import json
import io
import os
import re
import csv
import threading
import time
import multiprocessing as mp
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import requests
import pandas as pd
from dotenv import load_dotenv
from utils import *

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from camel.toolkits import FunctionTool, SearchToolkit, HumanToolkit, FileToolkit, BrowserToolkit
from camel.logger import get_logger
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Set, Tuple, Union, cast
from camel.loaders import Firecrawl

logger = get_logger(__name__)


class _RateLimiter:
    """Thread-safe sliding-window rate limiter: at most `max_calls` per `period` seconds,
    shared across all callers/threads that hold a reference to the same instance.

    Pass `calls`/`lock` as a multiprocessing.Manager().list()/Lock() to share the same
    budget across separate OS processes instead of just threads within one process —
    a plain deque+threading.Lock only enforces the cap within a single process, so
    each spawned worker process would otherwise get its own independent budget."""

    def __init__(self, max_calls: int, period: float = 60.0, calls=None, lock=None):
        self.max_calls = max_calls
        self.period = period
        self._calls = calls if calls is not None else deque()
        self._lock = lock if lock is not None else threading.Lock()

    def acquire(self):
        while True:
            with self._lock:
                now = time.monotonic()
                while len(self._calls) and now - self._calls[0] >= self.period:
                    del self._calls[0]
                if len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    return
                wait = self.period - (now - self._calls[0])
            time.sleep(max(wait, 0.05))


# Stay under the Firecrawl plan's per-minute cap even with multiple ThreadPoolExecutor
# workers hitting get_fulltext() concurrently. Override via FIRECRAWL_MAX_REQ_PER_MIN.
_firecrawl_rate_limiter = _RateLimiter(
    max_calls=int(os.getenv("FIRECRAWL_MAX_REQ_PER_MIN", "10")), period=60.0
)

class MeshQueryTool:
    """
    Enhanced PubMed MeSH query builder for PECO search.

    Logic:
      1. MeSH Terms:
         - IF standard_mesh_term exists (list): add each as "Term"[Mesh] to catch indexed papers.
      2. tiab Keywords:
         - Use the `terms` list (main standardized term + synonyms) as "Term"[tiab] to catch
           unindexed papers and ensure broad recall.
         - Falls back to `raw_term` if `terms` is not present (backward compatibility).
      3. Structure:
         - (Population OR ...) AND (Exposure OR ...) AND (Outcome OR ...) AND (Comparison OR ...)
         - Within each PECO element, group_id controls AND vs OR between distinct concepts.
         - Outcome is always OR regardless of group_ids.
    """

    def __init__(
            self,
            *,
            mesh_field: str = "Mesh",
            entry_terms_delimiter: str = ";",
            allow_comma_fallback: bool = True,
    ) -> None:
        self.mesh_field = mesh_field
        self.entry_terms_delimiter = entry_terms_delimiter
        self.allow_comma_fallback = allow_comma_fallback

    # -------------------- Public API --------------------

    def build_query(
            self,
            peco_input: str,
    ) -> str:
        """
        Builds a comprehensive PECO search query.
        """
        query_parts = []
        # 1. Parse the Single Input Blob
        data = self._parse_agent_input(peco_input)

        # 2. Extract PECO lists (safely handling missing keys)
        # Note: We use the helper to ensure even sub-fields are lists
        population = self._ensure_list(data.get("population"))
        exposure = self._ensure_list(data.get("exposure"))
        comparison = self._ensure_list(data.get("comparison"))
        outcome = self._ensure_list(data.get("outcome"))

        # Read per-element logical operators (from peco_operators field; default OR)
        # peco_operators values can be:
        #   - a plain string: "OR" or "AND"  (flat mode — all concepts joined by that op)
        #   - a dict: {"within_group": "OR", "between_groups": "AND"}  (nested group_id mode)
        peco_ops = data.get("peco_operators", {}) or {}

        def _resolve_ops(key):
            val = peco_ops.get(key, "OR")
            if isinstance(val, dict):
                within  = val.get("within_group",   "OR").upper()
                between = val.get("between_groups", "AND").upper()
            else:
                within = between = str(val).upper()
            return within, between

        pop_within,  pop_between  = _resolve_ops("population")
        int_within,  int_between  = _resolve_ops("exposure")
        comp_within, comp_between = _resolve_ops("comparison")
        out_within,  out_between  = _resolve_ops("outcome")

        # 3. Build query
        if population:
            query_parts.append(self._group_block_nested(population, pop_within, pop_between))

        if exposure:
            query_parts.append(self._group_block_nested(exposure, int_within, int_between))

        if comparison:
            query_parts.append(self._group_block_nested(comparison, comp_within, comp_between))

        # 4. Outcome Block
        if outcome:
            query_parts.append(self._group_block_nested(outcome, out_within, out_between))


        if not query_parts:
            raise ValueError("At least one PECO element (Population, Exposure, or Outcome) must be provided.")

        # Combine P AND I AND O
        full_query = " AND ".join(query_parts)

        
        # 4. Append Study Design
        # full_query = f"{full_query} AND {self._study_design_block()}"

        return full_query
    

    def refine_query(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """
        Refine query by extracting MeSH terms and keywords from initial search results.

        Args:
            query: Initial PubMed query string
            max_results: Number of papers to analyze for term extraction

        Returns:
            Dictionary with:
                - mesh_terms: List of unique MeSH terms found
                - keywords: List of unique keywords found
                - mesh_term_counts: Dict of MeSH term -> count
                - keyword_counts: Dict of keyword -> count
                - refined_query: Suggested refined query with top terms
        """
        pubmed = PubMedToolkit()
        searched_papers = pubmed.search_ids(
            query,
            max_results=max_results,
            sort="relevance",
            date_range=None,
            publication_type=None,
        )

        print(query)
        # Collect all MeSH terms and keywords
        all_mesh_terms: List[str] = []
        all_keywords: List[str] = []

        for paper_id in searched_papers:
            # Extract MeSH terms - searched_papers is a list of ID strings from search_ids()
            key_words, major_mesh, all_mesh = get_terms(paper_id)
            all_mesh_terms.extend(all_mesh)
            all_keywords.extend(key_words)

        # Count occurrences
        mesh_term_counts: Dict[str, int] = {}
        for term in all_mesh_terms:
            term_lower = term.lower()
            mesh_term_counts[term_lower] = mesh_term_counts.get(term_lower, 0) + 1

        keyword_counts: Dict[str, int] = {}
        for kw in all_keywords:
            kw_lower = kw.lower()
            keyword_counts[kw_lower] = keyword_counts.get(kw_lower, 0) + 1

        return {
            "mesh_term_counts": mesh_term_counts,
            "keyword_counts": keyword_counts,
        }

    # -------------------- Core construction --------------------

    def _parse_agent_input(self, input_data: Union[str, Dict]) -> Dict:
        """
        Safely converts the Agent's output (String or Dict) into a Python Dict.
        """
        if isinstance(input_data, dict):
            return input_data

        if isinstance(input_data, str):
            clean_str = input_data.strip()
            # Often agents wrap output in markdown code blocks, strip them:
            if clean_str.startswith("```json"):
                clean_str = clean_str.replace("```json", "").replace("```", "")
            elif clean_str.startswith("```"):
                clean_str = clean_str.replace("```", "")

            try:
                return json.loads(clean_str.strip())
            except json.JSONDecodeError:
                # Fallback: If parsing fails, return empty dict to avoid crash
                print(f"DEBUG: Failed to parse JSON input: {clean_str[:50]}...")
                return {}

        return {}

    def _ensure_list(self, data: Any) -> List[Dict]:
        """Ensures the PECO element data is a list of dicts."""
        if isinstance(data, list):
            return data
        return []

    def _group_block(self, concepts: List[Dict[str, Any]], operator: str = "OR") -> str:
        """Join multiple concepts within a PECO element using the given operator (OR or AND)."""
        blocks = [self._concept_block(c) for c in concepts]
        blocks = [b for b in blocks if b]
        if not blocks:
            return ""
        if len(blocks) == 1:
            return blocks[0]
        sep = f" {operator} "
        return "(" + sep.join(blocks) + ")"

    def _group_block_nested(
        self,
        concepts: List[Dict[str, Any]],
        within_op: str = "OR",
        between_op: str = "OR",
    ) -> str:
        """
        Build a PECO block that supports group_id-based nesting.

        Items sharing the same group_id are joined with within_op;
        distinct groups are then joined with between_op.

        If no item has a group_id, falls back to _group_block with within_op.
        """
        # Check if any concept uses group_id
        has_groups = any(c.get("group_id") is not None for c in concepts)
        if not has_groups:
            return self._group_block(concepts, within_op)

        # Partition by group_id; preserve insertion order
        groups: Dict[Any, List[Dict[str, Any]]] = {}
        for c in concepts:
            gid = c.get("group_id", "__ungrouped__")
            groups.setdefault(gid, []).append(c)

        group_blocks = []
        for gid, members in groups.items():
            block = self._group_block(members, within_op)
            if block:
                group_blocks.append(block)

        if not group_blocks:
            return ""
        if len(group_blocks) == 1:
            return group_blocks[0]
        sep = f" {between_op} "
        return "(" + sep.join(group_blocks) + ")"

    def _concept_block(self, concept: Dict[str, Any]) -> str:
        terms: List[str] = []

        # 1. Handle MeSH terms (singular or plural key)
        # PubMed quoting rules for MeSH:
        #   - "/" is an operator (heading/subheading) — NEVER quote across a slash
        #   - Quote the heading only when it contains a comma (comma breaks the parser)
        #   - Chained subheadings (term/sub1/sub2) must be split into separate terms
        std_mesh = concept.get("standard_mesh_term") or concept.get("standard_mesh_terms")
        if std_mesh:
            mesh_list = self._parse_entry_terms(std_mesh)
            for mesh_term in mesh_list:
                if not (mesh_term and mesh_term.strip()):
                    continue
                parts = [p.strip() for p in mesh_term.split("/")]
                heading = parts[0]
                subheadings = parts[1:]
                # Quote heading only if it contains a comma
                quoted_heading = f'"{heading}"' if "," in heading else heading
                if not subheadings:
                    terms.append(f"{quoted_heading}[{self.mesh_field}]")
                else:
                    # One entry per subheading (split chained subheadings)
                    for sub in subheadings:
                        terms.append(f"{quoted_heading}/{sub}[{self.mesh_field}]")

        # 2. Handle tiab keywords: use `terms` list (main term + synonyms) if available,
        #    fall back to `raw_term` for backward compatibility.
        # Multi-word phrases need quotes for [tiab] phrase searching;
        # single-word terms do not.
        keywords = concept.get("terms") or concept.get("term") or concept.get("raw_term") or concept.get("raw_terms")
        if keywords:
            kw_list = self._parse_entry_terms(keywords)
            for kw in kw_list:
                kw = (kw or "").strip()
                if not kw:
                    continue
                if " " in kw:
                    terms.append(f"\"{kw}\"[tiab]")
                else:
                    terms.append(f"{kw}[tiab]")

        # Deduplicate and join
        unique_terms = self._dedupe_preserve_order(terms)

        if not unique_terms:
            return ""

        return "(" + " OR ".join(unique_terms) + ")"

        # # Sort by frequency (most common first)
        # sorted_mesh = sorted(mesh_term_counts.items(), key=lambda x: x[1], reverse=True)
        # sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)


    # -------------------- Study design block --------------------

    @staticmethod
    # def _default_study_design_terms() -> List[str]:
    #     return [
    #         # RCT
    #         "\"Randomized Controlled Trials as Topic\"[MeSH]",
    #         "\"Randomized Controlled Trial\"[pt]",
    #         "\"Controlled Clinical Trial\"[pt]",
    #         # Observational
    #         "\"Observational Studies as Topic\"[MeSH]",
    #         "\"Observational Study\"[pt]",
    #         "\"Cohort Studies\"[MeSH]",
    #         "\"Case-Control Studies\"[MeSH]",
    #         "\"Cross-Sectional Studies\"[MeSH]",
    #         # Causal Inference Specific
    #         "\"Mendelian Randomization Analysis\"[MeSH]",
    #         "\"Causality\"[MeSH]",
    #         "\"Confounding Factors, Epidemiologic\"[MeSH]"
    #     ]

    def _study_design_block(self) -> str:
        parts = self._dedupe_preserve_order(self.study_design_terms)
        return "(" + " OR ".join(parts) + ")"

    # -------------------- Parsing Helpers --------------------

    def _parse_entry_terms(self, raw: Any) -> List[str]:
        if not raw:
            return []

        # Handle list input directly (from agent JSON arrays)
        if isinstance(raw, list):
            return [str(item).strip() for item in raw if item and str(item).strip()]

        s = str(raw).strip()
        if not s:
            return []

        # 1. Delimiter split (e.g. semicolon)
        if self.entry_terms_delimiter in s:
            terms = [t.strip() for t in s.split(self.entry_terms_delimiter) if t.strip()]
            return terms

        # 2. CSV parsing (handles quoted strings like "Falls, Accidental")
        csv_terms = self._try_parse_csv_line(s)
        if csv_terms:
            return csv_terms

        # 3. Fallback comma split (only if allowed and no quotes detected)
        if self.allow_comma_fallback and "," in s and '"' not in s:
            terms = [t.strip() for t in s.split(",") if t.strip()]
            return terms

        return [s]

    @staticmethod
    def _try_parse_csv_line(s: str) -> List[str]:
        if '"' not in s and "," not in s:
            return []
        try:
            reader = csv.reader(io.StringIO(s), skipinitialspace=True)
            row = next(reader, [])
            return [c.strip() for c in row if c and c.strip()]
        except Exception:
            return []

    @staticmethod
    def _escape_quotes(s: str) -> str:
        return (s or "").replace('"', '\\"').strip()

    @staticmethod
    def _dedupe_preserve_order(items: List[str]) -> List[str]:
        seen = set()
        out = []
        for it in items:
            key = it.lower().strip()
            if key and key not in seen:
                seen.add(key)
                out.append(it.strip())
        return out

def save_search_results(results_json: str, cols: List[str], filename: str = "literature_search.csv") -> str:
    """
    Saves the list of papers to a CSV file.
    Args:
        results_json: A JSON string or list of dicts containing the papers.
        cols: columns in the saved file
        filename: Name of the file to save (default: literature_search.csv).
    """
    if cols is None:
        cols = []
    try:
        if isinstance(results_json, str):
            data = json.loads(results_json)
        else:
            data = results_json

        df = pd.DataFrame(data)
        # Reorder columns for readability if they exist
        df = df

        df.to_csv(filename, index=False)
        return f"Successfully saved {len(df)} papers to {filename}."
    except Exception as e:
        return f"Error saving file: {str(e)}"



    
def get_keywords(paper_id: Union[str, int]) -> List[str]:
    r"""Get the keywords of a specific biomedical paper from MEDLINE/
    PubMed.

    Args:
        paper_id (Union[str, int]): PubMed ID of the paper.

    Returns:
        List[str]: A list of keywords.
    """
    params: Dict[str, Union[str, int]] = {
        "db": "pubmed",
        "id": str(paper_id),
        "retmode": "xml",
    }
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    try:
        response = requests.get(
            f"{BASE_URL}/efetch.fcgi", params=params
        )
        response.raise_for_status()
        try:
            root = ET.fromstring(response.content)

            # Extract keywords from KeywordList/Keyword tags
            keywords = root.findall(".//KeywordList/Keyword")
            keyword_list = [kw.text for kw in keywords if kw.text]
            return keyword_list

        except ET.ParseError:
            return []
    except requests.exceptions.RequestException as e:
        logger.error(
            f"Failed to retrieve keywords for ID {paper_id}: {e!s}"
        )
        return []

def get_terms(paper_id: Union[str, int]) -> List[str]:
    r"""Get the MeSH terms of a specific biomedical paper from MEDLINE/
    PubMed.

    Args:
        paper_id (Union[str, int]): PubMed ID of the paper.

    Returns:
        List[str]: A list of MeSH terms.
    """
    params: Dict[str, Union[str, int]] = {
        "db": "pubmed",
        "id": str(paper_id),
        "retmode": "xml",
        "api_key": os.environ.get("PUBMED_API_KEY")
    }
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    try:
        response = requests.get(
            f"{BASE_URL}/efetch.fcgi", params=params
        )
        response.raise_for_status()
        try:
            root = ET.fromstring(response.content)
            ############ GET KEYWORDS
            keywords = root.findall(".//KeywordList/Keyword")
            keyword_list = [kw.text for kw in keywords if kw.text]
            
            ############ GET MESH TERMS
            # Initialize lists before the loop
            all_mesh_list = []
            major_mesh_list = []

            # Extract MeSH terms from MeshHeadingList/MeshHeading/DescriptorName
            mesh_tree = root.findall('.//MeshHeadingList')
            for mesh_info in mesh_tree:
                for i in mesh_info:

                    full_mesh = ''
                    major_mesh = ''
                    
                    for m in i:

                        if m.tag=='DescriptorName':
                            descriptor=m.text
                            qualifier = ''
                            full_mesh+=descriptor
                            full_mesh += ''
                            full_mesh+=qualifier

                            major_mesh+=descriptor

                            major_mesh += ''
                            if m.attrib['MajorTopicYN'] == 'Y':
                                major_mesh +='*'
                            major_mesh+=qualifier

                        elif m.tag=='QualifierName':
                            descriptor=''
                            qualifier = m.text

                            full_mesh+=descriptor
                            full_mesh += '/'
                            full_mesh+=qualifier

                            if m.attrib['MajorTopicYN'] == 'Y':

                                major_mesh+=descriptor
                                major_mesh += '/'
                                major_mesh+=qualifier
                                major_mesh+='*'

                    all_mesh_list.append(full_mesh)
                    major_mesh_list.append(major_mesh)

            # Filter to only major topics (marked with *) after processing all terms
            major_mesh_list = [value for value in major_mesh_list if '*' in value]
            return keyword_list, major_mesh_list, all_mesh_list

        except ET.ParseError:
            return [], [], []
    except requests.exceptions.RequestException as e:
        logger.error(
            f"Failed to retrieve MeSH terms and keywords for ID {paper_id}: {e!s}"
        )
        return [], [], []


class PubMedToolkit:
    r"""A toolkit for interacting with PubMed's E-utilities API to access
    MEDLINE data.

    This toolkit provides functionality to search and retrieve papers from the
    PubMed database, including abstracts, citations, and other metadata.

    Args:
        timeout (Optional[float]): The timeout for API requests in seconds.
            (default: :obj:`None`)
    """

    def __init__(self, timeout: Optional[float] = None) -> None:
        r"""Initializes the PubMedToolkit."""
        self.timeout = timeout
        self.DEFAULT_PUBMED_API_KEY = os.environ.get('PUBMED_API_KEY')
        self.BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.session = requests.Session()

    def _make_request(
        self,
        endpoint: str,
        params: Dict[str, Union[str, int]],
        retries: int = 3,
    ) -> Optional[Dict[str, Any]]:
        r"""Makes a request to the PubMed/MEDLINE API with error handling and
        retries.

        Args:
            endpoint (str): The API endpoint to call.
            params (Dict[str, Union[str, int]]): Query parameters.
            retries (int, optional): Number of retry attempts.
                (default: :obj:`3`)

        Returns:
            Optional[Dict[str, Any]]: JSON response if successful, else None.
        """
        url = f"{self.BASE_URL}/{endpoint}"
        request_params = cast(Dict[str, Union[str, int]], params)
        headers = {
            'User-Agent': 'Mozilla/5.0',
        }

        # Use POST when query string would exceed safe URL length (~2000 chars).
        # PubMed E-utilities fully support POST for all endpoints.
        use_post = len(str(request_params.get("term", ""))) > 2000

        for attempt in range(retries):
            try:
                if use_post:
                    response = self.session.post(
                        url, data=request_params, timeout=self.timeout, headers=headers
                    )
                else:
                    response = self.session.get(
                        url, params=request_params, timeout=self.timeout, headers=headers
                    )
                response.raise_for_status()

                if not response.text:
                    logger.warning(
                        f"Empty response from PubMed API: {endpoint}"
                    )
                    return None

                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt == retries - 1:
                    logger.error(f"Failed to fetch data from PubMed: {e!s}")
                    return None
                logger.warning(f"Request attempt {attempt + 1} failed: {e!s}")
                time.sleep(2 ** attempt)  # back off before retrying (1s, 2s, ...)
            except ValueError as e:
                logger.error(f"Failed to parse JSON response: {e!s}")
                return None
        return None
    
    def search_ids(
            self,
            query: str,
            max_results: int = 10,
            sort: str = "relevance",
            date_range: Optional[Dict[str, str]] = None,
            publication_type: Optional[List[str]] = None,         
    ) -> List[str]:
        # Build query with filters

        filtered_query = query
        if publication_type:
            type_filter = " OR ".join(
                [f'"{pt}"[Publication Type]' for pt in publication_type]
            )
            filtered_query = f"({query}) AND ({type_filter})"
        if date_range:
            date_filter = (
                f"{date_range.get('from', '')}:"
                f"{date_range.get('to', '')}[Date - Publication]"
            )
            filtered_query = f"({filtered_query}) AND ({date_filter})"

        # Search for paper IDs
        search_params: Dict[str, Union[str, int]] = {
            "db": "pubmed",
            "term": filtered_query,
            "retmax": max_results,
            "sort": "relevance" if sort == "relevance" else "pub+date",
            "retmode": "json",
            'api_key': self.DEFAULT_PUBMED_API_KEY
        }

        search_data = self._make_request("esearch.fcgi", search_params)
        if not search_data or "esearchresult" not in search_data:
            logger.error("Failed to retrieve search results")
            return []

        paper_ids = search_data["esearchresult"].get("idlist", [])
        if not paper_ids:
            return []
        
        return paper_ids
        

    def search_papers(
        self,
        query: str,
        max_results: int = 10,
        sort: str = "relevance",
        date_range: Optional[Dict[str, str]] = None,
        publication_type: Optional[List[str]] = None,
    ) -> List[Dict[str, str]]:
        r"""Search for biomedical papers in MEDLINE via PubMed with advanced
        filtering options.

        Args:
            query (str): The search query string.
            max_results (int, optional): Maximum number of results to return.
                (default: :obj:`10`)
            sort (str, optional): Sort order - 'relevance' or 'date'.
                (default: :obj:`"relevance"`)
            date_range (Optional[Dict[str, str]], optional): Date range filter
                with 'from' and 'to' dates in YYYY/MM/DD format.
                (default: :obj:`None`)
            publication_type (Optional[List[str]], optional): Filter by
                publication types (e.g., ["Journal Article", "Review"]).
                (default: :obj:`None`)

        Returns:
            List[Dict[str, str]]: List of papers with their metadata.
        """
        paper_ids = self.search_ids(query=query, max_results=max_results, sort=sort,
                                    date_range=date_range, publication_type=publication_type)
        # Batch fetch summaries (much faster than one-by-one)
        print(f"  Batch fetching summaries for {len(paper_ids)} papers...")
        results = self._batch_get_summaries(paper_ids)

        # Batch fetch abstracts and DOIs from efetch XML (more reliable than esummary)
        print(f"  Batch fetching abstracts...")
        efetch_map = self._get_abstracts_batch(paper_ids, batch_size=100)

        # Merge abstract and DOI into results
        for paper in results:
            entry = efetch_map.get(paper["id"], {})
            paper["abstract"] = entry.get("abstract")
            if entry.get("doi"):
                paper["doi"] = entry["doi"]

        return results

    def _batch_get_summaries(self, paper_ids: List[str], batch_size: int = 100) -> List[Dict[str, Any]]:
        """Batch fetch paper summaries (metadata) for multiple papers."""
        results = []

        for i in range(0, len(paper_ids), batch_size):
            batch_ids = paper_ids[i:i + batch_size]
            ids_str = ",".join(batch_ids)

            summary_params = {
                "db": "pubmed",
                "id": ids_str,
                "retmode": "json",
                "api_key": self.DEFAULT_PUBMED_API_KEY
            }

            summary_data = self._make_request("esummary.fcgi", summary_params)
            if not summary_data or "result" not in summary_data:
                continue

            for paper_id in batch_ids:
                paper_data = summary_data["result"].get(str(paper_id))
                if not paper_data or paper_id == "uids":
                    continue

                # Handle authors
                authors = paper_data.get("authors", [])
                author_names = []
                for author in authors:
                    if isinstance(author, dict) and "name" in author:
                        author_names.append(author["name"])
                    elif isinstance(author, str):
                        author_names.append(author)

                # 1. Try articleids array (most reliable — typed entries)
                eloc_doi = ""
                for aid in paper_data.get("articleids", []):
                    if isinstance(aid, dict) and aid.get("idtype", "").lower() == "doi":
                        eloc_doi = aid.get("value", "").strip()
                        break
                # 2. Fall back to elocationid string if articleids had no DOI
                if not eloc_doi:
                    eloc = paper_data.get("elocationid", "")
                    if eloc.lower().startswith("doi:"):
                        eloc_doi = eloc[4:].strip()
                    elif eloc.lower().startswith("doi "):
                        eloc_doi = eloc[4:].strip()

                results.append({
                    "id": str(paper_id),
                    "title": paper_data.get("title", ""),
                    "authors": ", ".join(author_names),
                    "journal": paper_data.get("source", ""),
                    "pub_date": paper_data.get("pubdate", ""),
                    "doi": eloc_doi,
                    "publication_types": paper_data.get("pubtype", []),
                    "abstract": None,  # Will be filled by batch abstract fetch
                })

        return results
    
    def _get_abstracts_batch(self, paper_ids: List[Union[str, int]], batch_size: int = 200) -> Dict[str, Dict]:
        """
        Batch fetch abstracts and DOIs for multiple papers in fewer API calls.
        NCBI allows up to 200 IDs per request.

        Args:
            paper_ids: List of PubMed IDs
            batch_size: Number of IDs per request (max 200)

        Returns:
            Dict mapping paper_id -> {"abstract": str|None, "doi": str|None}
        """
        results = {}

        # Process in batches
        for i in range(0, len(paper_ids), batch_size):
            batch_ids = paper_ids[i:i + batch_size]
            ids_str = ",".join(str(pid) for pid in batch_ids)

            params = {
                "db": "pubmed",
                "id": ids_str,
                "retmode": "xml",
                'api_key': self.DEFAULT_PUBMED_API_KEY
            }

            try:
                response = requests.get(f"{self.BASE_URL}/efetch.fcgi", params=params, timeout=30)
                response.raise_for_status()
                root = ET.fromstring(response.content)

                # Parse each PubmedArticle in the response
                for article in root.findall(".//PubmedArticle"):
                    # Get PMID
                    pmid_elem = article.find(".//PMID")
                    if pmid_elem is None:
                        continue
                    pmid = pmid_elem.text

                    # Get abstract
                    abstract_texts = article.findall(".//Abstract/AbstractText")
                    if abstract_texts:
                        abstract = " ".join([elem.text for elem in abstract_texts if elem.text])
                    else:
                        abstract = None

                    # Get DOI: try ArticleIdList first, then ELocationID
                    doi_elem = article.find(".//ArticleIdList/ArticleId[@IdType='doi']")
                    if doi_elem is None:
                        doi_elem = article.find(".//ELocationID[@EIdType='doi']")
                    doi = doi_elem.text.strip() if doi_elem is not None else None

                    results[pmid] = {"abstract": abstract, "doi": doi}

            except Exception as e:
                logger.error(f"Batch fetch failed for IDs {i}-{i+batch_size}: {e}")
                for pid in batch_ids:
                    results[str(pid)] = {"abstract": None, "doi": None}

        return results

    def get_paper_details(
        self,
        paper_id: Union[str, int],
        include_references: bool = False,
    ) -> Optional[Dict[str, Any]]:
        r"""Get detailed information about a specific paper.

        Uses :meth:`_batch_get_summaries` and :meth:`_get_abstracts_batch`
        internally so the field schema is identical to :meth:`search_papers`.

        Args:
            paper_id (Union[str, int]): PubMed ID of the paper.
            include_references (bool, optional): Whether to include referenced
                paper IDs. (default: :obj:`False`)

        Returns:
            Optional[Dict[str, Any]]: Paper metadata dict, or ``None`` if
            the paper was not found.
        """
        pid = str(paper_id)

        # Fetch summary metadata (title, authors, journal, pub_date, doi …)
        summaries = self._batch_get_summaries([pid])
        if not summaries:
            return None
        paper = summaries[0]

        # Fetch abstract and DOI from efetch XML
        efetch_map = self._get_abstracts_batch([pid])
        entry = efetch_map.get(pid, {})
        paper["abstract"] = entry.get("abstract")
        if entry.get("doi"):
            paper["doi"] = entry["doi"]

        # Optionally fetch reference IDs via elink
        if include_references:
            ref_params: Dict[str, Union[str, int]] = {
                "db": "pubmed",
                "id": pid,
                "linkname": "pubmed_pubmed_refs",
                "retmode": "json",
            }
            ref_data = self._make_request("elink.fcgi", ref_params)
            references: List[str] = []
            if ref_data and "linksets" in ref_data:
                try:
                    references = ref_data["linksets"][0]["linksetdbs"][0]["links"]
                except (KeyError, IndexError):
                    logger.warning(f"No references found for paper ID: {paper_id}")
            paper["references"] = references
        else:
            paper["references"] = None

        return paper


def pubmed_search_tool(
        query: str,
        max_results: int = 10,
        sort: str = "relevance",
        date_range: Optional[Dict[str, str]] = None,
        publication_type: Optional[List[str]] = None,
        data_cols: Optional[List[str]] = None,
        if_save: bool = False,
        save_filename: Optional[str] = None,):
        r"""Search for biomedical papers in MEDLINE via PubMed with advanced
                filtering options.

                Args:
                    query (str): The search query string.
                    max_results (int, optional): Maximum number of results to return.
                        (default: :obj:`10`)
                    sort (str, optional): Sort order - 'relevance' or 'date'.
                        (default: :obj:`"relevance"`)
                    date_range (Optional[Dict[str, str]], optional): Date range filter
                        with 'from' and 'to' dates in YYYY/MM/DD format.
                        (default: :obj:`None`)
                    publication_type (Optional[List[str]], optional): Filter by
                        publication types (e.g., ["Journal Article", "Review"]).
                        (default: :obj:`None`)
        Returns:
                List[Dict[str, str]]: List of papers with their metadata.
            """
    

        # 1. Search Papers
        pubmed = PubMedToolkit()
        searched_papers = pubmed.search_papers(
            query,
            max_results=max_results,
            sort=sort,
            date_range=date_range,
            publication_type=publication_type
        )

        # 2. Early exit if no columns requested - return all columns
        if not data_cols:
            if if_save:
                filename = save_filename if save_filename else "pubmed_search_results.csv"
                save_search_results(searched_papers, cols=data_cols, filename=filename)
                # Return summary only to avoid context overflow
                return {
                    "status": "ok",
                    "saved_to": filename,
                    "num_papers": len(searched_papers),
                    "message": f"Successfully saved {len(searched_papers)} papers to {filename}"
                }
            return searched_papers

        # 3. Build filtered results with only specified columns
        final_results = []
        for paper in searched_papers:
            filtered_paper = {}
            for col in data_cols:
                if col == "citation_count":
                    # Skip citation_count - too slow (individual API call per paper)
                    filtered_paper["citation_count"] = None
                else:
                    filtered_paper[col] = paper.get(col, None)
            final_results.append(filtered_paper)

        # 5. Save if requested
        if if_save:
            filename = save_filename if save_filename else "pubmed_search_results.csv"
            save_search_results(final_results, cols=data_cols, filename=filename)
            # Return summary only to avoid context overflow
            return {
                "status": "ok",
                "saved_to": filename,
                "num_papers": len(final_results),
                "message": f"Successfully saved {len(final_results)} papers to {filename}"
            }

        return final_results


def pmid2biocxml(pmid):
    """Get BioC XML from PubMed Central"""
    base_url = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_xml/{pmid}/unicode"
    
    request_url = base_url.format(pmid=pmid)
    response = requests.get(request_url)

    return response.text


# Semantic Scholar rate limiter: max 1 request per second
_ss_lock = threading.Lock()
_ss_last_call = 0.0

def _ss_rate_limit():
    global _ss_last_call
    with _ss_lock:
        elapsed = time.time() - _ss_last_call
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)
        _ss_last_call = time.time()


def get_text_from_pdf_url(pdf_url, hard_timeout: float = 60.0):
    """
    Download PDF from URL and extract text content using PyMuPDF

    Args:
        pdf_url: URL to the PDF file
        hard_timeout: maximum total wall-clock seconds to wait for the whole download +
            extraction, enforced independently of requests' own timeout. requests' timeout
            only bounds the gap between reads, not total duration — a slow-but-nonzero
            trickle of data (common on flaky institutional repository servers) can keep
            resetting that per-chunk timer indefinitely without ever finishing. Running the
            work in a background thread and bounding it with future.result(timeout=...) is
            the only way to guarantee this function returns within hard_timeout regardless.

    Returns:
        Extracted text content as string, or None if extraction fails
    """
    def _download_and_extract():
        import fitz  # PyMuPDF
        from io import BytesIO

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "application/pdf,*/*",
            "Referer": pdf_url,
        }
        # Stream the body and track cumulative wall-clock time ourselves — requests' own
        # timeout only bounds the gap between reads, so a slow-but-nonzero trickle never
        # trips it even though the total download takes far longer than hard_timeout. This
        # is a second, independent safeguard alongside the outer future.result(timeout=...).
        start = time.monotonic()
        chunks = []
        with requests.get(pdf_url, headers=headers, timeout=30, stream=True) as response:
            if response.status_code != 200:
                print(f"    Failed to download PDF: HTTP {response.status_code}")
                return None
            for chunk in response.iter_content(chunk_size=65536):
                if time.monotonic() - start > hard_timeout:
                    print(f"    ✗ PDF download exceeded hard timeout ({hard_timeout:.0f}s) mid-stream, aborting")
                    return None
                if chunk:
                    chunks.append(chunk)

        # Open PDF from bytes
        pdf_bytes = BytesIO(b"".join(chunks))
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        # Extract text from all pages
        text = ""
        for page in doc:
            text += page.get_text()

        page_count = doc.page_count
        doc.close()
        print(f"    ✓ Extracted {len(text)} characters from PDF ({page_count} pages)")
        return text

    print(f"    Downloading PDF from {pdf_url[:50]}...")
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(_download_and_extract)
    try:
        return future.result(timeout=hard_timeout)
    except FuturesTimeoutError:
        print(f"    ✗ PDF download exceeded hard timeout ({hard_timeout:.0f}s), giving up")
        return None
    except ImportError:
        print(f"    Error: PyMuPDF not installed. Install with: pip install PyMuPDF")
        return None
    except Exception as e:
        print(f"    PDF extraction error: {e}")
        return None
    finally:
        # Don't block on a background thread that may still be stuck mid-download —
        # let it linger and get cleaned up whenever it eventually finishes or errors.
        pool.shutdown(wait=False)

def get_fulltext(pmid, doi=None, resource_folder=None):
    """
    Get full text from PubMed Central or publisher website

    This tool tries strategies in order:
    0. Local PDF file  (if resource_folder is provided and {pmid}.pdf exists)
    1. NCBI PubMed Central BioC XML (free, structured XML)
    2. Europe PMC full-text XML (free, covers many EU/UKRI-funded papers)
    3. Elsevier Full-Text API (covers ScienceDirect, Lancet, JACC, etc.)
    4. Wiley TDM API (covers onlinelibrary.wiley.com journals)
    5. Springer Nature OA API (covers Nature, BMC, Springer journals)
    6. PLOS API (all PLOS journals are open access)
    7. Semantic Scholar open-access PDF (free)
    8. OpenAlex open-access locations (free, repository copies)
    9. Unpaywall (free, finds legal open-access versions via DOI)
    10. CrossRef TDM links (free, publisher-registered text-mining links)
    11. citation_pdf_url meta tag (generic — covers Cambridge Core and others)
    12. Jina AI Reader (free/paid, scrapes publisher website via DOI)
    13. Firecrawl (paid, handles JavaScript-heavy publisher sites)

    Args:
        pmid: PubMed ID (required)
        doi: DOI (optional, needed for Unpaywall, Jina Reader and Firecrawl fallback)
        resource_folder: Optional path to a local folder containing full-text PDFs
                         named by PMID (e.g. /resource/24113040.pdf).  When provided
                         and the file exists, its text is returned immediately without
                         hitting any remote API.

    Returns:
        str: Full text content (XML or markdown), or None if not found

    Example:
        >>> content = get_fulltext('12345678', '10.1001/jama.2020.1234')
        >>> if content:
        >>>     print(f"Retrieved {len(content)} characters")
    """
    if resource_folder:
        pdf_path = os.path.join(resource_folder, f"{pmid}.pdf")
        if os.path.isfile(pdf_path):
            print(f"  Trying local PDF for PMID {pmid}...")
            try:
                import fitz
                doc = fitz.open(pdf_path)
                full_text = "\n".join(page.get_text() for page in doc).strip()
                page_count = doc.page_count
                doc.close()
                if full_text:
                    print(f"  ✓ Extracted {len(full_text)} chars from local PDF ({page_count} pages)")
                    return full_text
                print(f"  ✗ Local PDF yielded no text")
            except Exception as e:
                print(f"  ✗ Local PDF error: {e}")
        else:
            print(f"  ✗ Local PDF not found for PMID {pmid}")
 
    # ── Strategy 1: NCBI PMC BioC XML ──────────────────────────────────────
    print(f"  Trying PMC BioC XML for PMID {pmid}...")
    try:
        pmc_xml = pmid2biocxml(pmid)
        if pmc_xml and len(pmc_xml.strip()) > 100 and "<passage>" in pmc_xml:
            print(f"  ✓ PMC BioC XML ({len(pmc_xml)} chars)")
            return pmc_xml
        print(f"  ✗ PMC BioC: empty or invalid XML")
    except Exception as e:
        print(f"  ✗ PMC BioC error: {e}")
 
    # ── Strategy 2: Europe PMC full-text XML (NEW) ──────────────────────────
    # Covers many EU/UKRI/Wellcome/NIH-funded papers not in NCBI PMC.
    # Free, no API key required.
    print(f"  Trying Europe PMC for PMID {pmid}...")
    try:
        epmc_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/article/MED/{pmid}/fullTextXML"
        epmc_resp = requests.get(epmc_url, timeout=15)
        if epmc_resp.status_code == 200:
            epmc_xml = epmc_resp.text
            if epmc_xml and len(epmc_xml.strip()) > 500 and "<body" in epmc_xml:
                print(f"  ✓ Europe PMC full-text XML ({len(epmc_xml)} chars)")
                return epmc_xml
            print(f"  ✗ Europe PMC: too short or no <body>")
        elif epmc_resp.status_code == 404:
            print(f"  ✗ Europe PMC: not in Europe PMC")
        else:
            print(f"  ✗ Europe PMC: HTTP {epmc_resp.status_code}")
    except Exception as e:
        print(f"  ✗ Europe PMC error: {e}")
 
    # ── Strategy 3: Elsevier Full-Text API ──────────────────────────────────
    # view=FULL is required — without it, Elsevier returns only metadata (no text).
    els_api_key = os.getenv("ELSEVIER_API_KEY")
    publisher = doi_publisher(doi)
    if els_api_key and doi:
        print(f"  Trying Elsevier API for DOI {doi}...")
        try:
            els_resp = requests.get(
                f"https://api.elsevier.com/content/article/doi/{doi}",
                headers={
                    "X-ELS-APIKey":    els_api_key,
                    "X-ELS-Insttoken": os.getenv("ELSEVIER_INST_TOKEN", ""),
                    "Accept":          "text/xml",
                },
                params={"view": "FULL"},
                timeout=15,
            )
            if els_resp.status_code == 200:
                els_xml = els_resp.text
                # A genuine full-text response is large; metadata-only is tiny
                if els_xml and len(els_xml.strip()) > 2000:
                    print(f"  ✓ Elsevier full-text API ({len(els_xml)} chars)")
                    return els_xml
                print(f"  ✗ Elsevier API: response too short — likely metadata only")
            elif els_resp.status_code == 404:
                print(f"  ✗ Elsevier API: DOI not found (not an Elsevier journal)")
            elif els_resp.status_code == 401:
                print(f"  ✗ Elsevier API: invalid API key — body: {els_resp.text[:300]}")
            elif els_resp.status_code == 403:
                print(f"  ✗ Elsevier API: no entitlement (check insttoken / VPN) — body: {els_resp.text[:300]}")
            else:
                print(f"  ✗ Elsevier API: HTTP {els_resp.status_code}")
        except Exception as e:
            print(f"  ✗ Elsevier API error: {e}")
    elif not doi:
        print(f"  ⊘ Skipping Elsevier API (no DOI)")
 
    # ── Strategy 4: Wiley TDM API (MOVED UP) ───────────────────────────────
    wiley_api_key = os.getenv("WILEY_TDM_API_KEY")
    if wiley_api_key and doi:
        print(f"  Trying Wiley TDM API for DOI {doi}...")
        try:
            wiley_resp = requests.get(
                f"https://api.wiley.com/onlinelibrary/tdm/v1/articles/{doi}",
                headers={"Wiley-TDM-Client-Token": wiley_api_key, "Accept": "application/pdf"},
                timeout=20,
            )
            if wiley_resp.status_code == 200:
                content_type = wiley_resp.headers.get("Content-Type", "")
                if "pdf" in content_type.lower():
                    from io import BytesIO
                    import fitz
                    doc = fitz.open(stream=BytesIO(wiley_resp.content), filetype="pdf")
                    text = "".join(page.get_text() for page in doc)
                    page_count = doc.page_count
                    doc.close()
                    if text and len(text) > 500:
                        print(f"  ✓ Wiley TDM PDF ({len(text)} chars, {page_count} pages)")
                        return text
                    print(f"  ✗ Wiley TDM: PDF extracted no text")
                else:
                    print(f"  ✗ Wiley TDM: unexpected content type: {content_type}")
            elif wiley_resp.status_code == 404:
                print(f"  ✗ Wiley TDM: not a Wiley journal")
            elif wiley_resp.status_code == 401:
                print(f"  ✗ Wiley TDM: invalid API key")
            else:
                print(f"  ✗ Wiley TDM: HTTP {wiley_resp.status_code}")
        except Exception as e:
            print(f"  ✗ Wiley TDM error: {e}")
    elif not doi:
        print(f"  ⊘ Skipping Wiley TDM (no DOI)")
 
    # ── Strategy 5: Springer Nature TDM API (NEW) ───────────────────────────
    # Covers Nature, BMJ Open, Springer, BioMed Central journals.
    # Free API key: https://dev.springernature.com
    springer_api_key = os.getenv("SPRINGER_API_KEY")
    if springer_api_key and doi:
        print(f"  Trying Springer Nature API for DOI {doi}...")
        try:
            # Springer Nature OA API: DOI goes in query param, not URL path
            sn_resp = requests.get(
                "https://api.springernature.com/openaccess/jats",
                params={"q": f"doi:{doi}", "api_key": springer_api_key},
                timeout=15,
            )
            if sn_resp.status_code == 200:
                sn_text = sn_resp.text
                if sn_text and len(sn_text.strip()) > 500 and "<body" in sn_text:
                    print(f"  ✓ Springer Nature OA JATS XML ({len(sn_text)} chars)")
                    return sn_text
                print(f"  ✗ Springer Nature OA: too short or no body")
            elif sn_resp.status_code == 404:
                print(f"  ✗ Springer Nature: DOI not found or not OA")
            else:
                print(f"  ✗ Springer Nature: HTTP {sn_resp.status_code}")
        except Exception as e:
            print(f"  ✗ Springer Nature error: {e}")
    elif not doi:
        print(f"  ⊘ Skipping Springer Nature API (no DOI)")
 
    # ── Strategy 6: PLOS API (NEW) ──────────────────────────────────────────
    # All PLOS journals are fully open access — reliable full text for any PLOS DOI.
    # No API key required.
    if doi and (doi.startswith("10.1371") or publisher == "plos"):
        print(f"  Trying PLOS API for DOI {doi}...")
        try:
            plos_resp = requests.get(
                "https://api.plos.org/search",
                params={"q": f'id:"{doi}"', "fl": "id,body", "wt": "json"},
                timeout=15,
            )
            if plos_resp.status_code == 200:
                docs = plos_resp.json().get("response", {}).get("docs", [])
                if docs and docs[0].get("body"):
                    body = docs[0]["body"]
                    if len(body) > 500:
                        print(f"  ✓ PLOS API full text ({len(body)} chars)")
                        return body
                print(f"  ✗ PLOS API: no body in response")
            else:
                print(f"  ✗ PLOS API: HTTP {plos_resp.status_code}")
        except Exception as e:
            print(f"  ✗ PLOS API error: {e}")
 
    # ── Strategy 7: Semantic Scholar OA PDF ────────────────────────────────
    print(f"  Trying Semantic Scholar for PMID {pmid}...")
    try:
        _ss_rate_limit()
        ss_headers = {}
        ss_api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        if ss_api_key:
            ss_headers["x-api-key"] = ss_api_key

        # Request all PDF-related fields in one call
        ss_resp = requests.get(
            f"https://api.semanticscholar.org/graph/v1/paper/PMID:{pmid}"
            f"?fields=openAccessPdf,isOpenAccess,openAccessLocations",
            headers=ss_headers,
            timeout=10,
        )
        if ss_resp.status_code == 429:
            print(f"  ✗ Semantic Scholar rate limited — set SEMANTIC_SCHOLAR_API_KEY")
        elif ss_resp.status_code == 200:
            ss_data = ss_resp.json()

            # Collect all candidate PDF URLs from all locations
            pdf_urls = []

            # Primary openAccessPdf field
            oa_pdf = ss_data.get("openAccessPdf")
            if oa_pdf and oa_pdf.get("url"):
                pdf_urls.append(oa_pdf["url"])

            # All openAccessLocations (may include PubMed Central, repository, author copies)
            for loc in ss_data.get("openAccessLocations") or []:
                url = loc.get("pdfUrl") or loc.get("landingPageUrl")
                if url and url not in pdf_urls:
                    pdf_urls.append(url)

            if pdf_urls:
                for oa_url in pdf_urls:
                    print(f"  Semantic Scholar trying: {oa_url[:80]}...")
                    pdf_text = get_text_from_pdf_url(oa_url)
                    if pdf_text and len(pdf_text) > 500:
                        print(f"  ✓ Semantic Scholar PDF ({len(pdf_text)} chars)")
                        return pdf_text
                print(f"  ✗ Semantic Scholar: all {len(pdf_urls)} PDF location(s) failed")
            else:
                print(f"  ✗ Semantic Scholar: no OA PDF locations found")
        else:
            print(f"  ✗ Semantic Scholar: HTTP {ss_resp.status_code}")
    except Exception as e:
        print(f"  ✗ Semantic Scholar error: {e}")
 
    # ── Strategy 8: OpenAlex OA locations (NEW) ─────────────────────────────
    # OpenAlex often finds OA versions Unpaywall misses, especially repository copies.
    # Free, no API key required (but add email for polite pool = higher rate limits).
    if doi:
        print(f"  Trying OpenAlex for DOI {doi}...")
        try:
            oa_email = os.getenv("UNPAYWALL_EMAIL", "researcher@example.com")
            oa_resp = requests.get(
                f"https://api.openalex.org/works/doi:{doi}",
                params={"mailto": oa_email},
                timeout=10,
            )
            if oa_resp.status_code == 200:
                work = oa_resp.json()
                locations = work.get("open_access", {}).get("oa_locations", []) or []
                # Also check best_oa_location
                best = work.get("open_access", {}).get("best_oa_location")
                if best:
                    locations = [best] + [l for l in locations if l != best]
                pdf_tried = set()
                for loc in locations:
                    pdf_url = loc.get("pdf_url") or loc.get("url")
                    if not pdf_url or pdf_url in pdf_tried:
                        continue
                    pdf_tried.add(pdf_url)
                    if not pdf_url.endswith(".pdf") and "pdf" not in pdf_url.lower():
                        continue
                    print(f"  Trying OpenAlex PDF: {pdf_url[:60]}...")
                    pdf_text = get_text_from_pdf_url(pdf_url)
                    if pdf_text and len(pdf_text) > 500:
                        print(f"  ✓ OpenAlex PDF ({len(pdf_text)} chars)")
                        return pdf_text
                print(f"  ✗ OpenAlex: no usable PDF URL found")
            elif oa_resp.status_code == 404:
                print(f"  ✗ OpenAlex: DOI not found")
            else:
                print(f"  ✗ OpenAlex: HTTP {oa_resp.status_code}")
        except Exception as e:
            print(f"  ✗ OpenAlex error: {e}")
 
    # ── Strategy 9: Unpaywall ───────────────────────────────────────────────
    if doi:
        print(f"  Trying Unpaywall for DOI {doi}...")
        try:
            email = os.getenv("UNPAYWALL_EMAIL", "researcher@example.com")
            uw_resp = requests.get(
                f"https://api.unpaywall.org/v2/{doi}?email={email}", timeout=15
            )
            if uw_resp.status_code == 200:
                uw_data = uw_resp.json()
                best = uw_data.get("best_oa_location")
                all_locs = uw_data.get("oa_locations", [])
                repo_loc = next(
                    (l for l in all_locs if l != best and l.get("host_type") == "repository"),
                    None,
                )
                for loc in [l for l in [best, repo_loc] if l]:
                    label = loc.get("host_type", "?")
                    pdf_url = loc.get("url_for_pdf")
                    if pdf_url:
                        print(f"  Trying Unpaywall {label} PDF: {pdf_url[:60]}...")
                        pdf_text = get_text_from_pdf_url(pdf_url)
                        if pdf_text and len(pdf_text) > 500:
                            print(f"  ✓ Unpaywall {label} PDF ({len(pdf_text)} chars)")
                            return pdf_text
                        print(f"  ✗ Unpaywall {label} PDF failed")
                print(f"  ✗ Unpaywall: no direct PDF available")
            else:
                print(f"  ✗ Unpaywall: HTTP {uw_resp.status_code}")
        except requests.exceptions.Timeout:
            print(f"  ✗ Unpaywall timed out")
        except Exception as e:
            print(f"  ✗ Unpaywall error: {str(e)[:100]}")
 
    # ── Strategy 10: CrossRef TDM links ─────────────────────────────────────
    # CrossRef metadata sometimes carries a publisher-registered direct
    # text-mining download link. Free, no API key required, plain JSON —
    # no scraping/bot-detection risk. Some publishers gate their TDM link
    # behind a click-through token (Elsevier does); we opportunistically
    # reuse ELSEVIER_API_KEY for that header since it's the only TDM token
    # we hold, but most registered links don't require one at all.
    if doi:
        print(f"  Trying CrossRef TDM links for DOI {doi}...")
        try:
            cr_resp = requests.get(
                f"https://api.crossref.org/works/{doi}",
                params={"mailto": os.getenv("UNPAYWALL_EMAIL", "researcher@example.com")},
                timeout=15,
            )
            if cr_resp.status_code == 200:
                links = cr_resp.json().get("message", {}).get("link", []) or []
                tdm_links = [
                    l for l in links
                    if l.get("intended-application") == "text-mining" and l.get("URL")
                ]
                if not tdm_links:
                    print(f"  ✗ CrossRef: no text-mining links registered")
                click_token = os.getenv("ELSEVIER_API_KEY")
                cr_headers = {"CR-Clickthrough-Client-Token": click_token} if click_token else {}
                for link in tdm_links:
                    link_url = link["URL"]
                    content_type = (link.get("content-type") or "").lower()
                    print(f"  Trying CrossRef TDM link ({content_type or 'unknown'}): {link_url[:60]}...")
                    if "pdf" in content_type or link_url.lower().endswith(".pdf"):
                        text = get_text_from_pdf_url(link_url)
                    else:
                        try:
                            link_resp = requests.get(link_url, headers=cr_headers, timeout=20)
                            text = link_resp.text if link_resp.status_code == 200 else None
                        except Exception:
                            text = None
                    if text and len(text.strip()) > 500:
                        print(f"  ✓ CrossRef TDM link ({len(text)} chars)")
                        return text
                    print(f"  ✗ CrossRef TDM link: no usable content")
            else:
                print(f"  ✗ CrossRef: HTTP {cr_resp.status_code}")
        except Exception as e:
            print(f"  ✗ CrossRef error: {str(e)[:100]}")

    # ── Strategy 11: citation_pdf_url meta tag ──────────────────────────────
    # Most academic publishing platforms — including Cambridge Core — embed a
    # <meta name="citation_pdf_url"> tag on the article landing page; it's the
    # same tag Google Scholar relies on. Generic and cheap, and covers many
    # publishers (Cambridge among them) without needing site-specific scrapers.
    if doi:
        print(f"  Trying citation_pdf_url meta tag for DOI {doi}...")
        try:
            landing_resp = requests.get(
                f"https://doi.org/{doi}",
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
                },
                timeout=20,
            )
            if landing_resp.status_code == 200:
                match = re.search(
                    r'<meta[^>]+name=["\']citation_pdf_url["\'][^>]+content=["\']([^"\']+)["\']',
                    landing_resp.text, re.IGNORECASE,
                )
                if match:
                    pdf_url = match.group(1)
                    print(f"  Found citation_pdf_url: {pdf_url[:60]}...")
                    pdf_text = get_text_from_pdf_url(pdf_url)
                    if pdf_text and len(pdf_text) > 500:
                        print(f"  ✓ citation_pdf_url PDF ({len(pdf_text)} chars)")
                        return pdf_text
                    print(f"  ✗ citation_pdf_url: PDF fetch failed")
                else:
                    print(f"  ✗ No citation_pdf_url meta tag found")
            else:
                print(f"  ✗ Landing page: HTTP {landing_resp.status_code}")
        except requests.exceptions.Timeout:
            print(f"  ✗ Landing page timed out")
        except Exception as e:
            print(f"  ✗ citation_pdf_url error: {str(e)[:100]}")

    # ── Strategy 12: Jina AI Reader ────────────────────────────────────────
    if doi:
        print(f"  Trying Jina AI Reader for DOI {doi}...")
        try:
            jina_url = f"https://r.jina.ai/https://doi.org/{doi}"
            headers = {"X-Return-Format": "markdown"}
            jina_api_key = os.getenv("JINA_API_KEY")
            if jina_api_key:
                headers["Authorization"] = f"Bearer {jina_api_key}"
            resp = requests.get(jina_url, headers=headers, timeout=45)
            if resp.status_code == 200:
                content = resp.text
                _error_signals = [
                    "cookies turned off", "cookies are disabled",
                    "enable cookies", "javascript is disabled",
                    "access denied", "403 forbidden", "sign in to continue",
                ]
                if any(sig in content.lower() for sig in _error_signals):
                    print(f"  ✗ Jina: returned auth/cookie error page")
                elif len(content) > 10000:
                    print(f"  ✓ Jina Reader ({len(content)} chars)")
                    return content
                else:
                    print(f"  ✗ Jina: insufficient content")
            else:
                print(f"  ✗ Jina: HTTP {resp.status_code}")
        except requests.exceptions.Timeout:
            print(f"  ✗ Jina timed out (45s)")
        except Exception as e:
            print(f"  ✗ Jina error: {str(e)[:100]}")
 
    # ── Strategy 13: Firecrawl ─────────────────────────────────────────────
    if doi:
        print(f"  Trying Firecrawl for DOI {doi}...")
        try:
            from camel.loaders import Firecrawl
            firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")
            if not firecrawl_api_key:
                print(f"  ✗ FIRECRAWL_API_KEY not set")
            else:
                firecrawl = Firecrawl(api_key=firecrawl_api_key)

                # proxy="auto" makes Firecrawl retry server-side with enhanced
                # anti-bot proxies if the basic proxy gets blocked, within a
                # single billed request — no need to loop/retry on our side.
                _firecrawl_rate_limiter.acquire()
                result = firecrawl.scrape(
                    url=f"https://doi.org/{doi}",
                    params={
                        "formats": ["markdown"],
                        "onlyMainContent": True,
                        "timeout": 60000,
                        "waitFor": 2000,
                        "proxy": "auto",
                    },
                )
                if result and "markdown" in result and len(result["markdown"]) > 10000:
                    print(f"  ✓ Firecrawl ({len(result['markdown'])} chars)")
                    return result["markdown"]
                print(f"  ✗ Firecrawl: insufficient content")
        except ImportError:
            print(f"  ✗ Firecrawl not installed: pip install firecrawl-py")
        except Exception as e:
            print(f"  ✗ Firecrawl error: {str(e)[:200]}")
    else:
        print(f"  ⊘ Skipping web scraping (no DOI)")
 
    print(f"  ✗ No full text found for PMID {pmid}")
    return None

def parse_bioc_xml(path):
    """
    Parse BioC XML text to a list of dictionary of each passage
    and its metadata

    Parameters
    ----------
    path: str
        Path to the BioC formatted XML file.
        Or the input xml text.

    Return
    ------
    dict_bioc: list
        A list contains all dictionary of passage with its metadata.
        Metadata includes 'pmid', 'pmc', 'section' name of an article, and section 'text'
    """
    if os.path.exists(path):
        bioc_text = open(path, "r").read()
    else:
        bioc_text = path
    soup = BeautifulSoup(bioc_text, features="xml")
    passages = soup.find_all("passage")
    passage_dic = list()
    ref_dic = list()
    table_dic = list()

    for passage in passages:
        infons = passage.find_all("infon")
        section = [info.text for info in infons if info.get("key") == "section_type"]
        if len(section) == 0:
            section = ""
        else:
            section = section[0]

        pmid = [info.text for info in infons if info.get("key") == "article-id_pmid"]
        if len(pmid) == 0:
            pmid = ""
        else:
            pmid = pmid[0]

        pmcid = [info.text for info in infons if info.get("key") == "article-id_pmc"]
        if len(pmcid) == 0:
            pmcid = ""
        else:
            pmcid = pmcid[0]


        if section == "REF":
            texts = passage.find_all("text")
            texts = ". ".join([t.text for t in texts])
            ref_dic.append(
                {
                    "pmid": pmid,
                    "pmc": pmcid,
                    "section": section,
                    "content": texts,
                }
            )

        elif section == "TABLE":
            tab_ele_type = passage.find("infon", {"key": "type"})
            if tab_ele_type is not None: tab_ele_type = tab_ele_type.text
            tab_id = passage.find("infon", {"key": "id"})
            if tab_id is not None: tab_id = tab_id.text
            texts = passage.find_all("text")
            texts = ". ".join([t.text for t in texts])
            table_dic.append(
                {
                    "pmid": pmid,
                    "pmc": pmcid,
                    "section": section,
                    "tab_id": tab_id,
                    "tab_ele_type": tab_ele_type,
                    "content": texts,
                }
            )


        else:
            texts = passage.find_all("text")
            texts = ". ".join([t.text for t in texts])
            passage_dic.append(
                {
                    "pmid": pmid,
                    "pmc": pmcid,
                    "section": section,
                    "content": texts,
                }
            )

    return {
        "passage": passage_dic,
        "ref": ref_dic,
        "table": table_dic,
    }

def add_dois_to_csv(
        dois: List[str],
        csv_filename: str = "pubmed_results_exposure_outcome.csv",
        data_cols: Optional[List[str]] = None,
) -> str:
    """
    Fetch papers by DOI from PubMed, merge with an existing CSV, deduplicate,
    and re-save. Always preserves the original search results.

    Args:
        dois: List of DOI strings to add.
        csv_filename: Path to the existing CSV (must already exist).
        data_cols: Columns to fetch (same as used in pubmed_search_tool).

    Returns:
        A status message with counts.
    """
    # 1. Load existing CSV
    try:
        existing_df = pd.read_csv(csv_filename)
    except FileNotFoundError:
        return f"Error: {csv_filename} not found. Run the initial search first."

    # 2. Fetch each DOI via PubMed
    new_papers = []
    for doi in dois:
        results = pubmed_search_tool(
            query=f"{doi}",
            max_results=1,
            data_cols=data_cols,
        )
        if results:
            new_papers.extend(results)

    if not new_papers:
        return "No papers found for the provided DOIs. CSV unchanged."

    # 3. Merge with existing
    new_df = pd.DataFrame(new_papers)
    merged_df = pd.concat([existing_df, new_df], ignore_index=True)

    # 4. Deduplicate by 'id' (PMID) then 'doi'
    if "id" in merged_df.columns:
        merged_df = merged_df.drop_duplicates(subset=["id"], keep="first")
    if "doi" in merged_df.columns:
        merged_df = merged_df.drop_duplicates(subset=["doi"], keep="first")

    # 5. Re-save
    merged_df.to_csv(csv_filename, index=False)
    n_added = len(merged_df) - len(existing_df)
    return (
        f"Added {n_added} new paper(s) (fetched {len(new_df)}, duplicates removed). "
        f"CSV now has {len(merged_df)} papers total (was {len(existing_df)})."
    )


def extract_fulltext_content(content: str) -> str:
    """
    Reduce full-text content to plain readable text for LLM consumption.

    For BioC XML (from PubMed Central): strips all XML markup and extracts
    only the prose from <passage> elements and <table> elements, joined in
    document order.  This can reduce a 240 000-char BioC XML file to
    ~30 000-40 000 chars of actual text.

    For Markdown (from Jina / Firecrawl): returned as-is.

    Args:
        content: Raw full-text string (BioC XML or Markdown)

    Returns:
        Plain text string ready to pass to a language model.
    """
    if "<passage>" not in content or "<infon" not in content:
        return content  # already markdown / plain text

    parsed = parse_bioc_xml(content)

    parts = []

    # All non-reference passages in document order
    for p in parsed.get("passage", []):
        text = (p.get("content") or "").strip()
        sec  = (p.get("section") or "").upper()
        if text and sec != "REF":
            if sec:
                parts.append(f"[{sec}]\n{text}")
            else:
                parts.append(text)

    # All table content
    for t in parsed.get("table", []):
        text     = (t.get("content") or "").strip()
        tab_id   = t.get("tab_id") or ""
        ele_type = t.get("tab_ele_type") or ""
        if text:
            label = f"[TABLE {tab_id} {ele_type}]".strip()
            parts.append(f"{label}\n{text}")

    return "\n\n".join(parts)


def strip_references(text: str) -> str:
    """Remove reference/bibliography section and return clean content for LLM.

    For BioC XML: delegates to extract_fulltext_content (REF passages already skipped).
    For Markdown / plain text: cuts at the first References heading.
    """
    if "<passage>" in text and "<infon" in text:
        return extract_fulltext_content(text)
    cleaned = re.sub(
        r'\n{0,2}#{0,3}\s*(?:References?|Bibliography|Works Cited)\s*\n.*',
        '',
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return cleaned.strip()


def extract_sections_split(content: str) -> dict:
    """
    Split full-text content into methods and results sections for two-pass extraction.
    Returns dict with 'methods' and 'results' keys.

    Handles two formats:
    - BioC XML: splits by section_type infon tags
    - Plain text / Markdown (PDF): splits by common section header patterns
    For both formats, falls back to full text in both keys if split is ambiguous.
    """
    import re

    # ── BioC XML ──────────────────────────────────────────────────────────────
    if "<passage>" in content and "<infon" in content:
        parsed = parse_bioc_xml(content)
        METHODS_SECTIONS = {"METHODS", "INTRO", "TITLE", "ABSTRACT"}
        RESULTS_SECTIONS = {"RESULTS", "DISCUSS", "CONCL"}
        methods_parts, results_parts = [], []
        for p in parsed.get("passage", []):
            text = (p.get("content") or "").strip()
            sec = (p.get("section") or "").upper()
            if not text or sec == "REF":
                continue
            label = f"[{sec}]\n{text}" if sec else text
            if sec in METHODS_SECTIONS:
                methods_parts.append(label)
            elif sec in RESULTS_SECTIONS:
                results_parts.append(label)
        for t in parsed.get("table", []):
            text = (t.get("content") or "").strip()
            tab_id = t.get("tab_id") or ""
            ele_type = t.get("tab_ele_type") or ""
            if text:
                results_parts.append(f"[TABLE {tab_id} {ele_type}]\n{text}".strip())
        return {
            "methods": "\n\n".join(methods_parts),
            "results": "\n\n".join(results_parts),
        }

    # ── Plain text / Markdown (PDF) ───────────────────────────────────────────
    # Split on common section headers (markdown headings or all-caps lines)
    METHODS_HEADERS = r"(?im)^#{0,3}\s*(methods?|study design|participants?|eligibility|statistical analysis)\s*$"
    RESULTS_HEADERS = r"(?im)^#{0,3}\s*(results?|findings?|discussion|conclusions?)\s*$"

    methods_match = re.search(METHODS_HEADERS, content)
    results_match = re.search(RESULTS_HEADERS, content)

    if methods_match and results_match and methods_match.start() < results_match.start():
        methods_text = content[methods_match.start():results_match.start()].strip()
        results_text = content[results_match.start():].strip()
        return {"methods": methods_text, "results": results_text}

    # Fallback: pass full text to both calls
    return {"methods": content, "results": content}


def combine_blocks_text(blocks, format="xml"):
    # combine the blocks

    new_blocks = list(set(blocks))
    if format == "xml":
        new_block_strs = [f"<source id=\"{i}\"><content>{block}</content></source>" for i,block in enumerate(new_blocks)]
    else:
        new_block_strs = [f"[[citation:{i}]] {block}" for i,block in enumerate(new_blocks)]
    combined = '\n\n'.join(new_block_strs)
    return combined, new_blocks


def parse_paper_sections(content: str) -> Dict[str, str]:
    """
    Parse full-text content into four structured sections:
    methods, results, discussion, and tables.

    Handles two formats:
    - BioC XML (from PubMed Central): uses section_type infon tags via parse_bioc_xml
    - Markdown (from Jina/Firecrawl): uses heading patterns and table rows

    Args:
        content: Raw full-text content (BioC XML string or Markdown string)

    Returns:
        Dict with keys 'methods', 'results', 'discussion', 'tables'.
        Missing sections are empty strings.
    """
    if "<passage>" in content and "<infon" in content:
        return _parse_sections_from_bioc(content)
    else:
        return _parse_sections_from_markdown(content)


def _parse_sections_from_bioc(xml_content: str) -> Dict[str, str]:
    """Extract sections from BioC XML using parse_bioc_xml."""
    parsed = parse_bioc_xml(xml_content)

    # BioC section_type values for each target section
    method_types  = {"METHODS", "METHOD", "MATERIALS", "MATERIALS|METHODS"}
    result_types  = {"RESULTS", "RESULT", "RESULTS|DISCUSSION"}
    discuss_types = {"DISCUSS", "DISCUSSION", "RESULTS|DISCUSSION", "CONCLUSIONS", "CONCLUSION"}

    methods_parts  = []
    results_parts  = []
    discuss_parts  = []

    for p in parsed.get("passage", []):
        sec = (p.get("section") or "").upper().strip()
        text = (p.get("content") or "").strip()
        if not text:
            continue
        if sec in method_types:
            methods_parts.append(text)
        elif sec in result_types:
            results_parts.append(text)
        elif sec in discuss_types:
            discuss_parts.append(text)

    # Tables: join caption + element content per table
    table_parts = []
    for t in parsed.get("table", []):
        tab_id   = t.get("tab_id") or ""
        ele_type = t.get("tab_ele_type") or ""
        text     = (t.get("content") or "").strip()
        if text:
            label = f"[Table {tab_id} | {ele_type}]" if tab_id else "[Table]"
            table_parts.append(f"{label}\n{text}")

    return {
        "methods":    "\n\n".join(methods_parts),
        "results":    "\n\n".join(results_parts),
        "discussion": "\n\n".join(discuss_parts),
        "tables":     "\n\n".join(table_parts),
    }


def _parse_sections_from_markdown(md_content: str) -> Dict[str, str]:
    """Extract sections from Markdown using heading patterns and table rows."""
    lines = md_content.split("\n")

    # Map each heading to its line index and normalised name
    heading_re = re.compile(r"^#{1,4}\s+(.+)$")
    headings = []  # list of (line_idx, heading_text)
    for i, line in enumerate(lines):
        m = heading_re.match(line.strip())
        if m:
            headings.append((i, m.group(1).strip()))

    # Keyword patterns for each section
    method_re  = re.compile(r"method|material|participant|patient|design|setting|eligib|inclusion", re.IGNORECASE)
    result_re  = re.compile(r"result|finding|outcome|efficacy|effect", re.IGNORECASE)
    discuss_re = re.compile(r"discussion|conclusion|interpretation|implication|limitation", re.IGNORECASE)

    buckets: Dict[str, List[str]] = {"methods": [], "results": [], "discussion": []}

    for idx, (line_num, heading_text) in enumerate(headings):
        end = headings[idx + 1][0] if idx + 1 < len(headings) else len(lines)
        section_lines = lines[line_num:end]
        body = "\n".join(section_lines).strip()
        if not body:
            continue
        if method_re.search(heading_text):
            buckets["methods"].append(body)
        elif result_re.search(heading_text):
            buckets["results"].append(body)
        elif discuss_re.search(heading_text):
            buckets["discussion"].append(body)

    # Tables: collect caption lines and header rows of markdown tables
    table_parts = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Markdown table row
        if line.strip().startswith("|"):
            # Collect caption from the preceding non-empty line if it looks like a label
            caption = ""
            if i > 0:
                prev = lines[i - 1].strip()
                if re.match(r"^(Table|Tbl\.?)\s*\d+", prev, re.IGNORECASE):
                    caption = prev
            # Header row + separator
            header_row = line.strip()
            table_entry = f"{caption}\n{header_row}".strip() if caption else header_row
            table_parts.append(table_entry)
            # Skip remaining rows of this table
            while i < len(lines) and lines[i].strip().startswith("|"):
                i += 1
            continue
        # Standalone table caption (no table body found yet)
        if re.match(r"^(Table|Tbl\.?)\s*\d+", line.strip(), re.IGNORECASE):
            table_parts.append(line.strip())
        i += 1

    return {
        "methods":    "\n\n".join(buckets["methods"]),
        "results":    "\n\n".join(buckets["results"]),
        "discussion": "\n\n".join(buckets["discussion"]),
        "tables":     "\n\n".join(table_parts),
    }

