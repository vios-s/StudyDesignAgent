import os
import re
import json
import requests
import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict

logger = logging.getLogger(__name__)

from dotenv import load_dotenv
from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.toolkits import FunctionTool

from utils import *
from tools import *
from pubmed_tools import *
from prompts.literature_process import _query_refine_system_prompt, _query_refine_user_prompt


# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────

def _batch_dois_to_pmids(dois: List[str]) -> List[str]:
    """Convert a list of DOIs to PMIDs via NCBI idconv (batch, up to 100 per request)."""
    pmids = []
    for i in range(0, len(dois), 100):
        chunk = dois[i:i + 100]
        try:
            resp = requests.get(
                "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/",
                params={"ids": ",".join(chunk), "format": "json"},
                timeout=15,
            )
            resp.raise_for_status()
            for record in resp.json().get("records", []):
                pid = record.get("pmid")
                if pid:
                    pmids.append(str(pid))
        except Exception as e:
            logger.warning(f"idconv batch failed: {e}")
    return pmids


def get_references_of_paper(pmid: str, api_key: str = None) -> List[str]:
    """
    Fetch reference PMIDs of a paper.
    Primary: PubMed elink (fast, when available).
    Fallback: CrossRef API → NCBI idconv (DOI-based, when elink returns nothing).
    """
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    # Primary: elink
    params: Dict[str, str] = {
        "db": "pubmed",
        "id": str(pmid),
        "linkname": "pubmed_pubmed_refs",
        "retmode": "json",
    }
    if api_key:
        params["api_key"] = api_key
    try:
        resp = requests.get(f"{BASE_URL}/elink.fcgi", params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        linksetdbs = data.get("linksets", [{}])[0].get("linksetdbs", [])
        refs = [str(x) for x in (linksetdbs[0].get("links", []) if linksetdbs else [])]
        if refs:
            return refs
    except Exception as e:
        logger.warning(f"elink failed for PMID {pmid}: {e}")

    # Fallback: CrossRef
    logger.info(f"[PMID {pmid}] elink empty, trying CrossRef...")
    try:
        pubmed = PubMedToolkit()
        summaries = pubmed._batch_get_summaries([str(pmid)])
        elocationid = summaries[0].get("doi", "") if summaries else ""
        doi_match = re.search(r'10\.\d{4,}/\S+', elocationid)
        if not doi_match:
            logger.warning(f"[PMID {pmid}] No DOI found, skipping CrossRef")
            return []
        doi = doi_match.group(0).rstrip(".,;)")

        cr_resp = requests.get(
            f"https://api.crossref.org/works/{doi}",
            headers={"User-Agent": "StudyDesignAgent/1.0 (mailto:researcher@example.com)"},
            timeout=15,
        )
        cr_resp.raise_for_status()
        ref_list = cr_resp.json().get("message", {}).get("reference", [])
        ref_dois = [
            r["DOI"].strip() for r in ref_list if r.get("DOI")
        ]
        if not ref_dois:
            logger.warning(f"[PMID {pmid}] CrossRef returned no reference DOIs")
            return []
        print(f"[Line 1] CrossRef found {len(ref_dois)} reference DOIs for PMID {pmid}, converting to PMIDs...")
        return _batch_dois_to_pmids(ref_dois)
    except Exception as e:
        logger.warning(f"CrossRef fallback failed for PMID {pmid}: {e}")
        return []


def fetch_papers_by_pmids(pmids: List[str], data_cols: List[str]) -> List[Dict]:
    """Fetch paper metadata for a list of PMIDs using batch API calls."""
    if not pmids:
        return []
    pubmed = PubMedToolkit()
    summaries = pubmed._batch_get_summaries(pmids)
    abstracts_map = pubmed._get_abstracts_batch(pmids)
    for paper in summaries:
        paper["abstract"] = abstracts_map.get(paper["id"])
    return [{col: paper.get(col) for col in data_cols} for paper in summaries]


def deduplicate_papers(
    papers1: List[Dict], papers2: List[Dict], id_col: str = "id"
) -> pd.DataFrame:
    """Merge two paper lists and deduplicate by PMID."""
    df1 = pd.DataFrame(papers1) if papers1 else pd.DataFrame()
    df2 = pd.DataFrame(papers2) if papers2 else pd.DataFrame()
    combined = pd.concat([df1, df2], ignore_index=True)
    if id_col in combined.columns:
        combined = combined.drop_duplicates(subset=[id_col], keep="first")
    return combined.reset_index(drop=True)


def extract_query_from_response(text: str) -> str:
    """Parse the refined query string from an agent response."""
    clean = text.strip()
    # Strip markdown code fences
    if clean.startswith("```"):
        parts = clean.split("```")
        clean = parts[1] if len(parts) > 1 else clean
        if clean.startswith("json"):
            clean = clean[4:]
        clean = clean.strip()
    try:
        obj = json.loads(clean)
        if isinstance(obj, dict):
            return obj.get("Query", obj.get("query", text)).strip()
    except Exception:
        pass
    return text.strip()


# ──────────────────────────────────────────────────────────────────────────────
# Line workers (run concurrently)
# ──────────────────────────────────────────────────────────────────────────────

def run_line1(initial_query: str, data_cols: List[str], date_range: Dict = None) -> List[Dict]:
    """LINE 1: initial query → 20 SRs → extract included studies."""
    print("[Line 1] Searching systematic reviews / meta-analyses...")
    pubmed = PubMedToolkit()
    api_key = os.getenv("PUBMED_API_KEY")
    sr_pmids = pubmed.search_ids(
        initial_query,
        max_results=20,
        sort="relevance",
        publication_type=["Systematic Review", "Meta-Analysis"],
        date_range=date_range,
    )
    print(f"[Line 1] Found {len(sr_pmids)} systematic reviews")

    included_pmids: set = set()
    for sr_pmid in sr_pmids:
        included_pmids.update(get_references_of_paper(sr_pmid, api_key))
    included_pmids -= set(sr_pmids)
    print(f"[Line 1] Extracted {len(included_pmids)} unique included studies")

    papers = fetch_papers_by_pmids(list(included_pmids), data_cols)

    # Remove any references that are themselves reviews / SRs / meta-analyses
    _review_types = {"systematic review", "meta-analysis", "review"}
    before = len(papers)
    papers = [
        p for p in papers
        if not any(
            pt.lower() in _review_types
            for pt in (p.get("publication_types") or [])
        )
    ]
    print(f"[Line 1] Removed {before - len(papers)} review-type references")
    print(f"[Line 1] Done — {len(papers)} papers")
    return papers


def run_line2(
    initial_query: str,
    data_cols: List[str],
    model,
    research_question: str,
    data_input: str,
    date_range: Dict = None,
) -> List[Dict]:
    """LINE 2: initial query → 100 papers → refine → search 1 000 papers."""
    print("[Line 2] Extracting MeSH terms and keywords from 100 papers...")
    term_data = MeshQueryTool().refine_query(initial_query, max_results=100)
    mesh_term_counts = term_data.get("mesh_term_counts", {})
    keyword_counts   = term_data.get("keyword_counts",   {})
    print(f"[Line 2] {len(mesh_term_counts)} MeSH terms, {len(keyword_counts)} keywords extracted")

    mesh_query_tool_fn = FunctionTool(MeshQueryTool().build_query)
    refine_agent = ChatAgent(
        system_message=_query_refine_system_prompt(),
        tools=[mesh_query_tool_fn],
        model=model,
    )
    refine_response = refine_agent.step(
        _query_refine_user_prompt(research_question, data_input, mesh_term_counts, keyword_counts)
    )
    refined_query = extract_query_from_response(refine_response.msgs[0].content)

    # Line 2 targets primary studies; Line 1 already handles systematic reviews/meta-analyses
    # separately (via their reference lists), so exclude review-type publications here.
    refined_query = (
        f'({refined_query}) NOT ("Systematic Review"[Publication Type] OR '
        f'"Meta-Analysis"[Publication Type] OR Review[Publication Type])'
    )
    print(f"[Line 2] Refined query:\n{refined_query}")

    line2_raw = pubmed_search_tool(
        query=refined_query,
        max_results=2000,
        sort="relevance",
        date_range=date_range,
        data_cols=data_cols,
        if_save=False,
    )
    papers: List[Dict] = line2_raw if isinstance(line2_raw, list) else []
    print(f"[Line 2] Done — {len(papers)} papers")
    return papers


# ──────────────────────────────────────────────────────────────────────────────
# Agent system message (Line 2 – query refinement only)
# ──────────────────────────────────────────────────────────────────────────────



# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    load_dotenv()

    # ── Configuration ──────────────────────────────────────────────────────────
    save_folder = ""
    os.makedirs(save_folder, exist_ok=True)
    api_key = os.getenv("PUBMED_API_KEY")

    data_cols = [
        "id", "title", "authors", "pub_date", "journal",
        "publication_types", "abstract", "doi",
    ]


    # Example PECO input (output format of the PECO extraction step)
    data_input = ""
    peco = json.loads(data_input)
    _int_items = [item for item in peco.get("exposure", []) if item.get("raw_term")]
    exposure = _int_items[0]["raw_term"].split()[0] if _int_items else "exposure"

    _out_items = [item for item in peco.get("outcome", []) if item.get("raw_term")]
    outcome = _out_items[0]["raw_term"].split()[0] if _out_items else "outcome"

    date_range = {"from": "", "to": ""}

    research_question = "" 
    # ── Model ──────────────────────────────────────────────────────────────────
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
        model_type="openai/gpt-5.4-nano",
        url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model_config_dict={"temperature": 0.2, "max_tokens": 100000},
    )

    # ── Step 0: Generate Initial Query from PECO ───────────────────────────────
    print("\n=== Generating Initial Query ===")
    initial_query = MeshQueryTool().build_query(data_input)
    print(f"Initial query:\n{initial_query}\n")

    # ═══════════════════════════════════════════════════════════════════════════
    # Run LINE 1 and LINE 2 simultaneously
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n=== Running Line 1 and Line 2 simultaneously ===")
    with ThreadPoolExecutor(max_workers=2) as executor:
        future1 = executor.submit(run_line1, initial_query, data_cols, date_range)
        future2 = executor.submit(
            run_line2, initial_query, data_cols, model, research_question, data_input, date_range
        )
        line1_papers = future1.result()
        line2_papers = future2.result()

    # ═══════════════════════════════════════════════════════════════════════════
    # MERGE & DEDUPLICATE
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n=== Merging and Deduplicating Results ===")
    final_df = deduplicate_papers(line1_papers, line2_papers)

    print(f"  Line 1 papers      : {len(line1_papers)}")
    print(f"  Line 2 papers      : {len(line2_papers)}")
    print(f"  Combined (deduped) : {len(final_df)} unique papers")

    combined_path = os.path.join(save_folder, f"pubmed_results_{exposure}_{outcome}.csv")
    final_df.to_csv(combined_path, index=False)
    print(f"Saved combined results → {combined_path}")


if __name__ == "__main__":
    main()
