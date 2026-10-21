import json
import time
import random
import os
import io
import csv
import requests
import pandas as pd
from utils import *
import xml.etree.ElementTree as ET
from camel.toolkits import FunctionTool, SearchToolkit, HumanToolkit, FileToolkit, BrowserToolkit, PubMedToolkit
from camel.logger import get_logger
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Set, Tuple, Union
from camel.loaders import Firecrawl

logger = get_logger(__name__)

class ParseFile:
    def __init__(self, text):
        self.text = text

    def _parse_key_value_lines(self, text: str) -> List[Dict[str, str]]:
        rows = []
        for line in text.splitlines():
            if not line.strip():
                continue
            if ":" in line:
                k, v = line.split(":", 1)
                rows.append({"name": k.strip(), "description": v.strip()})
            elif "-" in line:  # allow "Name - Description"
                k, v = line.split("-", 1)
                rows.append({"name": k.strip(), "description": v.strip()})
        return rows

    def _parse_csv(self, text: str) -> List[Dict[str, str]]:
        rows = []
        try:
            reader = csv.reader(io.StringIO(text))
            for r in reader:
                if len(r) >= 2:
                    rows.append({"name": r[0].strip(), "description": r[1].strip()})
        except Exception:
            pass
        return rows

    def _parse_markdown_table(self, text: str) -> List[Dict[str, str]]:
        # crude MD table parser: header line with pipes + separator + rows
        lines = [ln for ln in text.splitlines() if "|" in ln]
        if len(lines) < 2:
            return []
        # try to detect columns
        header_cells = [c.strip().lower() for c in lines[0].strip("|").split("|")]
        try:
            name_idx = next(i for i, c in enumerate(header_cells) if c in ("name", "variable", "field"))
            desc_idx = next(i for i, c in enumerate(header_cells) if c in ("description", "desc", "details"))
        except StopIteration:
            return []
        rows = []
        for ln in lines[2:]:  # skip header + separator
            cells = [c.strip() for c in ln.strip("|").split("|")]
            if len(cells) <= max(name_idx, desc_idx):
                continue
            n, d = cells[name_idx], cells[desc_idx]
            if n:
                rows.append({"name": n, "description": d})
        return rows

    def _parse_json(self, text: str) -> List[Dict[str, str]]:
        try:
            data = json.loads(text)
            if isinstance(data, list):
                out = []
                for item in data:
                    if isinstance(item, dict):
                        # accept keys: name/variable, description/desc
                        name = item.get("name") or item.get("variable")
                        desc = item.get("description") or item.get("desc")
                        if name and desc:
                            out.append({"name": str(name).strip(), "description": str(desc).strip()})
                return out
        except Exception:
            pass
        return []

    def parse_variables_block(self) -> Dict[str, Any]:
        """
        Accepts JSON array, key:value lines, CSV (2 cols), or Markdown table.
        Returns: {"ok": True, "variables": [{"name":..., "description":...}, ...], "n": int}
                 or {"ok": False, "error": "..."}
        """
        text = self.text.strip()
        candidates = []
        for parser in (self._parse_json, self._parse_markdown_table, self._parse_csv, self._parse_key_value_lines):
            rows = parser(text)
            candidates.append(rows)
        # pick the richest parse
        rows = max(candidates, key=lambda r: len(r))
        # dedupe & clean
        uniq, seen = [], set()
        for r in rows:
            name = r.get("name", "").strip()
            desc = r.get("description", "").strip()
            if not name or not desc:
                continue
            key = (name.lower(), desc.lower())
            if key not in seen:
                seen.add(key)
                uniq.append({"name": name, "description": desc})
        if not uniq:
            return {
                "Error": "Could not parse any (name, description) pairs. Accepts JSON array; key: value lines; CSV (2 columns); or Markdown table."}
        return {"Variables": uniq, "n": len(uniq)}


class MeshLookUp:
    def __init__(self):
        self.LOOKUP_DESC = "https://id.nlm.nih.gov/mesh/lookup/descriptor"
        self.SPARQL_ENDPOINT = "https://id.nlm.nih.gov/mesh/sparql"

    def top1_mesh_descriptor(self, term: str) -> Tuple[str, str]:
        """
        Step 1: Get the UI (ID) and Standard Label.
        Uses exact match first, then contains match with validation.
        """
        term = term.strip()
        term_lower = term.lower()
        try:
            # Try exact match first
            params = {"label": term, "match": "exact", "limit": 1}
            r = requests.get(self.LOOKUP_DESC, params=params, timeout=10)
            r.raise_for_status()
            hits = r.json()

            if not hits:
                # Try "contains" but filter results to ensure semantic relevance
                params["match"] = "contains"
                params["limit"] = 1 # Get more results to filter
                r = requests.get(self.LOOKUP_DESC, params=params, timeout=10)
                r.raise_for_status()
                hits = r.json()

                # if hits:
                #     # Filter: prefer labels that START with the term or ARE the term
                #     # This prevents "adults" from matching "Latent Autoimmune Diabetes in Adults"
                #     filtered_hits = []
                #     for hit in hits:
                #         label_lower = hit["label"].lower()
                #         # Accept if: label starts with term, or term equals label, or label is singular/plural of term
                #         if (label_lower.startswith(term_lower) or
                #             label_lower == term_lower or
                #             label_lower == term_lower + "s" or
                #             label_lower + "s" == term_lower or
                #             term_lower.startswith(label_lower)):
                #             filtered_hits.append(hit)

                #     # Use filtered results if any, otherwise return None (no valid match)
                #     hits = filtered_hits if filtered_hits else []

            if not hits:
                raise RuntimeError(f"No descriptor found for term={term!r}")

            uri = hits[0]["resource"]
            ui = uri.rstrip("/").split("/")[-1]
            return ui, hits[0]["label"]

        except requests.RequestException as e:
            raise RuntimeError(f"Network error during MeSH lookup: {e}")

    def _get_details_sparql(self, ui: str) -> Dict[str, str]:
        """
        Step 2: Traverse Graph.
        Excludes 'PublicationType' from 'seeAlso' results.
        """
        ui = ui.strip()
        query = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX meshv: <http://id.nlm.nih.gov/mesh/vocab#>
        PREFIX mesh: <http://id.nlm.nih.gov/mesh/>

        SELECT DISTINCT ?scopeNote ?entryTerm ?seeAlsoLabel ?broaderLabel
        WHERE {{
            BIND(mesh:{ui} AS ?d)

            # 1. SCOPE NOTE (Property Path via Preferred Concept)
            OPTIONAL {{ ?d meshv:preferredConcept/meshv:scopeNote ?scopeNote . }}

            # 2. ENTRY TERMS (Property Path via Preferred Concept)
            OPTIONAL {{ ?d meshv:preferredConcept/meshv:term/meshv:prefLabel ?entryTerm . }}

        }}
        """

        try:
            r = requests.get(
                self.SPARQL_ENDPOINT,
                params={"query": query, "format": "JSON"},
                timeout=10
            )
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            return {
                "scope_note": f"Error: {e}",
            }

        results = data.get("results", {}).get("bindings", [])

        # Containers
        scope_note = ""
        entry_terms = set()
        see_also_refs = set()
        broader_descs = set()

        for item in results:
            if not scope_note and "scopeNote" in item:
                scope_note = item["scopeNote"]["value"]


        return {
            "scope_note": scope_note,
        }

    def look_up_tool(self, term: str) -> Dict[str, Any]:
        """
            Standardize the term with MeSH vocabulary
            Args:
                term: The searched term to be standardized.
        """
        try:
            ui, label = self.top1_mesh_descriptor(term)
            details = self._get_details_sparql(ui)

            return {
                'Original term': term,
                'Standard Mesh term': label,
                'MeSH UI': ui,
                'scope_note': details['scope_note'],
            }
        except RuntimeError as e:
            return {'Original term': term, 'Error': str(e)}











