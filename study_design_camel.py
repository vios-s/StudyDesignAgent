import os
import asyncio
import traceback
import textwrap
from getpass import getpass
from dotenv import load_dotenv
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.configs import ChatGPTConfig
from camel.toolkits import FunctionTool, SearchToolkit, HumanToolkit, FileToolkit, BrowserToolkit, PubMedToolkit
from camel.types import ModelPlatformType, ModelType
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from pydantic import BaseModel, Field
from pubmed_tools import *
from response_format import *
from typing import List, Optional, Union
from prompts.task_process import (
    _task_assessment_prompt,
    _variable_extract_prompt,
    _design_agent_system_prompt,
    _task_assessment_user_prompt,
    _peco_extraction_user_prompt,
    TASK_ASSESSOR_ROLE,
    TASK_ASSESSOR_PERSONA,
    PECO_EXTRACTOR_ROLE,
    PECO_EXTRACTOR_PERSONA,
)
from prompts.literature_process import (
    _query_generate_prompt,
    _literature_search_prompt
)

from test_search import (
    run_line1,
    run_line2,
    deduplicate_papers
)
from utils import *
from tools import *
from literature_screen import DualReviewerScreener
from test_browser import run_fulltext_screening, run_variable_extraction
from test_dag_construct import construct_dag_ensemble_async, save_dag_graph, plot_dag_with_scores
from adj_tools import compute_adjustment_set, find_causal_adjustment_set, build_metadata_match_nodes, \
    save_metadata_match_nodes, print_adjustment_summary
from metadata_match import match_variables, write_match_csv, print_match_summary
from pubmed_tools import get_fulltext

import re
import json
import hashlib
import networkx as nx


class WorkflowStopped(Exception):
    """Raised when the user stops the workflow at a human review checkpoint."""


def _slug(s: str) -> str:
    """Same filename sanitisation as test_browser.run_fulltext_screening / run_variable_extraction."""
    return re.sub(r'[^\w\-]', '_', s).strip('_')


def _file_digest(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def _load_json_list(path: str, required_keys: tuple):
    """Load a JSON list of objects; returns (data, error_message)."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return None, f"Could not parse JSON: {e}"
    if not isinstance(data, list):
        return None, "Expected a JSON list at the top level"
    for i, item in enumerate(data):
        if not isinstance(item, dict) or any(k not in item for k in required_keys):
            return None, f"Entry {i} must be an object with keys {list(required_keys)}"
    return data, None


def _validate_nodes_json(path: str):
    return _load_json_list(path, ("variable",))[1]


def _validate_edges_json(path: str):
    edges, error = _load_json_list(path, ("from", "to"))
    if error:
        return error
    G = nx.DiGraph([(e["from"], e["to"]) for e in edges])
    if not nx.is_directed_acyclic_graph(G):
        cycle = " → ".join(src for src, _ in nx.find_cycle(G))
        return f"Edges contain a cycle ({cycle}); the graph must be a DAG"
    return None


def _validate_included_csv(path: str):
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception as e:
        return f"Could not read CSV: {e}"
    if "id" not in df.columns:
        return "CSV must keep the 'id' column"
    return None


class StudyDesignFramework:
    def __init__(self, save_folder, date_range, human_in_the_loop: bool = True,
                 hidden_variables: Optional[List[str]] = None,
                 dag_num_runs: int = 5, dag_score_threshold: float = 0.5,
                 metadata_db_path: Optional[str] = "metadata_database/dataloch_db/dataloch_metadata.db",
                 fulltext_resource_folder: str = "test_case/low_dose_resource"):
        """
        Args:
            human_in_the_loop: if True, pause for human review after PECO extraction,
                full-text screening, DAG node construction, DAG construction and
                adjustment-set selection. If False, run the whole pipeline end-to-end.
            hidden_variables: DAG nodes treated as unobserved when computing the adjustment set.
            dag_num_runs / dag_score_threshold: ensemble DAG construction settings.
            metadata_db_path: metadata SQLite DB used to check that adjustment variables exist
                in the data. None skips metadata matching.
            fulltext_resource_folder: folder of local {pmid}.pdf files used by get_fulltext
                (same folder as full-text screening), e.g. for papers added during review.
        """
        self.workforce = None
        self.MeSH_url = 'https://meshb.nlm.nih.gov/'
        self.mesh_tool_kit = MeshLookUp()
        self.pubmed_tool_kit = PubMedToolkit()
        self.save_folder = save_folder
        self.date_range = date_range
        self.human_in_the_loop = human_in_the_loop
        self.hidden_variables = list(hidden_variables or [])
        self.dag_num_runs = dag_num_runs
        self.dag_score_threshold = dag_score_threshold
        self.metadata_db_path = metadata_db_path
        self.fulltext_resource_folder = fulltext_resource_folder


        # Workflow state
        self.state = {
            "research_question": "",
            "task_type": "",
            "peco_elements": {},
            "literature_results": [],
            "screening_results": {},
            "extracted_variables": {},
            "dag_nodes": [],
            "dag": [],
            "minimum_adjustment_set": [],
            "hidden_variables": [],
            "assumptions": [],
            "metadata_refinement": [],
            "human_feedback": []
        }

    def _human_checkpoint(self, stage: str, artifact_path: str, summary: str = "", validate=None) -> bool:
        """
        Pause so the user can review (and edit in place) an artifact before the next step.
        No-op when human_in_the_loop is False. Returns True if the user modified the file.
        """
        if not self.human_in_the_loop:
            return False

        before = _file_digest(artifact_path)
        print("\n" + "-" * 80)
        print(f"🧑 HUMAN REVIEW — {stage}")
        print("-" * 80)
        if summary:
            print(summary)
        print(f"Review file: {os.path.abspath(artifact_path)}")
        print("To make changes, edit and save the file in place (keep the same format).")
        while True:
            choice = input("  [Enter] continue   [q] stop the workflow\n> ").strip().lower()
            if choice in ("q", "quit", "stop"):
                logger.info(f"Workflow stopped by user at '{stage}' review")
                raise WorkflowStopped(f"Stopped by user at '{stage}' review")
            if choice not in ("", "c", "continue"):
                continue
            error = validate(artifact_path) if validate else None
            if error:
                print(f"⚠️  {error}. Fix the file and press Enter again.")
                continue
            modified = _file_digest(artifact_path) != before
            self.state["human_feedback"].append({"stage": stage, "artifact": artifact_path, "modified": modified})
            logger.info(f"Human review '{stage}': {'modified' if modified else 'confirmed'}")
            print("✓ Edits detected — continuing with the edited file." if modified else "✓ Confirmed — continuing.")
            return modified

    def create_design_agent(self, role_name: str, persona: str, task: str, tools=None) -> ChatAgent:
        """Create an OpenAI agent with specific role and persona"""

        msg_content = _design_agent_system_prompt(role_name, persona, task)

        sys_msg = BaseMessage.make_assistant_message(
            role_name=role_name,
            content=msg_content,
        )

        # Configure Claude model
        model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_type="openai/gpt-5.4",
            url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            model_config_dict={"temperature": 0.2, "max_tokens": 100000},
        )

        agent = ChatAgent(
            system_message=sys_msg,
            model=model,
            tools=tools or [],
            step_timeout=300,  # 5 minutes per step timeout
        )

        return agent
    def _run_task_assessment(self, research_aim):
        task_assessor = self.create_design_agent(
            role_name=TASK_ASSESSOR_ROLE,
            persona=TASK_ASSESSOR_PERSONA,
            task=_task_assessment_prompt(),
            tools=None,
        )
        class TaskResponse(BaseModel):
            task_type: str = Field(description="Select from causal, not_applicable, unclear")
            rationale: str = Field(description="1-3 sentences explaining your decision")
        response = task_assessor.step(_task_assessment_user_prompt(research_aim), response_format=TaskResponse)
        parsed_response = response.msgs[0].parsed
        self.state["task_type"] = parsed_response.task_type
        return parsed_response.task_type, parsed_response.rationale

    def _run_task_parser(self, research_aim):
        mesh_search_tool = FunctionTool(self.mesh_tool_kit.look_up_tool)
        # PECO review happens inside the agent via the human tool (Steps 4-5 of the prompt)
        human_loop_tool = HumanToolkit().get_tools() if self.human_in_the_loop else []

        # Create agents

        variable_extractor = self.create_design_agent(
            role_name=PECO_EXTRACTOR_ROLE,
            persona=PECO_EXTRACTOR_PERSONA,
            task=_variable_extract_prompt(human_in_the_loop=self.human_in_the_loop),
            tools=[mesh_search_tool, *human_loop_tool],
        )
        response = variable_extractor.step(_peco_extraction_user_prompt(research_aim), response_format=PECOExtraction)
        print(response.msgs[0].content)
        self.state["peco_elements"] = response.msgs[0].content
        return response.msgs[0].content

    def _run_literature_search(self, peco_result, research_question):
        peco = json.loads(peco_result)
        _int_items = [item for item in peco.get("exposure", []) if item.get("raw_term")]
        exposure = _int_items[0]["raw_term"].split()[0] if _int_items else "exposure"

        _out_items = [item for item in peco.get("outcome", []) if item.get("raw_term")]
        outcome = _out_items[0]["raw_term"].split()[0] if _out_items else "outcome"
        data_cols = [
            "id", "title", "authors", "pub_date", "journal",
            "publication_types", "abstract", "doi",
        ]
        
        model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_type="openai/gpt-5.4-nano",
            url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            model_config_dict={"temperature": 0.2, "max_tokens": 100000},
        )

        # ── Step 0: Generate Initial Query from PECO ───────────────────────────────
        print("\n=== Generating Initial Query ===")
        initial_query = MeshQueryTool().build_query(peco_result)
        logger.info(f"Initial query:\n{initial_query}")

        # ═══════════════════════════════════════════════════════════════════════════
        # Run LINE 1 and LINE 2 simultaneously
        # ═══════════════════════════════════════════════════════════════════════════
        print("\n=== Running Line 1 and Line 2 simultaneously ===")
        with ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(run_line1, initial_query, data_cols, self.date_range)
            future2 = executor.submit(
                run_line2, initial_query, data_cols, model, research_question, peco_result, self.date_range
            )
            line1_papers = future1.result()
            line2_papers = future2.result()

        # ═══════════════════════════════════════════════════════════════════════════
        # MERGE & DEDUPLICATE
        # ═══════════════════════════════════════════════════════════════════════════
        print("\n=== Merging and Deduplicating Results ===")
        final_df = deduplicate_papers(line1_papers, line2_papers)

        logger.info(f"Line 1 papers      : {len(line1_papers)}")
        logger.info(f"Line 2 papers      : {len(line2_papers)}")
        logger.info(f"Combined (deduped) : {len(final_df)} unique papers")

        combined_path = os.path.join(self.save_folder, f"pubmed_results_{exposure}_{outcome}.csv")
        final_df.to_csv(combined_path, index=False)
        logger.info(f"Saved combined results -> {combined_path}")

       
        print('Literature Search Done')
        self.state["literature_results"] = f'Literature search result saved into pubmed_results_{exposure}_{outcome}.csv'
        return combined_path

    async def _run_abstract_screen(self, peco_result, research_question, input_csv, output_csv=None):
        peco = json.loads(peco_result)
        peco_values = extract_peco_values(peco)
        population = peco_values["population"]
        exposure = peco_values["exposure"]
        comparison = peco_values["comparison"]
        outcome = peco_values["outcome"]

        print("\n=== PECO Elements for Abstract Screening ===")
        print(f"Population: {population}")
        print(f"Exposure: {exposure}")
        print(f"Comparison: {comparison}")
        print(f"Outcome: {outcome}")

        screener = DualReviewerScreener(
            exposure=exposure,
            outcome=outcome,
            population=population,
            comparison=comparison,
        )

        # ═══════════════════════════════════════════════════════════════════════════
        # Run Abstract Dual-Reviewer Screening
        # ═══════════════════════════════════════════════════════════════════════════
        print("\n=== Running Abstract Dual-Reviewer Screening ===")
        screened_df = await screener.screen_papers_dual(
            input_csv=input_csv,
            research_question=research_question,
            output_csv=output_csv,
        )

        output_path = output_csv or input_csv.replace("pubmed_results", "abstract_screen")
        n_include = int((screened_df["decision"] == "INCLUDE").sum())
        print(f"  Screened            : {len(screened_df)} papers")
        print(f"  Included            : {n_include} papers")
        print(f"Saved screening results → {output_path}")

        print('Abstract Screening Done')
        self.state["abstract_screening_results"] = f'Abstract screening result saved into {os.path.basename(output_path)}'
        return output_path

    async def _run_fulltext_screen(self, peco_result, abstract_included_csv):
        peco = json.loads(peco_result)
        peco_values = extract_peco_values(peco)
        population = peco_values["population"]
        exposure = peco_values["exposure"]
        comparison = peco_values["comparison"]
        outcome = peco_values["outcome"]

        print("\n=== PECO Elements for Full-Text Screening ===")
        print(f"Population: {population}")
        print(f"Exposure: {exposure}")
        print(f"Comparison: {comparison}")
        print(f"Outcome: {outcome}")

        # ═══════════════════════════════════════════════════════════════════════════
        # Run Full-Text Dual-Reviewer Screening
        # ═══════════════════════════════════════════════════════════════════════════
        print("\n=== Running Full-Text Dual-Reviewer Screening ===")
        input_df = pd.read_csv(abstract_included_csv)
        included_df = await run_fulltext_screening(
            input_df=input_df,
            exposure=exposure,
            outcome=outcome,
            save_folder=self.save_folder,
            population=population,
            comparison=comparison,
        )

        print(f"  Screened            : {len(input_df)} papers")
        print(f"  Included            : {len(included_df)} papers")

        # Human review: the user may delete rows to exclude papers, or add rows to include papers
        included_csv = self._out_path(f"fulltext_screen_{_slug(exposure)}_{_slug(outcome)}_included.csv")
        if self._human_checkpoint(
            "Full-text screening",
            included_csv,
            summary=f"{len(included_df)}/{len(input_df)} papers included.\n"
                    "  - Delete rows to exclude papers.\n"
                    "  - Add rows to include papers: 'id' (PMID) is required, 'doi' is optional. "
                    f"Their full text is fetched again (a local copy can be placed at "
                    f"{self.fulltext_resource_folder}/<pmid>.pdf).",
            validate=_validate_included_csv,
        ):
            edited = pd.read_csv(included_csv, encoding="utf-8-sig", dtype=str, keep_default_na=False)
            edited["id"] = edited["id"].str.strip()
            edited = edited[edited["id"] != ""].drop_duplicates("id")

            included_df["id"] = included_df["id"].astype(str)
            kept_df = included_df[included_df["id"].isin(set(edited["id"]))]
            added = edited[~edited["id"].isin(set(included_df["id"]))]
            added_df = await self._fetch_added_papers(added, input_df) if not added.empty else pd.DataFrame()

            print(f"  Removed by reviewer  : {len(included_df) - len(kept_df)} papers")
            print(f"  Added by reviewer    : {len(added_df)}/{len(added)} papers (full text retrieved)")
            included_df = pd.concat([kept_df, added_df], ignore_index=True)
            print(f"  Included after review: {len(included_df)} papers")

            # Write the reconciled list back so the CSV matches what goes into variable extraction
            included_df[[c for c in included_df.columns if c != "fulltext"]].to_csv(
                included_csv, index=False, encoding="utf-8-sig"
            )

        print('Full-Text Screening Done')
        self.state["fulltext_screening_results"] = f'Full-text screening included {len(included_df)}/{len(input_df)} papers'
        return included_df

    async def _fetch_added_papers(self, added: pd.DataFrame, input_df: pd.DataFrame) -> pd.DataFrame:
        """Fetch full text for papers the reviewer added to the full-text included list."""
        print(f"\n=== Fetching full text for {len(added)} reviewer-added paper(s) ===")
        input_by_id = {str(r["id"]): r for r in input_df.to_dict("records")}
        semaphore = asyncio.Semaphore(3)

        async def _fetch(row: dict):
            pmid = row["id"]
            # Start from the search/abstract-screen metadata if the paper is known, then apply reviewer fields
            record = dict(input_by_id.get(pmid, {}))
            record.update({k: v for k, v in row.items() if v != ""})
            doi = str(record.get("doi") or "").strip()
            doi = doi if doi and doi.lower() != "nan" else None
            async with semaphore:
                content = await asyncio.to_thread(get_fulltext, pmid, doi, self.fulltext_resource_folder)
            if not content:
                print(f"  ⚠️  Paper {pmid}: full text unavailable — not added "
                      f"(place {pmid}.pdf in {self.fulltext_resource_folder} to use a local copy)")
                return None
            print(f"  Paper {pmid}: full text retrieved")
            return {**record, "id": pmid, "doi": doi or "",
                    "fulltext_decision": "INCLUDE_BY_REVIEWER", "fulltext": content}

        records = await asyncio.gather(*(_fetch(r) for r in added.to_dict("records")))
        return pd.DataFrame([r for r in records if r])

    def _out_path(self, filename: str) -> str:
        return os.path.join(self.save_folder, filename) if self.save_folder else filename

    async def _run_variable_extraction(self, peco_result, included_df):
        """Returns the path of the DAG nodes JSON, or None if no nodes were produced."""
        peco = json.loads(peco_result)
        peco_values = extract_peco_values(peco)
        population = peco_values["population"]
        exposure = peco_values["exposure"]
        comparison = peco_values["comparison"]
        outcome = peco_values["outcome"]

        if included_df.empty:
            print("\nNo papers passed full-text screening — skipping variable extraction.")
            self.state["extracted_variables"] = "No papers available for variable extraction"
            self.state["dag_nodes"] = []
            return None

        # ═══════════════════════════════════════════════════════════════════════════
        # Run Variable Extraction → Deduplication → DAG Node Construction
        # ═══════════════════════════════════════════════════════════════════════════
        print("\n=== Running Variable Extraction ===")
        dag_df = await run_variable_extraction(
            included_df=included_df,
            exposure=exposure,
            outcome=outcome,
            save_folder=self.save_folder,
            population=population,
            comparator=comparison,
        )

        print(f"  DAG nodes extracted : {len(dag_df)}")
        if dag_df.empty:
            self.state["extracted_variables"] = f'No DAG nodes extracted from {len(included_df)} papers'
            self.state["dag_nodes"] = []
            return None

        # Human review: the user may rename/delete/add nodes or change roles in the JSON
        dag_nodes_json = self._out_path(f"extracted_variables_{_slug(exposure)}_{_slug(outcome)}_dag.json")
        self._human_checkpoint(
            "DAG node construction",
            dag_nodes_json,
            summary=f"{len(dag_df)} DAG nodes. Each node needs a 'variable' and 'roles' "
                    "(nodes whose only role is 'selection_variable' are kept out of the DAG).",
            validate=_validate_nodes_json,
        )
        dag_nodes, _ = _load_json_list(dag_nodes_json, ("variable",))

        print('Variable Extraction Done')
        self.state["extracted_variables"] = f'Extracted {len(dag_nodes)} distinct DAG nodes from {len(included_df)} papers'
        self.state["dag_nodes"] = dag_nodes
        return dag_nodes_json

    async def _run_dag_construction(self, peco_result, research_question, dag_nodes_json):
        """Returns the path of the DAG edges JSON, or None if construction failed."""
        peco_values = extract_peco_values(json.loads(peco_result))
        treatment = peco_values["exposure"]
        outcome = peco_values["outcome"]

        final_edges = await construct_dag_ensemble_async(
            json_path=dag_nodes_json,
            treatment=treatment,
            outcome=outcome,
            research_question=research_question,
            num_runs=self.dag_num_runs,
            score_threshold=self.dag_score_threshold,
        )
        if final_edges is None:
            self.state["dag"] = []
            return None

        # Output paths used by construct_dag_ensemble_async
        edges_json = dag_nodes_json.replace(".json", "_dag_edges.json")

        # Human review: the user may add/remove/reverse edges ({"from", "to", ...} objects)
        if self._human_checkpoint(
            "DAG construction",
            edges_json,
            summary=f"{len(final_edges)} edges between DAG nodes "
                    f"(plot: {os.path.abspath(dag_nodes_json.replace('.json', '_dag.png'))}). "
                    f"Treatment node: '{treatment}', outcome node: '{outcome}'.",
            validate=_validate_edges_json,
        ):
            final_edges, _ = _load_json_list(edges_json, ("from", "to"))
            # Regenerate the graph files so they match the edited edges
            save_dag_graph(final_edges, base_path=dag_nodes_json.replace(".json", "_dag"))
            try:
                plot_dag_with_scores(final_edges, treatment, outcome,
                                     out_path=dag_nodes_json.replace(".json", "_dag.png"))
            except Exception as e:
                print(f"⚠️  Could not re-plot the edited DAG: {e}")

        print('DAG Construction Done')
        self.state["dag"] = final_edges
        return edges_json

    def _run_adjustment_set(self, peco_result, dag_nodes_json, edges_json) -> dict:
        """Returns the compute_adjustment_set result (output_path = metadata_match input JSON)."""
        peco_values = extract_peco_values(json.loads(peco_result))

        result = compute_adjustment_set(
            dag_nodes_file=dag_nodes_json,
            dag_edges_file=edges_json,
            peco=peco_values,
            treatment=peco_values["exposure"],
            outcome=peco_values["outcome"],
            hidden_variables=self.hidden_variables,
        )
        print('Adjustment Set Selection Done')
        self.state["hidden_variables"] = list(self.hidden_variables)
        self.state["minimum_adjustment_set"] = _load_json_list(result["output_path"], ("variable",))[0]
        return result

    async def _run_metadata_refinement(self, peco_result, dag_nodes_json, edges_json, adj_result) -> dict:
        """
        Match adjustment/eligibility variables against the metadata DB and repair the set.

        For each variable not found in the DB:
          - eligibility variable (not needed for adjustment): drop it, record an assumption;
          - adjustment variable: treat it as hidden and recompute the adjustment set.
            If a non-empty set still exists, keep the variable hidden and match any new
            variables in the next round. If the set becomes empty, un-hide it, drop it from
            the set and record an assumption that the variable was not found.
        Repeats until every variable in the set is found in the DB.
        """
        peco_values = extract_peco_values(json.loads(peco_result))
        treatment, outcome = peco_values["exposure"], peco_values["outcome"]
        with open(edges_json, encoding="utf-8") as f:
            edges = json.load(f)
        with open(dag_nodes_json, encoding="utf-8") as f:
            dag_nodes = json.load(f)

        eligibility_vars = list(adj_result["eligibility_variables"])
        hidden = list(self.hidden_variables)
        # Full tigramite set (may overlap eligibility variables); adjustment vars = causal_set - eligibility
        causal_set = find_causal_adjustment_set(edges, treatment, outcome, hidden) or []
        assumed_missing = set()                             # not found and cannot be replaced
        assumptions, refinement_log = [], []
        match_results = {}                                  # variable -> metadata_match result

        round_no = 0
        while True:
            adjustment_vars = [v for v in causal_set if v not in eligibility_vars]
            current = list(dict.fromkeys(adjustment_vars + eligibility_vars))
            to_match = [v for v in current if v not in match_results]
            if to_match:
                round_no += 1
                print(f"\n--- Metadata matching round {round_no}: {len(to_match)} variable(s) ---")
                nodes = build_metadata_match_nodes(
                    [v for v in adjustment_vars if v in to_match],
                    [v for v in eligibility_vars if v in to_match],
                    dag_nodes,
                )
                for r in await match_variables(nodes, self.metadata_db_path):
                    match_results[r["variable"]] = r

            not_found = [v for v in current
                         if match_results.get(v, {}).get("status", "not_found") == "not_found"]
            if not not_found:
                break
            v = not_found[0]

            if v in eligibility_vars:
                eligibility_vars.remove(v)
                if v not in causal_set:
                    assumed_missing.add(v)
                    assumptions.append({
                        "variable": v,
                        "type": "eligibility_variable_not_found",
                        "assumption": f"Eligibility criterion '{v}' was not found in the metadata database; "
                                      f"the cohort cannot be restricted on it.",
                    })
                    refinement_log.append({"variable": v, "action": "dropped_eligibility"})
                    print(f"  ✗ '{v}' (eligibility) not found → dropped, listed as assumption")
                    continue

            new_set = find_causal_adjustment_set(edges, treatment, outcome, hidden + [v])
            new_set = [x for x in (new_set or []) if x not in assumed_missing]
            if new_set:
                hidden.append(v)
                causal_set = new_set
                refinement_log.append({"variable": v, "action": "hidden", "new_adjustment_set": new_set})
                print(f"  ✗ '{v}' not found → treated as hidden; new adjustment set: {new_set}")
            else:
                assumed_missing.add(v)
                causal_set = [x for x in causal_set if x != v]
                assumptions.append({
                    "variable": v,
                    "type": "adjustment_variable_not_found",
                    "assumption": f"Variable not found: '{v}' is needed to block confounding between "
                                  f"'{treatment}' and '{outcome}' but is not in the metadata database, and no "
                                  f"valid adjustment set exists without it. The analysis assumes the "
                                  f"confounding through '{v}' is negligible.",
                })
                refinement_log.append({"variable": v, "action": "assumed_not_found"})
                print(f"  ✗ '{v}' not found → adjustment set would be empty; removed and listed as assumption")

        adjustment_vars = [v for v in causal_set if v not in eligibility_vars]
        print_adjustment_summary(adjustment_vars, eligibility_vars)
        print(f"\n  Hidden variables : {hidden}")
        print(f"  Assumptions      : {[a['variable'] for a in assumptions]}")

        # Final variables in metadata_match input format, annotated with their match status
        final_nodes = build_metadata_match_nodes(adjustment_vars, eligibility_vars, dag_nodes)
        for node in final_nodes:
            node["metadata_match_status"] = match_results.get(node["variable"], {}).get("status", "not_found")
        final_json = save_metadata_match_nodes(
            final_nodes, dag_nodes_json.replace(".json", "_adjustment_final.json")
        )

        all_results = list(match_results.values())
        write_match_csv(all_results, dag_nodes_json.replace(".json", "_metadata_match_results.csv"))
        print_match_summary(all_results)

        summary = {
            "adjustment_set": adjustment_vars,
            "eligibility_variables": eligibility_vars,
            "hidden_variables": hidden,
            "assumptions": assumptions,
            "refinement_log": refinement_log,
            "match_status": {v: r["status"] for v, r in match_results.items()},
        }
        with open(dag_nodes_json.replace(".json", "_adjustment_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print('Metadata Matching Done')
        self.state["minimum_adjustment_set"] = final_nodes
        self.state["hidden_variables"] = hidden
        self.state["assumptions"] = assumptions
        self.state["metadata_refinement"] = refinement_log
        return {**summary, "output_path": final_json}

    def _review_adjustment_set(self, result: dict):
        """Human review of the final adjustment set JSON."""
        lines = [f"Causal adjustment set: {result['adjustment_set']}",
                 f"Eligibility variables: {result['eligibility_variables']}"]
        if self.state["hidden_variables"]:
            lines.append(f"Hidden variables: {self.state['hidden_variables']}")
        for a in self.state["assumptions"]:
            lines.append(f"Assumption: {a['assumption']}")
        if self._human_checkpoint(
            "Adjustment set",
            result["output_path"],
            summary="\n".join(lines),
            validate=_validate_nodes_json,
        ):
            self.state["minimum_adjustment_set"] = _load_json_list(result["output_path"], ("variable",))[0]

    async def run_study_design(self, research_aim: str):
        """Run a collaborative research session"""
        logger.info("=" * 80)
        logger.info("WORKFLOW EXECUTION STARTED")
        logger.info("=" * 80)
        logger.info(f"Research Question: {research_aim}")
        logger.info(f"Human-in-the-loop: {self.human_in_the_loop}")

        print(f"🔬 Starting study design on: {research_aim}")
        print(f"   Human-in-the-loop: {'ON' if self.human_in_the_loop else 'OFF (end-to-end)'}")
        print("=" * 80)
        self.state["research_question"] = research_aim

        # Task Gate
        logger.info("Starting Step 1: Task Assessment")
        print("=" * 80)
        print("STEP 1: Task Assessment - Assess if the task is Causal or not")
        print("=" * 80)
        task_type, rationale = self._run_task_assessment(research_aim)
        task_type = task_type.lower()
        print(f'The research question is: {task_type}')
        print(f'The reason for the classification is: {rationale}')
        logger.info(f"Task type identified: {task_type}")
        if task_type != "causal":
            print("\n ⚠️WORKFLOW STOPPED")
            print("This workflow only processes causal research questions.")
            print("=" * 80)

            return {
                "status": "stopped",
                "reason": f"Task type is '{task_type}', not 'causal'",
                "research_question": research_aim,
                "message": "This workflow is designed for causal questions only. "
                           "Please reformulate your question to investigate a causal relationship, "
                           "or use a different analysis approach for associational/descriptive questions."
            }

        print("✓ Task is CAUSAL - Proceeding with workflow")
        print("=" * 80)

        try:
            await self._run_pipeline(research_aim)
        except WorkflowStopped as e:
            print(f"\n ⚠️WORKFLOW STOPPED: {e}")
            print("=" * 80)
            return {"status": "stopped", "reason": str(e), **self.state}

        print("\n" + "=" * 60)
        print("📊 Study Design RESULTS")
        print("=" * 60)
        print(self.state)

        return self.state

    async def _run_pipeline(self, research_aim: str):
        """Steps 2-9. Raises WorkflowStopped if the user stops at a review checkpoint."""
        # Step 2: Task Parser
        print("STEP 2: Task Parsing - Extract PECO elements from the research question")
        peco_result = self._run_task_parser(research_aim)

        # Step 3: Literature Search
        print("\n" + "=" * 80)
        print("STEP 3: Literature Search - Finding Relevant Papers")
        print("=" * 80)
        search_result = self._run_literature_search(peco_result, research_aim)

        # Step 4: Abstract Screening
        print("\n" + "=" * 80)
        print("STEP 4: Abstract Screening - Title/Abstract Screening with Dual Reviewers")
        print("=" * 80)
        screen_result = await self._run_abstract_screen(peco_result, research_aim, search_result)

        # Step 5: Full-Text Screening
        print("\n" + "=" * 80)
        print("STEP 5: Full-Text Screening - Dual-Reviewer Review of Full Papers")
        print("=" * 80)
        abstract_included_csv = screen_result.replace(".csv", "_included.csv")
        fulltext_included_df = await self._run_fulltext_screen(peco_result, abstract_included_csv)

        # Step 6: Variable Extraction
        print("\n" + "=" * 80)
        print("STEP 6: Variable Extraction - Extracting DAG Nodes from Included Papers")
        print("=" * 80)
        dag_nodes_json = await self._run_variable_extraction(peco_result, fulltext_included_df)
        if dag_nodes_json is None:
            print("\nNo DAG nodes available — skipping DAG construction and adjustment set selection.")
            return

        # Step 7: DAG Construction
        print("\n" + "=" * 80)
        print("STEP 7: DAG Construction - Ensemble Edge Construction Between DAG Nodes")
        print("=" * 80)
        edges_json = await self._run_dag_construction(peco_result, research_aim, dag_nodes_json)
        if edges_json is None:
            print("\nDAG construction failed — skipping adjustment set selection.")
            return

        # Step 8: Adjustment Set
        print("\n" + "=" * 80)
        print("STEP 8: Adjustment Set - Minimal Adjustment Set + Eligibility Variables")
        print("=" * 80)
        adj_result = self._run_adjustment_set(peco_result, dag_nodes_json, edges_json)

        # Step 9: Metadata Matching — check variables exist in the data, repair the set if not
        if self.metadata_db_path and os.path.exists(self.metadata_db_path):
            print("\n" + "=" * 80)
            print("STEP 9: Metadata Matching - Check Adjustment Variables Against the Metadata DB")
            print("=" * 80)
            adj_result = await self._run_metadata_refinement(peco_result, dag_nodes_json, edges_json, adj_result)
        else:
            print(f"\nMetadata DB not found ({self.metadata_db_path}) — skipping metadata matching.")

        self._review_adjustment_set(adj_result)


async def run():
    set_up_log()
    load_dotenv()
    save_folder = "test_results"
    date_range = {"from": "1950/01/01", "to": "2026/03/30"}
    # True: pause for human review after PECO extraction, full-text screening, DAG nodes,
    # DAG edges and adjustment set. False: run the entire system end-to-end.
    human_in_the_loop = False
    framework = StudyDesignFramework(
        save_folder=save_folder,
        date_range=date_range,
        human_in_the_loop=human_in_the_loop,
        hidden_variables=[],       # DAG nodes to treat as unmeasured, e.g. ["birth size"]
        dag_num_runs=5,
        dag_score_threshold=0.5,
        metadata_db_path="metadata_database/cfps_db/cfps_metadata.db",  # None to skip Step 9
    )
    try:
        print("\nDescribe your research aim in 1-5 sentences (free text).")
        task_desc = input("> ").strip()
        result = await framework.run_study_design(task_desc)

    except KeyboardInterrupt:
        print("\n👋 Research session interrupted.")
    except Exception as e:
        print(f"❌ Error during research: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run())