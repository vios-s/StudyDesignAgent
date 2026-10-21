"""
Variable Extraction from Research Papers using Parallel Workers
Uses direct asyncio pattern for true parallel execution
Includes full-text screening with dual reviewers before variable extraction
"""

import os
import asyncio
import multiprocessing as mp
import time
import pandas as pd
import json
import textwrap
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from dotenv import load_dotenv
import pubmed_tools
from pubmed_tools import get_fulltext, extract_fulltext_content, extract_sections_split, strip_references
from typing import List, Dict, Any
from utils import *

import logging
logging.getLogger('camel').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)

from prompts.variable_process import _variable_extraction_task_prompt, _normalize_and_temporal_prompt, _deduplicate_variable_prompt, _build_dag_stage1_prompt, _build_dag_level1_prompt, _build_dag_level2_prompt
from literature_screen import DualReviewerScreener
from prompts.fulltext_screening_process import (
    _fulltext_criteria_generation_prompt,
    _fulltext_reviewer_task_prompt,
    _fulltext_reviewer_system_prompt,
)

load_dotenv()
extraction_model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
    model_type="openai/gpt-5.4",
    url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model_config_dict={"temperature": 0.2, "max_tokens": 100000},
)
reviewer_model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
    model_type="openai/gpt-5.4-nano",
    url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model_config_dict={"temperature": 0.2, "max_tokens": 100000},
)

# Model for deduplication — smaller context is sufficient
dedup_model = ModelFactory.create(
    model_platform=ModelPlatformType.ANTHROPIC,
    model_type="openai/gpt-5.4",
    url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model_config_dict={"temperature": 0.2, "max_tokens": 200000},
)

class FullTextScreener(DualReviewerScreener):
    """
    Full-text screening with dual reviewers.
    Generates more specific criteria than abstract screening since
    we have the complete paper. Applies specificity over sensitivity.
    """

    def _criteria_generation_prompt(self, num_criteria: int = 6) -> str:
        return _fulltext_criteria_generation_prompt(
            self.population, self.exposure, self.comparison, self.outcome, num_criteria
        )

    def _create_reviewer_agent(
        self, role_name: str, persona: str, criteria_text: str, n_criteria: int
    ) -> ChatAgent:
        """Create a full-text reviewer agent using Claude Sonnet for longer content."""
        task = _fulltext_reviewer_task_prompt(
            self.population, self.exposure, self.comparison, self.outcome, criteria_text, n_criteria
        )

        sys_msg = BaseMessage.make_assistant_message(
            role_name=role_name,
            content=_fulltext_reviewer_system_prompt(role_name, persona, task),
        )


        return ChatAgent(system_message=sys_msg, model=reviewer_model, token_limit=128000)


# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — Full-text screening
# ─────────────────────────────────────────────────────────────────────────────

async def screen_single_paper(
    worker_id: int,
    pmid: str,
    doi: str,
    screener: FullTextScreener,
    personas: List[Dict],
    criteria_text: str,
    n_criteria: int,
) -> Dict[str, Any]:
    """
    Fetch full text for one paper, parse into four sections, and run
    dual-reviewer screening on the structured content.

    Returns a dict with:
      - id, fulltext_decision, per-reviewer details
      - methods, results, discussion, tables  ← ready for variable extraction
    'fulltext_decision' is 'NO_FULLTEXT' when the paper cannot be retrieved.
    """
    base = {
        "id": pmid,
        "fulltext_decision": "NO_FULLTEXT",
        "fulltext_reviewer_1_decision": "",
        "fulltext_reviewer_1_rationale": "",
        "fulltext_reviewer_2_decision": "",
        "fulltext_reviewer_2_rationale": "",
        "fulltext_agreement": "",
        "fulltext": "",
    }

    # Fetch full text
    content = await asyncio.to_thread(get_fulltext, pmid, doi or None, 'test_case/low_dose_resource')
    if not content:
        print(f"  [Worker {worker_id}] Paper {pmid}: full text unavailable")
        return base

    # For BioC XML: strip markup and extract passage/table text only.
    # For Markdown: returned as-is.
    screen_content = extract_fulltext_content(content)

    # Dual-reviewer screening — fresh agents per paper for async safety
    screening_agents = [
        screener._create_reviewer_agent(
            p["role_name"], p["persona"], criteria_text, n_criteria
        )
        for p in personas
    ]
    screen_result = await screener._screen_paper_dual(
        agents=screening_agents,
        paper_id=pmid,
        paper_content=screen_content,
        n_criteria=n_criteria,
    )

    r1    = screen_result["reviewer_1"]
    r2    = screen_result["reviewer_2"]
    agree = screen_result["agreement"]

    # Decision rule: at least one reviewer votes INCLUDE → INCLUDE
    if r1["decision"] == "INCLUDE" or r2["decision"] == "INCLUDE":
        final = "INCLUDE"
    else:
        final = "EXCLUDE"

    print(
        f"  [Worker {worker_id}] Paper {pmid}: "
        f"R1={r1['decision']} R2={r2['decision']} "
        f"[{'AGREE' if agree else 'DISAGREE'}] → {final}"
    )

    return {
        "id": pmid,
        "fulltext_decision": final,
        "fulltext_reviewer_1_decision": r1["decision"],
        "fulltext_reviewer_1_rationale": r1["rationale"],
        "fulltext_reviewer_2_decision": r2["decision"],
        "fulltext_reviewer_2_rationale": r2["rationale"],
        "fulltext_agreement": agree,
        "fulltext": content,
    }


def _base_screen_result(pmid: str, decision: str = "NO_FULLTEXT", error: str = "") -> Dict[str, Any]:
    return {
        "id": pmid,
        "fulltext_decision": decision,
        "fulltext_reviewer_1_decision": "",
        "fulltext_reviewer_1_rationale": "",
        "fulltext_reviewer_2_decision": "",
        "fulltext_reviewer_2_rationale": "",
        "fulltext_agreement": "",
        "fulltext": "",
        "error": error,
    }


def _screen_paper_process_target(payload: Dict[str, Any], result_queue) -> None:
    """
    Entry point for a per-paper subprocess (spawned fresh, not a pooled worker).

    Rebuilds the screener from plain picklable PECO strings rather than
    receiving the parent's screener/model objects, since those hold live
    HTTP client state that isn't safe to pickle across a process boundary.
    Running each paper in its own process (instead of asyncio.to_thread)
    means a stuck network/LLM call can be force-killed by the parent via
    process.terminate() — a hung *thread* can't be cancelled once a blocking
    socket read is in flight, which is why the old asyncio version could
    stall forever on a single bad paper.
    """
    load_dotenv()
    pmid = payload["pmid"]
    shared_limiter = payload.get("firecrawl_limiter")
    if shared_limiter is not None:
        # Replace the per-process Firecrawl rate limiter with one backed by
        # Manager proxies shared across every worker process, so the plan's
        # req/min cap is enforced globally rather than once per process.
        calls, lock, max_calls = shared_limiter
        pubmed_tools._firecrawl_rate_limiter = pubmed_tools._RateLimiter(
            max_calls=max_calls, period=60.0, calls=calls, lock=lock,
        )
    try:
        screener = FullTextScreener(
            exposure=payload["exposure"],
            outcome=payload["outcome"],
            population=payload["population"],
            comparison=payload["comparison"],
            include_threshold=payload["include_threshold"],
        )
        result = asyncio.run(
            screen_single_paper(
                worker_id=payload["worker_id"],
                pmid=pmid,
                doi=payload["doi"],
                screener=screener,
                personas=payload["personas"],
                criteria_text=payload["criteria_text"],
                n_criteria=payload["n_criteria"],
            )
        )
        result_queue.put(result)
    except Exception as e:
        result_queue.put(_base_screen_result(pmid, decision="ERROR", error=str(e)))


def _run_screening_pool(
    papers: List[tuple],
    screener_config: Dict[str, Any],
    personas: List[Dict],
    criteria_text: str,
    n_criteria: int,
    num_workers: int,
    process_timeout: int,
) -> List[Dict[str, Any]]:
    """
    Synchronous supervisor: keeps up to `num_workers` papers screening
    concurrently, each in its own OS process, and force-kills any paper
    whose process runs past `process_timeout` seconds so one stuck paper
    can no longer stall the whole batch.
    """
    ctx = mp.get_context("spawn")
    results: List[Any] = [None] * len(papers)
    task_iter = iter(enumerate(papers))
    active: Dict[Any, Dict[str, Any]] = {}

    # Firecrawl's per-minute cap is account-wide, not per-process — a Manager-backed
    # list+lock lets every spawned worker draw from the same shared budget, matching
    # what the old single-process asyncio version did with a plain in-process deque.
    manager = ctx.Manager()
    firecrawl_limiter = (
        manager.list(),
        manager.Lock(),
        int(os.getenv("FIRECRAWL_MAX_REQ_PER_MIN", "10")),
    )

    def launch_next() -> bool:
        try:
            idx, (pmid, doi) = next(task_iter)
        except StopIteration:
            return False
        q = ctx.Queue()
        payload = {
            "worker_id": idx + 1,
            "pmid": pmid,
            "doi": doi,
            "personas": personas,
            "criteria_text": criteria_text,
            "n_criteria": n_criteria,
            "firecrawl_limiter": firecrawl_limiter,
            **screener_config,
        }
        proc = ctx.Process(target=_screen_paper_process_target, args=(payload, q), daemon=True)
        proc.start()
        active[proc] = {"queue": q, "start": time.monotonic(), "pmid": pmid, "idx": idx}
        return True

    for _ in range(num_workers):
        if not launch_next():
            break

    last_heartbeat = time.monotonic()
    abandoned: set = set()  # processes we gave up waiting on — never join() these again

    while active:
        for proc in list(active.keys()):
            info = active[proc]

            # Drain the queue every pass, regardless of alive-state. A paper's full
            # text can be tens of KB — bigger than the OS pipe buffer — so the child
            # blocks inside queue.put() until we read it out. If we only ever checked
            # the queue after is_alive() went False, we'd deadlock forever: the child
            # can't exit until put() returns, and put() can't return until we read.
            if "result" not in info:
                try:
                    info["result"] = info["queue"].get_nowait()
                except Exception:
                    pass

            if "result" in info:
                proc.join(5)
                if proc.is_alive():
                    proc.terminate()
                    proc.join(5)
                results[info["idx"]] = info["result"]
                try:
                    info["queue"].close()
                except Exception:
                    pass
                del active[proc]
                launch_next()
                continue

            elapsed = time.monotonic() - info["start"]
            timed_out = False
            if proc.is_alive() and elapsed > process_timeout and proc not in abandoned:
                print(f"  [Worker] Paper {info['pmid']}: TIMEOUT after {process_timeout}s — killing process")
                timed_out = True
                proc.terminate()
                proc.join(5)
                if proc.is_alive():
                    proc.kill()
                    proc.join(15)
                if proc.is_alive():
                    # A process stuck in an uninterruptible OS wait (e.g. hung NFS I/O
                    # on a network-mounted home dir) can survive even SIGKILL. Don't
                    # block the scheduler on it forever — abandon it, mark this paper
                    # as timed out, and let the OS reap the orphan whenever it can.
                    print(f"  [Worker] Paper {info['pmid']}: process would not die (stuck OS-level I/O?) — abandoning it")
                    abandoned.add(proc)

            if not proc.is_alive() or proc in abandoned:
                # Process exited/abandoned without us ever seeing a queued result
                # (crashed, killed, or genuinely produced nothing) — one last
                # non-blocking check, then fall back to a synthetic result.
                try:
                    result = info["queue"].get_nowait()
                except Exception:
                    result = _base_screen_result(
                        info["pmid"], decision="TIMEOUT" if (timed_out or proc in abandoned) else "ERROR"
                    )
                results[info["idx"]] = result
                try:
                    info["queue"].close()
                except Exception:
                    pass
                abandoned.discard(proc)
                del active[proc]
                launch_next()

        if active and time.monotonic() - last_heartbeat > 20:
            last_heartbeat = time.monotonic()
            status = ", ".join(
                f"{info['pmid']}({time.monotonic() - info['start']:.0f}s)" for info in active.values()
            )
            print(f"  [Scheduler] {len(active)} active: {status}")

        if active:
            time.sleep(0.5)

    manager.shutdown()
    return results


async def screen_papers_parallel(
    paper_list: List[str],
    paper_doi_list: List[str],
    screener: FullTextScreener,
    personas: List[Dict],
    criteria_text: str,
    n_criteria: int,
    num_workers: int = 5,
    process_timeout: int = 240,
) -> List[Dict[str, Any]]:
    """
    Screen papers in parallel, one OS process per paper (bounded to
    num_workers concurrent processes). A paper whose process exceeds
    process_timeout seconds is killed and recorded as TIMEOUT rather than
    hanging the whole run.

    Returns one dict per paper with the screening decision and full text.
    """
    print(f"\n{'=' * 60}")
    print(f"FULL-TEXT SCREENING: {len(paper_list)} papers | concurrency={num_workers} | timeout={process_timeout}s")
    print(f"{'=' * 60}")

    papers = list(zip(
        [str(p) for p in paper_list],
        [str(d) if pd.notna(d) else None for d in paper_doi_list],
    ))
    screener_config = {
        "exposure": screener.exposure,
        "outcome": screener.outcome,
        "population": screener.population,
        "comparison": screener.comparison,
        "include_threshold": screener.include_threshold,
    }

    all_results = await asyncio.to_thread(
        _run_screening_pool,
        papers, screener_config, personas, criteria_text, n_criteria, num_workers, process_timeout,
    )

    n_include     = sum(1 for r in all_results if r["fulltext_decision"] == "INCLUDE")
    n_exclude     = sum(1 for r in all_results if r["fulltext_decision"] == "EXCLUDE")
    n_no_fulltext = sum(1 for r in all_results if r["fulltext_decision"] == "NO_FULLTEXT")
    screened      = n_include + n_exclude
    n_agree       = sum(1 for r in all_results if r.get("fulltext_agreement") is True)

    print(f"\n{'=' * 60}")
    print(f"SCREENING COMPLETE")
    print(f"  Included     : {n_include}")
    print(f"  Excluded     : {n_exclude}")
    print(f"  No full text : {n_no_fulltext}")
    if screened:
        print(f"  Agreement    : {n_agree}/{screened} ({100 * n_agree // screened}%)")
    print(f"{'=' * 60}")

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# PART 2 — Variable extraction  (uses pre-parsed sections from screening CSV)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_json_response(raw: str) -> list:
    raw = raw.strip()
    if raw.startswith('```json'):
        raw = raw.split('```json')[1].split('```')[0].strip()
    elif raw.startswith('```'):
        raw = raw.split('```')[1].split('```')[0].strip()
    if '[' not in raw:
        return []
    return json_array_slice(raw)


def _flatten_passages(items) -> list:
    """Normalize a source_passages list: some earlier extraction runs stored
    a raw list-of-spans instead of a single string for one entry, so a few
    items in the list can themselves be lists. Flatten those in place."""
    out = []
    for item in items or []:
        if isinstance(item, list):
            out.extend(str(s) for s in item if s)
        elif item:
            out.append(str(item))
    return out


async def extract_single_paper(
    worker_id: int,
    exposure: str,
    outcome: str,
    pmid: str,
    fulltext: str,
    population: str = "",
    comparator: str = "",
) -> List[Dict[str, Any]]:
    """Single-pass variable extraction covering all paper sections."""
    fulltext = strip_references(fulltext)
    peco_parts = [f"--- EXPOSURE ---\n{exposure}", f"--- OUTCOME ---\n{outcome}"]
    if population:
        peco_parts.insert(0, f"--- POPULATION ---\n{population}")
    if comparator:
        peco_parts.append(f"--- COMPARATOR ---\n{comparator}")
    peco_header = "\n\n".join(peco_parts) + "\n\n"

    agent = ChatAgent(system_message=_variable_extraction_task_prompt(), model=extraction_model)
    prompt = (
        f"Extract variables from paper (PMID: {pmid}).\n\n"
        f"{peco_header}"
        f"--- PAPER CONTENT ---\n{fulltext}\n--- END ---"
    )
    try:
        r = await asyncio.to_thread(agent.step, prompt)
        variables = _parse_json_response(r.msgs[0].content)
    except Exception as e:
        print(f"  [Worker {worker_id}] Paper {pmid} ERROR - {str(e)[:50]}")
        return []

    # Stage 2: normalize names + add temporal prefixes (batch per paper)
    if variables:
        norm_agent = ChatAgent(system_message=_normalize_and_temporal_prompt(), model=extraction_model)
        norm_prompt = (
            f"{peco_header}"
            f"--- RAW VARIABLES (JSON) ---\n{json.dumps(variables, ensure_ascii=False)}\n--- END ---"
        )
        try:
            nr = await asyncio.to_thread(norm_agent.step, norm_prompt)
            normalized = _parse_json_response(nr.msgs[0].content)
            # conservation guard: only accept if count matches; else keep raw
            if len(normalized) == len(variables):
                variables = normalized
            else:
                print(f"  [Worker {worker_id}] Paper {pmid}: normalize count mismatch "
                      f"({len(normalized)} vs {len(variables)}) — keeping raw")
        except Exception as e:
            print(f"  [Worker {worker_id}] Paper {pmid} normalize ERROR - {str(e)[:50]} — keeping raw")

    for var in variables:
        var['paper_id'] = pmid
    print(f"  [Worker {worker_id}] Paper {pmid}: {len(variables)} variables extracted")
    return variables


async def extract_variables_parallel(
    included_df: pd.DataFrame,
    exposure: str,
    outcome: str,
    num_workers: int = 5,
    population: str = "",
    comparator: str = "",
) -> List[Dict[str, Any]]:
    """
    Extract variables from included papers using parallel workers.

    Reads pre-parsed sections (methods, results, discussion, tables) from
    included_df — the full-text screening output filtered to INCLUDE rows.
    """
    print(f"\n{'=' * 60}")
    print(f"VARIABLE EXTRACTION: {len(included_df)} papers | {num_workers} workers")
    print(f"{'=' * 60}")

    rows = included_df.to_dict("records")
    rows_per_worker = max(1, len(rows) // num_workers)

    async def worker_batch(worker_id: int, start_idx: int, end_idx: int):
        batch_results = []
        for row in rows[start_idx:end_idx]:
            variables = await extract_single_paper(
                worker_id=worker_id,
                exposure=exposure,
                outcome=outcome,
                pmid=str(row["id"]),
                fulltext=str(row.get("fulltext") or ""),
                population=population,
                comparator=comparator,
            )
            batch_results.extend(variables)
        return batch_results

    worker_tasks = []
    for w in range(num_workers):
        start_idx = w * rows_per_worker
        end_idx = start_idx + rows_per_worker if w < num_workers - 1 else len(rows)
        if start_idx >= len(rows):
            break
        print(f"  Worker {w+1}: papers {start_idx+1}-{end_idx}")
        worker_tasks.append(worker_batch(w + 1, start_idx, end_idx))

    print(f"\nStarting {len(worker_tasks)} parallel workers...")
    worker_results = await asyncio.gather(*worker_tasks)
    all_results = [v for batch in worker_results for v in batch]

    print(f"\n{'=' * 60}")
    print(f"EXTRACTION COMPLETE: {len(all_results)} total variables")
    print(f"{'=' * 60}")

    return all_results


async def deduplicate_variables(
    variables: List[Dict[str, Any]],
    population: str = "",
    exposure: str = "",
    comparator: str = "",
    outcome: str = "",
) -> List[Dict[str, Any]]:
    """
    Deduplicate variables using an LLM agent for semantic grouping,
    then reattach source passages from the original variables by paper_id.
    """
    if not variables:
        return []

    print(f"\n{'=' * 60}")
    print(f"Deduplicating {len(variables)} variables using LLM agent...")
    print(f"{'=' * 60}")

    # Prepare input for LLM — include entry_id so we can precisely recover role/source per cluster
    agent_input = [
        {
            'entry_id': idx,
            'variable': var.get('variable', ''),
            'role': var.get('role', ''),
            'knowledge_card': var.get('knowledge_card', ''),
            'paper_id': str(var.get('paper_id', '')).strip(),
        }
        for idx, var in enumerate(variables)
    ]

    peco_parts = [f"--- EXPOSURE ---\n{exposure}", f"--- OUTCOME ---\n{outcome}"]
    if population:
        peco_parts.insert(0, f"--- POPULATION ---\n{population}")
    if comparator:
        peco_parts.append(f"--- COMPARATOR ---\n{comparator}")
    peco_header = "\n\n".join(peco_parts) + "\n\n"

    sys_msg = BaseMessage.make_assistant_message(
        role_name="Deduplication Agent",
        content=_deduplicate_variable_prompt(),
    )
    dedup_agent = ChatAgent(system_message=sys_msg, model=dedup_model)

    user_msg = BaseMessage.make_user_message(
        role_name="User",
        content=(
            f"Here is the PECO element {peco_header}"
            f"Deduplicate the following variables:\n\n"
            f"{json.dumps(agent_input, ensure_ascii=False)}"
        ),
    )

    response = dedup_agent.step(user_msg)
    raw = response.msgs[0].content.strip()

    # Parse LLM output
    if '[' not in raw:
        print("Failed to parse deduplication response: No JSON array found")
        print("Raw response:", raw[:500])
        return []
    clusters = json_array_slice(raw)
    if not clusters:
        print("Failed to parse deduplication response: empty or malformed array")
        print("Raw response:", raw[:500])
        return []

    # Reassemble clusters: use entry_ids to precisely recover paper_id, role, source
    deduped = []
    for cluster in clusters:
        variable = (cluster.get('variable') or cluster.get('name') or
                    cluster.get('term') or cluster.get('canonical_name') or
                    cluster.get('canonical_variable') or '')
        entry_ids = cluster.get('entry_ids', [])

        paper_ids = []
        source_passages = []
        roles_set: set = set()
        for eid in entry_ids:
            if not isinstance(eid, int) or eid < 0 or eid >= len(variables):
                continue
            orig = variables[eid]
            pid = str(orig.get('paper_id', '')).strip()
            if pid and pid not in paper_ids:
                paper_ids.append(pid)
            src = orig.get('source', '')
            if isinstance(src, list):
                source_passages.extend(str(s) for s in src if s)
            elif src:
                source_passages.append(str(src))
            role = orig.get('role', '')
            if role:
                roles_set.add(role)

        # Anchors keep exactly their canonical role — never polluted by merged entries
        if "treatment" in roles_set:
            assigned_roles = ["treatment"]
        elif "outcome" in roles_set:
            assigned_roles = ["outcome"]
        else:
            assigned_roles = sorted(roles_set)

        deduped.append({
            'variable': variable,
            'roles': assigned_roles,
            'frequency': len(paper_ids),
            'knowledge_card': cluster.get('knowledge_card', ''),
            'paper_ids': paper_ids,
            'source_passages': source_passages,
        })

    deduped.sort(key=lambda x: x['frequency'], reverse=True)
    print(f"Deduplicated to {len(deduped)} unique variables")
    return deduped


def enforce_conservation(output_nodes, input_vars):
    placed = set()
    for n in output_nodes:
        placed.add(n['variable'])
        placed.update(n.get('children', []))
    dropped = [v for v in input_vars if v['variable'] not in placed]
    for v in dropped:
        output_nodes.append({
            'variable': v['variable'],
            'roles': v.get('roles', []),
            'knowledge_card': v.get('knowledge_card', ''),
            'source_passages': v.get('source_passages', []),
            'children': []
        })
    return output_nodes, dropped


async def build_dag_nodes(
    dedup_results: List[Dict[str, Any]],
    population: str,
    exposure: str,
    comparator: str,
    outcome: str,
) -> List[Dict[str, Any]]:
    """
    Post-deduplication step: assign parent-child structure and identify
    distinct DAG nodes by merging subtypes/instances under their parent.
    """
    if not dedup_results:
        return []

    print(f"\n{'=' * 60}")
    print(f"Building DAG node structure from {len(dedup_results)} deduplicated variables...")
    print(f"{'=' * 60}")

    agent_input = [
        {
            "variable": v.get("variable", ""),
            "roles": v.get("roles", []),
            "knowledge_card": v.get("knowledge_card", ""),
            "source_passages": v.get("source_passages", [])[:5],
        }
        for v in dedup_results
    ]

    peco_parts = [f"--- EXPOSURE ---\n{exposure}", f"--- OUTCOME ---\n{outcome}"]
    if population:
        peco_parts.insert(0, f"--- POPULATION ---\n{population}")
    if comparator:
        peco_parts.append(f"--- COMPARATOR ---\n{comparator}")
    peco_header = "\n\n".join(peco_parts) + "\n\n"

    json_only = "Return ONLY a valid JSON array. No reasoning, markdown, headings, or preamble.\n\n"

    # Pass 1: Stage 1 filtering
    # The model only returns {variable, roles} for survivors — knowledge_card and
    # source_passages are reattached from agent_input in Python rather than having
    # the model retype them, which would roughly double the output size for no reason.
    print("  [DAG pass 1/3] Filtering variables...")
    agent1 = ChatAgent(system_message=_build_dag_stage1_prompt(), model=extraction_model, token_limit=128000, step_timeout=600)
    r1 = await asyncio.to_thread(
        agent1.step,
        json_only + peco_header + f"Filter the following variables. Output JSON only:\n\n{json.dumps(agent_input, indent=2)}",
    )
    filtered_vars_raw = _parse_json_response(r1.msgs[0].content)
    if not filtered_vars_raw:
        return []
    agent_input_by_name = {v["variable"]: v for v in agent_input}
    filtered_vars = []
    for fv in filtered_vars_raw:
        name = fv.get("variable", "")
        orig = agent_input_by_name.get(name)
        filtered_vars.append({
            "variable": name,
            "roles": fv.get("roles", (orig or {}).get("roles", [])),
            "knowledge_card": (orig or {}).get("knowledge_card", ""),
            "source_passages": (orig or {}).get("source_passages", []),
        })
    print(f"  [DAG pass 1/3] {len(agent_input)} → {len(filtered_vars)} variables after filtering")

    # Pass 2: Level 1 vertical merges
    # Same reattachment pattern: the model only returns {variable, roles, children}.
    print("  [DAG pass 2/3] Applying vertical merges...")
    agent2 = ChatAgent(system_message=_build_dag_level1_prompt(), model=extraction_model, token_limit=128000, step_timeout=600)
    r2 = await asyncio.to_thread(
        agent2.step,
        json_only + peco_header + f"Apply vertical parent-child merging to the following variables. Output JSON only:\n\n{json.dumps(filtered_vars, indent=2)}",
    )
    level1_roots_raw = _parse_json_response(r2.msgs[0].content)
    if not level1_roots_raw:
        return []
    filtered_by_name = {v['variable']: v for v in filtered_vars}
    level1_roots = []
    for root in level1_roots_raw:
        name = root.get('variable', '')
        orig = filtered_by_name.get(name)
        level1_roots.append({
            'variable': name,
            'roles': root.get('roles', (orig or {}).get('roles', [])),
            'knowledge_card': (orig or {}).get('knowledge_card', ''),
            'source_passages': (orig or {}).get('source_passages', []),
            'children': root.get('children', []),
        })
    level1_roots, l1_dropped = enforce_conservation(level1_roots, filtered_vars)
    if l1_dropped:
        print(f"  [Level 1 conservation] Recovered {len(l1_dropped)} dropped variable(s): "
              + ", ".join(v['variable'] for v in l1_dropped))
    print(f"  [DAG pass 2/3] {len(level1_roots)} Level 1 roots")

    # Pass 3: Level 2 horizontal aggregation → final DAG nodes
    # The model omits knowledge_card for roots carried over unchanged; only genuinely
    # new synthesized parents get a model-authored knowledge_card (reattach can't help there).
    print("  [DAG pass 3/3] Applying horizontal aggregation...")
    agent3 = ChatAgent(system_message=_build_dag_level2_prompt(), model=extraction_model, token_limit=128000, step_timeout=600)
    r3 = await asyncio.to_thread(
        agent3.step,
        json_only + peco_header + f"Apply horizontal aggregation to the following Level 1 roots. Output JSON only:\n\n{json.dumps(level1_roots, indent=2)}",
    )
    dag_nodes_raw = _parse_json_response(r3.msgs[0].content)
    if not dag_nodes_raw:
        return []
    level1_roots_by_name = {r['variable']: r for r in level1_roots}
    dag_nodes = []
    for node in dag_nodes_raw:
        name = node.get('variable', '')
        orig = level1_roots_by_name.get(name)
        dag_nodes.append({
            'variable': name,
            'roles': node.get('roles', (orig or {}).get('roles', [])),
            'knowledge_card': orig.get('knowledge_card', '') if orig else (node.get('knowledge_card', '') or ''),
            'source_passages': (orig or {}).get('source_passages', []),
            'children': node.get('children', []),
        })

    # Build flat list of all Level 1 variables (roots + children) for conservation check
    filtered_by_name = {v['variable']: v for v in filtered_vars}
    seen_l2: set[str] = set()
    level1_all_vars: list[dict] = []
    for root in level1_roots:
        if root['variable'] not in seen_l2:
            level1_all_vars.append(root)
            seen_l2.add(root['variable'])
        for child_name in root.get('children', []):
            if child_name not in seen_l2:
                level1_all_vars.append(filtered_by_name.get(child_name, {
                    'variable': child_name, 'roles': [], 'knowledge_card': '', 'source_passages': []
                }))
                seen_l2.add(child_name)
    dag_nodes, l2_dropped = enforce_conservation(dag_nodes, level1_all_vars)
    if l2_dropped:
        print(f"  [Level 2 conservation] Recovered {len(l2_dropped)} dropped variable(s): "
              + ", ".join(v['variable'] for v in l2_dropped))

    # Collect all variable names that are listed as children of another node
    child_names: set[str] = set()
    for node in dag_nodes:
        for child in node.get("children", []):
            child_names.add(child)

    # Remove top-level entries that are children of another node
    dag_nodes = [n for n in dag_nodes if n.get("variable") not in child_names]

    # Merge paper_ids, source_passages, frequency back from dedup results.
    # For synthetic parent nodes (not in dedup_by_name), aggregate from children.
    dedup_by_name = {v["variable"]: v for v in dedup_results}
    for node in dag_nodes:
        name = node.get("variable", "")
        orig = dedup_by_name.get(name)
        if orig:
            node["paper_ids"] = orig.get("paper_ids", [])
            node["source_passages"] = _flatten_passages(orig.get("source_passages", []))
            node["frequency"] = orig.get("frequency", 0)
        else:
            # Synthetic parent — aggregate from children
            agg_paper_ids: list[str] = []
            agg_passages: list[str] = []
            for child in node.get("children", []):
                child_orig = dedup_by_name.get(child, {})
                agg_paper_ids.extend(child_orig.get("paper_ids", []))
                agg_passages.extend(_flatten_passages(child_orig.get("source_passages", [])))
            node["paper_ids"] = list(dict.fromkeys(agg_paper_ids))
            node["source_passages"] = agg_passages
            node["frequency"] = len(node["paper_ids"])

    print(f"DAG structure: {len(dag_nodes)} distinct nodes")
    return dag_nodes


async def generate_knowledge_card(
    exposure: str,
    outcome: str,
    raw_json_path: str,
    save_folder: str = None,
) -> List[Dict[str, Any]]:
    """
    Enrich raw extracted variables with knowledge cards via terminology lookup.

    Reads raw variables from raw_json_path, saves an intermediate knowledge_card
    JSON, then merges the cards back into each variable entry.

    Returns the enriched variable list.
    """
    print(f"\n{'=' * 60}")
    print(f"Enriching variables with knowledge cards...")
    from terminology_enriched import EnrichedTerminologyLookup

    def build_path(fn):
        return os.path.join(save_folder, fn) if save_folder else fn

    with open(raw_json_path, 'r', encoding='utf-8') as f:
        raw_variables = json.load(f)

    lookup = EnrichedTerminologyLookup(config_path="config/web_browser.json")
    knowledge_card_json = build_path(f"knowledge_card_{exposure}_{outcome}.json")
    enriched_json_path = await lookup.process_raw_variable_enriched(
        input_json=raw_json_path,
        output_json=knowledge_card_json,
        num_workers=5,
    )
    print(f"Enriched knowledge cards saved to: {enriched_json_path}")

    print(f"\n{'=' * 60}")
    print(f"Merging knowledge cards...")

    with open(enriched_json_path, 'r', encoding='utf-8') as f:
        enriched_data = json.load(f)

    knowledge_lookup = {
        entry['term'].strip().lower(): entry['knowledge card']
        for entry in enriched_data
    }

    enriched_results = []
    for entry in raw_variables:
        variable_name = entry.get('variable', '').strip().lower()
        enriched_entry = entry.copy()
        enriched_entry['knowledge_card'] = knowledge_lookup.get(variable_name, "")
        enriched_results.append(enriched_entry)

    print(f"Attached knowledge cards to {len(enriched_results)} variables")
    return enriched_results

# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────

async def run_fulltext_screening(
    input_df: pd.DataFrame,
    exposure: str,
    outcome: str,
    save_folder: str = None,
    population: str = None,
    comparison: str = None,
) -> pd.DataFrame:
    import re
    # Sanitize for use in filenames — replace spaces, parentheses, slashes, etc.
    exposure = re.sub(r'[^\w\-]', '_', exposure).strip('_')
    outcome = re.sub(r'[^\w\-]', '_', outcome).strip('_')
    """
    Part 1: Full-text screening.

    Reads PMIDs and DOIs from input_df ('id' and 'doi' columns required).
    Saves three files and returns the included-only DataFrame.

    Output files:
      fulltext_screen_{exposure}_{outcome}.csv         — all papers
      fulltext_screen_{exposure}_{outcome}_details.json — audit trail
      fulltext_screen_{exposure}_{outcome}_included.csv — INCLUDE rows
            (with methods/results/discussion/tables ready for extraction)
    """
    if save_folder:
        os.makedirs(save_folder, exist_ok=True)

    def build_path(save_folder, fn):
        return os.path.join(save_folder, fn) if save_folder else fn

    # Set up screener
    screener = FullTextScreener(
        exposure=exposure.replace("_", " "),
        outcome=outcome.replace("_", " "),
        population=population or "any population",
        comparison=comparison or "any comparison",
        include_threshold=1.0,
    )

    print(f"\n{'=' * 60}")
    print(f"FULL-TEXT SCREENING SETUP")
    print(f"{'=' * 60}")
    screener.generate_criteria()
    n_criteria   = len(screener.content_criteria)
    criteria_text = "\n".join(f"{i+1}. {c}" for i, c in enumerate(screener.content_criteria))
    print(f"Generated {n_criteria} criteria:")
    for i, c in enumerate(screener.content_criteria):
        print(f"  {i+1}. {c}")

    research_question = (
        f"Does {exposure.replace('_', ' ')} affect "
        f"{outcome.replace('_', ' ')} in {population or 'any population'}?"
    )
    print(f"\nDeriving reviewer personas...")
    personas = screener._derive_personas(research_question)
    print(f"Reviewer 1: {personas[0]['role_name']}")
    print(f"Reviewer 2: {personas[1]['role_name']}")
    print(f"Decision rule: INCLUDE if at least one reviewer votes INCLUDE")

    # Screen papers
    screen_results = await screen_papers_parallel(
        paper_list=input_df["id"].astype(str).tolist(),
        paper_doi_list=input_df["doi"].tolist(),
        screener=screener,
        personas=personas,
        criteria_text=criteria_text,
        n_criteria=n_criteria,
        num_workers=5,
    )

    # Merge screening columns back onto input df
    screen_cols = [
        "fulltext_decision",
        "fulltext_reviewer_1_decision", "fulltext_reviewer_1_rationale",
        "fulltext_reviewer_2_decision", "fulltext_reviewer_2_rationale",
        "fulltext_agreement",
        "fulltext",
    ]
    screen_map = {r["id"]: r for r in screen_results}
    result_df  = input_df.copy()
    result_df["id"] = result_df["id"].astype(str)
    for col in screen_cols:
        result_df[col] = result_df["id"].map(
            lambda pid, c=col: screen_map.get(pid, {}).get(c, "")
        )

    included_df = result_df[result_df["fulltext_decision"] == "INCLUDE"].copy()

    # Save CSV with only id + review columns (no full text)
    save_cols = [c for c in result_df.columns if c != "fulltext"]
    all_csv = build_path(save_folder, f"fulltext_screen_{exposure}_{outcome}.csv")
    result_df[save_cols].to_csv(all_csv, index=False, encoding='utf-8-sig')
    print(f"\nSaved all papers ({len(result_df)}) → {all_csv}")

    included_csv = build_path(save_folder, f"fulltext_screen_{exposure}_{outcome}_included.csv")
    included_df[save_cols].to_csv(included_csv, index=False, encoding='utf-8-sig')
    print(f"Saved included papers ({len(included_df)}) → {included_csv}")

    # Save audit trail JSON
    details_json = build_path(save_folder, f"fulltext_screen_{exposure}_{outcome}_details.json")
    with open(details_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "peco": {
                    "population": screener.population,
                    "exposure": screener.exposure,
                    "comparison": screener.comparison,
                    "outcome": screener.outcome,
                },
                "criteria": screener.content_criteria,
                "reviewers": [
                    {"role_name": p["role_name"], "persona": p["persona"]}
                    for p in personas
                ],
                "decision_rule": "INCLUDE if at least one reviewer votes INCLUDE",
                "screening_results": screen_results,
            },
            f, indent=2, ensure_ascii=False,
        )
    print(f"Saved details → {details_json}")

    return included_df


async def run_variable_extraction(
    included_df: pd.DataFrame,
    exposure: str,
    outcome: str,
    save_folder: str = None,
    population: str = "",
    comparator: str = "",
) -> pd.DataFrame:
    """
    Part 2: Variable extraction from pre-screened, pre-parsed papers.

    Reads sections (methods/results/discussion/tables) from included_df
    — the INCLUDE output of run_fulltext_screening — so no re-fetching needed.

    Saves:
      extracted_variables_{exposure}_{outcome}_raw.csv
      extracted_variables_{exposure}_{outcome}_deduplicated.csv
    """
    import re
    exposure = re.sub(r'[^\w\-]', '_', exposure).strip('_')
    outcome = re.sub(r'[^\w\-]', '_', outcome).strip('_')
    if save_folder:
        os.makedirs(save_folder, exist_ok=True)

    def build_path(fn):
        return os.path.join(save_folder, fn) if save_folder else fn

    # # Extract variables
    raw_variables = await extract_variables_parallel(
        included_df, exposure, outcome, num_workers=5,
        population=population, comparator=comparator,
    )

    if not raw_variables:
        print("No variables extracted.")
        return pd.DataFrame()

    # Save raw
    raw_json_path = build_path(f"extracted_variables_{exposure}_{outcome}_raw.json")
    with open(raw_json_path, 'w', encoding='utf-8') as f:
        json.dump(raw_variables, f, ensure_ascii=False, indent=2)
    print(f"\nRaw: {len(raw_variables)} variable entries → {raw_json_path}")

  
    # Step 3: Enrich variables with knowledge cards
    enriched_results = await generate_knowledge_card(
        exposure, outcome, raw_json_path, save_folder
    )

    # # Deduplicate
    dedup_results = await deduplicate_variables(
        enriched_results,
        population=population,
        exposure=exposure,
        comparator=comparator,
        outcome=outcome,
    )

    dedup_json = build_path(f"extracted_variables_{exposure}_{outcome}_deduplicated.json")
    with open(dedup_json, 'w', encoding='utf-8') as f:
        json.dump(dedup_results, f, ensure_ascii=False, indent=2)
    print(f"\nDeduplicated: {len(dedup_results)} unique entries → {dedup_json}")

    # Build DAG node structure
    dag_results = await build_dag_nodes(
        dedup_results,
        population, exposure,
        comparator, outcome
    )

    dag_json = build_path(f"extracted_variables_{exposure}_{outcome}_dag.json")
    with open(dag_json, 'w', encoding='utf-8') as f:
        json.dump(dag_results, f, ensure_ascii=False, indent=2)
    print(f"DAG nodes: {len(dag_results)} distinct nodes → {dag_json}")

    rows = []
    for entry in dag_results:
        pids     = entry.get("paper_ids", [])
        passages = _flatten_passages(entry.get("source_passages", []))
        roles    = entry.get("roles", [])
        rows.append({
            "variable":        entry.get("variable", ""),
            "roles":           " | ".join(roles),
            "frequency":       entry.get("frequency", len(pids)),
            "knowledge_card":  entry.get("knowledge_card", ""),
            "children":        " | ".join(entry.get("children", [])),
            "paper_ids":       " | ".join(str(p) for p in pids),
            "source_passages": " | ".join(passages),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("No DAG nodes generated; returning empty dataframe.")
        return pd.DataFrame(columns=[
            "variable",
            "roles",
            "frequency",
            "knowledge_card",
            "children",
            "paper_ids",
            "source_passages",
        ])
    df = df.sort_values("frequency", ascending=False).reset_index(drop=True)

    dag_csv = build_path(f"extracted_variables_{exposure}_{outcome}_dag.csv")
    df.to_csv(dag_csv, index=False, encoding="utf-8")
    print(f"DAG nodes CSV → {dag_csv}")

    print(f"\nTop DAG nodes by frequency:")
    print(df[["variable", "frequency", "children"]].head(20).to_string(index=False))

    return df


if __name__ == "__main__":
    save_folder  = ""

    peco_elements_json = """
    """


    peco_elements = parse_peco_json(peco_elements_json)
    peco_values = extract_peco_values(peco_elements)

    async def pipeline():
        # Part 1 — full-text screening
        details_path = ""
        with open(details_path, "r", encoding="utf-8") as f:
            details = json.load(f)

        included_df = pd.DataFrame(
            [r for r in details["screening_results"] if r.get("fulltext_decision") == "INCLUDE"]
        )
        

        # Part 2 — variable extraction (only on included papers)
        if not included_df.empty:
            await run_variable_extraction(
                included_df=included_df,
                exposure=peco_values['exposure'],
                outcome=peco_values['outcome'],
                save_folder=save_folder,
                population=peco_values['population'],
                comparator=peco_values['comparison'],
            )

    asyncio.run(pipeline())
