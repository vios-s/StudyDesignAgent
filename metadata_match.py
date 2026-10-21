"""
metadata_match_structured.py

Match DAG variables to a SAIL metadata SQLite database using:
1. LLM-generated structured metadata query plans
2. Anchor-first SQL search
3. Scored candidate ranking
4. Match classification: direct_match / derivable / weak_match / not_found

Why this version is stricter than the original:
- Broad EHR terms such as "diagnosis", "clinical event", "result", or "prescription"
  are NOT allowed to confirm a match by themselves.
- The search first requires variable-specific anchor terms.
- Category and modifier terms are used to support/rank candidates.
- Derived variables are distinguished from directly observable metadata fields.

Usage:
    python metadata_match.py \
        --dag <save_folder>/extracted_variables_<E>_<O>_dag_adjustment_metadata_match_input.json \
        --db metadata_database/<dataset>_db/<dataset>_metadata.db \
        --out <save_folder>/metadata_match_results.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any

from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from dotenv import load_dotenv
from prompts.metadata_process import _metadata_keyword_system_prompt

load_dotenv()


# ---------------------------------------------------------------------------
# LLM model
# ---------------------------------------------------------------------------

_model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
    model_type="openai/gpt-5.4",
    url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model_config_dict={
        "temperature": 0.1,
        "max_tokens": 4096,
    },
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

SQLITE_TEXT_TYPES = ("TEXT", "VARCHAR", "CHAR", "CLOB", "STRING")

CORE_METADATA_TABLES = {"_dataset_overview", "_table_metadata", "_column_metadata"}

DATASET_CATEGORY_EXPANSIONS = {
    "diagnosis": [
        "diagnosis",
        "morbidity",
        "primary care",
        "general practice",
        "GP",
        "patient episode",
        "clinical record",
    ],
    "prescription": [
        "prescription",
        "prescriptions",
        "prescribing",
        "dispensing",
        "medication",
        "drug",
        "pharmacy",
    ],
    "dispensing": [
        "dispensing",
        "prescription",
        "prescribing",
        "medication",
        "drug",
        "pharmacy",
    ],
    "laboratory": [
        "laboratory",
        "pathology",
        "result",
        "results",
        "test",
        "report",
        "observation",
    ],
    "microbiology": [
        "laboratory",
        "pathology",
        "result",
        "results",
        "test",
        "report",
        "observation",
        "microbiology",
    ],
    "procedure": [
        "procedure",
        "operation",
        "surgery",
        "intervention",
        "activity",
        "treatment",
    ],
    "hospitalisation": [
        "hospital",
        "admission",
        "inpatient",
        "episode",
        "discharge",
        "patient episode",
    ],
    "consultation": [
        "consultation",
        "encounter",
        "attendance",
        "visit",
        "primary care",
        "general practice",
        "GP",
    ],
    "demographic": [
        "demographic",
        "population",
        "cohort",
        "person",
        "patient",
        "linkage",
    ],
    "lifestyle": [
        "lifestyle",
        "survey",
        "questionnaire",
        "smoking",
        "alcohol",
        "behaviour",
        "behavior",
    ],
    "maternity": [
        "maternity",
        "pregnancy",
        "birth",
        "child",
        "mother",
        "obstetric",
    ],
    "death": [
        "death",
        "mortality",
        "cause of death",
        "vital",
    ],
}

GENERIC_DATASET_TERMS = {
    "result",
    "results",
    "test",
    "tests",
    "record",
    "records",
    "clinical",
    "health",
    "patient",
    "patients",
    "cohort",
    "data",
    "dataset",
}

SPECIALISED_DATASET_TERMS = {
    "covid": ["covid", "covid-19", "coronavirus", "sars-cov-2", "sars cov 2"],
    "cancer": ["cancer", "anti-cancer", "oncology", "tumour", "tumor"],
    "dementia": ["dementia", "alzheim"],
    "cystic fibrosis": ["cystic fibrosis"],
    "renal": ["renal", "kidney", "ckd", "chronic kidney"],
}

LAB_DATASET_HINTS = {
    "wrrs",
    "welsh results reports service",
    "pathology",
    "pathology data",
    "laboratory results",
}

GENERIC_ANCHOR_STOPWORDS = {
    "record",
    "records",
    "data",
    "patient",
    "patients",
    "health",
    "clinical",
    "information",
    "event",
    "events",
    "field",
    "fields",
    "value",
    "values",
    "code",
    "codes",
}


def clean_keyword_list(values: Any, max_len: int | None = None) -> list[str]:
    """Normalise a list of keywords."""
    if not isinstance(values, list):
        return []

    cleaned: list[str] = []
    seen: set[str] = set()

    for value in values:
        if value is None:
            continue
        kw = str(value).strip()
        if not kw:
            continue
        key = kw.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(kw)

    if max_len is not None:
        cleaned = cleaned[:max_len]
    return cleaned


def remove_generic_anchor_terms(anchor_keywords: list[str]) -> list[str]:
    """Remove anchors that are too generic to confirm a variable match."""
    filtered = []
    for kw in anchor_keywords:
        if kw.lower().strip() in GENERIC_ANCHOR_STOPWORDS:
            continue
        filtered.append(kw)
    return filtered


def sql_identifier(name: str) -> str:
    """Safely quote SQLite identifiers."""
    return '"' + name.replace('"', '""') + '"'


def row_text(row_data: dict[str, Any]) -> str:
    """Concatenate row values for scoring."""
    return " ".join(str(v) for v in row_data.values() if v is not None).lower()


def keyword_in_text(text: str, keyword: str) -> bool:
    """Return whether a keyword is present, using word boundaries for short terms."""
    kw_clean = keyword.lower().strip()
    if not kw_clean:
        return False

    # Short acronyms such as UTI should not match inside words like "routine".
    if re.fullmatch(r"[a-z0-9]{2,4}", kw_clean):
        return re.search(rf"\b{re.escape(kw_clean)}\b", text) is not None

    return kw_clean in text


def count_matches(text: str, keywords: list[str]) -> tuple[int, list[str]]:
    """Count distinct substring keyword matches in text."""
    matched: list[str] = []
    seen: set[str] = set()

    for kw in keywords:
        kw_clean = kw.lower().strip()
        if not kw_clean:
            continue
        if keyword_in_text(text, kw_clean) and kw_clean not in seen:
            matched.append(kw)
            seen.add(kw_clean)

    return len(matched), matched


def normalise_child_terms(children: Any) -> list[str]:
    """Extract searchable child term strings from a DAG node's children field."""
    if not isinstance(children, list):
        return []

    terms: list[str] = []
    seen: set[str] = set()
    for child in children:
        if isinstance(child, dict):
            value = (
                child.get("variable")
                or child.get("term")
                or child.get("name")
                or child.get("label")
                or ""
            )
        else:
            value = child

        term = str(value).strip()
        key = term.lower()
        if not term or key in seen:
            continue
        seen.add(key)
        terms.append(term)

    return terms


def matched_child_terms(hits: list[dict[str, Any]], children: Any) -> list[str]:
    """Return child terms found in candidate metadata hit text."""
    child_terms = normalise_child_terms(children)
    if not child_terms:
        return []

    matched: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        text = row_text(hit.get("row_data", {}))
        _, row_matches = count_matches(text, child_terms)
        for term in row_matches:
            key = term.lower().strip()
            if key and key not in seen:
                seen.add(key)
                matched.append(term)

    return matched


def extract_json_object(content: str) -> dict[str, Any]:
    """Extract the first JSON object from an LLM response."""
    content = content.strip()

    # Prefer direct JSON parsing.
    try:
        obj = json.loads(content)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Fallback: extract first {...} block.
    match = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Could not find JSON object in LLM response:\n{content}")

    obj = json.loads(match.group())
    if not isinstance(obj, dict):
        raise ValueError(f"LLM JSON response is not an object:\n{content}")
    return obj


# ---------------------------------------------------------------------------
# Schema introspection
# ---------------------------------------------------------------------------


def get_text_columns(conn: sqlite3.Connection) -> dict[str, list[str]]:
    """Return {table_name: [text_column, ...]} for every table in the DB."""
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]

    schema: dict[str, list[str]] = {}

    for table in tables:
        cursor.execute(f"PRAGMA table_info({sql_identifier(table)})")
        cols = cursor.fetchall()

        text_cols: list[str] = []
        for col in cols:
            # col tuple: cid, name, type, notnull, dflt_value, pk
            col_name = col[1]
            col_type = (col[2] or "").upper()

            if (
                col_type == ""
                or col_type in SQLITE_TEXT_TYPES
                or "TEXT" in col_type
                or "CHAR" in col_type
                or "CLOB" in col_type
                or "STRING" in col_type
            ):
                text_cols.append(col_name)

        if text_cols:
            schema[table] = text_cols

    return schema


def has_core_metadata_schema(conn: sqlite3.Connection) -> bool:
    """Return true when the DB has the SAIL metadata tables."""
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}
    return CORE_METADATA_TABLES.issubset(tables)


def load_dataset_overviews(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Load dataset-level summaries used for first-stage retrieval."""
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT dataset_id, name, summary, documentation, keywords
        FROM _dataset_overview
        ORDER BY dataset_id
        """
    )
    return [dict(row) for row in cursor.fetchall()]


def unique_keywords(values: list[str]) -> list[str]:
    """Deduplicate while preserving order."""
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value).strip()
        key = item.lower()
        if not item or key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def build_dataset_search_terms(plan: dict[str, Any]) -> list[str]:
    """
    Build broad dataset-routing terms.

    The dataset overview usually describes a data source ("prescribing",
    "laboratory results") rather than paper-level constructs such as
    "baseline urine bacterial susceptibility to prescribed antibiotic".
    """
    terms: list[str] = []

    for category in plan.get("ehr_categories", []):
        category_key = str(category).lower().strip()
        terms.extend(DATASET_CATEGORY_EXPANSIONS.get(category_key, []))
        terms.append(category)

    terms.extend(plan.get("category_keywords", []))

    # Anchors can help when the overview names a disease/program directly, but
    # they should not dominate routing for constructed variables.
    for anchor in plan.get("anchor_keywords", []):
        if len(str(anchor).strip()) > 3:
            terms.append(anchor)

    return unique_keywords(terms)


def term_score_multiplier(term: str) -> float:
    """Down-weight broad catalogue words that are poor discriminators."""
    term_key = term.lower().strip()
    if term_key in GENERIC_DATASET_TERMS:
        return 0.35
    if term_key in {"test result", "lab test", "pathology report"}:
        return 0.6
    return 1.0


def plan_text(plan: dict[str, Any]) -> str:
    """Collect plan text for speciality mismatch checks."""
    parts: list[str] = [str(plan.get("variable") or "")]
    for key in [
        "ehr_categories",
        "category_keywords",
        "anchor_keywords",
        "modifier_keywords",
    ]:
        parts.extend(str(item) for item in plan.get(key, []) or [])
    return " ".join(parts).lower()


def specialty_mismatch_penalty(
    name_text: str,
    summary_text: str,
    documentation_text: str,
    keywords_text: str,
    plan: dict[str, Any],
) -> tuple[int, list[str]]:
    """
    Penalize highly specialised datasets when the variable is not about that domain.

    This stops broad variables like "urine culture result" being pulled into
    COVID test datasets just because both contain "test results". Keyword-only
    occurrences are penalized lightly because many SAIL datasets carry project
    tags such as COVID-19/NCS even when they are not COVID datasets.
    """
    variable_text = plan_text(plan)
    penalty = 0
    matched_domains: list[str] = []

    for domain, terms in SPECIALISED_DATASET_TERMS.items():
        if any(keyword_in_text(variable_text, term) for term in terms):
            continue

        name_summary_doc = " ".join([name_text, summary_text, documentation_text])
        strong_match = any(keyword_in_text(name_summary_doc, term) for term in terms)
        keyword_match = any(keyword_in_text(keywords_text, term) for term in terms)

        if strong_match:
            penalty += 14
            matched_domains.append(domain)
        elif keyword_match:
            penalty += 3
            matched_domains.append(domain)

    return penalty, unique_keywords(matched_domains)


def lab_dataset_boost(name_text: str, summary_text: str, plan: dict[str, Any]) -> int:
    """Boost canonical broad lab/pathology sources for lab or microbiology variables."""
    categories = {str(c).lower().strip() for c in plan.get("ehr_categories", [])}
    if not categories.intersection({"laboratory", "microbiology"}):
        return 0

    text = " ".join([name_text, summary_text])
    if any(keyword_in_text(text, hint) for hint in LAB_DATASET_HINTS):
        return 8
    return 0


def score_dataset_overview(dataset: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    """Score one dataset overview against broad routing terms."""
    terms = build_dataset_search_terms(plan)
    negative_keywords = plan.get("negative_keywords", [])

    name_text = str(dataset.get("name") or "").lower()
    summary_text = str(dataset.get("summary") or "").lower()
    documentation_text = str(dataset.get("documentation") or "").lower()
    keywords_text = str(dataset.get("keywords") or "").lower()
    all_text = " ".join([name_text, summary_text, documentation_text, keywords_text])

    matched_terms: list[str] = []
    score = 0

    for term in terms:
        term_key = term.lower().strip()
        if not term_key:
            continue
        multiplier = term_score_multiplier(term_key)
        matched = False
        if keyword_in_text(name_text, term_key):
            score += 4 * multiplier
            matched = True
        if keyword_in_text(keywords_text, term_key):
            score += 3 * multiplier
            matched = True
        if keyword_in_text(summary_text, term_key):
            score += 2 * multiplier
            matched = True
        if keyword_in_text(documentation_text, term_key):
            score += 1 * multiplier
            matched = True
        if matched:
            matched_terms.append(term)

    _, matched_negatives = count_matches(all_text, negative_keywords)
    score -= 5 * len(matched_negatives)

    specialty_penalty, matched_specialty_penalties = specialty_mismatch_penalty(
        name_text=name_text,
        summary_text=summary_text,
        documentation_text=documentation_text,
        keywords_text=keywords_text,
        plan=plan,
    )
    score -= specialty_penalty
    score += lab_dataset_boost(name_text, summary_text, plan)

    return {
        "dataset_id": dataset.get("dataset_id"),
        "dataset_name": dataset.get("name") or "",
        "dataset_summary": dataset.get("summary") or "",
        "dataset_keywords": dataset.get("keywords") or "",
        "dataset_score": round(score, 2),
        "matched_dataset_terms": unique_keywords(matched_terms),
        "matched_dataset_negatives": unique_keywords(matched_negatives + matched_specialty_penalties),
    }


def retrieve_candidate_datasets(
    dataset_overviews: list[dict[str, Any]],
    plan: dict[str, Any],
    top_n: int = 10,
) -> list[dict[str, Any]]:
    """Return the best dataset candidates for the variable."""
    scored = [
        score_dataset_overview(dataset, plan)
        for dataset in dataset_overviews
    ]
    scored = [item for item in scored if item["dataset_score"] > 0]
    scored.sort(
        key=lambda item: (
            item["dataset_score"],
            len(item["matched_dataset_terms"]),
            -int(item["dataset_id"] or 0),
        ),
        reverse=True,
    )
    return scored[:top_n]


# ---------------------------------------------------------------------------
# LLM query-plan generation
# ---------------------------------------------------------------------------


def _normalise_query_plan(plan: dict[str, Any], variable: str) -> dict[str, Any]:
    """Defensively normalise the LLM query plan."""
    plan.setdefault("variable", variable)

    plan["ehr_categories"] = clean_keyword_list(plan.get("ehr_categories", []), max_len=3)
    plan["category_keywords"] = clean_keyword_list(plan.get("category_keywords", []), max_len=8)
    plan["anchor_keywords"] = clean_keyword_list(plan.get("anchor_keywords", []), max_len=12)
    plan["modifier_keywords"] = clean_keyword_list(plan.get("modifier_keywords", []), max_len=8)
    plan["negative_keywords"] = clean_keyword_list(plan.get("negative_keywords", []), max_len=6)

    plan["anchor_keywords"] = remove_generic_anchor_terms(plan["anchor_keywords"])

    if not isinstance(plan.get("is_directly_observable"), bool):
        plan["is_directly_observable"] = True

    if "derivation_note" not in plan:
        plan["derivation_note"] = None

    if not isinstance(plan.get("minimum_match_rule"), dict):
        plan["minimum_match_rule"] = {
            "must_match_anchor": True,
            "must_match_category_or_modifier": True,
        }
    else:
        plan["minimum_match_rule"].setdefault("must_match_anchor", True)
        plan["minimum_match_rule"].setdefault("must_match_category_or_modifier", True)

    return plan


def _fallback_query_plan(variable: str, knowledge_card: str, children: list[str]) -> dict[str, Any]:
    """Simple fallback plan if the LLM response cannot be parsed."""
    anchors = [variable]
    anchors.extend(children[:6])

    text = f"{variable} {knowledge_card}".lower()

    category_keywords: list[str] = []
    ehr_categories: list[str] = []

    if any(x in text for x in ["antibiotic", "drug", "medication", "prophylactic", "prescription"]):
        ehr_categories.append("prescription")
        category_keywords.extend(["prescription", "medication", "drug", "prescribing", "pharmacy"])

    if any(x in text for x in ["diagnosis", "disease", "syndrome", "condition", "uti", "infection"]):
        ehr_categories.append("diagnosis")
        category_keywords.extend(["diagnosis", "disease", "condition", "morbidity", "clinical event"])

    if any(x in text for x in ["laboratory", "lab", "culture", "resistance", "susceptibility", "sensitivity", "organism"]):
        ehr_categories.append("microbiology")
        category_keywords.extend(["microbiology", "culture", "laboratory", "result", "susceptibility", "sensitivity"])

    if any(x in text for x in ["hospital", "hospitalisation", "hospitalization", "admission", "inpatient"]):
        ehr_categories.append("hospitalisation")
        category_keywords.extend(["hospitalisation", "hospitalization", "admission", "inpatient", "hospital"])

    if any(x in text for x in ["age", "sex", "gender", "deprivation", "ethnicity", "demographic"]):
        ehr_categories.append("demographic")
        category_keywords.extend(["demographic", "age", "sex", "gender", "ethnicity"])

    if not category_keywords:
        ehr_categories.append("derived")
        category_keywords.extend(["diagnosis", "prescription", "laboratory", "result"])

    modifier_keywords = []
    for term in ["baseline", "history", "prior", "previous", "frequency", "count", "episode", "duration"]:
        if term in text:
            modifier_keywords.append(term)

    return _normalise_query_plan(
        {
            "variable": variable,
            "ehr_categories": ehr_categories,
            "category_keywords": category_keywords,
            "anchor_keywords": anchors,
            "modifier_keywords": modifier_keywords,
            "negative_keywords": [],
            "is_directly_observable": not any(
                x in text for x in ["frequency", "count", "history", "duration", "adherence", "time to"]
            ),
            "derivation_note": None,
            "minimum_match_rule": {
                "must_match_anchor": True,
                "must_match_category_or_modifier": True,
            },
        },
        variable,
    )


def _generate_query_plan_sync(
    variable: str,
    roles: list[str],
    knowledge_card: str,
    children: list[str] | None = None,
) -> dict[str, Any]:
    children = children or []
    agent = ChatAgent(system_message=_metadata_keyword_system_prompt(), model=_model)

    prompt_obj = {
        "variable": variable,
        "roles": roles,
        "knowledge_card": knowledge_card,
        "children": children,
    }

    response = agent.step(json.dumps(prompt_obj, ensure_ascii=False, indent=2))
    content = response.msgs[0].content.strip()

    try:
        plan = extract_json_object(content)
        return _normalise_query_plan(plan, variable)
    except Exception as exc:
        print(f"WARNING: Could not parse LLM query plan for {variable!r}: {exc}")
        return _fallback_query_plan(variable, knowledge_card, children)


async def generate_query_plan(
    variable: str,
    roles: list[str],
    knowledge_card: str,
    children: list[str] | None = None,
) -> dict[str, Any]:
    return await asyncio.to_thread(
        _generate_query_plan_sync,
        variable,
        roles,
        knowledge_card,
        children,
    )


# ---------------------------------------------------------------------------
# SQL search
# ---------------------------------------------------------------------------


def build_anchor_prefilter_sql(
    table: str,
    columns: list[str],
    anchor_keywords: list[str],
    limit: int,
) -> tuple[str, list[str]]:
    """
    Build SQL that prefilters rows by anchor keywords only.

    Category terms are deliberately excluded from this prefilter because they are
    too broad and cause false positives.
    """
    col_selects = ", ".join(sql_identifier(col) for col in columns)

    conditions: list[str] = []
    params: list[str] = []

    for kw in anchor_keywords:
        kw = kw.strip().lower()
        if not kw:
            continue
        for col in columns:
            conditions.append(f"LOWER({sql_identifier(col)}) LIKE ?")
            params.append(f"%{kw}%")

    if not conditions:
        # Deliberately returns no rows.
        return f"SELECT rowid, {col_selects} FROM {sql_identifier(table)} WHERE 1 = 0", []

    sql = f"""
        SELECT rowid, {col_selects}
        FROM {sql_identifier(table)}
        WHERE {' OR '.join(conditions)}
        LIMIT {int(limit)}
    """

    return sql, params


def score_row(row_data: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    """Score one metadata row against a query plan."""
    text = row_text(row_data)

    category_keywords = plan.get("category_keywords", [])
    anchor_keywords = plan.get("anchor_keywords", [])
    modifier_keywords = plan.get("modifier_keywords", [])
    negative_keywords = plan.get("negative_keywords", [])

    anchor_score, matched_anchors = count_matches(text, anchor_keywords)
    category_score, matched_categories = count_matches(text, category_keywords)
    modifier_score, matched_modifiers = count_matches(text, modifier_keywords)
    negative_score, matched_negatives = count_matches(text, negative_keywords)

    # Weights: anchors are most important; category/modifier support the match.
    total_score = (
        3 * anchor_score
        + 1 * category_score
        + 1 * modifier_score
        - 4 * negative_score
    )

    return {
        "score": total_score,
        "anchor_score": anchor_score,
        "category_score": category_score,
        "modifier_score": modifier_score,
        "negative_score": negative_score,
        "matched_anchors": matched_anchors,
        "matched_categories": matched_categories,
        "matched_modifiers": matched_modifiers,
        "matched_negatives": matched_negatives,
    }


def passes_minimum_match(
    score_info: dict[str, Any],
    min_score: int = 2,
    require_anchor: bool = True,
    require_category_or_modifier: bool = False,
) -> bool:
    """Gate rows before they enter the hit list."""
    if require_anchor and score_info["anchor_score"] < 1:
        return False

    # Only enforce category/modifier requirement when explicitly requested
    # (used in pass 2 where there is no anchor match guarantee)
    if require_category_or_modifier:
        if score_info["category_score"] < 1 and score_info["modifier_score"] < 1:
            return False

    if score_info["score"] < min_score:
        return False

    return True


def _run_prefilter(
    cursor: sqlite3.Cursor,
    schema: dict[str, list[str]],
    keywords: list[str],
    max_rows_per_table: int,
) -> list[tuple[str, int, dict]]:
    """Execute a prefilter across all tables; return (table, rowid, row_data) tuples."""
    results: list[tuple[str, int, dict]] = []
    seen: set[tuple[str, int]] = set()

    for table, columns in schema.items():
        sql, params = build_anchor_prefilter_sql(
            table=table,
            columns=columns,
            anchor_keywords=keywords,
            limit=max_rows_per_table,
        )
        try:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
        except sqlite3.OperationalError as exc:
            print(f"WARNING: SQL failed for table {table!r}: {exc}")
            continue

        for row in rows:
            if not row:
                continue
            rowid = int(row[0])
            row_key = (table, rowid)
            if row_key in seen:
                continue
            seen.add(row_key)
            row_data = dict(zip(columns, row[1:]))
            results.append((table, rowid, row_data))

    return results


def search_variable_structured(
    conn: sqlite3.Connection,
    schema: dict[str, list[str]],
    plan: dict[str, Any],
    max_rows_per_table: int = 500,
    min_score: int = 2,
) -> list[dict[str, Any]]:
    """
    Two-pass search:
    Pass 1 — anchor prefilter (specific terms). Hits require anchor_score >= 1.
    Pass 2 — category/modifier fallback if pass 1 returns nothing.
             These hits are tagged category_only=True and capped at weak_match.
    """
    cursor = conn.cursor()

    anchor_keywords = plan.get("anchor_keywords", [])
    category_keywords = plan.get("category_keywords", [])
    modifier_keywords = plan.get("modifier_keywords", [])

    # --- Pass 1: anchor prefilter ---
    anchor_rows = _run_prefilter(cursor, schema, anchor_keywords, max_rows_per_table)

    hits: list[dict[str, Any]] = []

    for table, rowid, row_data in anchor_rows:
        score_info = score_row(row_data, plan)
        # Pass 1: anchor required; no category/modifier hard requirement
        # (anchor alone at weight 3 already gives score=3, well above min_score=2)
        if not passes_minimum_match(score_info, min_score=min_score, require_anchor=True,
                                    require_category_or_modifier=False):
            continue
        hits.append({
            "source_table": table,
            "rowid": rowid,
            "row_data": row_data,
            "category_only": False,
            **score_info,
        })

    # --- Pass 2: category/modifier fallback (only if pass 1 found nothing) ---
    if not hits:
        fallback_keywords = list(dict.fromkeys(category_keywords + modifier_keywords))
        fallback_rows = _run_prefilter(cursor, schema, fallback_keywords, max_rows_per_table)

        for table, rowid, row_data in fallback_rows:
            score_info = score_row(row_data, plan)
            # Pass 2: no anchor required; must have at least one category/modifier hit
            # min_score=1 because category weight=1, so a single match gives score=1
            if not passes_minimum_match(score_info, min_score=1, require_anchor=False,
                                        require_category_or_modifier=True):
                continue
            hits.append({
                "source_table": table,
                "rowid": rowid,
                "row_data": row_data,
                "category_only": True,
                **score_info,
            })

    return sorted(
        hits,
        key=lambda h: (
            not h["category_only"],  # anchor hits first
            h["score"],
            h["anchor_score"],
            h["category_score"],
            h["modifier_score"],
            -h["negative_score"],
        ),
        reverse=True,
    )


def _fetch_metadata_column_rows(
    conn: sqlite3.Connection,
    dataset_ids: list[int] | None = None,
) -> list[tuple[int, dict[str, Any]]]:
    """
    Fetch joined dataset/table/column metadata rows.

    Returns tuples of (stable_row_number, row_data). The database has no single
    rowid after the join, so a deterministic sequence number is enough for output.
    """
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    where = ""
    params: list[Any] = []
    if dataset_ids:
        placeholders = ", ".join("?" for _ in dataset_ids)
        where = f"WHERE c.dataset_id IN ({placeholders})"
        params.extend(dataset_ids)

    cursor.execute(
        f"""
        SELECT
            o.dataset_id,
            o.name AS dataset_name,
            o.summary AS dataset_summary,
            o.keywords AS dataset_keywords,
            c.table_name,
            t.table_description,
            c.column_name,
            c.column_description,
            c.original_data_type,
            c.sensitive
        FROM _column_metadata c
        JOIN _dataset_overview o
             ON o.dataset_id = c.dataset_id
        LEFT JOIN _table_metadata t
             ON t.dataset_id = c.dataset_id
            AND t.table_name = c.table_name
        {where}
        ORDER BY o.dataset_id, c.table_name, c.column_name
        """,
        params,
    )

    return [
        (idx, dict(row))
        for idx, row in enumerate(cursor.fetchall(), start=1)
    ]


def _score_metadata_rows(
    rows: list[tuple[int, dict[str, Any]]],
    plan: dict[str, Any],
    dataset_scores: dict[int, dict[str, Any]],
    min_score: int,
    require_anchor: bool,
    require_category_or_modifier: bool,
    category_only: bool,
    global_fallback: bool,
) -> list[dict[str, Any]]:
    """Score joined metadata rows and return retained hits."""
    hits: list[dict[str, Any]] = []

    for rownum, row_data in rows:
        column_scoring_data = {
            "table_name": row_data.get("table_name"),
            "table_description": row_data.get("table_description"),
            "column_name": row_data.get("column_name"),
            "column_description": row_data.get("column_description"),
            "original_data_type": row_data.get("original_data_type"),
        }
        score_info = score_row(column_scoring_data, plan)
        if not passes_minimum_match(
            score_info,
            min_score=min_score,
            require_anchor=require_anchor,
            require_category_or_modifier=require_category_or_modifier,
        ):
            continue

        dataset_id = int(row_data.get("dataset_id") or 0)
        dataset_info = dataset_scores.get(dataset_id, {})
        dataset_score = float(dataset_info.get("dataset_score", 0) or 0)

        hits.append({
            "source_table": row_data.get("table_name") or "",
            "rowid": rownum,
            "row_data": row_data,
            "dataset_id": dataset_id,
            "dataset_name": row_data.get("dataset_name") or "",
            "dataset_score": dataset_score,
            "matched_dataset_terms": dataset_info.get("matched_dataset_terms", []),
            "category_only": category_only,
            "global_fallback": global_fallback,
            **score_info,
        })

    return hits


def search_variable_core_metadata(
    conn: sqlite3.Connection,
    dataset_overviews: list[dict[str, Any]],
    plan: dict[str, Any],
    candidate_dataset_count: int = 10,
    min_score: int = 2,
) -> list[dict[str, Any]]:
    """
    Dataset-aware search for the SAIL metadata schema.

    Stage 1 routes the variable to likely datasets using broad overview terms.
    Stage 2 searches table/column descriptions inside those datasets. If no
    row-level evidence is found, a global anchor fallback protects recall.
    """
    candidate_datasets = retrieve_candidate_datasets(
        dataset_overviews=dataset_overviews,
        plan=plan,
        top_n=candidate_dataset_count,
    )
    candidate_ids = [
        int(item["dataset_id"])
        for item in candidate_datasets
        if item.get("dataset_id") is not None
    ]
    dataset_scores = {
        int(item["dataset_id"]): item
        for item in candidate_datasets
        if item.get("dataset_id") is not None
    }

    hits: list[dict[str, Any]] = []

    if candidate_ids:
        candidate_rows = _fetch_metadata_column_rows(conn, candidate_ids)
        hits.extend(
            _score_metadata_rows(
                rows=candidate_rows,
                plan=plan,
                dataset_scores=dataset_scores,
                min_score=min_score,
                require_anchor=True,
                require_category_or_modifier=False,
                category_only=False,
                global_fallback=False,
            )
        )

        if not hits:
            hits.extend(
                _score_metadata_rows(
                    rows=candidate_rows,
                    plan=plan,
                    dataset_scores=dataset_scores,
                    min_score=1,
                    require_anchor=False,
                    require_category_or_modifier=True,
                    category_only=True,
                    global_fallback=False,
                )
            )

    # Recall fallback: if overview routing missed the source dataset, search all
    # columns but require variable-specific anchor evidence.
    if not hits:
        global_rows = _fetch_metadata_column_rows(conn, None)
        hits.extend(
            _score_metadata_rows(
                rows=global_rows,
                plan=plan,
                dataset_scores={},
                min_score=min_score,
                require_anchor=True,
                require_category_or_modifier=False,
                category_only=False,
                global_fallback=True,
            )
        )

    return sorted(
        hits,
        key=lambda h: (
            not h["global_fallback"],
            not h["category_only"],
            h["dataset_score"],
            h["score"],
            h["anchor_score"],
            h["category_score"],
            h["modifier_score"],
            -h["negative_score"],
        ),
        reverse=True,
    )


# ---------------------------------------------------------------------------
# Match classification
# ---------------------------------------------------------------------------


def classify_match(hits: list[dict[str, Any]], plan: dict[str, Any]) -> str:
    """Classify the variable-level match status."""
    if not hits:
        return "not_found"

    best = hits[0]
    directly_observable = bool(plan.get("is_directly_observable", True))
    category_only = bool(best.get("category_only", False))

    # Category-only hits (pass 2 fallback): cap at weak_match regardless of score
    if category_only:
        return "weak_match"

    anchor_score = best["anchor_score"]
    category_score = best["category_score"]
    modifier_score = best["modifier_score"]
    score = best["score"]

    strong = anchor_score >= 1 and category_score >= 1 and score >= 5
    acceptable = anchor_score >= 1 and (category_score >= 1 or modifier_score >= 1) and score >= 3
    anchor_only = anchor_score >= 1 and score >= 2  # specific anchor matched but no category/modifier

    if strong and directly_observable:
        return "direct_match"

    if (strong or acceptable) and not directly_observable:
        return "derivable"

    if acceptable or anchor_only:
        return "weak_match"

    return "not_found"


def diversify_hits_by_dataset(hits: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    """
    Keep only the best hit per dataset for output.

    The matcher scores metadata rows, but the user-facing question is usually
    "which datasets can provide this variable?" Repeating five columns from the
    same dataset makes the output look like multiple matches when it is really
    one dataset-level candidate.
    """
    selected: list[dict[str, Any]] = []
    seen_dataset_ids: set[Any] = set()

    for hit in hits:
        dataset_id = hit.get("dataset_id")
        dataset_key = dataset_id if dataset_id not in (None, "") else (
            hit.get("dataset_name"),
            hit.get("source_table"),
        )
        if dataset_key in seen_dataset_ids:
            continue
        seen_dataset_ids.add(dataset_key)
        selected.append(hit)
        if len(selected) >= top_k:
            break

    return selected


# ---------------------------------------------------------------------------
# Per-variable async task
# ---------------------------------------------------------------------------


async def process_node(
    node: dict[str, Any],
    db_path: str,
    schema: dict[str, list[str]],
    dataset_overviews: list[dict[str, Any]] | None,
    use_core_metadata_schema: bool,
    top_k: int,
    max_rows_per_table: int,
    min_score: int,
    candidate_dataset_count: int,
) -> dict[str, Any]:
    """Generate query plan and search DB for one DAG variable."""
    variable = str(node.get("variable", "")).strip()
    roles_list = node.get("roles", []) or []
    roles = ", ".join(str(r) for r in roles_list)
    knowledge_card = str(node.get("knowledge_card", ""))
    children = node.get("children", []) or []

    if not variable:
        return {
            "variable": "",
            "roles": roles,
            "status": "not_found",
            "plan": {},
            "hits": [],
            "rows": [],
        }

    plan = await generate_query_plan(
        variable=variable,
        roles=roles_list,
        knowledge_card=knowledge_card,
        children=children,
    )

    print(f"\n[{variable}]")
    print(f"  categories: {plan.get('ehr_categories', [])}")
    print(f"  anchor_keywords: {plan.get('anchor_keywords', [])}")
    print(f"  directly_observable: {plan.get('is_directly_observable')}")

    def _search() -> list[dict[str, Any]]:
        conn = sqlite3.connect(db_path)
        try:
            if use_core_metadata_schema and dataset_overviews is not None:
                return search_variable_core_metadata(
                    conn=conn,
                    dataset_overviews=dataset_overviews,
                    plan=plan,
                    candidate_dataset_count=candidate_dataset_count,
                    min_score=min_score,
                )
            return search_variable_structured(
                conn=conn,
                schema=schema,
                plan=plan,
                max_rows_per_table=max_rows_per_table,
                min_score=min_score,
            )
        finally:
            conn.close()

    hits = await asyncio.to_thread(_search)
    status = classify_match(hits, plan)
    child_terms = normalise_child_terms(children)
    if child_terms and status != "not_found":
        child_matches = matched_child_terms(hits, child_terms)
        if child_matches:
            print(f"  matched_children: {child_matches}")
        else:
            # Downgrade rather than reject — the parent concept itself may be matchable
            # even when DAG children are too specific to appear in column-level metadata.
            if status in ("direct_match", "derivable"):
                print(f"  matched_children: none; downgrading {status} → weak_match")
                status = "weak_match"
            else:
                print(f"  matched_children: none; keeping {status}")

    print(f"  hits: {len(hits)}")
    print(f"  status: {status}")

    rows: list[dict[str, Any]] = []

    base_row = {
        "variable": variable,
        "roles": roles,
        "match_status": status,
        "ehr_categories": "; ".join(plan.get("ehr_categories", [])),
        "category_keywords": "; ".join(plan.get("category_keywords", [])),
        "anchor_keywords": "; ".join(plan.get("anchor_keywords", [])),
        "modifier_keywords": "; ".join(plan.get("modifier_keywords", [])),
        "negative_keywords": "; ".join(plan.get("negative_keywords", [])),
        "is_directly_observable": plan.get("is_directly_observable"),
        "derivation_note": plan.get("derivation_note"),
    }

    if not hits:
        rows.append(
            {
                **base_row,
                "rank": 1,
                "score": 0,
                "anchor_score": 0,
                "category_score": 0,
                "modifier_score": 0,
                "negative_score": 0,
                "matched_anchors": "",
                "matched_categories": "",
                "matched_modifiers": "",
                "matched_negatives": "",
                "dataset_id": "",
                "dataset_name": "",
                "dataset_score": 0,
                "matched_dataset_terms": "",
                "global_fallback": False,
                "source_table": "",
                "rowid": "",
                "row_preview": "NO MATCH",
            }
        )
    else:
        display_hits = diversify_hits_by_dataset(hits, top_k)

        for rank, hit in enumerate(display_hits, start=1):
            row_preview = " | ".join(
                f"{k}={v}" for k, v in hit["row_data"].items() if v is not None and str(v).strip()
            )

            rows.append(
                {
                    **base_row,
                    "rank": rank,
                    "score": hit["score"],
                    "anchor_score": hit["anchor_score"],
                    "category_score": hit["category_score"],
                    "modifier_score": hit["modifier_score"],
                    "negative_score": hit["negative_score"],
                    "matched_anchors": "; ".join(hit["matched_anchors"]),
                    "matched_categories": "; ".join(hit["matched_categories"]),
                    "matched_modifiers": "; ".join(hit["matched_modifiers"]),
                    "matched_negatives": "; ".join(hit["matched_negatives"]),
                    "dataset_id": hit.get("dataset_id", ""),
                    "dataset_name": hit.get("dataset_name", ""),
                    "dataset_score": hit.get("dataset_score", 0),
                    "matched_dataset_terms": "; ".join(hit.get("matched_dataset_terms", [])),
                    "global_fallback": hit.get("global_fallback", False),
                    "source_table": hit["source_table"],
                    "rowid": hit["rowid"],
                    "row_preview": row_preview[:800],
                }
            )

    return {
        "variable": variable,
        "roles": roles,
        "status": status,
        "plan": plan,
        "hits": hits,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


MATCH_CSV_FIELDNAMES = [
    "variable",
    "roles",
    "match_status",
    "ehr_categories",
    "category_keywords",
    "anchor_keywords",
    "modifier_keywords",
    "negative_keywords",
    "is_directly_observable",
    "derivation_note",
    "rank",
    "score",
    "anchor_score",
    "category_score",
    "modifier_score",
    "negative_score",
    "matched_anchors",
    "matched_categories",
    "matched_modifiers",
    "matched_negatives",
    "dataset_id",
    "dataset_name",
    "dataset_score",
    "matched_dataset_terms",
    "global_fallback",
    "source_table",
    "rowid",
    "row_preview",
]


async def match_variables(
    dag_nodes: list[dict[str, Any]],
    db_path: str,
    top_k: int = 5,
    max_rows_per_table: int = 500,
    min_score: int = 2,
    candidate_dataset_count: int = 10,
    concurrency: int = 4,
) -> list[dict[str, Any]]:
    """
    Match DAG variable nodes against the metadata DB.

    Returns one result per node: {"variable", "roles", "status", "plan", "hits", "rows"},
    where status is direct_match / derivable / weak_match / not_found.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    dataset_overviews: list[dict[str, Any]] | None = None
    use_core_metadata_schema = False

    conn = sqlite3.connect(str(db_path))
    try:
        use_core_metadata_schema = has_core_metadata_schema(conn)
        if use_core_metadata_schema:
            dataset_overviews = load_dataset_overviews(conn)
        schema = get_text_columns(conn)
    finally:
        conn.close()

    if not schema:
        raise ValueError(f"No text columns found in the database {db_path}. Is it populated?")

    print(f"Schema tables with text columns: {list(schema.keys())}")
    if use_core_metadata_schema:
        print(
            f"Using dataset-aware metadata search across "
            f"{len(dataset_overviews or [])} dataset overviews."
        )
    print(f"Processing {len(dag_nodes)} variables...\n")

    semaphore = asyncio.Semaphore(concurrency)

    async def guarded_process(node: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            return await process_node(
                node=node,
                db_path=str(db_path),
                schema=schema,
                dataset_overviews=dataset_overviews,
                use_core_metadata_schema=use_core_metadata_schema,
                top_k=top_k,
                max_rows_per_table=max_rows_per_table,
                min_score=min_score,
                candidate_dataset_count=candidate_dataset_count,
            )

    return list(await asyncio.gather(*(guarded_process(node) for node in dag_nodes)))


def write_match_csv(results: list[dict[str, Any]], out_path: str | Path) -> None:
    rows_out: list[dict[str, Any]] = []
    for result in results:
        rows_out.extend(result["rows"])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MATCH_CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"\nResults written to {out_path}")


def write_query_plans(results: list[dict[str, Any]], plan_out_path: str | Path) -> None:
    plan_out_path = Path(plan_out_path)
    plan_out_path.parent.mkdir(parents=True, exist_ok=True)
    plan_records = [
        {
            "variable": r["variable"],
            "roles": r["roles"],
            "match_status": r["status"],
            "query_plan": r["plan"],
        }
        for r in results
    ]
    with open(plan_out_path, "w", encoding="utf-8") as f:
        json.dump(plan_records, f, indent=2, ensure_ascii=False)
    print(f"Query plans written to {plan_out_path}")


def print_match_summary(results: list[dict[str, Any]]) -> None:
    total = len(results)
    status_counts: dict[str, int] = {}
    for result in results:
        status_counts[result["status"]] = status_counts.get(result["status"], 0) + 1

    n_found = sum(status_counts.get(s, 0) for s in ["direct_match", "derivable", "weak_match"])
    n_not_found = status_counts.get("not_found", 0)

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total variables : {total}")
    print(f"Found (total)   : {n_found}  "
          f"[direct={status_counts.get('direct_match', 0)}, "
          f"derivable={status_counts.get('derivable', 0)}, "
          f"weak={status_counts.get('weak_match', 0)}]")
    print(f"Not found       : {n_not_found}")

    by_status = {s: [] for s in ["direct_match", "derivable", "weak_match", "not_found"]}
    for r in results:
        by_status.setdefault(r["status"], []).append(r)

    if by_status["not_found"]:
        print("\nNOT FOUND VARIABLES")
        for item in by_status["not_found"]:
            print(f"  - {item['variable']} [{item['roles']}]")

    if by_status["weak_match"]:
        print("\nWEAK MATCH VARIABLES (found via category terms only; specific field not confirmed)")
        for item in by_status["weak_match"]:
            print(f"  - {item['variable']} [{item['roles']}]")

    if by_status["derivable"]:
        print("\nDERIVABLE VARIABLES (found in DB; must be computed from raw records)")
        for item in by_status["derivable"]:
            note = item["plan"].get("derivation_note") or ""
            print(f"  - {item['variable']} [{item['roles']}]  → {note}")


async def _run(args: argparse.Namespace) -> None:
    dag_path = Path(args.dag)
    if not dag_path.exists():
        raise FileNotFoundError(f"DAG JSON not found: {dag_path}")

    with open(dag_path, encoding="utf-8") as f:
        dag_nodes = json.load(f)

    if not isinstance(dag_nodes, list):
        raise ValueError("DAG JSON must be a list of variable nodes.")

    results = await match_variables(
        dag_nodes,
        db_path=args.db,
        top_k=args.top_k,
        max_rows_per_table=args.max_rows_per_table,
        min_score=args.min_score,
        candidate_dataset_count=args.candidate_datasets,
        concurrency=args.concurrency,
    )

    write_match_csv(results, args.out)
    if args.plan_out:
        write_query_plans(results, args.plan_out)
    print_match_summary(results)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Match DAG variables to SAIL metadata DB using structured LLM query plans."
    )

    parser.add_argument(
        "--dag",
        default="",
        help="Path to the DAG JSON file.",
    )
    parser.add_argument(
        "--db",
        default="",
        help="Path to database.",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--plan-out",
        default="",
        help="Optional JSON output path for LLM query plans. Use empty string to disable.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Maximum candidate rows to keep per variable.",
    )
    parser.add_argument(
        "--max-rows-per-table",
        type=int,
        default=500,
        help="Maximum anchor-prefiltered rows to scan per metadata table.",
    )
    parser.add_argument(
        "--min-score",
        type=int,
        default=2,
        help="Minimum score for a pass-1 (anchor) candidate row to be retained.",
    )
    parser.add_argument(
        "--candidate-datasets",
        type=int,
        default=10,
        help="Number of dataset-overview candidates to keep before table/column search.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Number of variables to process in parallel.",
    )

    args = parser.parse_args()

    if args.plan_out == "":
        args.plan_out = None

    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
