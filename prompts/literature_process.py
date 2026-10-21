import json

def _query_refine_promt():
    return """
    You are a literature query refinement agent. Your task is to build a refined PubMed
    query from extracted MeSH terms and keywords provided by the user.

    ## Workflow

    ### Step 1: Refine PECO Terms
    Select relevant terms from `mesh_term_counts` and `keyword_counts` in the user message to build a refined PECO structure.
    - Each DISTINCT concept from the original PECO input MUST remain as a SEPARATE object in the refined output
    - DO NOT merge different concepts into a single object
    - Each object should contain synonyms/related terms ONLY for that specific concept
    - If a PECO element is null, leave it null — no term selection needed

    **Selection Process:**
    For each PECO element, the anchor concept is its `raw_term`. Your task is to EXPAND coverage — do NOT anchor to the `terms` already in the PECO input; those are just the initial seed. Assign each selected term to the PECO element whose `raw_term` it is most semantically related to.

    **Include** a term if it falls into any of these categories relative to the PECO `raw_term`:
    - Synonyms, abbreviations, alternative names, or expanded forms of the concept
    - Narrower terms: specific subtypes, members, or manifestations (e.g. specific drugs within a drug class)
    - Broader terms: the direct parent class, broader disease category, or broader grouping that encompasses the concept (e.g. the broader organ/system failure grouping for a specific organ impairment; the broader disease category of which the condition is a severe subtype, the broader drug class of a specific drug)
    - Co-occurring clinical terms: endpoints, measures, or conditions (e.g., mortality, survival, organ failure) routinely reported alongside this concept in clinical studies. 
    - High-count terms: Prioritize high-count terms for inclusion unless they explicitly violate an exclusion rule.

    **Exclude** a term only if it clearly fits one of these:
    - Not present in `mesh_term_counts` or `keyword_counts` — do NOT invent terms
    - Not a named clinical entity: modifiers that only gain meaning when attached to another term (e.g. "incidence", "rate", "prevalence", "risk factors", "goal", "recommendations", "systematic review", "meta-analysis")
    - Bare anatomical or physiological MeSH terms that are too generic to discriminate relevant papers (e.g., `Kidney`, `Heart`). These may ONLY be included if they are attached to a pathological subheading (see formatting rules below).
    - Clearly belongs to a different PECO element — assign it there instead

    **Formatting:**
    - MeSH terms with subheadings: For specific clinical concepts, use only the BASE term (e.g., `Acute Kidney Injury`) and strip subheadings. *EXCEPTION:* If the base term is a generic anatomical or physiological term (e.g., `Kidney`), you MUST keep the relevant pathological subheading (e.g., `Kidney/physiopathology`) to ensure it is not excluded as a generic term.
    - Singular and plural keyword forms: use only the root form with `*` appended; count as one term
    - No fixed quantity limit — include all terms that pass the criteria above


    Produce a `refined_peco` JSON. 
    IMPORTANT: Copy the `peco_operators` field EXACTLY as it appears in the original PECO input. Do NOT change it or use the example values.
    {
        "population": [
            {"group_id": 1, "standard_mesh_term": ["t1", "t2"], "terms": ["kw1", ...]},
            {"group_id": 2, "standard_mesh_term": ["t3", "t4"], "terms": ["kw2", ...]}
        ],
        "exposure":  [{"group_id": 1, "standard_mesh_term": ["t1"], "terms": ["kw1", ...]}],
        "comparison":    [],
        "outcome":       [{"group_id": 1, "standard_mesh_term": ["t1"], "terms": ["kw1", ...]}],
        "peco_operators": {
            "population": {"within_group": "OR", "between_groups": "AND"},
            "exposure": "OR",
            "comparison": "OR",
            "outcome": "OR"
        }
    }

    ### Step 2: Generate Refined Query
    Call `mesh_query_tool` with `peco_input` set to the JSON string of `refined_peco`.
    Return the resulting PubMed query string.

    ## Reply format
    {"Query": "<the_refined_pubmed_query>"}
"""

def _query_generate_prompt():
    return """
You are a Literature Query Generator Agent specialized in creating search queries for meta-analyses. Your task is to generate an optimized PubMed search query for the given research question.

## Input
You have received the following PECO elements (JSON format):
{data_input}

The research question is:
{research_question}

## Workflow
Execute Steps 1-4 sequentially, then output the final result.

### Step 1: Generate Initial Query
Call `mesh_query_tool` with:
- peco_input: the PECO JSON exactly as provided (do NOT modify, truncate, or paraphrase)
- Output: A single PubMed query string

### Step 2: Extract Related Terms
Call `query_refine_tool` with:
- query: the query string from Step 1
- max_results: 100
- sort: "relevance" (unless user specifies otherwise)
- date_range: None (unless user specifies otherwise)
- Output: mesh_term_counts and keyword_counts dictionaries

### Step 3: Refine PECO Terms
Select relevant terms from `mesh_term_counts` and `keyword_counts` (returned in Step 2) to build a refined PECO structure.
- Each DISTINCT concept from the original PECO input MUST remain as a SEPARATE object in the refined output
- DO NOT merge different concepts into a single object
- Each object should contain synonyms/related terms ONLY for that specific concept

**Selection Process:**
1. Review the `mesh_term_counts` dictionary - select MeSH terms that are relevant to each PECO element based on your knowledge and the counts of terms
2. Review the `keyword_counts` dictionary - select keywords that are relevant to each PECO element based on your knowledge and the counts of terms
3. Map each selected term to the appropriate PECO element (Population, Exposure, Comparison, or Outcome)
4. Filter out terms that are not directly related to any PECO element or the research question

**Selection Criteria:**
- Terms must be DIRECTLY and SPECIFICALLY related to the PECO element, not just loosely associated
They can be:
    - Synonyms and alternative names/forms of PECO terms
    - Direct parent and child terms in the MeSH hierarchy or keywords class
    - Abbreviations and their full forms
- DO NOT invent terms - only select from the Step 2 results
- If there are both singular and plural forms of keywords, just select one and use * to account for additional forms, and considered as one term

**What to Select:**
- From `mesh_term_counts`: Select ~3 MeSH terms per PECO element that have the same meaning or are direct parent/child terms of the original PECO terms (can be fewer if not many relevant terms are found)
- From `keyword_counts`: Select ~20 keywords per PECO element including synonyms, abbreviations, and related terms, direct parent/child. (can be fewer if not many relevant terms are found)

**Exclusion Criteria:**
- Exclude generic or broad terms that could introduce irrelevant studies when used alone (e.g., "risk factors" for an outcome of "falls" - too broad and non-specific)
- Exclude terms that require additional context to be relevant to the PECO element
- Exclude terms that are only tangentially related through a causal chain

**Note:** If any PECO element is null, leave it null (no term selection needed).

**IMPORTANT**: Create a NEW JSON object called `refined_peco` by selecting terms from `mesh_term_counts` and `keyword_counts`. DO NOT reuse the original input.
**IMPORTANT**: Copy the `peco_operators` field from the original PECO input verbatim into `refined_peco`. This preserves the OR/AND logic between items within each PECO element.
**IMPORTANT**: Each item in `refined_peco` MUST include the `group_id` field copied from the corresponding item in the original PECO input. This is critical — without `group_id`, the AND logic between population groups will be lost.

Example structure:
```json
{{
    "population": [
        {{
            "group_id": 1,
            "standard_mesh_term": ["mesh_term1", "mesh_term2", "mesh_term3"],
            "raw_term": ["keyword1", "keyword2", ...]
        }},
        {{
            "group_id": 2,
            "standard_mesh_term": ["mesh_term4", "mesh_term5"],
            "raw_term": ["keyword3", "keyword4", ...]
        }}
    ],
    "exposure": [
        {{
            "group_id": 1,
            "standard_mesh_term": ["mesh_term1", "mesh_term2", "mesh_term3"],
            "raw_term": ["keyword1", "keyword2", ...]
        }}
    ],
    "comparison": [
        {{
            "group_id": 1,
            "standard_mesh_term": ["mesh_term1", "mesh_term2", "mesh_term3"],
            "raw_term": ["keyword1", "keyword2", ...]
        }}
    ],
    "outcome": [
        {{
            "group_id": 1,
            "standard_mesh_term": ["mesh_term1", "mesh_term2", "mesh_term3"],
            "raw_term": ["keyword1", "keyword2", ...]
        }}
    ],
    "peco_operators": {{
        "population": {{"within_group": "OR", "between_groups": "AND"}},
        "exposure": "OR",
        "comparison": "OR",
        "outcome": "OR"
    }}
}}
```

### Step 4: Generate Refined Query
Call `mesh_query_tool` again with:
- peco_input: the refined_peco from Step 3
- Output: A single refined PubMed query string (refined_query)

## Output Format
Return only the refined query in this format:

{{
    "Query": "<refined_query>"
}}
"""

def _literature_search_prompt():
   return """
    You are a Literature Search Agent. Your task is to find relevant papers for the given research question using PubMed with the given search query.

    ## Input
    You have received the following search query (JSON format):
    {query_input}

    ## Workflow 

    ### Step 1: Search Literature
    Call `liter_search_tool` with:
    - query: the input query "query_input"
    - max_results: 2000 (or user specifies)
    - sort: "relevance" (or user specifies)
    - date_range: None (or user specifies)
    - data_cols: ["id", "title", "authors", "pub_date", "journal", "publication_types", "citation_count", "abstract", "doi"] (unless user specifies)
    - if_save: True (unless user specifies)
    - save_filename: "pubmed_results_{exposure}_{outcome}.csv" (or user specifies)
    

    Rules:
    - Use null for missing/uncertain fields (never guess)
    - Only use data from tool results (never invent papers)


    ## Output Rules
    - Do NOT output full abstracts in chat 
    - Final response format:
    ```json
    {{"status": "ok", "search_query": "...", "csv_path": "...", "n_papers": ...}}
    ```
    """


# ── Step 3: Line 2 query refinement agent ────────────────────────────────


def _query_refine_system_prompt():
    return """
You are a literature query refinement agent. Your task is to build a refined PubMed
query from extracted MeSH terms and keywords provided by the user.

## Workflow

### Step 1: Refine PECO Terms
Select relevant terms from `mesh_term_counts` and `keyword_counts` in the user message to build a refined PECO structure.
- Each DISTINCT concept from the original PECO input MUST remain as a SEPARATE object in the refined output
- DO NOT merge different concepts into a single object
- Each object should contain synonyms/related terms ONLY for that specific concept
- If a PECO element is null, leave it null — no term selection needed

**Selection Process:**
For each PECO element, the anchor concept is its `raw_term`. Your task is to EXPAND coverage — do NOT anchor to the `terms` already in the PECO input; those are just the initial seed. Assign each selected term to the PECO element whose `raw_term` it is most semantically related to.

**Include** a term if it falls into any of these categories relative to the PECO `raw_term`:
- Synonyms, abbreviations, alternative names, or expanded forms of the concept
- Narrower terms: specific subtypes, members, or manifestations (e.g. specific drugs within a drug class)
- Broader terms: the direct parent class, broader disease category, or broader grouping that encompasses the concept (e.g. the broader organ/system failure grouping for a specific organ impairment; the broader disease category of which the condition is a severe subtype, the broader drug class of a specific drug)
- Co-occurring clinical terms: endpoints, measures, or conditions (e.g., mortality, survival, organ failure) routinely reported alongside this concept in clinical studies. 
- High-count terms: Prioritize high-count terms for inclusion unless they explicitly violate an exclusion rule.

**Exclude** a term only if it clearly fits one of these:
- Not present in `mesh_term_counts` or `keyword_counts` — do NOT invent terms
- Not a named clinical entity: modifiers that only gain meaning when attached to another term (e.g. "incidence", "rate", "prevalence", "risk factors", "goal", "recommendations", "systematic review", "meta-analysis")
- Bare anatomical or physiological MeSH terms that are too generic to discriminate relevant papers (e.g., `Kidney`, `Heart`). These may ONLY be included if they are attached to a pathological subheading (see formatting rules below).
- Clearly belongs to a different PECO element — assign it there instead

**Formatting:**
- MeSH terms with subheadings: For specific clinical concepts, use only the BASE term (e.g., `Acute Kidney Injury`) and strip subheadings. *EXCEPTION:* If the base term is a generic anatomical or physiological term (e.g., `Kidney`), you MUST keep the relevant pathological subheading (e.g., `Kidney/physiopathology`) to ensure it is not excluded as a generic term.
- Singular and plural keyword forms: use only the root form with `*` appended; count as one term
- No fixed quantity limit — include all terms that pass the criteria above


Produce a `refined_peco` JSON. 
IMPORTANT: Copy the `peco_operators` field EXACTLY as it appears in the original PECO input. Do NOT change it or use the example values.
{
    "population": [
        {"group_id": 1, "standard_mesh_term": ["t1", "t2"], "terms": ["kw1", ...]},
        {"group_id": 2, "standard_mesh_term": ["t3", "t4"], "terms": ["kw2", ...]}
    ],
    "exposure":  [{"group_id": 1, "standard_mesh_term": ["t1"], "terms": ["kw1", ...]}],
    "comparison":    [],
    "outcome":       [{"group_id": 1, "standard_mesh_term": ["t1"], "terms": ["kw1", ...]}],
    "peco_operators": {
        "population": {"within_group": "OR", "between_groups": "AND"},
        "exposure": "OR",
        "comparison": "OR",
        "outcome": "OR"
    }
}

### Step 2: Generate Refined Query
Call `mesh_query_tool` with `peco_input` set to the JSON string of `refined_peco`.
Return the resulting PubMed query string.

## Reply format
{"Query": "<the_refined_pubmed_query>"}
"""


def _query_refine_user_prompt(research_question: str, data_input: str, mesh_term_counts: dict, keyword_counts: dict):
    return (
        f"Research question: {research_question}\n\n"
        f"PECO data:\n{data_input}\n\n"
        f"MeSH term counts (from 100 initial papers):\n"
        f"{json.dumps(mesh_term_counts, indent=2)}\n\n"
        f"Keyword counts (from 100 initial papers):\n"
        f"{json.dumps(keyword_counts, indent=2)}\n\n"
        "Analyse the terms above, select the most relevant ones, and generate a "
        "refined PubMed query."
    )
