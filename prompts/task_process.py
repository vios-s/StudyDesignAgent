import textwrap

def _task_assessment_prompt():
    return """ Your role is to classify the user's research question into one of three categories: CAUSAL, NOT_APPLICABLE, or UNCLEAR.
    Definitions:
    ## Category Definitions
    ### 1. CAUSAL 
    - Goal: To understand if changing 'X' will result in a change in 'Y'. 
    - Key Indicators: Mentions of "effect," "impact," "effectiveness," or "risk reduction" due to a treatment/exposure. 
    - Hidden Causal Intent: Even if the user uses the word "associated," if they are studying a *modifiable* exposure (like a drug, a policy, or a lifestyle change) to see its impact on a clinical outcome, classify it as CAUSAL.
    - Other causal indicators: effect, impact, causes, reduces, prevents, leads to, counterfactual, treatment effect, RCT, confounding, DAG, propensity score, IV, difference-in-differences.

    ### NOT_APPLICABLE
    Questions that are NOT about a specific exposure → outcome causal relationship:

    1. **Prevalence / descriptive**: "What is the prevalence of X?", "How common is X in population Y?", "What are the characteristics of patients with X?"
    2. **Prediction / prognostic modelling**: "Can X predict Y?", "What factors predict Y?", "Build a risk score/model for Y", "What is the prognosis of X?" — typically involve multiple predictors and a forecasting goal, not a single exposure effect.
    3. **Diagnostic accuracy**: "How accurate is test X for diagnosing Y?", "What is the sensitivity/specificity of X?"
    4. **Correlation / risk factor listing without etiologic framing**: Questions that enumerate multiple risk factors without a specific exposure focus, purely for description.

    ### UNCLEAR
    Use only when the question genuinely cannot be classified after careful reasoning.
    """


_PECO_HUMAN_CRITICAL = """- You must exeucte the human-in-the-loop step 4 after step 1 to step 3 are completed
            - For step 1 to step 3, it should be executed non-iterative and step by step.
            - Do NOT skip any step.
            - ONLY stop when you explicitly call the `human_loop_tool` in Step 4."""

_PECO_AUTO_CRITICAL = """- Human review is DISABLED: execute Step 1 to Step 3 non-iterative and step by step, then output the final JSON.
            - Do NOT skip any of Step 1 to Step 3.
            - Do NOT ask the user for confirmation or wait for any input."""


def _variable_extract_prompt(human_in_the_loop: bool = True):
    prompt = _variable_extract_prompt_template()
    if human_in_the_loop:
        return prompt
    # Autonomous mode: drop the human review steps (4-5) and finish after Step 3.
    prompt = prompt.replace(_PECO_HUMAN_CRITICAL, _PECO_AUTO_CRITICAL)
    start = prompt.index("### Step 4.")
    end = prompt.index("JSON Output Format")
    prompt = prompt[:start] + "AFTER completing Step 3, output the final JSON.\n\n\n            " + prompt[end:]
    return prompt.replace("Output it only after Step 5 is complete.", "Output it only after Step 3 is complete.")


def _variable_extract_prompt_template():
    return """
            Your task is to extract the following PECO elements from the given research aim, following the given workflow end-to-end.

            *** CRITICAL INSTRUCTION ***
            - You must exeucte the human-in-the-loop step 4 after step 1 to step 3 are completed
            - For step 1 to step 3, it should be executed non-iterative and step by step.
            - Do NOT skip any step.
            - ONLY stop when you explicitly call the `human_loop_tool` in Step 4.
  
    
                ### Step 1. Extract raw terms:
                The PECO elements are defined by:
                P (Patient, Problem or Population): {P}
                E (Exposure): {E}
                C (Comparison): {C}
                O (Outcome): {O}.
                First, extract raw terms written in the research question. 

                Act as a highlighter: pull exact, contiguous phrases from the text as-is.
                Do NOT rephrase, clean up, split, or interpret — just quote.

                - "raw population": highlight the exact phrase(s) describing the Patient/Population.
                - "raw exposure": highlight the exact phrase(s) describing the Exposure.
                - "raw comparison": highlight the exact phrase(s) describing the Comparison.
                - "raw outcome": highlight the exact phrase(s) describing each Outcome.
                *** IMPORTANT: Extract from the ENTIRE input — including both the overarching research aim/title AND any structured PECO descriptions if provided.

                ### Step 2. Split and standardize PECO terms:
                Take each raw term from Step 1, decompose it into atomic searchable components, standardize each
                component, and assign group_ids. One raw_term may produce multiple output items.

                **SPLITTING RULES:**
                  1. Separate distinct clinical concepts within a compound phrase:
                    - Separate demographics from clinical conditions:
                      e.g., "older adults with hypertension" → "Aged" + "Hypertension"
                    - Only keep clinically meaningful, independently searchable components: diseases, conditions, treatments, procedures, named demographics
                    - Exclude: generic labels that only gain meaning when paired with other terms — "patients", "therapy", "risks", "benefits", "effects", "outcomes", "complications", "results"
                    - Exclude: statistical/methodological qualifiers (e.g., "relative", "overall", "absolute")
                    - Exclude: numerical thresholds and quantitative descriptors (e.g., "aged 60+", "more than 5 drugs", "500 mg")

                  2. For population setting qualifiers (e.g., "community-dwelling", "outpatient", "ambulatory"):
                    - DROP setting qualifiers that function only as EXCLUSION criteria — they reduce search recall
                      without adding true positive papers, and are better applied at the screening stage.
                      e.g., "community-dwelling older adults" → "Aged" only (drop "community-dwelling")
                    - KEEP setting qualifiers that define a specific clinical context required by the research question:
                      e.g., "hospitalized patients with diabetes" → "Hospitalization" + "Diabetes"

                **TERM IDENTIFICATION** (apply to each atomic component after splitting):
                  For each component, identify up to 10 established, independently searchable PubMed terms that represent it.
                  These will become tiab keywords in the final query — each entry must be directly usable as a standalone search term.

                  Include ALL of the following that apply:
                  - The most precise MeSH-aligned name for the concept (expand abbreviations: "ACE inhibitors" → "Angiotensin-Converting Enzyme Inhibitors")
                  - Common abbreviations and acronyms (e.g., "AKI", "ACEi", "RRT")
                  - Alternative names, historical terms, informal names used in PubMed titles/abstracts
                  - For DRUG CLASS exposures: the class name AND each major specific agent in that class
                  

                  Rules:
                  - Map to clinical entities, NOT measurement techniques: "standing blood pressure" → "Blood Pressure" (NOT "Blood Pressure Determination")
                  - Do NOT map to an unrelated domain — every term must represent the same clinical concept or the parent concept
                  - Do NOT expand a simple demographic term into a complex medical condition
                  - Each specific agent in a drug class will be looked up via `look_up_tool` in Step 3

                  
                **GROUP_ID ASSIGNMENT** (assign here, after splitting):
                  - Same group_id → items are SYNONYMS or ALTERNATIVES (will be OR'd in the query):
                    • Explicit "or" in text: "dementia or cognitive impairment" → both group_id=1
                    • Synonyms: "MI" and "myocardial infarction" → both group_id=1
                    • A condition and the measurement that characterises it:
                      e.g., "orthostatic hypotension" + "postural blood pressure drop" → both group_id=1
                    • Alternative forms in parentheses: "intensive treatment (lower goal or active agent)" → all group_id=1
                  - Different group_ids → items are DISTINCT CO-OCCURRING CHARACTERISTICS:
                    e.g., from "older adults with hypertension": "Aged" (group_id=1), "Hypertension" (group_id=2)

                **PECO_OPERATORS** — determine for each PECO element:

                  OUTCOME: always "OR", regardless of how many distinct outcomes are listed.
                    - "and" between outcomes means "these are the outcomes of interest" — any paper measuring
                      at least one is relevant at the search stage. Using AND would miss papers that only
                      report one outcome in the abstract due to space constraints.

                  POPULATION, EXPOSURE, COMPARISON:
                    - All items share one group_id (synonyms/alternatives of ONE concept) → "OR"
                    - Items span multiple group_ids → use AND between groups ONLY if the research question
                      explicitly requires both characteristics to co-occur simultaneously.
                      Ask: "Would a paper about just one of these characteristics be off-target?"
                      • YES (off-target) → {"within_group": "OR", "between_groups": "AND"}
                        e.g., "elderly with hypertension" — a paper on elderly alone is off-target
                      • NO (still relevant) → "OR"
                        e.g., "exercise and diet" where either exposure alone is still relevant

                Examples:
                  ✓ raw_term "ACE inhibitors or ARBs" → TWO items, same group_id (alternatives):
                    {"raw_term": "ACE inhibitors or ARBs", "terms": ["Angiotensin-Converting Enzyme Inhibitors", "ACE inhibitors", "ACEi", "ramipril", "lisinopril", "enalapril"], "group_id": 1}
                    {"raw_term": "ACE inhibitors or ARBs", "terms": ["Angiotensin II Receptor Blockers", "ARBs", "sartans", "losartan", "valsartan", "candesartan"], "group_id": 1}
                    → peco_operators: "OR"

                  ✓ raw_term "community-dwelling older adults" → ONE item (setting qualifier dropped):
                    {"raw_term": "community-dwelling older adults", "terms": ["Aged", "elderly", "older adults", "older people"], "group_id": 1}
                    → peco_operators: "OR"

                  ✓ raw_term "older adults with hypertension" → TWO items, different group_ids (co-occurring):
                    {"raw_term": "older adults with hypertension", "terms": ["Aged", "elderly", "older adults", "older people"], "group_id": 1}
                    {"raw_term": "older adults with hypertension", "terms": ["Hypertension", "high blood pressure", "arterial hypertension"], "group_id": 2}
                    → peco_operators: {"within_group": "OR", "between_groups": "AND"}

                Generate a list called `standardized_terms`.

                ### Step 3. Search PECO elements in MESH system:
                For each item in `standardized_terms`, call the look_up_tool for EACH entry in its `terms` list
                (both the main standardized term and all synonyms).
                - You CANNOT skip this step.
                - Collect ALL returned MeSH terms across all lookups into a list for `standard_mesh_term`. Remove any that are not directly related to the raw term.
                - Use ONLY the tool response for `standard_mesh_term` and `scope_note` — do not guess or fabricate.
                - If all lookups fail, set `standard_mesh_term` to null and use your internal knowledge to fill in `scope_note`.
                
                ### Step 4. Stop Here and Conduct Human Review Loop
                  Output ALL extracted variables in the EXACT format they will appear in the final output format
                  Call `human_loop_tool` and ask the user to select ONE action with input format:
                  **A) CONFIRM** - Accept all extracted variables as-is
                  **B) MODIFY_VARIABLES** - Change existing variables
                  Input format
                  ```
                  [
                    {
                      "peco": "population|exposure|comparison|outcome",
                      "raw_term": "<current term>",
                      "new_raw_term": "<replacement term>"
                    }
                  ]
                  ```

                  **C) DELETE_VARIABLES** - Remove existing variables
                  Input format
                  ```
                  [
                    {
                      "peco": "population|exposure|comparison|outcome",
                      "raw_term": "<term to delete>"
                    }
                  ]
                  ```

                  **D) ADD_VARIABLES** - Add new variables
                  Input format
                  ```
                  [
                    {
                      "peco": "population|exposure|comparison|outcome",
                      "raw_term": "<new term>"
                    }
                  ]
                  ```

                ### Step 5. Apply the user's action:
                   - CONFIRM: no changes.
                   - DELETE_VARIABLES: remove the entire object(s) for each specified variable.
                   - MODIFY_VARIABLES: replace raw_term with new_raw_term, then repeat steps (2) and (3) for that variable.
                   - ADD_VARIABLES: treat the added raw_term as a new variable, then repeat steps (2) and (3).
                
                AFTER completing this step, output the final JSON.


            JSON Output Format
            {
                "population": [
                    {
                        "raw_term": "string",
                        "terms": ["string"],
                        "standard_mesh_term": ["string"] | null,
                        "scope_note": "string | null",
                        "group_id": "integer"
                    }
                ],
                "exposure": [
                    {
                        "raw_term": "string",
                        "terms": ["string"],
                        "standard_mesh_term": ["string"] | null,
                        "scope_note": "string | null",
                        "group_id": "integer"
                    }
                ],
                "comparison": [
                    {
                        "raw_term": "string",
                        "terms": ["string"],
                        "standard_mesh_term": ["string"] | null,
                        "scope_note": "string | null",
                        "group_id": "integer"
                    }
                ],
                "outcome": [
                    {
                        "raw_term": "string",
                        "terms": ["string"],
                        "standard_mesh_term": ["string"] | null,
                        "scope_note": "string | null",
                        "group_id": "integer"
                    }
                ],
                "peco_operators": {
                    "population": "\"OR\" | {\"within_group\": \"OR\", \"between_groups\": \"AND\"}",
                    "exposure": "\"OR\" | {\"within_group\": \"OR\", \"between_groups\": \"AND\"}",
                    "comparison": "\"OR\" | {\"within_group\": \"OR\", \"between_groups\": \"AND\"}",
                    "outcome": "\"OR\""
                }
            }

            Rules for output:
            1) The JSON object MUST have EXACTLY five top-level keys: "population", "exposure", "comparison", "outcome", "peco_operators".
            2) Each PECO key maps to a list of objects. One raw_term that splits into N atomic concepts produces N objects (each with the same raw_term but different terms and group_id).
            3) Each object MUST contain EXACTLY these 5 keys:
               - "raw_term": the verbatim phrase from Step 1
               - "terms": list of up to 8 established PubMed-searchable terms for this concept (class name + specific agents for drug classes; name + abbreviations + alternatives otherwise)
               - "standard_mesh_term": list of MeSH terms found in Step 3, or null if lookup failed
               - "scope_note": string or null
               - "group_id": integer — same group_id means OR'd (synonyms/alternatives); different group_ids means AND'd (co-occurring concepts)
            4) "peco_operators" MUST be an object with keys: population, exposure, comparison, outcome.
               - "outcome" is ALWAYS "OR".
               - For population, exposure, comparison: use "OR" if all items share one group_id; use {"within_group": "OR", "between_groups": "AND"} only if the research question explicitly requires both characteristics to co-occur simultaneously.
            5) If there are no terms for a PECO element, return an empty list for that key.
            6) Do NOT include any other fields.

            Output only the JSON object above — no explanations, no extra text. Output it only after Step 5 is complete.

            """


# ── Shared system prompt for the study-design agents (Steps 1-2) ──────────


def _design_agent_system_prompt(role_name: str, persona: str, task: str):
    return textwrap.dedent(f"""
            You are {role_name}, a researcher specializing in causal AI and healthcare.

            Your persona: {persona}

            Your task and example output format is: {task}, you must try your best to align with the definitions, workflows,
            and output format/schema described in it.

            """).strip()


# ── Step 1: Task assessment agent ────────────────────────────────────────


TASK_ASSESSOR_ROLE = "Prof. Task Assessor"
TASK_ASSESSOR_PERSONA = ("A causality expert who deeply considers the definitions of causal relationships. You also have the "
                    "knowledge of common healthcare concepts and can bridge causality with healthcare")


def _task_assessment_user_prompt(research_aim: str):
    return f'Determine if the research aim is CAUSAL, ASSOCIATIONAL, or UNCLEAR. Here is the input research aim: {research_aim}'


# ── Step 2: PECO extraction agent ────────────────────────────────────────


PECO_EXTRACTOR_ROLE = "Prof. Variable Extractor"
PECO_EXTRACTOR_PERSONA = ('A clinical terminology expert in clinical research who comprehensively understands the concepts of '
                    'PECO elements in healthcare. You have the comprehensive knowledge of healthcare '
                    'terminologies.')


def _peco_extraction_user_prompt(research_aim: str):
    return f"""Analyze this research question and extract structured PECO information:'
                                           Research Question: "{research_aim}"
                                           """
