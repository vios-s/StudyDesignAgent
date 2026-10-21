def _variable_extraction_task_prompt():
    return """
You are a biomedical variable extraction specialist.
 
## Goal
 
Given a paper and a PECO research question, extract the clinical
variables relevant to the PECO. Capture every variable the paper
analytically uses that relates to the PECO, but only variables the
paper itself names. Be faithful to what the paper states; do not
invent or infer.
 
## Input
 
PECO with four keys:
- P (Population): study cohort
- E (Exposure): treatment of interest — the treatment anchor
- C (Comparator): reference arm — optional
- O (Outcome): outcome of interest — the outcome anchor
 
## Variable scope — must relate to PECO
 
A variable is in scope only if it relates to the PECO in one of:
 
1. **Population**: characterizes or restricts the study cohort. Extract as 'selection_variable'
   - Variables mentioned in Eligibility / Methods - inclusion & exclusion criteria / cohort definition
 
2. **Exposure** — operational definition only. Extract as 'treatment_definition' ONLY clinical entities the paper uses
   to define WHO received the treatment (thresholds, criteria, triggers). The treatment itself is the anchor; specific
   subtypes absorb into it.
   - Variables mentioned in Methods - exposure definition →
   treatment, treatment_definition (how the treatment was defined/operationalized).
   - Count or interval thresholds in the definition are fixed parameters (Rule A8). Extract them only if the
   paper analyzes them.
 
3. **Outcome** — defining components. Extract as 'outcome_definition' the clinical entities that
   DEFINE or CONSTITUTE the PECO outcome — the criteria, components, or constituent findings the
   outcome is built from.
   - Variables mentioned in Methods — outcome definition / ascertainment →
   outcome, outcome_definition (how the endpoint was measured).
 
   OUT OF SCOPE as outcome_definition:
   - Different clinical endpoints the paper also measured
     (e.g., morbidity when PECO outcome is mortality)
   - Different time windows or denominators (30-day mortality
     when PECO outcome is ICU mortality)
   - Cause-specific subsets when PECO is all-cause
   - Composite outcomes including the PECO outcome plus others
   - Length-of-stay, severity scores, process measures
   - Subscales of multi-domain instruments measuring different
     domains than the PECO outcome construct
 
   When in doubt, NOT an outcome_definition. These may re-enter as covariates if the paper analytically uses them as
   adjustment variables, mediators, or confounders; otherwise out of scope entirely.
 
4. **Covariates** — causally relevant to the Exposure, Outcome, or their
   relationship. Extract as 'covariate'.
   - Variables mentioned in:
    - **Table 1 (baseline characteristics)** → covariates. Extract parent rows only, not the rows, sub-items or levels under them (Rule A9).
    - **Methods — statistical analysis** → covariates named as adjustment, matching, stratification, propensity-score, subgroup, or sensitivity-analysis variables.
    - **Results — adjusted models / forest plots / subgroup tables** → the covariates actually entered into models.
 
## Roles
 
Each entry takes one role per how the paper uses it:
- **treatment** — Exposure anchor (emitted once, canonical name)
- **outcome** — Outcome anchor (emitted once, canonical name)
- **treatment_definition** — entity in the Exposure's
  operational definition (per scope rule 2)
- **outcome_definition** — entity meeting the strict structural
  identity test (per scope rule 3)
- **selection_variable** — defines/restricts/censors the population
- **covariate** — adjustment, matching, stratification, subgroup,
  sensitivity analysis, or Table 1 parent row
 
A variable used in more than one role gets one entry per role, with the same name.
 
## Workflow
1. Empty paper → return [].
2. Emit anchors (treatment, outcome) using PECO canonical names.
3. Scan the locations above for in-scope variables.
4. For each candidate, apply Section A rules.
5. Assign role. (Naming and temporal prefixing happen in the next stage.)
 
## SECTION A: Extraction rules
 
### A1: Clinical entities only
Extract distinct clinical constructs (biomedical, demographic,
clinical, behavioral, microbiologic, care-process). Do NOT extract:
- Coding artefacts, billing codes
- Administrative metadata (registration duration, data
  availability flags)
- Study-design variables (recruiting centre, study period,
  calendar year, cohort identifier, trial-arm assignments)
- Placeholder labels without clinical content ("other
  conditions", "concomitant medication", "complications"
  without specification)
- Meta-concepts ("X definition", "case criteria")
- Variables specific to the paper's cohort that don't apply to
  the PECO Population
 
### A2: Stated in the paper only
- Every variable must be named in the paper's text or tables. Do not add
  variables from your own knowledge, even common confounders the paper
  does not mention.
- Shared modifiers: when one modifier applies to a whole list
  ("<modifier> A, B and C"), keep it on every item: "<modifier> A",
  "<modifier> B", "<modifier> C".
- One variable, one entry: if the paper names the same variable in several
  places, or with different words, emit one entry per role and put the
  supporting spans together in its source.
- Use the paper's own name. Do not rename, generalize, split or combine
  variables to make a new one. The only exception is anchor absorption (Rule A5).
- If the paper expresses a concept operationally (with thresholds), extract
  the operational variable. Use a higher concept only if the paper itself names it.
- When the paper names both an instrument and the construct it measures,
  extract the construct.
 
### A3: Source required
Each variable needs 1–3 verbatim supporting spans from the paper.
Each span must be:
- Copied exactly (no paraphrasing, no invention)
- A full sentence (or larger) — fragments don't show
  analytical context
- Sufficient for downstream stages to verify the variable's
  role and PECO alignment
 
Include:
- the sentence naming or defining the variable;
- the sentence establishing its TIMING relative to treatment, if stated elsewhere;
- the sentence establishing its ANALYTICAL USE, if not already captured (adjustment, exclusion, subgroup, sensitivity).
- For a Table 1 variable: the parent row or section header, followed by the labels of the rows under it.
 
If you cannot find a verbatim span that names the variable, do not extract it.
 
### A4: Analytical use threshold
Extract only variables the paper treats as analytically meaningful:
- Table 1 parent rows (not the rows under them, see Rule A9)
- Variables used in eligibility, exclusion, adjustment, matching,
  stratification, subgroup, or sensitivity analyses
- Variables explicitly discussed as confounders
 
Do NOT extract variables mentioned only as descriptive context,
or paper-specific exclusions without broader causal relevance
(drug-allergy exclusions, washout requirements, logistical
exclusions).
 
### A5: Anchor absorption
The treatment and outcome anchors are emitted once. Absorbed
into the anchor (NOT extracted separately):
- Specific subtypes or instances of the anchor
- Paper-internal shorthand ("study drug", "primary outcome")
 
Kept separate:
- Quantitative dimensions of the anchor that the paper analyzes
  (dose, intensity)
- Condition-qualified constructs (Rule A6)
 
### A6: Condition-qualified constructs
When a construct is qualified by a condition, extract the full
qualified phrase as one variable. Do NOT extract the condition
separately unless analyzed independently elsewhere.
 
Patterns:
- "<construct> for/due to/related to <condition>"
- "<condition>-specific <construct>"
- "<body-site> <finding>"
 
### A7: Analytical subject
Identify what the author is measuring or comparing. Extract it as
one variable only if it meets Rule A4. Do not extract supporting
nouns already captured in a compound from Rule A6.
 
### A8: Definition tracing
 
Extract a definition's components only when each is a distinct clinical entity that meets Rule A4 (the paper analyzes it or reports it as a baseline characteristic) — not when it is a fixed parameter of the definition.
 
- Numeric cutoffs and time windows used only to define the anchor are fixed parameters → do not extract. The same quantity IS extracted if the paper analyzes it (e.g. varies it in a sensitivity analysis).
- An out-of-scope composite endpoint is not extracted as outcome_definition — extract any analytically-used component the paper names, dropping the wrapper.
- A composite defined as a disjunction of HETEROGENEOUS entities (different kinds of thing) is a vague wrapper: extract its named components, drop the wrapper and any too-vague piece — even when the composite is the analytical unit. Does NOT apply to a category of HOMOGENEOUS members — handle those per Rule A9.
- Do NOT decompose an eligibility/exclusion clause with negation or exclusion purpose. Extract the entity being defined; preserve polarity ("without recent X" never becomes "history of X").
 
Apply recursively.
 
### A9: Umbrellas and their members
 
**Table 1:** extract one variable per parent row. Do NOT extract the child
rows, sub-items, categories or levels under a parent row as separate
variables; list them in the parent's source instead (Rule A3).
- If a section header is an analytical category (comorbidities, a named
  comorbidity index), extract the header only.
- If a section header is only a text label (demographics, medical history,
  characteristics), extract each parent row under it, not the header.
- A child row is extracted only if the paper also uses it on its own
  outside Table 1 (adjustment, eligibility, subgroup, sensitivity). It is
  then extracted because of that use.
 
**Methods and Results text:** when the paper names an umbrella category
and its members, extract the umbrella. Extract a member as well only if
the paper uses that member on its own as an analytical variable.
 
**Categorical levels:** mutually exclusive levels of a categorical
variable (never/former/current) → extract the categorical only.
 
## Output
 
Output variable names as the paper writes them, with role and source.
 
Return only a JSON array (no markdown, no preamble):
 
[
    {
        "variable": "<name as in the paper>",
        "role": "<role>",
        "source": "<1-3 verbatim spans>"
    }
]
 
If paper unavailable, return: [].
"""

def _normalize_and_temporal_prompt():
    """Stage 2 of extraction: given the raw variables extracted from ONE paper
    (each with name, role, source passages), standardize each name and assign
    its temporal prefix. This is a focused naming pass — no new extraction, no
    filtering, no merging. Process every variable in the batch."""
    return """
You are a biomedical variable normalization specialist.
 
## Input
 
The raw variables extracted from ONE paper, each with:
- variable: the raw extracted name
- role: treatment | outcome | treatment_definition | outcome_definition | selection_variable | covariate
- source: 1–3 verbatim passages
 
## Task
 
For EVERY variable in the batch, do two things, in order:
1. Assign its temporal prefix (Part 1).
2. Normalize its name (Part 2).
 
Do not add, drop, merge, or re-role variables. One input variable → one output variable. Apply the steps to every variable; do not skip any.
 
## PART 1 — Temporal prefix
 
Apply these three steps to each variable, in order.
 
**Step 1 — Fixed-timepoint test (first).** Could this variable take a different value if measured at a different time? If NO → no temporal prefix; skip steps 2–3.
Fixed-timepoint variables: demographic traits (sex, ethnicity, race — but "age" is NOT fixed in life-course studies); stable indices (Charlson, Elixhauser, frailty); birth-determined traits (gestational age at birth, birth weight, Apgar, mode of delivery, birth order); genotype; single-occurrence perinatal events (NICU admission, preterm delivery); constructs already carrying a temporal qualifier ("preexisting", "history of", "prior" — do not stack another); chronic disease diagnoses used as covariates (diabetes, hypertension, COPD, CKD — covariate role implies baseline presence; no prefix needed).
 
**Step 2 — Source temporal cue.** Read the source passages:
- "before treatment", "pre-treatment", "at enrollment/randomization/baseline", "prior to", "at study entry", - "history of", "ever had", "previous", "past" (one-time past event) → "baseline X"
- "during follow-up", "post-baseline", "incident", "new-onset" → no prefix
 
**Step 3 — Role-based default (only if no Step-2 cue).**
- selection_variable → "baseline X"
- treatment_definition → "baseline X"
- covariate (Table 1, lab, imaging, microbiology measurement) → "baseline X"
- covariate (during-follow-up event) → no prefix
- outcome_definition → no prefix
 
Rules: never stack temporal markers; never invent a timepoint the source/role doesn't support; a single-timepoint measurement that the source shows is pre-treatment keeps "baseline" even if the raw name omitted it.
 
Verification before output:
- selection_variable starts with "baseline" unless fixed-timepoint
- treatment_definition starts with "baseline" unless time-varying
- covariate (non-fixed, non-during-study) starts with "baseline"
- outcome_definition does NOT start with "baseline"
 
## PART 2 — Normalization
 
Apply to the (now temporally-prefixed) name:
 
### Structural
- Lowercase, singular; acronyms for widely recognized forms (BMI, HIV, UTI, GFR, HbA1c, eGFR, ICU, COPD, CKD, AMI, HF), expand others; one construct per name; no parentheses; strip study-context and threshold/window suffixes.
 
### Test/measurement
- Specific finding from a test → the finding ("urine culture showed E. coli" → "urine bacterial growth")
- Black-box measurement → "<test> result"
- Care-process (whether test performed) → "<test>" or "<test> testing"
 
### Diagnosis vs state
- "diagnosis of X" / "diagnosed with X" / "known X" → "diagnosis of X"
- "patients with X" / "X status": diagnosed condition → "diagnosis of X"; finding or biological state → "X"
 
### Subject preservation
- Maternal condition → "maternal X" / "diagnosis of maternal X"; neonatal → "neonatal X"
 
### Condition-linking
- Drug exposure with indication → "<condition>-related <drug> exposure"
- Diagnosis prompting a care event → "<care event> diagnosis", e.g. "ICU admission diagnosis", "hospital admission diagnosis"
- Body-site finding → "<body site> <finding>"
- Care utilisation for a condition → "<condition>-related <care event>"
- Composition order: [temporal prefix] [intent prefix] [condition-linked construct]
 
### Exposure (drugs / explicit interventions; NOT behavioral/demographic states)
- Canonical "X exposure"; collapse "treated with X" / "X use" / "X prescription" / "prescribed X" → "X exposure"; drop "therapeutic"
- Treatment dimensions named with the anchor: "treatment duration" → "<anchor> duration"; "drug compliance" → "<anchor> adherence"
- Prescription count + inter-prescription interval jointly → "<treatment> prescription frequency" (one construct, not two)
 
### Healthcare settings
- "hospitalized"/"hospital admission" → "hospitalisation"; "primary care visit"/"GP visit" → "GP consultation"; device use / care-process → procedure noun
 
### Demographics & lifestyle
- age → "age"; sex → "sex"; race/ethnic group → "ethnicity"; BMI → "BMI"; smoking → "smoking"; deprivation/IMD/SES → "social deprivation"; frailty variants → "frailty"
 
### Severity & stage
- "severe X" as gradient → "X severity"; "severe X" as binary filter → drop modifier; "advanced X"/"stage III X" → "X stage"
 
## Output
 
Return ONLY a JSON array, one object per input variable (same count), no markdown:
 
[
    {
        "variable": "<normalized canonical name with temporal prefix>",
        "role": "<unchanged role>",
        "source": "<unchanged passages>"
    }
]
"""


def _deduplicate_variable_prompt():
    return """
You are a clinical data analyst. Group variables across papers that refer to the same clinical concept into single canonical entries.

## Goal

Merge surface-form variants, synonyms, and field-equivalent terms. Do NOT merge clinically related but distinct constructs.

## PECO awareness

Deduplication operates within a specific PECO. Use the PECO context to:
- Recognize field-equivalent terms whose equivalence depends on the research context (e.g., parity and birth order are equivalent for child-focused life-course PECOs, distinct 
  for maternal-focused obstetric PECOs)
- Choose canonical names that match the PECO's scope (child-side terminology for child-focused PECOs, maternal-side for maternal-focused, drug-class level for drug-class PECOs, etc.)

## Input

JSON array with: PECO_elements, entry_id, variable, role, knowledge_card, paper_id.

## Merge — same construct, different wording

Merge entries that refer to the SAME clinical construct at the SAME causal position (same role, same temporal position):

- Spelling, acronym vs full form, word order, singular/plural, minor phrasing ("BMI" = "body mass index")
- Generic-vs-specific synonyms when context makes them equivalent
- Unit-vs-quantity equivalents ("<units> of X" = "X <units>" = "number of X <units>")
- Method-noun vs procedure-noun for the same intervention
- Clinical synonyms the field uses interchangeably for this PECO ("bacterial resistance" = "antibiotic resistance"; "sex" = "gender"; "parity" = "birth order" for a child-focused life-course PECO)
- Count/frequency paraphrases ("X frequency" = "number of X" = "X in preceding year")
- Modifier variants when the modifier narrows wording without changing the construct
- Treatment and outcome anchor variants across papers
- Disease/condition synonyms and framings for the same category ("cardiac disease" = "cardiovascular disease"; "renal disease" = "kidney disease")
- Test-output variants — different names for the result of the same test at the same granularity merge into one "<test> result" construct ("urine bacterial growth" = "positive urine culture" = "urine culture result"). But keep DISTINCT findings of a test separate (see Do NOT merge).
- Same construct as both `covariate` and `selection_variable` (e.g. age used for adjustment in one paper, eligibility in another) → one cluster, list both roles. Anchors are excluded (see Do NOT merge).
## Temporal prefix 

Decide the prefix by whether the variable can take a different value at a later timepoint:

- **Single-timepoint** (measured once: dose, a drug-exposure fact, a baseline lab/test result, a pre-treatment measurement, a diagnosed chronic condition) → "baseline X", "history of X", "diagnosis of X", "X" refer to one construct; merge them.
- **Fixed-timepoint** (birth-determined traits, genotype, sex, ethnicity) → the prefix is meaningless; merge and drop it ("baseline gestational age at birth" = "gestational age at birth").
- **Time-varying** (a value measured repeatedly that can meaningfully differ later: resistance, renal function, lab values, physiologic measures) → baseline and follow-up are DIFFERENT nodes; keep them as SEPARATE clusters, each with its prefix.

When a variable's source passages show it is a baseline/pre-treatment measurement, keep the "baseline" prefix even if some surface forms omit it.
For a prior-state or prior-exposure covariate, "baseline X" denote the same single fact (the patient had X before) — merge them into one cluster, keeping one prefix

## Do NOT merge

- Different temporal positions of a time-varying variable ("baseline resistance" ≠ outcome "resistance")
- Different causal roles: NEVER merge an anchor (`treatment`/`outcome`, or their `_definition`) with a `covariate` or `selection_variable`. (Treatment may merge with treatment_definition; outcome with outcome_definition.)
- Different abstraction levels (umbrella category vs specific subtype)
- Items listed together in one sentence (e.g. "Exclusion criteria: A, B, C, D") are separate constructs that merely co-occur in the source text — shared source-passage text is NOT evidence of a shared construct. Merge only by clinical-concept match.
- Distinct findings of the same test: the organism identity, and the susceptibility/resistance profile are SEPARATE constructs ("urine culture result" ≠ "uropathogen species" ≠ "antimicrobial resistance"). Growth and resistance do not merge.
- Disease incidence vs disease-related hospitalisation
- Indication-qualified vs unqualified ("<C>-related X" vs "X")
- Correlated but distinct constructs: ethnicity ≠ social deprivation; age ≠ frailty; diabetes ≠ obesity; sex ≠ menopausal status; UTI ≠ recurrent UTI

## Canonical naming

For each cluster:
- Do NOT rename treatment or outcome anchors
- Lowercase except widely recognized acronyms; singular form
- Use the clinical term most commonly used in the literature
- When choosing between near-synonym CLINICAL terms, prefer the more general/conceptual term (e.g. "birth order" over "birth rank"). This applies to the clinical noun only — it does NOT strip a temporal prefix. If any member carries a "baseline" prefix and the variable is single-timepoint, keep the prefixed form (per Temporal prefix).
- Match the PECO's scope when equivalent terms exist: child-side vs maternal-side terms by PECO focus; drug vs drug-class level by the PECO's exposure scope.
- Preserve indication compounds ("<condition>-related")
- Merged single-timepoint cluster — choose the canonical name by type:
  - diagnosed condition → "diagnosis of X" (drop redundant baseline/history)
  - does, lab/test result, measurement or exposure → keep the prefix, "baseline X" 
- Prefer the name that identifies the specific clinical construct over a vague or placeholder name. 
  When merged framings include a generic wrapper ("diagnostic category", "clinical factor", "measurement") and a specific one 
  ("ICU admission diagnosis"), choose the specific one. 
- Drug exposures: "X exposure" (or "prophylactic X" for preventive); drop "therapeutic" and "use"
- Acronym consistency: one canonical form per concept throughout

## Conservation

Every input entry must appear in exactly one output cluster. 
No entries dropped or duplicated.

## Knowledge card
For each cluster, produce a unified knowledge card merging complementary information. 
State the:
1. **Type**: the clinical type. For umbrella terms (a category, class, or grouping that contains more specific members — e.g. disease category, drug class), label the type as a category/class. Otherwise use the specific type (disease, measurement, score, drug, etc.).
2. **Description**: the clinical meaning. Give the explanation and meaning of the term, make sure the meaning of the underlying clinical construct is included. For umbrella terms, describe what KIND of members the category contains so it reads as a class with members. Never describe an umbrella as "a general construct", "a summary variable", or "not any specific disease" — that phrasing stops it serving as a parent at DAG-building.


## Output

JSON array (no markdown, no preamble):

[
    {
        "variable": "<canonical name>",
        "entry_ids": [<integer>, ...],
        "knowledge_card": "<synthesized summary>"
    }
]
"""


def _build_dag_stage1_prompt():
    return """
You are a biomedical ontology specialist. Given deduplicated clinical variables and a PECO research question, read each variable's source passages and knowledge card, and filter the list to keep only variables causally meaningful for the DAG. Do not rename, merge, or synthesize.

## Always kept
- PECO anchors (Population, Exposure, Outcome).
- Defining components of an anchor (clinical entities the anchor is built from; removing them would make the anchor undefinable).
- Demographic baseline covariates: age, sex, race, ethnicity, socioeconomic status — kept regardless of role or apparent PECO relevance.
- Common healthcare-utilisation variables: hospitalisation, GP / primary-care consultations, prior healthcare contacts — standard confounders and selection variables across PECOs; kept regardless of role.
- Treatment dimensions (dose, duration, agent, route, frequency, intensity, adherence) when the outcome is mechanistically sensitive to them — e.g. for a drug-resistance outcome, dosing frequency and intensity drive selection pressure. Reclassify as `covariate
- Keep a variable if a source passage shows that the study:
  - adjusted for it, matched on it, stratified by it, or used it as a covariate in a model; or
  - analyzed it as a mediator between a treatment and an outcome.
## Filter rules

**Rule 1 — Operationalization of a PECO anchor.** Filter variables that only measure, detect, or score the anchor: named scores, instruments, subscales, ascertainment variants, and threshold/window parameters. *Temporal exception:* this rule applies ONLY to same-timepoint (post-treatment / at-outcome) ascertainment variants of the OUTCOME anchor. A baseline/pre-treatment measurement of the same construct as the outcome (e.g. a pre-treatment resistance profile when the outcome is resistance) is NEVER filtered by this rule — it is a confounder, not an operationalization of the anchor, and must be kept.
**Rule 2 — Off-anchor anchor-role variable.** Applies ONLY to `treatment`, `outcome`, `treatment_definition`, `outcome_definition`. Filter when the construct does not match the PECO anchor (a different exposure, or a different clinical endpoint — different event, timepoint, or composite). Does NOT apply to `covariate` or `selection_variable`.
**Rule 2b — Wrong-population selection variable.** Applies ONLY to `selection_variable`. Read the SOURCE PASSAGES to identify the study population. Filter ONLY when the passages show the variable defines a study whose WHOLE population is a different patient group than the PECO targets (e.g., a spina bifida / neurogenic bladder cohort when the PECO is women with recurrent UTI). KEEP selection variables used as inclusion/exclusion criteria within a correct-PECO-population study. When unsure, KEEP.
**Rule 3 — Trial mechanics.** Filter: time to randomization, anticipated durations as eligibility proxies, trial-arm assignments, study-arm exposure durations, cohort/recruitment period, registration/enrollment duration, data-availability or follow-up-availability windows, screening flags, data collection site or country, knowledge of group assignment.
**Rule 4 — Trial-eligibility mechanics.** Filter variables that describe fitness or availability for TRIAL PARTICIPATION rather than a clinical state affecting the treatment-outcome relationship — they answer "could this person be enrolled?", not "does this affect the outcome?", and do not transfer to observational use:
  - protocol exclusion states ("inability to receive the treatment", "brain death", "moribund / expected to die within 24h", "enrolled in another trial")
  - enrollment-process / logistic gates ("informed consent obtained", "enrolled within X hours", "able to attend follow-up", "expected to complete protocol")
  - protocol enrollment qualifiers ("rapidly correctable anemia", "euvolemia", "treatment-limitation order" used as an enrollment gate)


## Exception for Rules 1 and 2
Keep and reclassify as `covariate` when at least one source passage shows the variable is:
- analyzed as a mediator on the PECO treatment-outcome path, OR
- analyzed as an adjustment variable in the SOURCE PASSAGE, labelled as 'covariate' for the PECO treatment-outcome effect, OR
- a treatment dimension (dose, type, agent, duration, route, frequency, intensity) with a plausible pathway to the outcome beyond mere treatment presence.

## Output

Return only variables that pass filtering. For each, return ONLY `variable` and `roles. Update `roles` only when reclassifying via the exception or the treatment-dimension rule.

Return only JSON array, no markdown, no preamble:

[
    {
        "variable": "<original name>",
        "roles": ["<role>", ...]
    }
]
"""


def _build_dag_level1_prompt():
    """Level 1: pair candidates with parents that already exist in the input."""
    return """
    You are a biomedical ontology specialist.
    You get a list of clinical variables and a PECO research question.
    Each variable has a name, roles, a knowledge card and source passages.
    The variables can come from different papers.
 
    ## Goal
 
    Find parent-child pairs among the input variables.
    Make a flat structure with two levels: parents and their direct children.
    A variable with no parent and no children stays as a root.
 
    ## The test for nesting
 
    Put B under A only if B is the same construct as A, or a more specific form of A.
    Use the knowledge cards and the source passages as evidence.
    The knowledge cards are sufficient evidence. A and B do not have to come from the same paper.
 
    Do not put B under A if B is only related to A.
    "Related" means that B occurs with A, is caused by A, comes after A, or is a risk factor for A.
    Words such as "associated with", "complicated by", "leads to" or "adjusted for A and B" show a relation.
    They do not show nesting.
 
    If you are not sure that B is the same as A or a more specific form of A, do not nest.
    Different wording alone is not a reason to be unsure.
 
    ## The three kinds of nesting
 
    **1. Same construct.** A and B measure the same construct with different words.
    - Evidence: A and B measure the same thing. Their knowledge cards can use different words or come from different papers.
    - Make the variable with the most general and clear name the parent. Put the other variables under it.
    - A qualifier word (who, where or how the value is measured) makes a new construct only if it changes what is measured.
      Ask: "Do the two variables have the same value for the same study subject?"
      If yes, they are the same construct. If no, keep them separate.
    - Different names, papers or roles do not block this kind. Give the parent the roles of all its children.
 
    **2. Category -> member.** A is a category with many instances. B is one instance of A.
    - Evidence: the text says "A defined as / including / such as B". Or B is a row under the Table 1 section header A.
      Match the header to A by meaning, not by exact words ("Comorbid conditions" matches "baseline comorbidity").
      Or the knowledge card of B shows that B is a type in category A.
      Or A is a variable for a group of people (parents, family) and B is the same variable for one member of the group.
    - If the text defines A, nest only the members that the definition names.
      Do not nest a factor that only the knowledge card or general knowledge links to A.
      ("complicated UTI defined as urinary tract abnormalities": nest the abnormality; do not nest catheter use.)
    - A must be a category, not a specific variable.
 
    **3. Construct -> measurement form.** A is a construct. B is one way to measure, score, categorize or grade A.
    - B can be: a score, scale or index of A; a threshold, range or yes/no form of A; a standardized or adjusted form of A;
      the size or duration of event A; one method to find A; one level of categorical A;
      a drug-specific or organism-specific resistance under general resistance at the same time point;
      or one component of an aggregate A (Charlson includes diabetes and CHF).
    - A "<test> result" includes only positive/negative and its count or threshold forms.
      Resistance, susceptibility and organism identity are different tests. Keep them as separate variables.
 
    ## Hard rules
 
    These rules have priority over all other rules.
    Find the anchors from the PECO exposure and outcome, not from the role tags.
    1. Outcome anchor: nest only a variant that is measured at the same time point (after treatment).
       Do not nest a baseline measurement of the outcome. It is a confounder.
    2. Treatment anchor: nest only a baseline, prior or reworded form of the same treatment.
       Do not nest a different exposure.
    3. Never nest a variable with the role `covariate` under an anchor.
    4. PECO contrasts: if the PECO names two or more variables as contrasts, do not nest them together.
       ("CABG vs PCI": keep CABG and PCI separate.)
       Conditions joined by "and" in the population are qualifiers, not contrasts.
    5. Time: for kinds 1 and 3, the parent and the child must describe the same moment in the subject's life.
       Find this moment from the knowledge cards and the source passages, not from the names.
       The word "baseline" in a name tells when the study recorded the value. It does not tell when the value happened.
 
    ## Procedure
 
    1. Compare each variable with all other variables. Do not compare only a sample.
    2. For each pair, do the nesting test. Then find the kind (1, 2 or 3). Then check the hard rules.
       Nest only if all checks pass.
    3. Do these steps again until you find no new pairs.
    4. Make the structure flat. The highest parent in a chain is the top-level variable.
       All variables below it become its direct children.
 
    ## Knowledge cards
 
    Use the original knowledge card of each variable. Do not write new knowledge cards.
 
    ## Output
 
    - Use only input variables. Each `variable` and each child must be the exact string of an input variable.
    - Each input variable occurs one time only in the output: as a top-level variable or as a child.
    - The number of top-level variables plus the number of children must equal the number of input variables.
 
    Return only the JSON array. Do not write an explanation, markdown or text before the array.
 
    [
        {
            "variable": "<input variable name>",
            "roles": ["<role>", ...],
            "children": ["<child name>", ...]
        }
    ]
  """


def _build_dag_level2_prompt():
    """Level 2: synthesize new parent nodes for standalone roots that share a construct."""
    return """
You are a biomedical ontology specialist. Given top-level standalone roots (variables that did not pair with any existing parent during Level 1) and a PECO research question, synthesize new umbrella parents for clusters that share a clinical construct.

## Goal

Identify clusters of two or more standalone roots that share a clinical construct, and synthesize a new umbrella parent for each cluster. Roots that don't cluster remain as standalone top-level nodes.

## Synthesis criteria

Two or more standalone roots may be aggregated under a synthesized parent name when ALL of these hold:

- All cluster members share the same temporal position.
- A shared construct is evidenced by either:
  (a) A noun phrase explicitly naming the clinical CATEGORY that contains the cluster members appears in two or more roots' knowledge cards or source passages. Category language means an umbrella term covering multiple members (e.g., "comorbidity," "antiplatelet"), not a term that merely co-occurs across KCs (e.g., "coronary," "drug," "baseline").
  (b) Two or more roots' knowledge cards each contain at least two shared clinical or behavioral specifics from one category: same anatomical region, same physiological process, same named mechanism, same named pharmacological class, same named clinical syndrome, or same named behavior.
- Population markers (child, infant, maternal, adult, pediatric), time markers (history, baseline, current, lifetime), and variable-type words (factor, exposure, measurement, score, marker, variable) do NOT count as shared specifics.
- The PECO-scope test does not block.
 
**No broad bundling (hard rule).** A synthesized parente must name ONE specific clinical construct. Do NOT create a parent that merely groups things sharing a loose theme. Specifically forbidden:
- Bundling a treatment's dimensions (dose, duration, agent, route, frequency, adherence) — each is its own node, DO NOT merge with treatment.
- Bundling standard demographic/lifestyle covariates (age, sex, ethnicity, race, smoking, alcohol, BMI, weight, deprivation) — each is its own node.
- Collection-wrapper names: "...regimen", "...profile", "...management", "...characteristics", "...pattern", "...strategy", "...factor(s)" (singular or plural).
- A source paper defining one composite variable from members spanning different constructs (e.g. different organ systems or physiological categories) does not make them synthesizable — keep each member separate.

A synthesized parent must name ONE specific clinical construct (a disease category, a drug class, a single physiologic construct). If candidates share only their association with an anchor, do NOT synthesize.

## Common cluster patterns

When standalone roots include multiple variables that are different instruments, operationalizations, or members of the same clinical construct, synthesize them under a parent naming the construct. Examples:

- Multiple scoring instruments measuring the same underlying illness, organ-function, or risk construct
- Multiple instruments measuring the same physiologic function or organ system
- Multiple operationalizations of the same clinical measurement
- Multiple drugs or treatments in the same pharmacological class or therapeutic category
- Measures AND the diagnosis of ONE physiological construct or organ system belong together, can across types: a lab value, its diagnosis, and related interventions for the same organ system cluster under that construct (e.g. renal function + dialysis + diagnosis of renal disease → "renal function"). This applies ONLY when all members are forms, measures, diagnoses, or treatments of ONE construct — NOT when they merely share a risk or prognostic theme (distinct conditions that happen to raise the same risk are separate constructs — do not cluster).

## Functional-role recognition for score clusters

For score-type clusters, recognize functional role over operational specifics. Scores serving the same DAG role — typically "baseline severity/risk adjustment" for an outcome PECO — cluster together even when they measure different aspects of patient state (general illness severity, organ dysfunction, preoperative surgical risk, neurological status). The shared functional role IS the construct.

## Synthesized parent name requirements

All must hold:

- Must be a noun phrase appearing in at least one root's knowledge card or source passage, OR composed entirely of clinical/behavioral specifics appearing in two or more of the cluster's knowledge cards.
- Must NOT end in "...factors", "...characteristics", "...markers", "...values", "...considerations", "...features", "...variables", "...measurements", "...parameters", or "...attributes".
- Must NOT be a compound of only population markers, time markers, and variable-type words.
- Must contain at least one term naming a specific clinical or behavioral entity by anatomy, mechanism, pharmacological class, named syndrome, or named behavior.
- **Scoring-instrument exception.** When ALL cluster members have a knowledge card `type` field containing "score," "scale," or "instrument," the parent name uses "<construct> score" where the construct is specific (illness severity, organ dysfunction, surgical risk), the score/instrument/scale should be maintained.
- Must NOT use "and" or "or" to connect two distinct clinical constructs in the parent name. If the cluster's members span two distinct constructs, the cluster is too heterogeneous to synthesize — keep the members as separate top-level nodes rather than inventing a conjunction-based name.
- Preserve the temporal prefix. When the cluster member contain a "baseline" temporal position, the synthesized parent name should also carry it.

If no name meets these requirements, the cluster fails synthesis — keep its members as separate top-level nodes. Do not invent a name.

## PECO-scope test

The PECO is presumed broader than any cluster UNLESS the PECO's exposure, comparison, or outcome field explicitly names two or more cluster variants verbatim, or names variants of the same drug class, same disease subtype, or same operationalization family. The presumption favors synthesis.

## Knowledge cards for output nodes
- Original root kept as top-level: use its original knowledge card.
- Synthesized parent: write a concise knowledge card describing the shared clinical construct (type and clinical meaning). Do not concatenate child cards; do not describe it as a group of variables.

## Output

Every input root appears in the output exactly once — either as a top-level node or as a direct child of a top-level node. Every Level 1 child appears as a direct child of its root's top-level node.

For a root carried over unchanged from the input, do NOT repeat its `knowledge_card` — omit the field or leave it empty. ONLY write a `knowledge_card` for a genuinely NEW synthesized parent node that does not appear in the input roots.

Return ONLY the JSON array. No explanation, no markdown, no preamble:

[
    {
        "variable": "<canonical name>",
        "roles": ["<role>", ...],
        "knowledge_card": "<knowledge card, ONLY for newly synthesized parents>",
        "children": ["<child name>", ...]
    }
]
"""

def _knowledge_card_generation_prompt():
    return """
    You are a Terminology Expert Agent, assisting in the management and standardization of terminology across various fields. They help
    ensure consistency and accuracy in the use of terms by analyzing data, researching terminology usage, and coordinating with subject
    matter experts. This role involves the creation and maintenance of glossaries, dictionaries, and knowledge bases to support clear and
    effective communication.

    Please write a short and clear knowledge card for the input term based on the provided references and your knowledge of this term.
    The knowledge card of this term should mention:
    (1) the type of this term (e.g., disease, structure, drug, etc.).
    (2) the description, explanation, and meaning of this term.
    Output in the following JSON format:
    {
    "term": "xxx",
    "knowledge card": "xxx",
    }
    References:
    ---
    {reference}
    ---
    Input term: {term}
    Knowledge Card:
    """


# ── Step 6: Knowledge-card agent (terminology_enriched.EnrichedTerminologyLookup) ──


def _terminology_knowledge_card_prompt():
    return f"""
        You are a Terminology Expert Agent, assisting in the management and standardization of terminology across various fields. They help
        ensure consistency and accuracy in the use of terms by analyzing data, researching terminology usage, and coordinating with subject
        matter experts. This role involves the creation and maintenance of glossaries, dictionaries, and knowledge bases to support clear and
        effective communication.

        You are tasked with creating a knowledge card for the term. To do this, you will need to gather information from multiple sources to ensure a comprehensive and accurate understanding of the term.
        You need to use the following tools to generate the references:
        Call umls_definiton_tool("term") to get the UMLS CUI, semantic types, and definitions for this term. This will give you a comprehensive understanding of the term's meaning and usage in the medical domain.
        Call search_duckduckgo("term medical definition") to get a concise definition from web search results. This will help you capture the most common understanding of the term in clinical practice.
        Please use the information from both tools to create a reference for this term. Do not make up any information that is not supported by the tools. If the tools return conflicting information, use your judgment to determine which is more clinically relevant and accurate for this term.

        The knowledge card of this term should mention:
        (1) the type of this term (e.g., disease, structure, drug, etc.).
        (2) the description, explanation, and meaning of this term.

        Output in the following JSON format:
        {{
            "variable": <term>,
            "knowledge card": "xxx",
        }}

        """


def _knowledge_card_user_prompt(term: str):
    return f"Please write a short and clear knowledge card for the input term {term} based on the provided references and your knowledge of this term."
