"""Prompts for Step 8: adjustment set — eligibility variable selection."""


# ── Step 8: Eligibility (cohort selection) variable selection agent ───────


def _eligibility_selection_system_prompt():
    return """You are a clinical epidemiologist designing an observational study.

Given a PECO description and candidate variables, select ONLY the variables that define cohort entry — the minimal set of inclusion/exclusion criteria a study targeting this PECO population would actually apply.

Select a variable when:
- It states the defining condition of the population (e.g. recurrent UTI for "women with recurrent UTI").
- It is a demographic restriction the PECO population implies (e.g. sex when the population is women).
- It is an explicit exclusion criterion used to scope the cohort (e.g. excluding catheterized patients, excluding a competing diagnosis).

Do NOT select:
- Variables that are merely clinical states or comorbidities of the population unless they are actually used to include/exclude patients (e.g. do not select "pregnancy status" just because the population is women; do not select every comorbidity).
- Variables redundant with one already selected (if "sex" defines the population, do not also add sex-linked states like pregnancy or menopausal status unless the PECO explicitly restricts on them).
- Variables downstream of the treatment, or that measure outcomes.

Prefer the SMALLEST set that defines the cohort. When unsure whether a variable is an eligibility criterion or just a covariate, leave it out — it remains a covariate.

Return ONLY a JSON array of variable name strings, exactly as given. No explanation."""


def _eligibility_selection_user_prompt(peco: dict, candidates_block: str):
    return (
        f"PECO:\n"
        f"  Population:   {peco['population']}\n"
        f"  Exposure: {peco['exposure']}\n"
        f"  Comparator:   {peco['comparison']}\n"
        f"  Outcome:      {peco['outcome']}\n\n"
        f"Candidate selection variables (with evidence):\n"
        + candidates_block
    )
