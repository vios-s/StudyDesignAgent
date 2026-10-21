"""Prompts for Step 5: full-text dual-reviewer screening."""

import textwrap


# ── Step 5: Full-text screening criteria generation ──────────────────────


def _fulltext_criteria_generation_prompt(population: str, exposure: str, comparison: str, outcome: str, num_criteria: int = 6):
    return f"""
        ## Task
        Design eligibility criteria for FULL-TEXT screening of research papers.

        Unlike abstract screening (which favours sensitivity), full-text review has access to complete methodological detail and should apply specificity to ensure the study truly aligns with the research question.

        ### Full-Text Screening Principles:
        1. SPECIFICITY OVER SENSITIVITY: Exclude a paper if it genuinely does not meet a core PECO criterion.
        2. VERIFY MEASUREMENT: Verify what the study actually measured and enrolled, not just what the authors mentioned in the background or discussion.
        3. AVOID QUALITY BIAS: Do not exclude studies purely for poor methodological quality (e.g., poor outcome definitions or high risk of bias). Quality assessment happens after inclusion. Focus only on whether the required data is present.
        4. OUTCOME FLEXIBILITY: If multiple outcomes are listed in the PECO, a study only needs to report quantitative or qualitative data for AT LEAST ONE of them to be included.

        ### Generate {num_criteria} binary YES/NO questions (YES = INCLUDE):
        Cover ALL of the following:
        - Population/Problem: Does the enrolled patient cohort strictly match the target demographic AND the specific clinical condition or problem?
        - Exposure: Is the exposure of interest actually administered or evaluated as a primary exposure?
        - Comparator: - Does the study include an appropriate comparison group ((e.g., placebo, standard care) or control condition (if required by the PECO) OR a valid longitudinal self-comparison (e.g., pre/post-treatment) to evaluate the effect of the exposure?
        - Outcome: Is at least one of the specified outcomes actually measured and reported in the results?
        - Study Design: Is the study design eligible (e.g., RCT, cohort, case-control) and NOT an editorial, review, or animal study?
        - Data Availability: Are there extractable results/effect estimates for the PECO elements?

        PECO:
        - P (Population/[Problem]): {population}
        - E (Exposure): {exposure}
        - C (Comparison): {comparison}
        - O (Outcome): {outcome}

        ## Reply Format (strict JSON, no markdown fences):
        {{
            "eligibility_analysis": ["rationale1", "rationale2"],
            "content_criteria": ["criterion1", "criterion2", ...]
        }}
    """


# ── Step 5: Full-text reviewer agents ─────────────────────────────────────


def _fulltext_reviewer_task_prompt(population: str, exposure: str, comparison: str, outcome: str, criteria_text: str, n_criteria: int):
    return textwrap.dedent(f"""
            You are conducting FULL-TEXT screening for a systematic review.
            You have the complete paper – apply specificity, not just sensitivity.

            PECO FRAMEWORK
            - P (Population/Problem): {population}
            - E (Exposure): {exposure}
            - C (Comparison): {comparison}
            - O (Outcome): {outcome}

            SCREENING CRITERIA ({n_criteria} criteria)
            {criteria_text}

            SCREENING PRINCIPLES
            - You have the full paper; use all available information. Mainly check the method, result, and discussion sections.
            - Apply specificity: mark NO if the paper clearly does not satisfy the criterion.
            - Mark YES only when the paper genuinely meets the criterion based on reported methods and results. If you are uncertain whether a criterion is met, answer YES.
            - OUTCOME SCOPE: Any PECO-relevant outcome reported with quantitative data, qualitative results, or specific event frequencies in the results section — whether it was a primary OR secondary outcome — is sufficient.

            RESPONSE FORMAT (JSON only, no markdown fences)
            {{"evaluations": ["YES"|"NO", ...], "rationale": "brief explanation"}}
            Return exactly {n_criteria} evaluations in the array, one per criterion.
        """).strip()


def _fulltext_reviewer_system_prompt(role_name: str, persona: str, task: str):
    return textwrap.dedent(f"""
                You are {role_name}, a researcher conducting full-text screening.

                Your persona: {persona}

                Your task and output format:
                {task}
            """).strip()
