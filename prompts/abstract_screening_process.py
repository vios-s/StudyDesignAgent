"""Prompts for Step 4: title/abstract dual-reviewer screening."""

import textwrap


# ── Step 4: Abstract screening criteria generation ───────────────────────


def _abstract_criteria_generation_prompt(num_criteria: int = 6):
    return f"""
        ## Task
        Your task is to design the eligibility criteria specifically for the TITLE AND ABSTRACT screening phase, following these 2 steps:

        ### Step 1: Abstract Screening Principles
        Before generating criteria, acknowledge the following rules for abstract screening:
        1. Abstracts are short and lack methodological detail. 
        2. SENSITIVITY OVER SPECIFICITY: We must not erroneously exclude papers. If an abstract is vague but broadly aligns with the PECO elements, it must pass to the full-text review.
        3. BROADEN CONCEPTS: Generalize specific PECO terms to their broader clinical categories (e.g., generalize a specific drug to "pharmacological interventions").
        4. THE "BENEFIT OF THE DOUBT": If an abstract lacks specific details regarding setting, exact age cutoffs, or specific secondary outcomes, we assume it MIGHT be relevant.

        Write a brief rationale (eligibility_analysis) for how you will broaden the exact PECO elements to be safely screened at the abstract level without losing relevant papers.

        ### Step 2: Generate Abstract-Level Questions
        Develop {num_criteria} binary screening questions to filter studies based on their abstract.
        
        CRITICAL RULES FOR QUESTIONS:
        - Frame the questions so that a "YES" means the paper should be INCLUDED/PASSED to full-text review.
        - Frame the questions so that a "NO" means the paper is EXPLICITLY IRRELEVANT (e.g., entirely wrong population, animal study, etc.).
        - Explicitly state in the question that if the abstract is unclear, the answer should default to "YES".
        
        The criteria should cover:
        - Population/problem relevance (1-2 criteria)
        - Exposure relevance (1-2 criteria)
        - Outcome relevance (1-2 criteria)
        - Study design appropriateness: Frame this question to confirm the paper IS original primary research. (e.g., "Does the abstract describe an original empirical study, and is NOT clearly a review, meta-analysis, editorial, or animal/in-vitro study? (1 criterion)
        
        - If any of the PECO element is Null, DO NOT generate criteria on that PECO element.


        CONTENT_CRITERIA n: ...

        ## Reply Format
        You should reply in a format like:

        {{
            "eligibility_analysis": ["rationale1", "rationale2", ...] \\ the bullet points of your analysis
            "content_criteria": ["criterion1", "criterion2", "..."] \\ the {num_criteria} binary abstract-based criteria
        }}
        """


def _abstract_paper_screening_prompt(population: str, exposure: str, comparison: str, outcome: str,
                                     paper_content: str, criteria_list: list) -> str:
    """TrialMind-style per-paper screening prompt (not used by the dual-reviewer pipeline)."""
    num_criteria = len(criteria_list)

    # Format criteria with numbered list
    criteria_text = "\n".join([
            f"{i+1}. {c['criterion']}"
            for i, c in enumerate(criteria_list)
        ])

    return f"""# CONTEXT #
        You are a clinical specialist tasked with assessing research papers for inclusion in a systematic review based on specific eligibility criteria.

        # OBJECTIVE #
        Evaluate each criterion for the given paper to determine its eligibility. 
        Provide a list of decisions ("YES", "NO") for each of the {num_criteria} eligibility criteria. You must deliver exactly {num_criteria} responses.

        # SCREENING PRINCIPLE #
        This task follows systematic review screening principles.

        When screening abstracts or limited paper content:
        - Prefer **sensitivity over specificity**.
        - Do **not exclude studies prematurely**.
        - If the abstract does not provide enough information to confidently exclude a study, answer **"YES"** rather than "NO".

        # IMPORTANT RULES #

        1. **Use only the information present in the provided paper content.**
        Do not infer or fabricate information.

        2. **If the information is insufficient to confidently evaluate a criterion, choose "YES".**

        3. **Broader Exposure Rule**
        Abstracts may describe exposures using broader categories.
        If the exposure of interest could plausibly be included within a broader category mentioned in the paper (for example, "CNS-active medications", "psychotropic drugs", or "psychoactive medications"), the criterion should be marked **"YES"**, not "NO", unless the paper clearly states the exposure is not included.
        Abstracts frequently omit specific drug dosages or treatment durations. If the correct exposure is mentioned but the specific dosage limits are absent or unclear, do NOT exclude the study. Answer "YES".

        4. **Population Uncertainty Rule**  
        If the study population may include the target population but is not explicitly restricted to it, select **"YES"** rather than "NO".

        5. **Outcome Mention Rule**  
        If the abstract describes the outcome using a synonym, a direct sub-category, OR a broader clinical umbrella term that could plausibly include the target outcome, choose **"YES"**.
        Any PECO outcome mentioned as an objective, measured in the methods, OR reported in the results is sufficient to satisfy the outcome criterion.

        6. **Study Design Rule**  
        Only mark "NO" if the article is clearly a review, systematic review, meta-analysis, editorial, or commentary.


        # PECO FRAMEWORK #
        - P (Patient, Problem or Population): {population}
        - E (Exposure): {exposure}
        - C (Comparison): {comparison}
        - O (Outcome): {outcome}
        

        # PAPER DETAILS #
        Paper Content: {paper_content}

        # EVALUATION CRITERIA #
        Number of Criteria: {num_criteria}
        Criteria for Inclusion:
        {criteria_text}

        # RESPONSE FORMAT #
        You are required to output a JSON object containing:
        1. "evaluations": A list of exactly {num_criteria} decisions (one per criterion, in order)
        2. "rationale": A brief explanation of your overall decision

        Each evaluation must be one of: "YES" or "NO". When in doubt, answer "YES".

        Example output:
        {{
            "evaluations": ["YES", "NO", "YES", ...],
            "rationale": "Brief explanation of screening decision"
        }}

        CRITICAL: The "evaluations" array must contain exactly {num_criteria} elements, no more, no less.
        """


# ── Reviewer persona derivation (shared by abstract and full-text screening)


def _persona_derivation_system_prompt():
    return "You are an expert in systematic review methodology."


def _persona_derivation_user_prompt(research_question: str):
    return (
        f"Research question: {research_question}\n\n"
            "Identify the two most relevant clinical or research domains for this question "
            "(e.g., clinical pharmacology, geriatric medicine, nephrology, epidemiology). "
            "For each domain write a short reviewer persona (2-3 sentences) describing their "
            "background and how they approach abstract screening.\n\n"
            "Reply in JSON only:\n"
            '[{"role_name": "...", "persona": "..."}, {"role_name": "...", "persona": "..."}]'
    )


def _fallback_reviewer_personas():
    return [
            {"role_name": "Clinical Epidemiologist", "persona": "You are a clinical epidemiologist with expertise in systematic review methodology. You favour sensitivity over specificity and pass borderline papers to full-text review."},
            {"role_name": "Clinical Domain Expert", "persona": "You are a clinical specialist in the exposure area. You assess whether a paper could plausibly contribute evidence, giving ambiguous abstracts the benefit of the doubt."},
        ]


# ── Step 4: Abstract reviewer agents ──────────────────────────────────────


def _abstract_reviewer_task_prompt(population: str, exposure: str, comparison: str, outcome: str, criteria_text: str, n_criteria: int):
    return textwrap.dedent(f"""
            Screen research papers for systematic review inclusion using the criteria below.

            PECO FRAMEWORK
            - P (Population): {population}
            - E (Exposure): {exposure}
            - C (Comparison): {comparison}
            - O (Outcome): {outcome}

            SCREENING CRITERIA ({n_criteria} criteria)
            {criteria_text}

            SCREENING PRINCIPLES
            - Prefer sensitivity over specificity at the title/abstract stage.
            - If the abstract is vague but broadly consistent with the PECO, answer YES.
            - Only answer NO when the paper is clearly and explicitly irrelevant.
            - If information is insufficient to decide, answer YES.

            RESPONSE FORMAT (JSON only, no markdown fences)
            {{"evaluations": ["YES"|"NO", ...], "rationale": "brief explanation"}}
            Return exactly {n_criteria} evaluations in the array, one per criterion.
        """).strip()


def _abstract_reviewer_system_prompt(role_name: str, persona: str, task: str):
    return textwrap.dedent(f"""
                You are {role_name}, a researcher specializing in systematic review.

                Your persona: {persona}

                Your task and output format: {task}
            """).strip()
