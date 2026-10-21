"""Prompts for Step 7: DAG construction (timing, expert team, phase prompts)."""

import textwrap


# ── Step 7: Timing classification (run once before DAG construction) ─────


def _timing_classification_prompt(treatment: str, outcome: str, research_question: str,
                        variables_block: str) -> str:
    return f"""You are a causal-inference methodologist. Classify WHEN each variable happens relative to the start of the treatment. Do not draw edges.

## Research question
{research_question}

## Treatment
{treatment}

## Outcome
{outcome}

## Labels
- pre: happens or is fixed before the treatment starts. Can be a cause of the treatment.
- concurrent: happens around the treatment start and reflects the conditions that lead to the treatment.
- post: happens after the treatment starts, including anything the treatment may change. Never a confounder.
- unclear: timing cannot be judged.

## Rules
1. Judge timing by when the thing the variable describes happens in the subject's life, not by when it was recorded.
2. Words such as "baseline" may refer to the study's survey time, not to a time before treatment. Do not label a variable "pre" because of its name alone.
3. Words such as "history of", "prior" or "before treatment" mean pre. Words such as "after", "during treatment" or "follow-up" mean post.
4. Traits fixed at or before birth are pre.
5. If the context passage states the timing, follow it.

## Output
Return only a JSON object that maps each variable name to one label. No other text.
{{"<variable name>": "pre|concurrent|post|unclear"}}

## Variables
{variables_block}
"""


TIMING_CLASSIFIER_ROLE = "Timing Classifier"
TIMING_CLASSIFIER_PERSONA = "A causal-inference methodologist who classifies variable timing relative to treatment."


# ── Step 7: DAG construction agents ───────────────────────────────────────


def _dag_agent_system_prompt(role_name: str, persona: str):
    return textwrap.dedent(f"""
    You are {role_name}, a researcher in causal inference and healthcare.

    Your persona: {persona}

    You work with a team to build a causal DAG from features extracted from observational studies.
    Follow the task instructions and the output format exactly.
    """).strip()


# Domain-expert team: key -> (role_name, persona)
DAG_TEAM = {
    "research_lead": (
        "Prof. Research Lead",
        "A thoughtful medical and causal AI researcher who excels at synthesizing complex information and identifying key insights",
    ),
    "clinical_expert": (
        "Dr. Clinical Researcher",
        "A clinician with deep expertise in the disease area who understands biological mechanisms, treatment pathways, and real-world patient care. " \
        "They identify clinically meaningful variables and propose plausible causal relationships based on medical knowledge and clinical experience.",
    ),
    "general_practitioner": (
        "Dr. General Practitioner",
        "A practising general practitioner with deep knowledge of primary care and "
        "real-world clinical decision-making. They contribute practical insight into "
        "how patients present, how care is delivered in routine settings, and how "
        "clinical, behavioural, and healthcare-system factors can influence both "
        "exposures and outcomes.",
    ),
    "biostatistics_expert": (
        "Dr. Statistician",
        "A quantitative researcher specializing in causal inference methods and statistical modeling. " \
        "They understand causal concepts, can formalizes domain knowledge into a coherent directed acyclic graph (DAG), "
        "and ensure the graph structure follows causal reasoning principles and that relationships are logically consistent and acyclic",
    ),
}


# ── Step 7: DAG construction task prompt ──────────────────────────────────


def _dag_construction_prompt(treatment: str, outcome: str, features_block: str):
    return f"""
    Your task is to build a causal DAG from the features below.

    ## Timing labels
    Each feature has a timing label relative to the start of the treatment:
    - pre: before the treatment starts.
    - concurrent: around the treatment start; reflects the conditions that lead to the treatment.
    - post: after the treatment starts.
    - unclear: unknown.
    Trust the timing label, not words in the feature name. A name may say "baseline" because of when it was recorded, not when it happened.

    ## Rules
    Apply the rules in this order. If two rules conflict, the earlier rule wins.
    1. Timing. A cause must happen before its effect. Post-treatment features cannot cause the treatment and are never confounders.
       If timing is unclear, draw the edge only if you can say why the cause comes first.
       A cause can be measured at the same time as its effect. Use the knowledge card to find when the cause happened,
       not when it was measured.
    2. Mechanism. Draw A → B only if you can name a specific way A changes B.
       Saying A "is associated with", "influences" or "is a risk factor for" B is not a mechanism.
       The knowledge card, the evidence and domain knowledge can give the mechanism.
    3. Direct edges only. A → B is direct only if A changes B by a mechanism that does not pass through
       any other feature in the list. If the mechanism passes through a feature C, draw A → C (and C → B) instead.
       Keep both A → C and A → B only if A also has a second mechanism that does not pass through C.
       Compare mechanisms, not words. If the reason for A → B describes the same mechanism as the path A → C → B,
       A → B is not direct.
       If A and B only share a common cause, do not connect them.
       The treatment is never C. A path through the treatment does not explain an edge to the outcome.
       This test decides every edge, also every edge to the treatment or the outcome.
    4. Overlapping features. Two features overlap if they measure the same thing for the same subject,
       or if one is only a score, threshold or category of the other.
       In each group of overlapping features, choose the most general one.
       Only the chosen feature gets edges to the treatment and the outcome, also under rules 5 and 9.
       A feature that is one part of a broader feature does not overlap with it.
       Give that part its own edges to the anchors if it has a mechanism that the broader feature does not carry.
       If a feature only measures the treatment or the outcome, draw its causes into that anchor, not into the feature.
    5. Demographics. Always consider age, sex, ethnicity and socioeconomic status as confounders.
       Use one feature for each of them (rule 4). Draw an edge from each one to the treatment and to the outcome.
       Leave out an edge only if timing forbids it, or if you are sure that the whole effect goes through
       other features (not the treatment). This rule has priority over rules 3 and 8.
       The mechanism may come from domain knowledge, but you must name it.
    6. Names. Use exact feature names. Do not rename, merge or invent nodes.
    7. No cycles.
    8. Only rule 5 adds edges to the anchors by default. Rule 3 decides all other edges to the anchors.
    9. Literature evidence. Read the role, the evidence and the knowledge card of each feature.
       If studies of this treatment or outcome adjusted for a feature, make sure it has a path to the treatment
       and a path to the outcome, unless rule 4 puts it in a group with another feature.
       Use rule 3 to decide if each path is a direct edge or goes through other features.
       An adjustment in a study is not a reason for a direct edge.

    ## Roles
    Each feature has a role from past studies. Use it as a starting hint only.
    - treatment_definition / outcome_definition: a way of measuring the treatment or the outcome (see rule 4).
    - covariates: a source study used it in its analysis. It can be a confounder, a cause of only the treatment
      or only the outcome, a mediator, or a cause of other covariates. Choose the position from timing,
      mechanism and the evidence (rule 9).

    ## Features
    - {treatment} | the treatment
    - {outcome} | the outcome
{features_block}
    """


# ── Step 7: Phase prompts (Clinical Researcher → GP → Statistician → Research Lead) ──


_GLOBAL_DAG_RULES = """
Always follow the rules above: exact feature names, no new nodes, direct edges only, no cycles.
"""


def _format_edges(edges: list) -> str:
    return "\n".join(f"{e['from']} → {e['to']} | reason: {e['reason']}" for e in edges) or "(none)"


def _phase1_prompt(research_question: str, prompt: str) -> str:
    return f"""
    Research question: {research_question}

    {prompt}
    {_GLOBAL_DAG_RULES}
    Phase 1 — Clinical Researcher: propose edges
    Build the graph in three steps. Propose an edge only if it passes every rule.
    Step 1. Covariates only. Find the direct causes among the covariates.
            Do not use the treatment or the outcome in this step.
    Step 2. Treatment. For each pre-treatment feature, use rule 3 to decide if it has a direct edge to the treatment.
    Step 3. Outcome. For each feature, use rule 3 to decide if it has a direct edge to the outcome.
            For each feature that the treatment changes, decide if it changes the outcome.
    In Steps 2 and 3, also do rule 5 for the demographic features.

    Output one line per edge and nothing else:
    PROPOSED: A → B | reason: <mechanism>
    """


def _phase2_prompt(research_question: str, prompt: str, current_edges: list,
                   consultation_notes: str = "") -> str:
    notes_block = (
        f"\n    Note from the Research Lead for this round:\n    {consultation_notes}\n"
        if consultation_notes else ""
    )
    return f"""
    Research question: {research_question}

    {prompt}
    {_GLOBAL_DAG_RULES}
    Phase 2 — General Practitioner: review edges

    Current edges:
    {_format_edges(current_edges)}
    {notes_block}
    Step 1. Check every current edge:
    - The reason must name a specific mechanism. If not, remove the edge.
    - If a mediator or common cause already in the graph fully explains the edge, remove it.
      The treatment is never this mediator.
    - The timing labels must allow the edge. If not, remove it.
    - If the direction is wrong, reverse it.
    - If one feature's definition contains the other, remove the edge.

    Step 2. Add only these missing edges:
    - demographic edges required by rule 5;
    - edges from the treatment to post-treatment features it directly changes;
    - edges from those post-treatment features to the outcome, if they change the outcome;
    - edges between covariates that carry an effect to the treatment or the outcome;
    - edges that rule 9 needs to link an adjusted feature to the treatment or the outcome.

    Output one line per edge and nothing else. Every current edge must appear exactly once as KEEP, REMOVE or REVERSE.
    KEEP: A → B | reason: ...
    REMOVE: A → B | reason: ...
    REVERSE: B → A | reason: ...   (write the new direction)
    ADD: A → B | reason: ...
    """


def _phase3_prompt(research_question: str, prompt: str, current_edges: list,
                   cycle_edges: list, shortcut_edges: list) -> str:
    cycle_block = "\n".join(f"- {u} → {v}" for u, v in cycle_edges) or "(none)"
    reason = {(e["from"], e["to"]): e.get("reason", "") for e in current_edges}
    shortcut_lines = []
    for u, v, path in shortcut_edges:
        shortcut_lines.append(f"- {u} → {v} | reason: {reason.get((u, v), '')}")
        shortcut_lines.append(f"  other path: {' → '.join(path)}")
        for a, b in zip(path, path[1:]):
            shortcut_lines.append(f"    {a} → {b} | reason: {reason.get((a, b), '')}")
    shortcut_block = "\n".join(shortcut_lines) or "(none)"
    return f"""
    Research question: {research_question}

    {prompt}
    {_GLOBAL_DAG_RULES}
    Phase 3 — Statistician: final check

    Edges after review:
    {_format_edges(current_edges)}

    Edges that are part of a cycle (found by code):
    {cycle_block}

    Edges that also have another directed path not through the treatment (found by code):
    {shortcut_block}

    Tasks:
    1. Break every cycle by removing the edge with the weakest mechanism.
    2. For each edge with another path, compare its reason with the reasons on the other path.
       If they describe the same mechanism, remove the edge.
       Keep it only if its reason names a second mechanism that does not go through that path.
    3. Remove any other edge that breaks the rules. Leave all other edges unchanged. Do not add edges.

    Output one line for every edge in "Edges after review" and nothing else:
    RETAINED: A → B | reason: ...
    REMOVED: A → B | reason: ...
    """


def _phase4_prompt(research_question: str, retained_edges: list,
                   phase1_result: str = "", phase2_result: str = "", phase3_result: str = "",
                   round_num: int = 1, max_rounds: int = 1) -> str:
    round_note = (
        f"This is review round {round_num} of at most {max_rounds}.\n\n"
        if max_rounds > 1 else ""
    )
    return f"""
    Research question: {research_question}

    Phase 4 — Research Lead: final summary
    {round_note}Below is the full discussion of this round.

    Phase 1 — Clinical Researcher proposed:
    {phase1_result}

    Phase 2 — General Practitioner's review:
    {phase2_result}

    Phase 3 — Statistician's final check:
    {phase3_result}

    The edge list below is fixed. Do not add, remove, rename or reverse any edge.
    Your tasks:
    1. Write a short summary of the causal story this DAG tells. You may reword each edge's reason to be clearer, without changing its claim.
    2. Decide if another review round is needed. Set "needs_further_consultation" to true only if you see a real problem
       that another review could fix: an implausible edge, a missing confounder, or an unresolved disagreement.
       Explain it briefly in "consultation_notes". Otherwise set it to false and leave the notes empty.

    Retained edges:
    {_format_edges(retained_edges)}

    Output JSON only, with exactly these {len(retained_edges)} edges:
    {{
      "synthesis": "short summary",
      "needs_further_consultation": true or false,
      "consultation_notes": "short reason if true, else empty string",
      "edges": [
        {{"from": "A", "to": "B", "reason": "reason"}}
      ]
    }}
    """
