import json
import os
import re
from pathlib import Path

import networkx as nx
import numpy as np
from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from dotenv import load_dotenv
from tigramite.causal_effects import CausalEffects

from utils import parse_peco_json, extract_peco_values
from prompts.adjustment_process import _eligibility_selection_system_prompt, _eligibility_selection_user_prompt

load_dotenv()

# ---------------------------------------------------------------------------
# LLM model
# ---------------------------------------------------------------------------

def _create_selection_model():
    return ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
        model_type="openai/gpt-5.4",
        url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model_config_dict={"temperature": 0.0, "max_tokens": 2048},
    )



def default_adjustment_output_path(dag_nodes_file: str) -> str:
    return str(
        Path(dag_nodes_file).with_name(
            Path(dag_nodes_file).stem + "_adjustment_metadata_match_input.json"
        )
    )


def _build_graph(edges: list, treatment: str, outcome: str):
    """Return (all_nodes, node_to_idx, tigramite graph matrix, networkx DiGraph)."""
    all_nodes = sorted(
        set(e["from"] for e in edges).union(e["to"] for e in edges) | {treatment, outcome}
    )
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}

    n = len(all_nodes)
    graph_matrix = np.full((n, n), "", dtype="<U3")

    for edge in edges:
        i = node_to_idx[edge["from"]]
        j = node_to_idx[edge["to"]]
        graph_matrix[i, j] = "-->"
        graph_matrix[j, i] = "<--"

    # NetworkX DiGraph for structural queries
    G = nx.DiGraph()
    G.add_nodes_from(all_nodes)
    for edge in edges:
        G.add_edge(edge["from"], edge["to"])

    return all_nodes, node_to_idx, graph_matrix, G


def find_causal_adjustment_set(edges: list, treatment: str, outcome: str,
                               hidden_variables: list = None):
    """
    Optimal minimal adjustment set via tigramite.
    Returns a list of variable names, or None if no valid adjustment set exists
    (e.g. a required confounder is hidden).
    """
    hidden_variables = list(hidden_variables or [])
    all_nodes, node_to_idx, graph_matrix, _ = _build_graph(edges, treatment, outcome)

    X = [(node_to_idx[treatment], 0)]
    Y = [(node_to_idx[outcome], 0)]
    H = [(node_to_idx[h], 0) for h in hidden_variables if h in node_to_idx]

    ce = CausalEffects(
        graph=graph_matrix,
        graph_type="dag",
        X=X,
        Y=Y,
        hidden_variables=H,
    )

    opt_set_indices = ce._get_adjust_set(minimize=True)
    if opt_set_indices is False or opt_set_indices is None:
        return None
    return [all_nodes[i[0]] for i in opt_set_indices]


def select_eligibility_variables(dag_nodes: list, edges: list, peco: dict,
                                 treatment: str, outcome: str,
                                 hidden_variables: list = None) -> list:
    """LLM selects the selection_variable nodes that define cohort entry for the PECO population."""
    hidden_variables = list(hidden_variables or [])
    _, _, _, G = _build_graph(edges, treatment, outcome)

    # -----------------------------------------------------------------------
    # Identify forbidden nodes (descendants of treatment = mediators /
    # post-treatment variables; should never be adjusted for)
    # -----------------------------------------------------------------------

    treatment_descendants = nx.descendants(G, treatment) if treatment in G else set()

    # Colliders: nodes with in-degree >= 2 that lie on paths between
    # treatment/outcome ancestors — conservative proxy: any node that is both
    # a descendant of a non-treatment ancestor AND has 2+ parents.
    # Simpler safe rule: just forbid all descendants of the treatment.
    forbidden = treatment_descendants | {treatment, outcome}

    # -----------------------------------------------------------------------
    # Collect candidate selection_variables (must exist in the DAG)
    # -----------------------------------------------------------------------

    candidate_selection_vars = [
        node["variable"]
        for node in dag_nodes
        if "selection_variable" in node.get("roles", [])
        and node["variable"] not in forbidden
        and node["variable"] not in hidden_variables
    ]

    print(f"\nCandidate selection variables ({len(candidate_selection_vars)}):")
    for v in candidate_selection_vars:
        print(f"  - {v}")

    if not candidate_selection_vars:
        return []

    # -----------------------------------------------------------------------
    # LLM selects population-relevant eligibility variables
    # -----------------------------------------------------------------------

    # Build a lookup from variable name -> (knowledge_card, first source passage)
    _node_evidence = {}
    for node in dag_nodes:
        name = node.get("variable", "")
        kc = (node.get("knowledge_card", "") or "").strip()
        passages = node.get("source_passages", []) or []
        first_passage = ""
        if isinstance(passages, list) and passages:
            first_passage = str(passages[0]).strip()
        elif isinstance(passages, str):
            first_passage = passages.strip()
        _node_evidence[name] = (kc[:300], first_passage[:400])

    def _format_candidate(v: str) -> str:
        kc, passage = _node_evidence.get(v, ("", ""))
        lines = [f"  - {v}"]
        if kc:
            lines.append(f"      knowledge_card: {kc}")
        if passage:
            lines.append(f"      evidence: {passage}")
        return "\n".join(lines)

    agent = ChatAgent(system_message=_eligibility_selection_system_prompt(), model=_create_selection_model())

    prompt = _eligibility_selection_user_prompt(
        peco, "\n".join(_format_candidate(v) for v in candidate_selection_vars)
    )

    response = agent.step(prompt)
    content = response.msgs[0].content.strip()

    # Parse JSON array from LLM response
    match = re.search(r"\[.*?\]", content, re.DOTALL)
    llm_selected: list[str] = []
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                llm_selected = [str(v).strip() for v in parsed if v]
        except json.JSONDecodeError:
            pass

    # Validate: only keep names that are actually in the candidate list
    eligibility_vars = [v for v in llm_selected if v in candidate_selection_vars]

    print(f"\nLLM-selected eligibility/selection variables ({len(eligibility_vars)}):")
    for v in eligibility_vars:
        print(f"  - {v}")
    return eligibility_vars


def build_metadata_match_nodes(adjustment_vars: list, eligibility_vars: list,
                               dag_nodes: list) -> list:
    """Combine adjustment + eligibility variables into metadata_match.py input nodes."""
    eligibility_var_set = set(eligibility_vars)
    combined = list(dict.fromkeys(list(adjustment_vars) + list(eligibility_vars)))  # dedup, preserve order

    dag_node_by_name = {
        str(node.get("variable", "")).strip(): node
        for node in dag_nodes
        if str(node.get("variable", "")).strip()
    }

    metadata_match_nodes = []
    for variable in combined:
        source_node = dag_node_by_name.get(variable)
        if source_node:
            node_out = dict(source_node)
            roles = list(node_out.get("roles", []) or [])
        else:
            # Fallback for graph nodes that are present in edges but absent from
            # the node metadata file. metadata_match.py only requires these fields.
            node_out = {
                "variable": variable,
                "roles": [],
                "knowledge_card": (
                    "Selected by the adjustment-set tool. No original DAG node "
                    "metadata was found for this variable."
                ),
                "children": [],
            }
            roles = []

        if variable in adjustment_vars and "covariate" not in roles:
            roles.append("covariate")
        if variable in eligibility_var_set:
            roles = [role for role in roles if role != "covariate"]
        if variable in eligibility_var_set and "selection_variable" not in roles:
            roles.append("selection_variable")

        node_out["roles"] = roles
        metadata_match_nodes.append(node_out)
    return metadata_match_nodes


def print_adjustment_summary(adjustment_vars: list, eligibility_vars: list) -> None:
    eligibility_var_set = set(eligibility_vars)
    combined = list(dict.fromkeys(list(adjustment_vars) + list(eligibility_vars)))
    print("\n" + "="*60)
    print("FINAL ADJUSTMENT VARIABLES")
    print("="*60)
    print(f"\n  Causal adjustment set  ({len(adjustment_vars)}):")
    for v in adjustment_vars:
        print(f"    {v}")
    print(f"\n  Study eligibility criteria  ({len(eligibility_vars)}):")
    for v in eligibility_vars:
        print(f"    {v}")
    print(f"\n  Combined set  ({len(combined)}):")
    for v in combined:
        marker = "[sel]" if v in eligibility_var_set else "[adj]"
        print(f"    {marker}  {v}")


def save_metadata_match_nodes(nodes: list, out_path: str) -> str:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nodes, f, indent=2, ensure_ascii=False)
    print(f"\nSaved metadata_match input ({len(nodes)} nodes): {out_path}")
    return str(out_path)


def compute_adjustment_set(
    dag_nodes_file: str,
    dag_edges_file: str,
    peco: dict,
    treatment: str,
    outcome: str,
    hidden_variables: list = None,
    out_path: str = None,
) -> dict:
    """
    Compute the optimal minimal adjustment set from a DAG, add LLM-selected
    eligibility variables, and save them in metadata_match.py input format.

    Args:
        peco: flattened PECO values {population, exposure, comparison, outcome}
              (output of utils.extract_peco_values)

    Returns:
        {"adjustment_set", "identifiable", "eligibility_variables", "combined", "output_path"}
    """
    hidden_variables = list(hidden_variables or [])
    out_path = out_path or default_adjustment_output_path(dag_nodes_file)

    with open(dag_edges_file, encoding="utf-8") as f:
        edges = json.load(f)

    with open(dag_nodes_file, encoding="utf-8") as f:
        dag_nodes = json.load(f)

    # Step 1: Optimal minimal adjustment set via tigramite
    adj_names = find_causal_adjustment_set(edges, treatment, outcome, hidden_variables)
    identifiable = adj_names is not None
    adj_names = adj_names or []
    print("Optimal minimal adjustment set:", adj_names if identifiable else "none (no valid adjustment set)")

    # Steps 2-4: LLM selects population-relevant eligibility variables
    eligibility_vars = select_eligibility_variables(
        dag_nodes, edges, peco, treatment, outcome, hidden_variables
    )

    # Step 5: Merge adjustment set with eligibility variables
    eligibility_var_set = set(eligibility_vars)
    adjustment_vars = [v for v in adj_names if v not in eligibility_var_set]
    combined = list(dict.fromkeys(adjustment_vars + eligibility_vars))  # dedup, preserve order
    print_adjustment_summary(adjustment_vars, eligibility_vars)

    # Step 6: Save combined variables in metadata_match.py input format
    nodes = build_metadata_match_nodes(adjustment_vars, eligibility_vars, dag_nodes)
    out_path = save_metadata_match_nodes(nodes, out_path)

    return {
        "adjustment_set": adjustment_vars,
        "identifiable": identifiable,
        "eligibility_variables": eligibility_vars,
        "combined": combined,
        "output_path": out_path,
    }


if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # PECO + study settings
    # -----------------------------------------------------------------------
    peco_elements_json = ""


    peco_elements = parse_peco_json(peco_elements_json)
    peco          = extract_peco_values(peco_elements)   # {population, exposure, comparison, outcome}

    treatment = ""
    outcome   = ""
    hidden_variables = []

    # hidden_variables: list[str] = []   # e.g. ["country", "pregnancy"]

    dag_edges_file = ""
    dag_nodes_file = ""
    metadata_match_nodes_file = str(
        Path(dag_nodes_file).with_name(
            Path(dag_nodes_file).stem + "_adjustment_metadata_match_input.json"
        )
    )

    compute_adjustment_set(
        dag_nodes_file=dag_nodes_file,
        dag_edges_file=dag_edges_file,
        peco=peco,
        treatment=treatment,
        outcome=outcome,
        hidden_variables=hidden_variables,
        out_path=metadata_match_nodes_file,
    )
