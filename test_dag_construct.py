import os
import re
import json
import textwrap
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dotenv import load_dotenv
from collections import defaultdict
import asyncio
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

from causalgraphicalmodels import CausalGraphicalModel
from test_browser import generate_knowledge_card
from prompts.dag_process import (
    _timing_classification_prompt,
    TIMING_CLASSIFIER_ROLE,
    TIMING_CLASSIFIER_PERSONA,
    _dag_agent_system_prompt,
    DAG_TEAM,
    _dag_construction_prompt,
    _phase1_prompt,
    _phase2_prompt,
    _phase3_prompt,
    _phase4_prompt,
)


load_dotenv()


# ── Temporal classification helpers ──────────────────────────────────────────

def _extract_json_object(text: str) -> dict:
    """Extract the first complete JSON object from text (handles surrounding prose)."""
    start = text.find("{")
    if start == -1:
        return {}
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return {}
    return {}




async def classify_node_timing(json_path: str, treatment: str, outcome: str,
                               research_question: str, agent_factory,
                               has_only_selection_role_fn) -> dict:
    """One focused LLM call that returns {{variable_name: 'pre'|'concurrent'|'post'|'unclear'}}."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    variables = []
    for entry in data:
        if has_only_selection_role_fn(entry):
            continue
        name = str(entry.get("variable", "")).strip()
        kc = str(entry.get("knowledge_card", "")).strip()
        passages = entry.get("source_passages", [])
        if isinstance(passages, list):
            first_passage = passages[0].strip()[:300] if passages else ""
        else:
            first_passage = str(passages).strip()[:300]
        if name:
            variables.append((name, kc, first_passage))

    variables_block = "\n".join(
        f"- {name}: {kc}" + (f"\n  context: {passage}" if passage else "")
        for name, kc, passage in variables
    )

    prompt = _timing_classification_prompt(treatment, outcome, research_question, variables_block)

    agent = agent_factory()
    response = await asyncio.to_thread(agent.step, prompt)
    text = response.msgs[0].content if response.msgs else "{}"

    timing = _extract_json_object(text)

    timing[treatment] = "treatment"
    timing[outcome] = "outcome"

    for name, _, _ in variables:
        timing.setdefault(name, "pre")

    return timing


def has_only_selection_role(entry: dict) -> bool:
    """Return True when a variable is only a study selection/eligibility node."""
    roles = entry.get("roles", [])
    if isinstance(roles, str):
        roles = [roles]
    roles = {str(role).strip() for role in roles if str(role).strip()}
    return roles == {"selection_variable"}



def create_dag_agent(role_name: str, persona: str, tools=None) -> ChatAgent:
    """Create a Claude agent with specific role and persona"""

    msg_content = _dag_agent_system_prompt(role_name, persona)

    sys_msg = BaseMessage.make_assistant_message(
        role_name=role_name,
        content=msg_content,
    )

    # Configure Claude model
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
        model_type="openai/gpt-5.4",
        url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model_config_dict={"temperature": 0.2, "max_tokens": 500000},
    )

    agent = ChatAgent(
        system_message=sys_msg,
        model=model,
        tools=tools or [],
    )

    return agent

def create_research_team() -> dict:
    """
    Create the domain-expert agents used to construct the DAG phase by phase.
    (No coordinator/task_agent/Workforce — phases are run as direct, code-orchestrated
    agent.step() calls; see _run_dag_construction_phases.)
    """
    return {
        key: create_dag_agent(role_name=role_name, persona=persona)
        for key, (role_name, persona) in DAG_TEAM.items()
    }












_EDGE_LINE_RE = re.compile(
    r'^[\s\-\*\d\.]*(PROPOSED|KEEP|ACCEPTED|ACCEPT|REMOVED|REMOVE|REVERSED|REVERSE|ADDED|ADD|RETAINED)\b[^:]*:'
    r'\s*(.+?)\s*(?:→|->)\s*(.+?)\s*(?:\|\s*reason:\s*(.*))?$',
    re.IGNORECASE,
)


def _clean_name(name: str) -> str:
    return name.strip().strip("*`\"' ").strip()


def _parse_edge_lines(text: str) -> list:
    """Parse 'LABEL: A → B | reason: ...' lines into (label, from, to, reason) tuples."""
    rows = []
    for line in text.splitlines():
        m = _EDGE_LINE_RE.match(line.strip())
        if m:
            label = m.group(1).upper()
            rows.append((label, _clean_name(m.group(2)), _clean_name(m.group(3)),
                         (m.group(4) or "").strip()))
    return rows


def _parse_proposed_edges(text: str) -> list:
    return [{"from": a, "to": b, "reason": r}
            for label, a, b, r in _parse_edge_lines(text) if label == "PROPOSED"]


def _parse_reviewed_edges(text: str, prev_edges: list) -> list:
    """
    Turn Phase 2's KEEP/REMOVE/REVERSE/ADD lines into the live edge list.
    A previous edge the GP did not mention at all is kept (with a warning), so a
    truncated reply does not silently delete edges.
    """
    live, mentioned = {}, set()
    for label, a, b, r in _parse_edge_lines(text):
        mentioned.update({(a, b), (b, a)})
        if label in ("KEEP", "ACCEPT", "ACCEPTED", "REVERSE", "REVERSED", "ADD", "ADDED"):
            live[(a, b)] = {"from": a, "to": b, "reason": r}
    for e in prev_edges:
        if (e["from"], e["to"]) not in mentioned:
            print(f"    ⚠ GP did not mention {e['from']} → {e['to']}; keeping it")
            live.setdefault((e["from"], e["to"]), e)
    return list(live.values())


def _parse_retained_edges(text: str) -> list:
    """Parse Phase 3's 'RETAINED: A → B | reason: ...' lines into edge dicts."""
    return [{"from": a, "to": b, "reason": r}
            for label, a, b, r in _parse_edge_lines(text) if label == "RETAINED"]


def _find_review_candidates(edges: list, treatment: str):
    """
    Find edges for the Statistician to check, in code:
    - cycle_edges: edges inside a cycle
    - shortcut_edges: A → B that also has another directed path A → ... → B
      that does not pass through the treatment (paths through the treatment are confounding, not redundancy)
    """
    G = nx.DiGraph((e["from"], e["to"]) for e in edges)
    cycle_edges = []
    for comp in nx.strongly_connected_components(G):
        if len(comp) > 1:
            cycle_edges += [(u, v) for u, v in G.edges(comp) if v in comp]

    shortcut_edges = []
    for u, v in G.edges():
        if u == treatment:
            continue  # keep treatment → X edges out of this check
        H = G.copy()
        H.remove_edge(u, v)
        if treatment in H and treatment not in (u, v):
            H.remove_node(treatment)
        if nx.has_path(H, u, v):
            shortcut_edges.append((u, v, nx.shortest_path(H, u, v)))
    return cycle_edges, shortcut_edges




async def _run_dag_construction_phases(prompt: str, research_question: str,
                                        treatment: str = "",
                                        run_id: int = None,
                                        semaphore: asyncio.Semaphore = None,
                                        max_rounds: int = 3) -> list:
    """
    Phase 1 (propose) runs once, then Phases 2-4 loop for up to max_rounds.
    Edges are parsed in Python after every phase, so each phase receives a clean edge list.
    Cycle and shortcut candidates for Phase 3 are found in code, not by the LLM.
    """
    team = create_research_team()
    tag = f"[Run {run_id}] " if run_id is not None else ""

    async def step(agent, content):
        if semaphore:
            async with semaphore:
                return await asyncio.to_thread(agent.step, content)
        return await asyncio.to_thread(agent.step, content)

    print(f"\n🔬 {tag}Starting DAG construction...")

    phase1_resp = await step(team["clinical_expert"], _phase1_prompt(research_question, prompt))
    phase1_result = phase1_resp.msgs[0].content if phase1_resp.msgs else ""
    current_edges = _parse_proposed_edges(phase1_result)
    print(f"  {tag}Phase 1 (propose) done: {len(current_edges)} edges")

    edges = []
    prev_pairs = None
    consultation_notes = ""

    for round_num in range(1, max_rounds + 1):
        rtag = f"{tag}[round {round_num}/{max_rounds}] "

        phase2_resp = await step(
            team["general_practitioner"],
            _phase2_prompt(research_question, prompt, current_edges, consultation_notes),
        )
        phase2_result = phase2_resp.msgs[0].content if phase2_resp.msgs else ""
        reviewed_edges = _parse_reviewed_edges(phase2_result, current_edges)
        print(f"  {rtag}Phase 2 (review) done: {len(reviewed_edges)} edges")

        cycle_edges, shortcut_edges = _find_review_candidates(reviewed_edges, treatment)
        phase3_resp = await step(
            team["biostatistics_expert"],
            _phase3_prompt(research_question, prompt, reviewed_edges, cycle_edges, shortcut_edges),
        )
        phase3_result = phase3_resp.msgs[0].content if phase3_resp.msgs else ""
        round_edges = _parse_retained_edges(phase3_result)
        print(f"  {rtag}Phase 3 (final check) done: {len(round_edges)} edges retained "
              f"({len(cycle_edges)} cycle / {len(shortcut_edges)} shortcut candidates)")

        if not round_edges:
            edges = round_edges
            break

        phase4_resp = await step(
            team["research_lead"],
            _phase4_prompt(research_question, round_edges,
                           phase1_result=phase1_result, phase2_result=phase2_result, phase3_result=phase3_result,
                           round_num=round_num, max_rounds=max_rounds),
        )
        phase4_text = phase4_resp.msgs[0].content if phase4_resp.msgs else ""
        synthesized = _extract_json_object(phase4_text)
        synthesis = synthesized.get("synthesis", "")
        llm_edges = synthesized.get("edges", [])
        needs_consultation = bool(synthesized.get("needs_further_consultation", False))
        consultation_notes = str(synthesized.get("consultation_notes", "")).strip()

        retained_pairs = {(e["from"], e["to"]) for e in round_edges}
        llm_pairs = {(e.get("from"), e.get("to")) for e in llm_edges}
        if llm_edges and llm_pairs == retained_pairs:
            edges = llm_edges  # same structure, keep the Research Lead's polished reasons
        else:
            if llm_edges:
                print(f"  {rtag}⚠ Research Lead's edge set didn't match the retained set exactly — "
                      f"keeping the Python-parsed retained edges as-is")
            edges = round_edges

        if synthesis:
            print(f"  {rtag}Synthesis: {synthesis}")

        converged_by_stability = prev_pairs is not None and retained_pairs == prev_pairs
        if not needs_consultation:
            print(f"  {rtag}Research Lead: converged, no further consultation needed")
            break
        if converged_by_stability:
            print(f"  {rtag}Edge set unchanged from the previous round despite a consultation "
                  f"request — stopping")
            break
        if round_num == max_rounds:
            print(f"  {rtag}Reached max consultation rounds — stopping")
            break

        print(f"  {rtag}Research Lead requested further consultation: "
              f"{consultation_notes or '(no notes given)'}")
        prev_pairs = retained_pairs
        current_edges = edges

    return edges


_USE_RE = re.compile(
    r"adjust|confound|covariat|control(?:led|ling)? for|stratif|match(?:ed|ing)|mediat|subgroup|sensitivity",
    re.IGNORECASE,
)


def _split_passages(source_passages) -> list:
    if isinstance(source_passages, str):
        return [source_passages] if source_passages.strip() else []
    if isinstance(source_passages, list):
        return [str(p) for p in source_passages if p and str(p).strip()]
    return []


def _evidence_sentences(source_passages, n: int = 3) -> str:
    """
    Pick up to n evidence sentences. Sentences that show analytical use (adjusted for,
    confounder, stratified, mediator, ...) come first, then the remaining sentences in order.
    """
    text = " ".join(p.strip() for p in _split_passages(source_passages))
    if not text:
        return ""
    sentences, seen = [], set()
    for s in re.split(r'(?<=[.!?])\s+', text):
        s = s.strip()
        if s and s not in seen:
            seen.add(s)
            sentences.append(s)
    use = [s for s in sentences if _USE_RE.search(s)]
    rest = [s for s in sentences if not _USE_RE.search(s)]
    return " ".join((use[:2] + rest + use[2:])[:n])


def _adjusted_count(source_passages) -> int:
    """Number of source passages that show the variable was adjusted for / used analytically."""
    return sum(1 for p in _split_passages(source_passages) if _USE_RE.search(p))


def _card_description(card) -> str:
    """Return the knowledge card's 'description' field (card may be a dict or a JSON string)."""
    if isinstance(card, str):
        try:
            card = json.loads(card)
        except (json.JSONDecodeError, TypeError):
            return card.strip()
    if isinstance(card, dict):
        return str(card.get("description", "")).strip()
    return str(card or "").strip()


def _top_sentences(source_passages, n: int = 3) -> str:
    """Kept for backward compatibility; now uses evidence-first selection."""
    return _evidence_sentences(source_passages, n)


def build_dag_prompt(json_path: str, treatment: str, outcome: str,
                     timing: dict = None) -> str:
    """
    Read the extracted-variables JSON and build the DAG construction prompt.
    timing: optional dict mapping variable name → 'pre'|'concurrent'|'post'|'treatment'|'outcome'|'unclear'
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    feature_lines = []
    for entry in data:
        if has_only_selection_role(entry):
            continue
        variable = str(entry.get("variable", "")).strip()
        knowledge_card = str(entry.get("knowledge_card", "")).strip()
        roles = entry.get("roles", [])
        roles_str = ", ".join(roles) if isinstance(roles, list) else str(roles)
        timing_label = timing.get(variable, "unclear") if timing else "unclear"
        passages = entry.get("source_passages", [])
        evidence = _evidence_sentences(passages, n=3)
        n_used = _adjusted_count(passages)
        definition = _card_description(entry.get("knowledge_card", ""))
        line = f"- {variable} | definition: {definition} | timing: {timing_label} | role: {roles_str}"
        if n_used:
            line += f" | adjusted for in {n_used} source passage(s)"
        if evidence:
            line += f" | evidence: {evidence}"
        feature_lines.append(line)

    features_block = "\n".join(feature_lines)

    dag_prompt = _dag_construction_prompt(treatment, outcome, features_block)
    return dag_prompt


def plot_dag(edges: list, treatment: str, outcome: str, out_path: str = "dag.png"):
    """
    Plot a causal DAG from a list of edges and save to a PNG file.
    Uses causalgraphicalmodels for visualization.

    Args:
        edges: list of {"from": ..., "to": ..., "reason": ...} dicts
        treatment: treatment variable name
        outcome: outcome variable name
        out_path: output file path
    """
    # Collect all nodes
    all_nodes = set()
    for e in edges:
        all_nodes.add(e["from"])
        all_nodes.add(e["to"])
    all_nodes.add(treatment)
    all_nodes.add(outcome)

    # Convert edges to tuple format
    edge_tuples = [(e["from"], e["to"]) for e in edges]

    # Create causal graphical model
    cgm = CausalGraphicalModel(
        nodes=list(all_nodes),
        edges=edge_tuples
    )

    # Draw and save
    dot = cgm.draw()
    output_base = out_path.rsplit('.', 1)[0] if '.' in out_path else out_path
    output_format = out_path.split('.')[-1] if '.' in out_path else 'png'
    dot.format = output_format
    dot.render(output_base, cleanup=True)
    print(f"DAG plot saved to: {out_path}")


def plot_dag_matplotlib_simple(edges: list, treatment: str, outcome: str, out_path: str = "dag.png"):
    """
    Fallback: Plot a simple causal DAG using matplotlib (no edge scores).
    """
    G = nx.DiGraph()
    for e in edges:
        G.add_edge(e["from"], e["to"])

    G.add_node(treatment)
    G.add_node(outcome)

    # Topological-level layout
    try:
        generations = {n: 0 for n in G.nodes()}
        for node in nx.topological_sort(G):
            for successor in G.successors(node):
                generations[successor] = max(generations[successor], generations[node] + 1)
    except nx.NetworkXUnfeasible:
        print("  Warning: Cycle detected; using spring layout.")
        pos = nx.spring_layout(G, seed=42, k=3)
        generations = None

    if generations is not None:
        levels: dict[int, list] = {}
        for node, lvl in generations.items():
            levels.setdefault(lvl, []).append(node)

        pos = {}
        max_level = max(levels.keys()) if levels else 0
        for lvl, nodes in levels.items():
            y = 1.0 - lvl / max(max_level, 1)
            n_nodes_level = len(nodes)
            for i, node in enumerate(sorted(nodes)):
                x = (i - (n_nodes_level - 1) / 2) * 0.25
                pos[node] = (x, y)

    # Node colours: orange = exposure, red = outcome, light-blue = adjustment
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        if node == treatment:
            node_colors.append("#F4A261")
            node_sizes.append(3500)
        elif node == outcome:
            node_colors.append("#F08080")
            node_sizes.append(3500)
        else:
            node_colors.append("#AEC6CF")
            node_sizes.append(2800)

    n_nodes = G.number_of_nodes()
    n_levels = len(set(generations.values())) if generations else 3
    max_nodes_per_level = max(len(v) for v in levels.values()) if generations else n_nodes

    fig_w = max(14, max_nodes_per_level * 3.5)
    fig_h = max(10, n_levels * 3)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_axis_off()

    labels = {n: "\n".join(textwrap.wrap(n, width=15)) for n in G.nodes()}

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, alpha=0.95,
                           edgecolors='#555555', linewidths=1.5)
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax,
                            font_size=9, font_weight='bold')
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True,
                           arrowstyle="-|>", arrowsize=15,
                           edge_color="#555555", width=1.5,
                           connectionstyle="arc3,rad=0.08",
                           min_source_margin=30, min_target_margin=30)

    legend_handles = [
        mpatches.Patch(color="#F4A261", label=f"Exposure ({treatment})"),
        mpatches.Patch(color="#F08080", label=f"Outcome ({outcome})"),
        mpatches.Patch(color="#AEC6CF", label="Adjustment variable"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=10, framealpha=0.9)
    ax.set_title(f"Causal DAG: {treatment} → {outcome}", fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor='white')
    plt.close(fig)
    print(f"DAG plot (matplotlib) saved to: {out_path}")


def save_dag_graph(edges: list, base_path: str):
    """
    Save the DAG as both an adjacency matrix (CSV) and an adjacency list (JSON).

    adjacency_matrix CSV:  nodes as both row and column headers; cell = 1 if edge exists.
    adjacency_list JSON:   {node: [list of successor nodes]} for every node.

    Args:
        edges:     list of {"from": ..., "to": ..., "reason": ...} dicts
        base_path: path prefix; files are written to <base_path>_adj_matrix.csv
                   and <base_path>_adj_list.json
    """
    G = nx.DiGraph()
    for e in edges:
        G.add_edge(e["from"], e["to"])

    nodes = sorted(G.nodes())

    # ── Adjacency matrix ─────────────────────────────────────────────────────
    matrix = pd.DataFrame(0, index=nodes, columns=nodes)
    for src, dst in G.edges():
        matrix.loc[src, dst] = 1

    matrix_path = base_path + "_adj_matrix.csv"
    matrix.to_csv(matrix_path)
    print(f"Adjacency matrix saved to: {matrix_path}  ({len(nodes)}×{len(nodes)})")

    # ── Adjacency list ────────────────────────────────────────────────────────
    adj_list = {node: sorted(G.successors(node)) for node in nodes}
    list_path = base_path + "_adj_list.json"
    with open(list_path, "w") as f:
        json.dump(adj_list, f, indent=2)
    print(f"Adjacency list  saved to: {list_path}")

    return matrix, adj_list


def plot_dag_with_scores(edges_with_scores: list, treatment: str, outcome: str, out_path: str = "dag.png"):
    """
    Plot a causal DAG with edge scores displayed on each edge.
    Uses causalgraphicalmodels for visualization.

    Args:
        edges_with_scores: list of {"from": ..., "to": ..., "score": ..., "reason": ...} dicts
        treatment: treatment variable name
        outcome: outcome variable name
        out_path: output file path (supports .png, .pdf, .svg)
    """
    # Collect all nodes
    all_nodes = set()
    for e in edges_with_scores:
        all_nodes.add(e["from"])
        all_nodes.add(e["to"])
    all_nodes.add(treatment)
    all_nodes.add(outcome)

    # Convert edges to tuple format
    edge_tuples = [(e["from"], e["to"]) for e in edges_with_scores]

    # Create causal graphical model
    cgm = CausalGraphicalModel(
        nodes=list(all_nodes),
        edges=edge_tuples
    )

    # Draw and customize with edge scores
    dot = cgm.draw()

    # Improve layout clarity: more spacing between ranks and siblings,
    # curved edges to reduce visual clutter, outcome pinned to bottom rank
    dot.graph_attr.update(ranksep='1.0', nodesep='0.5')
    dot.node_attr.update(fontsize='12')
    with dot.subgraph() as s:
        s.attr(rank='sink')
        s.node(outcome)

    # Colour nodes: orange = exposure, red = outcome, light-blue = adjustment
    # Wrap long labels onto multiple lines for readability
    for node in all_nodes:
        label = '\n'.join(textwrap.wrap(node, width=22))
        if node == treatment:
            dot.node(node, label=label, style='filled', fillcolor='#F4A261', fontcolor='black')
        elif node == outcome:
            dot.node(node, label=label, style='filled', fillcolor='#F08080', fontcolor='black')
        else:
            dot.node(node, label=label, style='filled', fillcolor='#AEC6CF', fontcolor='black')

    # Remove default edges and re-add with score labels
    dot.body = [line for line in dot.body if '->' not in line]
    for e in edges_with_scores:
        score = e.get("score", 1.0)
        dot.edge(e["from"], e["to"], label=f'{score:.2f}', fontcolor='red', fontsize='9')

    # Save
    output_base = out_path.rsplit('.', 1)[0] if '.' in out_path else out_path
    output_format = out_path.split('.')[-1] if '.' in out_path else 'png'
    dot.format = output_format
    dot.render(output_base, cleanup=True)
    print(f"DAG plot with scores saved to: {out_path}")


def plot_dag_matplotlib(edges_with_scores: list, treatment: str, outcome: str, out_path: str = "dag.png"):
    """
    Fallback: Plot a causal DAG using matplotlib/networkx.
    Improved version with better spacing and readability.
    """
    G = nx.DiGraph()
    edge_scores = {}
    for e in edges_with_scores:
        G.add_edge(e["from"], e["to"])
        edge_scores[(e["from"], e["to"])] = e.get("score", 1.0)

    G.add_node(treatment)
    G.add_node(outcome)

    # Topological-level layout with improved spacing
    try:
        generations = {n: 0 for n in G.nodes()}
        for node in nx.topological_sort(G):
            for successor in G.successors(node):
                generations[successor] = max(generations[successor], generations[node] + 1)
    except nx.NetworkXUnfeasible:
        print("  Warning: Cycle detected; using spring layout instead.")
        pos = nx.spring_layout(G, seed=42, k=3, iterations=100)
        generations = None

    if generations is not None:
        levels: dict[int, list] = {}
        for node, lvl in generations.items():
            levels.setdefault(lvl, []).append(node)

        pos = {}
        max_level = max(levels.keys()) if levels else 0

        for lvl, nodes in levels.items():
            y = 1.0 - lvl / max(max_level, 1)
            # Improved horizontal spacing
            n_nodes_level = len(nodes)
            for i, node in enumerate(sorted(nodes)):
                # Center nodes and add more spacing
                x = (i - (n_nodes_level - 1) / 2) * 0.25
                pos[node] = (x, y)

    # Node colours and sizes
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        if node == treatment:
            node_colors.append("#4C72B0")
            node_sizes.append(3500)
        elif node == outcome:
            node_colors.append("#C44E52")
            node_sizes.append(3500)
        else:
            node_colors.append("#E8E8E8")
            node_sizes.append(2800)

    # Calculate figure size based on graph structure
    n_nodes = G.number_of_nodes()
    n_levels = len(set(generations.values())) if generations else 3
    max_nodes_per_level = max(len(v) for v in levels.values()) if generations else n_nodes

    fig_w = max(14, max_nodes_per_level * 3.5)
    fig_h = max(10, n_levels * 3)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_axis_off()

    # Wrap long node labels
    labels = {n: "\n".join(textwrap.wrap(n, width=15)) for n in G.nodes()}

    # Draw nodes with border
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, alpha=0.95,
                           edgecolors='#555555', linewidths=1.5)

    # Draw labels
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax,
                            font_size=9, font_weight='bold', font_family='sans-serif')

    # Draw edges with varying thickness based on score
    for e in edges_with_scores:
        score = e.get("score", 1.0)
        width = 1.0 + score * 2.5
        alpha = 0.5 + score * 0.5
        nx.draw_networkx_edges(G, pos, ax=ax,
                               edgelist=[(e["from"], e["to"])],
                               arrows=True, arrowstyle="-|>", arrowsize=15,
                               edge_color="#555555", width=width, alpha=alpha,
                               connectionstyle="arc3,rad=0.08",
                               min_source_margin=30, min_target_margin=30)

    # Draw edge labels with scores
    edge_labels = {(e["from"], e["to"]): f'{e.get("score", 1.0):.2f}' for e in edges_with_scores}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax,
                                  font_size=8, font_color="#C44E52",
                                  font_weight="bold", bbox=dict(boxstyle='round,pad=0.2',
                                                                  facecolor='white',
                                                                  edgecolor='none',
                                                                  alpha=0.8))

    # Legend
    legend_handles = [
        mpatches.Patch(color="#F4A261", label=f"Exposure ({treatment})"),
        mpatches.Patch(color="#F08080", label=f"Outcome ({outcome})"),
        mpatches.Patch(color="#AEC6CF", label="Adjustment variable"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=10,
              framealpha=0.9, edgecolor='#CCCCCC')

    ax.set_title(f"Ensemble Causal DAG: {treatment} → {outcome}\n(Edge scores = proportion of runs agreeing)",
                 fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor='white')
    plt.close(fig)
    print(f"DAG plot (matplotlib) saved to: {out_path}")


async def construct_dag_single_run_async(json_path: str, treatment: str, outcome: str,
                                          research_question: str, run_id: int,
                                          semaphore: asyncio.Semaphore = None,
                                          timing: dict = None) -> list:
    """
    Async single DAG construction run. Returns edges list only (no saving).
    Runs Phases 1-3 as direct agent calls (see _run_dag_construction_phases) instead of a
    Workforce task, so there is no autonomous decomposition or quality-check/replan loop.
    """
    prompt = build_dag_prompt(json_path, treatment, outcome, timing=timing)
    return await _run_dag_construction_phases(prompt, research_question, treatment=treatment, run_id=run_id, semaphore=semaphore)


async def construct_dag_ensemble_async(json_path: str, treatment: str, outcome: str,
                                        research_question: str,
                                        num_runs: int = 10, score_threshold: float = 0.5):
    """
    Run DAG construction multiple times in parallel using asyncio and aggregate results.

    Uses 3-class classification for each variable pair:
      - Class 1: A → B (forward edge)
      - Class 2: B → A (reverse edge)
      - Class 3: No relationship

    Args:
        num_runs: Number of parallel runs (default 10)
        score_threshold: Minimum score to select an edge direction (default 0.5)
                        If max class score < threshold, no edge is created.

    Returns:
        Final edges with scores
    """
    print(f"=" * 70)
    print(f"🚀 ENSEMBLE DAG CONSTRUCTION (3-class classification)")
    print(f"   Runs: {num_runs} | Score threshold: {score_threshold}")
    print(f"   Treatment: {treatment} | Outcome: {outcome}")
    print(f"=" * 70)

    # Collect all nodes from input data
    with open(json_path, 'r', encoding='utf-8') as f:
        _data = json.load(f)
    all_nodes = {
        entry["variable"]
        for entry in _data
        if entry.get("variable") and not has_only_selection_role(entry)
    }
    all_nodes.add(treatment)
    all_nodes.add(outcome)
    all_nodes = sorted(all_nodes)

    # For 3-class classification: track votes for each direction
    # edge_votes[(A, B)] = {"forward": count, "reverse": count, "none": count}
    # forward = A→B, reverse = B→A, none = no edge between A and B
    edge_votes = defaultdict(lambda: {"forward": 0, "reverse": 0, "none": 0})
    edge_reasons = defaultdict(list)  # Store reasons for each direction

    # Classify node timing once — reused across all parallel runs
    print("\n⏱️  Classifying node timing relative to treatment...")

    def _timing_agent_factory():
        return create_dag_agent(role_name=TIMING_CLASSIFIER_ROLE, persona=TIMING_CLASSIFIER_PERSONA)

    timing = await classify_node_timing(
        json_path, treatment, outcome, research_question,
        agent_factory=_timing_agent_factory,
        has_only_selection_role_fn=has_only_selection_role,
    )
    print(f"   Timing classified for {len(timing)} variables: "
          + ", ".join(f"{v}={t}" for v, t in list(timing.items())[:5]) + " ...")

    print(f"\n🚀 Launching {num_runs} parallel DAG construction runs using asyncio...")

    # Semaphore limits concurrent runs to avoid API rate limits
    semaphore = asyncio.Semaphore(3)

    # Create all async tasks
    tasks = [
        construct_dag_single_run_async(
            json_path, treatment, outcome,
            research_question, run_id, semaphore, timing=timing
        )
        for run_id in range(1, num_runs + 1)
    ]

    # Run all tasks concurrently and collect results
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    all_edges = []
    successful_runs = 0

    for run_id, result in enumerate(results, start=1):
        if isinstance(result, Exception):
            print(f"  [Run {run_id}] ❌ Failed with exception: {result}")
        elif result:
            edges = result
            all_edges.append(edges)
            successful_runs += 1

            # Track which edges exist in this run
            run_edges = set()
            for e in edges:
                src, dst = e["from"], e["to"]
                if src in all_nodes and dst in all_nodes:
                    run_edges.add((src, dst))
                    edge_reasons[(src, dst)].append(e.get("reason", ""))

            # Update votes for all node pairs
            for i, node_a in enumerate(all_nodes):
                for node_b in all_nodes[i+1:]:  # Only check unique pairs (A, B) where A < B
                    if (node_a, node_b) in run_edges:
                        edge_votes[(node_a, node_b)]["forward"] += 1
                    elif (node_b, node_a) in run_edges:
                        edge_votes[(node_a, node_b)]["reverse"] += 1
                    else:
                        edge_votes[(node_a, node_b)]["none"] += 1

    if successful_runs == 0:
        print("❌ No successful runs. Cannot aggregate.")
        return None

    print(f"\n{'=' * 70}")
    print(f"📊 AGGREGATION RESULTS ({successful_runs}/{num_runs} successful runs)")
    print(f"   Classification approach: A→B vs B→A vs No Edge")
    print(f"   Threshold: max_score >= {score_threshold}")
    print(f"{'=' * 70}")

    # Classify each edge pair and select winner
    final_edges = []
    for (node_a, node_b), votes in edge_votes.items():
        # Calculate scores (proportion of runs voting for each class)
        forward_score = votes["forward"] / successful_runs
        reverse_score = votes["reverse"] / successful_runs
        none_score = votes["none"] / successful_runs

        # Find the class with maximum score
        max_score = max(forward_score, reverse_score, none_score)

        # Only add edge if max_score > threshold AND it's not "no relationship"
        if max_score > score_threshold:
            if forward_score == max_score:
                # A → B wins
                reasons = edge_reasons.get((node_a, node_b), [""])
                reason = max(set(reasons), key=reasons.count) if reasons else ""
                final_edges.append({
                    "from": node_a,
                    "to": node_b,
                    "score": round(forward_score, 2),
                    "reason": reason,
                    "votes": f"A→B:{votes['forward']}, B→A:{votes['reverse']}, None:{votes['none']}"
                })
            elif reverse_score == max_score:
                # B → A wins
                reasons = edge_reasons.get((node_b, node_a), [""])
                reason = max(set(reasons), key=reasons.count) if reasons else ""
                final_edges.append({
                    "from": node_b,
                    "to": node_a,
                    "score": round(reverse_score, 2),
                    "reason": reason,
                    "votes": f"A→B:{votes['forward']}, B→A:{votes['reverse']}, None:{votes['none']}"
                })
            # If none_score == max_score, we don't add any edge (no relationship wins)

    print(f"\n✓ Final edges before cycle removal (max class score >= {score_threshold}): {len(final_edges)}")

    # Remove cycles by greedily dropping the lowest-score edge that forms a cycle
    def _remove_cycles(edges: list) -> list:
        import networkx as nx
        # Sort ascending by score so we drop weakest edges first
        edges_sorted = sorted(edges, key=lambda e: e["score"])
        G = nx.DiGraph()
        kept = []
        for e in reversed(edges_sorted):  # add strongest first
            G.add_edge(e["from"], e["to"])
            if not nx.is_directed_acyclic_graph(G):
                G.remove_edge(e["from"], e["to"])
                print(f"  ✂ Removed cycle edge: {e['from']} → {e['to']} (score={e['score']:.2f})")
            else:
                kept.append(e)
        return kept

    final_edges = _remove_cycles(final_edges)
    print(f"✓ Final edges after cycle removal: {len(final_edges)}")
    for e in sorted(final_edges, key=lambda x: -x["score"]):
        print(f"  {e['from']} → {e['to']}: score={e['score']:.2f} ({e['votes']})")

    # Save final edges as JSON
    out_path = json_path.replace(".json", "_dag_edges.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(final_edges, f, indent=2, ensure_ascii=False)
    print(f"\n📁 Edges saved to: {out_path}")

    # Save adjacency matrix and adjacency list
    base = json_path.replace(".json", "_dag")
    save_dag_graph(final_edges, base_path=base)

    # Plot final DAG with scores
    if final_edges:
        plot_path = json_path.replace(".json", "_dag.png")
        try:
            plot_dag_with_scores(final_edges, treatment, outcome, out_path=plot_path)
        except Exception:
            plot_dag_matplotlib(final_edges, treatment, outcome, out_path=plot_path)

    return final_edges


def construct_dag(json_path: str, treatment: str = "ACE inhibitors", outcome: str = "falls", research_question: str=None):

    def _timing_agent_factory():
        return create_dag_agent(role_name=TIMING_CLASSIFIER_ROLE, persona=TIMING_CLASSIFIER_PERSONA)

    print("⏱️  Classifying node timing...")
    timing = asyncio.run(classify_node_timing(
        json_path, treatment, outcome, research_question or "",
        agent_factory=_timing_agent_factory,
        has_only_selection_role_fn=has_only_selection_role,
    ))

    prompt = build_dag_prompt(json_path, treatment, outcome, timing=timing)

    # Count features for logging
    with open(json_path, 'r', encoding='utf-8') as f:
        _data = json.load(f)
    dag_features = [entry for entry in _data if not has_only_selection_role(entry)]
    print(f"Building DAG with {len(dag_features)} features + treatment + outcome nodes")
    print(f"Treatment: {treatment}  |  Outcome: {outcome}")
    print(f"🔬 Starting finding dag related to: {research_question}")
    print("=" * 60)

    edges = asyncio.run(_run_dag_construction_phases(prompt, research_question or "", treatment=treatment))

    print("\n" + "=" * 60)
    print("📊 RESEARCH RESULTS")
    print("=" * 60)
    print(f"DAG constructed: {len(edges)} edges")
    for e in edges:
        print(f"  {e['from']} → {e['to']}: {e['reason'][:80]}...")

    # Save edges to JSON
    out_path = json_path.replace(".json", "_dag_edges.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(edges, f, indent=2, ensure_ascii=False)
    print(f"\nEdges saved to: {out_path}")

    # Plot and save the DAG image
    plot_path = json_path.replace(".json", "_dag.png")
    plot_dag(edges, treatment=treatment, outcome=outcome, out_path=plot_path)

    # Save adjacency matrix and adjacency list
    base = json_path.replace(".json", "_dag")
    save_dag_graph(edges, base_path=base)

    return edges


def construct_dag_ensemble(json_path: str, treatment: str, outcome: str,
                           research_question: str,
                           num_runs: int = 10, score_threshold: float = 0.5):
    """
    Synchronous wrapper for construct_dag_ensemble_async.
    Uses 3-class classification (A→B, B→A, No edge) for edge direction.
    Runs the async ensemble DAG construction using asyncio.run().
    """
    return asyncio.run(
        construct_dag_ensemble_async(
            json_path, treatment, outcome,
            research_question,
            num_runs, score_threshold
        )
    )


if __name__ == "__main__":
    import re

    enriched_json_path     = ""
    save_folder       = ""
    treatment = ""
    outcome = ""
    research_question = ""

    # enriched_json_path = os.path.join(
    #     save_folder,
    #     f"enriched_variables_{_slug(treatment)}_{_slug(outcome)}.json",
    # )

    async def main():

        # Step 2: Construct ensemble DAG from enriched variables
        await construct_dag_ensemble_async(
            json_path=enriched_json_path,
            treatment=treatment,
            outcome=outcome,
            research_question=research_question,
            num_runs=5,
            score_threshold=0.5,
        )

    asyncio.run(main())
 