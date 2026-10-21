
# Case study I: Prophylactic antibiotics and antibiotic resistance
 
Important intermediate outputs of the Study Design Agent for the research question:
 
> Estimate the effect of prophylactic antibiotic use on antibiotic resistance in women with rUTIs.
 
| | |
|---|---|
| Exposure (DAG treatment node) | `prophylactic antibiotic exposure` |
| Outcome (DAG outcome node) | `antibiotic resistance` |
| Population | Women with recurrent urinary tract infections (rUTIs) |
 
## At a glance
 
| Stage | Result |
|---|---|
| Full-text screening | 1,132 papers passed abstract screening: **8 included**, 795 excluded, 329 without retrievable full text |
| Knowledge cards | 168 variable terms (before deduplication) |
| DAG nodes | 48 nodes: 45 in the DAG + 3 eligibility-only nodes (`baseline pregnancy`, `baseline breastfeeding`, `history of renal transplantation`) |
| DAG edges | 118 edges over the 45 DAG nodes; 49 edges were chosen in all 5 runs. 23 edges point into the exposure and 26 into the outcome |
 
## Files
 
| File | Pipeline step | Content |
|---|---|---|
| `fulltext_screen_prophylactic antibiotic use_antibiotic resistance.csv` | Steps 4–5 | Every paper that passed abstract screening, with both screening rounds' decisions (see columns below) |
| `knowledge_card_prophylactic_antibiotic_use_antibiotic_resistance.json` | Step 6 | One knowledge card per extracted variable term: `{"term", "knowledge card"}` (type and description, from UMLS and web search) |
| `extracted_variables_prophylactic_antibiotic_use_antibiotic_resistance_dag.json` | Step 6 | DAG nodes after deduplication and grouping (see fields below) |
| `extracted_variables_prophylactic_antibiotic_use_antibiotic_resistance_dag_dag_edges.json` | Step 7 | Final DAG edges from the 5-run ensemble (see fields below) |
 
### Screening table columns
 
| Column | Meaning |
|---|---|
| `id`, `title`, `authors`, `pub_date`, `journal`, `publication_types`, `abstract`, `doi` | PubMed record (`id` is the PMID) |
| `decision` | Abstract-screening decision (`INCLUDE` for every row, since only abstract-included papers are listed) |
| `reviewer_1_decision`, `reviewer_1_rationale`, `reviewer_2_decision`, `reviewer_2_rationale`, `agreement` | The two abstract reviewers' votes and reasons, and whether they agreed |
| `fulltext_decision` | `INCLUDE`, `EXCLUDE`, or `NO_FULLTEXT` (full text could not be retrieved) |
| `fulltext_reviewer_1_*`, `fulltext_reviewer_2_*`, `fulltext_agreement` | The same, for full-text screening |
 
A paper is included when **at least one** of the two reviewers votes to include it.
 
### DAG node fields
 
| Field | Meaning |
|---|---|
| `variable` | Node name used in the DAG |
| `roles` | `treatment`, `outcome`, `covariate`, and/or `selection_variable` (a cohort eligibility criterion). Nodes whose only role is `selection_variable` are kept out of the DAG |
| `knowledge_card` | Definition of the variable |
| `children` | More specific variables merged into this node (e.g. `ciprofloxacin resistance` and `E. coli antibiotic resistance` under `antibiotic resistance`) |
| `paper_ids` | PMIDs of the included papers that report the variable |
| `frequency` | Number of those papers |
| `source_passages` | Sentences from the papers that mention the variable |
 
### DAG edge fields
 
| Field | Meaning |
|---|---|
| `from`, `to` | Directed edge: `from` causes `to` |
| `reason` | The mechanism given by the agents |
| `score` | Share of the 5 runs that chose this edge direction (0.6–1.0) |
| `votes` | Raw vote counts: `A→B` (this direction), `B→A` (reverse), `None` (no edge) |
 
 
## Reproduce
 
Run the pipeline (see the [main README](../../README.md)) with the research question above, `dag_num_runs=5` and `dag_score_threshold=0.5`. LLM outputs vary between runs, so a new run will not give an identical DAG.
