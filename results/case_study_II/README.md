# Case study II: Breastfeeding duration and cognitive performance
 
Important intermediate outputs of the Study Design Agent for the research question:
 
> Assess the association between breastfeeding duration and cognitive outcomes among youths in China.
 
| | |
|---|---|
| Exposure (DAG treatment node) | `breastfeeding duration` |
| Outcome (DAG outcome node) | `cognitive performance` |
| Population | Youths in China |
 
 
## At a glance
 
| Stage | Result |
|---|---|
| Full-text screening | 468 papers passed abstract screening: **13 included**, 364 excluded, 91 without retrievable full text |
| Knowledge cards | 225 variable terms (before deduplication) |
| DAG nodes | 66 nodes, all eligible for the DAG (`baseline age` and `ethnicity` are also eligibility criteria) |
| DAG edges | 80 edges over 40 nodes; 23 edges were chosen in all 5 runs. 18 edges point into the exposure and 18 into the outcome |
 
## Files
 
| File | Pipeline step | Content |
|---|---|---|
| `fulltext_screen_breastfeeding_duration_cognitive_performance.csv` | Steps 4–5 | Every paper that passed abstract screening, with both screening rounds' decisions (see columns below) |
| `knowledge_card_breastfeeding_duration_cognitive_performance.json` | Step 6 | One knowledge card per extracted variable term: `{"term", "knowledge card"}` (type and description, from UMLS and web search) |
| `extracted_variables_breastfeeding_duration_cognitive_performance_dag.json` | Step 6 | DAG nodes after deduplication and grouping (see fields below) |
| `extracted_variables_breastfeeding_duration_cognitive_performance_dag_dag_edges.json` | Step 7 | Final DAG edges from the 5-run ensemble (see fields below) |
 
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
| `children` | More specific variables merged into this node (e.g. `birth weight` and `birth length` under `birth size`; maternal and paternal education under `baseline parental education`) |
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
 
```
 
## Reproduce
 
Run the pipeline (see the [main README](../../README.md)) with the research question above, `dag_num_runs=5` and `dag_score_threshold=0.5`. LLM outputs vary between runs, so a new run will not give an identical DAG.
