# Study Design Agent

A multi-agent system that turns a free-text causal research question into an observational study design. It finds and screens the literature, extracts the variables used in prior studies, builds a causal DAG, derives a minimal adjustment set, and checks that each variable exists in a target dataset's metadata.

The pipeline can run **fully automated** (end-to-end) or **human-in-the-loop** (it pauses for your review after each key step). One switch controls this; see [Running the pipeline](#running-the-pipeline).

---

## Pipeline

| Step | What happens | Main output |
|---|---|---|
| 1. Task assessment | Classifies the question as causal, not applicable, or unclear. Non-causal questions stop here. | — |
| 2. PECO extraction | Extracts Population, Exposure, Comparison and Outcome, and maps each to MeSH terms. | PECO JSON |
| 3. Literature search | Two PubMed search lines run in parallel (systematic-review references + a refined primary-study query), then merged and deduplicated. | `pubmed_results_*.csv` |
| 4. Abstract screening | Two reviewer agents with derived personas screen titles/abstracts (sensitive: include if either says yes). | `abstract_screen_*_included.csv` |
| 5. Full-text screening | Full text is retrieved and screened by two reviewers (specific). | `fulltext_screen_*_included.csv` |
| 6. Variable extraction | Variables are extracted from each paper, enriched with knowledge cards (UMLS + web), deduplicated, and grouped into DAG nodes. | `extracted_variables_*_dag.json` |
| 7. DAG construction | An expert-agent team proposes, reviews and checks edges over several runs; edges are kept by majority vote, and cycles are removed. | `*_dag_edges.json`, `*_dag.png` |
| 8. Adjustment set | Computes the optimal minimal adjustment set (tigramite) and selects cohort-eligibility variables. | `*_adjustment_metadata_match_input.json` |
| 9. Metadata matching | Checks each variable against a dataset's metadata DB. Missing variables are treated as hidden and the adjustment set is recomputed. If no valid set remains, the variable is dropped and recorded as an assumption. | `*_adjustment_final.json`, `*_adjustment_summary.json` |

---

## Requirements

Install the Python packages (versions are pinned to the tested environment):

```bash
conda create -n studycamel python=3.12
conda activate studycamel
pip install -r requirements.txt
```

---

## API keys

Put your keys in a `.env` file in the project root. The code loads it automatically. 

### Required

| Variable | Used for | Get a key |
|---|---|---|
| `OPENROUTER_API_KEY` | All LLM agents (models `openai/gpt-5.4` and `openai/gpt-5.4-nano` via OpenRouter) | [openrouter.ai/settings/keys](https://openrouter.ai/settings/keys) (You can change it to the official openai setting)|
| `UMLS_API_KEY` | Knowledge cards for extracted variables (Step 6). Requires a free UMLS licence. | Apply for a licence at [uts.nlm.nih.gov](https://uts.nlm.nih.gov/uts/). The key is under **My Profile** (tick *Generate New API Key* if blank). See [NLM's guide](https://documentation.uts.nlm.nih.gov/rest/authentication.html). |

### Recommended

| Variable | Used for | Get a key |
|---|---|---|
| `PUBMED_API_KEY` | PubMed E-utilities (Step 3). Without a key, NCBI limits you to 3 requests/second; with one, 10/second. | Sign in to NCBI, open [Account Settings](https://account.ncbi.nlm.nih.gov/settings/) → **API Key Management** → *Create an API Key*. See [NLM's help article](https://support.nlm.nih.gov/knowledgebase/article/KA-05317/en-us). |
| `UNPAYWALL_EMAIL` | Your email address, sent with Unpaywall / OpenAlex / Crossref requests (no key needed) | [unpaywall.org/products/api](https://unpaywall.org/products/api) |

### Optional: extra full-text sources (Step 5)

Full text is tried from free sources first (local PDF → PubMed Central → Europe PMC → PLOS → OpenAlex → Unpaywall → Crossref). Each publisher source below is used only if its key is set; otherwise it is skipped. More keys means more papers retrieved.

| Variable | Source | Get a key |
|---|---|---|
| `ELSEVIER_API_KEY` | Elsevier / ScienceDirect full text | [dev.elsevier.com/apikey/manage](https://dev.elsevier.com/apikey/manage) |
| `ELSEVIER_INST_TOKEN` | Elsevier institutional token (subscribed content off-campus) | Request from Elsevier via [dev.elsevier.com/support](https://dev.elsevier.com/support.html) |
| `WILEY_TDM_API_KEY` | Wiley Online Library text and data mining | Sign in to Wiley Online Library, then *Get a Text and Data Mining Token* on the [TDM page](https://onlinelibrary.wiley.com/library-info/resources/text-and-datamining) (contact TDM@wiley.com if the page has moved) |
| `SPRINGER_API_KEY` | Springer Nature open-access API | Register at the [Springer Nature developer portal](https://dev.springernature.com/docs/quick-start/api-access) |
| `SEMANTIC_SCHOLAR_API_KEY` | Semantic Scholar open-access PDFs (works without a key, but with lower rate limits) | [semanticscholar.org/product/api](https://www.semanticscholar.org/product/api#api-key-form) |
| `JINA_API_KEY` | Jina Reader: scrapes publisher pages by DOI (works without a key, but with lower rate limits) | [jina.ai/reader](https://jina.ai/reader/) |
| `FIRECRAWL_API_KEY` | Firecrawl: last resort for JavaScript-heavy publisher sites (paid) | [firecrawl.dev/app/api-keys](https://www.firecrawl.dev/app/api-keys) |
| `FIRECRAWL_MAX_REQ_PER_MIN` | Firecrawl rate limit for your plan (default `10`) | — |

### Only for standalone tools

| Variable | Used for | Get a key |
|---|---|---|
| `OPENAI_API_KEY` | Only `python terminology_enriched.py --term <term>` (single-term lookup). The main pipeline does not use it. | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |

### `.env` template

```dotenv
# Required
OPENROUTER_API_KEY=
UMLS_API_KEY=

# Recommended
PUBMED_API_KEY=
UNPAYWALL_EMAIL=you@example.org

# Optional full-text sources
ELSEVIER_API_KEY=
ELSEVIER_INST_TOKEN=
WILEY_TDM_API_KEY=
SPRINGER_API_KEY=
SEMANTIC_SCHOLAR_API_KEY=
JINA_API_KEY=
FIRECRAWL_API_KEY=
FIRECRAWL_MAX_REQ_PER_MIN=10
```

---

## Configuration

Settings are in `run()` at the bottom of `study_design_camel.py`:

```python
save_folder = "test_results"                       # where all outputs are written
date_range = {"from": "1950/01/01", "to": "2026/03/30"}   # PubMed publication date range
human_in_the_loop = False                          # True = pause for review, False = fully automated

framework = StudyDesignFramework(
    save_folder=save_folder,
    date_range=date_range,
    human_in_the_loop=human_in_the_loop,
    hidden_variables=[],          # DAG nodes to treat as unmeasured, e.g. ["birth size"]
    dag_num_runs=5,               # number of ensemble DAG runs
    dag_score_threshold=0.5,      # an edge is kept only if > 50% of runs agree
    metadata_db_path="metadata_database/cfps_db/cfps_metadata.db",  # None skips Step 9
)
```

| Setting | Meaning |
|---|---|
| `save_folder` | Output folder. **Create it before running** (`mkdir -p test_results`). |
| `human_in_the_loop` | The switch between the two modes below. |
| `hidden_variables` | Variables you already know are unmeasured in your data; they are never used for adjustment. |
| `dag_num_runs`, `dag_score_threshold` | More runs give a more stable DAG at a higher LLM cost. |
| `metadata_db_path` | Metadata DB of the dataset you plan to analyse (see below), or `None` to skip Step 9. |
| `fulltext_resource_folder` | Folder of local PDFs named `<PMID>.pdf`, used before any online source (default `test_case/low_dose_resource`). |

### Metadata databases

| Path | Dataset |
|---|---|
| `metadata_database/cfps_db/cfps_metadata.db` | China Family Panel Studies (CFPS) |
| `metadata_database/dataloch_db/dataloch_metadata.db` | DataLoch (see its [README](metadata_database/dataloch_db/README.md)) |
| `metadata_database/sail_db/sail_metadata.db` | SAIL Databank |

---

## Running the pipeline

```bash
conda activate studycamel
python study_design_camel.py
```

You'll be asked for your research aim in 1–5 sentences, for example:

```
> Estimate the effect of prophylactic antibiotic use on antibiotic resistance in women with rUTIs.
```

### Mode 1: Fully automated (`human_in_the_loop = False`)

The pipeline runs Steps 1–9 without stopping. The PECO agent does not ask for confirmation, and no review checkpoints are shown. Use this mode for batch runs, benchmarking, or when you want a first draft of the design quickly. You can still review and edit every output file afterwards.

### Mode 2: Human-in-the-loop (`human_in_the_loop = True`)

The pipeline pauses at five checkpoints, and your edits are carried into the following steps.

**Checkpoint 1: PECO extraction (interactive, Step 2).** The agent shows the extracted PECO and asks you to choose:

- **A) CONFIRM**: accept as is
- **B) MODIFY_VARIABLES**: `[{"peco": "population|exposure|comparison|outcome", "raw_term": "<current>", "new_raw_term": "<replacement>"}]`
- **C) DELETE_VARIABLES**: `[{"peco": "...", "raw_term": "<term to delete>"}]`
- **D) ADD_VARIABLES**: `[{"peco": "...", "raw_term": "<new term>"}]`

Modified or added terms are re-standardised and looked up in MeSH again.

**Checkpoints 2–5: file review.** The pipeline prints a summary and the path of a file. Open the file, edit and save it in place if needed, then go back to the terminal:

- press **Enter** to continue with the file as it is now, or
- type **q** to stop the workflow (outputs so far are kept).

Edits are validated before continuing; if a file is invalid, you are told why and asked to fix it.

| # | After | File to review | What you can change |
|---|---|---|---|
| 2 | Full-text screening (Step 5) | `fulltext_screen_<E>_<O>_included.csv` | **Delete rows** to exclude papers. **Add rows** to include papers: `id` (PMID) is required, `doi` optional. Full text is fetched for added papers; if that fails, place `<PMID>.pdf` in `fulltext_resource_folder`. |
| 3 | DAG node construction (Step 6) | `extracted_variables_<E>_<O>_dag.json` | Rename, delete or add nodes, or change `roles`. Each node needs a `variable`; nodes whose only role is `selection_variable` are kept out of the DAG. |
| 4 | DAG construction (Step 7) | `extracted_variables_<E>_<O>_dag_dag_edges.json` | Add, remove or reverse edges (`{"from": ..., "to": ...}`). The graph must stay acyclic (cycles are rejected). The plot and adjacency files are regenerated. |
| 5 | Adjustment set (Steps 8–9) | `extracted_variables_<E>_<O>_dag_adjustment_final.json` | Add or remove adjustment/eligibility variables. Hidden variables and assumptions are shown in the summary. |

`<E>` and `<O>` are the exposure and outcome terms with non-alphanumeric characters replaced by `_`.

Each review is recorded in the workflow state under `human_feedback` (stage, file, and whether it was modified).

---

## Outputs

All files are written to `save_folder`:

| File | Content |
|---|---|
| `pubmed_results_*.csv` | Merged, deduplicated search results |
| `abstract_screen_*.csv` / `_included.csv` / `_details.json` | Abstract screening decisions, included papers, audit trail (criteria, reviewer personas) |
| `fulltext_screen_*.csv` / `_included.csv` / `_details.json` | Full-text screening decisions and audit trail |
| `extracted_variables_*_raw.json` | Variables extracted per paper |
| `knowledge_card_*.json` | UMLS/web knowledge cards |
| `extracted_variables_*_deduplicated.json` | Deduplicated variables |
| `extracted_variables_*_dag.json` / `.csv` | DAG nodes (variables, roles, children, sources) |
| `*_dag_edges.json`, `*_dag.png`, `*_dag_adj_matrix.csv`, `*_dag_adj_list.json` | DAG edges with vote scores, plot, adjacency matrix and list |
| `*_adjustment_metadata_match_input.json` | Adjustment + eligibility variables before metadata matching |
| `*_adjustment_final.json` | Final variables, each with its metadata match status |
| `*_adjustment_summary.json` | Final adjustment set, hidden variables, assumptions, and a log of each decision |
| `*_metadata_match_results.csv` | Matched datasets/fields for each variable |

Logs are written to `logs/workflow_<timestamp>.log` (full detail) and `logs/errors_<timestamp>.log`.

---

## Project structure

```
study_design_camel.py         Main entry point: StudyDesignFramework and the 9-step pipeline
test_search.py                Step 3: literature search lines and deduplication
literature_screen.py          Step 4: abstract dual-reviewer screening
test_browser.py               Steps 5–6: full-text screening, variable extraction, DAG nodes
terminology_enriched.py       Step 6: knowledge cards (UMLS + DuckDuckGo)
test_dag_construct.py         Step 7: ensemble DAG construction
adj_tools.py                  Step 8: adjustment set (tigramite) and eligibility variables
metadata_match.py             Step 9: metadata matching (also usable as a CLI)
pubmed_tools.py               PubMed, MeSH and full-text retrieval tools
tools.py, umls_toolkit.py     MeSH and UMLS lookups
response_format.py            Pydantic schema for PECO extraction
utils.py                      Logging and PECO helpers
prompts/                      All LLM prompts, one file per step:
  task_process.py               Steps 1–2
  literature_process.py         Step 3
  abstract_screening_process.py Step 4
  fulltext_screening_process.py Step 5
  variable_process.py           Step 6
  dag_process.py                Step 7
  adjustment_process.py         Step 8
  metadata_process.py           Step 9
metadata_database/            Metadata DBs for Step 9
```
Several modules can also be run on their own for a single step (e.g. `python test_dag_construct.py`, `python adj_tools.py`, `python metadata_match.py --dag ... --db ...`).

---


