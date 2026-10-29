# Batteries Corpus Audit (AKL) — Scripts

Scripts to generate the audit metrics and validation statistics required for the batteries paper and Supplementary Information (SI). All scripts have been tested end-to-end on datasets matching the Web of Science (WoS) column structure.

## Installation

```bash
pip install pandas pyarrow numpy scipy scikit-learn gensim openpyxl
```

Place this folder inside the project directory (where `config.py` is located). The scripts read:

- `config.DATA_FOLDER/df_data_batteries_130k.parquet` (main corpus dataset)
- `config.DATA_FOLDER/df_data_phrases.parquet` (sentences with challenges/directions/controversies)

Outputs are written to `./audit_outputs/` (and archived reference reports are available in `./audit_reports/`).

## Before Running

No manual configuration is required. Column names are automatically matched to `df_data_batteries.csv` and `df_data_phrases.csv` (for example, `Year Published`, `Total Times Cited Count`, `journal_if`, `first_country`, `topic_cluster_1`, and the join `phrases.doi -> main.DOI`). Script `00` displays the column mapping. If any mapping needs adjustment, edit `column_map.json` and re-run.

The citations-per-year formula does **not** need to be specified manually. Script `02` tests candidate combinations (citation column, `last_year`, fractional year factor, handling of 2026 records, "Review" filtering rule, P90 baseline, and year cutoff up to 2024) against the stored `citations_per_year` and `topic_rank_index` columns in the dataset, selecting the combination that reproduces the stored values.

## Execution Order

| # | Script | Item | Approx. Time |
|---|---|---|---|
| 0 | `00_inspect_columns.py` | Column mapping inspection | seconds |
| 1 | `01_corpus_composition.py` | 8 & 9 — Table 2 + corpus composition | ~1 min |
| 2 | `02_relevance_index.py` | 10 — RI, ranges, contributions, Spearman | seconds |
| 3 | `03_jnif_quadrants.py` | 16 — JNIF and impact quadrants | seconds |
| 4 | `04_topic_validation.py` | 11 — C_v, NPMI, UMass, diversity (ARI/NMI if layers 0 and 2 are present) | 10–60 min (C_v) |
| 5 | `05_intent_stats.py` | 12 — Challenges/directions/controversies statistics | ~2 min |
| 6a | `06a_make_validation_samples.py` | 12 & 13 — Validation spreadsheets for domain expert | seconds |
| 6b | `06b_score_validation.py` | 12 & 13 — Validation metrics (after expert annotation) | seconds |

**Run Step 0 first.** It writes `column_map.json` with the automatic column mapping. If any column appears as `null` or maps to the wrong field (e.g., country or impact factor), edit the JSON file and run the remaining scripts.

## Outputs

The `.txt` and `.json` files from `00` to `05` (and `06b`) contain aggregate statistics only and do not include raw titles or abstracts. The `05_*` CSV files can be used to regenerate the `q4_12_*` figures with the plotting scripts.

The `validation_*.xlsx` spreadsheets contain WoS text and are intended strictly for local domain-expert annotation; anonymized label-only CSVs (`validation_*_labels_only.csv`) are stored in `Data/Validation_and_Audit/`.

## Manual Validation (Step 6)

1. Run `06a`. Using random seed 42, it generates:
   - `validation_challenges.xlsx`, `validation_directions.xlsx`, and `validation_controversies.xlsx`:
     - `amostra` sheet: 100 categorized sentences, ~8 per category (disproportionate stratified sampling);
     - `none_check` sheet: 30 sentences classified as "No X", to estimate false negatives (optional).
   - `validation_research_type.xlsx`: 100 non-review documents.
2. The domain expert fills only the `expert_*` columns (via dropdown menus). If a sentence is not a challenge/direction/controversy, the expert selects `NOT_A_...`.
3. Run `06b`. It computes overall accuracy, per-category precision/recall/F1, confusion matrices, and the LaTeX tables `06b_*_metrics.tex`.

Setting `BLIND = True` in `06a` hides the predicted category from the domain expert (more rigorous, though more time-consuming). The default is `False`.

## Notes on Implemented Rules

- **Temporal exposure**: in the dataset extract, `citations_per_year` = `Total Times Cited Count` / (2025 − year + 1). Under that rule, 2026 documents would divide by zero. Scripts `02` and `03` report how many such records exist and how they are stored.
- **Reviews**: a document is counted as a review when its document type field contains the label `"Review"` in any position (including `"Review; Early Access"`). The research-type classification script uses exact equality (`== "Review"`). Script `01` reports both counts.
- **"No X" filtering** (`05` and `06a`): sentences whose group or raw label starts with `"No challenge / No research direction / No controversy"` (case-insensitive, with or without punctuation) are excluded from categorized statistics. The script also checks whether any variants escaped exact matching and which categories they fell into.
- **Countries** (`01` and `05`): if no dedicated country column is present, countries are extracted from the WoS address field (last comma-separated segment; US addresses with state/ZIP codes are normalized to `"USA"`).
