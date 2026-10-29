# Reproducibility Guide

This document provides step-by-step instructions to reproduce the data processing pipeline, validation audits, and analytical figures/tables for the study **"Charting the Global Landscape of Metal-Ion Battery Research from 128,675 Publications Using AI-Driven Literature Mining"**.

---

## 1. Two Levels of Reproducibility

Because Web of Science (WoS) licensing terms prohibit redistributing raw article titles, abstracts, and sentence text, this repository supports two distinct reproducibility workflows:

1. **Direct Reproduction of Validation Audits, Figures, and Tables (No WoS License Required)**:
   - Uses the enriched, text-free CSV files in `Data/` (`final_corpus_data.csv`, `df_data_phrases.csv`, `topic_relevance_index.csv`, `evoc_topic_layers_by_doi.csv`, and `Data/Validation_and_Audit/*.csv`).
   - All topic assignments, study types, working-ion chemistries (`is_li_ion`, `is_na_ion`, `is_k_ion`, `is_zn_ion`, `is_other_metal`), impact quadrants (`category_group`, `JNIF`), and sentence-level categories (`challenges_group`, `directions_group`, `controversies_group`) are already pre-computed and linked by `DOI` and `doi_phrase`.
2. **Full End-to-End Pipeline Re-execution from Raw Bibliometric Records (Requires WoS Core Collection Access)**:
   - Researchers with institutional access to the Web of Science Core Collection can export the full records for the `128,675` DOIs listed in `Data/final_corpus_IDs.csv` (or re-run the search query below) to re-hydrate the raw `Document Title`, `Abstract`, `Year Published`, `Total Times Cited Count`, `Document Type`, `Publication Name`, and `Author Address` columns, and then execute every script in `Pipeline/` from scratch.

---

## 2. Software and Hardware Environment

### Hardware Used in the Study
- **GPU**: NVIDIA GPU with $\ge 16\text{ GB}$ VRAM (CUDA 12.x) for running `KaLM-embedding-multilingual-mini-instruct-v2.5`, `sentence-transformers/all-mpnet-base-v2`, and the local `vLLM` server (`RedHatAI/gemma-3-4b-it-quantized.w8a8`).
- **System RAM**: $\ge 32\text{ GB}$ recommended for handling the $128,675$-document embedding matrix and the $910,492$-row sentence table (`df_data_phrases.csv`).

### Python Environment Setup

Python 3.10+ is recommended. Install the required packages in a virtual or Conda environment:

```bash
conda create -n batteries_nlp python=3.10 -y
conda activate batteries_nlp

# Core data science & visualization
pip install numpy pandas pyarrow scipy scikit-learn matplotlib seaborn plotly kaleido openpyxl tqdm requests

# NLP, topic modeling & embeddings
pip install torch sentence-transformers bertopic evoc umap-learn datamapplot spacy gensim string-grouper wordcloud country_converter kneed

# Local LLM serving & OpenAI-compatible client
pip install vllm openai
```

### Local vLLM Server Configuration (Required Only for Stage 4 & Stage 5 Extraction)

Scripts that perform zero-shot classification or open-label extraction connect to a local OpenAI-compatible `vLLM` server hosting `RedHatAI/gemma-3-4b-it-quantized.w8a8` on `http://localhost:8000/v1`:

```bash
vllm serve RedHatAI/gemma-3-4b-it-quantized.w8a8 \
    --host 0.0.0.0 \
    --port 8000 \
    --quantization compressed-tensors \
    --max-model-len 4096
```

---

## 3. Reconstructing the Raw Web of Science Corpus (Optional)

### 3.1 Search Query
The bibliographic retrieval was executed on **25 September 2025** in the **Web of Science Core Collection** using the following Topic (`TS`) query, restricted to `Language = English` and `Document Types = Article OR Review Article OR Early Access`:

```text
TS = (("Batter*") AND ("ion" OR "Metal-ion" OR "Metal ion" OR "M-ion"
      OR "Li-ion" OR "Na-ion" OR "K-ion" OR "Zn-ion" OR "Al-ion"
      OR "Mg-ion" OR "Ca-ion" OR "Lithium-ion" OR "Sodium-ion"
      OR "Potassium-ion" OR "Zinc-ion" OR "Aluminum-ion"
      OR "Magnesium-ion" OR "Calcium-ion")
      AND ("Cathod*" OR "Anod*" OR "Electrod*"))
```

### 3.2 Corpus Selection Funnel
1. **Initial WoS Export**: `129,789` records (`Data/initial_corpus_IDs.csv`).
2. **DOI Completeness Filter**: `1,114` records lacking a persistent DOI were removed, yielding **`128,675` unique DOI-identified publications** (`Data/final_corpus_IDs.csv`) spanning `1992–2026` (`128,127` with a non-empty abstract).
3. **Domain Relevance Verification**: Unlike broader materials queries, expert inspection of preliminary unsupervised clusters confirmed that all clusters belonged to metal-ion battery materials, electrochemistry, or cell engineering. Consequently, no supervised SetFit pruning was applied, and all `128,675` records were retained.
4. **Re-hydrating `final_corpus_data.csv`**: To run scripts that require raw WoS fields (`Document Title`, `Abstract`, `Year Published`, `Total Times Cited Count`, `Document Type`, `Publication Name`, `WOS Categories`, `Research Areas`), merge your WoS export with `Data/final_corpus_data.csv` on `DOI` and save as `Data/df_data_batteries_130k.parquet` (or `.csv`).

---

## 4. Step-by-Step Pipeline Execution (`Pipeline/`)

All scripts in `Pipeline/` read paths and column names from `Pipeline/config.py` and `Pipeline/utils.py`. By default, `config.py` points to `Batteries_Research/Data/` and `Batteries_Research/Results/`.

### Step 4.1 — Metadata Parsing, Sanitization, and Keyword Normalization
1. **Parse authors, affiliations, countries, macro-regions, CPY, and JIF**:
   ```bash
   cd Pipeline
   python 2025_10_16_prepare_df_data_v1.py
   ```
2. **Sanitize and deduplicate institution and country names**:
   Open and execute `Pipeline/2026_03_03_data_sanitization.ipynb` to apply similarity-based institution deduplication (`string_grouper` $\ge 0.85$ + manual dictionary) and country standardization.
3. **Normalize author keywords**:
   ```bash
   python 2025_11_11_normalize_keywords_in_df_data.py
   ```
   Generates `author_keywords_normalized_list`.

### Step 4.2 — Unsupervised Topic Discovery (`KaLM` + `EVoC` + `BERTopic`)
1. **Run `Pipeline/EVOC_v4.ipynb`**:
   - Concatenates `"Title: " + title + "\n\nAbstract: " + abstract` and prepends the instruction prompt:
     `"Instruct: Given the title and abstract of a scientific paper on battery materials and electrochemistry, generate a representation capturing the battery technology, active materials, and key findings to cluster similar research topics.\nQuery: "`
   - Encodes all `128,675` documents with `KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5`.
   - Clusters embeddings using `evoc.EVoC(noise_level=0.12, base_min_cluster_size=5)` inside `BERTopic`, retaining layers with $< 150$ clusters (`Layer 0: 67 topics`, `Layer 1: 23 topics`, `Layer 2: 7 topics`).
   - Reassigns outlier (`-1`) documents to their closest topic centroid by cosine similarity (`100%` corpus coverage).
   - Generates concise topic labels using `gpt-5-mini` (set `export OPENAI_API_KEY="your-key"` before running Cell 2/5) and exports the UMAP landscape (`Batteries_Articles_cluster_1.pdf`).
2. **Evaluate layer quality metrics and select Layer 1 ($K=23$)**:
   ```bash
   cd ../Validation
   python 2026_04_27_silhouette_davies_bouldin_indices.py
   ```
   Reproduces `Data/Validation_and_Audit/layer_quality_metrics.csv`, confirming that **Layer 1 ($K=23$)** maximizes the Standard CVI (`CVI2_v2_Score = 0.0724`, Silhouette $= 0.0319$, Davies–Bouldin $= 3.965$).
3. **Attach `topic_cluster_1` to the main dataset**:
   ```bash
   cd ../Pipeline
   python 2025_10_23_add_topic_column_to_df_data.py
   ```

### Step 4.3 — Composite Relevance Index and Impact Quadrants (`JNIF`)
1. **Compute Topic Relevance Index (with and without reviews)**:
   ```bash
   python 2025_11_06_add_relevance_index_df_data_v1.py
   ```
   Computes $\text{Index}_T = 0.5\,\text{mean\_CPY}_T + 0.3\,\text{prop\_top10}_T + 0.2\,\log(1+n_T)$, writes `Data/topic_relevance_index.csv`, and populates `topic_rank_index*` columns.
2. **Compute Impact Quadrants (`Star`, `Hidden Gem`, `Coattail`, `Long Tail`) and `JNIF`**:
   ```bash
   python 2026_01_08_create_JNIF_score_log_score_sqrt_columns_in_df_data.py
   ```

### Step 4.4 — LLM Study-Type and Working-Ion Extraction
*(Requires the local `vLLM` server running `RedHatAI/gemma-3-4b-it-quantized.w8a8`)*
1. **Classify documents into `Review`, `Experimental studies`, and `Computational and simulation studies`**:
   ```bash
   python 2025_11_11_classify_research_type.py
   ```
2. **Extract battery working-ion chemistries (`Li-ion`, `Na-ion`, `K-ion`, `Zn-ion`, `Other`)**:
   ```bash
   python 2026_04_07_extract_studied_metal_from_abstracts_v2.py
   ```
   Populates `studied_metal_metal_label`, `is_li_ion`, `is_na_ion`, `is_k_ion`, `is_zn_ion`, `is_other_metal`, and `is_irrelevant`.

### Step 4.5 — Sentence Segmentation and Extraction of Challenges, Directions, and Controversies
1. **Segment abstracts into sentences (`910,492` sentences across `128,127` abstracts)**:
   ```bash
   python 2025_11_21_split_abstract_sentences.py
   ```
2. **Extract open 2–5 word labels per sentence via `vLLM` (`Gemma-3-4B-IT`)**:
   ```bash
   python 2025_11_24_extract_challenges_from_sentences.py
   python 2025_11_24_extract_directions_from_sentences.py
   python 2025_11_24_extract_controversies_from_sentences.py
   ```
3. **Cluster distinct labels into 12 categories per vector (`all-mpnet-base-v2` + `MiniBatchKMeans`, `seed=42`) and name via `vLLM`**:
   - *Stage 1* (`_part_1.py`) encodes unique labels on GPU using `sentence-transformers/all-mpnet-base-v2`, frees GPU VRAM, clusters into `N_CLUSTERS = 12` with `MiniBatchKMeans(random_state=42, batch_size=512, max_iter=200)`, and extracts the 6 most frequent terms among the 20 closest labels to each centroid.
   - *Stage 2* (`_part_2.py`) queries the `vLLM` server (`temperature=0.2`) to assign a 3–5 word technical class name to each cluster and saves `*_label_classified_full.csv`, `*_label_classified_sample500.csv`, and `*_label_classification_summary.csv`.
   - *Stage 3* (`_part_3.py`) merges the consolidated `*_group` column back into `df_data_phrases`.
   ```bash
   # Challenges
   python 2025_11_24_create_challenge_classes_sentence_transformers_part_1.py
   python 2025_11_24_create_challenge_classes_sentence_transformers_part_2.py
   python 2025_11_24_create_challenge_classes_sentence_transformers_part_3.py

   # Research Directions
   python 2025_11_24_create_direction_classes_sentence_transformers_part_1.py
   python 2025_11_24_create_direction_classes_sentence_transformers_part_2.py
   python 2025_11_24_create_direction_classes_sentence_transformers_part_3.py

   # Controversies
   python 2025_11_24_create_controversy_classes_sentence_transformers_part_1.py
   python 2025_11_24_create_controversy_classes_sentence_transformers_part_2.py
   python 2025_11_24_create_controversy_classes_sentence_transformers_part_3.py

   # Normalize 'No Challenge' / 'No Research Direction' / 'No Controversy' group labels
   python 2026_10_14_fix_challenge_direction_controversy_labels.py
   ```

---

## 5. Running the Validation and Audit Suite (`Validation/`)

The `Validation/` folder contains the automated audit suite that verifies every quantitative claim in the manuscript and Supporting Information:

```bash
cd Validation

# 1. Score the expert human validation samples directly from the public text-free CSVs:
#    (Reproduces 99.0% accuracy for research_type, and 0.56 / 0.44 / 0.42 accuracy for
#     challenges / directions / controversies, plus Precision, Recall, F1, and Support tables)
python 06b_score_validation.py

# 2. If you have re-hydrated the WoS metadata columns (Year Published, Total Times Cited Count,
#    Document Type, Abstract), you can also re-run the full corpus audit suite:
python 00_inspect_columns.py
python 01_corpus_composition.py
python 01b_funnel.py
python 02_relevance_index.py
python 03_jnif_quadrants.py
python 04_topic_validation.py
python 05_intent_stats.py
```

Pre-generated outputs from all audit scripts are archived in `Validation/audit_reports/` (`.txt`, `.json`, and `.tex`) and `Data/Validation_and_Audit/` (`.csv`).

---

## 6. Reproducing Manuscript and SI Figures and Tables (`Analysis/`)

All scripts in `Analysis/` read from `Data/` via `Analysis/config.py` and `Analysis/utils.py`, and write figures (`.pdf`) and LaTeX tables to `Results/`.

### Scripts Executable Directly from the Public Text-Free CSVs in `Data/`
The following scripts depend **only** on the processed columns included in `Data/final_corpus_data.csv`, `Data/df_data_phrases.csv`, and `Data/topic_relevance_index.csv` (no raw WoS fields required):

```bash
cd Analysis

# Beyond-Lithium & Working-Ion Chemistry Analyses (Figure 3 & Table 3)
python q4_7_1_macro_and_micro_proportion.py
python q4_7_2_challenges_by_metal.py
python q4_7_3_donut_macro_proportion.py
python q4_7_4_metals_plot_intersections_v2.py
python q4_8_1_global_penetration_bar.py
python q4_8_2_geopolitical_heatmap.py
python q4_8_3_geographic_specialization_table.py
python q4_11_1_strategic_transition_matrix.py
python q4_11_2_portfolio_stacked_bar.py
python q4_11_3_escape_ratio_table.py
python 2026_04_14_ion_distribution_chart.py

# Geography, Institutions, Collaboration & Study Types
python q4_1_1_sankey_region_country_v1.py
python q4_2_2_Intitution_and_top_studied_topics_v2.py
python "q4_3_1_top _institutions.py"
python q4_3_2_cross_country_collaboration_map.py
python "q4_3_3_top first_and_last_author.py"
python q4_3_4_top_first_author.py
python q4_3_10_sankey_country_collaboration_v1.py
python q4_4_2_last_author_and_top_studied_topics.py
python q4_6_3_word_cloud.py
python q4_9_1_topics_and_research_types.py

# Sentence-Level Rankings & Country/Topic Biclustered Dendrograms (Figure 5 & SI)
python q4_12_2_challenge_country_dendrogram.py
python q4_12_3_challenge_topic_dendrogram.py
python q4_12_5_direction_country_dendrogram.py
python q4_12_6_direction_topic_dendrogram.py
python q4_12_7_challenge_ranking_chart.py
python q4_12_8_directions_ranking_chart.py
python q4_12_9_controversies_ranking_chart.py
python q4_12_11_controversies_country_dendrogram.py
python q4_12_12_controversies_topic_dendrogram.py
```

### Scripts Requiring Re-Hydrated Bibliometric Fields (`Year Published` / `Total Times Cited Count` / `Publication Name`)
Scripts that plot longitudinal time series by `Year Published` or compute raw citation totals (`q3_1_publications_per_year_and_period.py`, `q4_1_2_maps_by_decade_abs.py`, `q4_1_2_maps_by_decade_log.py`, `q4_1_3_*` to `q4_1_5_*`, `q4_2_1_*`, `q4_3_4_top_last_author_recent_affiliation.py` to `q4_3_9_*`, `q4_4_1_*`, `q4_4_4_*`, `q4_4_5_*`, `q4_5_*`, `q4_6_1_*`, `q4_6_2_*`, `q4_6_4_*`, `q4_6_5_*`, `q4_7_5_*`, `q4_10_*`, `q4_12_1_*`, `q4_12_4_*`, `q4_12_10_*`, and `2026_03_11_data_overview.ipynb`) can either be run after merging the WoS export by `DOI` (Section 3.2), or their pre-aggregated time-series tables can be inspected directly in `Data/Validation_and_Audit/` (`01_docs_by_year.csv`, `02_RI_*.csv`, and `05_*_by_year_*.csv`).
