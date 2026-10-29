####################################################################################################
# Sentence-Transformer clustering (Part 1)
####################################################################################################
# PART 1 performs the Sentence-Transformer embedding and clustering steps and then frees GPU memory

import pandas as pd
import numpy as np
import os
import json  # Import json to save top terms
from sklearn.cluster import KMeans, MiniBatchKMeans
from sentence_transformers import SentenceTransformer
import torch # Import torch to clear VRAM

import config

# ---------------------------------------------------------
# Create file with label counts
df = pd.read_csv(f"{config.DATA_FOLDER}/df_data_phrases.csv")

# Count occurrences of the target column
counts = df["directions_label"].value_counts().reset_index()

# Rename columns for clarity
counts.columns = ["directions_label", "count"]

# Save to a new CSV
counts.to_csv(f"{config.DATA_FOLDER}/directions_label_counts.csv", index=False)

print("File 'directions_label_counts.csv' generated successfully!")



# ---------------------------------------------------------
# SETTINGS
INPUT_CSV = f"{config.DATA_FOLDER}/directions_label_counts.csv"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
N_CLUSTERS = 12
RANDOM_SEED = 42

# --- Temporary Files ---
# File names saved by Stage 1 for Stage 2 to read
TEMP_DF_PATH = f"{config.DATA_FOLDER}/_temp_df_with_clusters.csv"
TEMP_TERMS_PATH = f"{config.DATA_FOLDER}/_temp_top_terms.json"
# ---------------------------------------------------------

def main_stage1():
    print("\n🔵 [STAGE 1] Starting clustering...")
    print("\n🔵 [STAGE 1] Loading CSV…")
    df = pd.read_csv(INPUT_CSV)
    labels = df["directions_label"].astype(str).tolist()
    counts = df["count"].tolist() if "count" in df.columns else [1]*len(labels)

    # ---------- Load ST Model on GPU ----------
    print(f"\n🔵 [STAGE 1] Loading sentence-transformer: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device="cuda")

    # ---------- Embeddings ----------
    print("🔵 [STAGE 1] Encoding labels into embeddings…")
    embeddings = model.encode(labels, batch_size=256, show_progress_bar=True)

    # --- FREE VRAM ---
    print("🔵 [STAGE 1] Embeddings generated. Removing ST model from VRAM...")
    del model
    torch.cuda.empty_cache()
    print("🔵 [STAGE 1] GPU memory freed.")
    # ------------------

    # ---------- Clustering (K-Means uses CPU) ----------
    print("\n🔵 [STAGE 1] Clustering embeddings (CPU)...")
    km = MiniBatchKMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_SEED,
        batch_size=512,
        max_iter=200
    )
    clusters = km.fit_predict(embeddings)

    # ---------- Build DataFrame ----------
    df_labels = pd.DataFrame({
        "directions_label": labels,
        "count": counts,
        "cluster": clusters
    })

    # ---------- Extract Top Terms per Cluster ----------
    print("🔵 [STAGE 1] Extracting representative terms per cluster…")
    top_terms_per_cluster = []
    for c in range(N_CLUSTERS):
        cluster_points = df_labels[df_labels.cluster == c]
        idxs = cluster_points.index.tolist()

        centroid = km.cluster_centers_[c]
        dists = np.linalg.norm(embeddings[idxs] - centroid, axis=1)
        closest = cluster_points.iloc[dists.argsort()[:20]]

        terms = closest["directions_label"].str.split().explode().value_counts().head(6).index.tolist()
        top_terms_per_cluster.append(terms)

    # ---------- Save State for Stage 2 ----------
    print(f"🔵 [STAGE 1] Saving temporary dataframe to: {TEMP_DF_PATH}")
    df_labels.to_csv(TEMP_DF_PATH, index=False)

    print(f"🔵 [STAGE 1] Saving top terms to: {TEMP_TERMS_PATH}")
    with open(TEMP_TERMS_PATH, 'w') as f:
        json.dump(top_terms_per_cluster, f)

    print("\n✅ [STAGE 1] COMPLETED!")
    print("--------------------------------------------------------------------")
    print("➡️  NEXT STEP:")
    print("1. LOAD YOUR VLLM MODEL (Gemma-3-4B).")
    print("2. WHEN READY, RUN THE SCRIPT: 2025_11_24_create_direction_classes_sentence_transformers_part_2.py")
    print("--------------------------------------------------------------------")


if __name__ == "__main__":
    main_stage1()
