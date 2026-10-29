####################################################################################################
# Sentence-Transformer clustering (Part 2)
####################################################################################################
# PART 2 uses the VLLM model to name the classes


import pandas as pd
import numpy as np
import os
import json
import requests # For VLLM requests

import config

# ---------------------------------------------------------
# SETTINGS
N_CLUSTERS = 12
SAMPLE_SIZE = 500
RANDOM_SEED = 42

# --- Temporary Files (INPUT FROM PART 1) ---
TEMP_DF_PATH = f"{config.DATA_FOLDER}/_temp_df_with_clusters.csv"
TEMP_TERMS_PATH = f"{config.DATA_FOLDER}/_temp_top_terms.json"

# --- Final Output Files ---
FULL_PATH = f"{config.DATA_FOLDER}/challenges_label_classified_full.csv"
SAMPLE_PATH = f"{config.DATA_FOLDER}/challenges_label_classified_sample500.csv"
SUMMARY_PATH = f"{config.DATA_FOLDER}/challenges_label_classification_summary.csv"

# --- VLLM API CONFIGURATION ---
VLLM_API_URL = "http://localhost:8000/v1/chat/completions" # ADJUST IF NEEDED
VLLM_MODEL_NAME = "RedHatAI/gemma-3-4b-it-quantized.w8a8"
# ---------------------------------------------------------

def make_class_name(top_terms, cluster_index):
    """
    Uses an LLM (via VLLM API) to generate a class name
    based on the most common terms in the cluster.
    """
    terms_str = ", ".join(top_terms)
    

    system_prompt = (
        "You are an expert Electrochemistry researcher specializing in metal-ion batteries. "
        "Your task is to generate a short, descriptive class name (3-5 words) for a cluster of **scientific challenges, obstacles, or limitations in battery research**, "
        "based on a list of keywords. "
        "The name should clearly summarize the **central specific mechanism of failure or bottleneck** "
        "(e.g., 'Dendrite Growth Issues', 'Cathode Material Degradation', 'Low Coulombic Efficiency', 'SEI Layer Instability', 'Slow Ion Diffusion Kinetics')."
    )

    user_prompt = (
        f"The most common keywords in this cluster of battery research challenges are: {terms_str}. "
        f"Based on these keywords, what is the best class name describing this specific technical obstacle? "
        f"Respond with ONLY the class name and nothing else."
    )

    payload = {
        "model": VLLM_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 20,
        "temperature": 0.2,
        "stop": ["\n"]
    }

    try:
        response = requests.post(VLLM_API_URL, headers={"Content-Type": "application/json"}, data=json.dumps(payload))
        response.raise_for_status()
        
        data = response.json()
        class_name = data['choices'][0]['message']['content'].strip()
        class_name = class_name.replace('"', '').replace("'", "").replace("Class Name: ", "")
        
        print(f"    > [Cluster {cluster_index}] VLLM generated: {class_name}")
        return class_name

    except requests.exceptions.RequestException as e:
        print(f"  [!] ERROR contacting VLLM API: {e}")
        print(f"  [!] Using fallback for terms: {terms_str}")
        verb = top_terms[0]
        noun = top_terms[1] if len(top_terms) > 1 else ""
        if verb.lower() == noun.lower() and len(top_terms) > 2:
            noun = top_terms[2]
        return f"{verb} {noun}".strip()

# ---------------------------------------------------------

def main_stage2():
    print("\n🔵 [STAGE 2] Starting class naming with VLLM...")
    print("          (Assuming VLLM is already loaded and running.)")

    # ---------- Load State from Stage 1 ----------
    print(f"🔵 [STAGE 2] Loading data from: {TEMP_DF_PATH}")
    try:
        df_labels = pd.read_csv(TEMP_DF_PATH)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {TEMP_DF_PATH}")
        print("Did you run '2025_11_24_create_challenge_classes_sentence_transformers_part_1.py' first?")
        return

    print(f"🔵 [STAGE 2] Loading terms from: {TEMP_TERMS_PATH}")
    try:
        with open(TEMP_TERMS_PATH, 'r') as f:
            top_terms_per_cluster = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {TEMP_TERMS_PATH}")
        print("Did you run '2025_11_24_create_challenge_classes_sentence_transformers_part_1.py' first?")
        return

    # ---------- Class Names (via VLLM) ----------
    print("\n🔵 [STAGE 2] Generating class names via VLLM API…")
    class_names = [make_class_name(terms, i) for i, terms in enumerate(top_terms_per_cluster)]
    df_labels["class_name"] = df_labels["cluster"].map(dict(enumerate(class_names)))
    print("🔵 [STAGE 2] Class names applied.")


    # ---------- Sampling 500 (balanced) ----------
    print("\n🔵 [STAGE 2] Sampling 500 balanced examples…")
    total = SAMPLE_SIZE
    per_cluster = total // N_CLUSTERS
    rng = np.random.RandomState(RANDOM_SEED)
    sampled_indices = []

    for c in range(N_CLUSTERS):
        idxs = df_labels[df_labels.cluster == c].index.tolist()
        rng.shuffle(idxs)
        sampled_indices.extend(idxs[:per_cluster])

    remaining = total - len(sampled_indices)
    if remaining > 0:
        pool = list(set(df_labels.index) - set(sampled_indices))
        rng.shuffle(pool)
        sampled_indices.extend(pool[:remaining])

    sampled_df = df_labels.loc[sampled_indices].reset_index(drop=True)

    # ---------- Save Final Files ----------
    df_labels.to_csv(FULL_PATH, index=False)
    sampled_df.to_csv(SAMPLE_PATH, index=False)

    summary = []
    for i in range(N_CLUSTERS):
        summary.append({
            "cluster": i,
            "class_name": class_names[i],
            "top_terms": ", ".join(top_terms_per_cluster[i]),
            "n_labels": int((df_labels.cluster == i).sum())
        })
    pd.DataFrame(summary).to_csv(SUMMARY_PATH, index=False)

    print("\n✅ [STAGE 2] DONE!")
    print("Full classification:", FULL_PATH)
    print("Sample of 500:", SAMPLE_PATH)
    print("Summary:", SUMMARY_PATH)

    # --- Optional: Clean up temporary files ---
    # print("\n🔵 [STAGE 2] Cleaning up temporary files...")
    # os.remove(TEMP_DF_PATH)
    # os.remove(TEMP_TERMS_PATH)
    # ---------------------------------------------


if __name__ == "__main__":
    main_stage2()
