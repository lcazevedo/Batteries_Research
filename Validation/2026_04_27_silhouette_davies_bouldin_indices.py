import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score
from joblib import parallel_backend
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from kneed import KneeLocator

# Path configuration
import config; project_path = os.path.join(config.DATA_FOLDER, "Validation_and_Audit") + "/"
suffix = "" 

# =========================
# CACHE CONTROL
# =========================
RECOMPUTE_METRICS = False  # <<<<<< CHANGE HERE

CACHE_PATH = "/tmp/evoc_cluster_metrics_batteries.pkl"



print("🚀 Loading EVoC pipeline data...")

# 1. Load Global Embeddings
with open(f'{project_path}embs_pred{suffix}.pkl', 'rb') as h:
    embs = pickle.load(h)

# 2. Load topic information (DataFrames with names/descriptions)
with open(f'{project_path}topic_infos{suffix}.pkl', 'rb') as h:
    topic_infos = pickle.load(h)

# 3. Load labels for each layer (Integer arrays corresponding to embs)
with open(f'{project_path}current_topics_infos{suffix}.pkl', 'rb') as h:
    current_topics_infos = pickle.load(h)


# =========================
# CACHE LOAD (if exists)
# =========================
if (not RECOMPUTE_METRICS) and os.path.exists(CACHE_PATH):
    print("⚡ Loading metrics from cache...")

    with open(CACHE_PATH, "rb") as f:
        final_metrics = pickle.load(f)

else:
    print("🚀 Recomputing metrics...")

    final_metrics = []

    print(f"Identified {len(current_topics_infos)} clustering layers.")

    for i in range(len(current_topics_infos)):
        print(f"\n📊 Processing Clustering Layer {i}...")

        labels = current_topics_infos[i]

        embs_filtered = embs
        labels_filtered = labels

        n_clusters = len(np.unique(labels_filtered))
        print(f"   -> Valid documents: {len(labels_filtered)}")
        print(f"   -> Number of topics in this layer: {n_clusters}")

        if n_clusters > 1:
            with parallel_backend('threading', n_jobs=16):

                sil = silhouette_score(
                    embs_filtered, 
                    labels_filtered, 
                    sample_size=65000, 
                    n_jobs=16, 
                    random_state=42
                )

                db = davies_bouldin_score(
                    embs_filtered,
                    labels_filtered
                )

                final_metrics.append({
                    "Layer": i,
                    "N_Topics": n_clusters,
                    "Silhouette_20k": round(sil, 4),
                    "Davies_Bouldin": round(db, 4)
                })

        else:
            print(f"   ⚠️ Skipping layer {i}: Insufficient clusters.")

    # =========================
    # SAVE CACHE
    # =========================
    print("\n💾 Saving metrics to cache...")

    with open(CACHE_PATH, "wb") as f:
        pickle.dump(final_metrics, f)


# Display Results Table
df_results = pd.DataFrame(final_metrics)
print("\n" + "="*50)
print("LAYER QUALITY REPORT (EVoC)")
print("="*50)
print(df_results.to_string(index=False))
print("="*50)

# Save results
file_path = f"{project_path}layer_quality_metrics.csv"
df_results.to_csv(file_path, index=False)

# 1. Read previously generated data
df_results = pd.read_csv(file_path)

# 2. Davies-Bouldin Normalization
db_min = df_results['Davies_Bouldin'].min()
db_max = df_results['Davies_Bouldin'].max()

df_results['DB_norm'] = (
    (df_results['Davies_Bouldin'] - db_min)
    / (db_max - db_min + 1e-9)
)

# 3. Empirical Silhouette Normalization for CVI2
sil_min = df_results['Silhouette_20k'].min()
sil_max = df_results['Silhouette_20k'].max()

df_results['S_norm'] = (
    (df_results['Silhouette_20k'] - sil_min)
    / (sil_max - sil_min + 1e-9)
)

# 3. Theoretical Silhouette Normalization
# Maps the interval [-1, 1] to [0, 1]
df_results['S_norm_v2'] = (
    df_results['Silhouette_20k'] + 1
) / 2


# =========================
# QUALITY SCORE Q
# =========================
df_results['Q_score'] = (
    df_results['S_norm'] * (1 - df_results['DB_norm'])
)
# avoid log(0)
eps = 1e-9
x = np.log(df_results['N_Topics'] + eps)
y = np.log(df_results['Q_score'] + eps)
X = x.values.reshape(-1, 1)
y = y.values
model = LinearRegression()
model.fit(X, y)
alpha_hat = -model.coef_[0]
print(f"\n📌 Automatically estimated alpha: {alpha_hat:.4f}")


# 4. Original CVI calculation
# df_results['CVI_Score'] = (
#     df_results['Silhouette_20k']
#     * (1 - df_results['DB_norm'])
# ) / np.log(df_results['N_Topics'])

# 5. CVI2 calculation
# df_results['CVI2_Score'] = (
#     df_results['S_norm']
#     * (1 - df_results['DB_norm'])
# ) / np.sqrt(df_results['N_Topics'])
df_results['CVI2_v2_Score'] = (
    df_results['S_norm_v2']
    * (1 - df_results['DB_norm'])
) / np.sqrt(df_results['N_Topics'])
# alpha = 0.25
# df_results['CVI2_alpha'] = (
#     df_results['S_norm']
#     * (1 - df_results['DB_norm'])
# ) / (df_results['N_Topics'] ** alpha)
# df_results['CVI_ensemble'] = (
#     0.4 * df_results['CVI_Score'] +
#     0.3 * df_results['CVI2_Score'] +
#     0.3 * df_results['CVI2_v2_Score']
# )
df_results['CVI_adaptive'] = (
    df_results['Q_score'] /
    (df_results['N_Topics'] ** alpha_hat)
)

# 6. Marginal Gain Calculation (Cluster Gain / ROI)
df_results['Gain_Silhouette'] = (
    df_results['Silhouette_20k'].diff()
    / df_results['N_Topics'].diff()
)

df_results['Gain_DB'] = (
    df_results['Davies_Bouldin'].diff()
    / df_results['N_Topics'].diff()
)


# # Check across alpha values
# for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
#     df_results[f'CVI_alpha_{alpha}'] = (
#         df_results['S_norm']
#         * (1 - df_results['DB_norm'])
#     ) / (df_results['N_Topics'] ** alpha)


# 7. Drop auxiliary columns and save
df_final = df_results.drop(columns=['DB_norm', 'S_norm', 'S_norm_v2'])


# Update CSV
df_final.to_csv(file_path, index=False)

# Formatted display
print("\n" + "="*80)
print("PARSIMONY AND MARGINAL GAIN ANALYSIS")
print("="*80)
print(df_final.to_string(index=False))
print("="*80)

# Interpretation suggestion:
# best_layer_cvi = df_final.loc[df_final['CVI_Score'].idxmax(), 'Layer']
# best_layer_cvi2 = df_final.loc[df_final['CVI2_Score'].idxmax(), 'Layer']
# best_layer_cvi2_alpha = df_final.loc[df_final['CVI2_alpha'].idxmax(), 'Layer']
# best_layer_cvi_ensemble = df_final.loc[df_final['CVI_ensemble'].idxmax(), 'Layer']

best_layer_cvi2_v2 = df_final.loc[df_final['CVI2_v2_Score'].idxmax(), 'Layer']
best_layer_cvi_adaptative = df_final.loc[df_final['CVI_adaptive'].idxmax(), 'Layer']

# print(
#     f"\n💡 Suggestion based on CVI: "
#     f"Layer {int(best_layer_cvi)} presents the best balance between cohesion and complexity."
# )

# print(
#     f"💡 Suggestion based on CVI2: "
#     f"Layer {int(best_layer_cvi2)} presents the best balance between cohesion and complexity."
# )

print(
    f"💡 Suggestion based on CVI2_v2: "
    f"Layer {int(best_layer_cvi2_v2)} presents the best balance between cohesion and complexity."
)

# print(
#     f"💡 Suggestion based on CVI2_alpha: "
#     f"Layer {int(best_layer_cvi2_alpha)} presents the best balance between cohesion and complexity."
# )

# print(
#     f"💡 Suggestion based on CVI_ensemble: "
#     f"Layer {int(best_layer_cvi_ensemble)} presents the best balance between cohesion and complexity."
# )

print(
    f"💡 Suggestion based on CVI_adaptive: "
    f"Layer {int(best_layer_cvi_adaptative)} presents the best balance between cohesion and complexity."
)



k = df_results['N_Topics'].values
q = df_results['Q_score'].values

kneedle = KneeLocator(
    k,
    q,
    curve='concave',
    direction='increasing'
)

x = kneedle.plot_knee()

print("Knee:", kneedle.knee)
if kneedle.knee is not None:
    best_layer_kneedle = df_results.loc[
        df_results['N_Topics'] == kneedle.knee,
        'Layer'
    ].iloc[0]

    print(
        f"💡 Suggestion based on Kneedle: "
        f"Layer {int(best_layer_kneedle)} presents the best balance between cohesion and complexity."
    )
else:
    print(
        "⚠️ Kneedle did not find a knee in the curve."
    )
