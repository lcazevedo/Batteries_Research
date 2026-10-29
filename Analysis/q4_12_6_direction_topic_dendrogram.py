from networkx import display
import utils
import config

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import textwrap
from matplotlib.ticker import MaxNLocator

# Load data
df_data = utils.get_df_data()
df_data_phrases = utils.get_df_data_phrases()

df_analysis = df_data_phrases.merge(
    df_data[[config.COLUMN_DOI, config.COLUMN_TOPIC_NAME]],
    left_on='doi',
    right_on=config.COLUMN_DOI,
    how='left'
)
direction_col = "directions_group"
topic_col = config.COLUMN_TOPIC_NAME

TOP_N_TOPICS = 50
TOP_N_DIRECTIONS = 15

fs1 = 70
fs2 = 50
fs3 = 37
fs4 = 50




# --- 1. DATA PREPARATION AND SORTING ---
df_filtered = df_analysis.dropna(subset=[direction_col, topic_col])
df_filtered = df_filtered[df_filtered[topic_col] != "UNKNOWN"]
df_filtered = df_filtered[
    ~df_filtered[direction_col].isin(['Uncategorized', 'No Research Direction'])
]
top_topics_sorted = df_filtered[topic_col].value_counts().head(TOP_N_TOPICS).index
top_directions_sorted = df_filtered[direction_col].value_counts().head(TOP_N_DIRECTIONS).index

df_heatmap = df_filtered[
    (df_filtered[topic_col].isin(top_topics_sorted)) &
    (df_filtered[direction_col].isin(top_directions_sorted))
]

# --- 2. CONTINGENCY TABLE CREATION ---
topic_direction_crosstab = pd.crosstab(
    df_heatmap[topic_col],
    df_heatmap[direction_col]
)
# Reorder rows and columns for consistency
topic_direction_crosstab = topic_direction_crosstab.reindex(
    index=top_topics_sorted.intersection(topic_direction_crosstab.index),
    columns=top_directions_sorted.intersection(topic_direction_crosstab.columns)
)

print("\n--- Count Table (Original) ---")
print(topic_direction_crosstab.head())

uncategorized_col_name = 'Uncategorized' # Verify exact column name
if uncategorized_col_name in topic_direction_crosstab.columns:
    topic_direction_crosstab = topic_direction_crosstab.drop(columns=[uncategorized_col_name])
    print(f"\n--- Column '{uncategorized_col_name}' removed from analysis ---")

##################################################################
#### HIERARCHICAL CLUSTERING
##################################################################

print("\n--- Starting Hierarchical Clustering ---")

# --- STEP 1: PROFILE NORMALIZATION (BY ROW) ---
# This removes the volume effect (hegemony) and focuses on the "research profile".
# Divide each direction count for a topic by the total directions for that topic.

# Calculate row total (topic)
row_totals = topic_direction_crosstab.sum(axis=1)

# Filter out topics that may have 0 directions (avoids division by zero)
topics_with_data = row_totals[row_totals > 0].index
topic_direction_crosstab_filtered = topic_direction_crosstab.loc[topics_with_data]
row_totals_filtered = row_totals.loc[topics_with_data]

# Divide cell count by row total to obtain proportion
df_profile = topic_direction_crosstab_filtered.div(row_totals_filtered, axis=0)

# Fill any resulting NaN values with 0
df_profile = df_profile.fillna(0)

print("\n--- Profile Table (Row-Normalized) ---")
print(df_profile.head())

# --- STEP 2: GENERATE CLUSTERMAP ---
# This creates a heatmap of the profiles and adds dendrograms
# that group topics (rows) and directions (columns) by similarity.

print("\n--- Generating Clustermap (Heatmap + Dendrogram) ---")

# Adjust figsize to fit the number of labels
# method='ward' is a robust clustering technique
# metric='euclidean' is the standard distance metric
# cmap='YlGnBu' is a good color palette for proportions (0 to 1)

#cbar_position_horizontal = (0.25, 0.08, 0.5, 0.03)
cbar_position_horizontal = (0.80, 0.95, 0.5, 0.03)

g = sns.clustermap(
    df_profile,
    method='ward',
    metric='euclidean',
    cmap='YlGnBu',  # Good palette for profiles (low to high)
    figsize=(20, 26),
    annot=False,  # Annotations would be too cluttered in a dense chart
    #cbar_kws={'label': 'Proportion of the Topic’s Research Effort'}
    cbar_pos=cbar_position_horizontal,
    cbar_kws={'orientation': 'horizontal'}
)
g.cax.set_xlabel('Proportion of the Topic’s\nResearch Effort', fontsize=fs2)
g.cax.tick_params(labelsize=fs3)

# Add titles and adjust labels
g.figure.suptitle('Hierarchical Clustering of Topics by Research Directions Profile', y=1.05, fontsize=fs1)
g.ax_heatmap.set_xlabel('Direction Category', fontsize=fs2)
g.ax_heatmap.set_ylabel('Topic', fontsize=fs2)
plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=fs4)
plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=fs4)

# Save figure
cluster_filename = f'{config.RESULTS_FOLDER}/q4_12_6_direction_topic_dendrogram.pdf'
plt.savefig(cluster_filename, dpi=300, bbox_inches='tight')
print(f"Clustermap saved to: {cluster_filename}")

####################
# Save data to CSV for statistical analysis
csv_filename = f'{config.RESULTS_FOLDER}/q4_12_6_direction_topic_dendrogram.csv'
df_profile.to_csv(csv_filename)
