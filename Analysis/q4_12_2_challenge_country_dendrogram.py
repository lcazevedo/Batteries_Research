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
    df_data[[config.COLUMN_DOI, config.COLUMN_FIRST_COUNTRY]],
    left_on='doi',
    right_on=config.COLUMN_DOI,
    how='left'
)
challenge_col = "challenges_group"
country_col = config.COLUMN_FIRST_COUNTRY

TOP_N_COUNTRIES = 20
TOP_N_CHALLENGES = 15

fs1 = 70
fs2 = 50
fs3 = 35
fs4 = 50




# --- 1. DATA PREPARATION AND SORTING ---
df_filtered = df_analysis.dropna(subset=[challenge_col, country_col])
df_filtered = df_filtered[df_filtered[country_col] != "UNKNOWN"]
df_filtered = df_filtered[
    ~df_filtered[challenge_col].isin(['Uncategorized', 'No Challenge'])
]
top_countries_sorted = df_filtered[country_col].value_counts().head(TOP_N_COUNTRIES).index
top_challenges_sorted = df_filtered[challenge_col].value_counts().head(TOP_N_CHALLENGES).index

df_heatmap = df_filtered[
    (df_filtered[country_col].isin(top_countries_sorted)) &
    (df_filtered[challenge_col].isin(top_challenges_sorted))
]

# --- 2. CONTINGENCY TABLE CREATION ---
country_challenge_crosstab = pd.crosstab(
    df_heatmap[country_col],
    df_heatmap[challenge_col]
)
# Reorder rows and columns for consistency
country_challenge_crosstab = country_challenge_crosstab.reindex(
    index=top_countries_sorted.intersection(country_challenge_crosstab.index),
    columns=top_challenges_sorted.intersection(country_challenge_crosstab.columns)
)

print("\n--- Count Table (Original) ---")
print(country_challenge_crosstab.head())

uncategorized_col_name = 'Uncategorized' # Verify exact column name
if uncategorized_col_name in country_challenge_crosstab.columns:
    country_challenge_crosstab = country_challenge_crosstab.drop(columns=[uncategorized_col_name])
    print(f"\n--- Column '{uncategorized_col_name}' removed from analysis ---")

##################################################################
#### HIERARCHICAL CLUSTERING
##################################################################

print("\n--- Starting Hierarchical Clustering ---")

# --- STEP 1: PROFILE NORMALIZATION (BY ROW) ---
# This removes the volume effect (hegemony) and focuses on the "research profile".
# Divide each challenge count for a country by the total challenges for that country.

# Calculate row total (country)
row_totals = country_challenge_crosstab.sum(axis=1)

# Filter out countries that may have 0 challenges (avoids division by zero)
countries_with_data = row_totals[row_totals > 0].index
country_challenge_crosstab_filtered = country_challenge_crosstab.loc[countries_with_data]
row_totals_filtered = row_totals.loc[countries_with_data]

# Divide cell count by row total to obtain proportion
df_profile = country_challenge_crosstab_filtered.div(row_totals_filtered, axis=0)

# Fill any resulting NaN values with 0
df_profile = df_profile.fillna(0)

print("\n--- Profile Table (Row-Normalized) ---")
print(df_profile.head())

# --- STEP 2: GENERATE CLUSTERMAP ---
# This creates a heatmap of the profiles and adds dendrograms
# that group countries (rows) and challenges (columns) by similarity.

print("\n--- Generating Clustermap (Heatmap + Dendrogram) ---")

# Adjust figsize to fit the number of labels
# method='ward' is a robust clustering technique
# metric='euclidean' is the standard distance metric
# cmap='YlGnBu' is a good color palette for proportions (0 to 1)

#cbar_position_horizontal = (0.25, 0.08, 0.5, 0.03)
cbar_position_horizontal = (0.70, 0.95, 0.5, 0.03)

g = sns.clustermap(
    df_profile,
    method='ward',
    metric='euclidean',
    cmap='YlGnBu',  # Good palette for profiles (low to high)
    figsize=(20, 22),
    annot=False,  # Annotations would be too cluttered in a dense chart
    #cbar_kws={'label': 'Proportion of the Country’s Research Effort'}
    cbar_pos=cbar_position_horizontal,
    cbar_kws={'orientation': 'horizontal'}
)
g.cax.set_xlabel('Proportion of the Country’s\nResearch Effort', fontsize=fs2)
g.cax.tick_params(labelsize=fs3)

# Add titles and adjust labels
g.figure.suptitle('Hierarchical Clustering of Countries by Challenge Profile', y=1.05, fontsize=fs1)
g.ax_heatmap.set_xlabel('Challenge Category', fontsize=fs2)
g.ax_heatmap.set_ylabel('Country', fontsize=fs2)
plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=fs4)
plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=fs4)

# Save figure
cluster_filename = f'{config.RESULTS_FOLDER}/q4_12_2_challenge_country_dendrogram.pdf'
plt.savefig(cluster_filename, dpi=300, bbox_inches='tight')
print(f"Clustermap saved to: {cluster_filename}")

####################
# Save data to CSV for statistical analysis
csv_filename = f'{config.RESULTS_FOLDER}/q4_12_2_challenge_country_dendrogram_data.csv'
df_profile.to_csv(csv_filename)

