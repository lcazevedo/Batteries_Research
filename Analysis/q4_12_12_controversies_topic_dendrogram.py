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
controversies_col = "controversies_group"
topic_col = config.COLUMN_TOPIC_NAME

TOP_N_TOPICS = 50
TOP_N_CONTROVERSIES = 15

fs1 = 74
fs2 = 55
fs3 = 45
fs4 = 55


# Format chemical formulas
import re
def chemical_subscript(text):
    """
    Converts numbers into subscripts using matplotlib mathtext.
    E.g.: MnO2 -> MnO$_2$
    """
    return re.sub(r'([A-Za-z\)])(\d+)', r'\1$_{\2}$', text)

# --- 1. DATA PREPARATION AND SORTING ---
df_filtered = df_analysis.dropna(subset=[controversies_col, topic_col])
df_filtered = df_filtered[df_filtered[topic_col] != "UNKNOWN"]
df_filtered = df_filtered[
    ~df_filtered[controversies_col].isin(['Uncategorized', 'No Controversy'])
]  

top_topics_sorted = df_filtered[topic_col].value_counts().head(TOP_N_TOPICS).index
top_controversies_sorted = df_filtered[controversies_col].value_counts().head(TOP_N_CONTROVERSIES).index

df_heatmap = df_filtered[
    (df_filtered[topic_col].isin(top_topics_sorted)) &
    (df_filtered[controversies_col].isin(top_controversies_sorted))
]

# --- 2. CONTINGENCY TABLE CREATION ---
topic_controversies_crosstab = pd.crosstab(
    df_heatmap[topic_col],
    df_heatmap[controversies_col]
)
# Reorder rows and columns for consistency
topic_controversies_crosstab = topic_controversies_crosstab.reindex(
    index=top_topics_sorted.intersection(topic_controversies_crosstab.index),
    columns=top_controversies_sorted.intersection(topic_controversies_crosstab.columns)
)

print("\n--- Count Table (Original) ---")
print(topic_controversies_crosstab.head())

uncategorized_col_name = 'Uncategorized' # Verify exact column name
if uncategorized_col_name in topic_controversies_crosstab.columns:
    topic_controversies_crosstab = topic_controversies_crosstab.drop(columns=[uncategorized_col_name])
    print(f"\n--- Column '{uncategorized_col_name}' removed from analysis ---")

##################################################################
#### HIERARCHICAL CLUSTERING
##################################################################

print("\n--- Starting Hierarchical Clustering ---")

# --- STEP 1: PROFILE NORMALIZATION (BY ROW) ---
# This removes the volume effect (hegemony) and focuses on the "research profile".
# Divide each controversy count for a topic by the total controversies for that topic.

# Calculate row total (topic)
row_totals = topic_controversies_crosstab.sum(axis=1)

# Filter out topics that may have 0 controversies (avoids division by zero)
topics_with_data = row_totals[row_totals > 0].index
topic_controversies_crosstab_filtered = topic_controversies_crosstab.loc[topics_with_data]
row_totals_filtered = row_totals.loc[topics_with_data]

# Divide cell count by row total to obtain proportion
df_profile = topic_controversies_crosstab_filtered.div(row_totals_filtered, axis=0)

# Fill any resulting NaN values with 0
df_profile = df_profile.fillna(0)

print("\n--- Profile Table (Row-Normalized) ---")
print(df_profile.head())

# --- STEP 2: GENERATE CLUSTERMAP ---
# This creates a heatmap of the profiles and adds dendrograms
# that group topics (rows) and controversies (columns) by similarity.

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
    figsize=(18, 30),
    annot=False,  # Annotations would be too cluttered in a dense chart
    #cbar_kws={'label': 'Proportion of the Topic’s Research Effort'}
    cbar_pos=cbar_position_horizontal,
    cbar_kws={'orientation': 'horizontal'}
)
g.cax.set_xlabel('Proportion of the Topic’s\nResearch Effort', fontsize=fs2)
g.cax.tick_params(labelsize=fs3)

# Add titles and adjust labels
g.figure.suptitle('Hierarchical Clustering of Topics by Research Controversies Profile', y=1.05, fontsize=fs1)
g.ax_heatmap.set_xlabel('Controversies Category', fontsize=fs2)
g.ax_heatmap.set_ylabel('Topic', fontsize=fs2)
plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=fs4)
#plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=fs4)


# --- Adjust X-axis labels (Controversies) with chemical subscripts ---
xticklabels = [label.get_text() for label in g.ax_heatmap.get_xticklabels()]
xticklabels = [chemical_subscript(label) for label in xticklabels]

g.ax_heatmap.set_xticklabels(
    xticklabels,
    rotation=55,
    ha='right',
    fontsize=fs4
)

# Y-axis labels (topics) remain unchanged
plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=fs4)

# Save figure
cluster_filename = f'{config.RESULTS_FOLDER}/q4_12_12_controversies_topic_dendrogram.pdf'
plt.savefig(cluster_filename, dpi=300, bbox_inches='tight')
print(f"Clustermap saved to: {cluster_filename}")

####################
# Save data to CSV for statistical analysis
csv_filename = f'{config.RESULTS_FOLDER}/q4_12_12_controversies_topic_dendrogram_data.csv'
df_profile.to_csv(csv_filename)


# ====================================================================
# --- STEP 3: EXPORT FULL TABLES FOR THE LATEX APPENDIX ---
# ====================================================================
print("\n--- Exporting normalized matrix and legend to LaTeX format ---")

latex_filename = f'{config.RESULTS_FOLDER}/q4_12_12_controversies_topic_dendrogram_data.tex'

# Make a copy of the original dataframe
df_profile_latex = df_profile.copy()

# ====================================================================
# Synchronize table order with dendrogram order
# ====================================================================
# Extract final order computed by clustermap (object 'g')
ordem_linhas = g.dendrogram_row.reordered_ind
ordem_colunas = g.dendrogram_col.reordered_ind

# Reorder dataframe to match the chart
df_profile_latex = df_profile_latex.iloc[ordem_linhas, ordem_colunas]
# ====================================================================

# Escape special LaTeX characters in row names (topics)
df_profile_latex.index = df_profile_latex.index.str.replace('&', r'\&', regex=False)
df_profile_latex.index = df_profile_latex.index.str.replace('_', r'\_', regex=False)

# Escape special characters in original column names in the NEW order
original_columns = df_profile_latex.columns.str.replace('&', r'\&', regex=False).str.replace('_', r'\_', regex=False)

# Identify number of columns and create C1, C2... labels
num_cols = len(df_profile_latex.columns)
c_labels = [f"C{i+1}" for i in range(num_cols)]

with open(latex_filename, 'w', encoding='utf-8') as f:
    # --- START LANDSCAPE ENVIRONMENT ---
    f.write("\\begin{landscape}\n")
    f.write("\\setlength{\\tabcolsep}{4pt}\n")
    f.write("\\small\n\n")

    # ==========================================
    # TABLE 1: FREQUENCY MATRIX
    # ==========================================
    col_def = "l*{" + str(num_cols) + "}{r}"
    f.write(f"\\begin{{longtable}}{{{col_def}}}\n")
    f.write("\\caption{Relative frequency matrix of scientific controversies distributed by research topic.}\\label{tbl:baterias_matriz_controversias} \\\\\n")
    f.write("\\toprule\n")
    
    # Header (Page 1)
    header_row = "\\textbf{Research Topic} & " + " & ".join([f"\\textbf{{{c}}}" for c in c_labels]) + " \\\\\n"
    f.write(header_row)
    f.write("\\midrule\n")
    f.write("\\endfirsthead\n\n")
    
    # Header (Continuation pages)
    f.write("\\toprule\n")
    header_row_cont = "\\textbf{Research Topic} (Cont.) & " + " & ".join([f"\\textbf{{{c}}}" for c in c_labels]) + " \\\\\n"
    f.write(header_row_cont)
    f.write("\\midrule\n")
    f.write("\\endhead\n\n")
    
    # Footer (Intermediate pages)
    f.write("\\midrule\n")
    f.write(f"\\multicolumn{{{num_cols + 1}}}{{r}}{{\\textit{{Continued on next page}}}} \\\\\n")
    f.write("\\bottomrule\n")
    f.write("\\endfoot\n\n")
    
    # Footer (Last page)
    f.write("\\bottomrule\n")
    f.write("\\endlastfoot\n\n")

    # Insert matrix data
    for index, row in df_profile_latex.iterrows():
        # Format with 4 decimal places
        formatted_values = [f"{val:.4f}" for val in row.values]
        linha_latex = f"{index} & " + " & ".join(formatted_values) + r" \\"
        f.write(linha_latex + "\n")
        
    f.write("\n\\end{longtable}\n\n")

    # --- SEPARATOR SPACE BETWEEN TABLES ---
    f.write("\\vspace{2em}\n\n")

    # ==========================================
    # TABLE 2: CONTROVERSIES LEGEND
    # ==========================================
    f.write("\\begin{longtable}{l p{18cm}}\n")
    f.write("\\caption{Description of the controversy category codes used in the previous matrix.}\\label{tbl:legenda_controversias} \\\\\n")
    f.write("\\toprule\n")
    f.write("\\textbf{Code} & \\textbf{Debate / Controversy Description} \\\\\n")
    f.write("\\midrule\n")
    f.write("\\endfirsthead\n\n")
    
    f.write("\\toprule\n")
    f.write("\\textbf{Code} & \\textbf{Debate / Controversy Description (Cont.)} \\\\\n")
    f.write("\\midrule\n")
    f.write("\\endhead\n\n")
    
    f.write("\\bottomrule\n")
    f.write("\\endfoot\n\n")
    f.write("\\bottomrule\n")
    f.write("\\endlastfoot\n\n")

    # Insert ordered codes and original descriptions
    for sigla, desc in zip(c_labels, original_columns):
        f.write(f"\\textbf{{{sigla}}} & {desc} \\\\\n")
        
    f.write("\n\\end{longtable}\n\n")

    # --- END LANDSCAPE ENVIRONMENT ---
    f.write("\\end{landscape}\n")

print(f"Complete LaTeX file successfully generated at: {latex_filename}")
print("Open the generated file, copy its contents, and paste into the appendix.")