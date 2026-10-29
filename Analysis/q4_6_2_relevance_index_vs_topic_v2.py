import utils
import config

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import textwrap

from matplotlib.patches import Ellipse
import numpy as np



# Load data
df_data = utils.get_df_data()


# --- 1. Data Configuration ---

# Initial year for filtering
START_YEAR = 2014


fs1 = 26
fs2 = 24
fs3 = 20



# Ensure the year column is numeric, handling errors
df_data[config.COLUMN_YEAR] = pd.to_numeric(df_data[config.COLUMN_YEAR], errors='coerce')

# Remove rows where essential data is missing
df_data = df_data.dropna(subset=[config.COLUMN_YEAR, config.COLUMN_TOPIC_NAME, config.COLUMN_TOPIC_RANK_INDEX_WO_REVIEW])

# Convert year to integer
df_data[config.COLUMN_YEAR] = df_data[config.COLUMN_YEAR].astype(int)

print("Data loaded. Processing...")

# Filter dataframe by years ( >= START_YEAR)
df_filtered = df_data[df_data[config.COLUMN_YEAR] >= START_YEAR].copy()

# 1. Calculate total document count (X-axis)
# Group by topic and count the number of publications
doc_counts = df_filtered.groupby(config.COLUMN_TOPIC_NAME).size().reset_index(name="document_count")

# 2. Obtain relevance value (Y-axis)
# Get the unique relevance value for each topic
relevance_df = df_data[[config.COLUMN_TOPIC_NAME, config.COLUMN_TOPIC_RANK_INDEX_WO_REVIEW]].drop_duplicates()

# 3. Combine data into a single dataframe
# Merge document counts with relevance values
topic_indexes = pd.merge(doc_counts, relevance_df, on=config.COLUMN_TOPIC_NAME, how="left")

# 4. Rename columns to match plotting code
topic_indexes = topic_indexes.rename(columns={
    config.COLUMN_TOPIC_NAME: "topic",
    config.COLUMN_TOPIC_RANK_INDEX_WO_REVIEW: "relevance"
})

# Ensure there are no null values in relevance (optional, but safe)
topic_indexes["relevance"] = topic_indexes["relevance"].fillna(0)

print("Dataframe 'topic_indexes' created successfully. Generating chart...")
# print(topic_indexes.head()) # Uncomment to inspect dataframe


# --- 3. Chart Generation (Original Code) ---
# This section is the plotting code, now using the prepared data.

# Generate "ranking" (raw values, as in the original code)
topic_indexes['rank1'] = topic_indexes['document_count']
topic_indexes['rank2'] = topic_indexes['relevance']

# Calculate ranking midpoints (for label display logic)
rank1_half = (topic_indexes['rank1'].max() / 2) * 1.5
rank2_half = (topic_indexes['rank2'].max() / 2) * 1.5

# Generate a unique color for each topic
topics = topic_indexes['topic'].unique()
colors = cm.get_cmap('tab20', len(topics)) # 'tab20' is a good colormap for many topics
color_dict = {topic: colors(i) for i, topic in enumerate(topics)}

# Configure white background
plt.style.use('default')
fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
ax.set_facecolor('white')

alto = []
direita = []
outros = []
coords_alto = []
coords_direita = []
coords_outros = []
# Plot points and conditional labels
for i, row in topic_indexes.iterrows():
    x, y = row['rank1'], row['rank2']
    color = color_dict[row['topic']]
    ax.scatter(x, y, color=color, s=100) # s=100 is the marker size


    if x > rank1_half:
        # Wrap label text across multiple lines
        label = "\n".join(textwrap.wrap(str(row['topic']), 12))
        direita.append(label)
        coords_direita.append((x, y))

        # ax.annotate(
        #     label,
        #     (x, y),
        #     xytext=(0, -10),  # vertical offset of 10 pixels downward
        #     textcoords='offset points',
        #     ha='center',
        #     va='top',
        #     fontsize=fs3,
        #     color="black"
        # )

    if y > 7: #rank2_half:
        # Wrap label text across multiple lines
        label =  "\n".join(textwrap.wrap(str(row['topic']), 40))
        alto.append(label)
        coords_alto.append((x, y))

        ax.annotate(
            label,
            (x, y),
            xytext=(0, -10),  # vertical offset of 10 pixels downward
            textcoords='offset points',
            ha='left',
            va='top',
            fontsize=fs3,
            color="black"
        )

    if x < rank1_half and y < 7:
        label = str(row['topic'])
        outros.append(label)
        coords_outros.append((x, y))

# Force axes to start at zero
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)

# Axes and title
ax.set_xlabel('Document Count', fontsize=fs2)
ax.set_ylabel('Relevance', fontsize=fs2)
ax.set_title('Comparison Between Relevance and\nQuantity of Documents per Topic', fontsize=fs1)
plt.xticks(fontsize=fs2, rotation=45)
plt.yticks(fontsize=fs2)
ax.grid(True)


def draw_ellipse(ax, coords, color, alpha=0.15, scale=1.3, label=None, y_mean_factor = 1):
    if len(coords) < 2:
        return
    
    coords = np.array(coords)
    x = coords[:, 0]
    y = coords[:, 1]
    
    # Center
    x_mean = x.mean()
    y_mean = y.mean()
    
    # Dimensions
    width = (x.max() - x.min()) * scale
    height = (y.max() - y.min()) * scale
    
    ellipse = Ellipse(
        (x_mean, y_mean),
        width=width,
        height=height,
        facecolor=color,
        edgecolor=color,
        alpha=alpha,
        zorder=0
    )
    
    ax.add_patch(ellipse)

    # Add label
    if label:
        ax.text(
            x_mean,
            (y_mean - height/2 - 0.1 * height)*y_mean_factor,  # below the ellipse
            label,
            ha='center',
            va='center',
            fontsize=fs3,
            color=color,
            alpha=0.8
        )

#draw_ellipse(ax, coords_alto, 'red', scale = 1.8)
draw_ellipse(ax, coords_direita, 'blue', scale = 1.6, label="Mainstream Li-ion\nResearch", y_mean_factor=0.9)
draw_ellipse(ax, coords_outros, 'green', scale = 1.5, label="Specialized & Emerging Chemistries", y_mean_factor=1.1)    


# Adjust layout to prevent labels from being cut off
plt.tight_layout()


# To save the chart, uncomment the line below
fig.savefig(f'{config.RESULTS_FOLDER}/q4_6_2_relevance_index_vs_topic.pdf', dpi=300, bbox_inches='tight')
print("Chart saved as 'q4_6_2_relevance_index_vs_topic.pdf'")

print("\n\nHigh:")
print(alto)
print("\n\nRight:")
print(direita)
print("\n\nOthers:")
print(outros)

################
# Suggested caption based on the divisions obtained:
    # \caption{Scatter plot comparing the Relevance index and Document Count per research topic. The visualization categorizes the data into two primary clusters highlighted by shaded ellipses:
    # \begin{itemize}
    #     \item \textbf{Specialized \& Emerging Chemistries (Green):} This cluster is positioned on the left side of the plot, representing topics with lower document counts, generally ranging between 2,000 and 5,000.
    #     \item \textbf{Mainstream Li-ion Research (Blue):} This cluster is located on the right side of the plot, encompassing topics with higher document counts, typically between 8,000 and 10,000+.
    # \end{itemize}
    # A single data point, labeled \textit{"Aqueous Zn-ion batteries: anode/electrolyte,"} is positioned independently at the top center of the graph, outside of the two main highlighted clusters. The vertical axis measures Relevance on a scale from 0 to 8, while the horizontal axis tracks the Document Count.}


# --- 4. Export data to LaTeX table using longtable ---

# Adjust special characters (BEFORE any export)
topic_indexes["topic"] = (
    topic_indexes["topic"]
    .str.replace("&", "\\&", regex=False)
    .str.replace("LiFePO₄", "\\ce{LiFePO4}", regex=False)
    .str.replace("MoS₂", "\\ce{MoS2}", regex=False)
    .str.replace("TiO₂", "\\ce{TiO2}", regex=False)
)

# Create group column
def classify_group(row):
    if row['rank2'] > 7:
        return 'High Relevance'
    elif row['rank1'] > rank1_half:
        return 'Mainstream Li-ion Research'
    else:
        return 'Specialized \\& Emerging Chemistries'

topic_indexes['group'] = topic_indexes.apply(classify_group, axis=1)

# Sort data
topic_indexes_sorted = topic_indexes.sort_values(
    by=['group', 'rank2', 'rank1'],
    ascending=[True, False, False]
)

# Select columns
df_latex = topic_indexes_sorted[['topic', 'rank1', 'rank2', 'group']].rename(columns={
    'topic': 'Topic',
    'rank1': 'Document Count',
    'rank2': 'Relevance',
    'group': 'Group'
})

# (Optional) wrap long lines
df_latex['Topic'] = df_latex['Topic'].apply(
    lambda x: "\\newline ".join(textwrap.wrap(x, 40))
)

# Generate longtable
latex_table = df_latex.to_latex(
    index=False,
    float_format="%.2f",
    column_format="p{6cm} c c p{5cm}",
    longtable=True,
    escape=False,
    caption=(
        "Summary of research topics showing the relationship between document count "
        "and relevance index. Topics are categorized into three groups: high relevance, "
        "mainstream Li-ion research, and specialized \\& emerging chemistries."
    ),
    label="tab:relevance_vs_documents_long"
)

# Save file
output_path = f'{config.RESULTS_FOLDER}/q4_6_2_relevance_index_vs_topic.tex'

with open(output_path, 'w') as f:
    f.write(latex_table)

print(f"LaTeX table (longtable) saved to: {output_path}")

