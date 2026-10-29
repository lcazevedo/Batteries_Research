import utils
import config

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import textwrap
import seaborn as sns

# Load data
topic_index_df = utils.get_topic_index_df()

# Define columns
# IMPORTANT: columns from topic_relevance_index.parquet
col_direita = "composite_index_wo_review"
col_esquerda = "composite_index"
rank_direita = "rank_index_wo_review"
rank_esquerda = "rank_index"


# Sort data by the right column
df_plot = topic_index_df.sort_values(by=col_direita, ascending=False).reset_index(drop=True)

# Generate gradient color palette
n = len(df_plot)
palette_direita = sns.color_palette("viridis", n)
palette_esquerda = sns.color_palette("plasma", n)

# Define font sizes
fs1 = 70  # Title
fs2 = 55  # Axes and labels
fs3 = 25  # (unused)

# Create figure
fig, ax = plt.subplots(figsize=(16, 33))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Plot left bars (with reviews)
sns.barplot(
    data=df_plot,
    x=-df_plot[col_esquerda],
    y="topic",
    palette=palette_direita,
    orient='h',
    ax=ax
)

# Plot right bars (without reviews)
sns.barplot(
    data=df_plot,
    x=df_plot[col_direita],
    y="topic",
    palette=palette_esquerda,
    orient='h',
    ax=ax
)

# Add values to the bars
for i, (val_esq, val_dir, rank_esq, rank_dir) in enumerate(zip(df_plot[col_esquerda], df_plot[col_direita], df_plot[rank_esquerda], df_plot[rank_direita])):
    # Left
    ax.text(
        -val_esq - 0.005, i, 
        f" ({rank_esq}) {val_esq:.1f}",
        va='center', ha='right',
        fontsize=fs2, color='black'
    )
    # Right
    ax.text(
        val_dir + 0.005, i, 
        f"{val_dir:.1f} ({rank_dir})",
        va='center', ha='left',
        fontsize=fs2, color='black'
    )

# Adjust Y-axis labels (topics)
y_lbls = ax.get_yticklabels()
lbls = ["\n".join(textwrap.wrap(label.get_text(), 80)) for label in y_lbls]
ax.set_yticklabels(lbls)

# Central line on X-axis (zero)
ax.axvline(0, color='black', linewidth=1.5)

ax.set_xlim(-25, 25)

# Adjust X-axis ticks (visually removing the negative sign)
xticks = ax.get_xticks()
ax.set_xticks(xticks)
ax.set_xticklabels([abs(int(x)) if x != 0 else 0 for x in xticks], fontsize=fs2, rotation=90)

# Axis settings
plt.xlabel("Relevance index", fontsize=fs2)
plt.ylabel("")
plt.yticks(fontsize=fs2)

# Title
plt.title("Topics Ranked by Relevance Index\n", fontsize=fs1, loc='right')

# Add top labels
# Get current axis limits
x_min, x_max = ax.get_xlim()
#y_max = len(df_plot) - 0.5
y = -0.5

# Text for the left side
ax.text(
    x_min * 0.4,  # x position (adjustable as needed)
    y,     # slightly above the bars
    "With Review",
    fontsize=fs2,
    ha='center',
    va='bottom'
)

# Text for the right side
ax.text(
    x_max * 0.65,  # x position
    y,
    "Without Review",
    fontsize=fs2,
    ha='center',
    va='bottom'
)

# Remove legend
ax.legend().remove()

# 8. SAVE THE CHART (formerly step 7)
file_name_plot = f'{config.RESULTS_FOLDER}/q4_6_1_topics_ranked_by_relevance_index_violin.pdf'
try:
    # Use bbox_inches='tight' to include the legend in the saved PDF
    plt.savefig(file_name_plot, dpi=300, bbox_inches='tight')
    print(f"Chart successfully saved as: {file_name_plot}")
except Exception as e:
    print(f"Error saving chart: {e}")

topic_index_df["topic"] = topic_index_df["topic"].str.replace("&", "\&").str.replace("LiFePO₄", "\ce{LiFePO4}").str.replace("MoS₂", "\ce{MoS2}").str.replace("TiO₂", "\ce{TiO2}")
topic_index_df = topic_index_df.rename(columns={"topic": "Topic Name", col_esquerda: "Relevance Index", col_direita: "Relevance Index Without Review"})

df_plot = topic_index_df.sort_values(by="Relevance Index", ascending=False)
tabular_str = df_plot[["Topic Name", "Relevance Index"]].to_latex(
    index=False,
    float_format="%.2f",
    column_format="l c",
    escape=False
)

# Build full structure
latex_table = f"""
\\begin{{table}}[H]
\\begin{{threeparttable}}
\\centering
\\caption{{Topic Ranking by Composite Relevance Index (Reviews Included).}}
\\label{{tab:q4_6_1_topics_ranked_by_relevance_index_violin}}

{tabular_str}

\\begin{{tablenotes}}
    \\footnotesize 
    \\item This table presents the topics ranked by the composite relevance index, where data from review articles were included in the calculation.
\\end{{tablenotes}}

\\end{{threeparttable}}
\\end{{table}}
"""

# Save to file
with open(f'{config.RESULTS_FOLDER}/q4_6_1_topics_ranked_by_relevance_index.tex', 'w') as f:
    f.write(latex_table)

df_plot = topic_index_df.sort_values(by="Relevance Index Without Review", ascending=False)
tabular_str = df_plot[["Topic Name", "Relevance Index Without Review"]].to_latex(
    index=False, 
    float_format="%.2f",
    column_format="l c",
    escape=False    
)

# Build full structure
latex_table = f"""
\\begin{{table}}[H]
\\begin{{threeparttable}}
\\centering
\\caption{{Topic Ranking by Composite Relevance Index (Reviews Excluded).}}
\\label{{tab:q4_6_1_topics_ranked_by_relevance_index_violin}}

{tabular_str}

\\begin{{tablenotes}}
    \\footnotesize 
    \\item This table presents the topics ranked by the composite relevance index, where data from review articles were excluded to better reflect primary research activity.
\\end{{tablenotes}}

\\end{{threeparttable}}
\\end{{table}}
"""

# Save to file
with open(f'{config.RESULTS_FOLDER}/q4_6_1_topics_ranked_by_relevance_index_wo_review.tex', 'w') as f:
    f.write(latex_table)


print("\nScript finished.")