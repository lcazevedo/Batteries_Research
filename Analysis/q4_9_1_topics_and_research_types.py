import utils
import config

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Load data
df_data = utils.get_df_data()

df_data = df_data[df_data[config.COLUMN_RESEARCH_TYPE] != "Hybrid"]
df_data[config.COLUMN_RESEARCH_TYPE] = df_data[config.COLUMN_RESEARCH_TYPE].replace('Computational and simulation studies', 'Computational and\nsimulation studies')

fs1 = 28
fs2 = 22
fs3 = 18

eixo_y = config.COLUMN_TOPIC_NAME
eixo_x = config.COLUMN_RESEARCH_TYPE

# Group data to count topic combinations
grouped = (
    df_data.groupby([eixo_x, eixo_y])
    .size()
    .reset_index(name="count")
)

# ---- Sort by row total (eixo_y) ----
# Total sum per eixo_y
order_y = grouped.groupby(eixo_y)['count'].sum().sort_values(ascending=False).index

# Apply this order to the DataFrame (to ensure consistency)
grouped[eixo_y] = pd.Categorical(grouped[eixo_y], categories=order_y, ordered=True)

# Create pivot table
pivot = grouped.pivot_table(
    index=eixo_y,
    columns=eixo_x,
    values="count",
    fill_value=0
)

# Convert to integers
pivot = pivot.astype(int)

plt.figure(figsize=(12, 18))

sns.heatmap(
    pivot,
    cmap="Reds",
    annot=True,
    fmt="d",
    linewidths=0.5,
    cbar_kws={"label": "Count"},
    cbar=False,  
    annot_kws={"size": fs2} 
)

plt.xticks(rotation=45, ha="right", fontsize=fs2)
plt.yticks(fontsize=fs2)
plt.title("Distribution of Research Types across Thematic Topics", fontsize=fs1, loc="right", pad=20)
plt.xlabel("")
plt.ylabel("")

plt.tight_layout()


# # Create figure and axes
# plt.figure(figsize=(12, 18))
# plt.gca().set_facecolor("white")  # chart area background
# plt.gcf().patch.set_facecolor("white")  # figure background

# # Scatter plot (bubbles) with salmon color
# scatter = sns.scatterplot(
#     data=grouped,
#     x=eixo_x,
#     y=eixo_y,
#     size="count",
#     color="salmon",
#     sizes=(50, 2000),
#     legend=False  # Remove legend
# )

# # Add text inside circles for counts > 1000
# for _, row in grouped.iterrows():
#     if row["count"] > 5000: #100:
#         plt.text(
#             row[eixo_x],
#             row[eixo_y],
#             str(row["count"]),
#             color="black",
#             ha="center",
#             va="center",
#             fontsize=fs3,
#             #fontweight="bold"
#         )
#     else:
#         plt.text(
#             row[eixo_x],
#             row[eixo_y],
#             str(row["count"]),
#             color="black",
#             ha="center",
#             va="bottom",
#             fontsize=fs3,
#             #fontweight="bold"
#         )

# # Get unique y-axis values (topics)
# y_labels = grouped[eixo_y].unique()
# # Define additional margin (e.g., 1 visual unit)
# plt.ylim(len(y_labels)-0.5, -0.5)

# # Visual improvements
# plt.xticks(rotation=45, ha="right", fontsize=fs2)
# plt.yticks(fontsize=fs2)
# plt.xlabel("")
# plt.ylabel("")
# plt.title("Distribution of Research Types across Thematic Topics", fontsize=fs1, loc="right", pad=20)
# plt.grid(True)

# ax = plt.gca()
# x_labels = grouped[eixo_x].unique()
# plt.xlim(-0.5, len(x_labels) - 0.5)




plt.tight_layout()

plt.savefig(f'{config.RESULTS_FOLDER}/q4_9_1_topics_and_research_types.pdf', bbox_inches='tight') #, dpi=300)


# Sort rows by total
pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

# ---- Highlight the largest value in each row in bold ----
# Convert data to string to insert LaTeX commands
pivot_str = pivot.astype(str)

for i in pivot.index:
    max_val = pivot.loc[i].max()
    # Replace the largest value with itself in bold (\textbf{})
    pivot_str.loc[i, pivot.loc[i] == max_val] = pivot_str.loc[i, pivot.loc[i] == max_val].apply(
        lambda x: f"\\textbf{{{x}}}"
    )

# Export to LaTeX without escaping LaTeX characters
pivot_str.to_latex(
    f'{config.RESULTS_FOLDER}/q4_9_1_topics_and_research_types.tex',
    escape=False
)

print(pivot)


