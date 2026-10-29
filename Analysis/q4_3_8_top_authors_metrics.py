import utils
import config

import pandas as pd
import numpy as np
import textwrap
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt

from adjustText import adjust_text   # pip install adjustText


# Load data
df_data = utils.get_df_data()



# Define variables for font size
sa = 28  # Font size for subplot titles
sb = 28  # Font size for axes and legends
sc = 32
sd = 24  # Font size for author names       

qtd_authors = 15

# Process data for first authors
first_author_stats = df_data.groupby(config.COLUMN_FIRST_AUTHOR_FULL_NAME).agg(
    publications=(config.COLUMN_FIRST_AUTHOR_FULL_NAME, 'size'),
    total_citations=(config.COLUMN_CITATIONS_COUNT, 'sum'),
    mean_citations=(config.COLUMN_CITATIONS_COUNT, 'mean')
).reset_index()
top_10_first_authors = first_author_stats.nlargest(qtd_authors, 'publications')

# Process data for last authors
last_author_stats = df_data.groupby(config.COLUMN_LAST_AUTHOR_FULL_NAME).agg(
    publications=(config.COLUMN_LAST_AUTHOR_FULL_NAME, 'size'),
    total_citations=(config.COLUMN_CITATIONS_COUNT, 'sum'),
    mean_citations=(config.COLUMN_CITATIONS_COUNT, 'mean')
).reset_index()
top_10_last_authors = last_author_stats.nlargest(qtd_authors, 'publications')

# Format author names to title case
top_10_first_authors[config.COLUMN_FIRST_AUTHOR_FULL_NAME] = top_10_first_authors[config.COLUMN_FIRST_AUTHOR_FULL_NAME].str.title()
top_10_last_authors[config.COLUMN_LAST_AUTHOR_FULL_NAME] = top_10_last_authors[config.COLUMN_LAST_AUTHOR_FULL_NAME].str.title()


# Create charts
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 20)) # Slightly increased height

# Add an overall title for the entire figure
fig.suptitle(f'Analysis of Top {qtd_authors} Authors\nby Publication and Citation Metrics',
             fontsize=sc,           # Larger than subplot titles (sa=20)
             fontweight='bold')   # Bold for emphasis


# First chart: Top First Authors
scatter1 = ax1.scatter(top_10_first_authors['publications'], top_10_first_authors['total_citations'],
                       c=top_10_first_authors['mean_citations'], cmap='viridis', s=150)

# --- CHANGE 1: Collect texts for adjustment ---
texts_ax1 = []
for i, txt in enumerate(top_10_first_authors[config.COLUMN_FIRST_AUTHOR_FULL_NAME]):
    texts_ax1.append(ax1.text(top_10_first_authors['publications'].iloc[i],
                             top_10_first_authors['total_citations'].iloc[i],
                             txt,
                             fontsize=sd,
                             ha='left', va='center_baseline'))

# Call adjust_text to position texts without overlap
adjust_text(texts_ax1, ax=ax1, arrowprops=dict(arrowstyle='->', color='black', lw=0.5), expand=(1.5, 1.5))

# texts = [plt.text(x[i], y[i], 'Text%s' %i, ha='center', va='center') for i in range(len(x))]
# adjust_text(texts, arrowprops=dict(arrowstyle='->', color='grey', lw=0.5))
# --------------------------------------------------

ax1.set_title('(A) First authors', loc='left', fontsize=sa, fontweight='bold')
ax1.set_xlabel('Number of Publications', fontsize=sb)
ax1.set_ylabel('Total Citations', fontsize=sb)
ax1.tick_params(axis='both', which='major', labelsize=sb)
ax1.set_facecolor('white')
ax1.grid(False)
cbar1 = fig.colorbar(scatter1, ax=ax1)
cbar1.set_label('Mean Citations', fontsize=sb)
cbar1.ax.tick_params(labelsize=sb)


# Second chart: Top Last Authors
scatter2 = ax2.scatter(top_10_last_authors['publications'], top_10_last_authors['total_citations'],
                       c=top_10_last_authors['mean_citations'], cmap='viridis', s=150)

# --- CHANGE 2: Collect texts for adjustment ---
texts_ax2 = []
for i, txt in enumerate(top_10_last_authors[config.COLUMN_LAST_AUTHOR_FULL_NAME]):
    texts_ax2.append(ax2.text(top_10_last_authors['publications'].iloc[i],
                             top_10_last_authors['total_citations'].iloc[i],
                             txt,
                             fontsize=sd,
                             ha='left', va='center_baseline'))

# Call adjust_text again for the second chart
adjust_text(texts_ax2, ax=ax2, arrowprops=dict(arrowstyle='->', color='black', lw=0.5), expand=(1.8, 1.8))
# --------------------------------------------------

ax2.set_title('(B) Last authors', loc='left', fontsize=sa, fontweight='bold')
ax2.set_xlabel('Number of Publications', fontsize=sb)
ax2.set_ylabel('Total Citations', fontsize=sb)
ax2.tick_params(axis='both', which='major', labelsize=sb)
ax2.set_facecolor('white')
ax2.grid(False)
cbar2 = fig.colorbar(scatter2, ax=ax2)
cbar2.set_label('Mean Citations', fontsize=sb)
cbar2.ax.tick_params(labelsize=sb)


# Adjust layout to prevent overlap and save figure
plt.tight_layout(pad=3.0)

plt.savefig(f"{config.RESULTS_FOLDER}/q4_3_8_top_authors_metrics.pdf") 

#plt.show()


# ==========================================================
# Create LaTeX table with data from both charts
# ==========================================================

import os

table_file = f"{config.RESULTS_FOLDER}/q4_3_8_top_authors_metrics.tex"

with open(table_file, "w") as f:

    f.write("\\begin{table}[htbp]\n")
    f.write("\\centering\n")
    f.write("\\small\n")
    f.write("\\begin{tabular}{l l r r r}\n")
    f.write("\\hline\n")
    f.write(" & Author & Publications & Total citations & Mean citations \\\\\n")
    f.write("\\hline\n")

    # ---------------------------
    # First authors
    # ---------------------------
    n_first = len(top_10_first_authors)

    for i, row in top_10_first_authors.iterrows():

        author = row[config.COLUMN_FIRST_AUTHOR_FULL_NAME]
        pubs = int(row["publications"])
        total_cit = int(row["total_citations"])
        mean_cit = row["mean_citations"]

        if i == top_10_first_authors.index[0]:
            f.write(
                f"\\multirow{{{n_first}}}{{*}}{{\\rotatebox[origin=c]{{90}}{{First author}}}} "
                f"& {author} & {pubs} & {total_cit} & {mean_cit:.1f} \\\\\n"
            )
        else:
            f.write(
                f"& {author} & {pubs} & {total_cit} & {mean_cit:.1f} \\\\\n"
            )

    # Thicker separator line
    f.write("\\hline\\hline\n")

    # ---------------------------
    # Last authors
    # ---------------------------
    n_last = len(top_10_last_authors)

    for i, row in top_10_last_authors.iterrows():

        author = row[config.COLUMN_LAST_AUTHOR_FULL_NAME]
        pubs = int(row["publications"])
        total_cit = int(row["total_citations"])
        mean_cit = row["mean_citations"]

        if i == top_10_last_authors.index[0]:
            f.write(
                f"\\multirow{{{n_last}}}{{*}}{{\\rotatebox[origin=c]{{90}}{{Last author}}}} "
                f"& {author} & {pubs} & {total_cit} & {mean_cit:.1f} \\\\\n"
            )
        else:
            f.write(
                f"& {author} & {pubs} & {total_cit} & {mean_cit:.1f} \\\\\n"
            )

    f.write("\\hline\n")
    f.write("\\end{tabular}\n")
    f.write("\\caption{Publication and citation metrics for the top authors.}\n")
    f.write("\\label{tab:top_authors_metrics}\n")
    f.write("\\end{table}\n")

print("LaTeX table saved to:", table_file)
