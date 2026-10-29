###########################################################################
# identifying the elite countries of publications
###########################################################################

import pandas as pd
import plot_likert
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
import numpy as np

import config
import utils

# --- Load the dataset ---
df_data = utils.get_df_data()

# Adjust internal format of plot_likert (label formatting)
plot_likert.__internal__.BAR_LABEL_FORMAT = "%.1f"

# --------------------------------------------------------------------
# GENERAL PARAMETERS
# --------------------------------------------------------------------
MIN_PUBLICATIONS = 10
TOP_N = 5_000_000         # maximum number of countries considered before plotting
N_COUNTRIES = 25          # final number of countries displayed in the chart

fs1 = 28
fs2=22

# Only the period 1993-2024 (as in the original code)
start, end = 1993, 2024

# --------------------------------------------------------------------
#  Filter the dataframe by the chosen period
# --------------------------------------------------------------------
df_period = df_data[(df_data[config.COLUMN_YEAR] >= start) & (df_data[config.COLUMN_YEAR] <= end)]
df_period = df_period[df_period[config.COLUMN_FIRST_COUNTRY] != 'UNKNOWN']

df_period['JNIF_quartile'] = pd.qcut(df_period[config.COLUMN_JNIF], q=4, labels=['Q4', 'Q3', 'Q2', 'Q1'])  

score = 'JNIF_quartile'

tmp = pd.crosstab(df_period[config.COLUMN_FIRST_COUNTRY], df_period[score])
tmp['tot'] = tmp.sum(axis=1)
tmp = tmp.sort_values(by='tot',ascending=False).head(20)


# --------------------------------------------------------------------
# 6. Create percentage table (matching the Likert chart)
# --------------------------------------------------------------------

df_latex = tmp[['Q1', 'Q2', 'Q3', 'Q4']].copy()

# Convert counts to percentages per country
df_latex = df_latex.div(df_latex.sum(axis=1), axis=0) * 100

# Round (value) — optional
df_latex = df_latex.round(1)

# Rename columns (optional, cleaner for paper)
df_latex = df_latex.rename(columns={
    'Q1': 'Q1 (highest JNIF)',
    'Q2': 'Q2',
    'Q3': 'Q3',
    'Q4': 'Q4 (lowest JNIF)'
})
latex_table = df_latex.to_latex(
    index=True,
    escape=False,
    float_format="%.1f",
    column_format="lcccc",
    caption="Distribution of Journal Normalized Impact Factor (JNIF) quartiles for the top 20 most productive countries (1993--2024).",
    label="tab:jnif_quartiles_country"
)

with open(config.RESULTS_FOLDER + "/q4_1_4_JNIF_per_quartile.tex", "w") as f:
    f.write(latex_table)




fig = plot_likert.plot_counts(tmp[['Q1', 'Q2', 'Q3', 'Q4']],scale=['Q1', 'Q2', 'Q3', 'Q4'],
                    compute_percentages=True, 
                    bar_labels=True,
                    bar_labels_color="snow",
                    figsize=(15,16),
                    colors=['#ffffff00','firebrick','lightcoral','cornflowerblue','darkblue'])

fig.set_xlabel("Percentage of papers in each category")
fig.set_ylabel("Countries")


# --- Visual adjustments ---
ax = fig.axes

for patch in ax.patches:
    current_height = patch.get_height()
    new_height = current_height * 1.5  # increase thickness
    patch.set_height(new_height)

ax.figure.suptitle("Distribution of JNIF Quartiles for the\nTop 20 Most Productive Countries (1993–2024)", fontsize=fs1 + 8, y=1.02)

ax.set_facecolor("white")
fig.patch.set_facecolor("white")
ax.grid(False)

# Borders
for spine in ax.spines.values():
    spine.set_edgecolor("black")
    spine.set_linewidth(1.5)

# Axis titles
ax.set_xlabel(ax.get_xlabel(), fontsize=fs1)
ax.set_ylabel(ax.get_ylabel(), fontsize=fs1)

# Ticks
# --- Find the vertical line of the true zero ---
zero_line = None
for line in ax.lines:
    xdata = line.get_xdata()
    # vertical line: all x values are equal
    if len(xdata) > 1 and np.allclose(xdata, xdata[0]):
        zero_line = xdata[0]
        break

# Fallback if not found
if zero_line is None:
    xmin, xmax = ax.get_xlim()
    zero_line = (xmin + xmax) / 2

# --- Symmetric limits ---
xmin, xmax = ax.get_xlim()
half_range = max(abs(xmin - zero_line), abs(xmax - zero_line))
ax.set_xlim(zero_line - half_range, zero_line + half_range)

# --- Clean ticks: 0, ±10, ±20, ...
n_steps = int(np.floor(half_range / 10))
ticks = [zero_line]
for i in range(1, n_steps + 1):
    ticks.append(zero_line - 10 * i)
    ticks.append(zero_line + 10 * i)

ticks = sorted(ticks)
ax.set_xticks(ticks)

# --- Formatting ---
ax.xaxis.set_major_formatter(
    FuncFormatter(lambda x, pos: f"{int(round(abs(x - zero_line)))}%")
)

ax.tick_params(axis='x', labelsize=fs1, rotation=90)
ax.tick_params(axis='y', labelsize=fs1)

# Legend
# leg = ax.get_legend()
# if leg is not None:
    
#     for text in leg.get_texts():
#         text.set_fontsize(fs1)
#     leg.set_title(leg.get_title().get_text(), prop={"size": fs1})


handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, ncol=4, loc="lower center", fontsize=fs1, bbox_to_anchor=(0.5, -0.18),frameon=False)


# Bar labels
for txt in ax.texts:
    txt.set_fontsize(fs2)

plt.tight_layout()
plt.savefig(config.RESULTS_FOLDER+"/q4_1_4_JNIF_per_quartile.pdf", dpi=300, bbox_inches='tight')
