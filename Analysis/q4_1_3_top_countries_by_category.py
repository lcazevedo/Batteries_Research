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
fs2=25

# Only the period 1993-2024 (as in the original code)
start, end = 1993, 2024

# --------------------------------------------------------------------
#  Filter the dataframe by the chosen period
# --------------------------------------------------------------------
df_period = df_data[(df_data[config.COLUMN_YEAR] >= start) & (df_data[config.COLUMN_YEAR] <= end)]

# --------------------------------------------------------------------
# OPTIONAL FILTERS — uncomment ONE to apply
# --------------------------------------------------------------------

# # JNIF Filter — papers above expected
# threshold_percent = 0.75
# df_period = df_period[df_period['JNIF'] >= df_period['JNIF'].quantile(threshold_percent)]

# # score_log Filter = w1*log(citations+1) + w2*log(jif+1)
# threshold_percent = 0.75
# df_period = df_period[df_period['score_log'] >= df_period['score_log'].quantile(threshold_percent)]

# # score_sqrt Filter = sqrt(citations * JIF)
# threshold_percent = 0.75
# df_period = df_period[df_period['score_sqrt'] >= df_period['score_sqrt'].quantile(threshold_percent)]

# --------------------------------------------------------------------
# 1. Total publications per country in the entire period (global denominator)
# --------------------------------------------------------------------
country_totals = (
    df_period
    .groupby(config.COLUMN_FIRST_COUNTRY)
    .size()
    .reset_index(name='total_period')
)

# Apply minimum threshold and sorting
country_totals = (
    country_totals[country_totals['total_period'] >= MIN_PUBLICATIONS]
    .sort_values(by='total_period', ascending=False)
    .head(TOP_N)
)

# --------------------------------------------------------------------
# 2. For each category, calculate percentages and select top_N
# --------------------------------------------------------------------
df_top_list = []
categories = df_data[config.COLUMN_CATEGORY_GROUP].unique()

for cat in categories:

    df_cat = df_period[df_period[config.COLUMN_CATEGORY_GROUP] == cat]
    if df_cat.empty:
        continue

    # Count per country within the category
    counts = (
        df_cat
        .groupby(config.COLUMN_FIRST_COUNTRY)
        .size()
        .reset_index(name='publications')
    )

    # Integrate global country totals (for percentage calculation)
    counts = counts.merge(country_totals, on=config.COLUMN_FIRST_COUNTRY, how='left')

    # Remove countries that did not reach MIN_PUBLICATIONS
    counts = counts.dropna(subset=['total_period'])

    # Calculate percentage relative to total output
    counts['percentage'] = (counts['publications'] / counts['total_period']) * 100

    # Select top countries (usually all)
    top_n = counts.sort_values(by='percentage', ascending=False).head(TOP_N)

    # Add category as column
    top_n['categoria'] = cat

    df_top_list.append(top_n)

# Consolidate everything
df_top_n = pd.concat(df_top_list, ignore_index=True)

# --------------------------------------------------------------------
# 3. Choose the countries to be displayed in the chart
# --------------------------------------------------------------------
# Sort by total output and get the top N_COUNTRIES
top_countries = (
    df_top_n.sort_values(by='total_period', ascending=False)[config.COLUMN_FIRST_COUNTRY]
    .unique()[:N_COUNTRIES]
)

# Remove unknown countries and filter only the selected top ones
df_filtrado = df_top_n[
    (df_top_n[config.COLUMN_FIRST_COUNTRY] != "UNKNOWN") &
    (df_top_n[config.COLUMN_FIRST_COUNTRY].isin(top_countries))
]

# --------------------------------------------------------------------
# 4. Pivot the dataframe to the Likert chart format
# --------------------------------------------------------------------
df_likert = (
    df_filtrado
    .pivot_table(
        index=config.COLUMN_FIRST_COUNTRY,
        columns='categoria',
        values='percentage',
        aggfunc='sum',
        fill_value=0
    )
)

# Create auxiliary variable to sort countries
df_likert["sort_value"] = (
    df_likert["Hidden Gem (Low IF, High Cit)"] +
    df_likert["Star (High IF, High Cit)"]
)

# Sort countries and remove auxiliary variable
df_likert = df_likert.sort_values(by="sort_value", ascending=False)
df_likert = df_likert.drop(columns=["sort_value"])

# Sort categories on the X-axis
categorias_ordenadas = [
    "Long Tail (Low IF, Low Cit)",
    "Coattail (High IF, Low Cit)",
    "Hidden Gem (Low IF, High Cit)",
    "Star (High IF, High Cit)"
]

df_likert = df_likert.reindex(columns=categorias_ordenadas, fill_value=0)

# --------------------------------------------------------------------
# ADDITION: append the total number of publications to the country name
# --------------------------------------------------------------------
# Merge with country_totals to recover total_period
df_likert = df_likert.merge(
    country_totals[[config.COLUMN_FIRST_COUNTRY, 'total_period']],
    left_index=True,
    right_on=config.COLUMN_FIRST_COUNTRY,
    how='left'
)

# Create new name for the Y-axis: "Country (N)"
df_likert['country_label'] = df_likert.apply(
    lambda row: f"{row[config.COLUMN_FIRST_COUNTRY]} ({int(row['total_period'])})",
    axis=1
)

# Make country_label the new index (to appear on the Y-axis)
df_likert = df_likert.set_index('country_label')

# Remove original auxiliary column
df_likert = df_likert.drop(columns=[config.COLUMN_FIRST_COUNTRY, 'total_period'])

print(df_likert)


# --------------------------------------------------------------------
# 6. Prepare LaTeX table from the chart data
# --------------------------------------------------------------------

# Copy so as not to affect the chart
df_latex = df_likert.copy()

# # Round percentages (e.g., 1 decimal place)
# df_latex = df_latex.round(1)

# # Optional: rename columns to something shorter / paper-friendly
# df_latex = df_latex.rename(columns={
#     "Long Tail (Low IF, Low Cit)": "Long Tail",
#     "Coattail (High IF, Low Cit)": "Coattail",
#     "Hidden Gem (Low IF, High Cit)": "Hidden Gem",
#     "Star (High IF, High Cit)": "Star"
# })

latex_table = df_latex.to_latex(
    index=True,
    escape=False,
    float_format="%.1f",
    column_format="lcccc",
    caption="Country-level distribution of research impact categories (1993--2024). Percentages refer to the share of papers of each country in each category.",
    label="tab:country_likert",
)

# Save to .tex file
with open(config.RESULTS_FOLDER + "/q4_1_3_top_countries_by_category.tex", "w") as f:
    f.write(latex_table)


# --------------------------------------------------------------------
# 5. Create Likert chart
# --------------------------------------------------------------------
fig = plot_likert.plot_counts(
    df_likert,
    scale=categorias_ordenadas,
    compute_percentages=True,
    bar_labels=True,
    bar_labels_color="snow",
    figsize=(18, 20),
    colors=[
        '#ffffff00',
        'firebrick',
        'lightcoral',
        'cornflowerblue',
        'darkblue'
    ],
)


fig.set_xlabel("Percentage of papers in each category")
fig.set_ylabel("Countries")

# --- Visual adjustments ---
ax = fig.axes

ax.figure.suptitle("Country-Level Distribution of Research Impact Categories (1993–2024)\nSorted by High-Utility Output", fontsize=fs1 + 8, y=1.02)

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
n_steps = int(np.floor(half_range / 10))+1
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
fig.legend(handles, labels, ncol=2, loc="lower center", fontsize=fs1, bbox_to_anchor=(0.5, -0.22), frameon=False)


# Bar labels
for txt in ax.texts:
    txt.set_fontsize(fs2)
    # --- Remove percentage symbol from labels ---
    txt.set_text(txt.get_text().replace("%", ""))

# --- Thicken bars (reduce vertical space between countries) ---
thickness_factor = 1.6  # 1.0 = default | 1.2–1.6 usually looks good for papers

for patch in ax.patches:
    y = patch.get_y()
    h = patch.get_height()

    new_h = h * thickness_factor

    # Recenter bar at the same Y
    patch.set_y(y - (new_h - h) / 2)
    patch.set_height(new_h)


plt.tight_layout()
plt.savefig(config.RESULTS_FOLDER+"/q4_1_3_top_countries_by_category.pdf", dpi=300, bbox_inches='tight')
