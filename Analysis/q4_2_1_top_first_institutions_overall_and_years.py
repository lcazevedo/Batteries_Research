import utils
import config

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# -----------------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------------

df_data = utils.get_df_data()

# -----------------------------------------------------------------
# PARAMETERS
# -----------------------------------------------------------------

N = 15

fs1 = 50
fs2 = 44
fs3 = 44
fs_legend = 44

markers_list = ['o','s','v','^','D','<','>','p','*','X','P','H','d','1','2']

# -----------------------------------------------------------------
# CLEANING AND FILTERS
# -----------------------------------------------------------------

df_data = df_data[df_data[config.COLUMN_FIRST_INSTITUTION] != "Unknown"]

df_data[config.COLUMN_YEAR] = pd.to_numeric(df_data[config.COLUMN_YEAR])
df_data = df_data[df_data[config.COLUMN_YEAR] <= 2024]

# -----------------------------------------------------------------
# TOP N INSTITUTIONS
# -----------------------------------------------------------------

institution_counts = df_data[config.COLUMN_FIRST_INSTITUTION].value_counts()
top_n_data = institution_counts.head(N)

top_n_data_sorted = top_n_data.sort_values(ascending=True)
top_n_list = top_n_data.index.tolist()

# -----------------------------------------------------------------
# PREPARE TEMPORAL DATA
# -----------------------------------------------------------------

df_top = df_data[df_data[config.COLUMN_FIRST_INSTITUTION].isin(top_n_list)]

df_grouped = df_top.groupby([config.COLUMN_YEAR, config.COLUMN_FIRST_INSTITUTION]).size()
df_pivot = df_grouped.unstack(level=config.COLUMN_FIRST_INSTITUTION, fill_value=0)
df_pivot = df_pivot.sort_index()

if not df_pivot.empty:
    last_year_data = df_pivot.iloc[-1]
    sorted_institutions = last_year_data.sort_values(ascending=False).index
else:
    sorted_institutions = df_pivot.columns

# -----------------------------------------------------------------
# COMBINED FIGURE
# -----------------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(
    1,2,
    figsize=(32,26),
    gridspec_kw={'width_ratios':[0.8,2.2]}
)

# ================================================================
# (b) EVOLUTION PLOT
# ================================================================

for i, inst in enumerate(sorted_institutions):

    marker = markers_list[i % len(markers_list)]

    ax2.plot(
        df_pivot.index,
        df_pivot[inst],
        label=inst,
        marker=marker,
        markersize=16,
        linewidth=3
    )

ax2.set_xlabel("Year", fontsize=fs2)
ax2.set_ylabel("Frequency", fontsize=fs2)

ax2.tick_params(axis='x', labelsize=fs2, rotation=45)
ax2.tick_params(axis='y', labelsize=fs2)

ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
ax2.grid(True, linestyle='--', alpha=0.6)

legend = ax2.legend(
    loc='upper center',
    bbox_to_anchor=(0.5,-0.25),
    fontsize=fs_legend,
    ncol=2,
    frameon=False,
    title="Institutions",
    title_fontsize=fs_legend+2
)

# label subplot
ax2.text(-0.08,1.03,"(b)", transform=ax2.transAxes,
         fontsize=fs1)

# ================================================================
# (a) BARH TOP INSTITUTIONS
# ================================================================

bars = ax1.barh(
    top_n_data_sorted.index,
    top_n_data_sorted.values,
    color='DodgerBlue',
    edgecolor='black',      # BORDER
    linewidth=3
)

ax1.set_xlabel("Frequency", fontsize=fs2)
ax1.tick_params(axis='both', labelsize=fs2)

for i, bar in enumerate(bars):

    width = bar.get_width()

    if i >= len(bars)-2:
        ha='right'
        x_pos=width-20
        color='white'
    else:
        ha='left'
        x_pos=width+20
        color='black'

    ax1.text(
        x_pos,
        bar.get_y()+bar.get_height()/2,
        f"{int(width)}",
        va='center',
        ha=ha,
        fontsize=fs3,
        color=color
    )

# label subplot
ax1.text(-0.15,1.03,"(a)", transform=ax1.transAxes,
         fontsize=fs1)

# -----------------------------------------------------------------
# FINAL ADJUSTMENT
# -----------------------------------------------------------------

plt.tight_layout()

file_name = f"{config.RESULTS_FOLDER}/q4_2_1_top_first_institutions_overall_and_years.pdf"

plt.savefig(file_name, dpi=300, bbox_inches='tight')

print("Figure saved to:", file_name)



# -----------------------------------------------------------------
# GENERATE LATEX TABLE (2014–2025)
# -----------------------------------------------------------------

# -----------------------------------------------------------------
# GENERATE LATEX TABLE (2014–2025)
# -----------------------------------------------------------------

print("Generating LaTeX table of publications per year...")

start_year = 2014
end_year = 2025

# ---------------------------------------------------------------
# RELOAD DATA (WITHOUT YEAR FILTER)
# ---------------------------------------------------------------

df_table_data = utils.get_df_data()

df_table_data = df_table_data[
    df_table_data[config.COLUMN_FIRST_INSTITUTION] != "Unknown"
]

df_table_data[config.COLUMN_YEAR] = pd.to_numeric(
    df_table_data[config.COLUMN_YEAR]
)

# keep only institutions in Top N
df_table_data = df_table_data[
    df_table_data[config.COLUMN_FIRST_INSTITUTION].isin(top_n_list)
]

# filter desired years
df_table_data = df_table_data[
    (df_table_data[config.COLUMN_YEAR] >= start_year) &
    (df_table_data[config.COLUMN_YEAR] <= end_year)
]

# ---------------------------------------------------------------
# GROUP
# ---------------------------------------------------------------

df_grouped = df_table_data.groupby(
    [config.COLUMN_YEAR, config.COLUMN_FIRST_INSTITUTION]
).size()

df_table = df_grouped.unstack(level=config.COLUMN_FIRST_INSTITUTION, fill_value=0)

df_table = df_table.sort_index()

# transpose so institutions are on rows
df_table = df_table.T

# ensure all columns 2014–2025
years = list(range(start_year, end_year + 1))
df_table = df_table.reindex(columns=years, fill_value=0)

# ensure same order as chart
df_table = df_table.loc[sorted_institutions]

# reset index
df_table = df_table.reset_index()
df_table.rename(columns={'index': 'Institution'}, inplace=True)

# -----------------------------------------------------------------
# GENERATE LATEX STRING
# -----------------------------------------------------------------

years = list(range(start_year, end_year+1))

header_years = " & ".join(map(str, years[:-1])) + f" & {years[-1]}\\tnote{{1}}"

latex_lines = []

latex_lines.append("\\begin{landscape}")
latex_lines.append("\\centering")
latex_lines.append("\\begin{table}")
latex_lines.append("\\begin{threeparttable}")

latex_lines.append(
f"\\caption{{Publications per year for the top {N} most productive first-author institutions ({start_year}--{end_year}).}}"
)

latex_lines.append("\\label{tab:pub_by_first_institutions_year}")

latex_lines.append(
"\\begin{tabular}{l" + "c"* (len(years)-1) + ">{\\columncolor[gray]{0.9}}c}"
)

latex_lines.append("\\toprule")

latex_lines.append(f" & {header_years} \\\\")

latex_lines.append("\\midrule")

for _, row in df_table.iterrows():

    inst = row[config.COLUMN_FIRST_INSTITUTION]

    values = " & ".join(str(row[y]) for y in years)

    latex_lines.append(f"{inst} & {values} \\\\")

latex_lines.append("\\bottomrule")

latex_lines.append("\\end{tabular}")

latex_lines.append("\\begin{tablenotes}")
latex_lines.append("\\footnotesize")
latex_lines.append(
"\\item[1] The gray column refers to partial data for 2025 (data available until October 2025)."
)

latex_lines.append("\\end{tablenotes}")

latex_lines.append("\\end{threeparttable}")
latex_lines.append("\\end{table}")
latex_lines.append("\\end{landscape}")

latex_table = "\n".join(latex_lines)

# -----------------------------------------------------------------
# SAVE FILE
# -----------------------------------------------------------------

file_name_latex = f"{config.RESULTS_FOLDER}/q4_2_1_top_first_institutions_decade.tex"

with open(file_name_latex,"w",encoding="utf-8") as f:
    f.write(latex_table)

print("LaTeX table saved to:", file_name_latex)
