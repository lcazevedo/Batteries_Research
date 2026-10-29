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
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import PercentFormatter


# Load data
df_data = utils.get_df_data()
df_data_phrases = utils.get_df_data_phrases()

df_analysis = df_data_phrases.merge(
    df_data[[config.COLUMN_DOI, config.COLUMN_YEAR]],
    left_on='doi',
    right_on=config.COLUMN_DOI,
    how='left'
)
direction_col = "directions_group"
year_col = config.COLUMN_YEAR
doi_original_col = 'doi'

NUM_DIRECTIONS_TO_PLOT = 20  # Number of directions to plot

# --- Font size definitions ---
ft1 = 40  # Main title font size
ft2 = 38  # Axis titles font size (X and Y)
ft3 = 38  # Axis labels/ticks font size (X and Y)

# --- 1. DATA PREPARATION ---
df_temporal = df_analysis.dropna(subset=[direction_col, year_col])
df_temporal[year_col] = df_temporal[year_col].astype(int)
df_temporal = df_temporal[df_temporal[year_col] >= 2014]
df_temporal = df_temporal[df_temporal[year_col] <= 2025]

# Filter out null or 'Uncategorized' data in the grouped column
df_ranked = df_analysis.dropna(subset=[direction_col])
df_ranked = df_ranked[
    ~df_ranked[direction_col].isin(['Uncategorized', 'No Research Direction'])
]

# Create ranking using value_counts() on the grouped column
direction_ranking = df_ranked[direction_col].value_counts().head(20)

top_directions = direction_ranking.head(NUM_DIRECTIONS_TO_PLOT).index
df_temporal_top = df_temporal[df_temporal[direction_col].isin(top_directions)]

# --- 2. ABSOLUTE FREQUENCIES ---
absolute_crosstab = pd.crosstab(
    df_temporal_top[year_col],
    df_temporal_top[direction_col]
)

# --- 3. RELATIVE FREQUENCIES (% per year) ---
total_pubs_per_year = df_temporal.groupby(year_col)[doi_original_col].nunique()
direction_pubs_per_year = df_temporal_top.groupby([year_col, direction_col])[doi_original_col].nunique()
numerator_crosstab = direction_pubs_per_year.unstack(fill_value=0)
relative_crosstab = numerator_crosstab.div(total_pubs_per_year, axis=0).fillna(0) * 100

# --- Sort columns consistently ---
sorted_columns = [col for col in top_directions if col in absolute_crosstab.columns]
absolute_crosstab = absolute_crosstab[sorted_columns]
relative_crosstab = relative_crosstab[sorted_columns]

# --- 4. CONVERT TO LONG FORMAT (for sns.lineplot) ---
df_abs = absolute_crosstab.reset_index().melt(
    id_vars=year_col, var_name='Direction Category', value_name='Number of Mentions'
)
df_rel = relative_crosstab.reset_index().melt(
    id_vars=year_col, var_name='Direction Category', value_name='% of Publications'
)

import matplotlib as mpl

mpl.rcParams['axes.edgecolor'] = 'black'
mpl.rcParams['xtick.color'] = 'black'
mpl.rcParams['ytick.color'] = 'black'
mpl.rcParams['grid.color'] = 'black'
mpl.rcParams['grid.linewidth'] = 0.2
mpl.rcParams['patch.edgecolor'] = 'black'

# --- 5. STYLE CONFIGURATION ---
#plt.style.use('seaborn-v0_8')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(23, 20), sharex=True, gridspec_kw={'hspace': 0.15})

# Force white background on figure and axes
fig.patch.set_facecolor('white')
ax1.set_facecolor('white')
ax2.set_facecolor('white')

# --- 6. CHART 1: ABSOLUTE ---
sns.lineplot(
    data=df_abs,
    x=year_col,
    y='Number of Mentions',
    hue='Direction Category',
    style='Direction Category',
    markers=True,
    dashes=False,
    linewidth=5,
    markersize=28,
    palette="viridis",      
    ax=ax1
)

        
ax1.set_title('Evolution of Directions (Absolute Values)', fontsize=ft1)
ax1.set_ylabel('Number of Mentions', fontsize=ft2)
ax1.set_xlabel('')
ax1.tick_params(axis='both', labelsize=ft3)
ax1.yaxis.set_major_locator(MultipleLocator(3000))

# --- 7. CHART 2: RELATIVE ---
sns.lineplot(
    data=df_rel,
    x=year_col,
    y='% of Publications',
    hue='Direction Category',
    style='Direction Category',
    markers=True,
    dashes=False,
    linewidth=5,
    markersize=28,
    palette="viridis",      
    ax=ax2,
    legend=False  # legend will be unified
)

ax2.set_title('Share of Publications by Direction (Relative %)', fontsize=ft1)
ax2.set_xlabel('Year', fontsize=ft2)
ax2.set_ylabel('% of Publications', fontsize=ft2)
ax2.tick_params(axis='both', labelsize=ft3)
ax2.yaxis.set_major_locator(MultipleLocator(10))
ax2.yaxis.set_major_formatter(PercentFormatter())

# --- GRID AND BORDERS (apply AFTER lineplots) ---
ax1.grid(True, color='black', linewidth=1)
ax2.grid(True, color='black', linewidth=1)

for ax in [ax1, ax2]:
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(2.2)
        
from matplotlib.ticker import PercentFormatter
ax2.yaxis.set_major_formatter(PercentFormatter())

# --- 8. REMOVE LOCAL LEGEND FROM FIRST CHART ---
ax1.get_legend().remove()  # Remove automatic legend created by Seaborn

# --- 9. GLOBAL LEGEND BELOW CHARTS ---
handles, labels = ax1.get_legend_handles_labels()
labels_2lines = [textwrap.fill(l, width=32) for l in labels]
fig.legend(
    handles,
    labels_2lines,
    loc='lower center',           # Center horizontally
    bbox_to_anchor=(0.5, -0.32),  # Increase space between chart and legend
    ncol=2,
    fontsize=ft3,
    title_fontsize='x-large',
    frameon=False
)

# --- 10. FINAL LAYOUT ADJUSTMENT ---
plt.subplots_adjust(bottom=0.18, hspace=0.15)  # Reserve space for legend and adjust vertical spacing

# --- 11. GLOBAL TITLE (optional) ---
fig.suptitle(
    f'Evolution of Research Directions in the Literature',
    fontsize=ft1+15,
    y=0.98
)


#################################################
# Save high-resolution figure
# --- 12. SAVE AND SHOW ---
plt.savefig(
    f'{config.RESULTS_FOLDER}/q4_12_4_direction_evolution_absolute_relative.pdf',
    dpi=600,
    bbox_inches='tight'
)

###########################################
# Export data as LaTeX for statistical analysis
df_abs.to_latex(
    f'{config.RESULTS_FOLDER}/q4_12_4_direction_evolution_absolute.csv',
    index=False
)
df_rel.to_latex(
    f'{config.RESULTS_FOLDER}/q4_12_4_direction_evolution_relative.csv',
    index=False
)


