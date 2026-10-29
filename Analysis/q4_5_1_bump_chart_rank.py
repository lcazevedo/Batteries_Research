import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import utils
import config
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as ticker



# Font settings (maintaining visual standard)
fs1 = 30  # Title
fs2 = 26  # Axes and labels
marker_size = 18

# 1. Define function that generates the correct English ordinal suffix
def to_ordinal(n, pos):
    n = int(n)
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f"{n}{suffix}"

# 1. Loading
df_data = utils.get_df_data()
df_phrases = utils.get_df_data_phrases()

# 2. Merging and Corrected Temporal Filter (2006-2025)
df_merged = df_phrases.merge(df_data[[config.COLUMN_DOI, config.COLUMN_YEAR]], 
                             left_on='doi', right_on='DOI', how='inner')

# CHANGE: Filter for the last two decades (20 years)
df_merged = df_merged[(df_merged[config.COLUMN_YEAR] >= 2006) & (df_merged[config.COLUMN_YEAR] <= 2025)]

# Filter: Remove non-informative directions
df_filtered = df_merged[df_merged['directions_group'] != 'No Research Direction'].copy()

# 3. Create 5-year periods adjusted to 2006
def get_quinquennium_2006(year):
    # Logic: (2006-2010), (2011-2015), (2016-2020), (2021-2025*)
    start_year = ((year - 2006) // 5) * 5 + 2006
    end_year = start_year + 4
    if start_year == 2021:
        return "2021-2025*"
    return f"{start_year}-{end_year}"

df_filtered['Period'] = df_filtered[config.COLUMN_YEAR].apply(get_quinquennium_2006)

# 4. Frequency and Ranking Calculation
counts = df_filtered.groupby(['Period', 'directions_group']).size().reset_index(name='Frequency')
counts['Rank'] = counts.groupby('Period')['Frequency'].rank(method='first', ascending=False)

# Select overall Top 10 for this 20-year period
top_10_overall = df_filtered['directions_group'].value_counts().nlargest(10).index
counts_top = counts[counts['directions_group'].isin(top_10_overall)]
max_rank_found = int(counts_top['Rank'].max())

# 5. Pivot for the Bump Chart
all_periods = ["2006-2010", "2011-2015", "2016-2020", "2021-2025*"]
pivot_ranks = counts_top.pivot(index='Period', columns='directions_group', values='Rank')
pivot_ranks = pivot_ranks.reindex(all_periods)

# 6. Bump Chart Generation
plt.figure(figsize=(16, 14))
sns.set_style("whitegrid")

# Lines with Viridis gradient and large markers
ax = sns.lineplot(data=pivot_ranks, 
                  markers=True, 
                  dashes=False, 
                  linewidth=5, 
                  ms=marker_size, 
                  palette='viridis')

# Y-Axis adjustments (Inverted ranking)
plt.gca().invert_yaxis()
plt.yticks(range(1, max_rank_found + 1)) # Show from 1 down to the lowest rank
plt.ylim(max_rank_found + 0.5, 0.5)        # Margin so markers are not clipped

# Titles and Labels
plt.title('Shift in Research Priorities: Evolution of Direction Ranking\n(Two Decades: 2006-2025)', fontsize=fs1, pad=30)
plt.xlabel('Period', fontsize=fs2)
plt.ylabel('Rank (1st is Highest Priority)', fontsize=fs2)
plt.setp(ax.get_xticklabels(), fontsize=fs2)
plt.setp(ax.get_yticklabels(), fontsize=fs2)
ax.xaxis.grid(False)
ax.yaxis.set_major_formatter(FuncFormatter(to_ordinal))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# Legend below
plt.legend(title='', loc='upper center', bbox_to_anchor=(0.5, -0.12), 
           ncol=1, frameon=False, fontsize=fs2)

plt.tight_layout()

# Saving PDF
plt.savefig(f'{config.RESULTS_FOLDER}/q4_5_1_bump_chart_rank.pdf', dpi=300)

def to_ordinal_str(n):
    if pd.isna(n): 
        return "-"  # In case the theme has no rank in that period
    n = int(n)
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f"{n}{suffix}"

# 1. Transpose and apply formatting to each cell
table_latex = pivot_ranks.T.applymap(to_ordinal_str)

# 2. Generating LaTeX string
latex_output = table_latex.to_latex(
    caption="Evolution of Research Priority Ranking (1st = Highest Priority) from 2006 to 2025*",
    label="tab:directions_rank_evolution",
    column_format="l" + "c" * len(table_latex.columns),
    position="htbp",
    index=True,
    index_names=True,
    escape=True
)

with open(f'{config.RESULTS_FOLDER}/q4_5_1_bump_chart_rank.tex', 'w', encoding='utf-8') as f:
    f.write(latex_output)

print("Bump chart and rank table generated successfully.")
