import utils
import config

import pandas as pd
import numpy as np
import textwrap
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt

# Load data
df_data = utils.get_df_data()


fs1 = 30
fs2 = 22

# 1. Find the top authors with the most publications
top_10_last_authors = df_data[config.COLUMN_LAST_AUTHOR_FULL_NAME].value_counts().nlargest(15).index

# 2. Filter the DataFrame to include only the top authors
df_top_10 = df_data[df_data[config.COLUMN_LAST_AUTHOR_FULL_NAME].isin(top_10_last_authors)]

# 3. Calculate the collaboration period for each author-coauthor pair
collaboration_periods = df_top_10.groupby([config.COLUMN_LAST_AUTHOR_FULL_NAME, config.COLUMN_FIRST_AUTHOR_FULL_NAME])[config.COLUMN_YEAR].agg(['min', 'max']).reset_index()
collaboration_periods['period'] = collaboration_periods['max'] - collaboration_periods['min'] + 1

# 4. For each author, find the minimum, maximum, and mean collaboration period
author_collaboration_summary = collaboration_periods.groupby(config.COLUMN_LAST_AUTHOR_FULL_NAME)['period'].agg(['min', 'max', 'mean']).reset_index()

# Sort authors to match the order of the top authors
author_collaboration_summary[config.COLUMN_LAST_AUTHOR_FULL_NAME] = pd.Categorical(
    author_collaboration_summary[config.COLUMN_LAST_AUTHOR_FULL_NAME],
    categories=top_10_last_authors,
    ordered=True
)
author_collaboration_summary = author_collaboration_summary.sort_values(config.COLUMN_LAST_AUTHOR_FULL_NAME)
author_collaboration_summary[config.COLUMN_LAST_AUTHOR_FULL_NAME] = author_collaboration_summary[config.COLUMN_LAST_AUTHOR_FULL_NAME].str.title()  # Name formatting

# --- 5. Create horizontal bar chart (image style) ---

# Define color palette inspired by the image
num_authors = len(author_collaboration_summary)
cmap = plt.get_cmap('cividis') 
colors = cmap(np.linspace(0.2, 0.8, num_authors))

# Figure settings
fig, ax = plt.subplots(figsize=(10, 8))
# fig.patch.set_facecolor('#FBF5EF') # Cream/beige background color
# ax.set_facecolor('#FBF5EF')
fig.patch.set_facecolor('white') # White background color
ax.set_facecolor('white')

# Y position for bars
y_pos = np.arange(len(author_collaboration_summary[config.COLUMN_LAST_AUTHOR_FULL_NAME]))

# Bar width is the difference between maximum and minimum period
bar_widths = author_collaboration_summary['max'] - author_collaboration_summary['min']

# Starting point of bars is the minimum period
bar_starts = author_collaboration_summary['min']

# Plot horizontal bars representing the [min, max] interval
ax.barh(y_pos, bar_widths, left=bar_starts, align='center', height=0.5,
        color=colors, alpha=0.8, edgecolor=colors, linewidth=1.5)

# Plot mean markers over the bars
ax.scatter(author_collaboration_summary['mean'], y_pos,
           marker='|', color='black', s=100, zorder=10, linewidths=1.5)

# --- Style and Label Adjustments ---

# Move X-axis to the top
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')

# Remove chart borders (spines) for a cleaner look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# Configure Y-axis ticks
ax.set_yticks(y_pos)
ax.set_yticklabels(author_collaboration_summary[config.COLUMN_LAST_AUTHOR_FULL_NAME], fontfamily='serif', fontsize=fs2)
ax.invert_yaxis()  # Invert axis so the top author is at the top

# Limit X-axis with a small margin
ax.set_xlim(0, author_collaboration_summary['max'].max() * 1.05)

# Add subtle vertical grid
ax.grid(axis='x', linestyle='-', color='gray', alpha=0.3, linewidth=0.5)
ax.grid(axis='y', linestyle='', alpha=0) # Remove Y-axis grid

# Force display of all integers on X-axis
ax.xaxis.set_major_locator(mticker.MultipleLocator(base=1))

# Add titles, subtitles, and source using fig.text for precise positioning
# fig.text(0.1, 0.95, 'CO-AUTHOR JOURNEY',
#          fontsize=16, fontweight='bold', ha='left', va='top', fontfamily='serif')
fig.text(0.5, 1, 'Collaboration longevity for the\ntop 15 last authors',
         fontsize=fs1, ha='center', va='top', fontfamily='serif')
# fig.text(0.1, 0.02, 'SOURCE: Institutional data analysis',
#          fontsize=8, ha='left', va='bottom', color='gray', fontfamily='serif')

# X-axis label
ax.set_xlabel('Longevity in years', fontsize=fs2, fontfamily='serif', labelpad=10)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.tick_params(axis='x', labelsize=fs2)


plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.9]) # Adjust layout to fit text

plt.savefig(f"{config.RESULTS_FOLDER}/q4_3_7_collaboration_longevity_by_last_author.pdf") 

#plt.show()

############
# Generate latex table for analysis

# Rename columns for clarity
table_df = author_collaboration_summary.rename(columns={
    config.COLUMN_LAST_AUTHOR_FULL_NAME: 'Author',
    'min': 'Min',
    'max': 'Max',
    'mean': 'Mean'
})

# Format mean with 2 decimal places
table_df['Mean'] = table_df['Mean'].map(lambda x: f"{x:.2f}")

latex_table = table_df.to_latex(
    index=False,
    column_format='lccc',
    caption='Collaboration longevity by last author (years).',
    label='tab:collaboration_longevity',
    escape=False
)

print(latex_table)
file_name_latex_combined = f'{config.RESULTS_FOLDER}/q4_3_7_collaboration_longevity_by_last_author.tex'
with open(file_name_latex_combined, 'w', encoding='utf-8') as f:
    f.write(latex_table)
