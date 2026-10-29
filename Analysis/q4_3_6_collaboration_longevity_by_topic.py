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

# 1. Calculate collaboration period for each pair (config.COLUMN_LAST_AUTHOR_FULL_NAME, config.COLUMN_FIRST_AUTHOR_FULL_NAME)
collaboration_periods = df_data.groupby([config.COLUMN_LAST_AUTHOR_FULL_NAME, config.COLUMN_FIRST_AUTHOR_FULL_NAME])[config.COLUMN_YEAR].agg(['min', 'max']).reset_index()
collaboration_periods['period'] = collaboration_periods['max'] - collaboration_periods['min'] + 1

# 2. Associate each collaboration period with the corresponding topics
# First, get unique author-topic pairs from the original DataFrame
unique_author_topics = df_data[[config.COLUMN_LAST_AUTHOR_FULL_NAME, config.COLUMN_FIRST_AUTHOR_FULL_NAME, config.COLUMN_TOPIC_NAME]].drop_duplicates()

# Now, merge collaboration periods with topics
topic_collaboration_data = pd.merge(
    unique_author_topics,
    collaboration_periods[[config.COLUMN_LAST_AUTHOR_FULL_NAME, config.COLUMN_FIRST_AUTHOR_FULL_NAME, 'period']],
    on=[config.COLUMN_LAST_AUTHOR_FULL_NAME, config.COLUMN_FIRST_AUTHOR_FULL_NAME]
)

# 3. For each topic, find minimum, maximum, and mean collaboration duration
topic_summary = topic_collaboration_data.groupby(config.COLUMN_TOPIC_NAME)['period'].agg(['min', 'max', 'mean']).reset_index()

# Sort topics by name for chart consistency
topic_summary = topic_summary.sort_values(config.COLUMN_TOPIC_NAME)


# --- 4. Create horizontal bar chart ---

# Define color palette
num_topics = len(topic_summary)
cmap = plt.get_cmap('cividis')
colors = cmap(np.linspace(0.2, 0.8, num_topics))

# Figure settings
fig, ax = plt.subplots(figsize=(13, 14))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Y position for bars
y_pos = np.arange(len(topic_summary[config.COLUMN_TOPIC_NAME]))

# Bar width is the difference between maximum and minimum period
bar_widths = topic_summary['max'] - topic_summary['min']
# Starting point of bars is the minimum period
bar_starts = topic_summary['min']

# Plot horizontal bars representing the [min, max] interval
ax.barh(y_pos, bar_widths, left=bar_starts, align='center', height=0.5,
        color=colors, alpha=0.8, edgecolor=colors, linewidth=1.5)

# Plot mean markers over the bars
ax.scatter(topic_summary['mean'], y_pos,
           marker='|', color='black', s=150, zorder=10, linewidths=2)

# --- Style and Label Adjustments ---

# Move X-axis to the top
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')

# Remove chart borders (spines)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# Configure Y-axis ticks
ax.set_yticks(y_pos)
ax.set_yticklabels(topic_summary[config.COLUMN_TOPIC_NAME], fontfamily='serif', fontsize=fs2)
ax.invert_yaxis()  # Invert axis so the first topic is at the top

# Limit X-axis starting at 0 with a small margin
ax.set_xlim(0, topic_summary['max'].max() * 1.05)

# Add vertical grid
ax.grid(axis='x', linestyle='-', color='gray', alpha=0.3, linewidth=0.5)
ax.grid(axis='y', linestyle='', alpha=0)

# Force display of integer ticks on X-axis
#ax.xaxis.set_major_locator(mticker.MultipleLocator(base=1))
ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=7, integer=True))

# Add titles and labels
fig.text(0.1, 0.95, 'Collaboration Longevity by Topic',
         fontsize=fs1, ha='left', va='top', fontfamily='serif')

ax.set_xlabel('Longevity in years', fontsize=fs2, fontfamily='serif', labelpad=10)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

ax.tick_params(axis='x', labelsize=fs2)

plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.9])

plt.savefig(f"{config.RESULTS_FOLDER}/q4_3_6_collaboration_longevity_by_topic.pdf") 

#plt.show()


############
# Generate latex table for analysis

# Rename columns for clarity
table_df = topic_summary.rename(columns={
    config.COLUMN_TOPIC_NAME: 'Topic',
    'min': 'Min',
    'max': 'Max',
    'mean': 'Mean'
})

# Format mean with 2 decimal places
table_df['Mean'] = table_df['Mean'].map(lambda x: f"{x:.2f}")

latex_table = table_df.to_latex(
    index=False,
    column_format='lccc',
    caption='Collaboration longevity by topic (years).',
    label='tab:collaboration_longevity',
    escape=False
)

print(latex_table)
file_name_latex_combined = f'{config.RESULTS_FOLDER}/q4_3_6_collaboration_longevity_by_topic.tex'
with open(file_name_latex_combined, 'w', encoding='utf-8') as f:
    f.write(latex_table)
