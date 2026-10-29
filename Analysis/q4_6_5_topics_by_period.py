import utils
import config

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# Load data
df_data = utils.get_df_data()

# -----------------------------------------------------------------
# CONFIGURATION PARAMETERS
# -----------------------------------------------------------------

# --- TEMPORAL CONTROL ---
bin_size = 5        # E.g.: 10 for decades, 5 for 5-year groups
start_year = 1990    # Start year 
end_year = 2024      # End year

# --- VISUAL SETTINGS (ADJUST INDIVIDUALLY HERE) ---

# 1. CHART Parameters
chart_figsize = (30, 20)  # Width x Height (in inches)
fs_title = 70            # Title font size
fs_axis_labels = 60       # Axis labels font size (X and Y)
fs_tick_labels = 45       # Axis numbers/dates font size
line_width = 6
marker_size = 20

# 2. LEGEND Parameters
legend_figsize = (30, 25) # Legend figure size (needs to be tall for 50 items)
fs_legend_text = 50       # Legend text font size
legend_cols = 2           # Number of columns

markers_list = ['o', 's', 'v', '^', 'D', '<', '>', 'p', '*', 'X', 'P', 'H', 'd', '1', '2']
N = 50

# -----------------------------------------------------------------
# DATA PROCESSING (Same as original)
# -----------------------------------------------------------------
df_data_cleaned = df_data[df_data[config.COLUMN_TOPIC_NAME] != "Unknown"]
df_data_cleaned[config.COLUMN_YEAR] = pd.to_numeric(df_data_cleaned[config.COLUMN_YEAR], errors='coerce')
df_data_cleaned = df_data_cleaned.dropna(subset=[config.COLUMN_YEAR])
df_data_cleaned[config.COLUMN_YEAR] = df_data_cleaned[config.COLUMN_YEAR].astype(int)

if start_year:
    df_data_cleaned = df_data_cleaned[df_data_cleaned[config.COLUMN_YEAR] >= start_year]
if end_year:
    df_data_cleaned = df_data_cleaned[df_data_cleaned[config.COLUMN_YEAR] <= end_year]

actual_max_year = df_data_cleaned[config.COLUMN_YEAR].max()

topic_counts = df_data_cleaned[config.COLUMN_TOPIC_NAME].value_counts()
top_n_list = topic_counts.head(N).index.tolist()

df_top_n = df_data_cleaned[df_data_cleaned[config.COLUMN_TOPIC_NAME].isin(top_n_list)].copy()
df_top_n['Period'] = (df_top_n[config.COLUMN_YEAR] // bin_size) * bin_size

df_grouped = df_top_n.groupby(['Period', config.COLUMN_TOPIC_NAME]).size()
df_pivot = df_grouped.unstack(level=config.COLUMN_TOPIC_NAME, fill_value=0)
df_pivot = df_pivot.sort_index()

if not df_pivot.empty:
    sorted_topics = df_pivot.iloc[-1].sort_values(ascending=False).index
else:
    sorted_topics = df_pivot.columns

x_values = df_pivot.index.tolist()
x_labels = []
if bin_size > 1:
    for start_p in x_values:
        theoretical_end = start_p + bin_size - 1
        label = f"{start_p}-{actual_max_year}*" if theoretical_end > actual_max_year else f"{start_p}-{theoretical_end}"
        x_labels.append(label)
else:
    x_labels = [str(y) for y in x_values]

# Base filename
file_name_base = f'{config.RESULTS_FOLDER}/q4_6_5_topics_evolution_{bin_size}y_labeled'


# =================================================================
# STEP 1: GENERATE ONLY THE CHART
# =================================================================
print("Generating chart (without legend)...")

fig, ax = plt.subplots(figsize=chart_figsize)

# Plot data
lines = [] # Store references for the legend later
for i, topic_name in enumerate(sorted_topics):
    marker = markers_list[i % len(markers_list)]
    # Note: Keep label here to extract later, but do not call ax.legend()
    line, = ax.plot(df_pivot.index, df_pivot[topic_name], label=topic_name, 
                    marker=marker, markersize=marker_size, linewidth=line_width)
    lines.append(line)

# Visual Settings
title_text = f'Evolution of Topics\n({bin_size}-Year Periods)'
if bin_size == 10: title_text = 'Evolution of Topics by Decade'

ax.set_title(title_text, fontsize=fs_title, pad=20, weight='bold')
ax.set_xlabel('Time Period', fontsize=fs_axis_labels, labelpad=30)
ax.set_ylabel('Frequency', fontsize=fs_axis_labels, labelpad=30)

ax.tick_params(axis='both', which='major', labelsize=fs_tick_labels, length=20, width=5, pad=15)
ax.set_xticks(x_values)
ax.set_xticklabels(x_labels, fontsize=fs_tick_labels, rotation=45, ha='right')

ax.grid(visible=True, axis='x', linestyle='--', alpha=0.6, linewidth=3)
ax.grid(visible=True, axis='y', linewidth=3)

# Adjust margins to use space well (since there is no legend in the way)
plt.subplots_adjust(top=0.92, bottom=0.15, left=0.10, right=0.98)

plt.tight_layout()

# Save Chart
file_chart = f"{file_name_base}_chart.pdf"
try:
    plt.savefig(file_chart, dpi=300) # Do not use bbox_inches='tight' if keeping exact figsize
    print(f"--> Chart saved: {file_chart}")
except Exception as e:
    print(f"Error saving chart: {e}")
    
# Capture handles and labels to use in the next figure
handles, labels = ax.get_legend_handles_labels()
plt.close(fig) # Close chart figure to free memory


# =================================================================
# STEP 2: GENERATE ONLY THE LEGEND
# =================================================================
print("Generating separate legend...")

# Create a new empty figure dedicated only to the legend
fig_leg = plt.figure(figsize=legend_figsize)
ax_leg = fig_leg.add_subplot(111)

# Turn off axes (no borders or numbers needed in the legend)
ax_leg.axis('off')

# Create legend in the center of the figure
legend = ax_leg.legend(
    handles, labels,
    loc='center',          # Center in figure
    fontsize=fs_legend_text,
    ncol=legend_cols,
    frameon=False,         # No surrounding box
    markerscale=2.0,       # Increase symbol size for visibility
    labelspacing=1.2,      # Vertical spacing
    columnspacing=1.5,     # Spacing between columns
    labelcolor='black'     # Legend text color
)

# Save Legend
file_legend = f"{file_name_base}_legend.pdf"
try:
    # Use bbox_inches='tight' here to crop excess whitespace around the legend
    plt.savefig(file_legend, dpi=300, bbox_inches='tight')
    print(f"--> Legend saved: {file_legend}")
except Exception as e:
    print(f"Error saving legend: {e}")

plt.close(fig_leg)


# -----------------------------------------------------------------
# LATEX TABLE EXPORT
# -----------------------------------------------------------------
print("Generating LaTeX table...")

try:
    # 1. Create copy for manipulation
    df_export = df_pivot.copy()
    
    # 2. Replace numeric index (Start Year) with Formatted Labels (e.g., 2020-2024*)
    df_export.index = x_labels
    
    # 3. Transpose table (Rows = Topics, Columns = Periods)
    # This is essential because N=50 would not fit the page width
    df_export = df_export.T
    
    # 4. Reorder rows (Topics) according to legend order (most frequent at the end)
    # Ensure only topics present in the dataframe are selected
    valid_sorted_topics = [t for t in sorted_topics if t in df_export.index]
    df_export = df_export.loc[valid_sorted_topics]
    
    # 5. Save to LaTeX
    file_name_tex = f"{file_name_base}.tex"
    
    # float_format="%.0f" ensures no decimal places appear (e.g., 10.0 -> 10)
    df_export.to_latex(file_name_tex, index=True, float_format="%.0f", caption=f"Topic Evolution by {bin_size}-Year Period")
    
    print(f"LaTeX table saved: {file_name_tex}")

except Exception as e:
    print(f"Error saving LaTeX table: {e}")

print("-" * 30)
print("Script finished.")