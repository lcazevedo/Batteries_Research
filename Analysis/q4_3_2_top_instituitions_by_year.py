import utils
import config

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator # Important to ensure year ticks are integers

# Load data
df_data = utils.get_df_data()

# -----------------------------------------------------------------
# CONFIGURATION PARAMETERS
# -----------------------------------------------------------------

# Number of institutions in Top N (from your previous script)
N = 15

# Font sizes (from your previous script)
fs1 = 50  # Title
fs2 = 40  # Axis labels and ticks
# fs3 (for bar labels) is not used here, but we define fs_legend

# New: Font size for legend (fs3=38 would be too large for 15 items)
fs_legend = 35 

# New: List of markers to differentiate lines
# List with 15 distinct markers
markers_list = ['o', 's', 'v', '^', 'D', '<', '>', 'p', '*', 'X', 'P', 'H', 'd', '1', '2']

# -----------------------------------------------------------------
# DATA PROCESSING (PART 1: IDENTIFY TOP N)
# -----------------------------------------------------------------

# 1. Remove "Unknown" entries if any exist
df_data_cleaned = df_data[df_data['first_institution'] != "Unknown"]

# 2. YEAR FILTER ADDED (as requested)
# Filter years up to 2024
# Ensure year column is numeric for filtering
df_data_cleaned[config.COLUMN_YEAR] = pd.to_numeric(df_data_cleaned[config.COLUMN_YEAR])
df_data_cleaned = df_data_cleaned[df_data_cleaned[config.COLUMN_YEAR] <= 2024]


# 3. Count total frequency of each institution (after filters)
institution_counts = df_data_cleaned['first_institution'].value_counts()

# 4. Select Top N
top_n_data = institution_counts.head(N)
top_n_list = top_n_data.index.tolist()

print(f"Top {N} institutions identified (post-filter):")
print(top_n_list)
print("-" * 30)

# -----------------------------------------------------------------
# DATA PROCESSING (PART 2: PREPARE FOR TIME SERIES)
# -----------------------------------------------------------------

print("Preparing data for the evolution chart...")

# 1. Filter DataFrame (already filtered) to contain ONLY Top N institutions
df_top_n_only = df_data_cleaned[df_data_cleaned['first_institution'].isin(top_n_list)]

# 2. Group by YEAR (config.COLUMN_YEAR) and INSTITUTION, then count occurrences
# .size() is faster than .count() when there are no NaNs
df_grouped = df_top_n_only.groupby([config.COLUMN_YEAR, 'first_institution']).size()


# 3. Pivot data (Unstack)
# We want Years as index (rows) and Institutions as columns
# fill_value=0 ensures years without publications for an inst. appear as 0
df_pivot = df_grouped.unstack(level='first_institution', fill_value=0)

# 4. Ensure index (year) is sorted
df_pivot = df_pivot.sort_index()

# 5. NEW: Sort institutions by the count of the last year (for the legend)
if not df_pivot.empty:
    # Take the last row (last year)
    last_year_data = df_pivot.iloc[-1]
    # Sort institutions (columns) by that row, in descending order
    sorted_institutions = last_year_data.sort_values(ascending=False).index
else:
    # Fallback if dataframe is empty after filters
    sorted_institutions = df_pivot.columns

print("Pivoted data ready. Legend order (based on last year):")
print(sorted_institutions.tolist())
print("-" * 30)

# -----------------------------------------------------------------
# CHART GENERATION (LINE CHART)
# -----------------------------------------------------------------

print("Generating evolution chart...")

fig, ax = plt.subplots(figsize=(24, 20)) # Wider figure for legend

# Iterate over each institution (column in df_pivot) and plot
# UPDATED: Iterate in 'sorted_institutions' order for the legend
for i, institution_name in enumerate(sorted_institutions):
    
    # Select a marker from the list, wrapping around if N > len(markers_list)
    marker = markers_list[i % len(markers_list)]
    
    ax.plot(df_pivot.index,                # Years (X)
            df_pivot[institution_name],    # Count (Y)
            label=institution_name,        # Label for legend
            marker=marker,                 # Marker
            markersize=12,                 # Marker size
            linewidth=3)                   # Line width

# Configure Title and Labels
ax.set_title(f'Evolution of Top {N} Institutions', fontsize=fs1, pad=20, weight='bold')
ax.set_xlabel('Year', fontsize=fs2, labelpad=15)
ax.set_ylabel('Frequency', fontsize=fs2, labelpad=15)

# Configure ticks (axis marks)
# UPDATED: 45-degree rotation on X-axis
ax.tick_params(axis='y', labelsize=fs2) # Y-axis ticks
ax.tick_params(axis='x', labelsize=fs2, rotation=45) # X-axis ticks

# Ensure X-axis (Year) shows only integers
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

# Add grid for easier reading
ax.grid(True, linestyle='--', alpha=0.6)

# Add Legend
legend = ax.legend(
    loc='upper center', 
    bbox_to_anchor=(0.5, -0.2), # Position: 0.5=Center(H), -0.2=Below chart (adjust if needed)
    fontsize=fs_legend,
    title='Institutions', # Updated legend title
    title_fontsize=fs_legend + 2, # Slightly larger legend title
    ncol=3, # Set number of columns
    frameon=False # Remove legend border
)

# Adjust layout
# bbox_inches='tight' is essential here to ensure external legend is included
plt.tight_layout()

# Save chart
file_name_plot = f'{config.RESULTS_FOLDER}/q4_3_2_evolution_top_n_institutions.pdf'
try:
    # Use bbox_inches='tight' to include legend in saved PDF
    plt.savefig(file_name_plot, dpi=300, bbox_inches='tight')
    print(f"Chart saved successfully as: {file_name_plot}")
except Exception as e:
    print(f"Error saving chart: {e}")

print("-" * 30)

# No LaTeX table needed for this chart unless requested.

print("\nScript finished.")
