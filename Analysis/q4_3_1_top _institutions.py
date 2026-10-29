
import utils
import config

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df_data = utils.get_df_data()


# -----------------------------------------------------------------
# CONFIGURATION PARAMETERS
# -----------------------------------------------------------------

# Number of institutions in Top N
N = 15

# Font sizes
fs1 = 50  # Title
fs2 = 40  # Axis labels and ticks
fs3 = 38  # Data labels (for bars)


# -----------------------------------------------------------------
# DATA PROCESSING
# -----------------------------------------------------------------

# 1. Count the frequency of each institution in the 'first_institution' column
df_data = df_data[df_data['first_institution'] != "Unknown"]  # Remove "Unknown" entries if any exist
institution_counts = df_data['first_institution'].value_counts()

# 2. Select Top N
top_n_data = institution_counts.head(N)

# 3. Prepare data for the chart (sort for barh)
# Reverse order (ascending=True) so matplotlib
# plots the largest bar at the top of the horizontal chart.
top_n_data_sorted = top_n_data.sort_values(ascending=True)

print(f"Top {N} data processed.")
print(top_n_data)
print("-" * 30)

# -----------------------------------------------------------------
# CHART GENERATION (HORIZONTAL BAR CHART)
# -----------------------------------------------------------------

print("Generating chart...")

# Use a horizontal bar chart (barh) which is better for long labels
fig, ax = plt.subplots(figsize=(20, 18))

# Create bars
bars = ax.barh(top_n_data_sorted.index, top_n_data_sorted.values, color='DodgerBlue')

# Configure Title and Labels with defined font sizes
ax.set_title(f'Top {N} Institutions', fontsize=fs1, pad=20, weight='bold')
ax.set_xlabel('Frequency', fontsize=fs2, labelpad=15)
ax.set_ylabel('Institution', fontsize=fs2, labelpad=15)

# Configure ticks (axis marks)
ax.tick_params(axis='both', labelsize=fs2)

# Add data labels (count) at the end of each bar
for i, bar in enumerate(bars):
    width = bar.get_width()

    # Check if it is the last bar (which is Top 1, the longest)
    if i == len(bars) - 1:
        # Top 1: Place text INSIDE the bar
        ha = 'right'
        x_pos = width - 0.5  # X position (slightly BEFORE the end of the bar)
        color = 'white'      # Text color (white to contrast with the bar)
    else:
        # Other bars: Place text OUTSIDE the bar
        ha = 'left'
        x_pos = width + 0.5   # X position (slightly AFTER the end of the bar)
        color = 'black'     # Text color (default)

    ax.text(x_pos,  # X position (slightly after the end of the bar)
            bar.get_y() + bar.get_height() / 2, # Y position (center of the bar)
            f'{width}', # The text (value)
            ha=ha,
            va='center',
            fontsize=fs3,
            color=color)

# Adjust layout and margins to prevent labels from being cut off
plt.tight_layout()

# Save chart
file_name_plot = f'{config.RESULTS_FOLDER}/q4_3_1_top_n_institutions.pdf'
try:
    plt.savefig(file_name_plot, dpi=300, bbox_inches='tight')
    print(f"Chart saved successfully as: {file_name_plot}")
except Exception as e:
    print(f"Error saving chart: {e}")

print("-" * 30)

# -----------------------------------------------------------------
# LATEX TABLE GENERATION
# -----------------------------------------------------------------

print("Generating LaTeX table...")

# Convert Series (top_n_data) to a DataFrame to use .to_latex
# Use original data (top_n_data, not 'sorted') for the table
# because we want the ranking from largest to smallest.
df_latex = top_n_data.reset_index()
df_latex.columns = ['Institution', 'Frequency']

# Add ranking column
df_latex.index = df_latex.index + 1
df_latex = df_latex.reset_index().rename(columns={'index': 'Rank'})


# Generate LaTeX code as a string
# Requires the 'booktabs' package (use \usepackage{booktabs} in your .tex)
# for \toprule, \midrule, \bottomrule lines
try:

    latex_table_string = df_latex.to_latex(
        index=False,
        caption=f'Top {N} Institutions by Frequency.',
        label='tab:top_N_institutions',
        position='htbp', # LaTeX positioning suggestion (Here, Top, Bottom, Page)
        column_format='rlr' # r=right, l=left, c=center (Rank, Institution, Count)
    )

    # Save LaTeX code to a .tex file
    file_name_latex = f'{config.RESULTS_FOLDER}/q4_3_1_top_n_institutions.tex'
    
    with open(file_name_latex, 'w', encoding='utf-8') as f:
        f.write(latex_table_string)
        
    print(f"LaTeX table saved successfully as: {file_name_latex}")
    
    # Optional: Print generated LaTeX code to console
    # print("\n--- Generated LaTeX Code ---")
    # print(latex_table_string)
    # print("---------------------------\n")

except ImportError:
    print("Error: The 'pandas' library (required for .to_latex) was not found.")
except Exception as e:
    print(f"Error generating or saving LaTeX file: {e}")


print("\nScript finished.")
