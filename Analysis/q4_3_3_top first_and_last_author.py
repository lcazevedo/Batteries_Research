import utils
import config

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df_data = utils.get_df_data()


# -----------------------------------------------------------------
# CONFIGURATION PARAMETERS
# -----------------------------------------------------------------

# Number of authors in Top N
N = 15

# Font sizes
fs1 = 56  # Title
fs2 = 44  # Axis labels and ticks
fs3 = 42  # Data labels (for bars)


# -----------------------------------------------------------------
# DATA PROCESSING (FOR BOTH COLUMNS)
# -----------------------------------------------------------------

# --- Processing for FULL NAME ---
author_counts_full = df_data[config.COLUMN_FIRST_AUTHOR_FULL_NAME].value_counts()
top_n_data_full = author_counts_full.head(N)
top_n_data_sorted_full = top_n_data_full.sort_values(ascending=True)

print(f"Top {N} data (Full Name) processed.")
print(top_n_data_full)
print("-" * 30)

# --- Processing for LAST NAME ---
author_counts_last = df_data[config.COLUMN_LAST_AUTHOR_FULL_NAME].value_counts()
top_n_data_last = author_counts_last.head(N)
top_n_data_sorted_last = top_n_data_last.sort_values(ascending=True)

print(f"Top {N} data (Last Name) processed.")
print(top_n_data_last)
print("-" * 30)


# -----------------------------------------------------------------
# PLOTTING HELPER FUNCTION
# -----------------------------------------------------------------

def plot_bars_on_ax(ax, data_sorted, title, ylabel):
    """Helper function to draw the bar chart on a specific axis."""
    # Create bars
    bars = ax.barh(data_sorted.index, data_sorted.values, color='DodgerBlue')

    # Configure Title and Labels
    ax.set_title(title, fontsize=fs1, pad=20, x=0.3)
    ax.set_xlabel('Frequency', fontsize=fs2, labelpad=15)
    ax.set_ylabel(ylabel, fontsize=fs2, labelpad=15)

    # Configure ticks
    ax.tick_params(axis='both', labelsize=fs2)

    # Add data labels (count)
    for i, bar in enumerate(bars):
        width = bar.get_width()

        # Place text inside the top 2 bars
        if (i == len(bars) - 1) or (i == len(bars) - 2) or ("Last" in title and (i == len(bars) - 3 or i == len(bars) - 4 or i == len(bars) - 5)):
            ha = 'right'
            x_pos = width - 0.5  
            color = 'white'      
        else:
            ha = 'left'
            x_pos = width + 0.5   
            color = 'black'     

        ax.text(x_pos,  
                bar.get_y() + bar.get_height() / 2, 
                f'{width}', 
                ha=ha,
                va='center',
                fontsize=fs3,
                color=color)

# -----------------------------------------------------------------
# CHART GENERATION (SIDE BY SIDE)
# -----------------------------------------------------------------

print("Generating chart with subplots...")

# Create a figure with 1 row and 2 columns of axes
# Increase figsize to accommodate both charts
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 16))

# Plot first chart (Full Name) on axis 1
plot_bars_on_ax(ax1, 
                top_n_data_sorted_full, 
                f'Top {N} First Authors', 
                'First Author')

# Plot second chart (Last Name) on axis 2
plot_bars_on_ax(ax2, 
                top_n_data_sorted_last, 
                f'Top {N} Last Authors', 
                'Last Author')

# Adjust overall layout
plt.tight_layout(pad=3.0)

# Save chart (both subplots in a single PDF)
file_name_plot = f'{config.RESULTS_FOLDER}/q4_3_3_top_first_and_last_author.pdf'
try:
    plt.savefig(file_name_plot, dpi=300, bbox_inches='tight')
    print(f"Chart saved successfully as: {file_name_plot}")
except Exception as e:
    print(f"Error saving chart: {e}")

print("-" * 30)

# -----------------------------------------------------------------
# LATEX TABLE GENERATION (FULL NAME - FILE 1)
# -----------------------------------------------------------------

print("Generating LaTeX table (Full Name)...")

df_latex_full = top_n_data_full.reset_index()
df_latex_full.columns = ['Author (Full Name)', 'Frequency']
df_latex_full.index = df_latex_full.index + 1
df_latex_full = df_latex_full.reset_index().rename(columns={'index': 'Rank'})

try:
    latex_table_string_full = df_latex_full.to_latex(
        index=False,
        caption=f'Top {N} Authors (Full Name).', # Corrected from Institutions to Authors
        label='tab:top_N_authors_full',
        position='htbp', 
        column_format='rlr'
    )
    file_name_latex_full = f'{config.RESULTS_FOLDER}/q4_3_3_top_first_author.tex'
    with open(file_name_latex_full, 'w', encoding='utf-8') as f:
        f.write(latex_table_string_full)
    print(f"LaTeX table (Full Name) saved as: {file_name_latex_full}")
except Exception as e:
    print(f"Error generating LaTeX table (Full Name): {e}")

print("-" * 30)

# -----------------------------------------------------------------
# LATEX TABLE GENERATION (LAST NAME - FILE 2)
# -----------------------------------------------------------------

print("Generating LaTeX table (Last Name)...")

df_latex_last = top_n_data_last.reset_index()
df_latex_last.columns = ['Author (Last Name)', 'Frequency']
df_latex_last.index = df_latex_last.index + 1
df_latex_last = df_latex_last.reset_index().rename(columns={'index': 'Rank'})

try:
    latex_table_string_last = df_latex_last.to_latex(
        index=False,
        caption=f'Top {N} Authors (Last Name).',
        label='tab:top_N_authors_last',
        position='htbp', 
        column_format='rlr'
    )
    file_name_latex_last = f'{config.RESULTS_FOLDER}/q4_3_3_top last_author.tex'
    with open(file_name_latex_last, 'w', encoding='utf-8') as f:
        f.write(latex_table_string_last)
    print(f"LaTeX table (Last Name) saved as: {file_name_latex_last}")
except Exception as e:
    print(f"Error generating LaTeX table (Last Name): {e}")

print("-" * 30)

# -----------------------------------------------------------------
# LATEX TABLE GENERATION (SIDE-BY-SIDE EXAMPLE - FILE 3)
# -----------------------------------------------------------------
# This code generates a LaTeX table that uses 'minipage' to 
# position the two data tables side by side.
# -----------------------------------------------------------------

print("Generating LaTeX table (Side-by-Side Example)...")

try:
    # Generate the *inner* table strings (without \begin{table})
    # Use df_latex_full and df_latex_last already created
    
    # Escape special LaTeX characters (such as '_') in names
    def escape_latex(df, cols):
        df_copy = df.copy()
        for col in cols:
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].astype(str).str.replace('_', '\\_', regex=False)
        return df_copy
    
    df_latex_full_safe = escape_latex(df_latex_full, ['Author (Full Name)'])
    df_latex_last_safe = escape_latex(df_latex_last, ['Author (Last Name)'])

    # Generate only the 'tabular' code
    core_table_full = df_latex_full_safe.to_latex(
        index=False, 
        column_format='r p{5cm} r', # 'p' allows line wrapping for long names
        header=['Rank', 'Author (Full Name)', 'Freq.']
    )
    
    core_table_last = df_latex_last_safe.to_latex(
        index=False, 
        column_format='r p{5cm} r',
        header=['Rank', 'Author (Last Name)', 'Freq.']
    )

    # Build the final LaTeX string
    # (May require the \usepackage{caption} package in your .tex)
    
    latex_string_combined = f"""
\\begin{{table}}[htbp]
\\centering
\\caption{{Comparison: Top ${N}$ Authors (Full Name and Last Name)}}
\\label{{tab:top_N_authors_combined}}

\\begin{{minipage}}{{0.48\\textwidth}}
    \\centering
    \\caption*{{Top ${N}$ (Full Name)}} % caption* (with asterisk) is an unnumbered title
    \\scriptsize % Smaller font to fit
    {core_table_full}
\\end{{minipage}}%
\\hfill % Flexible space between minipages
\\begin{{minipage}}{{0.48\\textwidth}}
    \\centering
    \\caption*{{Top ${N}$ (Last Name)}}
    \\scriptsize
    {core_table_last}
\\end{{minipage}}

\\end{{table}}
"""

    file_name_latex_combined = f'{config.RESULTS_FOLDER}/q4_3_3_top_first_and_last_author.tex'
    with open(file_name_latex_combined, 'w', encoding='utf-8') as f:
        f.write(latex_string_combined)
    print(f"LaTeX table (Side by Side) saved as: {file_name_latex_combined}")

except Exception as e:
    print(f"Error generating LaTeX table (Side by Side): {e}")


print("\nScript finished.")
