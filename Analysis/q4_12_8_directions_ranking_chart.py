from networkx import display
import utils
import config

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import textwrap
from matplotlib.ticker import MaxNLocator

# Load data
df_data = utils.get_df_data_phrases()

# --- Font size definitions ---
ft1 = 48 # Main title font size
ft2 = 42 # Axis titles font size (X and Y)
ft3 = 38 # Axis labels/ticks font size (X and Y)

# --- Column name with directions already grouped by script 6 ---
grouped_direction_col = f'directions_group'

if grouped_direction_col not in df_data.columns:
    print(f"ERROR: Column '{grouped_direction_col}' was not found.")
    print("Make sure script '6_group_directions.py' has been executed and the dataset has been updated with the latest data.")
else:
    print(f"\n--- Table: Top 20 Most Cited Direction Categories (from '{grouped_direction_col}') ---")
    
    # Filter out null or 'Uncategorized' data in the grouped column
    df_ranked = df_data.dropna(subset=[grouped_direction_col])
    df_ranked = df_ranked[
        ~df_ranked[grouped_direction_col].isin(['Uncategorized', 'No Research Direction'])
    ]
    
    # Create ranking using value_counts() on the grouped column
    direction_ranking = df_ranked[grouped_direction_col].value_counts().head(20)
    
    # Convert to DataFrame for cleaner display
    df_ranking_table = direction_ranking.to_frame(name="Number of Mentions")
    
    total_top20 = df_ranking_table["Number of Mentions"].sum()
    df_ranking_table["Percentage (%)"] = (
        df_ranking_table["Number of Mentions"] / total_top20 * 100
    ).round(2)

    
    # --- Prepare and export to LaTeX with updated names ---
    df_latex = df_ranking_table.copy()
    df_latex.index.name = "Direction"
    df_latex.rename(columns={"Number of Mentions": "Frequency"}, inplace=True)
    
    # Save LaTeX table
    df_latex.to_latex(f'{config.RESULTS_FOLDER}/q4_12_8_direction_ranking_table.tex',
                      float_format="%.2f")

    # --- Chart: Ranking Visualization ---
    print("\n--- Chart: Top 20 Most Cited Direction Categories (from Manual Grouping) ---")
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 17))
    
    ax = sns.barplot(
        x=direction_ranking.values,
        y=direction_ranking.index,
        palette="viridis"
    )
    
    # Define a small offset for the text
    offset = ax.get_xlim()[1] * 0.01

    for p in ax.patches:
        width = p.get_width()
        
        # If the bar is very long, place the text inside
        if width > 34000:
            ax.text(width - offset, # X position (slightly before the end of the bar)
                    p.get_y() + p.get_height() / 2., # Y position (middle of the bar)
                    f'{int(width)}',
                    ha='right', # Right-align
                    va='center',
                    fontsize=ft3,
                    color='white') # White color for contrast
        # If short, place the text outside
        else:
            ax.text(width + offset, # X position (end of the bar + offset)
                    p.get_y() + p.get_height() / 2.,
                    f'{int(width)}',
                    ha='left', # Left-align
                    va='center',
                    fontsize=ft3,
                    color='black') # Black color

    # Get current Y-axis labels
    y_labels = [label.get_text() for label in ax.get_yticklabels()]
    # Apply line wrapping to each label
    wrapped_labels = [textwrap.fill(label, width=50) for label in y_labels]
    # Set new wrapped labels on the chart
    ax.set_yticklabels(wrapped_labels)

    plt.title('Ranking of Main Technological\nand Economic Research Directions', fontsize=ft1, pad=20, loc="right")
    plt.xlabel('Number of Sentences', fontsize=ft2)
    plt.ylabel('Direction Category', fontsize=ft2)
    
    plt.xticks(fontsize=ft3, rotation=45)
    plt.yticks(fontsize=ft3)

    # Adjust X-axis limit
    #plt.xlim(right=ax.get_xlim()[1] * 1.15)
    
    # tight_layout is important after line wrapping to adjust spacing
    plt.tight_layout()
        
    plt.savefig(f'{config.RESULTS_FOLDER}/q4_12_8_direction_ranking_chart.pdf', dpi=300, bbox_inches='tight')
