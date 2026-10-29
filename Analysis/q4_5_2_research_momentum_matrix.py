import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
from adjustText import adjust_text
import utils
import config

# Font settings
fs1 = 30  # Title
fs2 = 24  # Axes and Quadrants
fs_labels = 24 # Bubble labels

# 1. Loading and Filtering (2006-2025)
df_data = utils.get_df_data()
df_phrases = utils.get_df_data_phrases()

df_merged = df_phrases.merge(df_data[[config.COLUMN_DOI, config.COLUMN_YEAR]], 
                             left_on='doi', right_on='DOI', how='inner')
df_merged = df_merged[(df_merged[config.COLUMN_YEAR] >= 2006) & (df_merged[config.COLUMN_YEAR] <= 2025)]
df_filtered = df_merged[df_merged['directions_group'] != 'No Research Direction'].copy()

# 2. Split into P1 (2006-2015) and P2 (2016-2025)
def get_window(year):
    return 'P1' if 2006 <= year <= 2015 else 'P2'

df_filtered['Window'] = df_filtered[config.COLUMN_YEAR].apply(get_window)

# 3. Presence and Momentum Calculation
counts = df_filtered.groupby(['directions_group', 'Window']).size().unstack(fill_value=0)

# Market Share % calculation in each window
total_p1 = counts['P1'].sum()
total_p2 = counts['P2'].sum()

matrix_df = pd.DataFrame(index=counts.index)
matrix_df['Total_Presence'] = counts.sum(axis=1)
matrix_df['Share_P1'] = (counts['P1'] / total_p1) * 100
matrix_df['Share_P2'] = (counts['P2'] / total_p2) * 100
matrix_df['Momentum'] = matrix_df['Share_P2'] - matrix_df['Share_P1']

# Select most relevant themes (e.g., Top 15 by presence)
top_themes = matrix_df.nlargest(15, 'Total_Presence').copy()

# 4. Chart Generation
plt.figure(figsize=(14, 14))
sns.set_style("white") # Clean style without grid so quadrants stand out

# Scatter Plot (Bubbles)
ax = sns.scatterplot(data=top_themes, 
                     x='Total_Presence', 
                     y='Momentum', 
                     size='Total_Presence', 
                     sizes=(20, 500), # Larger bubbles for visual impact
                     hue='Momentum', 
                     palette='viridis', 
                     legend=False, 
                     edgecolor='black', 
                     alpha=0.7)

# 5. Adding labels with TextWrap and AdjustText
texts = []
for i, name in enumerate(top_themes.index):
    # Wrap name into lines of at most 20 characters
    wrapped_name = textwrap.fill(name, width=20)
    texts.append(plt.text(top_themes.Total_Presence.iloc[i], 
                          top_themes.Momentum.iloc[i], 
                          wrapped_name, 
                          fontsize=fs_labels, 
                          ha='center', va='center'))

# Automatically adjust position to avoid overlapping bubbles or other texts
adjust_text(texts, 
            only_move={'points':'y', 'text':'xy'}, # Allow moving text in any direction
            expand_points=(1.8, 1.8),              # Increase bubble repulsion area
            expand_text=(1.2, 1.2),                # Increase spacing between names
            force_points=0.5,                      # Point repulsion force
            arrowprops=dict(arrowstyle='->', color='gray', lw=1, alpha=0.5))


# 6. Quadrant Dividing Lines (X-axis at median, Y-axis at 0)
plt.axhline(0, color='black', lw=1.5, ls='-')
plt.axvline(top_themes['Total_Presence'].median(), color='black', lw=1.5, ls='-')

# 7. Quadrant Text Positioning (At edges/corners)
# Use transform=ax.transAxes to position relative to plot area (0 to 1)
ax.text(0.02, 0.9, 'EMERGING', transform=ax.transAxes, fontsize=fs2, 
        fontweight='bold', color='darkorange', va='top', ha='left')

ax.text(0.98, 0.96, 'RISING STARS', transform=ax.transAxes, fontsize=fs2, 
        fontweight='bold', color='green', va='top', ha='right')

ax.text(0.98, 0.04, 'MAINSTREAM', transform=ax.transAxes, fontsize=fs2, 
        fontweight='bold', color='blue', va='bottom', ha='right')

# 8. Final Axis and Title Adjustments
plt.title('Research Momentum Matrix: Priorities Shift\n(2006-2015 vs. 2016-2025)', fontsize=fs1, pad=40)
plt.xlabel('Cumulative Research Presence (Total Mentions)', fontsize=fs2)
plt.ylabel('Research Momentum\n(Market Share % Change)', fontsize=fs2)

plt.setp(ax.get_xticklabels(), fontsize=fs2-4)
plt.setp(ax.get_yticklabels(), fontsize=fs2-4)

plt.tight_layout()
plt.savefig(f'{config.RESULTS_FOLDER}/q4_5_4_momentum_matrix.pdf', dpi=300)

print("Momentum Matrix generated with label wrapping and auto-adjustment.")


plt.savefig(f'{config.RESULTS_FOLDER}/q4_5_2_research_momentum_matrix.pdf', dpi=300)

print("Research Momentum Matrix generated successfully.")



# --- UPDATED CODE: LATEX TABLE WITHOUT % SYMBOL IN CELLS ---

table_momentum = top_themes.copy()
table_momentum.columns = ['Total Presence', 'Share 2006-2015 (%)', 'Share 2016-2025 (%)', 'Momentum (Δ%)']
table_momentum = table_momentum.sort_values(by='Momentum (Δ%)', ascending=False)

latex_momentum = table_momentum.to_latex(
    caption="Research Momentum Metrics: Market Share and Growth Comparison (2006-2015 vs. 2016-2025).",
    label="tab:research_momentum_matrix",
    column_format="l" + "c" * len(table_momentum.columns),
    position="htbp",
    index=True,
    index_names=True,
    escape=True,
    formatters={
        'Total Presence': lambda x: f"{int(x)}",
        'Share 2006-2015 (%)': lambda x: f"{x:.2f}", # % removed
        'Share 2016-2025 (%)': lambda x: f"{x:.2f}", # % removed
        'Momentum (Δ%)': lambda x: f"{x:+.2f}"       # % removed, sign kept
    }
)

# 3. Save .tex file
with open(f'{config.RESULTS_FOLDER}/q4_5_2_research_momentum_matrix.tex', 'w', encoding='utf-8') as f:
    f.write(latex_momentum)

print("Momentum Matrix LaTeX table generated successfully.")
