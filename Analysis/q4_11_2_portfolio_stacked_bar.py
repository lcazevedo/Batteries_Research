import utils
import config
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# ==========================================
# GLOBAL FONT CONFIGURATION
# ==========================================
fs1 = 32  # Main Title font size
fs2 = 26  # Axis Labels font size (X and Y)
fs3 = 26  # Text font size (Country names, X-axis, and Legends)
fs_bar = 24 # Font size of numbers written INSIDE the bars

# 1. Load data
df_data = utils.get_df_data()

# 2. Focus only on documents mentioning at least one of the 4 main metals in the analysis
mask_main_metals = (df_data['is_li_ion'] == True) | \
                    (df_data['is_na_ion'] == True) | \
                    (df_data['is_k_ion'] == True) | \
                    (df_data['is_zn_ion'] == True)

df_valido = df_data[mask_main_metals].copy()

# 3. Clean Country column
col_pais = config.COLUMN_FIRST_COUNTRY
df_valido = df_valido.dropna(subset=[col_pais])
df_valido[col_pais] = df_valido[col_pais].astype(str).str.strip()

# 4. Create the 3 Mutually Exclusive Categories
df_valido['has_li'] = df_valido['is_li_ion'] == True
df_valido['has_nextgen'] = (df_valido['is_na_ion'] == True) | \
                            (df_valido['is_k_ion'] == True) | \
                            (df_valido['is_zn_ion'] == True)

df_valido['Li_Only'] = df_valido['has_li'] & ~df_valido['has_nextgen']
df_valido['NextGen_Only'] = ~df_valido['has_li'] & df_valido['has_nextgen']
df_valido['Mixed'] = df_valido['has_li'] & df_valido['has_nextgen']

# 5. Group by Country
df_paises = df_valido.groupby(col_pais).agg(
    Total_Main_Docs=('has_li', 'count'), # Count all rows in the group
    Li_Only_Count=('Li_Only', 'sum'),
    NextGen_Only_Count=('NextGen_Only', 'sum'),
    Mixed_Count=('Mixed', 'sum')
)

# Select the top 15 absolute producers to analyze
df_top15 = df_paises.nlargest(15, 'Total_Main_Docs').copy()

# 6. Calculate Percentages (Ensures each row sums to 100%)
# Row total = Li_Only + NextGen_Only + Mixed
df_top15['Portfolio_Total'] = df_top15['Li_Only_Count'] + df_top15['NextGen_Only_Count'] + df_top15['Mixed_Count']

df_top15['Li_Only_Pct'] = (df_top15['Li_Only_Count'] / df_top15['Portfolio_Total']) * 100
df_top15['Mixed_Pct'] = (df_top15['Mixed_Count'] / df_top15['Portfolio_Total']) * 100
df_top15['NextGen_Only_Pct'] = (df_top15['NextGen_Only_Count'] / df_top15['Portfolio_Total']) * 100

# 7. SORTING: Sort countries by highest Next-Gen share (creates a stepped visual layout)
df_top15 = df_top15.sort_values(by=['NextGen_Only_Pct', 'Mixed_Pct'], ascending=True)

# ==============================================================================
# 8. CHART GENERATION (100% STACKED BAR)
# ==============================================================================
fig, ax = plt.subplots(figsize=(14, 14))

paises = df_top15.index
li_vals = df_top15['Li_Only_Pct'].values
mix_vals = df_top15['Mixed_Pct'].values
next_vals = df_top15['NextGen_Only_Pct'].values

# Colors (Classic Blue for Lithium, Orange for Transition/Mixed, Vibrant Green for Next-Gen)
c_li = '#4C72B0'
c_mix = '#E67E22'
c_next = '#55A868'

# Plot the 3 stacked segments (using 'left' parameter to offset bars horizontally)
p1 = ax.barh(paises, li_vals, color=c_li, edgecolor='black', height=0.7, label='Li-ion Only')
p2 = ax.barh(paises, mix_vals, left=li_vals, color=c_mix, edgecolor='black', height=0.7, label='Mixed (Comparative)')
p3 = ax.barh(paises, next_vals, left=li_vals + mix_vals, color=c_next, edgecolor='black', height=0.7, label='Next-Gen Only')

# Add exact numbers inside bars (only if segment is wider than 4% to avoid overlapping text)
def add_labels(bars):
    for bar in bars:
        width = bar.get_width()
        if width > 4.0: 
            x = bar.get_x() + width / 2
            y = bar.get_y() + bar.get_height() / 2
            ax.text(x, y, f'{width:.1f}%', ha='center', va='center', color='white', fontsize=fs_bar, weight='bold')

add_labels(p1)
add_labels(p2)
add_labels(p3)

# Titles and Axes
ax.set_title('Internal Research Portfolio:\nLithium vs. Next-Generation Distribution', fontsize=fs1, pad=40, color='black')
ax.set_xlabel('Percentage of Internal Portfolio (%)', fontsize=fs2, labelpad=10, color='black')
ax.set_ylabel('Country of Origin', fontsize=fs2, labelpad=10, color='black')

ax.set_xlim(0, 100) # Force X-axis to go up to exactly 100%
ax.tick_params(axis='both', colors='black', labelsize=fs3)

# Legend at the bottom of the chart
#ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=3, fontsize=fs3, frameon=False)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=3, fontsize=fs3, frameon=False)

# Border adjustments
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

# Save Chart
if not os.path.exists(config.RESULTS_FOLDER):
    os.makedirs(config.RESULTS_FOLDER)
file_name_img = f'{config.RESULTS_FOLDER}/q4_11_2_portfolio_stacked_bar.pdf'
plt.savefig(file_name_img, dpi=300, bbox_inches='tight')
plt.close()
print(f"Stacked Bar Chart generated and saved to: {file_name_img}")

# ==============================================================================
# 9. LATEX TABLE GENERATION
# ==============================================================================
# Reverse order again for the table (highest NextGen to lowest, top to bottom)
df_top15_table = df_top15.sort_values(by=['NextGen_Only_Pct', 'Mixed_Pct'], ascending=False)

latex_str = r"""\begin{table}[htpb]
\centering
\begin{threeparttable}
\caption{National research portfolios decomposed by strictly exclusive and mixed technological focus.}
\label{tab:portfolio_distribution}
\begin{tabular}{ p{3.0cm} >{\raggedleft\arraybackslash}p{1.8cm} >{\raggedleft\arraybackslash}p{2.0cm} >{\raggedleft\arraybackslash}p{2.0cm} >{\raggedleft\arraybackslash}p{2.0cm} }
\hline
Country & Valid & Li-ion & Mixed & Next-Gen \\
& Pubs & Only (\%) & Focus (\%) & Only (\%) \\
\hline
"""

for pais, row in df_top15_table.iterrows():
    pais_safe = str(pais).replace('&', r'\&').replace('_', ' ')
    total = int(row['Portfolio_Total'])
    li_pct = row['Li_Only_Pct']
    mix_pct = row['Mixed_Pct']
    next_pct = row['NextGen_Only_Pct']
    
    latex_str += f"{pais_safe} & {total} & {li_pct:.1f} & {mix_pct:.1f} & {next_pct:.1f} \\\\\n"

latex_str += r"""\hline
\end{tabular}
\begin{tablenotes}
\small
\item Note: The table analyzes the internal portfolio of the top fifteen producing nations, bounded strictly to publications investigating lithium, sodium, potassium, or zinc-ion systems. The categories are mutually exclusive. Mixed Focus denotes documents that comparatively investigate lithium-ion alongside at least one of the targeted alternative chemistries.
\end{tablenotes}
\end{threeparttable}
\end{table}"""

# Save Table
file_name_tex = f'{config.RESULTS_FOLDER}/q4_11_2_portfolio_stacked_bar.tex'
with open(file_name_tex, 'w', encoding='utf-8') as f:
    f.write(latex_str)
print(f"Portfolio LaTeX Table generated and saved to: {file_name_tex}")

