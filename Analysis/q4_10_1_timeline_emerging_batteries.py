import utils
import config
import pandas as pd
import matplotlib.pyplot as plt
import os

# ==========================================
# GLOBAL FONT CONFIGURATION
# ==========================================
fs1 = 26  # Main Title font size
fs2 = 22  # Axis Labels font size (X and Y)
fs3 = 22  # Text font size (X-axis, Legends)

# 1. Load data
df_data = utils.get_df_data()

# 2. Clean Year column and apply decadal temporal filter (2014 to 2024)
# Deliberately exclude 2025 to avoid incomplete-year distortions
col_ano = config.COLUMN_YEAR
df_valido = df_data.dropna(subset=[col_ano]).copy()
df_valido[col_ano] = pd.to_numeric(df_valido[col_ano], errors='coerce')
df_valido = df_valido.dropna(subset=[col_ano])

df_decada = df_valido[(df_valido[col_ano] >= 2014) & (df_valido[col_ano] <= 2024)].copy()

# 3. Group by Year and sum occurrences (True) of each technology
df_evolucao = df_decada.groupby(col_ano).agg(
    Na_Docs=('is_na_ion', 'sum'),
    K_Docs=('is_k_ion', 'sum'),
    Zn_Docs=('is_zn_ion', 'sum')
).reset_index()

# Ensure year is treated as an integer for the X-axis
df_evolucao[col_ano] = df_evolucao[col_ano].astype(int)

# ==============================================================================
# 4. LINE CHART GENERATION
# ==============================================================================
fig, ax = plt.subplots(figsize=(12, 8))

anos = df_evolucao[col_ano]
na_vals = df_evolucao['Na_Docs']
k_vals = df_evolucao['K_Docs']
zn_vals = df_evolucao['Zn_Docs']

# Plot lines 
# Modern high-contrast colors (Orange for Sodium, Purple for Potassium, Green for Zinc)
ax.plot(anos, na_vals, color='#E67E22', linewidth=3.5, marker='o', markersize=12, markeredgecolor='black', label='Sodium-ion (Na)')
ax.plot(anos, k_vals, color='#9B59B6', linewidth=3.5, marker='o', markersize=12, markeredgecolor='black', label='Potassium-ion (K)')
ax.plot(anos, zn_vals, color='#2ECC71', linewidth=3.5, marker='o', markersize=12, markeredgecolor='black', label='Zinc-ion (Zn)')

# Titles and Axes
ax.set_title('Decadal Evolution: Attention on Emerging Metal-Ion\nBatteries (2014-2024)', fontsize=fs1, pad=20, color='black')
ax.set_xlabel('Publication Year', fontsize=fs2, labelpad=10, color='black')
ax.set_ylabel('Absolute Publication Volume', fontsize=fs2, labelpad=10, color='black')

# X-axis adjustment (Force showing all years, step by 1)
ax.set_xticks(anos)
ax.set_xticklabels(anos, rotation=45, ha='right')

# Grid and general design
ax.tick_params(axis='both', colors='black', labelsize=fs3)
ax.grid(True, linestyle=':', alpha=0.6, color='black', zorder=0)

# Position legend in the upper left corner to avoid covering lines rising on the right
ax.legend(loc='upper left', fontsize=fs3, frameon=True, edgecolor='black', facecolor='white', framealpha=0.9)

# Solid black borders
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.5)

plt.tight_layout()

# 5. Save Chart
if not os.path.exists(config.RESULTS_FOLDER):
    os.makedirs(config.RESULTS_FOLDER)
    
file_name_img = f'{config.RESULTS_FOLDER}/q4_10_1_timeline_emerging_batteries.pdf'
plt.savefig(file_name_img, dpi=300, bbox_inches='tight')
plt.close()

print(f"Timeline Chart generated and saved to: {file_name_img}")

# ==============================================================================
# 6. LATEX TABLE GENERATION (TEMPORAL EVOLUTION)
# ==============================================================================
# Ensure data is sorted chronologically from oldest to newest
df_evolucao = df_evolucao.sort_values(by=col_ano, ascending=True)

latex_str = r"""\begin{table}[htpb]
\centering
\begin{threeparttable}
\caption{Decadal Evolution: Absolute Publication Volume of Emerging Metal-Ion Batteries (2014-2024)}
\label{tab:q4_10_1_timeline_emerging_batteries}
% Columns: Year (Centered, 2.5cm) and 3 right-aligned data columns (2.5cm)
\begin{tabular}{ >{\centering\arraybackslash}p{2.5cm} >{\raggedleft\arraybackslash}p{2.5cm} >{\raggedleft\arraybackslash}p{2.5cm} >{\raggedleft\arraybackslash}p{2.5cm} }
\hline
\multicolumn{1}{c}{\shortstack{Publication \\ Year}} & \multicolumn{1}{c}{\shortstack{Sodium-ion \\ Pubs}} & \multicolumn{1}{c}{\shortstack{Potassium-ion \\ Pubs}} & \multicolumn{1}{c}{\shortstack{Zinc-ion \\ Pubs}} \\
\hline
"""

for _, row in df_evolucao.iterrows():
    ano = int(row[col_ano])
    na_docs = int(row['Na_Docs'])
    k_docs = int(row['K_Docs'])
    zn_docs = int(row['Zn_Docs'])
    
    latex_str += f"{ano} & {na_docs} & {k_docs} & {zn_docs} \\\\\n"

latex_str += r"""\hline
\end{tabular}
\begin{tablenotes}
\small
\item Note: The table details the absolute number of publications dedicated to sodium, potassium, and zinc-ion technologies annually from 2014 to 2024. The data captures the comparative historical growth and current dimensional scale of each targeted beyond-lithium field, explicitly excluding documents published in 2025 to avoid partial-year statistical distortions.
\end{tablenotes}
\end{threeparttable}
\end{table}"""

# Save LaTeX file
file_name_tex = f'{config.RESULTS_FOLDER}/q4_10_1_timeline_emerging_batteries.tex'
with open(file_name_tex, 'w', encoding='utf-8') as f:
    f.write(latex_str)
    
print(f"Temporal Evolution LaTeX Table generated and saved to: {file_name_tex}")