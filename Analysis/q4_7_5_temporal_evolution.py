import utils
import config
import pandas as pd
import matplotlib.pyplot as plt
import os

fs1 = 24
fs2 = 18

# 1. Load data
df_data = utils.get_df_data()

# 2. Identify and clean the Year column
ano_col = 'Year Published' if 'Year Published' in df_data.columns else 'Year'
df_data = df_data.dropna(subset=[ano_col])
df_data[ano_col] = df_data[ano_col].astype(int)

# 3. Filter valid documents
mask_any_metal = (df_data['is_li_ion'] == True) | \
                 (df_data['is_na_ion'] == True) | \
                 (df_data['is_k_ion'] == True) | \
                 (df_data['is_zn_ion'] == True) | \
                 (df_data['is_other_metal'] == True)

df_valido = df_data[mask_any_metal].copy()

# 4. DATA CLEANING FILTER: Keep only up to 2025
df_valido = df_valido[df_valido[ano_col] <= 2025]

# 5. Group by year and sum the occurrences
cols_metais = {
    'is_li_ion': 'Lithium-ion',
    'is_na_ion': 'Sodium-ion',
    'is_zn_ion': 'Zinc-ion',
    'is_k_ion': 'Potassium-ion',
    'is_other_metal': 'Other Alternatives'
}

df_grouped = df_valido.groupby(ano_col)[list(cols_metais.keys())].sum()

# Filter to display in the table and chart only from year 2000 onwards
df_grouped = df_grouped[df_grouped.index >= 2000]

# ==============================================================================
# 6. CHART GENERATION
# ==============================================================================
fig, ax = plt.subplots(figsize=(12, 7))

cores = {
    'is_li_ion': '#4C72B0',      
    'is_na_ion': '#55A868',      
    'is_zn_ion': '#8172B3',      
    'is_k_ion': '#C44E52',       
    'is_other_metal': '#DD8452'  
}

marcadores = {
    'is_li_ion': 'o',  
    'is_na_ion': 's',  
    'is_zn_ion': 'D',  
    'is_k_ion': '^',   
    'is_other_metal': 'v' 
}

# Plot line for each column
for col, label in cols_metais.items():
    ax.plot(df_grouped.index, df_grouped[col], 
            label=label, 
            color=cores[col], 
            marker=marcadores[col], 
            linewidth=2.5,
            markersize=6,
            alpha=0.85)
    
# Titles and Axes
ax.set_title('Temporal Evolution of Metal-Ion Battery Publications', fontsize=fs1, pad=20, color='black')
ax.set_xlabel('Year of Publication', fontsize=fs2, labelpad=10, color='black')
ax.set_ylabel('Number of Publications', fontsize=fs2, labelpad=10, color='black')

# STRICT-BLACK VISUAL ADJUSTMENTS
# 1. Grid lines in black
ax.grid(True, linestyle='--', alpha=0.5, color='black')

# 2. Axis tick labels in black
ax.tick_params(axis='both', colors='black', labelsize=12)

# 3. Chart borders (Spines) in solid black
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.0)

ax.legend(fontsize=fs2, loc='upper left')
ax.set_xlim(df_grouped.index.min(), 2025)
ax.tick_params(axis='both', colors='black', labelsize=fs2)

plt.tight_layout()

if not os.path.exists(config.RESULTS_FOLDER):
    os.makedirs(config.RESULTS_FOLDER)
    
file_name_img = f'{config.RESULTS_FOLDER}/q4_7_5_temporal_evolution.pdf'
plt.savefig(file_name_img, dpi=300, bbox_inches='tight')
plt.close()

print(f"Temporal evolution chart generated and saved to: {file_name_img}")

# ==============================================================================
# 7. LATEX TABLE GENERATION (WITH THREEPARTTABLE)
# ==============================================================================
latex_str = r"""\begin{table}[htpb]
\centering
\begin{threeparttable}
\caption{Annual Publication Volumes by Metal-Ion Battery Category}
\label{tab:q4_7_5_temporal_evolution}
\begin{tabular}{lrrrrr}
\hline
\textbf{Year} & \textbf{Li-ion} & \textbf{Na-ion} & \textbf{Zn-ion} & \textbf{K-ion} & \textbf{Other Metals} \\
\hline
"""

# Iterate over chronologically sorted data
for year, row in df_grouped.iterrows():
    # Ensure year and counts are printed as clean integers
    y_str = str(int(year))
    li_val = int(row['is_li_ion'])
    na_val = int(row['is_na_ion'])
    zn_val = int(row['is_zn_ion'])
    k_val = int(row['is_k_ion'])
    other_val = int(row['is_other_metal'])
    
    latex_str += f"{y_str} & {li_val} & {na_val} & {zn_val} & {k_val} & {other_val} \\\\\n"

latex_str += r"""\hline
\end{tabular}
\begin{tablenotes}
\small
\item Note: The table presents the absolute volume of publications per year from 2000 to 2025. Documents intersecting multiple categories are accounted for in each respective column. 
\end{tablenotes}
\end{threeparttable}
\end{table}"""

file_name_tex = f'{config.RESULTS_FOLDER}/q4_7_5_temporal_evolution.tex'
with open(file_name_tex, 'w', encoding='utf-8') as f:
    f.write(latex_str)
    
print(f"Temporal evolution LaTeX table generated and saved to: {file_name_tex}")