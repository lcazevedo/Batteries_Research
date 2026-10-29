import utils
import config
import pandas as pd
import os

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

# 4. Create Absolute Count Masks (overlap allowed)
df_valido['has_li'] = df_valido['is_li_ion'] == True
df_valido['has_nextgen'] = (df_valido['is_na_ion'] == True) | \
                            (df_valido['is_k_ion'] == True) | \
                            (df_valido['is_zn_ion'] == True)

# 5. Group by Country
df_paises = df_valido.groupby(col_pais).agg(
    Total_Docs=(col_pais, 'count'),
    Li_Docs=('has_li', 'sum'),
    NextGen_Docs=('has_nextgen', 'sum')
)

# 6. Select top 15 producers for analysis
df_top15 = df_paises.nlargest(15, 'Total_Docs').copy()

# 7. Calculate Ratio and convert to percentage
df_top15['Escape_Ratio'] = (df_top15['NextGen_Docs'] / df_top15['Li_Docs']) * 100

# 8. SORTING: From highest escape ratio to lowest escape ratio
df_top15 = df_top15.sort_values(by='Escape_Ratio', ascending=False)

# ==============================================================================
# 9. LATEX TABLE GENERATION
# ==============================================================================
# Fixed-width tabular configuration with right-aligned numeric columns
latex_str = r"""\begin{table}[htpb]
\centering
\begin{threeparttable}
\caption{The Beyond-Lithium Pivot: Ratio of Next-Generation to Lithium-ion Publications}
\label{tab:q4_11_3_escape_ratio_table}
% Country (3.5cm) and numeric columns (right-aligned)
\begin{tabular}{ p{3.5cm} >{\raggedleft\arraybackslash}p{1.8cm} >{\raggedleft\arraybackslash}p{1.8cm} >{\raggedleft\arraybackslash}p{1.8cm} >{\raggedleft\arraybackslash}p{3cm} }
\hline
Country & {\shortstack{Valid \\ Pubs}} & {\shortstack{Li-ion \\ Pubs}} & {\shortstack{Next-Gen \\ Pubs}} & {\shortstack{Next-Gen per \\ Li-ion (\%)}} \\
\hline
"""

for pais, row in df_top15.iterrows():
    pais_safe = str(pais).replace('&', r'\&').replace('_', ' ')
    total = int(row['Total_Docs'])
    li_docs = int(row['Li_Docs'])
    next_docs = int(row['NextGen_Docs'])
    ratio = row['Escape_Ratio']
    
    latex_str += f"{pais_safe} & {total} & {li_docs} & {next_docs} & {ratio:.1f} \\\\\n"

latex_str += r"""\hline
\end{tabular}
\begin{tablenotes}
\small
\item Note: The table highlights the comparative research velocity between emerging and established chemistries among the top fifteen producing nations. The metric normalizes the data, illustrating the proportional volume of next-generation documents (investigating sodium, potassium, or zinc) published relative to the lithium-ion research output within a given country.
\end{tablenotes}
\end{threeparttable}
\end{table}"""

# 10. Save table
if not os.path.exists(config.RESULTS_FOLDER):
    os.makedirs(config.RESULTS_FOLDER)
    
file_name_tex = f'{config.RESULTS_FOLDER}/q4_11_3_escape_ratio_table.tex'
with open(file_name_tex, 'w', encoding='utf-8') as f:
    f.write(latex_str)
    
print(f"Escape Ratio LaTeX Table generated and saved to: {file_name_tex}")

