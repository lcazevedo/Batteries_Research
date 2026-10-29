import utils
import config
import pandas as pd
import os


# 1. Load data
df_data = utils.get_df_data()

# 2. Filter valid documents
mask_any_metal = (df_data['is_li_ion'] == True) | \
                    (df_data['is_na_ion'] == True) | \
                    (df_data['is_k_ion'] == True) | \
                    (df_data['is_zn_ion'] == True) | \
                    (df_data['is_other_metal'] == True)

df_valido = df_data[mask_any_metal].copy()

# 3. Clean Country column
col_pais = config.COLUMN_FIRST_COUNTRY
df_valido = df_valido.dropna(subset=[col_pais])
df_valido[col_pais] = df_valido[col_pais].astype(str).str.strip()

# 4. Create mask for General Alternative Focus (without mathematical overlap)
# If an article studies Na and K, it counts as 1 "Alternative Focus"
df_valido['is_alt_focus'] = df_valido['is_na_ion'] | df_valido['is_k_ion'] | df_valido['is_zn_ion']

# 5. Group data by Country
df_paises = df_valido.groupby(col_pais).agg(
    Total_Docs=(col_pais, 'count'),
    Na_Docs=('is_na_ion', 'sum'),
    K_Docs=('is_k_ion', 'sum'),
    Zn_Docs=('is_zn_ion', 'sum'),
    Alt_Docs=('is_alt_focus', 'sum')
)

# 6. Isolate the top 15 science producers by absolute volume
df_top15 = df_paises.nlargest(15, 'Total_Docs').copy()

# 7. Calculate "Internal Focus" (National Penetration Rate)
df_top15['Na_pct'] = (df_top15['Na_Docs'] / df_top15['Total_Docs']) * 100
df_top15['K_pct'] = (df_top15['K_Docs'] / df_top15['Total_Docs']) * 100
df_top15['Zn_pct'] = (df_top15['Zn_Docs'] / df_top15['Total_Docs']) * 100
df_top15['Alt_pct'] = (df_top15['Alt_Docs'] / df_top15['Total_Docs']) * 100

# 8. STRATEGIC SORTING: Sort from the country most focused on alternatives to the least focused
df_top15 = df_top15.sort_values('Alt_pct', ascending=False)

# ==============================================================================
# 9. LATEX TABLE GENERATION (WITH THREEPARTTABLE AND TWO-LINE HEADER)
# ==============================================================================
# Removed \textbf{} and split the header into two logical lines
latex_str = r"""\begin{table}[htpb]
\centering
\begin{threeparttable}
\caption{Technological Specialization: Relative Geographic Focus on Alternative Metal-Ion Batteries}
\label{tab:geographic_specialization}
% Define: Country with 3cm (left-aligned) and the other 5 columns with 1.8cm (right-aligned)
\begin{tabular}{ p{3.0cm} >{\raggedleft\arraybackslash}p{1.8cm} >{\raggedleft\arraybackslash}p{1.8cm} >{\raggedleft\arraybackslash}p{1.8cm} >{\raggedleft\arraybackslash}p{1.8cm} >{\raggedleft\arraybackslash}p{2.0cm} }
\hline
Country & Total & Na-ion & K-ion & Zn-ion & Alternative \\
 & Pubs & (\%) & (\%) & (\%) & Focus (\%) \\
\hline
"""

# Populate table rows
for pais, row in df_top15.iterrows():
    pais_safe = str(pais).replace('&', r'\&').replace('_', ' ')
    
    total = int(row['Total_Docs'])
    na_pct = row['Na_pct']
    k_pct = row['K_pct']
    zn_pct = row['Zn_pct']
    alt_pct = row['Alt_pct']
    
    latex_str += f"{pais_safe} & {total} & {na_pct:.1f} & {k_pct:.1f} & {zn_pct:.1f} & {alt_pct:.1f} \\\\\n"

latex_str += r"""\hline
\end{tabular}
\begin{tablenotes}
\small
\item Note: The table evaluates the internal research focus of the top 15 highest-producing countries in the global battery literature. Percentages represent the proportion of each nation's total domestic publications dedicated to specific alternative working ions. The Alternative Focus column quantifies the absolute percentage of national literature investigating at least one of these three beyond-lithium technologies, accounting for internal document overlaps.
\end{tablenotes}
\end{threeparttable}
\end{table}"""

# 10. Save LaTeX table
if not os.path.exists(config.RESULTS_FOLDER):
    os.makedirs(config.RESULTS_FOLDER)
    
file_name_tex = f'{config.RESULTS_FOLDER}/q4_8_3_geographic_specialization_table.tex'
with open(file_name_tex, 'w', encoding='utf-8') as f:
    f.write(latex_str)
    
print(f"Geographic Specialization LaTeX Table generated and saved to: {file_name_tex}")

