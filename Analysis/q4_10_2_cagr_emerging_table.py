import utils
import config
import pandas as pd
import os


# 1. Load data
df_data = utils.get_df_data()

# 2. Clean and standardize Year column
col_ano = config.COLUMN_YEAR
df_valido = df_data.dropna(subset=[col_ano]).copy()
df_valido[col_ano] = pd.to_numeric(df_valido[col_ano], errors='coerce')
df_valido = df_valido.dropna(subset=[col_ano])

# 3. Filter only the years that form the analyzed decade (2014 and 2024)
ano_inicio = 2014
ano_fim = 2024
df_anos = df_valido[df_valido[col_ano].isin([ano_inicio, ano_fim])]

# 4. Calculate absolute volumes for the start year (2014)
v_inicio_na = df_anos[df_anos[col_ano] == ano_inicio]['is_na_ion'].sum()
v_inicio_k = df_anos[df_anos[col_ano] == ano_inicio]['is_k_ion'].sum()
v_inicio_zn = df_anos[df_anos[col_ano] == ano_inicio]['is_zn_ion'].sum()

# 5. Calculate absolute volumes for the end year (2024)
v_fim_na = df_anos[df_anos[col_ano] == ano_fim]['is_na_ion'].sum()
v_fim_k = df_anos[df_anos[col_ano] == ano_fim]['is_k_ion'].sum()
v_fim_zn = df_anos[df_anos[col_ano] == ano_fim]['is_zn_ion'].sum()

# 6. Calculate CAGR for each technology: ((End / Start) ^ (1 / Time)) - 1
t = ano_fim - ano_inicio # 10-year time window

cagr_na = ((v_fim_na / v_inicio_na) ** (1 / t) - 1) * 100 if v_inicio_na > 0 else 0
cagr_k = ((v_fim_k / v_inicio_k) ** (1 / t) - 1) * 100 if v_inicio_k > 0 else 0
cagr_zn = ((v_fim_zn / v_inicio_zn) ** (1 / t) - 1) * 100 if v_inicio_zn > 0 else 0

# 7. Create DataFrame and sort by highest acceleration (CAGR)
dados_cagr = [
    {'Tech': 'Sodium-ion (Na)', 'Inicio': int(v_inicio_na), 'Fim': int(v_fim_na), 'CAGR': cagr_na},
    {'Tech': 'Potassium-ion (K)', 'Inicio': int(v_inicio_k), 'Fim': int(v_fim_k), 'CAGR': cagr_k},
    {'Tech': 'Zinc-ion (Zn)', 'Inicio': int(v_inicio_zn), 'Fim': int(v_fim_zn), 'CAGR': cagr_zn}
]
df_cagr = pd.DataFrame(dados_cagr)
df_cagr = df_cagr.sort_values(by='CAGR', ascending=False)

# ==============================================================================
# 8. LATEX TABLE GENERATION (SCIENTIFIC STANDARD)
# ==============================================================================
latex_str = r"""\begin{table}[H]
\centering
\begin{threeparttable}
\caption{Decadal Compound Annual Growth Rate (CAGR) of Emerging Metal-Ion Batteries}
\label{tab:q4_10_2_cagr_emerging_table}
% Fixed-width configuration with headers aligned via \shortstack
\begin{tabular}{ p{3.5cm} >{\raggedleft\arraybackslash}p{2.0cm} >{\raggedleft\arraybackslash}p{2.0cm} >{\raggedleft\arraybackslash}p{2.5cm} }
\hline
Emerging & \multicolumn{1}{>{\raggedleft\arraybackslash}p{2.0cm}}{\shortstack[r]{2014 \\ Volume}} & \multicolumn{1}{>{\raggedleft\arraybackslash}p{2.0cm}}{\shortstack[r]{2024 \\ Volume}} & \multicolumn{1}{>{\raggedleft\arraybackslash}p{2.5cm}}{\shortstack[r]{Decadal \\ CAGR (\%)\tnote{*}}} \\
\hline
"""

for _, row in df_cagr.iterrows():
    tech = row['Tech']
    v_inicio = row['Inicio']
    v_fim = row['Fim']
    cagr = row['CAGR']
    
    latex_str += f"{tech} & {v_inicio} & {v_fim} & {cagr:.1f} \\\\\n"

latex_str += r"""\hline
\end{tabular}
\begin{tablenotes}
\small
\item[*] The Compound Annual Growth Rate (CAGR) measures the mean annual growth rate necessary to expand the absolute publication volume from the 2014 baseline to the 2024 peak, isolating the acceleration velocity of each individual alternative chemistry.
\end{tablenotes}
\end{threeparttable}
\end{table}"""

# 9. Save LaTeX file
if not os.path.exists(config.RESULTS_FOLDER):
    os.makedirs(config.RESULTS_FOLDER)
    
file_name_tex = f'{config.RESULTS_FOLDER}/q4_10_2_cagr_emerging_table.tex'
with open(file_name_tex, 'w', encoding='utf-8') as f:
    f.write(latex_str)
    
print(f"CAGR LaTeX Table generated and saved to: {file_name_tex}")

