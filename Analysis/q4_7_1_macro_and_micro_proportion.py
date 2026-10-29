import utils
import config

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df_data = utils.get_df_data()

# 2. Filter out irrelevant documents
# Only keep the row if AT LEAST ONE of the metals is True
mask_any_metal = (df_data['is_li_ion'] == True) | \
                 (df_data['is_na_ion'] == True) | \
                 (df_data['is_k_ion'] == True) | \
                 (df_data['is_zn_ion'] == True) | \
                 (df_data['is_other_metal'] == True)

df_valido = df_data[mask_any_metal].copy()
total_docs = len(df_valido)

# 3. Create MACRO masks
col_alt = ['is_na_ion', 'is_k_ion', 'is_zn_ion', 'is_other_metal']
has_any_alt = df_valido[col_alt].any(axis=1)

mask_li_only = df_valido['is_li_ion'] & ~has_any_alt
mask_pure_alt = ~df_valido['is_li_ion'] & has_any_alt
mask_mixed = df_valido['is_li_ion'] & has_any_alt

# MACRO counts
count_li_only = mask_li_only.sum()
count_pure_alt = mask_pure_alt.sum()
count_mixed = mask_mixed.sum()

# 4. Helper function for MICRO breakdown
def calcular_micro(mask_grupo):
    df_grupo = df_valido[mask_grupo]
    total_grupo = len(df_grupo)
    if total_grupo == 0:
        return {"Na": 0, "K": 0, "Zn": 0, "Other": 0}, total_grupo
        
    contagens = {
        "Na-ion": df_grupo['is_na_ion'].sum(),
        "K-ion": df_grupo['is_k_ion'].sum(),
        "Zn-ion": df_grupo['is_zn_ion'].sum(),
        "Other Metals": df_grupo['is_other_metal'].sum()
    }
    return contagens, total_grupo

micro_pure_alt, total_pure_alt = calcular_micro(mask_pure_alt)
micro_mixed, total_mixed = calcular_micro(mask_mixed)

# 5. Generate LaTeX code
latex_str = r"""\begin{table}[htpb]
\centering
\caption{Macro and Micro Proportion of Metal-Ion Battery Publications}
\label{tab:macro_micro_proportion}
\begin{tabular}{lrrr}
\hline
\textbf{Category} & \textbf{Publications} & \textbf{\% of Total} & \textbf{\% within Group} \\
\hline
\textbf{Lithium-Only} & \textbf{""" + str(count_li_only) + r"""} & \textbf{""" + f"{(count_li_only/total_docs)*100:.1f}\%" + r"""} & \textbf{-} \\
\hline
\textbf{Pure Alternative} & \textbf{""" + str(count_pure_alt) + r"""} & \textbf{""" + f"{(count_pure_alt/total_docs)*100:.1f}\%" + r"""} & \textbf{-} \\
"""
# Insert micro rows for Pure Alternative
if total_pure_alt > 0:
    for metal, count in sorted(micro_pure_alt.items(), key=lambda item: item[1], reverse=True):
        latex_str += f"\\hspace{{4mm}} $\\hookrightarrow$ {metal} & {count} & {(count/total_docs)*100:.1f}\\% & {(count/total_pure_alt)*100:.1f}\\% \\\\\n"

latex_str += r"""\hline
\textbf{Mixed (Comparative)} & \textbf{""" + str(count_mixed) + r"""} & \textbf{""" + f"{(count_mixed/total_docs)*100:.1f}\%" + r"""} & \textbf{-} \\
"""
# Insert micro rows for Mixed
if total_mixed > 0:
    for metal, count in sorted(micro_mixed.items(), key=lambda item: item[1], reverse=True):
        latex_str += f"\\hspace{{4mm}} $\\hookrightarrow$ {metal} & {count} & {(count/total_docs)*100:.1f}\\% & {(count/total_mixed)*100:.1f}\\% \\\\\n"

latex_str += r"""\hline
\textbf{Total Relevant Publications} & \textbf{""" + str(total_docs) + r"""} & \textbf{100.0\%} & \textbf{-} \\
\hline
\multicolumn{4}{l}{\footnotesize \textit{Note: Sub-category percentages may sum to >100\% as studies can cover multiple metals.}} \\
\end{tabular}
\end{table}"""

# Save string to a .tex file
file_name_latex = f'{config.RESULTS_FOLDER}/q4_7_1_macro_and_micro_proportion.tex'
with open(file_name_latex, 'w', encoding='utf-8') as f:
    f.write(latex_str)