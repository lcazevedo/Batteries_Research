import utils
import config
import pandas as pd
import os

# 1. Load data
df_data = utils.get_df_data()
df_phrases = utils.get_df_data_phrases()

# Standardize merge column name (DOI) to avoid case sensitivity errors
df_data_merge = df_data.rename(columns={'DOI': 'doi', 'Doi': 'doi', 'DOi': 'doi'})
df_phrases_merge = df_phrases.rename(columns={'DOI': 'doi'})

# Convert DOIs to lowercase strings to ensure exact matching
df_data_merge['doi'] = df_data_merge['doi'].astype(str).str.lower()
df_phrases_merge['doi'] = df_phrases_merge['doi'].astype(str).str.lower()

# 2. Filter valid documents
mask_any_metal = (df_data_merge['is_li_ion'] == True) | \
                 (df_data_merge['is_na_ion'] == True) | \
                 (df_data_merge['is_k_ion'] == True) | \
                 (df_data_merge['is_zn_ion'] == True) | \
                 (df_data_merge['is_other_metal'] == True)

df_valido = df_data_merge[mask_any_metal].copy()

# 3. Merge datasets
cols_to_merge = ['doi', 'is_li_ion', 'is_na_ion', 'is_k_ion', 'is_zn_ion']
df_merged = pd.merge(df_phrases_merge, df_valido[cols_to_merge], on='doi', how='inner')

# 4. Clean Challenges
df_merged = df_merged.dropna(subset=['challenges_group'])
termos_invalidos = ['No Challenge', 'None', 'N/A', '']
df_merged = df_merged[~df_merged['challenges_group'].isin(termos_invalidos)]

# 5. Calculate Frequencies by Metal
metals = {
    'Li-ion': 'is_li_ion',
    'Na-ion': 'is_na_ion',
    'K-ion': 'is_k_ion',
    'Zn-ion': 'is_zn_ion'
}

dict_challenges = {}
for metal_label, col_name in metals.items():
    df_metal = df_merged[df_merged[col_name] == True]
    counts = df_metal['challenges_group'].value_counts()
    percentages = (counts / counts.sum()) * 100
    dict_challenges[metal_label] = percentages
    
df_challenges = pd.DataFrame(dict_challenges).fillna(0)

# 6. Select and Separate Challenges
top_challenges = set()
for metal_label in metals.keys():
    top_n = df_challenges[metal_label].nlargest(15).index.tolist()
    top_challenges.update(top_n)
    
df_table = df_challenges.loc[list(top_challenges)].copy()

# Split into two groups based on Li-ion dominance
max_alt_value = df_table[['Na-ion', 'K-ion', 'Zn-ion']].max(axis=1)
mask_li_dominant = df_table['Li-ion'] > max_alt_value

df_li_dominant = df_table[mask_li_dominant].copy()
df_alt_dominant = df_table[~mask_li_dominant].copy()

# Sort each group individually
df_li_dominant['mean'] = df_li_dominant.mean(axis=1)
df_li_dominant = df_li_dominant.sort_values(by='mean', ascending=False).drop(columns=['mean'])

df_alt_dominant['mean'] = df_alt_dominant.mean(axis=1)
df_alt_dominant = df_alt_dominant.sort_values(by='mean', ascending=False).drop(columns=['mean'])

# 7. Generate LaTeX code
latex_str = r"""\begin{table}[htpb]
\centering
\begin{threeparttable}
\caption{Research Challenges by Metal-Ion Battery Type}
\label{tab:q4_7_2_challenges_by_metal}
\begin{tabular}{clrrrr}
\hline
& \textbf{Challenge Group} & \textbf{Li-ion} & \textbf{Na-ion} & \textbf{K-ion} & \textbf{Zn-ion} \\
\hline
"""

# Updated helper function to support two lines of text
def format_group(df_group, text_line1, text_line2):
    group_str = ""
    n_rows = len(df_group)
    for i, (challenge, row) in enumerate(df_group.iterrows()):
        safe_challenge = str(challenge).replace('&', r'\&').replace('%', r'\%')
        
        li_val = f"{row['Li-ion']:.1f}\\%" if row['Li-ion'] > 0 else "-"
        na_val = f"{row['Na-ion']:.1f}\\%" if row['Na-ion'] > 0 else "-"
        k_val = f"{row['K-ion']:.1f}\\%" if row['K-ion'] > 0 else "-"
        zn_val = f"{row['Zn-ion']:.1f}\\%" if row['Zn-ion'] > 0 else "-"
        
        if i == 0:
            # Create LaTeX block to break the vertical text line while keeping it centered
            rotated_text = f"\\rotatebox[origin=c]{{90}}{{\\begin{{tabular}}{{@{{}}c@{{}}}}\\textbf{{{text_line1}}} \\\\ \\textbf{{{text_line2}}}\\end{{tabular}}}}"
            group_str += f"\\multirow{{{n_rows}}}{{*}}{{{rotated_text}}} & {safe_challenge} & {li_val} & {na_val} & {k_val} & {zn_val} \\\\\n"
        else:
            group_str += f" & {safe_challenge} & {li_val} & {na_val} & {k_val} & {zn_val} \\\\\n"
    return group_str

# Call the function passing the separated words
latex_str += format_group(df_li_dominant, "Li-ion", "Dominant")
latex_str += r"\hline" + "\n"
latex_str += format_group(df_alt_dominant, "Alternatives", "Dominant")

latex_str += r"""\hline
\end{tabular}
\begin{tablenotes}
\small
\item \textit{Note:} Values represent the relative frequency of each challenge group within the specific metal-ion category. The table is divided into challenges where lithium-ion represents the highest relative frequency, and challenges where at least one emerging alternative metal surpasses lithium-ion.
\end{tablenotes}
\end{threeparttable}
\end{table}"""

# 8. Save string to a .tex file
if not os.path.exists(config.RESULTS_FOLDER):
    os.makedirs(config.RESULTS_FOLDER)
    
file_name_latex = f'{config.RESULTS_FOLDER}/q4_7_2_challenges_by_metal.tex'
with open(file_name_latex, 'w', encoding='utf-8') as f:
    f.write(latex_str)
    
print(f"Table generated and saved to: {file_name_latex}")