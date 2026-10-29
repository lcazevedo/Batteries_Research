import utils
import config
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
import os

fs1 = 18

# 1. Load data
df_data = utils.get_df_data()

# 2. Filter valid documents (remove irrelevant ones)
mask_any_metal = (df_data['is_li_ion'] == True) | \
                 (df_data['is_na_ion'] == True) | \
                 (df_data['is_k_ion'] == True) | \
                 (df_data['is_zn_ion'] == True) | \
                 (df_data['is_other_metal'] == True)

df_valido = df_data[mask_any_metal].copy()

# Total documents for percentage calculation
total_docs = len(df_valido)

# 3. Create the 3 main masks for the Venn diagram
m_li = df_valido['is_li_ion'] == True
m_na = df_valido['is_na_ion'] == True
m_others = (df_valido['is_k_ion'] == True) | \
           (df_valido['is_zn_ion'] == True) | \
           (df_valido['is_other_metal'] == True)

# 4. Calculate the ACTUAL mathematical values
only_li = (m_li & ~m_na & ~m_others).sum()
only_na = (~m_li & m_na & ~m_others).sum()
li_and_na = (m_li & m_na & ~m_others).sum()

only_others = (~m_li & ~m_na & m_others).sum()
li_and_others = (m_li & ~m_na & m_others).sum()
na_and_others = (~m_li & m_na & m_others).sum()

all_three = (m_li & m_na & m_others).sum()

real_values = {
    '100': only_li,
    '010': only_na,
    '110': li_and_na,
    '001': only_others,
    '101': li_and_others,
    '011': na_and_others,
    '111': all_three
}

# 5. VISUAL WEIGHTS (Area sizes)
visual_subsets = (
    only_li * 0.6,       
    only_na * 12.0,       
    li_and_na * 20.0,     
    only_others * 3.0,   
    li_and_others * 3.5, 
    na_and_others * 4.0, 
    all_three * 4.0      
)

# 6. Configure and generate the chart with adjusted sizes
fig, ax = plt.subplots(figsize=(10, 8))

v = venn3(subsets=visual_subsets, 
          set_labels=('Lithium-ion', 'Sodium-ion', 'Other Alternative\nMetals (K, Zn, etc.)'),
          set_colors=('#4C72B0', '#55A868', '#DD8452'),
          alpha=0.7)

# 7. Overwrite text labels with ACTUAL VALUES and PERCENTAGES
for region_id, true_value in real_values.items():
    label = v.get_label_by_id(region_id)
    if label:
        # Calculate percentage relative to valid total
        pct = (true_value / total_docs) * 100
        
        # Insert text on two lines (Absolute value on top, percentage below)
        label.set_text(f"{true_value}\n({pct:.1f}%)")
        label.set_fontsize(12)
        # Removed label.set_fontweight('bold')
        
# 8. Improve group label font size
for text in v.set_labels:
    if text:
        text.set_fontsize(14)

# Removed weight='bold' from title
plt.title('Research Crossover in Metal-Ion Battery Technologies', fontsize=fs1, pad=20)
    
if not os.path.exists(config.RESULTS_FOLDER):
    os.makedirs(config.RESULTS_FOLDER)
    
file_name_img = f'{config.RESULTS_FOLDER}/q4_7_4_metals_plot_intersections.pdf'
plt.savefig(file_name_img, dpi=300, bbox_inches='tight')
plt.close()

print(f"Venn diagram generated and saved to: {file_name_img}")



# ==============================================================================
# 9. Export Venn data to a LaTeX table for analysis
# ==============================================================================

# Dictionary mapping Venn IDs to readable article labels
region_labels = {
    '100': 'Lithium-ion Only',
    '010': 'Sodium-ion Only',
    '001': 'Other Alternatives Only',
    '110': 'Lithium & Sodium',
    '101': 'Lithium & Others',
    '011': 'Sodium & Others',
    '111': 'Lithium, Sodium & Others'
}

latex_table = r"""\begin{table}[htpb]
\centering
\caption{Publication Counts by Intersection Region in the Venn Diagram}
\label{tab:venn_data_export}
\begin{tabular}{lrr}
\hline
\textbf{Intersection Region} & \textbf{Publications} & \textbf{\% of Total} \\
\hline
"""

# Dynamically compute rows
for region_id, label in region_labels.items():
    val = real_values[region_id]
    pct = (val / total_docs) * 100
    
    # Escape & character so LaTeX does not break the column
    safe_label = label.replace('&', r'\&')
    latex_table += f"{safe_label} & {val} & {pct:.1f}\\% \\\\\n"

# Total row for verification
total_sum = sum(real_values.values())
latex_table += r"""\hline
\textbf{Total Valid Documents} & \textbf{""" + str(total_sum) + r"""} & \textbf{100.0\%} \\
\hline
\end{tabular}
\end{table}"""

# Save file
file_name_tex = f'{config.RESULTS_FOLDER}/q4_7_4_metals_plot_intersections.tex'
with open(file_name_tex, 'w', encoding='utf-8') as f:
    f.write(latex_table)

print(f"Auxiliary table with Venn data exported to: {file_name_tex}")