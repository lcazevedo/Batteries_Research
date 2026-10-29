import utils
import config
import pandas as pd
import matplotlib.pyplot as plt
import os

fs1 = 22
fs2 = 18

# 1. Load data
df_data = utils.get_df_data()

# 2. Filter valid documents (strict metal filter)
mask_any_metal = (df_data['is_li_ion'] == True) | \
                    (df_data['is_na_ion'] == True) | \
                    (df_data['is_k_ion'] == True) | \
                    (df_data['is_zn_ion'] == True) | \
                    (df_data['is_other_metal'] == True)

df_valido = df_data[mask_any_metal].copy()
total_docs = len(df_valido)

# 3. Calculate absolute sums and percentages for the 3 target metals
contagens = {
    'Sodium-ion (Na)': df_valido['is_na_ion'].sum(),
    'Potassium-ion (K)': df_valido['is_k_ion'].sum(),
    'Zinc-ion (Zn)': df_valido['is_zn_ion'].sum()
}

# Convert to percentage
porcentagens = {metal: (count / total_docs) * 100 for metal, count in contagens.items()}

# Sort dictionary from smallest to largest
# (ax.barh draws from bottom to top, so the largest will be at the top)
porcentagens_ordenadas = dict(sorted(porcentagens.items(), key=lambda item: item[1]))

metais = list(porcentagens_ordenadas.keys())
valores_pct = list(porcentagens_ordenadas.values())

# 4. Configure color palette (Same colors as Temporal Chart for consistency)
cores_map = {
    'Sodium-ion (Na)': '#55A868',    # Green
    'Zinc-ion (Zn)': '#8172B3',      # Purple
    'Potassium-ion (K)': '#C44E52'   # Red
}
cores_barras = [cores_map[m] for m in metais]

# 5. Chart generation
fig, ax = plt.subplots(figsize=(10, 5))

barras = ax.barh(metais, valores_pct, color=cores_barras, height=0.6, alpha=0.85, edgecolor='black')

# Add exact values at the end of each bar
for barra in barras:
    largura = barra.get_width()
    # Position text slightly to the right of the bar end
    ax.text(largura + 0.5, 
            barra.get_y() + barra.get_height()/2, 
            f'{largura:.1f}%', 
            va='center', ha='left', fontsize=fs2, color='black')

# Titles and Axes
ax.set_title('Global Penetration Rate of Alternative Metal-Ion Technologies', fontsize=fs1, pad=20, color='black', loc='right')
ax.set_xlabel('Percentage of Global Research Efforts (%)', fontsize=fs2, labelpad=10, color='black')

# STRICT-BLACK VISUAL ADJUSTMENTS
# Adjust X-axis limit to leave space for the longest bar's text
max_val = max(valores_pct)
ax.set_xlim(0, max_val + 5)

# Vertical grid in black
ax.grid(True, axis='x', linestyle='--', alpha=0.5, color='black')
ax.set_axisbelow(True) # Ensure grid stays behind the bars

# Axis labels in black
ax.tick_params(axis='both', colors='black', labelsize=fs2)

# Borders (Spines) in solid black
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.0)
    
# Hide top and right borders for a cleaner look (optional, but recommended)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

# 6. Save chart
if not os.path.exists(config.RESULTS_FOLDER):
    os.makedirs(config.RESULTS_FOLDER)
    
file_name_img = f'{config.RESULTS_FOLDER}/q4_8_1_global_penetration_bar.pdf'
plt.savefig(file_name_img, dpi=300, bbox_inches='tight')
plt.close()

print(f"Horizontal bar chart generated and saved to: {file_name_img}")
