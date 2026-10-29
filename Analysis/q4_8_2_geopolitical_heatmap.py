import utils
import config
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm  # Required import for Log scale
import os

fs1 = 24  # Main Title font size
fs2 = 20  # Axis Labels (X and Y) and Colorbar Title font size
fs3 = 20  # Numbers font size (inside cells and on X/Y axis ticks)

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

# 4. Group by Country and sum
df_paises = df_valido.groupby(col_pais)[['is_na_ion', 'is_k_ion', 'is_zn_ion']].sum()

# 5. Select Top 15 Countries in alternatives and sort columns
df_paises['Total_Alt'] = df_paises.sum(axis=1)
df_top15 = df_paises.nlargest(15, 'Total_Alt').copy()
df_top15 = df_top15.drop(columns=['Total_Alt'])

rename_cols = {
    'is_na_ion': 'Sodium-ion\n(Na)',
    'is_k_ion': 'Potassium-ion\n(K)',
    'is_zn_ion': 'Zinc-ion\n(Zn)'
}
df_top15 = df_top15.rename(columns=rename_cols)
ordem_colunas = ['Sodium-ion\n(Na)', 'Potassium-ion\n(K)', 'Zinc-ion\n(Zn)']
df_top15 = df_top15[ordem_colunas]

# ==========================================
# 6. CHART GENERATION
# ==========================================
fig, ax = plt.subplots(figsize=(10, 8))

# Use norm=LogNorm() for logarithmic color scaling
heatmap = sns.heatmap(
    df_top15, 
    cmap='YlGnBu', 
    annot=True, 
    fmt='d', 
    linewidths=1.0, 
    linecolor='black',
    norm=LogNorm(), # Apply logarithmic scale to colors
    annot_kws={'size': fs3, 'color': 'black'} # Font of numbers INSIDE cells
)

for text in ax.texts:
    try:
        # Get numeric value written in cell
        valor = int(text.get_text())
        # If greater than 2000 (very dark blue cells), change font to white
        if valor > 2000:
            text.set_color('white')
        else:
            text.set_color('black')
    except ValueError:
        pass

# 7. Visual Adjustments and Texts
ax.set_title('Absolute Publication Volume by Country and Technology', fontsize=fs1, pad=20, color='black')
ax.set_ylabel('Country of Origin (First Author)', fontsize=fs2, labelpad=10, color='black')
ax.set_xlabel('Alternative Metal-Ion Technology', fontsize=fs2, labelpad=10, color='black')

# X and Y scale font
ax.tick_params(axis='both', colors='black', labelsize=fs3)
plt.xticks(rotation=0) 

# Add thick border around the entire heatmap
for _, spine in ax.spines.items():
    spine.set_visible(True)
    spine.set_linewidth(1.5)
    spine.set_edgecolor('black')

# 8. Adjust Side Colorbar
cbar = heatmap.collections[0].colorbar
# Configure colorbar title font
cbar.set_label('Number of Publications (Logarithmic Scale)', size=fs2, color='black', labelpad=10)
# Configure colorbar numbers font
cbar.ax.tick_params(labelsize=fs3, colors='black')

plt.tight_layout()

# 8. Save chart
if not os.path.exists(config.RESULTS_FOLDER):
    os.makedirs(config.RESULTS_FOLDER)
    
file_name_img = f'{config.RESULTS_FOLDER}/q4_8_2_geopolitical_heatmap.pdf'
plt.savefig(file_name_img, dpi=300, bbox_inches='tight')
plt.close()

print(f"Geopolitical Heatmap generated and saved to: {file_name_img}")