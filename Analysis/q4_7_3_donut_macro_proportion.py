import utils
import config
import pandas as pd
import matplotlib.pyplot as plt
import os

fs1 = 22
fs2 = 18

# 1. Load data
df_data = utils.get_df_data()

# 2. Filter valid documents (remove irrelevant ones)
mask_any_metal = (df_data['is_li_ion'] == True) | \
                 (df_data['is_na_ion'] == True) | \
                 (df_data['is_k_ion'] == True) | \
                 (df_data['is_zn_ion'] == True) | \
                 (df_data['is_other_metal'] == True)

df_valido = df_data[mask_any_metal].copy()

# 3. Create exclusive MACRO masks
col_alt = ['is_na_ion', 'is_k_ion', 'is_zn_ion', 'is_other_metal']
has_any_alt = df_valido[col_alt].any(axis=1)

mask_li_only = df_valido['is_li_ion'] & ~has_any_alt
mask_pure_alt = ~df_valido['is_li_ion'] & has_any_alt
mask_mixed = df_valido['is_li_ion'] & has_any_alt

# Absolute counts
count_li_only = mask_li_only.sum()
count_pure_alt = mask_pure_alt.sum()
count_mixed = mask_mixed.sum()

# 4. Prepare data for the chart
labels = ['Lithium-Only', 'Pure Alternative', 'Mixed (Comparative)']
sizes = [count_li_only, count_pure_alt, count_mixed]

# Elegant color palette for scientific articles (Blue, Green, Orange/Red)
colors = ['#4C72B0', '#55A868', '#DD8452']

# 5. Configure and generate the chart
fig, ax = plt.subplots(figsize=(8, 6))

# Generate pie chart with labels and percentages
wedges, texts, autotexts = ax.pie(
    sizes, 
    labels=labels, 
    colors=colors, 
    autopct='%1.1f%%', 
    startangle=90, 
    pctdistance=0.80, # Distance of percentage text from the center
    textprops={'fontsize': fs2, 'color': 'black'}
)

# Style percentage font
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_weight('bold')

# Add white circle in the center to create the donut appearance
centre_circle = plt.Circle((0,0), 0.60, fc='white')
ax.add_artist(centre_circle)

# Adjust aspect ratio to be a perfect circle
ax.axis('equal')  

# Chart title
plt.title('Macro Proportion of Metal-Ion Battery Publications', fontsize=fs1, pad=20, weight='bold')
plt.tight_layout()

# 6. Save high-resolution chart
if not os.path.exists(config.RESULTS_FOLDER):
    os.makedirs(config.RESULTS_FOLDER)
    
file_name_img = f'{config.RESULTS_FOLDER}/q4_7_3_donut_macro_proportion.pdf'
plt.savefig(file_name_img, dpi=300, bbox_inches='tight')
plt.close() # Important to free memory

print(f"Donut chart generated and saved to: {file_name_img}")
