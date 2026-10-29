import utils
import config
import pandas as pd
import matplotlib.pyplot as plt
import os

# ==========================================
# 4. GLOBAL FONT CONFIGURATION (Ideal for Slides)
# ==========================================
fs1 = 28  # Main Title Size (larger for presentations)
fs2 = 24  # Axis Label Size
fs3 = 24  # Text Size (X-Axis)
fs_bar = 24 # Label Size at the top of the bars

# 1. Load data
df_data = utils.get_df_data()

# 2. Define the "Valid Document Universe" (Articles mentioning at least 1 metal)
mask_any_metal = (df_data['is_li_ion'] == True) | \
                    (df_data['is_na_ion'] == True) | \
                    (df_data['is_k_ion'] == True) | \
                    (df_data['is_zn_ion'] == True) | \
                    (df_data['is_other_metal'] == True)
                    
df_valido = df_data[mask_any_metal].copy()

# The absolute denominator for percentages
total_docs_validos = len(df_valido)

# 3. Calculate totals for each type (allowing overlaps)
total_li = df_valido['is_li_ion'].sum()
total_na = df_valido['is_na_ion'].sum()
total_k = df_valido['is_k_ion'].sum()
total_zn = df_valido['is_zn_ion'].sum()
total_others = df_valido['is_other_metal'].sum()

# Create a simple DataFrame
dados = {
    'Ion Type': ['Lithium-ion (Li)', 'Sodium-ion (Na)', 'Potassium-ion (K)', 'Zinc-ion (Zn)', 'Other Metals'],
    'Total Publications': [total_li, total_na, total_k, total_zn, total_others]
}
df_plot = pd.DataFrame(dados)

# Calculate the Percentage (Presence Rate)
df_plot['Percentage'] = (df_plot['Total Publications'] / total_docs_validos) * 100

# Sort from largest to smallest
df_plot = df_plot.sort_values(by='Total Publications', ascending=False)



# ==============================================================================
# 5. BAR CHART GENERATION
# ==============================================================================
fig, ax = plt.subplots(figsize=(12, 8)) # Slightly wider figure to fit 5 bars

# Define colors (adding gray for 'Other Metals')
cores_map = {
    'Lithium-ion (Li)': '#4C72B0',    # Blue
    'Sodium-ion (Na)': '#E67E22',     # Orange
    'Potassium-ion (K)': '#9B59B6',   # Purple
    'Zinc-ion (Zn)': '#2ECC71',       # Green
    'Other Metals': '#95A5A6'         # Elegant gray
}
cores = [cores_map[ion] for ion in df_plot['Ion Type']]

# Plot the bars (making them slightly wider 'width=0.65' for slide visuals)
bars = ax.bar(df_plot['Ion Type'], df_plot['Total Publications'], 
                color=cores, edgecolor='black', linewidth=1.5, width=0.65, zorder=3)

# Add numbers and percentages on top of each bar
for bar, pct in zip(bars, df_plot['Percentage']):
    yval = bar.get_height()
    
    # Formatting text with line break: "Number \n (Percentage%)"
    texto_rotulo = f"{int(yval)}\n({pct:.1f}%)"
    
    # The offset ensures the text floats slightly above the bar
    ax.text(bar.get_x() + bar.get_width()/2, yval + (df_plot['Total Publications'].max() * 0.02), 
            texto_rotulo, ha='center', va='bottom', fontsize=fs_bar, color='black')

# Titles and Axes
ax.set_title('Overall Publication Presence by Metal-Ion System', fontsize=fs1, pad=25, color='black')
ax.set_ylabel('Absolute Volume & Presence Rate (%)', fontsize=fs2, labelpad=15, color='black')

# Adjust Y-axis limits to leave space for two-line text on top of the tallest bar
max_y = df_plot['Total Publications'].max()
ax.set_ylim(0, max_y * 1.15)

# Visual adjustments (Grid and Ticks)
ax.tick_params(axis='both', colors='black', labelsize=fs3)
# Slightly rotate X-axis if names start crowding
ax.set_xticklabels(df_plot['Ion Type'], rotation=15, ha='center')

ax.grid(axis='y', linestyle=':', alpha=0.6, color='black', zorder=0)

# Solid black borders
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.5)

# Remove top and right borders for a cleaner slide
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

# 6. Save the chart
if not os.path.exists(config.RESULTS_FOLDER):
    os.makedirs(config.RESULTS_FOLDER)
        

    
file_name_img = f'{config.RESULTS_FOLDER}/grafico_extra_distribuicao_ions.pdf'
plt.savefig(file_name_img, dpi=300, bbox_inches='tight')
plt.close()

print(f"Distribution Bar Chart generated and saved to: {file_name_img}")
