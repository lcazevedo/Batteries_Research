import utils
import config
import pandas as pd
import matplotlib.pyplot as plt
import os
from adjustText import adjust_text  # Import text adjustment library

# ==========================================
# GLOBAL FONT CONFIGURATION
# ==========================================
fs1 = 26  # Main Title font size
fs2 = 22  # Axis Labels font size (X and Y)
fs3 = 22  # Text font size (Country names, axis numbers)
fs_quad = 24 # Background Quadrant labels font size

# 1. Load data
df_data = utils.get_df_data()

# 2. Filter valid documents (Any listed metal)
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

# 4. Create Strategic Masks
# Exclusive Lithium (True for Li, False for EVERYTHING else)
df_valido['is_li_only'] = (df_valido['is_li_ion'] == True) & \
                            (df_valido['is_na_ion'] == False) & \
                            (df_valido['is_k_ion'] == False) & \
                            (df_valido['is_zn_ion'] == False) & \
                            (df_valido['is_other_metal'] == False)

# Next Generation (Na, K, or Zn) - May or may not include mixed Lithium
df_valido['is_next_gen'] = df_valido['is_na_ion'] | df_valido['is_k_ion'] | df_valido['is_zn_ion']

# 5. Group by Country
df_paises = df_valido.groupby(col_pais).agg(
    Total_Docs=(col_pais, 'count'),
    Li_Only_Docs=('is_li_only', 'sum'),
    NextGen_Docs=('is_next_gen', 'sum')
)

# 6. Select Top 15 (or Top 20) largest absolute producers for clean analysis
df_top15 = df_paises.nlargest(15, 'Total_Docs').copy()

# 7. Calculate Percentages (X and Y Axes)
df_top15['X_Li_Only_Pct'] = (df_top15['Li_Only_Docs'] / df_top15['Total_Docs']) * 100
df_top15['Y_NextGen_Pct'] = (df_top15['NextGen_Docs'] / df_top15['Total_Docs']) * 100

# ==============================================================================
# 8. SCATTER PLOT GENERATION
# ==============================================================================
fig, ax = plt.subplots(figsize=(14, 9)) # Slightly increased figure size to accommodate size 22 fonts

# Define quadrant dividing lines (using Top 15 mean)
x_mid = df_top15['X_Li_Only_Pct'].mean()
y_mid = df_top15['Y_NextGen_Pct'].mean()

# Plot quadrant lines (Black dashed)
ax.axvline(x=x_mid, color='black', linestyle='--', alpha=0.6, linewidth=1.5)
ax.axhline(y=y_mid, color='black', linestyle='--', alpha=0.6, linewidth=1.5)

# Add points (Blue circles with black border)
ax.scatter(df_top15['X_Li_Only_Pct'], df_top15['Y_NextGen_Pct'], 
            color='#4C72B0', edgecolor='black', s=150, alpha=0.85, zorder=3)

# ---------------------------------------------------------
# Expand chart limits BEFORE adjust_text
# ---------------------------------------------------------
x_min, x_max = df_top15['X_Li_Only_Pct'].min(), df_top15['X_Li_Only_Pct'].max()
y_min, y_max = df_top15['Y_NextGen_Pct'].min(), df_top15['Y_NextGen_Pct'].max()

# Add extra space on horizontal and vertical edges
x_factor = 0.25
y_factor = 0.25
ax.set_xlim(x_min - (x_max - x_min)*x_factor, x_max + (x_max - x_min)*x_factor)
ax.set_ylim(y_min - (y_max - y_min)*(y_factor+.1), y_max + (y_max - y_min)*y_factor)
# ---------------------------------------------------------

# ---------------------------------------------------------
# APPLY ADJUST_TEXT
# ---------------------------------------------------------
texts = []
x_vals = df_top15['X_Li_Only_Pct'].values
y_vals = df_top15['Y_NextGen_Pct'].values

for pais, row in df_top15.iterrows():
    t = ax.text(row['X_Li_Only_Pct'], row['Y_NextGen_Pct'], pais, fontsize=fs3, color='black', zorder=4)
    texts.append(t)

# Optimized parameters for large fonts in tight corners
adjust_text(texts, 
            x=x_vals, 
            y=y_vals, 
            expand_points=(2.0, 2.0), # Keep space around the point
            expand_text=(1.5, 1.5),   # Require more whitespace around text itself
            force_points=1.5,         # Push text away from point
            force_text=2.5,           # Push text away from other text
            lim=500,                  # Increase maximum solver iterations
            arrowprops=dict(arrowstyle='-', color='gray', lw=1.5, alpha=0.7))
# ---------------------------------------------------------

# ---------------------------------------------------------
# BACKGROUND ANNOTATIONS (NAMING THE QUADRANTS)
# ---------------------------------------------------------
# Calculate limits to position texts in corners
x_min, x_max = ax.get_xlim()
y_min, y_max = ax.get_ylim()

offset_x = (x_max - x_min) * 0.02
offset_y = (y_max - y_min) * 0.02

# Upper Left Quadrant (High Next-Gen, Low Li-Only)
ax.text(x_min + offset_x, y_max - offset_y + 0.3, 'PIONEERS\n(High Alt, Low Li-Only)', 
        fontsize=fs_quad, color='gray', alpha=0.5, va='top', ha='left', weight='bold')

# Lower Right Quadrant (Low Next-Gen, High Li-Only)
ax.text(x_max - offset_x, y_min + offset_y - 0.3, 'TRADITIONALISTS\n(High Li-Only, Low Alt)', 
        fontsize=fs_quad, color='gray', alpha=0.5, va='bottom', ha='right', weight='bold')
# ---------------------------------------------------------

# 9. Titles, Axes, and STRICT-BLACK Visual Adjustments
ax.set_title('Strategic Transition Matrix: Next-Generation vs. Established Li-ion Focus', fontsize=fs1, pad=20, color='black')
ax.set_xlabel('Internal Focus on Exclusive Lithium-ion Research (%)', fontsize=fs2, labelpad=10, color='black')
ax.set_ylabel('Internal Focus on Next-Generation Batteries (%)', fontsize=fs2, labelpad=10, color='black')

ax.grid(True, linestyle=':', alpha=0.5, color='black', zorder=0)
ax.tick_params(axis='both', colors='black', labelsize=fs3)

# Solid black borders
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.0)

plt.tight_layout()

# 10. Save Chart
if not os.path.exists(config.RESULTS_FOLDER):
    os.makedirs(config.RESULTS_FOLDER)
    
file_name_img = f'{config.RESULTS_FOLDER}/q4_11_1_strategic_transition_matrix.pdf'
plt.savefig(file_name_img, dpi=300, bbox_inches='tight')
plt.close()

print(f"Strategic Transition Matrix generated and saved to: {file_name_img}")


# ==============================================================================
# 11. STRATEGIC MATRIX LATEX TABLE GENERATION (GROUPED)
# ==============================================================================
# Group classification (Pioneers vs Traditionalists) based on Y-axis mean
# Variable y_mid was already calculated in step 8
df_pioneers = df_top15[df_top15['Y_NextGen_Pct'] >= y_mid].sort_values('Y_NextGen_Pct', ascending=False)
df_tradition = df_top15[df_top15['Y_NextGen_Pct'] < y_mid].sort_values('Y_NextGen_Pct', ascending=False)

latex_str = r"""\begin{table}[htpb]
\centering
\begin{threeparttable}
\caption{Strategic classification of national research portfolios: Pioneers vs. Traditionalists in the beyond-lithium transition.}
\label{tab:q4_11_1_strategic_transition_matrix}
% Extra column for Group (l), Country (3.0cm), and remaining values
\begin{tabular}{ l p{3.0cm} >{\raggedleft\arraybackslash}p{1.6cm} >{\raggedleft\arraybackslash}p{1.6cm} >{\raggedleft\arraybackslash}p{1.6cm} >{\raggedleft\arraybackslash}p{1.6cm} >{\raggedleft\arraybackslash}p{1.8cm} }
\hline
Group & Country & Total & Li-Only & Next-Gen & Li-Only & Next-Gen \\
 & & Pubs & Pubs & Pubs & Focus (\%) & Focus (\%) \\
\hline
"""

# --- PIONEERS Group ---
num_pion = len(df_pioneers)
first = True
for pais, row in df_pioneers.iterrows():
    cat_cell = rf"\multirow{{{num_pion}}}{{*}}{{\rotatebox{{90}}{{Pioneers}}}}" if first else ""
    pais_safe = str(pais).replace('&', r'\&').replace('_', ' ')
    
    total = int(row['Total_Docs'])
    li_only = int(row['Li_Only_Docs'])
    next_gen = int(row['NextGen_Docs'])
    li_pct = row['X_Li_Only_Pct']
    next_pct = row['Y_NextGen_Pct']
    
    latex_str += f"{cat_cell} & {pais_safe} & {total} & {li_only} & {next_gen} & {li_pct:.1f} & {next_pct:.1f} \\\\\n"
    first = False

latex_str += r"\hline " + "\n"

# --- TRADITIONALISTS Group ---
num_trad = len(df_tradition)
first = True
for pais, row in df_tradition.iterrows():
    cat_cell = rf"\multirow{{{num_trad}}}{{*}}{{\rotatebox{{90}}{{Traditionalists}}}}" if first else ""
    pais_safe = str(pais).replace('&', r'\&').replace('_', ' ')
    
    total = int(row['Total_Docs'])
    li_only = int(row['Li_Only_Docs'])
    next_gen = int(row['NextGen_Docs'])
    li_pct = row['X_Li_Only_Pct']
    next_pct = row['Y_NextGen_Pct']
    
    latex_str += f"{cat_cell} & {pais_safe} & {total} & {li_only} & {next_gen} & {li_pct:.1f} & {next_pct:.1f} \\\\\n"
    first = False

# Table closing and footer with text adjusted for the classification
latex_str += r"""\hline
\end{tabular}
\begin{tablenotes}
\small
\item Note: The table categorizes countries into two strategic groups based on the mean focus on next-generation technologies. The Pioneers group represents nations with above-average relative commitment to sodium, potassium, or zinc-ion research. The Traditionalists group includes established research hubs maintaining a higher concentration on exclusive lithium-ion systems.
\end{tablenotes}
\end{threeparttable}
\end{table}"""

# Save LaTeX file keeping the original filename
file_name_tex = f'{config.RESULTS_FOLDER}/q4_11_1_strategic_transition_matrix.tex'
with open(file_name_tex, 'w', encoding='utf-8') as f:
    f.write(latex_str)
    
print(f"Strategic Matrix LaTeX Table (Grouped) generated and saved to: {file_name_tex}")
