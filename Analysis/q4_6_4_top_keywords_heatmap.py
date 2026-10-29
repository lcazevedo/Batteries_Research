import utils
import config

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Load data
df_data = utils.get_df_data()

# Font sizes
fs1 = 30
fs2 = 23
fs3 = 23

# --- 1. Data Preparation (The Missing Step) ---

print("Starting data preparation...")

# Define column names based on 'config'
year_col = config.COLUMN_YEAR
keyword_list_col = config.COLUMN_AUTHORS_KEYWORDS_NORM_LIST
keyword_col = 'keyword' # Name of the new column after 'explode'
N_TOP_KEYWORDS = 15     # Number of keywords to show
N_YEARS_TO_SHOW = 11    # Number of years to show (last 11)

# Select only necessary columns
df_prep = df_data[[year_col, keyword_list_col]]
df_prep = df_prep[(df_prep[year_col] >= 2014) & (df_prep[year_col] <= 2025)]

# ** KEY STEP: Explode keyword list **
# Transforms each item in the list into a separate row
df_exploded = df_prep.explode(keyword_list_col)
df_exploded = df_exploded.rename(columns={keyword_list_col: keyword_col})

# Clean null values that may have been generated
df_exploded = df_exploded.dropna(subset=[keyword_col])

# ** KEY STEP: Create 'kw_grouped' **
# Count the frequency of each keyword per year
kw_grouped = df_exploded.groupby([year_col, keyword_col]).size().reset_index(name='count')

print("Data grouped and counted.")

# --- 2. Filtering and Pivot Logic (Adapted Code) ---

# 1. Identify overall Top N keywords
total_counts = kw_grouped.groupby(keyword_col)['count'].sum()
# Use min(N_TOP_KEYWORDS, total_counts.nunique()) to avoid errors if there are fewer keywords
top_n = min(N_TOP_KEYWORDS, len(total_counts)) 
top_keywords = total_counts.nlargest(top_n).index

# 2. Filter only those N words
df_top = kw_grouped[kw_grouped[keyword_col].isin(top_keywords)]

# 3. Filter the last N years
max_year = df_top[year_col].max()
min_year = max_year - (N_YEARS_TO_SHOW - 1)
df_top = df_top[df_top[year_col] >= min_year]

# 4. Pivot DataFrame
# (index=keyword, columns=year, values=count)
pivot = df_top.pivot_table(index=keyword_col, columns=year_col, values='count', fill_value=0)

# 5. Calculate total for sorting and normalization
pivot['total'] = pivot.sum(axis=1)

# 6. Sort by total (most frequent at the top)
pivot = pivot.sort_values(by='total', ascending=False)

# 7. Normalize to 0-100% (dividing by total)
# .iloc[:, :-1] selects all year columns except 'total'
pivot.iloc[:, :-1] = pivot.iloc[:, :-1].div(pivot['total'], axis=0) * 100

# 8. Remove 'total' column after calculation
pivot = pivot.drop(columns=['total'])

print("Pivoted and normalized matrix created.")

# --- 3. Plot Heatmap ---

plt.figure(figsize=(15, 10))
# 'pivot' now contains percentages
ax = sns.heatmap(pivot, cmap='YlGnBu', annot=False, cbar=True) # Set cbar=True to display the colorbar


cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=fs3)

# Get colormap
cmap = plt.cm.get_cmap('YlGnBu')

# Function to determine text color based on background color brightness
def get_text_color(rgb_color):
    # Calculate color luminance (brightness) (human perception)
    # Formula L = 0.299*R + 0.587*G + 0.114*B (for colors in 0-1)
    luminance = (0.299 * rgb_color[0] + 0.587 * rgb_color[1] + 0.114 * rgb_color[2])
    return "white" if luminance < 0.5 else "black" # Threshold of 0.5 is common, can be adjusted

# Manually annotate with percentages (converted to int)
for i in range(pivot.shape[0]):           # Rows (terms)
    for j in range(pivot.shape[1]):       # Columns (years)
        value = int(pivot.iloc[i, j])
        
        # Map cell value to a color in the colormap
        # Normalize value to the [0, 1] interval of the colormap
        # Maximum value in pivot is 100 (percentage), and minimum is 0
        norm_value = value / 100.0 
        rgb_color = cmap(norm_value) # Get RGB color (actually RGBA)
        
        # Determine text color based on cell color brightness
        if pivot.columns[j] == max_year:
            text_color = "gray"  # Gray color for the most recent year
        else:
            text_color = get_text_color(rgb_color)
        
        # Optional: Highlight the last year (max_year) with a different color if desired, 
        # though this may conflict with contrast logic. 
        # Keeping dynamic text color for all.
        # current_col_year = pivot.columns[j]
        # text_color = "gray" if current_col_year == max_year else text_color

        ax.text(j + 0.5, i + 0.5, f"{value}", color=text_color, 
                ha='center', va='center', fontsize=fs3)

# Titles and labels
plt.title(f"Frequency (%) of Top {top_n} Authors Keywords by Year", fontsize=fs1, loc='right', pad=30)
plt.xlabel("Year", fontsize=fs2)
plt.ylabel("Keyword", fontsize=fs2)
plt.xticks(rotation=45, fontsize=fs2)
plt.yticks(fontsize=fs2)

plt.tight_layout()

# --- 4. Save and Show Image ---

output_filename = f"{config.RESULTS_FOLDER}/q4_6_4_top_keywords_heatmap.pdf"

# ** FIX: Use plt.savefig() to save the matplotlib chart **
try:
    plt.savefig(output_filename, format='pdf', bbox_inches='tight')
    print(f"Heatmap successfully saved as '{output_filename}'")

except Exception as e:
    print(f"Error saving chart: {e}")