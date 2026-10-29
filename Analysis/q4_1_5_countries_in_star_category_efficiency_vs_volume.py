import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text

import config
import utils

# --- Load the dataset ---
df_data = utils.get_df_data()


# --- 1. Visual Settings and Parameters ---
start_year = 1993
end_year = 2024
top_n_countries = 20  # Limit on the number of countries in the chart

# Font size configuration
fs_title = 34          # Chart title
fs_axis_title = 28     # Axis titles (X and Y)
fs_axis_label = 28     # Axis labels (Numbers/Ticks)
fs_point_label = 26    # Point labels (Country names)
fs_annotation = 26     # Explanatory texts (Regions, Notes)

# --- 2. Data Processing ---

# Filter by period
df_period = df_data[(df_data[config.COLUMN_YEAR] >= start_year) & (df_data[config.COLUMN_YEAR] <= end_year)].copy()

# REMOVE "UNKNOWN"
# Important to do this before calculating totals so it does not take up space in the ranking
df_period = df_period[df_period[config.COLUMN_FIRST_COUNTRY] != 'UNKNOWN']

# Step 2.1: Calculate total publications per country (Volume)
country_totals = (
    df_period
    .groupby(config.COLUMN_FIRST_COUNTRY)
    .size()
    .reset_index(name='total_period')
)

# Step 2.2: Filter only the Top N countries by volume
top_countries_list = (
    country_totals
    .sort_values('total_period', ascending=False)
    .head(top_n_countries)[config.COLUMN_FIRST_COUNTRY]
    .tolist()
)

# Step 2.3: Calculate the number of "Stars" per country
# Filter only the rows that are "Star"
stars_counts = (
    df_period[df_period[config.COLUMN_CATEGORY_GROUP] == "Star (High IF, High Cit)"]
    .groupby(config.COLUMN_FIRST_COUNTRY)
    .size()
    .reset_index(name='star_count')
)

# Step 2.4: Join the tables (Total + Stars)
# Use the Top N to filter the final dataframe
df_plot = country_totals[country_totals[config.COLUMN_FIRST_COUNTRY].isin(top_countries_list)].merge(
    stars_counts, 
    on=config.COLUMN_FIRST_COUNTRY, 
    how='left'
)

# Fill with 0 in case a country has no "Star" paper
df_plot['star_count'] = df_plot['star_count'].fillna(0)

# Step 2.5: Calculate the percentage of Stars (Quality)
df_plot['star_percentage'] = (df_plot['star_count'] / df_plot['total_period']) * 100


# --- 2.6: Save chart data to CSV ---
output_csv = f"{config.RESULTS_FOLDER}/q4_1_5_countries_star_category.csv"

df_plot.sort_values("total_period", ascending=False).to_csv(
    output_csv,
    index=False
)

# --- 3. Chart Generation ---

plt.figure(figsize=(14, 10))

# Plot the points
plt.scatter(
    df_plot['total_period'], 
    df_plot['star_percentage'], 
    color='#1f77b4', 
    s=150, 
    alpha=0.7, 
    edgecolors='k'
)

# Set logarithmic scale for the X-axis (Volume)
plt.xscale('log')

# --- 4. Annotations and Labels ---

# List to store text objects
texts = []

# Add each country name to the list (without fixed plotting yet)
for i, row in df_plot.iterrows():
    # Note that manual offset (+ 0.8) was removed as the library will handle this
    t = plt.text(
        row['total_period'], 
        row['star_percentage'], 
        row[config.COLUMN_FIRST_COUNTRY], 
        fontsize=fs_point_label,      
        ha='center', 
        va='bottom', 
        #fontweight='bold'
    )
    texts.append(t)

# Axis Titles
plt.xlabel('Total Publication Volume (Log Scale)', fontsize=fs_axis_title)
plt.ylabel('% of Papers in "Star" Category', fontsize=fs_axis_title)

# Chart Title
plt.title(f'Efficiency vs. Volume: Top {top_n_countries} Countries ({start_year}-{end_year})', fontsize=fs_title)

# Adjust Axis Labels (Ticks)
plt.tick_params(axis='both', which='major', labelsize=fs_axis_label)

# Grid
plt.grid(True, which="both", ls="--", alpha=0.5)

# --- 5. Marking Strategic Regions (Annotations) ---

# Get current limits to position relative texts
x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()

# Top Right Corner
plt.text(x_max * 0.9, y_max * 0.92, 'Consistency Leaders\n(High Volume, High Quality)', 
         fontsize=fs_annotation, color='green', ha='right') #, weight='bold')

# Top Left Corner
plt.text(x_min * 1.1, y_max * 0.85, 'Niche Elite\n(High Efficiency)', 
         fontsize=fs_annotation, color='purple', ha='left') #, weight='bold')

# Bottom/Middle Right Corner
plt.text(x_max * 0.9, y_min + (y_max-y_min)*0.12, 'Scale Giants\n(Massive Volume)', 
         fontsize=fs_annotation, color='red', ha='right') #, weight='bold')

# Footnote
plt.text(0.3, 0.02, 
         "Note: Countries at the bottom may perform highly in\n'Hidden Gems', not captured in this 'Stars' chart.", 
         transform=plt.gca().transAxes, fontsize=fs_annotation - 4, 
         bbox=dict(facecolor='yellow', alpha=0.2))

# --- 6. AUTOMATIC TEXT ADJUSTMENT CALL ---
# arrowprops draws a line if the text moves too far from the point
adjust_text(
    texts, 
    arrowprops=dict(
        arrowstyle='->',
        color='black',
        lw=0.5,
        shrinkA=20,   # shrink near text
        #shrinkB=5    # shrink near point
    ),
    expand_points=(1.2, 1.2)
)


plt.tight_layout()

# Save or display
plt.savefig(config.RESULTS_FOLDER+"/q4_1_5_countries_in_star_category_efficiency_vs_volume.pdf", dpi=300, bbox_inches='tight')
