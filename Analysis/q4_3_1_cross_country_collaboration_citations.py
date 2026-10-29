import utils
import config

import pandas as pd
import numpy as np
import textwrap
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import seaborn as sns

from adjustText import adjust_text   # pip install adjustText


# Load data
df_data = utils.get_df_data()

# Fonts
fs1 = 44  # title
fs2 = 34  # axis titles
fs3 = 32  # data labels on axes

# Remove duplicates in country lists per document
df_data['unique_countries'] = df_data[config.COLUMN_AUTHORS_COUNTRY].apply(lambda x: list(set(x)))
df_data['n_countries'] = df_data['unique_countries'].apply(len)

# Calculate citations per year
current_year = pd.Timestamp.now().year
df_data['citations_per_year'] = df_data[config.COLUMN_CITATIONS_COUNT] / (current_year - df_data[config.COLUMN_YEAR] + 1)

# Create chart
plt.figure(figsize=(12, 8))

# Boxplot without visual outliers
sns.boxplot(
    x='n_countries',
    y='citations_per_year',
    data=df_data,
    showfliers=False,     # Hide boxplot outliers (we will display them with stripplot)
    width=0.6,
    palette='pastel'
)

# Stripplot with transparency to show point density
sns.stripplot(
    x='n_countries',
    y='citations_per_year',
    data=df_data,
    color='black',
    alpha=0.3,
    jitter=True,         # Spread points horizontally
    size=4
)

# Log scale on Y-axis
plt.yscale('log')

# Customization
plt.title('Mean Citations/Year at\nDifferent Levels of Collaboration', fontsize=fs1)
plt.xlabel('Cross-Country Collaboration', fontsize=fs2)
plt.ylabel('Mean citations per year\n(log scale)', fontsize=fs2)
plt.xticks(fontsize=fs3)
plt.yticks(fontsize=fs3)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.gca().set_facecolor('white')  # White background

plt.tight_layout()


plt.savefig(f"{config.RESULTS_FOLDER}/q4_3_1_cross_country_collaboration_citations.pdf") 

# Filter only positive values
df_plot = df_data[df_data['citations_per_year'] > 0]

# Calculate statistics grouped by number of countries
summary_table = df_plot.groupby('n_countries')['citations_per_year'].agg([
    ('# Documents', 'count'),
    ('Min', 'min'),
    ('1st Quartile', lambda x: x.quantile(0.25)),
    ('Median', 'median'),
    ('3rd Quartile', lambda x: x.quantile(0.75)),
    ('Max', 'max'),
    ('Mean', 'mean'),
]).reset_index()

# Rename number of countries column
summary_table.rename(columns={'n_countries': 'Countries'}, inplace=True)

# Save as LaTeX
latex_output = summary_table.to_latex(index=False, float_format="%.2f")

# Save to .tex file
with open(f"{config.RESULTS_FOLDER}/q4_3_1_cross_country_collaboration_citations.tex", "w") as f:
    f.write(latex_output)
