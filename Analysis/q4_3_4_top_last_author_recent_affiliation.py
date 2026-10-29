import utils
import config

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df_data = utils.get_df_data()


# --- Part 1: Calculate Overall Statistics per Author ---
print("Calculating overall statistics per author...")
# Groupby is now done only by 'last_author'
author_stats = df_data.groupby(config.COLUMN_LAST_AUTHOR_FULL_NAME).agg(
    publications=(config.COLUMN_CITATIONS_COUNT, 'count'),
    total_citations=(config.COLUMN_CITATIONS_COUNT, 'sum'),
    mean_citations=(config.COLUMN_CITATIONS_COUNT, 'mean')
).reset_index()


# --- Part 2: Determine the Most Recent Affiliation of Each Author ---
print("Determining the most recent affiliation of each author...")
# Sort data by author and, in descending order, by year
df_sorted_by_year = df_data.sort_values([config.COLUMN_LAST_AUTHOR_FULL_NAME, config.COLUMN_YEAR], ascending=[True, False])
# For each author, take the first entry, which corresponds to the most recent year
latest_affiliations = df_sorted_by_year.drop_duplicates(subset=config.COLUMN_LAST_AUTHOR_FULL_NAME, keep='first')[[config.COLUMN_LAST_AUTHOR_FULL_NAME, config.COLUMN_LAST_INSTITUTION]]


# --- Part 3: Combine Statistics and Recent Affiliation ---
print("Combining overall statistics with most recent affiliation...")
# Merge the two DataFrames on the 'last_author' column
df_merged = pd.merge(author_stats, latest_affiliations, on=config.COLUMN_LAST_AUTHOR_FULL_NAME)


# --- Part 4: Sort, Select Top 20, and Format ---
# Sort by total publication count of the author
df_final = df_merged.sort_values('publications', ascending=False)
df_top20 = df_final.head(20).copy()

df_top20[config.COLUMN_LAST_AUTHOR_FULL_NAME] = df_top20[config.COLUMN_LAST_AUTHOR_FULL_NAME].str.title()  # Name formatting

# Renaming columns
df_top20.rename(columns={
    config.COLUMN_LAST_AUTHOR_FULL_NAME: 'Author',
    config.COLUMN_LAST_INSTITUTION: 'Last Affiliation',
    'publications': '# Pubs',
    'total_citations': 'Tot Cites',
    'mean_citations': 'Mean Cites'
}, inplace=True)

# Data type conversion
df_top20['# Pubs'] = df_top20['# Pubs'].astype(int)
df_top20['Tot Cites'] = df_top20['Tot Cites'].astype(int)
df_top20['Mean Cites'] = df_top20['Mean Cites'].round(1)

# REORDERING COLUMNS so that 'Last Affiliation' is LAST
df_top20 = df_top20[['Author', '# Pubs', 'Tot Cites', 'Mean Cites', 'Last Affiliation']]


# --- Part 5: Generate LaTeX Table ---
print("Generating LaTeX code for the table...")

# Format for 5 columns: Author (text), 3x Stats (numbers), Affiliation (text)
column_format = "p{4cm} r r r p{5cm}"

# Adjusting header order to match the new column order
# Final order: Author, # Pubs, Tot Cites, Mean Cites, Last Affiliation
custom_header = [
    "Author",
    "\\shortstack[r]{{\\#\\\\Pubs}}",
    '\\shortstack[r]{{Tot\\\\Cites}}',
    '\\shortstack[r]{{Mean\\\\Cites}}',
    'Last Affiliation'
]

latex_table = df_top20.to_latex(
    buf=None,
    caption="Top 20 Last Authors by Total Publications and Their Most Recent Affiliation",
    label="tab:q4_3_4_top_last_author_recent_affiliation",
    index=False,
    column_format=column_format,
    escape=False,  # IMPORTANT: 'False' so LaTeX commands in the header work
    float_format="%.1f",
    header=custom_header
)

print("\n--- TABLE LATEX CODE ---")
print(latex_table)

file_name_latex_combined = f'{config.RESULTS_FOLDER}/q4_3_4_top_last_author_recent_affiliation.tex'
with open(file_name_latex_combined, 'w', encoding='utf-8') as f:
    f.write(latex_table)



print("\nScript finished.")
