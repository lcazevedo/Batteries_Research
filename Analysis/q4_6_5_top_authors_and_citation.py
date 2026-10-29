import utils
import config

import pandas as pd
import numpy as np
import textwrap

# Load data
df_data = utils.get_df_data()

# Define which author type to analyze
author_type = "first"   # last or first

# Table of the top 20 authors with their most recent institution
author_column = config.COLUMN_LAST_AUTHOR_FULL_NAME if author_type == "last" else config.COLUMN_FIRST_AUTHOR_FULL_NAME
institution_column = config.COLUMN_LAST_INSTITUTION if author_type == "last" else config.COLUMN_FIRST_INSTITUTION

# --- Part 1: Calculate General Statistics per Author ---
print("Calculating general statistics per author...")
# Groupby is performed by author column
author_stats = df_data.groupby(author_column).agg(
    publications=(config.COLUMN_CITATIONS_COUNT, 'count'),
    total_citations=(config.COLUMN_CITATIONS_COUNT, 'sum'),
    mean_citations=(config.COLUMN_CITATIONS_COUNT, 'mean')
).reset_index()


# --- Part 2: Determine Most Recent Affiliation for Each Author ---
print("Determining most recent affiliation for each author...")
# Sort data by author ascending and year descending
df_sorted_by_year = df_data.sort_values([author_column, config.COLUMN_YEAR], ascending=[True, False])
# For each author, take the first entry, which corresponds to the most recent year
latest_affiliations = df_sorted_by_year.drop_duplicates(subset=author_column, keep='first')[[author_column, institution_column]]


# --- Part 3: Combine Statistics and Recent Affiliation ---
print("Combining general statistics with most recent affiliation...")
# Merge the two DataFrames on author column
df_merged = pd.merge(author_stats, latest_affiliations, on=author_column)


# --- Part 4: Sort, Select Top 20, and Format ---
# Sort by total publication count per author
df_final = df_merged.sort_values('publications', ascending=False)
df_top20 = df_final.head(20).copy()

df_top20[author_column] = df_top20[author_column].str.title()  # Name formatting

# Rename columns
df_top20.rename(columns={
    author_column: 'Author',
    institution_column: 'Last Affiliation',
    'publications': '# Pubs',
    'total_citations': 'Tot Cites',
    'mean_citations': 'Mean Cites'
}, inplace=True)

# Data type conversion
df_top20['# Pubs'] = df_top20['# Pubs'].astype(int)
df_top20['Tot Cites'] = df_top20['Tot Cites'].astype(int)
df_top20['Mean Cites'] = df_top20['Mean Cites'].round(1)

# REORDER COLUMNS so that 'Last Affiliation' is LAST
df_top20 = df_top20[['Author', '# Pubs', 'Tot Cites', 'Mean Cites', 'Last Affiliation']]


# --- Part 5: Generate LaTeX Table ---
print("Generating LaTeX code for the table...")

# Format for 5 columns: Author (text), 3x Stats (numbers), Affiliation (text)
column_format = "p{4cm} r r r p{5cm}"

# Adjust header order to match the new column order
# Final order: Author, # Pubs, Tot Cites, Mean Cites, Last Affiliation
custom_header = [
    "Author",
    "\\shortstack[r]{{\\#\\\\Pubs}}",
    '\\shortstack[r]{{Tot\\\\Cites}}',
    '\\shortstack[r]{{Mean\\\\Cites}}',
    'Last Affiliation'
]

author_ref = "Last Author" if author_type == "last" else "First Author"
latex_table = df_top20.to_latex(
    f"{config.RESULTS_FOLDER}/q4_6_5_top_{author_type}_authors_and_citations.tex",
    caption=f"Top 20 {author_ref} by Total Publications and Their Most Recent Affiliation",
    label="tab:top_authors_and_citations",
    index=False,
    column_format=column_format,
    escape=False,  # IMPORTANT: 'False' so that LaTeX commands in the header work
    float_format="%.1f",
    header=custom_header
)

print("\n--- TABLE LATEX CODE ---")
print(latex_table)