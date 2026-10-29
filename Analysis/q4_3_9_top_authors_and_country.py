import utils
import config

import pandas as pd
import numpy as np
import textwrap

from tqdm import tqdm
import numpy as np
import plotly.graph_objects as go


tqdm.pandas()

# Load data
df_data = utils.get_df_data()

fs1 = 42
fs2 = 32
fs3 = 30

# Filter data: remove "UNKNOWN" countries
df_filtered = df_data[df_data[config.COLUMN_FIRST_COUNTRY] != "Unknown"]

# Select top 10 authors by number of publications
print("Processing top_authors...")
top_authors = df_filtered[config.COLUMN_FIRST_AUTHOR_FULL_NAME].value_counts().nlargest(10).index
print("Processing df_top...")
df_top = df_filtered[df_filtered[config.COLUMN_FIRST_AUTHOR_FULL_NAME].isin(top_authors)]

# Group data by author and country
print("Processing summary...")
summary = df_top.groupby([config.COLUMN_FIRST_AUTHOR_FULL_NAME, config.COLUMN_FIRST_COUNTRY]).agg(
    publications=(config.COLUMN_FIRST_AUTHOR_FULL_NAME, 'count'),
    citations=(config.COLUMN_CITATIONS_COUNT, 'sum')
).reset_index()

# Only authors with 5 or more publications will have a visible label
summary['label'] = summary.progress_apply(
    lambda row: row[config.COLUMN_FIRST_AUTHOR_FULL_NAME] if row['publications'] >= 5 else '', axis=1
)

# Calculate sizeref
max_citations = summary['citations'].max()
sizeref = 2. * max_citations / (40. ** 2)
sizeref = sizeref * (2/3)

# Determine text positions
summary['textposition'] = 'top center'
summary.loc[summary[config.COLUMN_FIRST_AUTHOR_FULL_NAME].str.strip() == 'Liu, Yang', 'textposition'] = 'bottom center'
summary.loc[summary[config.COLUMN_FIRST_AUTHOR_FULL_NAME].str.strip() == 'Wang, Jie', 'textposition'] = 'bottom center'
summary.loc[summary[config.COLUMN_FIRST_AUTHOR_FULL_NAME].str.strip() == 'Wang, Wei', 'textposition'] = 'middle left'
#summary.loc[summary[config.COLUMN_FIRST_AUTHOR_FULL_NAME].str.strip() == 'Wang, Bo', 'textposition'] = 'bottom center'
summary.loc[summary[config.COLUMN_FIRST_AUTHOR_FULL_NAME].str.strip() == 'Li, Wei', 'textposition'] = 'middle right'


# Create figure
fig = go.Figure()

# Create one trace per country
print("Adding traces...")
for country in summary[config.COLUMN_FIRST_COUNTRY].unique():
    country_data = summary[summary[config.COLUMN_FIRST_COUNTRY] == country]
    
    fig.add_trace(go.Scatter(
        x=country_data['publications'],
        y=country_data['citations'],
        mode='markers+text',
        name=country,
        text=country_data['label'],
        textposition=country_data['textposition'],
        textfont=dict(size=fs2),
        marker=dict(
            size=country_data['citations'] / sizeref,
            sizemode='area',
            sizeref=1,
            line=dict(width=1, color='DarkSlateGrey')
        ),
        hovertext=country_data[config.COLUMN_FIRST_AUTHOR_FULL_NAME],
        hoverinfo='text',
        showlegend=True
    ))

# Layout
fig.update_layout(
    width=1300,
    height=1000,
    plot_bgcolor='white',
    paper_bgcolor='white',
    title="Top 10 Authors by Number of Publications and Countries",
    title_font=dict(size=fs1),
    xaxis=dict(
        title='No. of Publications',
        title_font=dict(size=fs2),
        tickfont=dict(size=fs2),
        showline=True,
        linecolor='black',
        gridcolor='lightgray',
        gridwidth=1,
        griddash='dot'
    ),
    yaxis=dict(
        title='No. of Citations',
        title_font=dict(size=fs2),
        tickfont=dict(size=fs2),
        showline=True,
        linecolor='black',
        gridcolor='lightgray',
        gridwidth=1,
        griddash='dot'
    ),
    legend_title_text='Countries',
    legend_title_font=dict(size=fs2),
    legend=dict(font=dict(size=fs2))
)

# Save image
fig.write_image( f'{config.RESULTS_FOLDER}/q4_3_9_top_authors_and_country.pdf')

# ==============================
# Export grouped and sorted LaTeX table
# ==============================

df_latex = summary.copy()

# Rename columns
df_latex = df_latex.rename(columns={
    config.COLUMN_FIRST_AUTHOR_FULL_NAME: 'Author',
    config.COLUMN_FIRST_COUNTRY: 'Country',
    'publications': 'Publications',
    'citations': 'Citations'
})

# ==============================
# CORRECT SORTING
# ==============================

# 1. Calculate total publications per country
country_order = (
    df_latex.groupby('Country')['Publications']
    .sum()
    .sort_values(ascending=False)
)

# 2. Sort dataframe:
df_latex = df_latex.sort_values(
    by=['Country', 'Publications', 'Citations'],
    ascending=[True, False, False]
)

# 3. Apply custom country order
df_latex['Country'] = pd.Categorical(
    df_latex['Country'],
    categories=country_order.index,
    ordered=True
)

df_latex = df_latex.sort_values(
    by=['Country', 'Publications'],
    ascending=[True, False]
)

# ==============================
# Optional formatting
# ==============================

df_latex['Citations'] = df_latex['Citations'].map('{:,}'.format)

# ==============================
# Build LaTeX table
# ==============================

lines = []

lines.append(r'\begin{table}[ht]')
lines.append(r'\centering')
lines.append(r'\caption{Top authors grouped by country (ordered by productivity)}')
lines.append(r'\label{tab:top_authors_country}')
lines.append(r'\begin{tabular}{llrr}')
lines.append(r'\toprule')
lines.append(r'Country & Author & Publications & Citations \\')
lines.append(r'\midrule')

# Group in the correct order
for country in country_order.index:
    group = df_latex[df_latex['Country'] == country].reset_index(drop=True)
    n = len(group)

    for i, row in group.iterrows():
        author = row['Author']
        pubs = row['Publications']
        cites = row['Citations']

        if i == 0:
            lines.append(
                rf'\multirow{{{n}}}{{*}}{{{country}}} & {author} & {pubs} & {cites} \\'
            )
        else:
            lines.append(
                rf' & {author} & {pubs} & {cites} \\'
            )

    lines.append(r'\midrule')

lines.append(r'\bottomrule')
lines.append(r'\end{tabular}')
lines.append(r'\end{table}')

latex_table = '\n'.join(lines)

# Save file
with open(f"{config.RESULTS_FOLDER}/q4_3_9_top_authors_and_country.tex", "w") as f:
    f.write(latex_table)

print("LaTeX table exported successfully!")
