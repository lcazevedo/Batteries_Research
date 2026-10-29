
import utils
import config

import pandas as pd
import plotly.express as px
import numpy as np

df_data = utils.get_df_data()

# Define periods and create interval dictionary
periods = {
    "1993-2004": (1993, 2004),
    "2005-2014": (2005, 2014),
    "2015-2024": (2015, 2024),
    "1993-2024": (1993, 2024)
}

# Count by country in each period
country_counts = {}

for label, (start, end) in periods.items():
    df_period = df_data[(df_data[config.COLUMN_YEAR] >= start) & (df_data[config.COLUMN_YEAR] <= end)]
    counts = df_period[config.COLUMN_FIRST_COUNTRY].value_counts().reset_index()
    counts.columns = ['country', 'publications']
    country_counts[label] = counts

# Find maximum value to standardize color scale
max_value = max(df['publications'].max() for df in country_counts.values())

# Generate charts separately
for label, df_counts in country_counts.items():
    fig = px.choropleth(
        df_counts,
        locations='country',
        locationmode='country names',
        color=df_counts['publications'], #'publications',
        color_continuous_scale='YlOrRd',
        range_color=(0, max_value), #(0, max_value),
        labels={'publications': 'Publications'},
        title=label
    )

    fig.update_layout(
        title=dict(
            text=label,
            x=0.5,
            xanchor='center',
            font=dict(size=20)
        ),
        coloraxis_colorbar=dict(
            title='Publications',
            title_font=dict(size=20),
            tickfont=dict(size=20)
        ),
        margin=dict(l=0, r=0, t=50, b=0)
    )

    # Save as image
    fig.write_image(f"{config.RESULTS_FOLDER}/q4_1_2_mapa_publicacoes_abs_{label.replace('-', '_')}.jpg", width=900, height=500)

    # Show interactively (optional)
    #fig.show()

# Generate latex table of data
# Initialize set with top 10 countries from each period
top_countries = set()

# Identify top 10 countries per period
for period, df in country_counts.items():
    df_clean = df[df['country'] != 'UNKNOWN']
    top10 = df_clean.nlargest(10, 'publications')['country']
    top_countries.update(top10)

# Create consolidated DataFrame with relevant countries
consolidated_df = pd.DataFrame({'country': sorted(top_countries)})

for period, df in country_counts.items():
    df_clean = df[df['country'] != 'UNKNOWN']
    df_period = df_clean[df_clean['country'].isin(top_countries)]

    df_grouped = df_period.groupby('country', as_index=False)['publications'].sum()
    df_grouped.rename(columns={'publications': period}, inplace=True)

    consolidated_df = consolidated_df.merge(df_grouped, on='country', how='left')

# Fill missing values with zero and convert to integers
consolidated_df.fillna(0, inplace=True)
consolidated_df = consolidated_df.set_index('country').astype(int)

# Sort by values from period 1993-2024, if it exists
if '1993-2024' in consolidated_df.columns:
    consolidated_df = consolidated_df.sort_values(by='1993-2024', ascending=False)

# Generate LaTeX
latex_table = consolidated_df.to_latex(
    index=True,
    caption='Top 10 countries by period in number of publications on H₂ electrochemistry',
    label='tab:top10_publications_country',
    bold_rows=True,
    column_format='l' + 'r' * len(consolidated_df.columns)
)

tex_file = f"{config.RESULTS_FOLDER}/q4_1_2_publications_by_decade_abs.tex"
with open(tex_file, "w") as f:
    f.write(latex_table)
