import utils
import config

import pandas as pd
import plotly.graph_objects as go

df_data = utils.get_df_data()

# 1. Obtain and sort the 6 most frequent regions
top_regions = df_data[df_data[config.COLUMN_FIRST_COUNTRY_REGION] != "Others"][config.COLUMN_FIRST_COUNTRY_REGION].value_counts().nlargest(6).index

# 2. Filter the DataFrame to contain only these regions
df_filtered = df_data[df_data[config.COLUMN_FIRST_COUNTRY_REGION].isin(top_regions)].copy()

# 3. Convert the 'country_region' column to an ordered categorical type.
# This ensures that the regions appear in descending order in the chart.
df_filtered[config.COLUMN_FIRST_COUNTRY_REGION] = pd.Categorical(df_filtered[config.COLUMN_FIRST_COUNTRY_REGION], categories=top_regions, ordered=True)

# 4. Group, count, and select the 5 most frequent countries for each region
top_countries_by_region = (
    df_filtered.groupby([config.COLUMN_FIRST_COUNTRY_REGION, config.COLUMN_FIRST_COUNTRY], observed=True)
    .size()
    .reset_index(name='count')
    .sort_values([config.COLUMN_FIRST_COUNTRY_REGION, 'count'], ascending=[True, False]) # 'country_region' will respect the categorical order
    .groupby(config.COLUMN_FIRST_COUNTRY_REGION, observed=True)
    .head(5)
)

# 5. Calculate the totals for each region (sum of the counts of its filtered countries)
region_totals = top_countries_by_region.groupby(config.COLUMN_FIRST_COUNTRY_REGION, observed=True)['count'].sum()

# 6. Create the labels for the diagram nodes, including the counts
# Region labels: "Region Name (Total)"
region_labels = [f"{region} ({count})" for region, count in region_totals.items()]

# Country labels: "Country Name (Count)"
country_labels = [f"{row[config.COLUMN_FIRST_COUNTRY]} ({row['count']})" for idx, row in top_countries_by_region.iterrows()]

# Final list of all labels (nodes)
labels = region_labels + country_labels

# 7. Map the labels to numerical indices, which will be used to create the links
label_to_index = {label: i for i, label in enumerate(labels)}

# 8. Create the lists of 'source', 'target', and 'value' for the links
# Maps the source regions to their respective indices
sources = top_countries_by_region[config.COLUMN_FIRST_COUNTRY_REGION].apply(
    lambda region: label_to_index[f"{region} ({region_totals[region]})"]
)

# Maps the target countries to their respective indices
targets = top_countries_by_region.apply(
    lambda row: label_to_index[f"{row[config.COLUMN_FIRST_COUNTRY]} ({row['count']})"], axis=1
)

# Link values are the counts for each country
values = top_countries_by_region['count']

# ==============================
# 9. Create the Sankey Diagram figure
# ==============================

fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=30,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=labels,  # Uses the new labels with counts
    ),
    link=dict(
        source=sources,
        target=targets,
        value=values
    )
)])

fig.update_layout(
    title_text="Top 6 Regions → Top 5 Countries (referring to the first author)",
    font_size=24,
    width=1000,
    height=1000,
    font=dict(
        size=20,
        color="black"
    )
)

fig.write_image(f'{config.RESULTS_FOLDER}/q4_1_1_Sankey_region_country.pdf')



# reference latex table:
# Create pivoted table
df_table = top_countries_by_region.pivot(
    index=config.COLUMN_FIRST_COUNTRY_REGION,
    columns=config.COLUMN_FIRST_COUNTRY,
    values='count'
).fillna(0).astype(int)

# Export to LaTeX
latex_table = df_table.to_latex(
    caption="Top 5 countries for each of the top 6 regions (first author affiliation). Values correspond to the flows shown in the Sankey diagram.",
    label="tab:sankey_region_country",
    column_format="l" + "r"*len(df_table.columns)
)

# Save file
with open(f"{config.RESULTS_FOLDER}/q4_1_1_Sankey_region_country_table.tex", "w") as f:
    f.write(latex_table)
