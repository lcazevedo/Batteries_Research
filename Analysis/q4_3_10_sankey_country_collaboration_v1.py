
import utils
import config

import pandas as pd
import plotly.graph_objects as go
from itertools import tee
from collections import Counter

df_data = utils.get_df_data()


fs1 = 32
fs2 = 28

# =============================
# Helper function for consecutive pairs
# =============================
def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

# =============================
# Select the top 5 countries
# =============================
country_counter = Counter()
for ctr_list in df_data[config.COLUMN_AUTHORS_COUNTRY]:
    country_counter.update(set(ctr_list))  

top5_countries = [country for country, _ in country_counter.most_common(10)]

# =============================
# Generate flows with at most 3 unique countries per doc
# =============================
country_links = []

for _, row in df_data.iterrows():
    countries = [c for c in row[config.COLUMN_AUTHORS_COUNTRY] if c in top5_countries]

    # Remove duplicates while preserving order
    seen = set()
    unique_countries = []
    for c in countries:
        if c not in seen:
            seen.add(c)
            unique_countries.append(c)

    if len(unique_countries) >= 2:
        limited_countries = unique_countries[:3]  # Limit to at most 3 countries
        labeled_countries = [f"{c} ({i+1})" for i, c in enumerate(limited_countries)]
        for source, target in pairwise(labeled_countries):
            country_links.append((source, target))

# =============================
# Create DataFrame with flows
# =============================
df_links = pd.DataFrame(country_links, columns=['source', 'target'])
df_links['value'] = 1
df_links = df_links.groupby(['source', 'target']).agg({'value': 'sum'}).reset_index()

# =============================
# Create nodes and indices
# =============================
nodes = sorted(set(df_links['source']) | set(df_links['target']))
node_map = {label: i for i, label in enumerate(nodes)}

df_links['source_idx'] = df_links['source'].map(node_map)
df_links['target_idx'] = df_links['target'].map(node_map)

# =============================
# Sankey Diagram (corrected)
# =============================

source = df_links['source_idx'].tolist()
target = df_links['target_idx'].tolist()
values = df_links['value'].tolist()


fig = go.Figure(data=[go.Sankey(
    arrangement="snap",
    node=dict(
        pad=30,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=nodes,
    ),
    link=dict(
        source=source,
        target=target,
        value=values
    )
)])

fig.update_layout(
    title_text="Collaborations between the 10 most productive countries",
    font_size=fs1,
    width=1100,
    height=900,
    font=dict(
        size=fs2,
        color="black"
    )
)

fig.write_image(f'{config.RESULTS_FOLDER}/q4_3_10_sankey_country_collaboration.pdf')


# =============================
# Prepare table for LaTeX
# =============================

df_latex = df_links.copy()

# Remove level index (1), (2), (3) from names
df_latex['source_clean'] = df_latex['source'].str.replace(r' \(\d+\)', '', regex=True)
df_latex['target_clean'] = df_latex['target'].str.replace(r' \(\d+\)', '', regex=True)

# Rename columns to something more friendly
df_latex = df_latex[['source_clean', 'target_clean', 'value']]
df_latex.columns = ['Source Country', 'Target Country', 'Number of Collaborations']

# Sort by number of collaborations (descending)
df_latex = df_latex.sort_values(by='Number of Collaborations', ascending=False)
