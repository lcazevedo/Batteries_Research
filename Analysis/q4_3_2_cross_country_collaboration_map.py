import utils
import config

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as pc

# Load data
df_data = utils.get_df_data()

# Fonts
fs1 = 40  # Title
fs2 = 30  # Axis titles
fs3 = 28  # Data labels

def generate_color_bar_shapes(scale, x_start=1.0, x_end=1.015, y_start=0.0, y_end=1.0, steps=100, gamma=0.6):
    colors = pc.sample_colorscale(scale, [(i / steps) ** gamma for i in range(steps)])
    shapes = []
    for i in range(steps):
        y0 = y_start + (y_end - y_start) * i / steps
        y1 = y_start + (y_end - y_start) * (i + 1) / steps
        shapes.append(dict(
            type="rect",
            xref="paper", yref="paper",
            x0=x_start, x1=x_end,
            y0=y0, y1=y1,
            fillcolor=colors[i],
            line=dict(width=0)
        ))
    return shapes

# ---------------------------
#  Normalize country names
# ---------------------------
def normalize_country_name(name):
    if not isinstance(name, str):
        return name
    name = name.strip()
    name_upper = name.upper()
    # Common mappings / variants -> name recognized by Plotly / ISO
    mapping = {
        'PEOPLES R CHINA': 'China',
        'PEOPLE\'S R CHINA': 'China',
        'PEOPLES R. CHINA': 'China',
        'PEOPLES R. CHINA': 'China',
        'UNITED STATES OF AMERICA': 'United States',
        'UNITED STATES': 'United States',
        'USA': 'United States',
        'U.S.A.': 'United States',
        'ENGLAND': 'United Kingdom',
        'SCOTLAND': 'United Kingdom',
        'WALES': 'United Kingdom',
        'NORTHERN IRELAND': 'United Kingdom',
        'NORTH IRELAND': 'United Kingdom',
        'UK': 'United Kingdom',
        'RUSSIAN FEDERATION': 'Russia',
        'IRAN (ISLAMIC REPUBLIC OF)': 'Iran',
        'SOUTH KOREA': 'South Korea',
        'S. KOREA': 'South Korea',
        'KOREA, SOUTH': 'South Korea',
        'KOREA, NORTH': 'North Korea',
        'CZECH REPUBLIC': 'Czechia',
        'BOLIVIA (PLURINATIONAL STATE OF)': 'Bolivia',
        'Viet Nam'.upper(): 'Vietnam',
        'JAPAN': 'Japan',
        'PEOPLE R CHINA': 'China'
    }
    if name_upper in mapping:
        return mapping[name_upper]
    # fallback: Title case (Plotly expects "Belgium", "Brazil", etc.)
    return name.title()

# Normalize country lists in column (and create countries_unique)
df_data['countries_unique'] = df_data[config.COLUMN_AUTHORS_COUNTRY].apply(
    lambda lst: list({normalize_country_name(c) for c in lst if isinstance(c, str)})
)

# Normalize first_country (in case of variants)
df_data[config.COLUMN_FIRST_COUNTRY] = df_data[config.COLUMN_FIRST_COUNTRY].apply(
    lambda c: normalize_country_name(c) if isinstance(c, str) else c
)

# ---------------------------
# Counts per country
# ---------------------------
pubs_by_country = df_data[config.COLUMN_FIRST_COUNTRY].value_counts().reset_index()
pubs_by_country.columns = ['country', 'count']

# Avoid zeros / NaNs and convert country to Title case
pubs_by_country['country'] = pubs_by_country['country'].astype(str)
pubs_by_country = pubs_by_country[pubs_by_country['country'] != 'nan']

# Log for color mapping (keep real count for legends)
pubs_by_country['log_count'] = np.log1p(pubs_by_country['count'])

# ---------------------------
# Build collaboration counts
# ---------------------------
collaborations = Counter()
for country_list in df_data['countries_unique']:
    if len(country_list) > 1:
        for pair in combinations(sorted(country_list), 2):
            collaborations[pair] += 1

# ---------------------------
# Coordinates per country (lat, lon)
# Added several of the countries listed.
# Values are approximate centroids.
# ---------------------------
country_coords = {
    'United States': (39.5, -98.35),
    'Brazil': (-14.235, -51.9253),
    'China': (35.8617, 104.1954),
    'Germany': (51.1657, 10.4515),
    'India': (20.5937, 78.9629),
    'United Kingdom': (55.3781, -3.4360),
    'France': (46.6034, 2.2137),
    'Canada': (56.1304, -106.3468),
    'Japan': (36.2048, 138.2529),
    'Australia': (-25.2744, 133.7751),
    'Italy': (41.8719, 12.5674),
    'Spain': (40.4637, -3.7492),
    'Russia': (61.5240, 105.3188),
    'South Korea': (35.9078, 127.7669),
    'South Africa': (-30.5595, 22.9375),
    'Saudi Arabia': (23.8859, 45.0792),
    'Singapore': (1.3521, 103.8198),
    'Taiwan': (23.6978, 120.9605),
    'Iran': (32.4279, 53.6880),
    'Switzerland': (46.8182, 8.2275),
    'Poland': (51.9194, 19.1451),
    'Egypt': (26.8206, 30.8025),
    'Sweden': (60.1282, 18.6435),
    'Portugal': (39.3999, -8.2245),
    'Czechia': (49.8175, 15.4730),
    'Denmark': (56.2639, 9.5019),
    'Vietnam': (14.0583, 108.2772),
    'Turkey': (38.9637, 35.2433),
    'Malaysia': (4.2105, 101.9758),
    'Israel': (31.0461, 34.8516),
    'United Arab Emirates': (23.4241, 53.8478),
    'Mexico': (23.6345, -102.5528),
    'Norway': (60.4720, 8.4689),
    'Netherlands': (52.1326, 5.2913),
    # Requested additions
    'Belgium': (50.5039, 4.4699),
    'Thailand': (15.8700, 100.9925),
    'Pakistan': (30.3753, 69.3451),
    'Austria': (47.5162, 14.5501),
    'Indonesia': (-0.7893, 113.9213),
    'Finland': (61.9241, 25.7482),
    'Morocco': (31.7917, -7.0926),
    'Ireland': (53.4129, -8.2439),
    'Kazakhstan': (48.0196, 66.9237),
    'Slovenia': (46.1512, 14.9955),
    'Iraq': (33.2232, 43.6793),
    'Argentina': (-38.4161, -63.6167),
    'Qatar': (25.3548, 51.1839),
    'Bulgaria': (42.7339, 25.4858),
    'Ukraine': (48.3794, 31.1656),
    'Bangladesh': (23.6850, 90.3563),
    'Greece': (39.0742, 21.8243),
    'Romania': (45.9432, 24.9668),
    'Serbia': (44.0165, 21.0059),
    'Chile': (-35.6751, -71.5430),
    'Slovakia': (48.6690, 19.6990),
    'Tunisia': (33.8869, 9.5375),
    'Nigeria': (9.0820, 8.6753),
    'New Zealand': (-40.9006, 174.8860),
    'Latvia': (56.8796, 24.6032),
    'Estonia': (58.5953, 25.0136),
    'Hungary': (47.1625, 19.5033),
    'Colombia': (4.5709, -74.2973),
    'Lithuania': (55.1694, 23.8813),
    'Croatia': (45.1, 15.2),
    'Sri Lanka': (7.8731, 80.7718),
    'Algeria': (28.0339, 1.6596),
    'North Korea': (40.3399, 127.5101),
    'Newfoundland': (49.5, -55.0),  # example if it appears
    # if more countries are missing, add below
}

# ---------------------------
# Choropleth (ensure country names match)
# ---------------------------
# Group pubs_by_country by normalized names (in case of duplicates)
pubs_by_country['country'] = pubs_by_country['country'].apply(normalize_country_name)
pubs_by_country = pubs_by_country.groupby('country', as_index=False)[['count', 'log_count']].sum()

# Create choropleth using raw count with fixed range_color
fig = px.choropleth(
    pubs_by_country,
    locations="country",
    locationmode="country names",
    color="count",
    color_continuous_scale="Reds",
    projection="natural earth",
    range_color=(0, pubs_by_country['count'].max()),
)



# ---------------------------
# Add collaboration lines (filter top N to avoid clutter)
# ---------------------------
# Build sorted list of strongest collaborations
collab_items = sorted(collaborations.items(), key=lambda x: x[1], reverse=True)

# Control parameters (adjust as needed)
TOP_COLLAB = 500    # only draw the top N collaborations (reduces clutter)
MAX_LINE_WIDTH = 3.0  # maximum line width on map

# Find min/max for width normalization
weights = [w for (_, _), w in collab_items]
min_w = min(weights) if weights else 0
max_w = max(weights) if weights else 1

count_added = 0
for (country1, country2), weight in collab_items[:TOP_COLLAB]:
    # only draw if both have coordinates
    if country1 in country_coords and country2 in country_coords:
        lat1, lon1 = country_coords[country1]
        lat2, lon2 = country_coords[country2]
        # normalize width between 0.4 and MAX_LINE_WIDTH
        if max_w > min_w:
            norm = (weight - min_w) / (max_w - min_w)
        else:
            norm = 0.0
        lw = 0.4 + norm * (MAX_LINE_WIDTH - 0.4)
        fig.add_trace(go.Scattergeo(
            lon=[lon1, lon2],
            lat=[lat1, lat2],
            mode='lines',
            line=dict(width=lw, color='chocolate'),
            opacity=0.25 + 0.6 * norm,  # stronger => more opaque
            hoverinfo='text',
            text=f"{country1} ↔ {country2}: {weight} pubs"
        ))
        count_added += 1
    else:
        if country1 not in country_coords:
            print(f"Missing coordinates for {country1}, skipping collaboration with {country2}")
        if country2 not in country_coords:
            print(f"Missing coordinates for {country2}, skipping collaboration with {country1}")
print(f"Lines added: {count_added} (top {TOP_COLLAB})")

# ---------------------------
# Legend / colorbar annotations (kept your approach)
# ---------------------------
# Desired ticks
ticks_real = [0, 10000, 20000, 30000, 40000, 50000, 60000, 70000]

annotations = []
n = len(ticks_real) - 1

# This offset value represents half the visual font height in relative (paper) coordinates
# Can be slightly adjusted (e.g., 0.02 to 0.03) depending on the final height of your figure
offset = 0.035


for i, real in enumerate(ticks_real):
    # Calculate base position (from 0 to 1)
    base_pos = i / n

    # Apply compression so text fits exactly between 0 and 1
    y_pos = offset + (1 - 2 * offset) * base_pos

    label = f'{real}' if real < 1000 else f'{int(real/1000)}K'

    annotations.append(dict(
        x=1.02,
        y=y_pos,
        xref='paper',
        yref='paper',
        text=label,
        showarrow=False,
        font=dict(size=fs2),
        xanchor='left',
        yanchor='middle'
    ))

# colorbar title
annotations.append(dict(
    x=0.995,
    y=1.15,
    xref='paper',
    yref='paper',
    text='No. of<br>Publications',
    showarrow=False,
    font=dict(size=fs2),
    xanchor='center'
))

color_bar_shapes = generate_color_bar_shapes(
    scale='Reds',
    x_start=0.98,
    x_end=1.01,
    y_start=0,
    y_end=1,
    steps=100
)

# ---------------------------
# Final layout
# ---------------------------
fig.update_layout(
    title_text='International Collaboration',
    title_font_size=fs1,
    title_x=0.5,
    geo=dict(
        showland=True,
        landcolor='white',
        showframe=False,
        showcountries=True,
    ),
    margin=dict(l=0, r=140, t=100, b=0),  # more space on the right
    showlegend=False,
    height=700,
    width=1250,
    annotations=annotations,
    coloraxis_showscale=False,
    shapes=color_bar_shapes
)

# Save
fig.write_image(f"{config.RESULTS_FOLDER}/q4_3_2_cross_country_collaboration_map.pdf")
# fig.show()


# ==========================================
# EXPORT DATA FOR STATISTICAL ANALYSIS
# ==========================================

# 1. Export heatmap (Choropleth) data
# Contains country, publication count (count), and log (log_count)
caminho_csv_paises = f"{config.RESULTS_FOLDER}/q4_3_2_cross_country_collaboration_map_publicacoes_por_pais.csv"
pubs_by_country.to_csv(caminho_csv_paises, index=False, encoding='utf-8')
print(f"Country data exported to: {caminho_csv_paises}")

# 2. Export collaboration line data
# First convert the list of tuples 'collab_items' into a Pandas DataFrame
dados_colaboracao = []
for (country1, country2), weight in collab_items:
    dados_colaboracao.append({
        'pais_1': country1,
        'pais_2': country2,
        'quantidade_colaboracoes': weight
    })

df_colaboracoes = pd.DataFrame(dados_colaboracao)

caminho_csv_colaboracoes = f"{config.RESULTS_FOLDER}/q4_3_2_cross_country_collaboration_map_colaboracoes_entre_paises.csv"
df_colaboracoes.to_csv(caminho_csv_colaboracoes, index=False, encoding='utf-8')
print(f"Collaboration data exported to: {caminho_csv_colaboracoes}")
