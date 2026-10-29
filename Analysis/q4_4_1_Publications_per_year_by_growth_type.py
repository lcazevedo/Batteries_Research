import utils
import config

import pandas as pd
import numpy as np
import textwrap
import matplotlib.pyplot as plt

from tqdm import tqdm
import numpy as np
import plotly.graph_objects as go
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import LinearRegression

tqdm.pandas()

# Load data
df_data = utils.get_df_data()
topic_index_df = utils.get_topic_index_df()

# ======================
# Settings
# ======================
# Font sizes
fs1 = 30  # chart title
fs2 = 18  # axes
fs3 = 15  # legend

# ======================
# Data preparation
# ======================
# Filter year range
df_filtered = df_data[df_data[config.COLUMN_YEAR].between(2014, 2024)]

# Create dictionary: {topic_name: rank}
rank_dict = dict(
    topic_index_df[['topic', 'rank_index_wo_review']].values
)


# Count publications per year and topic
yearly_counts = df_filtered.groupby([config.COLUMN_YEAR, config.COLUMN_TOPIC_NAME]).size().reset_index(name='count')
pivot_table = yearly_counts.pivot(index=config.COLUMN_YEAR, columns=config.COLUMN_TOPIC_NAME, values='count').fillna(0)

# ======================
# Classify trends
# ======================
linear_topics = []
flat_topics = []
exp_topics = []

years = pivot_table.index.values.reshape(-1, 1)

for topic in pivot_table.columns:
    counts = pivot_table[topic].values

    if np.count_nonzero(counts) < 5:
        continue  # ignore topics with sparse data

    # Linear fit
    lin_model = LinearRegression()
    lin_model.fit(years, counts)
    pred_linear = lin_model.predict(years)

    # R² of linear fit
    ss_res = np.sum((counts - pred_linear) ** 2)
    ss_tot = np.sum((counts - np.mean(counts)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Check if it looks exponential
    with np.errstate(divide='ignore'):
        log_counts = np.log1p(counts)  # log(1 + x) to avoid log(0)

    exp_model = LinearRegression()
    exp_model.fit(years, log_counts)
    pred_exp = exp_model.predict(years)

    exp_r2 = 1 - np.sum((log_counts - pred_exp) ** 2) / np.sum((log_counts - np.mean(log_counts)) ** 2)

    slope = lin_model.coef_[0]

    if r2 > 0.9 and slope > 2:
        linear_topics.append(topic)
    elif exp_r2 > 0.9 and exp_model.coef_[0] > 0.08:
        exp_topics.append(topic)
    else:
        flat_topics.append(topic)


# ======================
# Plotting function
# ======================
def plot_group(ax, topics, letter, title):
    from itertools import cycle
    import matplotlib.pyplot as plt

    # Color palette and markers
    color_cycle = plt.cm.get_cmap('tab20', len(topics)).colors
    marker_styles = ['o', 's', 'v', '^', '<', '>', 'D', 'P', '*', 'X', 'H']
    color_iterator = cycle(color_cycle)
    marker_iterator = cycle(marker_styles)

    for topic in topics:
        # Get topic rank; if not found, leave empty
        rank = rank_dict.get(topic, '')
        rank_label = f"({rank}) " if rank else ""
        
        
        # Format topic name
        label_wrapped = ' '.join(topic.split(maxsplit=1))

        # Final label with rank
        label_final = f"{rank_label}{label_wrapped}"

        # Color and marker for each line
        color = next(color_iterator)
        marker = next(marker_iterator)

        ax.plot(
            pivot_table.index,
            pivot_table[topic],
            label=label_final,
            color=color,
            marker=marker,
            linewidth=2,
            markersize=8
        )

    ax.text(
        0.01, 0.98, letter,
        transform=ax.transAxes,
        fontsize=fs2+1,
        va='top',
        ha='left',
    )

    ax.set_title(title, fontsize=fs2)
    ax.set_xlabel("", fontsize=fs2)
    ax.set_ylabel("No. of Publications", fontsize=fs2)
    ax.tick_params(labelsize=fs2)
    ax.set_xticks(pivot_table.index[::max(len(pivot_table.index)//4,1)])

    # Rotating X-axis labels
    ax.tick_params(axis='x', labelrotation=45)

    # Remove grid lines
    ax.grid(False)

    # Legend with white background and reduced width
    leg = ax.legend(
        fontsize=fs3,
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        frameon=True,
        fancybox=False,
        framealpha=1,
        borderpad=0.5,
        handlelength=1.5,
        handletextpad=0.5,
        borderaxespad=0.5
    )
    leg.get_frame().set_facecolor('white')
    leg.get_frame().set_linewidth(0)

    ax.set_facecolor('white')

# ======================
# Generate charts
# ======================
fig = plt.figure(figsize=(10, 15), facecolor='white')
gs = GridSpec(3, 1)

#plt.suptitle("Classification of Publication Growth Patterns\nin Metal-Ion Battery Topics", fontsize=fs1, y=1.0)


ax1 = fig.add_subplot(gs[0])
plot_group(ax1, linear_topics, "(a)", "Linear Growth")

ax2 = fig.add_subplot(gs[1])
plot_group(ax2, flat_topics, "(b)", "Plateau")

ax3 = fig.add_subplot(gs[2])
plot_group(ax3, exp_topics, "(c)", "Exponential Growth")
# If you want to enable logarithmic scale on the last chart:
# ax3.set_yscale('log')

# Separator lines
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Update real positions
fig.canvas.draw()

# Axis positions
pos1 = ax1.get_position()
pos2 = ax2.get_position()
pos3 = ax3.get_position()

# Midpoint between charts
y_sep1 = (pos1.y0 + pos2.y1) / 2
y_sep2 = (pos2.y0 + pos3.y1) / 2

for y in [y_sep1, y_sep2]:
    y -= 0.01
    fig.add_artist(
        plt.Line2D(
            [pos1.x0, pos1.x1+0.5],
            [y, y],
            transform=fig.transFigure,
            color='black',
            linewidth=1.5
        )
    )



plt.savefig(f'{config.RESULTS_FOLDER}/q4_4_1_Publications_per_year_by_growth_type.pdf', bbox_inches='tight') #, dpi=300)



#################################
# LATEX table


# Function to build table for a group
def build_group_table(topics, group_label):
    if not topics:
        return pd.DataFrame()

    df = pivot_table[topics].T  # transpose: topics on rows

    # Add rank to topic name
    new_index = []
    for topic in df.index:
        rank = rank_dict.get(topic, '')
        rank_label = f"({rank}) " if rank else ""
        new_index.append(f"{rank_label}{topic}")

    df.index = new_index
    df.index.name = 'Tópico'

    # Add group column
    df.insert(0, 'Grupo', group_label)

    return df.reset_index()

rank_dict = dict(topic_index_df[['topic', 'rank_index_wo_review']].values)


# Build tables
df_linear = build_group_table(linear_topics, 'Linear')
df_flat = build_group_table(flat_topics, 'Plateau')
df_exp = build_group_table(exp_topics, 'Exponential')

# Concatenate the three tables
df_concat = pd.concat([df_linear, df_flat, df_exp])

# Sort to keep organized (optional)
df_concat = df_concat.set_index(['Grupo', 'Tópico'])

# Fill NaN with zero and ensure integers
df_concat = df_concat.fillna(0).astype(int)


# Generate raw LaTeX
#latex_buffer = StringIO()
latex_table = df_concat.to_latex(
    f'{config.RESULTS_FOLDER}/q4_4_1_Publications_per_year_by_growth_type.tex',
    #buf=latex_buffer,
    multirow=True,
    index=True,
    na_rep="0",
    float_format="%.0f",
    caption="Growth in the Number of Publications by Topic",
    label="tab:q4_4_1_Publications_per_year_by_growth_type",
    bold_rows=False,
    column_format="l" * 2 + "r" * len(df_concat.columns)  # l for group and topic, r for years
)

#latex_table = latex_buffer.getvalue()

latex_table
