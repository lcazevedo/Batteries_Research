import utils
import config

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df_data = utils.get_df_data()

# -----------------------------------------------------------------
# CONFIGURATION PARAMETERS
# -----------------------------------------------------------------

# N: Number of institutions in Top N (from your previous scripts)
N = 10

# M: Number of topics in Top M to display individually (PER INSTITUTION)
M_TOPICS = 5 

# Font sizes (maintaining visual identity of previous charts)
fs1 = 60  # Title
fs2 = 45  # Axis labels and ticks
fs_legend = 45 # Legend font
fs_internal = 32 # Font for percentages INSIDE bars

# Minimum percentage for a segment to display its label (avoids clutter)
MIN_PCT_FOR_LABEL = 7 # (in %)

# Thickness of horizontal bars
BAR_WIDTH = 0.3

# -----------------------------------------------------------------
# PROCESSING (PART 1: IDENTIFY TOP N INSTITUTIONS)
# -----------------------------------------------------------------

# 1. Remove "Unknown" entries if any exist
df_data_cleaned = df_data[df_data[config.COLUMN_FIRST_INSTITUTION] != "Unknown"]
df_data_cleaned[config.COLUMN_FIRST_INSTITUTION] = df_data_cleaned[config.COLUMN_FIRST_INSTITUTION].str.title()

# 2. Apply year filter (for consistency with previous script)
df_data_cleaned[config.COLUMN_YEAR] = pd.to_numeric(df_data_cleaned[config.COLUMN_YEAR])
#df_data_cleaned = df_data_cleaned[df_data_cleaned[config.COLUMN_YEAR] <= 2024]


# 3. Count total frequency of each institution (after filters)
institution_counts = df_data_cleaned[config.COLUMN_FIRST_INSTITUTION].value_counts()

# 4. Select Top N
top_n_data = institution_counts.head(N)
top_n_list = top_n_data.index.tolist()

print(f"Top {N} institutions identified (post-filter):")
print(top_n_list)
print("-" * 30)

# -----------------------------------------------------------------
# PROCESSING (PART 2: IDENTIFY AND GROUP TOPICS)
# (BLOCK MODIFIED FOR TOP M PER INSTITUTION)
# -----------------------------------------------------------------

# 1. Filter DataFrame to contain ONLY Top N institutions
df_top_n_only = df_data_cleaned[df_data_cleaned[config.COLUMN_FIRST_INSTITUTION].isin(top_n_list)]

# 2. Identify Top M topics for EACH institution and build a master set
all_top_m_topics_set = set()

for inst_name in top_n_list:
    # Filter df for this institution
    df_inst = df_top_n_only[df_top_n_only[config.COLUMN_FIRST_INSTITUTION] == inst_name]
    
    # Count *its* topics
    inst_topic_counts = df_inst[config.COLUMN_TOPIC_NAME].value_counts()
    
    # Get Top M for *this* institution
    top_m_for_inst = inst_topic_counts.head(M_TOPICS).index.tolist()
    
    # Add to master set (sets handle duplicates)
    all_top_m_topics_set.update(top_m_for_inst)

# 3. Create an ordered list of these topics (sorted by global frequency)
#    This helps keep the most important topics first in the legend/stack.
global_topic_counts = df_top_n_only[config.COLUMN_TOPIC_NAME].value_counts()

all_top_m_topics_list = sorted(
    list(all_top_m_topics_set), 
    key=lambda x: global_topic_counts.get(x, 0), 
    reverse=True
)

print(f"Total unique topics (Top {M_TOPICS} per institution) identified: {len(all_top_m_topics_list)}")
print("These topics will be plotted individually:")
print(all_top_m_topics_list)

# 4. Create a new column 'topic_plot' grouping "other" topics
#    A topic is "Others" if it is NOT in our master set 'all_top_m_topics_set'
df_top_n_only['topic_plot'] = df_top_n_only[config.COLUMN_TOPIC_NAME].apply(
    lambda x: x if x in all_top_m_topics_set else 'Others'
)

# 5. Define column (topic) order in the chart
ordered_topics_for_plot = all_top_m_topics_list + ['Others']
print("-" * 30)


# -----------------------------------------------------------------
# PROCESSING (PART 3: PIVOT DATA)
# (This block remains the same, now using reprocessed data from Part 2)
# -----------------------------------------------------------------

# 1. Group by Institution and Plotted Topic
df_grouped = df_top_n_only.groupby([config.COLUMN_FIRST_INSTITUTION, 'topic_plot']).size()

# 2. Pivot (Unstack) to have Institutions as rows and Topics as columns
df_pivot = df_grouped.unstack(level='topic_plot', fill_value=0)

# 3. Reorder columns (Topics)
#    Ensure 'Others' exists, even if not generated (rare case)
if 'Others' not in df_pivot.columns and 'Others' in ordered_topics_for_plot:
     df_pivot['Others'] = 0
     
df_pivot = df_pivot.reindex(columns=ordered_topics_for_plot, fill_value=0)

# 4. Reorder index (Institutions) by total count (Top N)
# And reverse order (iloc[::-1]) so barh plots #1 at the top
df_plot_data = df_pivot.reindex(top_n_data.index).iloc[::-1]

print("Pivoted data ready for plotting.")
print(df_plot_data.head())
print("-" * 30)

# -----------------------------------------------------------------
# CHART GENERATION (STACKED BAR CHART)
# (Section modified to fix plot width)
# -----------------------------------------------------------------

print("Generating stacked bar chart...")

# Colormap
num_colors_needed = len(ordered_topics_for_plot)
if num_colors_needed <= 10:
    cmap = plt.get_cmap('tab10')
elif num_colors_needed <= 20:
    cmap = plt.get_cmap('tab20')
else:
    print(f"Warning: Number of topics ({num_colors_needed}) exceeds 20. 'tab20' colors may repeat.")
    cmap = plt.get_cmap('tab20')

# Figure
fig_height = max(20, N * 1.5 + 4) 
fig, ax = plt.subplots(figsize=(24, fig_height))

# 1. Plot stacked bar chart
df_plot_data.plot(
    kind='barh', 
    stacked=True, 
    ax=ax, 
    cmap=cmap,
    width=BAR_WIDTH 
)

# 2. Configure Title and Labels
ax.set_title(f'Top {M_TOPICS} Topic Distribution by\n Institution (Top {N})', fontsize=fs1, pad=20)
ax.set_xlabel('Frequency (No. of Documents)', fontsize=fs2, labelpad=15)
ax.set_ylabel('Institution', fontsize=fs2, labelpad=15)

# 3. Configure Ticks
ax.tick_params(axis='both', labelsize=fs2)

# 4. Add Percentage Labels
# Get totals for each institution (to calculate percentage)
institution_totals = df_plot_data.sum(axis=1)

for i, (inst_name, row) in enumerate(df_plot_data.iterrows()):
    
    current_x = 0
    y_center = i 
    offset_toggle = 1 
    y_offset_amount = BAR_WIDTH / 1.2 

    for topic_name, width in row.items():
        if width == 0:
            continue
            
        total_for_inst = institution_totals.loc[inst_name]
        if total_for_inst == 0:
            continue

        pct = (width / total_for_inst) * 100
        
        if pct >= MIN_PCT_FOR_LABEL:
            x_pos = current_x + (width / 2)
            y_pos = y_center + (y_offset_amount * offset_toggle)
            
            ax.text(
                x_pos, 
                y_pos, 
                f'{pct:.0f}%', 
                ha='center', 
                va='center', 
                fontsize=fs_internal, 
                color='black',
                weight='normal'
            )
            
            offset_toggle *= -1
            
        current_x += width

# 5. CONFIGURE LEGEND (MODIFIED METHOD)

# 5.1. Determine number of legend columns
num_legend_items = len(ordered_topics_for_plot)
legend_cols = 1
if num_legend_items > 7:
    legend_cols = 2
if num_legend_items > 16:
    legend_cols = 3

# 5.2. Capture legend 'handles' and 'labels' generated by the plot
handles, labels = ax.get_legend_handles_labels()

# 5.3. Remove internal legend from axes (ax)
#      Crucial so tight_layout does not account for it
if ax.get_legend() is not None:
    ax.get_legend().remove()

# 6. ADJUST LAYOUT (NOW WITHOUT LEGEND)
#    Now tight_layout is called WITHOUT the legend present.
#    It expands 'ax' to use the available space.
#    'rect' reserves 5% space at the bottom for the legend.
plt.tight_layout(rect=[0, 0.05, 1, 1]) 

# 7. ADD LEGEND BACK TO THE FIGURE (fig)
fig.legend(
    handles,
    labels,
    loc='lower center',       # Position: bottom center of FIGURE
    bbox_to_anchor=(0.5, -0.6), # (0.5=horizontal center, 0.01=bottom of figure)
    fontsize=fs_legend,
    ncol=1, #legend_cols,
    frameon=False
)

# 8. SAVE CHART
file_name_plot = f'{config.RESULTS_FOLDER}/q4_4_5_institution_and_top_studied_topics.pdf'
try:
    # Use bbox_inches='tight' to include legend in saved PDF
    plt.savefig(file_name_plot, dpi=300, bbox_inches='tight')
    print(f"Chart saved successfully as: {file_name_plot}")
except Exception as e:
    print(f"Error saving chart: {e}")

# -----------------------------------------------------------------
# LATEX TABLE GENERATION (INSTITUTION x TOPICS)
# -----------------------------------------------------------------

print("Generating LaTeX table (Institutions x Topics)...")

def escape_latex(text):
    """Escape special LaTeX characters."""
    if not isinstance(text, str):
        return text
    return text.replace('&', '\\&').replace('_', '\\_').replace('%', '\\%')

latex_string = ""

latex_string += "\\setlength{\\LTleft}{0pt}\n"
latex_string += "\\setlength{\\LTright}{0pt}\n"

latex_string += f"\\begin{{longtable}}{{p{{4cm}} l p{{7cm}} l}}\n"

# Caption and label (inside longtable)
latex_string += f"\\caption{{Top ${N}$ Institution \\& Top ${M_TOPICS}$ Topics.}} \\\\\n"
latex_string += f"\\label{{tab:q4_4_5_institution_and_top_studied_topics}} \\\\\n"

# Header (first page)
latex_string += "\\toprule\n"
latex_string += f"Institution (Top ${N}$) & \\shortstack{{Total\\\\Docs}} & Topics (Top ${M_TOPICS}$) & \\shortstack{{Total\\\\Docs}} \\\\ \n"
latex_string += "\\midrule\n"
latex_string += "\\endfirsthead\n"

# Header (subsequent pages)
latex_string += "\\toprule\n"
latex_string += f"Institution (Top ${N}$) & \\shortstack{{Total\\\\Docs}} & Topics (Top ${M_TOPICS}$) & \\shortstack{{Total\\\\Docs}} \\\\ \n"
latex_string += "\\midrule\n"
latex_string += "\\endhead\n"

first_inst = True

for inst_name in top_n_list:
    if not first_inst:
        latex_string += "\\midrule\n"
    
    total_docs = institution_counts.loc[inst_name]
    safe_inst_name = escape_latex(inst_name).replace('TiO₂', '\\ce{TiO2}')
    
    df_inst = df_top_n_only[df_top_n_only[config.COLUMN_FIRST_INSTITUTION] == inst_name]
    inst_topic_counts = df_inst[config.COLUMN_TOPIC_NAME].value_counts()
    top_m_for_inst = inst_topic_counts.head(M_TOPICS)

    # Convert to list to access by index
    topics_list = list(top_m_for_inst.items())

    # --- 1. First row (institution + first topic) ---
    if len(topics_list) > 0:
        first_topic, first_count = topics_list[0]
        safe_topic_name = escape_latex(first_topic).replace('TiO₂', '\\ce{TiO2}')
        
        latex_string += f"{safe_inst_name} & {total_docs} & {safe_topic_name} & {first_count} \\\\ \n"

        # --- 2. Remaining topics ---
        for topic, count in topics_list[1:]:
            safe_topic_name = escape_latex(topic).replace('TiO₂', '\\ce{TiO2}')
            latex_string += f" & & {safe_topic_name} & {count} \\\\ \n"
    else:
        # In case there are no topics
        latex_string += f"{safe_inst_name} & {total_docs} & & \\\\ \n"
    
    first_inst = False

latex_string += "\\bottomrule\n"
latex_string += "\\end{longtable}\n"

file_name_latex = f'{config.RESULTS_FOLDER}/q4_4_5_institution_and_top_studied_topics.tex'

try:
    with open(file_name_latex, 'w', encoding='utf-8') as f:
        f.write(latex_string)
    print(f"LaTeX table saved successfully as: {file_name_latex}")
except Exception as e:
    print(f"Error saving LaTeX table: {e}")


print("\nScript finished.")
