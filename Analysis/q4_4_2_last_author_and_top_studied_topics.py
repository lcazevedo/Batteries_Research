import utils
import config

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

from PIL import Image

# Load data
df_data = utils.get_df_data()

# -----------------------------------------------------------------
# CONFIGURATION PARAMETERS
# -----------------------------------------------------------------

# Font
fs1, fs2, fs2_5, fs3 = 90, 64, 64, 56
fs_legenda = 60

largura = 36
altura = 34

# N: Number of authors in Top N (from your previous scripts)
N = 10

# M: Number of topics in Top M to display individually
# Remaining topics will be grouped into 'Others'
M_TOPICS = 5 

# Data processing
top_journals = (
    df_data[config.COLUMN_FIRST_AUTHOR_FULL_NAME]
    .value_counts()
    .nlargest(N)
    .index
)

df_top_journals = df_data[df_data[config.COLUMN_FIRST_AUTHOR_FULL_NAME].isin(top_journals)].copy()
df_top_journals[config.COLUMN_FIRST_AUTHOR_FULL_NAME] = df_top_journals[config.COLUMN_FIRST_AUTHOR_FULL_NAME].apply(
    lambda x: x.replace('Institute', 'Inst.').replace("(IIT System)", "").replace("&", "&").replace(" - China", "")
)

grouped = (
    df_top_journals
    .groupby([config.COLUMN_FIRST_AUTHOR_FULL_NAME, config.COLUMN_TOPIC_NAME])
    .size()
    .reset_index(name='count')
)

def get_top_topics_per_journal(df):
    return (
        df.sort_values([config.COLUMN_FIRST_AUTHOR_FULL_NAME, 'count'], ascending=[True, False])
          .groupby(config.COLUMN_FIRST_AUTHOR_FULL_NAME)
          .head(M_TOPICS)
    )

top_topics = get_top_topics_per_journal(grouped)

pivot_df = (
    top_topics
    .pivot_table(index=config.COLUMN_FIRST_AUTHOR_FULL_NAME, columns=config.COLUMN_TOPIC_NAME, values='count', fill_value=0)
)
pivot_df = pivot_df.loc[pivot_df.sum(axis=1).sort_values(ascending=True).index]

def simplify_label(label):
    parts = str(label).split('_')
    new_label = parts[1] if len(parts) > 1 else label
    new_label = re.sub(r'(\D)(\d+)', lambda m: f"{m.group(1)}$_{m.group(2)}$", new_label)
    return new_label

# ========== Step 1: chart without legend ==========
fig1, ax1 = plt.subplots(figsize=(largura, altura), facecolor='white')
plot = pivot_df.plot(
    kind='barh', stacked=True, colormap='tab20', ax=ax1, width=0.3
)

ax1.set_facecolor('white')
ax1.set_title('Topics by First Author (top 10 authors, top 5 topics)', fontsize=fs1,loc='right')
ax1.set_xlabel('No. of Documents', fontsize=fs2)
ax1.set_ylabel('', fontsize=fs2)
ax1.tick_params(axis='x', labelsize=fs2)
ax1.tick_params(axis='y', labelsize=fs2)
ax1.spines['bottom'].set_linestyle('-')
ax1.spines['left'].set_linestyle('-')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(axis='x', linestyle=':', linewidth=1.2)
ax1.grid(axis='y', visible=False)

# Remove the legend
ax1.get_legend().remove()

# Add percentage labels
row_totals = pivot_df.sum(axis=1)
toggle_per_row = {}

for container in plot.containers:
    for rect in container:
        width = rect.get_width()
        if width > 1:
            y_center = rect.get_y() + rect.get_height() / 2
            y_index = int(y_center)
            journal_name = plot.get_yticklabels()[y_index].get_text()
            total = row_totals[journal_name]
            percent = (width / total) * 100
            if journal_name not in toggle_per_row:
                toggle_per_row[journal_name] = True
            x = rect.get_x() + width / 2
            y = rect.get_y() + rect.get_height() + 0.05 if toggle_per_row[journal_name] else rect.get_y() - 0.05
            va = 'bottom' if toggle_per_row[journal_name] else 'top'
            ax1.text(x, y, f'{percent:.0f}%', ha='center', va=va, fontsize=fs3, color='black')
            toggle_per_row[journal_name] = not toggle_per_row[journal_name]

fig1.tight_layout()
fig1.savefig(f"{config.RESULTS_FOLDER}/q4_4_2_last_author_and_top_studied_topics_img1.jpg", bbox_inches='tight')

# ========== Step 2: isolated legend ==========
fig2, ax2 = plt.subplots(figsize=(largura, 8), facecolor='white')
handles, labels = plot.get_legend_handles_labels()
new_labels = [simplify_label(label) for label in labels]
ax2.legend(
    handles, new_labels,
    loc='center', ncol=2,
    fontsize=fs_legenda, frameon=False
)
ax2.axis('off')
fig2.savefig(f"{config.RESULTS_FOLDER}/q4_4_2_last_author_and_top_studied_topics_img2.jpg", bbox_inches='tight')

# ========== Step 3: combine images ==========
img1 = Image.open(f"{config.RESULTS_FOLDER}/q4_4_2_last_author_and_top_studied_topics_img1.jpg")
img2 = Image.open(f"{config.RESULTS_FOLDER}/q4_4_2_last_author_and_top_studied_topics_img2.jpg")

# Resize legend to match chart width
img2 = img2.resize((img1.width, img2.height+100))
combined_height = img1.height + img2.height
combined_img = Image.new('RGB', (img1.width, combined_height), color='white')
combined_img.paste(img1, (0, 0))
combined_img.paste(img2, (0, img1.height))

combined_img.save(f'{config.RESULTS_FOLDER}/q4_4_2_last_author_and_top_studied_topics.pdf')





# ========== Step 4: LaTeX Table Generation ==========

# Calculate total documents per author in top N
total_por_instituicao = df_top_journals[config.COLUMN_FIRST_AUTHOR_FULL_NAME].value_counts()

# Helper function to escape special LaTeX characters (such as '&')
def escape_latex(texto):
    texto = str(texto)
    # Avoid escaping an '&' that has already been escaped
    texto = texto.replace('\\&', '&').replace('&', '\\&')
    return texto

latex_code = [
    r"\begin{longtable}{",
    r"    >{\raggedright\arraybackslash}p{0.20\textwidth}",
    r"    c",
    r"    >{\raggedright\arraybackslash}p{0.55\textwidth}",
    r"    c",
    r"}",
    r"\caption{Ten leading first authors ranked by publication output and their five most frequently investigated research topics.} ",
    r"\label{tab:q4_4_2_last_author_and_top_studied_topics} \\",
    "",
    r"% --- First Head ---",
    r"\toprule",
    r"Top 10 & Total & Topics (Top 5) & Docs \\ ",
    r"\midrule",
    r"\endfirsthead",
    "",
    r"% --- Standard Head (for next pages) ---",
    r"\toprule",
    r"Top 10 & Total & Topics (Top 5) & Docs \\ ",
    r"\midrule",
    r"\endhead",
    "",
    r"% --- Footer ---",
    r"\midrule",
    r"\multicolumn{4}{l}{\small Table \thetable\ continued on the next page} \\",
    r"\bottomrule",
    r"\endfoot",
    "",
    r"% --- Last Footer ---",
    r"\bottomrule",
    r"\endlastfoot",
    "",
    r"% --- Data ---"
]

# Sort authors by total publications (descending order)
instituicoes_ordenadas = total_por_instituicao.index.tolist()

for inst in instituicoes_ordenadas:
    total_docs = total_por_instituicao[inst]
    inst_safe = escape_latex(inst)
    
    # Get the topics corresponding to this author in df top_topics
    df_inst = top_topics[top_topics[config.COLUMN_FIRST_AUTHOR_FULL_NAME] == inst]
    df_inst = df_inst.sort_values('count', ascending=False)
    
    for idx, (_, row) in enumerate(df_inst.iterrows()):
        topic_name = row[config.COLUMN_TOPIC_NAME]
        count = row['count']
        
        # Simplify topic name using your function and escape text for LaTeX
        clean_topic = escape_latex(simplify_label(topic_name))
        
        if idx == 0:
            # First row: print Author, Total, Topic, and Document Count
            latex_code.append(f"{inst_safe} & {total_docs} & {clean_topic} & {count} \\\\")
        else:
            # Subsequent rows: hide Author and Total
            latex_code.append(f" & & {clean_topic} & {count} \\\\")
            
    # Spacing at the end of each author block
    latex_code.append(r"\addlinespace")
    latex_code.append("")

latex_code.append(r"\end{longtable}")

# Save result to a .tex file
caminho_tabela_latex = f'{config.RESULTS_FOLDER}/q4_4_2_last_author_and_top_studied_topics.tex'
with open(caminho_tabela_latex, 'w', encoding='utf-8') as f:
    f.write("\n".join(latex_code))

print(f"LaTeX table generated successfully at: {caminho_tabela_latex}")


print("\nScript finished.")
