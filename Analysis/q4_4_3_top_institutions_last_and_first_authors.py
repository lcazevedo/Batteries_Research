import utils
import config

import pandas as pd
import numpy as np
import textwrap


# Load data
df_data = utils.get_df_data()

# Ignore rows where 'last_institution' is null or an empty/whitespace string.
df_data[config.COLUMN_LAST_INSTITUTION].replace(r'^\s*$', np.nan, regex=True, inplace=True)
df_data.dropna(subset=[config.COLUMN_LAST_INSTITUTION], inplace=True)

# Ensure total citations column is integer for weighted average calculation
df_data[config.COLUMN_CITATIONS_COUNT] = df_data[config.COLUMN_CITATIONS_COUNT].astype(int).fillna(0)

# format author names to title case
# df_data[config.COLUMN_LAST_AUTHOR_FULL_NAME] = df_data[config.COLUMN_LAST_AUTHOR_FULL_NAME].str.title()
# df_data[config.COLUMN_FIRST_AUTHOR_FULL_NAME] = df_data[config.COLUMN_FIRST_AUTHOR_FULL_NAME].str.title()

# Aggregations
agg_inst = df_data.groupby(config.COLUMN_LAST_INSTITUTION).agg(
    total_pubs=(config.COLUMN_LAST_AUTHOR_FULL_NAME, 'count'),
    total_cites=(config.COLUMN_CITATIONS_COUNT, 'sum')
).sort_values('total_pubs', ascending=False)
agg_inst['avg_cites'] = (agg_inst['total_cites'] / agg_inst['total_pubs'])

agg_last_author = df_data.groupby([config.COLUMN_LAST_INSTITUTION, config.COLUMN_LAST_AUTHOR_FULL_NAME]).agg(
    total_pubs=(config.COLUMN_FIRST_AUTHOR_FULL_NAME, 'count'),
    total_cites=(config.COLUMN_CITATIONS_COUNT, 'sum')
).sort_values([config.COLUMN_LAST_INSTITUTION, 'total_pubs'], ascending=[True, False])
agg_last_author['avg_cites'] = (agg_last_author['total_cites'] / agg_last_author['total_pubs'])

agg_first_author = df_data.groupby([config.COLUMN_LAST_INSTITUTION, config.COLUMN_LAST_AUTHOR_FULL_NAME, config.COLUMN_FIRST_AUTHOR_FULL_NAME]).agg(
    total_pubs=(config.COLUMN_YEAR, 'count'),
    total_cites=(config.COLUMN_CITATIONS_COUNT, 'sum')
).sort_values([config.COLUMN_LAST_INSTITUTION, config.COLUMN_LAST_AUTHOR_FULL_NAME, 'total_pubs'], ascending=[True, True, False])
agg_first_author['avg_cites'] = (agg_first_author['total_cites'] / agg_first_author['total_pubs'])


# --- Step 3: Helper Function for Text Wrapping ---
def wrap_text(text, width=25):
    """Wrap text and format it for LaTeX with \shortstack."""
    if len(text) <= width:
        return text
    wrapped_lines = textwrap.wrap(text, width, break_long_words=True, break_on_hyphens=False)
    latex_lines = r' \\ '.join(wrapped_lines)
    return f"\\shortstack[l]{{{latex_lines}}}"

# --- Step 4: Build LaTeX String ---

top_15_institutions = agg_inst.head(10).index

latex_string = []

# Add commands to compact the table
latex_string.append(r"\begingroup")
latex_string.append(r"\setlength{\tabcolsep}{3pt} % Reduce space between columns")
latex_string.append(r"\renewcommand{\arraystretch}{0.6} % Reduce row height")

# LaTeX Table Header
# **CHANGE 1: The last two columns are now 'r' for right alignment.**
latex_string.append(r"\begin{longtable}{p{3.8cm} p{3.8cm} p{4.2cm} r r}")
latex_string.append(r"\captionsetup{justification=centering, skip=0cm}")
latex_string.append(r"\caption{Top 10 Institutions With Breakdown by\\Top 3 Last Authors and Top 5 First Authors.}\\")
latex_string.append(r"\label{tab:q4_4_3_top_institutions_last_and_first_authors} \\")
latex_string.append(r"\hline")
# **CHANGE 2 (Recommended): Numeric column headers right-aligned.**
latex_string.append(r"Institution & Last Author & First Author &\shortstack[r]{Total\\Pubs} & \shortstack[r]{Avg.\\Cites} \\")
latex_string.append(r"\hline")
latex_string.append(r"\endfirsthead")
latex_string.append(r"\multicolumn{5}{c}%")
latex_string.append(r"{{\tablename\ \thetable{} -- continued from previous page}} \\")
latex_string.append(r"\hline")
# **CHANGE 2 (Recommended): Repeated for the next page.**
latex_string.append(r"Institution & Last Author & First Author &\shortstack[r]{Total\\Pubs} & \shortstack[r]{Avg.\\Cites} \\")
latex_string.append(r"\hline")
latex_string.append(r"\endhead")
latex_string.append(r"\hline \multicolumn{5}{r}{{Continued on next page}} \\ \hline")
latex_string.append(r"\endfoot")
latex_string.append(r"\hline")
latex_string.append(r"\endlastfoot")

# Main loop to build table rows
for inst_name in top_15_institutions:
    inst_data = agg_inst.loc[inst_name]
    
    # **ADJUSTED**: Institution row now uses \multicolumn to span 3 columns
    # This avoids wrapping the institution name.
    inst_name_adjusted = inst_name.replace('&', r'\&')  # Escape '&' character for LaTeX
    latex_string.append(
        f"\\multicolumn{{3}}{{l}}{{{inst_name_adjusted}}} & "
        f"{int(inst_data['total_pubs'])} & "
        f"{inst_data['avg_cites']:.1f} \\\\"
    )
    
    # Processing Last Authors for this institution
    last_authors_in_inst = agg_last_author.loc[inst_name]
    
    num_last_authors = len(last_authors_in_inst)
    if num_last_authors <= 4:
        authors_to_show = last_authors_in_inst
        others_authors = pd.DataFrame()
    else:
        authors_to_show = last_authors_in_inst.head(3)
        others_authors = last_authors_in_inst.iloc[3:]

    for la_name, la_data in authors_to_show.iterrows():
        # Last Author row
        latex_string.append(
            f"& {wrap_text(la_name)} & & "
            f"{int(la_data['total_pubs'])} & "
            f"{la_data['avg_cites']:.1f} \\\\"
        )
        
        # Processing First Authors for this Last Author
        first_authors_in_la = agg_first_author.loc[(inst_name, la_name)]
        
        num_first_authors = len(first_authors_in_la)
        if num_first_authors <= 6:
            first_authors_to_show = first_authors_in_la
            others_first_authors = pd.DataFrame()
        else:
            first_authors_to_show = first_authors_in_la.head(5)
            others_first_authors = first_authors_in_la.iloc[5:]
            
        for fa_name, fa_data in first_authors_to_show.iterrows():
            # First Author row
            latex_string.append(
                f"& & {wrap_text(fa_name)} & "
                f"{int(fa_data['total_pubs'])} & "
                f"{fa_data['avg_cites']:.1f} \\\\"
            )
            
        # "Others" row for First Authors (if any)
        if not others_first_authors.empty:
            others_pubs = others_first_authors['total_pubs'].sum()
            others_cites_sum = others_first_authors['total_cites'].sum()
            others_avg_cites = (others_cites_sum / others_pubs) if others_pubs > 0 else 0
            latex_string.append(
                f"& & Others & "
                f"{int(others_pubs)} & "
                f"{others_avg_cites:.1f} \\\\"
            )
            
        # Separator line for Last Author
        latex_string.append(r"\cline{2-5}")
        
    # "Others" row for Last Authors (if any)
    if not others_authors.empty:
        others_pubs = others_authors['total_pubs'].sum()
        others_cites_sum = others_authors['total_cites'].sum()
        others_avg_cites = (others_cites_sum / others_pubs) if others_pubs > 0 else 0
        latex_string.append(
            f"& Others & & "
            f"{int(others_pubs)} & "
            f"{others_avg_cites:.1f} \\\\"
        )
        latex_string.append(r"\cline{2-5}")
        
    # Separator line for Institution
    latex_string.append(r"\hline")

latex_string.append(r"\end{longtable}")
latex_string.append(r"\endgroup")

# --- Step 5: Save Final Result ---
final_latex_code = "\n".join(latex_string)
# Save string to a .tex file
file_name_latex = f'{config.RESULTS_FOLDER}/q4_4_3_top_institutions_last_and_first_authors.tex'
try:
    with open(file_name_latex, 'w', encoding='utf-8') as f:
        f.write(final_latex_code)
    print(f"LaTeX table saved successfully as: {file_name_latex}")
except Exception as e:
    print(f"Error saving LaTeX table: {e}")


print("\n--- GENERATED LATEX CODE (WITH FINAL ADJUSTMENTS) ---\n")
