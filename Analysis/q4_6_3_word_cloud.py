import utils
import config

import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re # For more robust cleaning

# Load data
df_data = utils.get_df_data()

# --- Configuration ---
MAX_WORDS = 100
MAX_WORDS_LATEX = 30

# --- 1. Keyword Processing (Optimized) ---
all_keywords_list = []
for keyword_list in df_data[config.COLUMN_AUTHORS_KEYWORDS_NORM_LIST].dropna():
    all_keywords_list.extend(keyword_list)

#pd.DataFrame(all_keywords_list).to_csv(f"{config.RESULTS_FOLDER}/q4_6_3_word_counts.csv")

# --- 2. Count Word Frequency ---
word_counts = Counter(all_keywords_list)


# --- 2.5. Export LaTeX Table (NEW BLOCK) ---
print("Generating LaTeX table...")

# a. Convert Counter to DataFrame
df_counts = pd.DataFrame(word_counts.items(), columns=['Keyword', 'Frequency'])

# b. Sort and take Top N words (to match the Word Cloud)
df_top_counts = df_counts.sort_values(by='Frequency', ascending=False).head(MAX_WORDS_LATEX)

# c. Define output filename
output_latex_file = f"{config.RESULTS_FOLDER}/q4_6_3_word_counts.tex"

# 1. Calculate split point (round up)
n_rows = len(df_top_counts)
n_split = (n_rows + 1) // 2

# 2. Create left half (first N rows)
df_left = df_top_counts.iloc[:n_split].reset_index(drop=True)

# 3. Create right half (remaining rows)
df_right = df_top_counts.iloc[n_split:].reset_index(drop=True)

# 4. Combine both DataFrames side by side (axis=1)
df_latex_combined = pd.concat([df_left, df_right], axis=1)

# 5. Fill with empty string in case of an odd number of items
#    (The last row of the right column would otherwise be 'NaN')
df_latex_combined = df_latex_combined.fillna('')


# e. (MODIFIED) Generate LaTeX string
#    Now using 'df_latex_combined'
latex_table_string = df_latex_combined.to_latex(
    index=False,
    # We need 4 headers now
    header=['Keyword', 'Freq.', 'Keyword', 'Freq.'],
    caption=f'Top {MAX_WORDS_LATEX} keywords.',
    label='tab:word_counts',
    escape=True,
    # (Optional, but recommended) Define column alignment
    # 'l' (left), 'r' (right). The '|' adds a vertical bar
    # to visually separate the two halves.
    column_format='l r | l r' 
)

# f. (Originally 'e.') Save string to .tex file
try:
    with open(output_latex_file, 'w', encoding='utf-8') as f:
        f.write(latex_table_string)
    print(f"LaTeX table successfully saved as '{output_latex_file}'")
except Exception as e:
    print(f"Error saving LaTeX file: {e}")

# --- END OF NEW BLOCK ---


# --- 3. Generate Word Cloud ---
print("Generating word cloud...")
wc = WordCloud(
    background_color="white",
    max_words=MAX_WORDS,
    width=1600,
    height=800,
    colormap='plasma',  #'cividis',
    contour_width=1,
    contour_color='steelblue',
    random_state=42
)
wc.generate_from_frequencies(word_counts)

# --- 4. Save and Show Image ---
output_filename = f"{config.RESULTS_FOLDER}/q4_6_3_word_cloud.pdf"
wc.to_file(output_filename)
print(f"Image successfully saved as '{output_filename}'")

# --- 5. Show image (Optional) ---
#plt.figure(figsize=(15, 7))
#plt.imshow(wc, interpolation='bilinear')
#plt.axis("off")
#plt.show()