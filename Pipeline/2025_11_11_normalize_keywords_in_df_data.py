import utils
import config

import pandas as pd
from string_grouper import match_strings
from string_grouper import group_similar_strings
import re
from tqdm import tqdm

tqdm.pandas() 

# === 1. Load df_data (main dataset) ===
df_data = utils.get_df_data()

# --- 2. Extract All Unique Keywords ---

# a. Get the column, drop null values (NaN), and split strings by ';'
all_keywords_series = df_data[config.COLUMN_AUTHORS_KEYWORDS].dropna().str.split(';')

# b. Explode the series so each keyword gets its own row
all_keywords_exploded = all_keywords_series.explode()

# c. Apply BASIC cleaning: strip whitespace and convert to lowercase
#    This is crucial for clustering to work properly.
basic_normalized_keywords = all_keywords_exploded.str.strip().str.lower()

# d. Get a list of unique keywords (as a Pandas Series)
unique_keywords = pd.Series(basic_normalized_keywords.unique()).dropna()

print(f"Found {len(unique_keywords)} unique keywords to cluster.")


# --- 3. Cluster Similar Words ---

print("Starting string grouping (this may take a moment)...")

# 'group_similar_strings' is the right function for this case.
# It returns a pandas Series where:
#    - INDEX is the original string (e.g., 'catalysts')
#    - VALUE is the "master_string" (e.g., 'catalyst')
grouped_series = group_similar_strings(
    strings_to_group=unique_keywords, 
    min_similarity=0.8 # Adjust this value if needed
)

# 'grouped_matches' has columns: [original_string, master_string, similarity]
# The 'master_string' is the canonical representative word for each group.


# --- 4. Create the Replacement Map ---

# 'grouped_series' is already the map we need.
# We only need to convert it to a dictionary.
# .to_dict() produces: {'variation': 'master_string'}
replacement_map_auto = grouped_series.to_dict()
print("Automatic map created.")



# --- 5. Apply Map Back to the DataFrame ---

# **** STEP 2: MANUAL MAP ****
# Here we fix abbreviations/synonyms that
# string_grouper did not catch.
# Important: Use ALREADY CLEANED (lowercase) words
manual_abbreviation_map = {
    'li-ion battery': 'lithium-ion battery',
    'li ion battery': 'lithium-ion battery',
    'li-ion batteries': 'lithium-ion battery',
    'li -ion batteries': 'lithium-ion battery',
    'li -ion battery': 'lithium-ion battery',
    'li ion batteries': 'lithium-ion battery',
    'lithium -ion battery': 'lithium-ion battery',
    'lithium-ion batteries': 'lithium-ion battery',
    'lithium ion batteries': 'lithium-ion battery',
    'lithium ion battery': 'lithium-ion battery',
    'lithium -ion batteries': 'lithium-ion battery',
    'batteries li-ion': 'lithium-ion battery',
    'batteries-li-ion': 'lithium-ion battery',
    'batteries - li-ion': 'lithium-ion battery',
    'lithium-ion batteries (libs)': 'lithium-ion battery',
    'lithium-ion battery (lib)': 'lithium-ion battery',
    'batteries - lithium': 'lithium-ion battery',
    'lithium batteries': 'lithium-ion battery',
    'lithium battery': 'lithium-ion battery',
    'cathode materials': 'cathode material',
    '2d materials': '2d material',
    'all-solid-state batteries': 'all-solid-state battery',
    'aluminum-ion batteries': 'aluminum-ion battery',
    'aqueous li-ion batteries': 'lithium-ion battery',
    'aqueous lithium ion batteries': 'lithium-ion battery',
    'aqueous lithium ion battery': 'lithium-ion battery',
    'aqueous lithium-ion batteries': 'lithium-ion battery',
    'aqueous zinc -ion batteries': 'zinc-ion battery',
    'aqueous zinc -ion battery': 'zinc-ion battery',
    'aqueous zinc ion batteries': 'zinc-ion battery',
    'aqueous zinc ion battery': 'zinc-ion battery',
    'aqueous zinc-ion batteries': 'zinc-ion battery',
    'aqueous zinc-ion batteries (azibs)': 'zinc-ion battery',
    'aqueous zn ion batteries': 'zinc-ion battery',
    'aqueous zn ion battery': 'zinc-ion battery',
    'aqueous zn-ion batteries': 'zinc-ion battery',
    'aqueous zn-ion battery': 'zinc-ion battery',
    'aqueous zn batteries': 'zinc-ion battery',
    'aqueous zinc batteries': 'zinc-ion battery',
    'batteries': 'battery',
    'battery management system (bms)': 'battery management system',
    'battery management systems': 'battery management system',
    'battery management systems (bmss)': 'battery management system',
    'additives': 'additive',
    'anode materials': 'anode material',
    'aqueous batteries': 'aqueous battery',
    'binders': 'binder',
    'carbon nanofibers': 'carbon nanofiber',
    'carbon nanotubes': 'carbon nanotube',
    'cathodes': 'cathode',
    'composites': 'composite',
    'deep eutectic solvents': 'deep eutectic solvent',
    'dendrites': 'dendrite',
    'dft': 'density functional theory',
    'dft calculations': 'density functional theory',
    'dft calculation': "density functional theory",
    'density functional theory calculations': 'density functional theory',
    'density functional theory (dft)': 'density functional theory',
    'density functional calculations': 'density functional theory',
    'density functional theory calculation': 'density functional theory',
    'electric vehicles': 'electric vehicle',
    'electrochemical performances': 'electrochemical performance',
    'electrochemical properties': 'electrochemical property',
    'electrode materials': 'electrode material',
    'electrolyte additives': 'electrolyte additive',
    'fast charging': 'fast-charging',
    'first-principles calculations': 'first-principles calculation',
    'full cells': 'full cell',
    'gel polymer electrolytes': 'gel polymer electrolyte',
    'heterostructures': 'heterostructure',
    'interfaces': 'interface',
    'ionic liquids': 'ionic liquid',
    'lithium metal batteries': 'lithium metal battery',
    'lithium-sulfur batteries': 'lithium-sulfur battery',
    'metal-organic frameworks': 'metal-organic framework',
    'mxenes': 'mxene',
    'nanocomposites': 'nanocomposite',
    'nanoparticles': 'nanoparticle',
    'nanostructures': 'nanostructure',
    'potassium ion batteries': 'potassium-ion battery',
    'potassium ion battery': 'potassium-ion battery',
    'potassium-ion batteries': 'potassium-ion battery',
    'silicon anodes': 'silicon anode',
    'sodium -ion batteries': 'sodium-ion battery',
    'sodium ion batteries': 'sodium-ion battery',
    'sodium ion battery': 'sodium-ion battery',
    'sodium-ion batteries': 'sodium-ion battery',
    'sodium-ion batteries': 'sodium-ion battery',
    'solid electrolyte interphase (sei)': 'solid electrolyte interphase',
    'solid-state batteries': 'solid-state battery',
    'solid-state electrolytes': 'solid-state electrolyte',
    'supercapacitors': 'supercapacitor',
    'zinc ion batteries': 'zinc-ion battery',
    'zinc ion battery': 'zinc-ion battery',
    'zinc-ion batteries': 'zinc-ion battery',
    'zn-ion batteries': 'zinc-ion battery',
    'zn-ion battery': 'zinc-ion battery',
    'zn anodes': 'zn anode',
    'zinc metal anode': 'zn metal anode',
    'na-ion batteries': 'sodium-ion battery',
    'na-ion batteiies': 'sodium-ion battery',
    'na-ion batteries (nibs)': 'sodium-ion battery',
    'na-ion batteries (sibs)': 'sodium-ion battery',
    'na-ion batteries cathode material': 'sodium-ion battery',
    'na-ion battery': 'sodium-ion battery',
    'na-ion battery (sib)': 'sodium-ion battery',
    'na-ion battery 3': 'sodium-ion battery',
    'anodes': 'anode',
    'carbon nanotuber': 'carbon nanotube',
    'electrolytes': 'electrolyte',


    # Add other abbreviations you find here
}

# Helper function (using 2 maps)
def normalize_keyword_string(text, auto_map, manual_map):
    if pd.isna(text):
        return text
    
    words = text.split(';')
    normalized_words = []
    
    for word in words:
        # 1. Apply basic cleaning
        cleaned_word = word.strip().lower()
        
        if not cleaned_word:
            continue
            
        # 2. (Step 1) Apply automatic map (typos/plurals)
        master_word = auto_map.get(cleaned_word, cleaned_word)
        
        # 3. (Step 2) Apply manual map (abbreviations)
        final_word = manual_map.get(master_word, master_word)
        
        normalized_words.append(final_word)
    
    return normalized_words #';'.join(normalized_words)



# Apply the function to each row of the keywords column
print("Applying map to DataFrame to create the new column...")


df_data[config.COLUMN_AUTHORS_KEYWORDS_NORM_LIST] = df_data[config.COLUMN_AUTHORS_KEYWORDS].progress_apply(
    lambda text: normalize_keyword_string(text, replacement_map_auto, manual_abbreviation_map)
)


print("Saving updated DataFrame...")
utils.save_df_data(df_data)
utils.export_df_to_csv(df_data)

print("Process completed.")
