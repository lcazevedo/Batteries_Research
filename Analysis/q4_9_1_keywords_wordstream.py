import utils
import config

import pandas as pd
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
from matplotlib.ticker import MaxNLocator
import altair as alt

# Load data
df_data = utils.get_df_data()


##########################################################
###
### This first section generates the top 100 keywords for Gemini to classify
###
##########################################################
# # List of exceptions: short words to keep
# # (very common in battery chemistry)
# SHORT_EXCEPTIONS = {'li', 'na', 'zn', 'kv', 'mg', 'ca', 'fe', 'co', 'ni', 
#                     'dft', 'sem', 'tem', 'xps', 'eis', 'cvd'}

# def clean_and_validate_keyword(keyword):
#     """
#     Cleans a single keyword.
#     Converts to lowercase, removes extra spaces.
#     Rejects very short words (except those in the list) or numeric strings.
#     """
#     try:
#         # Convert to string (in case it is another type) and clean
#         kw = str(keyword).lower().strip()
        
#         # Reject if empty after cleaning
#         if not kw:
#             return None
            
#         # Reject if purely numeric
#         if kw.isnumeric():
#             return None
            
#         # Reject if too short (<= 2 letters), 
#         # unless in the exception list
#         if len(kw) <= 2 and kw not in SHORT_EXCEPTIONS:
#             return None
            
#         # Remove unwanted trailing punctuation (e.g., "li-ion.")
#         kw = re.sub(r"[.,!?;]$", "", kw) 
            
#         return kw
#     except:
#         # Catch any conversion error
#         return None

# # --- 2. Load and Process Data ---



# print("Starting keyword cleaning and counting...")

# # Remove rows where keywords are missing (NaN)
# df_keywords = df_data.dropna(subset=[config.COLUMN_AUTHORS_KEYWORDS])

# # Step 1: Split (Explode)
# # Splits string "a; b; c" into a list ["a", " b", " c"]
# # and then explodes the list so each item becomes a row.
# df_exploded = df_keywords[config.COLUMN_AUTHORS_KEYWORDS].str.split(';').explode()

# # Step 2: Clean and Validate
# # Apply cleaning function to each keyword
# df_cleaned = df_exploded.apply(clean_and_validate_keyword)

# # Remove rejected keywords (which returned None)
# df_cleaned = df_cleaned.dropna()

# # Step 3: Count and Filter
# print("Counting finished. Getting Top 100 keywords...")
# top_100_keywords = df_cleaned.value_counts().head(100)

# # --- 3. Display Result ---
# print("\n--- Top 100 Most Frequent Keywords ---")
# print(top_100_keywords)
# print("\n-------------------------------------------------")
# print("\nCopy this list and send it for the next step (classification).")

# # Optional: Save list to a file
# top_100_keywords.to_csv(f"{config.RESULTS_FOLDER}/top_100_keywords.csv")
# print("List also saved to 'top_100_keywords.csv'")


##########################################################
###
### Keywords were passed to Gemini for classification, generating the dictionary below
###
##########################################################

# This chart cannot be generated in Python; an R script was used in Google Colab