import pandas as pd
from tqdm import tqdm
import numpy as np

tqdm.pandas() 

# Import settings from config.py
import config
import utils

df_data = utils.get_df_data()



# 1. Calculate the thresholds (Medians)
# We use medians because they are robust to outliers (Power Law distribution)
citation_threshold = df_data[config.COLUMN_CITATIONS_PER_YEAR].median()  #.quantile(.75) #
if_threshold = df_data[config.COLUMN_JOURNAL_IF].median()   #.quantile(.75) # 

# 2. Define a function to categorize each row
def get_quadrant(row):
    high_cit = row[config.COLUMN_CITATIONS_PER_YEAR] >= citation_threshold
    high_if = row[config.COLUMN_JOURNAL_IF] >= if_threshold
   
    if high_if and high_cit:
        return "Star (High IF, High Cit)"
    elif not high_if and high_cit:
        return "Hidden Gem (Low IF, High Cit)"
    elif high_if and not high_cit:
        return "Coattail (High IF, Low Cit)"
    else:
        return "Long Tail (Low IF, Low Cit)"

# 3. Apply the function
df_data[config.COLUMN_CATEGORY_GROUP] = df_data.apply(get_quadrant, axis=1)

# Optional: View the counts to see how balanced your groups are
print("\n\n category_group")
print(df_data[config.COLUMN_CATEGORY_GROUP].value_counts())

#######################################
# Create column with JNIF = citations of paper / expected citations (JIF)
df_data[config.COLUMN_JNIF] = df_data.apply(
    lambda row: row[config.COLUMN_CITATIONS_COUNT] / row[config.COLUMN_JOURNAL_IF] 
    if row[config.COLUMN_JOURNAL_IF] > 0 else None,
    axis=1
)


#######################################
# Create column with score_log = w1*log(citations+1) + w2*log(jif+1).
w1 = 0.7  # weight of citations
w2 = 0.3  # weight of JIF
df_data['score_log'] = (
    w1 * np.log(df_data[config.COLUMN_CITATIONS_COUNT] + 1) +
    w2 * np.log(df_data[config.COLUMN_JOURNAL_IF] + 1)
)


#######################################
# Create column with score_sqrt = sqrt(citations * JIF)
df_data['score_sqrt'] = np.sqrt(df_data[config.COLUMN_CITATIONS_COUNT].fillna(0) * df_data[config.COLUMN_JOURNAL_IF].fillna(0))




# === 7. (Optional) save final result ===
print("Saving updated DataFrame...")
utils.save_df_data(df_data)
utils.export_df_to_csv(df_data)

print("✅ Columns added successfully!")
