import utils
import config

import pandas as pd

from tqdm import tqdm

# === 1. Load df_data (main dataset) ===
df_data = utils.get_df_data()

# === 2. Drop old column "topic_cluster_4" if it exists ===
if "topic_cluster_4" in df_data.columns:
    df_data = df_data.drop(columns=["topic_cluster_4"])

# === 3. Read topics spreadsheet ===
excel_path = f"{config.RESULTS_FOLDER}/papers_pred_cluster1.xlsx"

# Read the "Topics" sheet (only to retrieve topic names)
topics_df = pd.read_excel(excel_path, sheet_name="Topics")

# Extract topic names (part after "_")
topic_names = [t.split("_", 1)[-1] if "_" in t else t for t in topics_df["Name"]]
subscripts = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
topic_full_names = [t.translate(subscripts) for t in topics_df["Name"]]

# === 4. Build a global dictionary of DOI -> abbreviated topic ===
doi_to_topic = {}

# For each sheet (topic) in the Excel file, load and register DOIs
for i, topic in tqdm(enumerate(topic_full_names), total=len(topic_full_names)):
    try:
        sheet_name=topic[:31].replace(":", "_").replace("/", "_")
        df_topic = pd.read_excel(excel_path, sheet_name=sheet_name)
        if "DOI" not in df_topic.columns:
            raise ValueError(f"Sheet '{topic}' does not contain column 'doi'. Please check the Excel file.")
            #continue  # skip sheets without "doi" column
        
        # Map DOIs to abbreviated topic name
        for doi in df_topic["DOI"].dropna():
            doi_to_topic[doi] = topic_names[i]
    except Exception as e:
        raise ValueError(f"Error processing sheet '{sheet_name}': {e}")

# === 5. Create the new column in df_data ===
df_data[config.COLUMN_TOPIC_NAME] = df_data[config.COLUMN_DOI].map(doi_to_topic)

# === 6. (Optional) Verify results ===
#print(df_data["topic_cluster_4"].value_counts(dropna=False).head())

print("Saving updated DataFrame...")
utils.save_df_data(df_data)
utils.export_df_to_csv(df_data)
