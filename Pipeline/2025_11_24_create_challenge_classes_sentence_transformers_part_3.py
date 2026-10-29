####################################################################################################
# Sentence-Transformer clustering (Part 3)
####################################################################################################
# PART 3 merges the verified classes from Part 2 into df_data_phrases

import pandas as pd
import os
import sys
import config  # Import configuration file
import numpy as np

# --- 1. Path Definitions (from config) ---

# File containing LLM-generated class names
FULL_PATH = f"{config.DATA_FOLDER}/challenges_label_classified_full.csv"

# Main target file
DF_DATA_PATH_PARQUET = f"{config.DATA_FOLDER}/df_data_phrases.parquet"

# Final CSV output
DF_DATA_PATH_CSV = f"{config.DATA_FOLDER}/df_data_phrases.csv"

def main():
    print("Starting classification merge script...")
    
    # --- 2. Load Files ---
    try:
        print(f"Loading classification file: {FULL_PATH}")
        df_classification = pd.read_csv(FULL_PATH)
        
        print(f"Loading target Parquet file: {DF_DATA_PATH_PARQUET}")
        df_target = pd.read_parquet(DF_DATA_PATH_PARQUET)
        df_target['challenges_label'] = df_target['challenges_label'].replace('', np.nan)
        df_target['challenges_label'] = df_target['challenges_label'].fillna('No Challenge')

    except FileNotFoundError as e:
        print(f"\n[ERROR] File not found: {e.filename}")
        print("Please verify that the paths in your 'config.py' are correct.")
        sys.exit(1)

    print("Files loaded successfully.")

    # --- 3. Prepare Source Data ---
    
    # Select only required columns and rename 'class_name' to 'challenges_group'
    # Ensure one name per label by dropping duplicates.
    df_source_map = df_classification[['challenges_label', 'class_name']].rename(
        columns={'class_name': 'challenges_group'}
    ).drop_duplicates(subset=['challenges_label'])

    print("Classification data prepared and renamed to 'challenges_group'.")

    # --- 4. Critical Validation ---
    print("Starting validation...")
    
    # Get set of labels from target file (Parquet)
    target_labels = set(df_target['challenges_label'])
    
    # Get set of labels from source file (classification CSV)
    source_labels = set(df_source_map['challenges_label'])
    
    # Check if any label in target is missing from source
    missing_labels = target_labels - source_labels
    
    if len(missing_labels) > 0:
        print("\n" + "="*50)
        print("[FATAL ERROR] Execution aborted.")
        print(f"{len(missing_labels)} 'challenges_label' entries from the Parquet file were not found in the classification file.")
        print("This means the classification file is incomplete.")
        print("\nExamples of missing labels:")
        # Print first 5 missing labels as an example
        for i, label in enumerate(list(missing_labels)):
            if i >= 5:
                break
            print(f"  - {label}")
        print("="*50)
        sys.exit(1) # Abort execution
    
    print("Validation completed successfully. All labels in the Parquet file have a corresponding classification.")

    # --- 5. Data Merge ---
    print("Starting data merge...")
    
    # Perform a left merge, bringing 'challenges_group' into df_target
    df_final = pd.merge(
        df_target,
        df_source_map,
        on='challenges_label',
        how='left' # 'left' ensures all rows from df_target are preserved
    )

    # Safety check (should not happen given the validation above)
    if df_final['challenges_group'].isnull().any():
        print("[WARNING] An unexpected error occurred and some 'challenges_group' entries are null after the merge.")
    else:
        print("Merge completed successfully. Column 'challenges_group' added.")

    # --- 6. Save Final Files ---
    try:
        # 6a. Save to Parquet (overwriting original)
        print(f"Saving updated data to Parquet: {DF_DATA_PATH_PARQUET}")
        df_final.to_parquet(DF_DATA_PATH_PARQUET, index=False)
        
        # 6b. Save to CSV
        print(f"Saving updated data to CSV: {DF_DATA_PATH_CSV}")
        # 'utf-8-sig' is recommended for CSVs to open properly in Excel
        df_final.to_csv(DF_DATA_PATH_CSV, index=False, encoding='utf-8-sig')

    except Exception as e:
        print(f"\n[ERROR] Failed to save final files: {e}")
        sys.exit(1)
        
    print("\n" + "="*50)
    print("✅ Process completed successfully!")
    print("The Parquet file was updated and a new CSV file was generated.")
    print("="*50)

if __name__ == "__main__":
    main()
