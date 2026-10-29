import pandas as pd
import config
import sys

# --- Path Definitions ---
DF_DATA_PATH_PARQUET = f"{config.DATA_FOLDER}/df_data_phrases.parquet"
DF_DATA_PATH_CSV = f"{config.DATA_FOLDER}/df_data_phrases.csv"

def main():
    print(f"Loading Parquet file: {DF_DATA_PATH_PARQUET}")
    try:
        df = pd.read_parquet(DF_DATA_PATH_PARQUET)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {DF_DATA_PATH_PARQUET}")
        sys.exit(1)

    print("Applying grouping fixes...")

    # Fix for Controversies
    if 'controversies_label' in df.columns and 'controversies_group' in df.columns:
        mask_controversy = df['controversies_label'] == "No Controversy"
        df.loc[mask_controversy, 'controversies_group'] = "No Controversy"
        print(f"  -> {mask_controversy.sum()} records updated in 'controversies_group'.")

    # Fix for Challenges
    if 'challenges_label' in df.columns and 'challenges_group' in df.columns:
        mask_challenge = df['challenges_label'] == "No Challenge"
        df.loc[mask_challenge, 'challenges_group'] = "No Challenge"
        print(f"  -> {mask_challenge.sum()} records updated in 'challenges_group'.")

    # Fix for Directions
    if 'directions_label' in df.columns and 'directions_group' in df.columns:
        mask_direction = df['directions_label'] == "No Research Direction"
        df.loc[mask_direction, 'directions_group'] = "No Research Direction"
        print(f"  -> {mask_direction.sum()} records updated in 'directions_group'.")

    print("\nSaving corrected files...")
    try:
        # Save to Parquet
        df.to_parquet(DF_DATA_PATH_PARQUET, index=False)
        print(f"✅ Parquet saved successfully: {DF_DATA_PATH_PARQUET}")
        
        # Save to CSV
        df.to_csv(DF_DATA_PATH_CSV, index=False, encoding='utf-8-sig')
        print(f"✅ CSV saved successfully: {DF_DATA_PATH_CSV}")
    except Exception as e:
        print(f"[ERROR] Failed to save files: {e}")

if __name__ == "__main__":
    main()
