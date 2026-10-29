import os
import sys
import pandas as pd
import config


def get_df_data():
    if os.path.exists(config.DATA_FILE_PATH):
        df = pd.read_parquet(config.DATA_FILE_PATH)
        src = config.DATA_FILE_PATH
    elif os.path.exists(os.path.splitext(config.DATA_FILE_PATH)[0] + ".csv"):
        src = os.path.splitext(config.DATA_FILE_PATH)[0] + ".csv"
        df = pd.read_csv(src)
    elif os.path.exists(config.FINAL_CORPUS_CSV_PATH):
        src = config.FINAL_CORPUS_CSV_PATH
        df = pd.read_csv(src)
    else:
        raise FileNotFoundError(
            f"No corpus file found in {config.DATA_FOLDER}. "
            f"Expected: {config.DATA_FILE_PATH} or {config.FINAL_CORPUS_CSV_PATH}"
        )
    print(f"Data loaded from: {src}")
    print(f"Number of rows in DataFrame: {len(df)}")
    return df


def save_df_data(df: pd.DataFrame):
    df.to_parquet(config.DATA_FILE_PATH, index=False)
    print(f"Data saved to: {config.DATA_FILE_PATH}")


def export_df_to_csv(df: pd.DataFrame):
    csv_path = os.path.splitext(config.DATA_FILE_PATH)[0] + ".csv"
    df.to_csv(csv_path, index=False)
    print(f"Data exported to CSV at: {csv_path}")


def get_topic_index_df():
    if os.path.exists(config.TOPIC_RELEVANCE_INDEX_FILE_PATH):
        src = config.TOPIC_RELEVANCE_INDEX_FILE_PATH
        df = pd.read_parquet(src)
    else:
        src = config.TOPIC_RELEVANCE_INDEX_CSV_PATH
        df = pd.read_csv(src)
    print(f"Data loaded from: {src}")
    return df


def get_df_data_phrases():
    if os.path.exists(config.DATA_PHRASES_FILE_PATH):
        src = config.DATA_PHRASES_FILE_PATH
        df = pd.read_parquet(src)
    else:
        src = config.DATA_PHRASES_CSV_PATH
        df = pd.read_csv(src)
    print(f"Data loaded from: {src}")
    print(f"Number of rows in DataFrame: {len(df)}")
    return df
