import utils
import config
import numpy as np
import pandas as pd
from tqdm import tqdm

tqdm.pandas() 

df_data = utils.get_df_data()


last_year = 2024
months_in_current_year = 9  # query ran on 2025-09-23

current_year_factor = months_in_current_year / 12

# list of topics
topics_cluster4 = df_data[config.COLUMN_TOPIC_NAME].unique().tolist()

# create a dict with papers per topic
dict_cluster4 = {}
for topic in tqdm(topics_cluster4, desc="Creating papers/topic list"):
    papers_topic = df_data[df_data[config.COLUMN_TOPIC_NAME] == topic].copy()

    papers_topic = papers_topic[[config.COLUMN_DOI, config.COLUMN_CITATIONS_COUNT, config.COLUMN_YEAR, config.COLUMN_DOCUMENT_TYPE]].reset_index()

    dict_cluster4[topic] = papers_topic

# First, aggregate global statistics per year for z-score calculation.
# Remove the incomplete final year so it does not skew the calculation.
# Computed both with Review and without Review.
all_docs = pd.concat(dict_cluster4.values(), ignore_index=True)
all_docs[config.COLUMN_DOCUMENT_TYPE] = all_docs[config.COLUMN_DOCUMENT_TYPE].fillna("")
all_docs_wo_review = all_docs[~all_docs[config.COLUMN_DOCUMENT_TYPE].str.contains("Review")]
# Identify the latest year (e.g., 2025)
ultimo_ano = all_docs[config.COLUMN_YEAR].max()
# Filter DataFrame, excluding the latest year
docs_filtrados = all_docs[all_docs[config.COLUMN_YEAR] < ultimo_ano]
docs_filtrados_wo_review = all_docs_wo_review[all_docs_wo_review[config.COLUMN_YEAR] < ultimo_ano]
# Calculate mean and standard deviation per year (excluding the latest year)
year_stats = (
    docs_filtrados.groupby(config.COLUMN_YEAR)[config.COLUMN_CITATIONS_COUNT]
    .agg(["mean", "std"])
    .rename(columns={"mean": "year_mean", "std": "year_std"})
    .reset_index()
)
year_stats_wo_review = (
    docs_filtrados_wo_review.groupby(config.COLUMN_YEAR)[config.COLUMN_CITATIONS_COUNT]
    .agg(["mean", "std"])
    .rename(columns={"mean": "year_mean_wo_review", "std": "year_std_wo_review"})
    .reset_index()
)


# Function to compute the relevance index of a topic
def compute_topic_index(topic_df):

    ######################################################################
    # Original Relevance Index
    df = topic_df.copy()
    df = df[df[config.COLUMN_CITATIONS_COUNT].notna() & df[config.COLUMN_YEAR].notna()]
    
    # Calculate citations per year
    df[config.COLUMN_CITATIONS_PER_YEAR] = df[config.COLUMN_CITATIONS_COUNT] / (last_year - df[config.COLUMN_YEAR] + 1 + current_year_factor) 

    # Merge with yearly statistics
    df = df.merge(year_stats, on=config.COLUMN_YEAR, how="left")
    
    # Yearly Z-score
    df["z_score"] = (df[config.COLUMN_CITATIONS_COUNT] - df["year_mean"]) / df["year_std"]
    df["z_score"] = df["z_score"].fillna(0)

    # Identify top 10% per year
    top_percentile_flags = []
    for year in df[config.COLUMN_YEAR].unique():
        year_notes = all_docs[all_docs[config.COLUMN_YEAR] == year][config.COLUMN_CITATIONS_COUNT]
        if len(year_notes) >= 10:
            threshold = np.percentile(year_notes, 90)
            top_percentile_flags.extend(df[df[config.COLUMN_YEAR] == year][config.COLUMN_CITATIONS_COUNT] >= threshold)
        else:
            top_percentile_flags.extend([False] * sum(df[config.COLUMN_YEAR] == year))
    df["top_10_percent"] = top_percentile_flags[:len(df)]

    # Composite index calculation
    mean_cpy = df[config.COLUMN_CITATIONS_PER_YEAR].mean()
    mean_z = df["z_score"].mean()
    prop_top10 = df["top_10_percent"].mean()
    n_docs = len(df)

    # Composite index (weighted) *** Original ****
    index = (
        0.5 * mean_cpy +
        0.3 * prop_top10 +
        0.2 * np.log1p(n_docs)
    )

    ######################################################################
    # Relevance Index without review
    df = topic_df.copy()
    df[config.COLUMN_DOCUMENT_TYPE] = df[config.COLUMN_DOCUMENT_TYPE].fillna("")
    df = df[~df[config.COLUMN_DOCUMENT_TYPE].str.contains("Review")]
    df = df[df[config.COLUMN_CITATIONS_COUNT].notna() & df[config.COLUMN_YEAR].notna()]
    
    # Calculate citations per year
    df["citations_per_year_wo_review"] = df[config.COLUMN_CITATIONS_COUNT] / (last_year - df[config.COLUMN_YEAR] + 1 + current_year_factor)  

    # Merge with yearly statistics
    df = df.merge(year_stats_wo_review, on=config.COLUMN_YEAR, how="left")
    
    # Yearly Z-score
    df["z_score_wo_review"] = (df[config.COLUMN_CITATIONS_COUNT] - df["year_mean_wo_review"]) / df["year_std_wo_review"]
    df["z_score_wo_review"] = df["z_score_wo_review"].fillna(0)

    # Identify top 10% per year
    top_percentile_flags_wo_review = []
    for year in df[config.COLUMN_YEAR].unique():
        year_notes_wo_review = all_docs_wo_review[all_docs_wo_review[config.COLUMN_YEAR] == year][config.COLUMN_CITATIONS_COUNT]
        if len(year_notes_wo_review) >= 10:
            threshold_wo_review = np.percentile(year_notes_wo_review, 90)
            top_percentile_flags_wo_review.extend(df[df[config.COLUMN_YEAR] == year][config.COLUMN_CITATIONS_COUNT] >= threshold_wo_review)
        else:
            top_percentile_flags_wo_review.extend([False] * sum(df[config.COLUMN_YEAR] == year))
    df["top_10_percent_wo_review"] = top_percentile_flags_wo_review[:len(df)]

    # Composite index calculation
    mean_cpy_wo_review = df["citations_per_year_wo_review"].mean()
    mean_z_wo_review = df["z_score_wo_review"].mean()
    prop_top10_wo_review = df["top_10_percent_wo_review"].mean()
    n_docs_wo_review = len(df)


    # Composite index (weighted) *** without Review ****
    index_wo_review = (
        0.5 * mean_cpy_wo_review +
        0.3 * prop_top10_wo_review +
        0.2 * np.log1p(n_docs_wo_review)
    )



    return {
        "n_documents": n_docs,
        "mean_citations_per_year": mean_cpy,
        "mean_z_score": mean_z,
        "prop_top_10_percent": prop_top10,
        "composite_index": index,

        "n_documents_wo_review": n_docs_wo_review,
        "mean_citations_per_year_wo_review": mean_cpy_wo_review,
        "mean_z_score_wo_review": mean_z_wo_review,
        "prop_top_10_percent_wo_review": prop_top10_wo_review,
        "composite_index_wo_review": index_wo_review,
    }

# Apply to all topics
topic_indices = {}
for topic, df in tqdm(dict_cluster4.items(), desc="Calculating relevance indices per topic"):
    topic_indices[topic] = compute_topic_index(df)


# Convert to DataFrame
topic_index_df = pd.DataFrame.from_dict(topic_indices, orient="index").reset_index()
topic_index_df = topic_index_df.rename(columns={"index": "topic"})
topic_index_df = topic_index_df.sort_values(by="composite_index", ascending=False)

# Ensure numeric values
topic_index_df["mean_citations_per_year"] = topic_index_df["mean_citations_per_year"].astype(float)
topic_index_df["mean_z_score"] = topic_index_df["mean_z_score"].astype(float)
topic_index_df["prop_top_10_percent"] = topic_index_df["prop_top_10_percent"].astype(float)
topic_index_df["composite_index"] = topic_index_df["composite_index"].astype(float)
topic_index_df["mean_citations_per_year_wo_review"] = topic_index_df["mean_citations_per_year_wo_review"].astype(float)
topic_index_df["mean_z_score_wo_review"] = topic_index_df["mean_z_score_wo_review"].astype(float)
topic_index_df["prop_top_10_percent_wo_review"] = topic_index_df["prop_top_10_percent_wo_review"].astype(float)
topic_index_df["composite_index_wo_review"] = topic_index_df["composite_index_wo_review"].astype(float)


def ordinal(n):
    """
    Convert an integer into an ordinal string, such as 1st, 2nd, 3rd, etc.
    """
    if 10 <= n % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1:'st', 2:'nd', 3:'rd'}.get(n % 10, 'th')
    return f"{n}{suffix}"

# Generate ranks
topic_index_df["rank_index"] = (
    topic_index_df["composite_index"]
    .rank(ascending=False, method="min")
    .astype(int)
    .map(ordinal)
)

topic_index_df["rank_index_wo_review"] = (
    topic_index_df["composite_index_wo_review"]
    .rank(ascending=False, method="min")
    .astype(int)
    .map(ordinal)
)

topic_index_df.to_csv(f"{config.DATA_FOLDER}/topic_relevance_index.csv", index=False)
topic_index_df.to_parquet(f"{config.DATA_FOLDER}/topic_relevance_index.parquet", index=False)

# Add columns to df_data
df_data[config.COLUMN_TOPIC_RANK_INDEX] = df_data.progress_apply(lambda row: topic_index_df.loc[topic_index_df["topic"] == row[config.COLUMN_TOPIC_NAME], "composite_index"].values[0], axis=1)
df_data[config.COLUMN_TOPIC_RANK_INDEX_POSITION] = df_data.progress_apply(lambda row: topic_index_df.loc[topic_index_df["topic"] == row[config.COLUMN_TOPIC_NAME], "rank_index"].values[0], axis=1)
df_data[config.COLUMN_TOPIC_RANK_INDEX_WO_REVIEW] = df_data.progress_apply(lambda row: topic_index_df.loc[topic_index_df["topic"] == row[config.COLUMN_TOPIC_NAME], "composite_index_wo_review"].values[0], axis=1)
df_data[config.COLUMN_TOPIC_RANK_INDEX_WO_REVIEW_POSITION] = df_data.progress_apply(lambda row: topic_index_df.loc[topic_index_df["topic"] == row[config.COLUMN_TOPIC_NAME], "rank_index_wo_review"].values[0], axis=1)

print("Saving updated DataFrame...")
utils.save_df_data(df_data)
utils.export_df_to_csv(df_data)

print("Process completed.")
