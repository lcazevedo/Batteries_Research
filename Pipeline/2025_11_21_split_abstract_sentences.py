#####################################
# Sentence segmentation pipeline for abstracts

import pandas as pd
from spacy.lang.en import English
from spacy.language import Language
from tqdm import tqdm
import utils
import config
import multiprocessing

tqdm.pandas() 

df_data = utils.get_df_data()

# ---------------------------------------------------------
# Settings: list of abbreviations (without the period)
# add other abbreviations you find here
# ---------------------------------------------------------
ABBREVS = {"ca", "e.g", "i.e", "fig", "figs", "vs", "et", "al"}  # all in lower-case

# ---------------------------------------------------------
# Build lightweight pipeline: sentencizer + custom component
# (uses simple English(), no need to download en_core_web_sm)
# ---------------------------------------------------------
nlp = English()
nlp.add_pipe("sentencizer")  # creates initial sentence boundaries

@Language.component("custom_sentencizer_after")
def custom_sentencizer_after(doc):
    """
    After the default sentencizer, looks for patterns such as:
      (a) token == 'ca' and next token == '.'  -> merge (do not split after period)
      (b) token == 'ca.' (single token ending in '.') -> also avoid split
    If one of these cases is detected, sets is_sent_start = False on the token that would otherwise start the next sentence.
    """
    for i, token in enumerate(doc):
        lower = token.text.lower()

        candidate_start_idx = None

        # case 1: separate token 'ca' followed by '.'
        if lower in ABBREVS and i + 1 < len(doc) and doc[i + 1].text == ".":
            candidate_start_idx = i + 2

        # case 2: token already includes period, e.g. 'ca.' as a single token
        elif any(lower == f"{a}." for a in ABBREVS):
            candidate_start_idx = i + 1

        if candidate_start_idx is None:
            continue

        # skip extra punctuation that may appear (e.g., '.' followed by ')' etc.)
        j = candidate_start_idx
        while j < len(doc) and doc[j].is_punct:
            j += 1

        if j < len(doc):
            # mark that this token is NOT a sentence start
            doc[j].is_sent_start = False

    return doc

# add the component after sentencizer
nlp.add_pipe("custom_sentencizer_after", after="sentencizer")

# ---------------------------------------------------------
# batch processing with tqdm
# ---------------------------------------------------------

rows = []
abstracts = df_data[config.COLUMN_ABSTRACT].fillna("").tolist()
dois      = df_data[config.COLUMN_DOI].tolist()

# --- optimizations and settings ---
# larger batch size reduces overhead; n_process uses multiple cores when available
PIPE_BATCH = getattr(config, "PIPE_BATCH_SIZE", 256)
PIPE_N_PROCESS = getattr(config, "PIPE_N_PROCESS", max(1, multiprocessing.cpu_count() - 1))

# Filter empty abstracts to avoid unnecessary processing
pairs = [(dois[i].strip() if isinstance(dois[i], str) else dois[i], a)
         for i, a in enumerate(abstracts) if a and a.strip()]

if not pairs:
    print("⚠️ No valid abstracts found. Nothing to process.")
else:
    dois_filtered = [p[0] for p in pairs]
    texts = [p[1] for p in pairs]

    # separate lists (faster than appending dicts)
    doi_phrases = []
    dois_out = []
    phrases = []

    # try using spaCy multiprocessing; fall back if unsupported
    try:
        pipe_iter = nlp.pipe(texts, batch_size=PIPE_BATCH, n_process=PIPE_N_PROCESS)
    except TypeError:
        pipe_iter = nlp.pipe(texts, batch_size=PIPE_BATCH)

    for idx, doc in enumerate(tqdm(pipe_iter, total=len(texts), desc="Processing abstracts")):
        doi = dois_filtered[idx]
        for pos, sent in enumerate(doc.sents):
            phrase = sent.text.strip()
            if phrase:
                doi_phrases.append(f"{doi}_phrase_{pos}")
                dois_out.append(doi)
                phrases.append(phrase)

    # create DataFrame efficiently
    df_phrases = pd.DataFrame({"doi_phrase": doi_phrases, "doi": dois_out, "phrase": phrases})

    # save Parquet (faster/more compact). Optional CSV (kept for compatibility).
    parquet_path = f"{config.DATA_FOLDER}/df_data_phrases.parquet"
    csv_path = f"{config.DATA_FOLDER}/df_data_phrases.csv"

    df_phrases.to_parquet(parquet_path, index=False)
    try:
        df_phrases.to_csv(csv_path, index=False)
        print(f"✅ Files created: '{parquet_path}', '{csv_path}'")
    except Exception:
        print(f"✅ Parquet file created: '{parquet_path}' (failed to generate CSV)")
