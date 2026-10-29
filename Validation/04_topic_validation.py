"""
04_topic_validation.py -- item 11: topic model validation metrics
('Topic Model Validation Metrics' and 'Topic Model Stability and Sensitivity
Analysis' in the SI).

Computes for the selected layer (K = 23):
  * c-TF-IDF keywords (top-30 per topic), removal of generic terms (present in
    >= half of the topic lists), and truncation to top-10;
  * coherence C_v, NPMI, and UMass (gensim) and topic diversity;
  * ARI and NMI between adjacent layers of the EVoC hierarchy;
  * optional: cosine silhouette per topic, if an embeddings .npy file is
    provided in EMBEDDINGS_PATH (matching the row order of the main dataset).

Requires: pip install gensim scikit-learn
Runtime: C_v across 128k abstracts can take tens of minutes. For a quick test,
set MAX_DOCS_COHERENCE (e.g., 30000); for the final figures, use None.

Outputs: audit_outputs/04_topic_validation.{txt,json} + 04_topic_keywords.csv.
"""
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from common import (RANDOM_SEED, Report, load_main, require, resolve_columns, save_csv,
                    topic_layer_columns)

MAX_DOCS_COHERENCE = None      # None = all documents
EMBEDDINGS_PATH = None         # e.g., r"/path/to/embeddings_kalm.npy" (optional)
SILHOUETTE_SAMPLE = 65000
TOP_CANDIDATES, TOP_WORDS = 30, 10


def ctfidf_top_words(texts: pd.Series, topics: pd.Series, top_n: int):
    vec = CountVectorizer(stop_words="english", min_df=5, lowercase=True,
                          token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]+\b")
    X = vec.fit_transform(texts)
    vocab = np.array(vec.get_feature_names_out())
    labels = topics.values
    uniq = pd.unique(labels)
    rows = []
    for t in uniq:
        rows.append(np.asarray(X[labels == t].sum(axis=0)).ravel())
    tf = np.vstack(rows).astype(float)
    tf_norm = tf / tf.sum(axis=1, keepdims=True)
    A = tf.sum() / len(uniq)
    idf = np.log(1 + A / (tf.sum(axis=0) + 1e-12))
    score = tf_norm * idf
    words = {t: list(vocab[np.argsort(-score[i])[:top_n]]) for i, t in enumerate(uniq)}
    return words, vec.build_analyzer()


def main():
    rep = Report("04_topic_validation")
    df = load_main()
    cols = resolve_columns(df)
    require(cols, ["abstract", "topic"], "04_topic_validation")
    topic_col = cols["topic"]
    text = (df[cols["title"]].fillna("").astype(str) + ". " if cols["title"] else "") + \
        df[cols["abstract"]].fillna("").astype(str)
    topics = df[topic_col].astype(str)
    K = topics.nunique()
    rep.h(f"LAYER {topic_col}: K = {K}, N = {len(df):,}")

    # ---------------------------------------------------------------- c-TF-IDF
    cand, analyzer = ctfidf_top_words(text, topics, TOP_CANDIDATES)
    freq = pd.Series([w for ws in cand.values() for w in ws]).value_counts()
    generic = set(freq[freq >= K / 2].index)
    rep.p(f"Generic terms removed (in >= {K / 2:.1f} lists): {sorted(generic)}")
    final = {t: [w for w in ws if w not in generic][:TOP_WORDS] for t, ws in cand.items()}
    if any(len(ws) < 3 for ws in final.values()):
        rep.p("[warning] generic term removal left topics with < 3 words; "
              "removal disabled (words = top-10 from c-TF-IDF).")
        final = {t: ws[:TOP_WORDS] for t, ws in cand.items()}
        rep.data["generic_removal_disabled"] = True
    kw = pd.DataFrame({"topic": list(final), "top10": [", ".join(v) for v in final.values()]})
    save_csv(kw, "04_topic_keywords", index=False)
    rep.data["generic_terms_removed"] = sorted(generic)
    rep.data["keywords"] = final

    all_words = [w for ws in final.values() for w in ws]
    diversity = len(set(all_words)) / len(all_words)
    rep.p(f"Topic diversity (unique words / {TOP_WORDS}*K): {diversity:.3f}")
    rep.data["topic_diversity"] = diversity

    # --------------------------------------------------------------- coherence
    try:
        from gensim.corpora import Dictionary
        from gensim.models.coherencemodel import CoherenceModel
        idx = text.index
        if MAX_DOCS_COHERENCE and len(idx) > MAX_DOCS_COHERENCE:
            idx = text.sample(MAX_DOCS_COHERENCE, random_state=RANDOM_SEED).index
            rep.p(f"[coherence computed on a sample of {MAX_DOCS_COHERENCE:,} documents]")
        tokens = [analyzer(t) for t in text.loc[idx]]
        dictionary = Dictionary(tokens)
        topics_words = [ws for ws in final.values() if ws]
        corpus = [dictionary.doc2bow(t) for t in tokens]
        procs = max(1, (os.cpu_count() or 2) - 1)
        res = {}
        for measure in ["c_v", "c_npmi", "u_mass"]:
            cm = CoherenceModel(topics=topics_words, texts=tokens, corpus=corpus,
                                dictionary=dictionary, coherence=measure, processes=procs)
            per_topic = cm.get_coherence_per_topic()
            res[measure] = {"mean": float(np.mean(per_topic)),
                            "per_topic": dict(zip(final.keys(), map(float, per_topic)))}
            rep.p(f"{measure:7s} = {res[measure]['mean']:.3f}")
        rep.data["coherence"] = res
        rep.data["coherence_n_docs"] = len(idx)
    except ImportError:
        rep.p("[warning] gensim not installed: pip install gensim  (coherence not computed)")

    # ------------------------------------------------------- ARI/NMI stability
    layers = topic_layer_columns(df)
    rep.h(f"HIERARCHICAL STABILITY (layers: {layers})")
    stab = []
    for a, b in zip(layers[:-1], layers[1:]):
        la, lb = df[a].astype(str), df[b].astype(str)
        ari = adjusted_rand_score(la, lb)
        nmi = normalized_mutual_info_score(la, lb)
        stab.append({"layer_a": a, "K_a": la.nunique(), "layer_b": b, "K_b": lb.nunique(),
                     "ARI": round(ari, 3), "NMI": round(nmi, 3)})
    if stab:
        rep.table(pd.DataFrame(stab))
    rep.data["hierarchical_stability"] = stab

    # ----------------------------------------------------------- silhouette/topic
    if EMBEDDINGS_PATH:
        from sklearn.metrics import silhouette_samples
        E = np.load(EMBEDDINGS_PATH, mmap_mode="r")
        assert E.shape[0] == len(df), "embeddings and main dataset have different row counts"
        rng = np.random.RandomState(RANDOM_SEED)
        idx = rng.choice(len(df), size=min(SILHOUETTE_SAMPLE, len(df)), replace=False)
        s = silhouette_samples(np.asarray(E[idx]), topics.values[idx], metric="cosine")
        per = pd.Series(s, index=topics.values[idx]).groupby(level=0).mean().sort_values()
        rep.h("COSINE SILHOUETTE PER TOPIC -- sample")
        rep.table(per.to_frame("silhouette"))
        rep.data["silhouette_cosine_mean"] = float(s.mean())
        rep.data["silhouette_cosine_per_topic"] = per.to_dict()
    rep.save()


if __name__ == "__main__":
    main()
