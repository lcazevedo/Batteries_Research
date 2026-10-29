"""
05_intent_stats.py -- item 12 (part 1): recomputes all statistics for
challenges, directions, and controversies after 'No X' filtering.

For each vector:
  * total sentences, how many are 'No X' (by *_group column and by raw *_label),
    extraction errors, and categorized sentences;
  * most frequent raw labels -- to check for 'No X' variants that escaped exact
    matching ('No challenge.', 'No Challenge Identified', ...);
  * category ranking (sentences and %), number of distinct categories;
  * annual evolution: absolute mentions and % of documents mentioning the category;
  * countries x categories and topics x categories (row proportions, used for
    Ward clustermaps);
  * 5-year period rankings and momentum (2006-2015 vs 2016-2025).

All tables are saved to CSV (audit_outputs/05_<vector>_*.csv) for regenerating
the q4_12_* figures.

Adjust below if column names differ. The join key phrases -> document is
detected automatically; if needed, set JOIN_KEY_PHRASES and JOIN_KEY_MAIN.
"""
import re

import numpy as np
import pandas as pd

from common import (COLUMN_CANDIDATES, Report, load_main, load_phrases, resolve_columns,
                    save_csv)

VECTORS = {
    "challenges": ("challenges_label", "challenges_group"),
    "directions": ("directions_label", "directions_group"),
    "controversies": ("controversies_label", "controversies_group"),
}
JOIN_KEY_PHRASES = None   # e.g., "UT (Unique WOS ID)"
JOIN_KEY_MAIN = None      # e.g., "UT (Unique WOS ID)"
YEARS = range(2014, 2026)
TOP_COUNTRIES = 20

NONE_RE = re.compile(r"^\s*[\"']?\s*no\s+(research\s+)?(challenge|direction|controvers|research\s+direction)",
                     re.IGNORECASE)
ERROR_RE = re.compile(r"extraction\s+error|llm_(parse|request)_error", re.IGNORECASE)


def classify_status(series: pd.Series) -> pd.Series:
    s = series.astype("string")
    empty = s.isna() | (s.str.strip() == "") | s.str.lower().isin(["nan", "none"])
    status = pd.Series("category", index=series.index)
    status[s.fillna("").str.match(NONE_RE)] = "none"
    status[s.fillna("").str.contains(ERROR_RE)] = "error"
    status[empty] = "empty"
    return status


def find_join(ph: pd.DataFrame, df: pd.DataFrame, cols: dict):
    if JOIN_KEY_PHRASES and JOIN_KEY_MAIN:
        return JOIN_KEY_PHRASES, JOIN_KEY_MAIN
    if "doi" in ph.columns and cols.get("doi"):          # current structure: phrases.doi -> main.DOI
        return "doi", cols["doi"]
    prefs = ([cols.get("doc_id")] if cols.get("doc_id") else []) + COLUMN_CANDIDATES["doc_id"][1]
    common = set(ph.columns) & set(df.columns)
    for c in prefs:
        if c and c in common:
            return c, c
    return None, None


def norm_key(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.lower()


BOILERPLATE_RE = re.compile(r"copyright|all rights reserved|\(c\)\s*\d{4}|published by elsevier|"
                            r"john wiley|this article is protected", re.IGNORECASE)


def rank_by_period(m: pd.DataFrame, start=2006, end=2025, step=5) -> pd.DataFrame:
    out = {}
    for a in range(start, end + 1, step):
        b = a + step - 1
        sub = m[(m.year >= a) & (m.year <= b)]
        out[f"{a}-{b}"] = sub["group"].value_counts().rank(ascending=False, method="min").astype(int)
    return pd.DataFrame(out)


def momentum(m: pd.DataFrame) -> pd.DataFrame:
    p1 = m[(m.year >= 2006) & (m.year <= 2015)]["group"].value_counts(normalize=True) * 100
    p2 = m[(m.year >= 2016) & (m.year <= 2025)]["group"].value_counts(normalize=True) * 100
    tot = m[(m.year >= 2006) & (m.year <= 2025)]["group"].value_counts()
    t = pd.DataFrame({"total_presence_2006_2025": tot, "share_2006_2015_pct": p1,
                      "share_2016_2025_pct": p2}).fillna(0)
    t["momentum_pp"] = t["share_2016_2025_pct"] - t["share_2006_2015_pct"]
    return t.sort_values("total_presence_2006_2025", ascending=False)


def main():
    rep = Report("05_intent_stats")
    ph = load_phrases()
    df = load_main()
    cols = resolve_columns(df)
    kp, km = find_join(ph, df, cols)
    rep.h("JOIN PHRASES -> DOCUMENTS")
    meta = None
    if kp:
        rep.p(f"key: phrases[{kp}] -> main[{km}]")
        keep = {km: "key"}
        for k in ["year", "country", "topic"]:
            if cols.get(k):
                keep[cols[k]] = k
        meta = df[list(keep)].rename(columns=keep)
        meta["year"] = pd.to_numeric(meta.get("year"), errors="coerce")
        meta["key"] = norm_key(meta["key"])
        n_nokey = int(meta["key"].isna().sum())
        n_dup = int(meta["key"].dropna().duplicated().sum())
        rep.p(f"documents without {km}: {n_nokey:,} (cannot link phrases) | duplicated {km}: {n_dup:,}")
        rep.data["docs_without_key"], rep.data["docs_duplicated_key"] = n_nokey, n_dup
        meta = meta.dropna(subset=["key"]).drop_duplicates("key")
        ph["key"] = norm_key(ph[kp])
        ph = ph.merge(meta, on="key", how="left")
        rep.p(f"phrases without matching document in main dataset: {int(ph['year'].isna().sum()):,}")
        n_docs_ph = ph["key"].nunique()
        rep.p(f"documents with at least one phrase: {n_docs_ph:,} of {len(df):,}")
        rep.p(f"phrases per document: mean {len(ph) / n_docs_ph:.2f}, "
              f"median {ph.groupby('key').size().median():.0f}")
        rep.data["docs_with_phrases"] = int(n_docs_ph)
        docs_per_year_all = meta["year"].value_counts()
    else:
        rep.p("[warning] join key not found: only global statistics will be computed. "
              "Populate JOIN_KEY_PHRASES/JOIN_KEY_MAIN.")
        docs_per_year_all = None
    rep.p(f"TOTAL SENTENCES: {len(ph):,}")
    rep.data["n_sentences"] = len(ph)

    # ---------------------------------------------- 'No X' diagnostics
    rep.h("GENERAL 'No X' DIAGNOSTICS")
    stat = {v: classify_status(ph[g]) for v, (_, g) in VECTORS.items() if g in ph.columns}
    for v, st in stat.items():
        rep.p(f"  {v:14s} sentences with category: {100 * (st == 'category').mean():5.1f}% | "
              f"'No X': {100 * (st == 'none').mean():5.1f}%")
    if len(stat) == 3:
        allcat = np.logical_and.reduce([st == "category" for st in stat.values()])
        nonecat = np.logical_and.reduce([st == "none" for st in stat.values()])
        rep.p(f"  sentences categorized across ALL THREE vectors simultaneously: {100 * allcat.mean():.1f}%")
        rep.p(f"  sentences 'No X' across all three vectors: {100 * nonecat.mean():.1f}%")
        rep.data["pct_sentences_categorized_in_all_three"] = float(100 * allcat.mean())
        rep.data["pct_sentences_none_in_all_three"] = float(100 * nonecat.mean())
    text_col = "phrase" if "phrase" in ph.columns else None
    if text_col:
        bp = ph[text_col].astype(str).str.contains(BOILERPLATE_RE)
        rep.p(f"\n  copyright/publisher boilerplate sentences (cannot contain research intent): {int(bp.sum()):,}")
        for v, st in stat.items():
            rate = (st[bp] == "category").mean() if bp.any() else float("nan")
            rep.p(f"    {v:14s} assigned a category (definite false positive): {100 * rate:.1f}%")
            rep.data[f"boilerplate_categorized_pct_{v}"] = float(100 * rate) if bp.any() else None
        # sentence position in abstract (initial sentences are typically background/objective)
        if "doi_phrase" in ph.columns:
            pos = pd.to_numeric(ph["doi_phrase"].astype(str).str.extract(r"_phrase_(\d+)$")[0], errors="coerce")
            for v, st in stat.items():
                t = pd.crosstab(pos.clip(upper=8), st, normalize="index")
                if "category" in t:
                    rep.p(f"  {v}: % categorized by sentence position (0..8+): "
                          f"{(100 * t['category']).round(1).to_dict()}")

    for vec, (lab_col, grp_col) in VECTORS.items():
        if grp_col not in ph.columns:
            rep.p(f"\n[warning] column {grp_col} not found; vector {vec} skipped")
            continue
        R = {}
        rep.h(f"VECTOR: {vec.upper()}")
        st_grp = classify_status(ph[grp_col])
        R["status_by_group_column"] = st_grp.value_counts().to_dict()
        rep.p(f"Status by {grp_col} column: {R['status_by_group_column']}")
        if lab_col in ph.columns:
            st_lab = classify_status(ph[lab_col])
            R["status_by_raw_label"] = st_lab.value_counts().to_dict()
            rep.p(f"Status by raw label {lab_col}: {R['status_by_raw_label']}")
            ct = pd.crosstab(st_lab.rename("label"), st_grp.rename("group"))
            rep.p("Crosstab raw label vs. group:")
            rep.table(ct, floatfmt="{:,.0f}")
            R["crosstab_label_vs_group"] = ct.to_dict()
            top_raw = ph[lab_col].astype(str).value_counts().head(30)
            rep.p("30 most frequent raw labels (check 'No X' variants):")
            rep.table(top_raw.to_frame("n"))
            R["top_raw_labels"] = top_raw.to_dict()
            R["n_distinct_raw_labels"] = int(ph[lab_col].nunique())
            rep.p(f"distinct raw labels: {R['n_distinct_raw_labels']:,}")
            # 'No X' in raw label but assigned a category group = leakage
            leak = (st_lab == "none") & (st_grp == "category")
            R["none_label_but_categorized"] = int(leak.sum())
            if leak.any():
                rep.p(f"!! {int(leak.sum()):,} sentences with 'No X' raw label (variant) but assigned "
                      f"a category. Groups they fall into: "
                      f"{ph.loc[leak, grp_col].value_counts().head(5).to_dict()}")

        cat_mask = st_grp == "category"
        if lab_col in ph.columns:
            cat_mask &= classify_status(ph[lab_col]) == "category"   # excludes 'No X' variants
        m = ph.loc[cat_mask, [grp_col] + [c for c in ["key", "year", "country", "topic"] if c in ph]]
        m = m.rename(columns={grp_col: "group"})
        n_cat = len(m)
        R["n_categorized_sentences"] = n_cat
        R["pct_categorized_of_all"] = 100 * n_cat / len(ph)
        rep.p(f"\nCategorized sentences: {n_cat:,} ({R['pct_categorized_of_all']:.2f}% of {len(ph):,})")

        rank = m["group"].value_counts()
        rank_df = pd.DataFrame({"sentences": rank, "pct": 100 * rank / n_cat})
        if "key" in m:
            rank_df["documents"] = m.groupby("group")["key"].nunique()
        rep.p(f"Distinct categories: {rank.size}")
        rep.table(rank_df, floatfmt="{:,.2f}")
        R["ranking"] = rank_df.reset_index().to_dict(orient="records")
        save_csv(rank_df, f"05_{vec}_ranking")

        if "year" in m:
            yr = m[m.year.isin(YEARS)]
            absy = pd.crosstab(yr["group"], yr["year"])
            save_csv(absy, f"05_{vec}_by_year_absolute_mentions")
            docs_cat_year = yr.groupby(["group", "year"])["key"].nunique().unstack(fill_value=0)
            docs_vec_year = yr.groupby("year")["key"].nunique()
            rel_vec = docs_cat_year.div(docs_vec_year, axis=1) * 100
            save_csv(rel_vec.round(2), f"05_{vec}_by_year_pct_docs_with_vector")
            if docs_per_year_all is not None:
                rel_all = docs_cat_year.div(docs_per_year_all.reindex(docs_cat_year.columns), axis=1) * 100
                save_csv(rel_all.round(2), f"05_{vec}_by_year_pct_all_docs")
            rep.p("\n% of documents (among those with at least one phrase of the vector in that year) "
                  "mentioning each category, first and last year:")
            first, last = min(rel_vec.columns), max(rel_vec.columns)
            rep.table(rel_vec[[first, last]].round(1), floatfmt="{:,.1f}")
            R["pct_docs_first_last_year"] = rel_vec[[first, last]].round(2).to_dict()

            rbp = rank_by_period(m)
            save_csv(rbp, f"05_{vec}_rank_by_5y_period")
            mom = momentum(m)
            save_csv(mom.round(3), f"05_{vec}_momentum")
            rep.p("\nMomentum (2006-2015 vs 2016-2025):")
            rep.table(mom.round(2), floatfmt="{:,.2f}")
            R["momentum"] = mom.round(3).reset_index().to_dict(orient="records")

        if "country" in m:
            top_c = m.groupby("country")["key"].nunique().sort_values(ascending=False).head(TOP_COUNTRIES).index
            cc = pd.crosstab(m.loc[m.country.isin(top_c), "country"], m.loc[m.country.isin(top_c), "group"],
                             normalize="index")
            save_csv(cc.round(4), f"05_{vec}_country_x_category_rowprop")
        if "topic" in m:
            tc = pd.crosstab(m["topic"], m["group"], normalize="index")
            save_csv(tc.round(4), f"05_{vec}_topic_x_category_rowprop")
            R["n_topics_in_topic_matrix"] = int(tc.shape[0])
            rep.p(f"\nTopic x category matrix: {tc.shape[0]} topics x {tc.shape[1]} categories")
        rep.data[vec] = R
    rep.save()


if __name__ == "__main__":
    main()
