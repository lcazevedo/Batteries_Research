"""
01_corpus_composition.py -- items 8 and 9: metadata completeness (Table 2 of
the paper) and corpus composition (Section 3.2 of the paper; S-2.2 of the SI).

Computes across the full main dataset:
  * N, time span, annual distribution (including partial 2025 and 2026 Early Access);
  * completeness of each field (absolute and %), formatted for Table 2;
  * incomplete records (missing title/abstract/year) and duplicates (DOI / ID);
  * document types (multi-valued labels) and languages;
  * WoS categories and research areas (full counts, for SI longtable);
  * number of countries, institutions, journals, and authors; total and mean citations;
  * mean citations of reviews vs. non-reviews;
  * size of each topic in each EVoC layer (verifying sum = N).

Outputs: audit_outputs/01_corpus_composition.{txt,json} + CSVs + LaTeX table.
"""
import re
from collections import Counter

import numpy as np
import pandas as pd

from common import (OUT_DIR, Report, is_filled, is_review, load_main, resolve_columns,
                    save_csv, split_multi, topic_layer_columns)


# ---------------------------------------------------------------------------
# Countries and institutions extracted from the WoS address field (C1)
# "[Author A; Author B] Univ X, Dept Y, City, Country; [Author C] Inst Z, ..., USA"
# ---------------------------------------------------------------------------
_US_STATE_ZIP = re.compile(r"\b[A-Z]{2}\s+\d{5}(-\d{4})?\s+USA$")


def parse_addresses(value) -> list[tuple[str, str]]:
    """List of (institution, country) tuples in order of appearance."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        parts = [str(v) for v in value]
    else:
        s = str(value)
        parts = re.split(r";\s*(?=\[)", s) if "[" in s else s.split(";")
    out = []
    for p in parts:
        p = re.sub(r"^\s*\[[^\]]*\]\s*", "", p).strip().rstrip(".")
        if not p:
            continue
        segs = [x.strip() for x in p.split(",") if x.strip()]
        if not segs:
            continue
        inst = segs[0]
        country = segs[-1]
        if country.endswith("USA") or _US_STATE_ZIP.search(country):
            country = "USA"
        out.append((inst, country))
    return out


def looks_like_addresses(series: pd.Series) -> bool:
    sample = series.dropna().astype(str).head(2000)
    if sample.empty:
        return False
    return (sample.str.count(",") >= 2).mean() > 0.5


def label_counts(series: pd.Series) -> pd.Series:
    c = Counter()
    for v in series:
        c.update(split_multi(v))
    return pd.Series(c, dtype="int64").sort_values(ascending=False)


def main():
    rep = Report("01_corpus_composition")
    df = load_main()
    cols = resolve_columns(df)
    N = len(df)
    rep.data["N"] = N
    rep.h(f"CORPUS: N = {N:,} documents")

    # ------------------------------------------------------------------ years
    if cols["year"]:
        y = pd.to_numeric(df[cols["year"]], errors="coerce")
        rep.p(f"Period: {int(y.min())}--{int(y.max())} | missing year: {int(y.isna().sum())}")
        by_year = y.value_counts().sort_index()
        rep.data["year_min"], rep.data["year_max"] = int(y.min()), int(y.max())
        rep.data["docs_by_year"] = {int(k): int(v) for k, v in by_year.items()}
        rep.p("Documents in recent years:",
              {int(k): int(v) for k, v in by_year.loc[by_year.index >= 2018].items()})
        last5 = by_year.loc[(by_year.index >= 2020) & (by_year.index <= 2024)].sum()
        rep.p(f"Share 2020-2024: {last5:,} ({100 * last5 / N:.2f}%)")
        rep.data["share_2020_2024"] = float(100 * last5 / N)
        save_csv(by_year.rename("n_docs").to_frame(), "01_docs_by_year")

    # ------------------------------------------------------------- completeness
    rep.h("METADATA COMPLETENESS (Table 2)")
    table2_keys = [("Sources", "source"), ("Abstracts", "abstract"), ("Affiliation", "affiliations"),
                   ("Author(s)", "authors"), ("DOI", "doi"), ("Keywords - Authors", "author_keywords"),
                   ("Keywords - Plus", "keywords_plus"), ("References", "references"),
                   ("Year", "year")]
    extra_keys = [("Title", "title"), ("Times cited", "citations"), ("Impact factor", "impact_factor"),
                  ("Country (first author)", "country"), ("Institution (first author)", "institution"),
                  ("Document type", "doc_type"), ("Language", "language"),
                  ("WoS categories", "wos_categories"), ("Research areas", "research_areas")]
    comp = []
    for label, key in table2_keys + extra_keys:
        col = cols.get(key)
        if col is None:
            comp.append({"campo": label, "coluna": None, "n": None, "pct": None,
                         "tabela2": (label, key) in table2_keys})
            continue
        if key == "impact_factor":
            n = int((pd.to_numeric(df[col], errors="coerce") > 0).sum())
        else:
            n = int(is_filled(df[col]).sum())
        comp.append({"campo": label, "coluna": col, "n": n, "pct": round(100 * n / N, 2),
                     "tabela2": (label, key) in table2_keys})
    comp_df = pd.DataFrame(comp)
    comp_df["n"] = comp_df["n"].astype("Int64")
    rep.table(comp_df, floatfmt="{:,.2f}")
    rep.data["completeness"] = comp
    save_csv(comp_df, "01_completeness", index=False)

    tex = ["\\begin{tabular}{lrr} \\toprule",
           "Entries & \\multicolumn{2}{c}{Completeness} \\\\ \\midrule"]
    for r in comp:
        if r["tabela2"] and r["n"] is not None:
            tex.append(f"{r['campo']:<20s} & \\num{{{r['n']}}} & \\SI{{{r['pct']:.2f}}}{{\\percent}} \\\\")
    tex.append("\\bottomrule\n\\end{tabular}")
    (OUT_DIR / "01_table2_completeness.tex").write_text("\n".join(tex), encoding="utf-8")

    # ------------------------------------------------- incomplete/duplicates
    rep.h("INCOMPLETE RECORDS AND DUPLICATES")
    inc = {}
    for key in ["title", "abstract", "year", "authors", "affiliations", "citations"]:
        if cols.get(key):
            inc[key] = int((~is_filled(df[cols[key]])).sum())
    rep.p("Records missing field:", inc)
    rep.data["missing_by_field"] = inc
    if cols["title"] and cols["abstract"]:
        both = (~is_filled(df[cols["title"]])) | (~is_filled(df[cols["abstract"]]))
        rep.p(f"Missing title OR abstract: {int(both.sum())}")
        rep.data["missing_title_or_abstract"] = int(both.sum())
    if cols["abstract"]:
        short = df[cols["abstract"]].astype(str).str.len() <= 20
        rep.p(f"Abstract with <= 20 characters (treated as 'Not_Applicable' in research_type): "
              f"{int(short.sum())}")
        rep.data["abstract_le_20_chars"] = int(short.sum())
    for key in ["doi", "doc_id"]:
        c = cols.get(key)
        if c:
            s = df[c][is_filled(df[c])].astype(str).str.strip().str.lower()
            d = int(s.duplicated().sum())
            rep.p(f"Duplicates by {key} ({c}): {d}")
            rep.data[f"duplicates_{key}"] = d

    # ------------------------------------------------------ document types and languages
    if cols["doc_type"]:
        rep.h("DOCUMENT TYPES (labels; a record may have more than one)")
        dt = label_counts(df[cols["doc_type"]])
        rep.table(dt.to_frame("n"))
        rep.data["doc_type_labels"] = dt.to_dict()
        combos = df[cols["doc_type"]].astype(str).value_counts()
        rep.p("\nExact combinations (raw field value):")
        rep.table(combos.to_frame("n"))
        rep.data["doc_type_raw_values"] = combos.to_dict()
        rev = is_review(df[cols["doc_type"]])
        rev_exact = (df[cols["doc_type"]].astype(str).str.strip() == "Review")
        rep.p(f"\nReviews ('Review' label in any position): {int(rev.sum()):,}")
        rep.p(f"Reviews (field exactly == 'Review', rule in research_type classifier): "
              f"{int(rev_exact.sum()):,}")
        rep.data["n_reviews_any_label"] = int(rev.sum())
        rep.data["n_reviews_exact"] = int(rev_exact.sum())
    if cols["language"]:
        rep.h("LANGUAGES")
        lang = df[cols["language"]].astype(str).value_counts()
        rep.table(lang.to_frame("n"))
        rep.data["languages"] = lang.to_dict()

    # ------------------------------------------- WoS categories and research areas
    for key, name in [("wos_categories", "WOS CATEGORIES"), ("research_areas", "RESEARCH AREAS")]:
        if cols.get(key):
            rep.h(f"{name} (full count; not mutually exclusive)")
            lc = label_counts(df[cols[key]])
            rep.p(f"Distinct categories: {len(lc)} | sum of labels: {int(lc.sum()):,}")
            top3 = int(lc.head(3).sum())
            rep.p(f"Top-3 sum to {top3:,} labels ({100 * top3 / lc.sum():.1f}% of all labels)")
            rep.table(lc.head(40).to_frame("n"))
            rep.data[key] = lc.to_dict()
            save_csv(lc.to_frame("n"), f"01_{key}")

    # ------------------------------------------ countries, institutions, etc.
    rep.h("ENTITIES: COUNTRIES, INSTITUTIONS, JOURNALS, AUTHORS")
    ent = {}
    if cols["affiliations"] and looks_like_addresses(df[cols["affiliations"]]):
        parsed = df[cols["affiliations"]].map(parse_addresses)
        any_c, any_i, first_c, first_i = set(), set(), Counter(), Counter()
        for lst in parsed:
            for inst, ctry in lst:
                any_i.add(inst)
                any_c.add(ctry)
            if lst:
                first_i[lst[0][0]] += 1
                first_c[lst[0][1]] += 1
        ent.update({"countries_any_author": len(any_c), "institutions_any_author": len(any_i),
                    "countries_first_address": len(first_c),
                    "institutions_first_address": len(first_i),
                    "docs_without_parsable_address": int((parsed.map(len) == 0).sum())})
        rep.p("(countries/institutions extracted from address field, 1st segment = institution, "
              "last = country)")
    for key, name in [("countries_all_authors", "countries_any_author_list_column"),
                      ("institutions_all_authors", "institutions_any_author_list_column")]:
        if cols.get(key):
            s = set()
            for v in df[cols[key]]:
                s.update(x for x in split_multi(v) if x.lower() not in {"nan", "none", "unknown"})
            ent[name] = len(s)
    if cols["country"]:
        ent["countries_from_country_column"] = int(df[cols["country"]].dropna().astype(str).nunique())
        top = df[cols["country"]].astype(str).value_counts().head(15)
        rep.data["top_first_countries"] = top.to_dict()
    if cols["institution"]:
        ent["institutions_from_institution_column"] = int(
            df[cols["institution"]].dropna().astype(str).nunique())
    if cols["source"]:
        ent["journals"] = int(df[cols["source"]].dropna().astype(str).str.upper().str.strip().nunique())
    if cols["authors"]:
        a = set()
        for v in df[cols["authors"]]:
            a.update(x.strip().lower() for x in split_multi(v))
        ent["authors_unique_full_name_strings"] = len(a)
    for k, v in ent.items():
        rep.p(f"  {k:40s} {v:,}")
    rep.data["entities"] = ent

    # -------------------------------------------------------------- citations
    if cols["citations"]:
        rep.h("CITATIONS")
        cit = pd.to_numeric(df[cols["citations"]], errors="coerce").fillna(0)
        rep.p(f"Total citations: {int(cit.sum()):,} | mean per document: {cit.mean():.2f} | "
              f"median: {cit.median():.0f}")
        rep.data["citations_total"] = int(cit.sum())
        rep.data["citations_mean"] = float(cit.mean())
        if cols["doc_type"]:
            rev = is_review(df[cols["doc_type"]])
            m_rev, m_non = cit[rev].mean(), cit[~rev].mean()
            rep.p(f"Mean citations: reviews = {m_rev:.1f} | non-reviews = {m_non:.1f} | "
                  f"ratio = {m_rev / m_non:.2f}x")
            rep.data["citations_mean_review"] = float(m_rev)
            rep.data["citations_mean_non_review"] = float(m_non)
            rep.data["citations_ratio_review_non_review"] = float(m_rev / m_non)

    # ---------------------------------------------------- topic sizes
    layers = topic_layer_columns(df)
    if cols["topic"] and cols["topic"] not in layers:
        layers.append(cols["topic"])
    for c in layers:
        rep.h(f"TOPICS -- column {c}")
        vc = df[c].value_counts(dropna=False)
        rep.p(f"Number of topics: {vc.size} | sum = {int(vc.sum()):,} (N = {N:,}) | "
              f"nulls: {int(df[c].isna().sum())} | label -1: {int((df[c].astype(str) == '-1').sum())}")
        rep.table(vc.to_frame("n_docs"))
        rep.data[f"topic_sizes__{c}"] = {str(k): int(v) for k, v in vc.items()}
        save_csv(vc.to_frame("n_docs"), f"01_topic_sizes__{c}")

    rep.save()


if __name__ == "__main__":
    main()
