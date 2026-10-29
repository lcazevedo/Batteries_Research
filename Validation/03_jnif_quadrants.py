"""
03_jnif_quadrants.py -- item 16: definitions of JNIF and impact quadrants
('Scientometric Indicators for Research Impact').

Computes:
  * N with impact factor (journal_if > 0) and N without IF (by document type);
  * medians of IF and citations_per_year (stored column) and counts in each
    quadrant (Star, Hidden Gem, Coattail, Long Tail);
  * verifies whether the medians reproduce the stored category_group column;
  * JNIF = citations / IF; verifies against the stored JNIF column; quartile thresholds;
  * the same statistics for 1993--2024 (time span of the SI figures).

Outputs: audit_outputs/03_jnif_quadrants.{txt,json}.
"""
import numpy as np
import pandas as pd

from common import Report, exposure_years, load_main, require, resolve_columns

QUAD_SHORT = {"star": "Star", "hidden": "Hidden Gem", "coattail": "Coattail", "long": "Long Tail"}


def short_quad(v) -> str | None:
    s = str(v).lower()
    for k, name in QUAD_SHORT.items():
        if s.startswith(k):
            return name
    return None


def block(d: pd.DataFrame, label: str, rep: Report) -> dict:
    res = {"n_total": len(d)}
    rep.h(f"{label}: {len(d):,} documents")
    has_if = d["IF"] > 0
    res["n_with_IF"], res["n_without_IF"] = int(has_if.sum()), int((~has_if).sum())
    rep.p(f"With IF: {res['n_with_IF']:,} | without IF: {res['n_without_IF']:,} "
          f"({100 * res['n_without_IF'] / len(d):.2f}%)")
    if "doc_type" in d:
        top = d.loc[~has_if, "doc_type"].astype(str).value_counts().head(8).to_dict()
        rep.p("Without IF by document type:", top)
        res["without_IF_by_doc_type"] = top

    q = d.loc[has_if & np.isfinite(d["cpy"])].copy()
    t_if, t_cpy = q["IF"].median(), q["cpy"].median()
    res["median_IF"], res["median_CPY"] = float(t_if), float(t_cpy)
    rep.p(f"Median IF = {t_if:.3f} | median CPY = {t_cpy:.3f} (n = {len(q):,})")
    hi_if, hi_c = q["IF"] >= t_if, q["cpy"] >= t_cpy
    q["quad"] = np.select([hi_if & hi_c, ~hi_if & hi_c, hi_if & ~hi_c],
                          ["Star", "Hidden Gem", "Coattail"], "Long Tail")
    vc = q["quad"].value_counts()
    rep.p("Recalculated quadrants:", {k: f"{v:,} ({100 * v / len(q):.1f}%)" for k, v in vc.items()})
    res["quadrants"] = vc.to_dict()
    if "stored_quad" in q:
        sq = q["stored_quad"].map(short_quad)
        ok = sq.notna()
        agree = (sq[ok] == q.loc[ok, "quad"]).mean()
        rep.p(f"Agreement with stored category_group: {100 * agree:.2f}% of {int(ok.sum()):,} docs")
        rep.p("Stored category_group:", sq.value_counts().to_dict())
        res["agreement_with_stored_quadrant_pct"] = float(100 * agree)
        res["stored_quadrants"] = sq.value_counts().to_dict()

    q["JNIF_calc"] = q["cit"] / q["IF"]
    if "stored_jnif" in q:
        diff = (q["JNIF_calc"] - q["stored_jnif"]).abs()
        rep.p(f"Recalculated vs. stored JNIF: max |diff| = {diff.max():.6f}")
        res["jnif_max_abs_diff_vs_stored"] = float(diff.max())
    p25, p50, p75 = q["JNIF_calc"].quantile([0.25, 0.5, 0.75])
    res["JNIF_quartiles"] = {"P25": float(p25), "P50": float(p50), "P75": float(p75)}
    rep.p(f"JNIF: Q1 <= {p25:.2f} < Q2 <= {p50:.2f} < Q3 <= {p75:.2f} < Q4 | mean {q['JNIF_calc'].mean():.2f}")
    return res


def main():
    rep = Report("03_jnif_quadrants")
    df = load_main()
    cols = resolve_columns(df)
    require(cols, ["year", "citations", "impact_factor"], "03_jnif_quadrants")
    d = pd.DataFrame({
        "year": pd.to_numeric(df[cols["year"]], errors="coerce"),
        "cit": pd.to_numeric(df[cols["citations"]], errors="coerce").fillna(0),
        "IF": pd.to_numeric(df[cols["impact_factor"]], errors="coerce").fillna(0),
    })
    if cols.get("citations_per_year"):
        d["cpy"] = pd.to_numeric(df[cols["citations_per_year"]], errors="coerce")
        rep.p(f"CPY = stored column {cols['citations_per_year']}")
    else:
        d["cpy"] = d["cit"] / exposure_years(d["year"]).clip(lower=1)
        rep.p("CPY recalculated (stored column not found)")
    if cols.get("doc_type"):
        d["doc_type"] = df[cols["doc_type"]]
    if cols.get("stored_quadrant"):
        d["stored_quad"] = df[cols["stored_quadrant"]]
    if cols.get("stored_jnif"):
        d["stored_jnif"] = pd.to_numeric(df[cols["stored_jnif"]], errors="coerce")
    rep.p(f"Citations: {cols['citations']} | IF: {cols['impact_factor']} (JCR year: VERIFY)")
    n_inf = int((~np.isfinite(d["cpy"])).sum())
    rep.p(f"Non-finite CPY (e.g., year 2026 with denominator 0): {n_inf} docs, excluded from quadrants")
    rep.data["n_cpy_not_finite"] = n_inf
    d = d.dropna(subset=["year"])
    rep.data["all_years"] = block(d, "FULL CORPUS", rep)
    rep.data["1993_2024"] = block(d[(d.year >= 1993) & (d.year <= 2024)], "PERIOD 1993-2024", rep)
    rep.save()


if __name__ == "__main__":
    main()
