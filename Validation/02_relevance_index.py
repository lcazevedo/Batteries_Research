"""
02_relevance_index.py -- item 10: Citation-Based Topic Relevance Index (RI) per
topic, following the audit methodology in the Supplementary Information
('Citation-Based Topic Relevance Index').

Stage A -- reverse-engineers the exact formula used:
  * tests combinations of (citation column, last_year, current-year fraction)
    against the stored citations_per_year column, document by document;
  * tests combinations of (CPY formula, handling of years > last_year,
    'Review' rule, P90 baseline, year cutoff) against the stored
    topic_rank_index and topic_rank_index_wo_review columns;
  * selects the combination that reproduces the stored values (lowest max |diff|).
Stage B -- using the identified formula:
  * per-topic table (n, mean_CPY, prop_top10, log(1+n), index, rank);
  * component ranges and mean percentage contribution of each term;
  * Spearman rank correlation across weight scenarios, monthly citation rate,
    and standardized (z-score) components;
  * comparison against the published SI tables and review vs. non-review mean citations.

Outputs: audit_outputs/02_relevance_index.{txt,json} + CSVs.
"""
import itertools

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import Report, is_review, load_main, require, resolve_columns, save_csv

BASE_W = (0.5, 0.3, 0.2)
SCENARIOS = {
    "Original Baseline": (0.50, 0.30, 0.20),
    "Scenario 1: Equal Weights": (1 / 3, 1 / 3, 1 / 3),
    "Scenario 2: Impact Focus": (0.70, 0.15, 0.15),
    "Scenario 3: Excellence Focus": (0.30, 0.50, 0.20),
}

PUBLISHED_WITHOUT_REVIEW = {
    "Aqueous Zn-ion batteries: anode/electrolyte": 8.28, "Sodium-ion batteries and electrolytes": 6.88,
    "Vanadium-based cathodes for aqueous ZIBs": 6.82, "Li-ion battery SoH/aging, thermal & charging": 5.92,
    "Sustainable recycling of lithium-ion batteries": 5.89, "2D materials for metal-ion battery anodes": 5.86,
    "Carbon anode materials for ion batteries": 5.64, "Layered cathodes for sodium-ion batteries": 5.59,
    "Solid-state electrolytes for Li batteries": 5.50, "Organic electrode materials for batteries": 5.39,
    "Electrolytes, SEI and additives for Li batteries": 5.22, "MoS2 and molybdenum-based battery anodes": 5.20,
    "Advanced anode materials for Li-ion batteries": 5.01, "Antimony & bismuth battery anodes": 5.00,
    "Lithium-ion battery cathode materials": 4.85, "Silicon anodes for lithium-ion batteries": 4.81,
    "Polymer electrolytes & separators for batteries": 4.73, "Lithium-ion battery electrodes and modeling": 4.50,
    "Vanadium cathode materials for batteries": 4.39, "Iron oxide-based anodes for Li-ion batteries": 4.22,
    "Tin-based anode materials for ion batteries": 4.22, "TiO2/LTO anode materials for Li/Na batteries": 4.07,
    "LiFePO4 cathodes for lithium-ion batteries": 3.46,
}
PUBLISHED_WITH_REVIEW = {
    "Aqueous Zn-ion batteries: anode/electrolyte": 8.87, "Sodium-ion batteries and electrolytes": 8.52,
    "Li-ion battery SoH/aging, thermal & charging": 7.11, "Vanadium-based cathodes for aqueous ZIBs": 7.00,
    "2D materials for metal-ion battery anodes": 6.76, "Solid-state electrolytes for Li batteries": 6.71,
    "Sustainable recycling of lithium-ion batteries": 6.64, "Carbon anode materials for ion batteries": 6.30,
    "Organic electrode materials for batteries": 6.00, "Electrolytes, SEI and additives for Li batteries": 5.89,
    "Layered cathodes for sodium-ion batteries": 5.85, "Advanced anode materials for Li-ion batteries": 5.30,
    "MoS2 and molybdenum-based battery anodes": 5.29, "Polymer electrolytes & separators for batteries": 5.28,
    "Lithium-ion battery cathode materials": 5.12, "Antimony & bismuth battery anodes": 5.12,
    "Silicon anodes for lithium-ion batteries": 5.07, "Lithium-ion battery electrodes and modeling": 4.94,
    "Vanadium cathode materials for batteries": 4.49, "Tin-based anode materials for ion batteries": 4.27,
    "Iron oxide-based anodes for Li-ion batteries": 4.25, "TiO2/LTO anode materials for Li/Na batteries": 4.19,
    "LiFePO4 cathodes for lithium-ion batteries": 3.60,
}


def _norm_name(s) -> str:
    return (str(s).replace("\\ce{", "").replace("}", "").replace("\\&", "&")
            .replace("$_4$", "4").replace("$_2$", "2").strip().lower())


# ---------------------------------------------------------------------------
# Index computation for a given configuration
# ---------------------------------------------------------------------------
def build_rate(d: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Applies year cutoff, temporal exposure, and future-year handling."""
    x = d.copy()
    if cfg["year_cut"] is not None:
        x = x[x["year"] <= cfg["year_cut"]]
    x["cit"] = x[cfg["cit"]]
    expo = cfg["last_year"] - x["year"] + 1 + cfg["factor"]
    bad = expo <= 0
    if cfg["future"] == "drop":
        x, expo = x[~bad], expo[~bad]
    else:  # 'clip': minimum exposure = current-year fraction (or 1 year if fraction = 0)
        expo = expo.where(~bad, cfg["factor"] if cfg["factor"] > 0 else 1.0)
    x["expo"] = expo
    x["rate"] = x["cit"] / x["expo"]
    return x


def add_top10(x: pd.DataFrame, base: pd.DataFrame | None = None) -> pd.DataFrame:
    """top10 = citations >= P90 of publication year; P90 computed on 'base' (default: x itself)."""
    ref = x if base is None else base
    p90 = ref.groupby("year")["cit"].quantile(0.9)
    x = x.copy()
    x["top10"] = (x["cit"] >= x["year"].map(p90)).astype(float)
    return x


def components(x: pd.DataFrame, rate_col: str = "rate") -> pd.DataFrame:
    g = x.groupby("topic")
    out = pd.DataFrame({"n": g.size(), "mean_rate": g[rate_col].mean(), "prop_top10": g["top10"].mean()})
    out["log1p_n"] = np.log1p(out["n"])
    return out


def index_from(comp: pd.DataFrame, w, standardize=False) -> pd.Series:
    X = comp[["mean_rate", "prop_top10", "log1p_n"]]
    if standardize:
        X = (X - X.mean()) / X.std(ddof=0)
    return w[0] * X["mean_rate"] + w[1] * X["prop_top10"] + w[2] * X["log1p_n"]


def compute(d: pd.DataFrame, cfg: dict, with_reviews: bool) -> pd.DataFrame:
    x = build_rate(d, cfg)
    rev = x["review_any"] if cfg["review_rule"] == "any" else x["review_exact"]
    sub = x if with_reviews else x[~rev]
    sub = add_top10(sub, base=x if cfg["p90_base"] == "all" else None)
    comp = components(sub)
    comp["index"] = index_from(comp, BASE_W)
    return comp


# ---------------------------------------------------------------------------
def main():
    rep = Report("02_relevance_index")
    df = load_main()
    cols = resolve_columns(df)
    require(cols, ["year", "citations", "doc_type", "topic"], "02_relevance_index")

    d = pd.DataFrame({
        "year": pd.to_numeric(df[cols["year"]], errors="coerce"),
        "cit_total": pd.to_numeric(df[cols["citations"]], errors="coerce").fillna(0),
        "topic": df[cols["topic"]].astype(str),
        "review_any": is_review(df[cols["doc_type"]]),
        "review_exact": df[cols["doc_type"]].astype(str).str.strip() == "Review",
    })
    cit_opts = {"total": "cit_total"}
    if cols.get("citations_wos_core") and cols["citations_wos_core"] != cols["citations"]:
        d["cit_core"] = pd.to_numeric(df[cols["citations_wos_core"]], errors="coerce").fillna(0)
        cit_opts["wos_core"] = "cit_core"
    rep.p(f"Citations 'total' = {cols['citations']} | 'wos_core' = {cols.get('citations_wos_core')}")
    rep.p(f"Missing year (excluded): {int(d['year'].isna().sum())}")
    d = d.dropna(subset=["year"])
    rep.p(f"Documents by year >= 2025: {d.loc[d.year >= 2025, 'year'].value_counts().sort_index().to_dict()}")

    # ------------------------------------------------- A1: citations_per_year
    rep.h("A1. WHICH FORMULA GENERATED THE STORED citations_per_year COLUMN?")
    if cols.get("citations_per_year"):
        stored = pd.to_numeric(df.loc[d.index, cols["citations_per_year"]], errors="coerce")
        res = []
        for (cn, cc), ly, fac in itertools.product(cit_opts.items(), [2024, 2025], [0.0, 0.75]):
            expo = ly - d["year"] + 1 + fac
            calc = d[cc] / expo
            ok = stored.notna() & np.isfinite(calc) & (expo > 0)
            match = (np.abs(stored[ok] - calc[ok]) < 1e-6).mean()
            res.append({"citacoes": cn, "last_year": ly, "fracao": fac, "pct_docs_iguais": 100 * match})
        res = pd.DataFrame(res).sort_values("pct_docs_iguais", ascending=False)
        rep.table(res, floatfmt="{:,.2f}")
        rep.data["cpy_column_formula_search"] = res.to_dict(orient="records")
        for y in [2024, 2025, 2026]:
            s = stored[d["year"] == y]
            if len(s):
                rep.p(f"  stored citations_per_year, year {y}: n={len(s)}, "
                      f"infinites={int(np.isinf(s).sum())}, nulls={int(s.isna().sum())}, "
                      f"median={s[np.isfinite(s)].median():.2f}")

    # --------------------------------------------- A2: stored topic_rank_index
    rep.h("A2. WHICH CONFIGURATION REPRODUCES THE STORED INDEX (topic_rank_index[_wo_review])?")
    targets = {}
    for key, wr in [("stored_ri", True), ("stored_ri_wo_review", False)]:
        if cols.get(key):
            targets[wr] = df.loc[d.index].groupby(d["topic"])[cols[key]].first()
    grid = []
    space = itertools.product(cit_opts.values(), [2024, 2025], [0.0, 0.75], ["clip", "drop"],
                              ["any", "exact"], ["subset", "all"], [None, 2024])
    for cc, ly, fac, fut, rr, pb, yc in space:
        cfg = dict(cit=cc, last_year=ly, factor=fac, future=fut, review_rule=rr, p90_base=pb, year_cut=yc)
        row = dict(cfg)
        for wr, tgt in targets.items():
            idx = compute(d, cfg, wr)["index"]
            row[f"maxdiff_{'with' if wr else 'without'}_review"] = float(
                np.nanmax(np.abs(idx.reindex(tgt.index) - tgt)))
        grid.append(row)
    grid = pd.DataFrame(grid)
    diff_cols = [c for c in grid.columns if c.startswith("maxdiff")]
    if diff_cols:
        grid["score"] = grid[diff_cols].max(axis=1)
        grid = grid.sort_values("score")
        rep.p("Top 10 configurations (max |difference| between recalculated and stored index):")
        rep.table(grid.head(10), floatfmt="{:,.5f}")
        best = grid.iloc[0]
        rep.data["ri_formula_search_top10"] = grid.head(10).to_dict(orient="records")
        if best["score"] > 0.01:
            rep.p("!! No combination reproduces the stored index (diff > 0.01). Check the original "
                  "code; Stage B uses the closest approximation.")
    else:
        rep.p("[info] topic_rank_index columns not found; using default configuration.")
        best = pd.Series(dict(cit="cit_total", last_year=2025, factor=0.0, future="clip",
                              review_rule="any", p90_base="subset", year_cut=None, score=np.nan))
    cfg = {k: best[k] for k in ["cit", "last_year", "factor", "future", "review_rule", "p90_base", "year_cut"]}
    cfg["year_cut"] = None if pd.isna(cfg["year_cut"]) else int(cfg["year_cut"])
    rep.p(f"\nADOPTED CONFIGURATION: {cfg}")
    rep.data["config_used"] = {k: (v.item() if hasattr(v, "item") else v) for k, v in cfg.items()}

    # ---------------------------------------------------------- B: analysis
    rep.h("REVIEWS vs. NON-REVIEWS")
    rev = d["review_any"] if cfg["review_rule"] == "any" else d["review_exact"]
    m_rev, m_non = d.loc[rev, cfg["cit"]].mean(), d.loc[~rev, cfg["cit"]].mean()
    rep.p(f"Reviews: {int(rev.sum()):,} docs, mean {m_rev:.1f} citations | non-reviews: "
          f"{int((~rev).sum()):,}, mean {m_non:.1f} | ratio {m_rev / m_non:.2f}x")
    rep.data["review_mean_citations"], rep.data["non_review_mean_citations"] = float(m_rev), float(m_non)

    for wr, label, published in [(False, "without_review", PUBLISHED_WITHOUT_REVIEW),
                                 (True, "with_review", PUBLISHED_WITH_REVIEW)]:
        R = {}
        x = build_rate(d, cfg)
        rv = x["review_any"] if cfg["review_rule"] == "any" else x["review_exact"]
        sub = x if wr else x[~rv]
        sub = add_top10(sub, base=x if cfg["p90_base"] == "all" else None)
        sub["rate_month"] = sub["rate"] / 12
        comp = components(sub)
        comp["index"] = index_from(comp, BASE_W)
        comp = comp.sort_values("index", ascending=False)
        comp["rank"] = np.arange(1, len(comp) + 1)
        rep.h(f"RI -- {label} (n docs = {len(sub):,}; topics = {len(comp)})")
        rep.table(comp, floatfmt="{:,.4f}")
        save_csv(comp, f"02_RI_{label}")
        R["table"] = comp.reset_index().to_dict(orient="records")

        ranges = {k: (float(comp[k].min()), float(comp[k].max())) for k in ["mean_rate", "prop_top10", "log1p_n"]}
        contrib = pd.DataFrame({
            "alpha*mean_CPY": BASE_W[0] * comp["mean_rate"] / comp["index"],
            "beta*prop_top10": BASE_W[1] * comp["prop_top10"] / comp["index"],
            "gamma*log(1+n)": BASE_W[2] * comp["log1p_n"] / comp["index"],
        }).mean() * 100
        rep.p("\nUnweighted component ranges:")
        for k, v in ranges.items():
            rep.p(f"  {k:12s} {v[0]:.4f} -- {v[1]:.4f}")
        rep.p("Mean percentage contribution of each term to the index:")
        for k, v in contrib.items():
            rep.p(f"  {k:18s} {v:.1f}%")
        R["ranges"], R["mean_contribution_pct"] = ranges, contrib.round(2).to_dict()

        base = comp["index"]
        sens = []
        for name, w in SCENARIOS.items():
            rho = spearmanr(base, index_from(comp, w).loc[base.index]).statistic
            sens.append({"scenario": name, "alpha": round(w[0], 2), "beta": round(w[1], 2),
                         "gamma": round(w[2], 2), "spearman_rho": round(float(rho), 4)})
        rep.p("\nWeight sensitivity (Spearman vs. baseline ranking):")
        rep.table(pd.DataFrame(sens))
        R["weight_sensitivity"] = sens

        comp_m = components(sub, "rate_month")
        rho_m = spearmanr(base, index_from(comp_m, BASE_W).loc[base.index]).statistic
        idx_z = index_from(comp, BASE_W, standardize=True)
        rho_z = spearmanr(base, idx_z.loc[base.index]).statistic
        t_base = list(base.sort_values(ascending=False).index[:5])
        t_z = list(idx_z.sort_values(ascending=False).index[:5])
        rep.p(f"\nCitations per MONTH: Spearman = {rho_m:.3f}")
        rep.p(f"Standardized components (z-score): Spearman = {rho_z:.3f} | top-3 identical in same "
              f"order? {t_base[:3] == t_z[:3]} | top-5 identical as set? {set(t_base) == set(t_z)}")
        R.update(spearman_per_month=float(rho_m), spearman_standardized=float(rho_z),
                 top5_base=t_base, top5_standardized=t_z)

        pub = {_norm_name(k): v for k, v in published.items()}
        mine = {_norm_name(k): v for k, v in comp["index"].items()}
        common = sorted(set(pub) & set(mine))
        if len(common) >= 3:
            a = np.array([mine[k] for k in common]); b = np.array([pub[k] for k in common])
            rho = spearmanr(a, b).statistic
            rep.p(f"\nComparison with current SI table: {len(common)} topics | Spearman = {rho:.3f} | "
                  f"max |diff| = {np.max(np.abs(a - b)):.3f}")
            R["vs_published_SI"] = {"matched": len(common), "spearman": float(rho),
                                    "max_abs_diff": float(np.max(np.abs(a - b)))}
        rep.data[label] = R
    rep.save()


if __name__ == "__main__":
    main()
