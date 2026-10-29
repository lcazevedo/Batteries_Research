"""
06a_make_validation_samples.py -- items 12 and 13: generates spreadsheets for
manual domain-expert validation (for LOCAL use only: contains WoS sentences and
abstracts).

Generates in audit_outputs/:
  validation_challenges.xlsx, validation_directions.xlsx, validation_controversies.xlsx
     sheet 'amostra'   : 100 categorized sentences, disproportionate stratified
                         sample (~8 per category, seed 42);
     sheet 'none_check': 30 sentences classified as 'No X', to estimate how many
                         'No X' sentences were actually challenge/direction/controversy;
     sheet 'categorias': list of valid categories (populates the dropdown menu).
  validation_research_type.xlsx
     100 randomly sampled non-review documents (seed 42), with title, abstract,
     and predicted class (experimental vs. computational).

The domain expert fills ONLY the 'expert_*' columns (dropdown menu). Afterwards,
run 06b_score_validation.py.

BLIND = True hides the predicted category from the domain expert (saved
separately in *_gabarito.csv). More rigorous, though more time-consuming.
Default: False.
"""
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.worksheet.datavalidation import DataValidation

from common import (OUT_DIR, RANDOM_SEED, Report, is_review, load_main, load_phrases,
                    resolve_columns)

N_SAMPLE, N_NONE, N_RTYPE = 100, 30, 100   # 100 sentences per vector
BLIND = False
SENTENCE_COL = "phrase"
VECTORS = {
    "challenges": ("challenges_label", "challenges_group", "NOT_A_CHALLENGE"),
    "directions": ("directions_label", "directions_group", "NOT_A_DIRECTION"),
    "controversies": ("controversies_label", "controversies_group", "NOT_A_CONTROVERSY"),
}
RTYPE_CLASSES = ["Computational and simulation studies", "Experimental studies"]

# reuses the same 'No X' rule from script 05
from importlib import import_module
_s05 = import_module("05_intent_stats")
classify_status = _s05.classify_status


def stratified(df: pd.DataFrame, by: str, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    groups = sorted(df[by].unique())
    per = n // len(groups)
    picked = []
    for g in groups:
        idx = df.index[df[by] == g].tolist()
        rng.shuffle(idx)
        picked.extend(idx[:per])
    rest = list(set(df.index) - set(picked))
    rng.shuffle(rest)
    picked.extend(rest[: n - len(picked)])
    out = df.loc[picked].sample(frac=1, random_state=seed)  # shuffles review order
    return out


def add_dropdown(path, sheet, col_letter, n_rows, list_ref):
    wb = load_workbook(path)
    ws = wb[sheet]
    dv = DataValidation(type="list", formula1=list_ref, allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"{col_letter}2:{col_letter}{n_rows + 1}")
    for c, w in {"A": 10, "B": 18, "C": 90, "D": 35, "E": 35, "F": 35}.items():
        ws.column_dimensions[c].width = w
    wb.save(path)


def col_letter(df: pd.DataFrame, col: str) -> str:
    return chr(ord("A") + list(df.columns).index(col))


def main():
    rep = Report("06a_make_validation_samples")
    ph = load_phrases()
    df = load_main()
    cols = resolve_columns(df)
    key = "doi_phrase" if "doi_phrase" in ph.columns else cols.get("doc_id")

    for vec, (lab, grp, not_a) in VECTORS.items():
        if grp not in ph.columns:
            rep.p(f"[warning] {grp} missing; {vec} skipped")
            continue
        st = classify_status(ph[grp])
        if lab in ph.columns:
            st_l = classify_status(ph[lab])
            cat = ph[(st == "category") & (st_l == "category")]
            none = ph[(st == "none") | (st_l == "none")]
        else:
            cat, none = ph[st == "category"], ph[st == "none"]
        cats = sorted(cat[grp].unique())
        s = stratified(cat, grp, N_SAMPLE, RANDOM_SEED)
        sample = pd.DataFrame({
            "sample_id": range(1, len(s) + 1),
            "doc_key": s[key] if key in s else s.index,
            "sentence": s[SENTENCE_COL] if SENTENCE_COL in s else "",
            "extracted_label": s[lab] if lab in s else "",
            "predicted_category": s[grp].values,
            "expert_category": "",
            "expert_note": "",
        })
        gabarito = None
        if BLIND:
            gabarito = sample[["sample_id", "predicted_category"]]
            sample = sample.drop(columns=["predicted_category"])
        nn = none.sample(min(N_NONE, len(none)), random_state=RANDOM_SEED)
        none_df = pd.DataFrame({
            "sample_id": range(1, len(nn) + 1),
            "doc_key": nn[key] if key in nn else nn.index,
            "sentence": nn[SENTENCE_COL] if SENTENCE_COL in nn else "",
            "extracted_label": nn[lab] if lab in nn else "",
            "expert_is_" + vec[:-1]: "",          # SIM / NAO (YES / NO)
            "expert_category_if_yes": "",
        })
        cat_list = pd.DataFrame({"categorias": cats + [not_a]})
        path = OUT_DIR / f"validation_{vec}.xlsx"
        with pd.ExcelWriter(path, engine="openpyxl") as w:
            sample.to_excel(w, sheet_name="amostra", index=False)
            none_df.to_excel(w, sheet_name="none_check", index=False)
            cat_list.to_excel(w, sheet_name="categorias", index=False)
        ref = f"categorias!$A$2:$A${len(cat_list) + 1}"
        add_dropdown(path, "amostra", col_letter(sample, "expert_category"), len(sample), ref)
        add_dropdown(path, "none_check", col_letter(none_df, "expert_is_" + vec[:-1]), len(none_df),
                     '"SIM,NAO"')
        add_dropdown(path, "none_check", col_letter(none_df, "expert_category_if_yes"), len(none_df), ref)
        if gabarito is not None:
            gabarito.to_csv(OUT_DIR / f"validation_{vec}_gabarito.csv", index=False)
        rep.p(f"{vec}: {len(sample)} sentences ({len(cats)} categories, ~{N_SAMPLE // len(cats)} per "
              f"category) + {len(none_df)} 'No X' -> {path}")
        rep.data[vec] = {"n_sample": len(sample), "n_categories": len(cats),
                         "per_category": s[grp].value_counts().to_dict(),
                         "n_none_check": len(none_df)}

    # ----------------------------------------------------------- research type
    if cols.get("research_type") and cols.get("abstract"):
        rt = df[cols["research_type"]].astype(str)
        rep.h("DISTRIBUTION OF research_type (full dataset)")
        rep.table(rt.value_counts().to_frame("n"))
        rep.data["research_type_counts"] = rt.value_counts().to_dict()
        elig = df[(~is_review(df[cols["doc_type"]])) & rt.isin(RTYPE_CLASSES)] if cols.get("doc_type") \
            else df[rt.isin(RTYPE_CLASSES)]
        s = elig.sample(min(N_RTYPE, len(elig)), random_state=RANDOM_SEED)
        out = pd.DataFrame({
            "sample_id": range(1, len(s) + 1),
            "doc_key": s[cols["doi"]] if cols.get("doi") else s.index,
            "title": s[cols["title"]] if cols.get("title") else "",
            "abstract": s[cols["abstract"]],
            "predicted_research_type": s[cols["research_type"]],
            "expert_research_type": "",
            "expert_note": "",
        })
        path = OUT_DIR / "validation_research_type.xlsx"
        with pd.ExcelWriter(path, engine="openpyxl") as w:
            out.to_excel(w, sheet_name="amostra", index=False)
            pd.DataFrame({"classes": RTYPE_CLASSES}).to_excel(w, sheet_name="classes", index=False)
        add_dropdown(path, "amostra", col_letter(out, "expert_research_type"), len(out),
                     f"classes!$A$2:$A${len(RTYPE_CLASSES) + 1}")
        rep.p(f"research_type: {len(out)} documents -> {path}")
    rep.save()


if __name__ == "__main__":
    main()
