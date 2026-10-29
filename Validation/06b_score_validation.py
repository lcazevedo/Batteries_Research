"""
06b_score_validation.py -- computes validation metrics from the expert-annotated
spreadsheets (06a) or anonymized label-only CSV files. Run after the 'expert_*'
columns have been populated.

For challenges / directions / controversies:
  * overall accuracy (NOT_A_X counts as an error), extraction precision
    (1 - NOT_A_X fraction);
  * per-category precision, recall, F1, and support + macro average,
    confusion matrix (rows = domain expert);
  * none_check sheet: fraction of 'No X' items that the expert considered X
    (false negative estimate of extraction).
For research_type: error rate and per-class metrics.

Outputs: audit_outputs/06b_score_validation.{txt,json} + .tex tables.
"""
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from common import OUT_DIR, Report

VECTORS = {"challenges": "NOT_A_CHALLENGE", "directions": "NOT_A_DIRECTION",
           "controversies": "NOT_A_CONTROVERSY"}


def latex_table(rep_dict: dict, labels: list[str], caption: str) -> str:
    L = ["\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}} p{0.50\\linewidth} c c c c @{}}",
         "\\toprule",
         "\\textbf{Category} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1-Score} & \\textbf{Support} \\\\",
         "\\midrule"]
    for lab in labels:
        r = rep_dict[lab]
        L.append(f"{lab} & {r['precision']:.2f} & {r['recall']:.2f} & {r['f1-score']:.2f} & {int(r['support'])} \\\\")
    m = rep_dict["macro avg"]
    L += ["\\midrule",
          f"\\textbf{{Macro Average}} & \\textbf{{{m['precision']:.2f}}} & \\textbf{{{m['recall']:.2f}}} & "
          f"\\textbf{{{m['f1-score']:.2f}}} & \\textbf{{{int(m['support'])}}} \\\\",
          "\\bottomrule", "\\end{tabular*}", f"% {caption}"]
    return "\n".join(L)


def score_vector(vec: str, not_a: str, rep: Report) -> dict | None:
    path_xlsx = OUT_DIR / f"validation_{vec}.xlsx"
    val_dir = OUT_DIR.parent.parent / "Data" / "Validation_and_Audit"
    path_csv = val_dir / f"validation_{vec}_labels_only.csv"
    if path_xlsx.exists():
        try:
            a = pd.read_excel(path_xlsx, sheet_name="sample")
        except ValueError:
            a = pd.read_excel(path_xlsx, sheet_name=0)
        if "predicted_category" not in a.columns:   # BLIND mode
            gab_path = OUT_DIR / f"validation_{vec}_key.csv"
            g = pd.read_csv(gab_path)
            a = a.merge(g, on="sample_id")
    elif path_csv.exists():
        a = pd.read_csv(path_csv)
    else:
        return None
    a = a[a["expert_category"].notna() & (a["expert_category"].astype(str).str.strip() != "")]
    if a.empty:
        rep.p(f"{vec}: spreadsheet not yet filled")
        return None
    rep.h(f"{vec.upper()}: {len(a)} sentences reviewed")
    y_true = a["expert_category"].astype(str).str.strip()
    y_pred = a["predicted_category"].astype(str).str.strip()
    res = {"n_reviewed": len(a)}
    res["accuracy_overall"] = accuracy_score(y_true, y_pred)
    res["extraction_precision"] = 1 - (y_true == not_a).mean()
    rep.p(f"Overall accuracy (NOT_A counts as error): {res['accuracy_overall']:.3f}")
    rep.p(f"Extraction precision (sentence is genuinely a {vec[:-1]}): {res['extraction_precision']:.3f}")

    labels = sorted(set(y_pred) | (set(y_true) - {not_a}))
    cr = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    tab = pd.DataFrame({lab: cr[lab] for lab in labels}).T
    rep.table(tab, floatfmt="{:,.2f}")
    m = cr["macro avg"]
    rep.p(f"Macro: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1-score']:.3f}")
    res["per_category"] = tab.round(4).to_dict(orient="index")
    res["macro"] = {k: float(v) for k, v in m.items()}
    cm_labels = labels + ([not_a] if (y_true == not_a).any() else [])
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=cm_labels),
                      index=[f"true:{l}" for l in cm_labels], columns=[f"pred:{l}" for l in cm_labels])
    rep.p("Confusion matrix (rows = expert, columns = pipeline):")
    rep.table(cm, floatfmt="{:,.0f}")
    res["confusion_matrix"] = cm.to_dict()
    (OUT_DIR / f"06b_{vec}_metrics.tex").write_text(
        latex_table(cr, labels, f"{vec}: validation on {len(a)} sentences"), encoding="utf-8")

    if path_xlsx.exists():
        try:
            nc = pd.read_excel(path_xlsx, sheet_name="none_check")
            col = [c for c in nc.columns if c.startswith("expert_is_")][0]
            nc = nc[nc[col].notna() & (nc[col].astype(str).str.strip() != "")]
            if len(nc):
                fn = (nc[col].astype(str).str.upper().str.strip().isin(["SIM", "YES"])).mean()
                res["none_check_n"] = len(nc)
                res["none_check_false_negative_rate"] = float(fn)
                rep.p(f"none_check: {len(nc)} reviewed; {100 * fn:.1f}% were actually {vec[:-1]} (false 'No X')")
        except ValueError:
            pass
    return res


def main():
    rep = Report("06b_score_validation")
    for vec, not_a in VECTORS.items():
        r = score_vector(vec, not_a, rep)
        if r:
            rep.data[vec] = r

    path_xlsx = OUT_DIR / "validation_research_type.xlsx"
    val_dir = OUT_DIR.parent.parent / "Data" / "Validation_and_Audit"
    path_csv = val_dir / "validation_research_type_labels_only.csv"
    a = None
    if path_xlsx.exists():
        try:
            a = pd.read_excel(path_xlsx, sheet_name="sample")
        except ValueError:
            a = pd.read_excel(path_xlsx, sheet_name=0)
    elif path_csv.exists():
        a = pd.read_csv(path_csv)
    if a is not None:
        a = a[a["expert_research_type"].notna() & (a["expert_research_type"].astype(str).str.strip() != "")]
        if len(a):
            yt = a["expert_research_type"].astype(str).str.strip()
            yp = a["predicted_research_type"].astype(str).str.strip()
            err = (yt != yp).mean()
            rep.h(f"RESEARCH TYPE: {len(a)} documents reviewed")
            rep.p(f"Error rate: {100 * err:.1f}% | accuracy {100 * (1 - err):.1f}%")
            cr = classification_report(yt, yp, output_dict=True, zero_division=0)
            rep.table(pd.DataFrame(cr).T, floatfmt="{:,.2f}")
            rep.p(pd.crosstab(yt.rename("expert"), yp.rename("predicted")).to_string())
            rep.data["research_type"] = {"n": len(a), "error_rate": float(err), "report": cr}
    rep.save()


if __name__ == "__main__":
    main()
