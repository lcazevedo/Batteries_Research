"""
00_inspect_columns.py -- lists the columns of both datasets and shows the
automatic mapping used by the remaining scripts.

Generates:
  audit_outputs/00_inspect_columns.txt / .json
  column_map.json  (only if it does not exist yet) -> edit if any mapping is
                    wrong or empty; the other scripts read it first.

(Does not contain raw titles or abstracts: long text fields are truncated to 60 characters.)
"""
import json
from pathlib import Path

import pandas as pd

from common import (COLUMN_MAP_FILE, PHRASES_PATH, Report, load_main, resolve_columns,
                    topic_layer_columns)

LONG_TEXT_HINTS = ("abstract", "title", "phrase", "sentence", "cited references", "references")


def describe(df: pd.DataFrame, rep: Report):
    rows = []
    for c in df.columns:
        s = df[c]
        ex = s.dropna()
        ex = ex.iloc[0] if len(ex) else None
        ex_txt = str(ex)
        if any(h in str(c).lower() for h in LONG_TEXT_HINTS):
            ex_txt = f"<text, {len(ex_txt)} chars>"
        elif len(ex_txt) > 60:
            ex_txt = ex_txt[:57] + "..."
        rows.append({"coluna": c, "dtype": str(s.dtype), "nao_nulos": int(s.notna().sum()),
                     "n_unicos": int(s.astype(str).nunique()) if len(s) < 3_000_000 else None,
                     "exemplo": ex_txt})
    t = pd.DataFrame(rows)
    rep.table(t, max_rows=500)
    return rows


def main():
    rep = Report("00_inspect_columns")
    df = load_main()
    rep.h(f"MAIN DATASET: {len(df):,} rows, {df.shape[1]} columns")
    rep.data["main_columns"] = describe(df, rep)

    cols = resolve_columns(df)
    rep.h("AUTOMATIC MAPPING (logical key -> column)")
    for k, v in cols.items():
        rep.p(f"  {k:20s} -> {v}")
    rep.data["resolved"] = cols
    layers = topic_layer_columns(df)
    rep.p(f"\n  detected topic layers: {layers}")
    for c in layers:
        rep.p(f"    {c}: {df[c].nunique()} distinct values")
    rep.data["topic_layers"] = {c: int(df[c].nunique()) for c in layers}

    if Path(PHRASES_PATH).exists():
        ph = pd.read_parquet(PHRASES_PATH)
        rep.h(f"PHRASES DATASET: {len(ph):,} rows, {ph.shape[1]} columns")
        rep.data["phrases_columns"] = describe(ph, rep)
        common_cols = sorted(set(ph.columns) & set(df.columns))
        rep.p(f"\n  columns in common with main dataset (join key candidates): {common_cols}")
        rep.data["phrases_common_columns"] = common_cols
    else:
        rep.p(f"\n[warning] phrases dataset not found at {PHRASES_PATH}")

    if not COLUMN_MAP_FILE.exists():
        with open(COLUMN_MAP_FILE, "w", encoding="utf-8") as f:
            json.dump(cols, f, ensure_ascii=False, indent=2)
        rep.p(f"\nTemplate written to {COLUMN_MAP_FILE} -- edit null/incorrect values if needed.")
    rep.save()


if __name__ == "__main__":
    main()
