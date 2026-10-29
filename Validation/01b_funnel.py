"""
01b_funnel.py -- record funnel: raw WoS export (129,789 records, 71 columns)
-> analyzed corpus (128,675 documents). Supports Section 3.1 of the paper.

Usage: python 01b_funnel.py /path/to/raw_export.csv
       (or define RAW_DATA_PATH in common.py)

Reports how many raw records lack a DOI, how many DOIs are duplicated, how many
were excluded from the final corpus, and the document type/year of excluded records.
"""
import sys

import pandas as pd

from common import RAW_DATA_PATH, Report, is_filled, load_main, resolve_columns


def main():
    raw_path = sys.argv[1] if len(sys.argv) > 1 else RAW_DATA_PATH
    if not raw_path:
        sys.exit("Please provide the raw export path: python 01b_funnel.py <file.csv>")
    rep = Report("01b_funnel")
    df = load_main()
    cols = resolve_columns(df, verbose=False)
    print(f"Reading raw export: {raw_path}")
    raw = pd.read_csv(raw_path)  # pd.read_parquet(raw_path)
    rc = resolve_columns(raw, verbose=False)
    f = {"raw_records": len(raw), "final_records": len(df)}

    doi = raw[rc["doi"]].astype("string").str.strip().str.lower()
    has = is_filled(raw[rc["doi"]])
    f["raw_without_doi"] = int((~has).sum())
    f["raw_duplicated_doi_extra_copies"] = int(doi[has].duplicated().sum())
    f["raw_unique_doi"] = int(doi[has].nunique())
    fin = set(df[cols["doi"]].astype("string").str.strip().str.lower().dropna())
    f["final_doi_found_in_raw"] = len(fin & set(doi[has]))
    f["final_doi_not_in_raw"] = len(fin - set(doi[has]))
    for key in ["abstract", "title", "year"]:
        if rc.get(key):
            f[f"raw_without_{key}"] = int((~is_filled(raw[rc[key]])).sum())

    ids = set(df[cols["doc_id"]].astype(str))
    out = raw[~raw[rc["doc_id"]].astype(str).isin(ids)]
    f["raw_records_not_in_final"] = len(out)
    f["excluded_without_doi"] = int((~is_filled(out[rc["doi"]])).sum())
    out_doi = out[rc["doi"]].astype("string").str.strip().str.lower()
    f["excluded_with_doi_already_in_final"] = int(out_doi.isin(fin).sum())
    f["excluded_by_doc_type"] = out[rc["doc_type"]].astype(str).value_counts().head(10).to_dict()
    f["excluded_by_year"] = pd.to_numeric(out[rc["year"]], errors="coerce").value_counts().sort_index().to_dict()
    f["excluded_by_language"] = out[rc["language"]].astype(str).value_counts().head(5).to_dict()

    rep.h("FUNNEL: RAW EXPORT -> ANALYZED CORPUS")
    for k, v in f.items():
        rep.p(f"  {k:36s} {v}")
    rep.data = f
    rep.save()


if __name__ == "__main__":
    main()
