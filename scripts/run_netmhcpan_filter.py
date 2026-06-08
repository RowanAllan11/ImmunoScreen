from __future__ import annotations

import argparse
import pandas as pd
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# make repo and src importable
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from src.netmhcpan import parse_netmhcpan_xls_wide_tsv, read_unique_peptides_tsv, read_peptide_variant_map_tsv, infer_kmer_range_from_df


def main() -> int:
    ap = argparse.ArgumentParser(description="Filter netMHCpan outputs and join variant positions (new tabular fragmentation).")

    ap.add_argument(
        "--netmhcpan-xls",
        type=Path,
        nargs="+",
        required=True,
        help="One or more netMHCpan .xls (tab-delimited) outputs (can be multiple k-mer runs).",
    )
    ap.add_argument("--peptides", type=Path, required=True, help="Path to unique_peptides.tsv")
    ap.add_argument("--peptide-map", type=Path, required=True, help="Path to peptide_variant_map.tsv")

    ap.add_argument("-o", "--outdir", type=Path, required=True, help="Output directory")
    ap.add_argument("--st", required=True, help="ST tag for output filename stem (e.g. VR5)")
    ap.add_argument(
        "--combined-name",
        default=None,
        help="Optional explicit filename stem for combined output (no extension).",
    )

    ap.add_argument("--el-rank-threshold", type=float, default=2.0, help="Keep rows with EL_rank < threshold")
    ap.add_argument("--dedup", action="store_true", help="Deduplicate identical (allele, peptide_id, variant_id, start, end) keeping best rank")

    args = ap.parse_args()

    uniq = read_unique_peptides_tsv(args.peptides)
    pmap = read_peptide_variant_map_tsv(args.peptide_map)

    dfs: list[pd.DataFrame] = []
    for xls_path in args.netmhcpan_xls:
        nm = parse_netmhcpan_xls_wide_tsv(xls_path)
        dfs.append(nm)

    nm_all = pd.concat(dfs, ignore_index=True)

    # Filter first (reduces fan-out)
    nm_all["netMHCpan_EL_rank"] = pd.to_numeric(nm_all["netMHCpan_EL_rank"], errors="coerce")
    nm_all = nm_all.dropna(subset=["netMHCpan_EL_rank"])
    nm_all = nm_all[nm_all["netMHCpan_EL_rank"] < float(args.el_rank_threshold)].copy()

    # Map (peptide,k) -> peptide_id, occurrence_count
    merged = nm_all.merge(
        uniq,
        on=["peptide", "k"],
        how="left",
        validate="many_to_one",
    )
    n_missing = int(merged["peptide_id"].isna().sum())
    if n_missing:
        raise ValueError(
            f"{n_missing} filtered netMHCpan rows could not be mapped to peptide_id via (peptide,k). "
            "Ensure the netMHCpan input peptides came from this exact unique_peptides.tsv."
        )

    # Fan-out to variant occurrences with positional + metadata
    out = merged.merge(pmap, on="peptide_id", how="left", validate="many_to_many")

    n_missing_var = int(out["variant_id"].isna().sum())
    if n_missing_var:
        raise ValueError(
            f"{n_missing_var} rows missing variant mapping after join. "
            "Check peptide_variant_map.tsv matches unique_peptides.tsv."
        )

    out["length"] = out["k"]

    # Optional dedup
    if args.dedup:
        dedup_keys = ["allele", "peptide_id", "variant_id", "start", "end"]
        out = out.sort_values(["netMHCpan_EL_rank"], ascending=True).drop_duplicates(dedup_keys, keep="first")

    # Output columns similar to mhcflurry output you showed.
    # criteria is included if present in peptide map.
    base_cols = [
        "allele",
        "peptide_id",
        "peptide",
        "k",
        "variant_id",
        "start",
        "end",
        "length",
        "occurrence_count",
    ]
    meta_cols = [c for c in pmap.columns if c not in {"peptide_id", "variant_id", "start", "end"}]
    score_cols = ["netMHCpan_EL_score", "netMHCpan_EL_rank"]
    if "netMHCpan_BA_score" in out.columns:
        score_cols.append("netMHCpan_BA_score")
    if "netMHCpan_BA_rank" in out.columns:
        score_cols.append("netMHCpan_BA_rank")

    out_cols = base_cols + meta_cols + score_cols
    for c in out_cols:
        if c not in out.columns:
            out[c] = pd.NA

    out = out[out_cols].sort_values(["allele", "netMHCpan_EL_rank", "variant_id", "peptide_id"]).reset_index(drop=True)

    kmer_tag = infer_kmer_range_from_df(out)
    title = args.combined_name or f"{args.st}_{kmer_tag}_netMHCpan"

    args.outdir.mkdir(parents=True, exist_ok=True)
    out_path = args.outdir / f"{title}.tsv"
    out.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote: {out_path} ({len(out)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


"""
python -m src.netmhc_filter \
  --netmhcpan-xls data/output/netmhcpan/VR5_9mer/variants_vr5_9_9mer.netmhcpan.xls \
  --peptides data/output/fragmentation/variants_vr5_9/unique_peptides.tsv \
  --peptide-map data/output/fragmentation/variants_vr5_9/peptide_variant_map.tsv \
  --el-rank-threshold 2.0 \
  --st VR5 \
  -o data/output/netmhcpan_filtered
"""