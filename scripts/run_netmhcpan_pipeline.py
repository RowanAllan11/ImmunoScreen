from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.netmhcpan import (
    run_netmhcpan_for_k,
    DEFAULT_NETMHC_PATH,
    parse_netmhcpan_xls_wide_tsv,
    read_unique_peptides_tsv,
    read_peptide_variant_map_tsv,
    infer_kmer_range_from_df,
)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run netMHCpan, filter hits, and map peptides back to variants."
    )

    ap.add_argument("--peptides", type=Path, required=True)
    ap.add_argument("--peptide-map", type=Path, required=True)
    ap.add_argument("--alleles", type=Path, required=True)
    ap.add_argument("--kmers", type=int, nargs="+", required=True)

    ap.add_argument("--netmhcpan", type=Path, default=DEFAULT_NETMHC_PATH)
    ap.add_argument("--outdir", type=Path, default=REPO_ROOT / "data/output/netmhcpan",
    help="Base NetMHCpan output directory.",
)

    ap.add_argument("--el-rank-threshold", type=float, default=2.0)
    ap.add_argument("--dedup", action="store_true")
    ap.add_argument("--output-format", choices=["xls", "txt"], default="xls")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[])

    args = ap.parse_args()

    run_label = args.peptides.parent.name
    run_dir = args.outdir / run_label
    raw_dir = run_dir / "raw"

    raw_dir.mkdir(parents=True, exist_ok=True)

    # 1. Run netMHCpan for each k-mer
    raw_outputs = []

    for k in args.kmers:
        out_path = run_netmhcpan_for_k(
            peptides_tsv=args.peptides,
            k=k,
            alleles_path=args.alleles,
            netmhcpan_path=args.netmhcpan,
            outdir=raw_dir,
            output_format=args.output_format,
            extra=list(args.extra),
        )
        raw_outputs.append(out_path)
        print(f"Wrote raw netMHCpan output: {out_path}")

    # 2. Read mapping files
    uniq = read_unique_peptides_tsv(args.peptides)
    pmap = read_peptide_variant_map_tsv(args.peptide_map)

    # 3. Parse all netMHCpan outputs
    dfs = []
    for xls_path in raw_outputs:
        nm = parse_netmhcpan_xls_wide_tsv(xls_path)
        dfs.append(nm)

    nm_all = pd.concat(dfs, ignore_index=True)

    # 4. Filter by EL rank
    nm_all["netMHCpan_EL_rank"] = pd.to_numeric(
        nm_all["netMHCpan_EL_rank"],
        errors="coerce",
    )

    nm_all["netMHCpan_EL_rank_pass"] = (
        nm_all["netMHCpan_EL_rank"].notna()
        & (
            nm_all["netMHCpan_EL_rank"]
            < float(args.el_rank_threshold)
        )
    )

    # 5. Join to peptide IDs
    merged = nm_all.merge(
        uniq,
        on=["peptide", "k"],
        how="left",
        validate="many_to_one",
    )

    if merged["peptide_id"].isna().any():
        n_missing = int(merged["peptide_id"].isna().sum())
        raise ValueError(
            f"{n_missing} NetMHCpan rows could not be mapped to peptide_id. "
            "Ensure the predictions were generated from this exact "
            "unique_peptides.tsv."
        )

    # 6. Fan out to variant occurrences
    out = merged.merge(
        pmap,
        on="peptide_id",
        how="left",
        validate="many_to_many",
    )

    if out["variant_id"].isna().any():
        n_missing = int(out["variant_id"].isna().sum())
        raise ValueError(
            f"{n_missing} rows missing variant mapping after join."
        )

    out["length"] = out["k"]

    # 7. Optional deduplication
    if args.dedup:
        dedup_keys = ["allele", "peptide_id", "variant_id", "start", "end"]
        out = (
            out.sort_values("netMHCpan_EL_rank", ascending=True)
            .drop_duplicates(dedup_keys, keep="first")
        )

    # 8. Choose output columns
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

    meta_cols = [
        c for c in pmap.columns
        if c not in {"peptide_id", "variant_id", "start", "end"}
    ]

    score_cols = [
        "netMHCpan_EL_score",
        "netMHCpan_EL_rank",
        "netMHCpan_EL_rank_pass",
    ]

    for optional_col in ["netMHCpan_BA_score", "netMHCpan_BA_rank"]:
        if optional_col in out.columns:
            score_cols.append(optional_col)

    out_cols = base_cols + meta_cols + score_cols

    for c in out_cols:
        if c not in out.columns:
            out[c] = pd.NA

    out = (
        out[out_cols]
        .sort_values(["allele", "netMHCpan_EL_rank", "variant_id", "peptide_id"])
        .reset_index(drop=True)
    )

    # 9. Write final output
    out_path = run_dir / "predictions_mapped.tsv"

    out.to_csv(out_path, sep="\t", index=False)

    n_pass = int(out["netMHCpan_EL_rank_pass"].sum())
    n_total = len(out)

    print(f"Wrote mapped NetMHCpan output: {out_path}")
    print(
        f"Passing EL-rank threshold: "
        f"{n_pass:,}/{n_total:,} rows "
        f"(threshold < {args.el_rank_threshold})"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())



"""
python -m scripts.run_netmhcpan_pipeline \
  --peptides data/output/fragmentation/VR5_V3__k9/unique_peptides.tsv \
  --peptide-map data/output/fragmentation/VR5_V3__k9/peptide_variant_map.tsv \
  --alleles data/input/alleles/netmhcpan/allele_single.txt \
  --kmers 9 \
  --el-rank-threshold 2.0 \
  --dedup
"""
