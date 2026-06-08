from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.netmhcpan import _read_alleles
from src.mhcflurry import (
    _write_unique_peptides_allele_input,
    run_mhcflurry_predict,
    filter_and_expand_mhcflurry_predictions,
)

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run MHCflurry, filter predictions, and expand to variant rows."
    )

    ap.add_argument("--unique-peptides", type=Path, required=True)
    ap.add_argument("--peptide-map", type=Path, required=True)
    ap.add_argument("--alleles", type=Path, required=True)

    ap.add_argument("--outdir", type=Path, default=REPO_ROOT / "data/output")
    ap.add_argument("--tag", type=str, required=True)

    ap.add_argument("--mhcflurry-predict", default="mhcflurry-predict")
    ap.add_argument("--affinity-percentile-threshold", type=float, default=2.0)
    ap.add_argument("--no-flanking", action="store_true", default=True)
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[])

    args = ap.parse_args()

    raw_dir = args.outdir / "mhcflurry" / args.tag
    filtered_dir = args.outdir / "mhcflurry_filtered" / args.tag

    raw_dir.mkdir(parents=True, exist_ok=True)
    filtered_dir.mkdir(parents=True, exist_ok=True)

    alleles = _read_alleles(args.alleles)

    mhc_input_csv = raw_dir / f"{args.tag}.unique_peptides.input.csv"
    raw_mhcflurry_tsv = raw_dir / f"{args.tag}.unique_peptides.mhcflurry.tsv"
    filtered_tsv = filtered_dir / f"{args.tag}_MHCflurry.tsv"

    n = _write_unique_peptides_allele_input(
        unique_peptides_tsv=args.unique_peptides,
        alleles=alleles,
        out_csv=mhc_input_csv,
    )

    if n == 0:
        raise RuntimeError(
            f"0 peptide/allele rows written from: {args.unique_peptides}"
        )

    print(f"Wrote MHCflurry input: {mhc_input_csv} ({n} rows)")

    run_mhcflurry_predict(
        mhcflurry_predict=args.mhcflurry_predict,
        mhc_in=mhc_input_csv,
        out_path=raw_mhcflurry_tsv,
        no_flanking=args.no_flanking,
        extra=list(args.extra),
    )

    print(f"Wrote raw MHCflurry output: {raw_mhcflurry_tsv}")

    out_df = filter_and_expand_mhcflurry_predictions(
        unique_peptides_tsv=args.unique_peptides,
        peptide_map_tsv=args.peptide_map,
        mhcflurry_tsv=raw_mhcflurry_tsv,
        affinity_percentile_threshold=args.affinity_percentile_threshold,
        alleles_path=args.alleles,
    )

    out_df.to_csv(filtered_tsv, sep="\t", index=False)

    print(f"Wrote filtered MHCflurry output: {filtered_tsv} ({len(out_df)} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


"""
python scripts/run_mhcflurry_pipeline.py \
  --unique-peptides data/output/fragmentation/variants_vr5_9/unique_peptides.tsv \
  --peptide-map data/output/fragmentation/variants_vr5_9/peptide_variant_map.tsv \
  --alleles data/input/alleles/mhcflurry/allele_single.txt \
  --tag VR5_v3_9mer \
  --affinity-percentile-threshold 2.0 \
  --outdir data/output
"""
