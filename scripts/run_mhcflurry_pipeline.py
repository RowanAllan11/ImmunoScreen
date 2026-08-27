from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.mhcflurry import (
    _write_unique_peptides_allele_input,
    run_mhcflurry_predict,
    filter_and_expand_mhcflurry_predictions,
)

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run MHCflurry, filter predictions, and expand to variant rows."
    )

    ap.add_argument("--peptides", type=Path, required=True)
    ap.add_argument("--peptide-map", type=Path, required=True)
    ap.add_argument("--alleles", nargs="+", required=True, help="MHCflurry allele names")

    ap.add_argument("--outdir", type=Path, default=REPO_ROOT / "data/output/mhcflurry")

    ap.add_argument("--mhcflurry-predict", default="mhcflurry-predict")
    ap.add_argument("--affinity-percentile-threshold", type=float, default=2.0)
    ap.add_argument("--no-flanking", action="store_true", default=True)
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[])

    args = ap.parse_args()

    peptides_tsv = args.peptides.resolve()
    peptide_map_tsv = args.peptide_map.resolve()

    run_label = peptides_tsv.parent.name

    run_dir = args.outdir / run_label
    run_dir.mkdir(parents=True, exist_ok=True)

    mhc_input_csv = run_dir / "input.csv"
    raw_mhcflurry_tsv = run_dir / "predictions.tsv"
    mapped_tsv = run_dir / "predictions_mapped.tsv"

    alleles = [
        allele.strip()
        for allele in args.alleles
        if allele.strip()
    ]


    n_input_rows = _write_unique_peptides_allele_input(
        unique_peptides_tsv=peptides_tsv,
        alleles=alleles,
        out_csv=mhc_input_csv,
    )

    if n_input_rows == 0:
        raise RuntimeError(
            f"Zero peptide–allele rows were written from: "
            f"{peptides_tsv}"
        )

    print(f"Run label: {run_label}")
    print(
        f"Wrote MHCflurry input: {mhc_input_csv} "
        f"({n_input_rows:,} rows)"
    )

    run_mhcflurry_predict(
        mhcflurry_predict=args.mhcflurry_predict,
        mhc_in=mhc_input_csv,
        out_path=raw_mhcflurry_tsv,
        no_flanking=args.no_flanking,
        extra=list(args.extra),
    )

    print(f"Wrote raw MHCflurry predictions: {raw_mhcflurry_tsv}")

    out_df = filter_and_expand_mhcflurry_predictions(
        unique_peptides_tsv=peptides_tsv,
        peptide_map_tsv=peptide_map_tsv,
        mhcflurry_tsv=raw_mhcflurry_tsv,
        affinity_percentile_threshold=args.affinity_percentile_threshold,
        alleles=alleles,
    )

    out_df.to_csv(
        mapped_tsv,
        sep="\t",
        index=False,
    )

    pass_col = "MHCflurry_affinity_percentile_pass"
    n_pass = int(out_df[pass_col].fillna(False).sum())

    print(f"Wrote mapped MHCflurry predictions: {mapped_tsv}")
    print(
        f"Passing affinity-percentile threshold: "
        f"{n_pass:,}/{len(out_df):,} rows "
        f"(< {args.affinity_percentile_threshold})"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
