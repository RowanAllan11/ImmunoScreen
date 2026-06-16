from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.netmhcpan import _read_alleles
from src.mhcflurry import _write_unique_peptides_allele_input, run_mhcflurry_predict


def main() -> int:
    ap = argparse.ArgumentParser(description="Run mhcflurry-predict on fragmented peptides.")
    ap.add_argument("--peptides", type=Path, required=True, help="Path to unique_peptides.tsv from scripts/run_fragmentation.py")
    ap.add_argument("--alleles", type=Path, required=True, help="Alleles .txt (one allele per line)")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "data/output/mhcflurry",
        help="Output directory (default: data/output/mhcflurry)",
    )

    args = ap.parse_args()

    unique_peptides_tsv = args.peptides
    if not unique_peptides_tsv.exists():
        raise FileNotFoundError(f"unique_peptides.tsv not found: {unique_peptides_tsv}")

    alleles = _read_alleles(args.alleles)

    run_label = unique_peptides_tsv.parent.name

    if not run_label:
        raise ValueError(
            f"Could not derive a run label from: {unique_peptides_tsv}"
        )

    run_out_dir = args.out_dir / run_label
    run_out_dir.mkdir(parents=True, exist_ok=True)

    mhc_input_path = run_out_dir / "input.csv"
    predictions_path = run_out_dir / "predictions.tsv"



    n_rows = _write_unique_peptides_allele_input(
        unique_peptides_tsv=unique_peptides_tsv,
        alleles=alleles,
        out_csv=mhc_input_path,
    )
    if n_rows == 0:
        raise RuntimeError(f"0 (peptide, allele) rows written from: {unique_peptides_tsv}")

    mhcflurry_predict = "mhcflurry-predict"


    # Run mhcflurry-predict
    run_mhcflurry_predict(mhcflurry_predict, mhc_input_path, predictions_path)

    print(f"Run label: {run_label}")
    print(f"Peptide–allele rows: {n_rows:,}")
    print(f"Wrote MHCflurry input: {mhc_input_path}")
    print(f"Wrote MHCflurry predictions: {predictions_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
python scripts/run_mhcflurry.py \
    --peptides data/output/fragmentation/VR5_V3__k9/unique_peptides.tsv \
    --alleles data/input/alleles/mhcflurry/alleles_h2.txt
"""
