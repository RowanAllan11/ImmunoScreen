from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from netmhcpan import _read_alleles
from mhcflurry import _write_unique_peptides_allele_input, run_mhcflurry_predict


def main() -> int:
    ap = argparse.ArgumentParser(description="Run mhcflurry-predict on fragmented peptides.")
    ap.add_argument("--unique-peptides", type=Path, required=True, help="Path to unique_peptides.tsv from scripts/run_fragmentation.py --tabular")
    ap.add_argument("--alleles", type=Path, required=True, help="Alleles .txt (one allele per line)")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "data/output/mhcflurry",
        help="Output directory (default: data/output/mhcflurry)",
    )
    ap.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Output tag/prefix for filenames (default: parent directory name of unique-peptides)",
    )
    args = ap.parse_args()

    unique_peptides_tsv = args.unique_peptides
    if not unique_peptides_tsv.exists():
        raise FileNotFoundError(f"unique_peptides.tsv not found: {unique_peptides_tsv}")

    alleles = _read_alleles(args.alleles)
    mhcflurry_predict = "mhcflurry-predict"

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = args.tag or unique_peptides_tsv.parent.name
    mhc_in = out_dir / f"{tag}.unique_peptides.input.csv"
    out_path = out_dir / f"{tag}.unique_peptides.mhcflurry.tsv"

    n = _write_unique_peptides_allele_input(unique_peptides_tsv, alleles, mhc_in)
    if n == 0:
        raise RuntimeError(f"0 (peptide, allele) rows written from: {unique_peptides_tsv}")

    # delegate to helper
    out_path = run_mhcflurry_predict(mhcflurry_predict, mhc_in, out_path)

    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
python scripts/run_mhcflurry.py \
  --unique-peptides data/output/fragmentation/variants_vr5_9/unique_peptides.tsv \
  --alleles data/input/alleles/mhcflurry/alleles_h2.txt \
  --out-dir data/output/mhcflurry/VR5_9mer \
  --tag VR5_v3_9mer
"""