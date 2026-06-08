from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.bigmhc import create_bigmhc_input_csv, _infer_bigmhc_score_col, _read_bigmhc_output

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

BIGMHC_PREDICT = REPO_ROOT / "tools" / "bigmhc" / "src" / "predict.py"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Use BigMHC to predict immunogenicity for peptides in a combined TSV and merge scores back."
    )
    ap.add_argument("--i", type=Path, required=True, help="Input combined TSV (must contain 'peptide' column).")
    ap.add_argument("--out", type=Path, default=None, help="Output TSV path. Default: <input>.bigmhc.tsv")
    ap.add_argument("--m", type=str, default="el", help="BigMHC -m (example: el). Default: el")
    ap.add_argument("--t", type=int, default=2, help="BigMHC -t. Default: 2")
    ap.add_argument("--d", type=str, default="cpu", help='BigMHC -d (e.g. "cpu" or "cuda"). Default: cpu')
    ap.add_argument(
        "--score-col",
        type=str,
        default=None,
        help="Score column to merge back (e.g. BigMHC_IM or BigMHC_EL). Default: auto-detect BigMHC_*",
    )

    ap.add_argument(
        "--default-tgt",
        type=int,
        default=0,
        help="Dummy target label for BigMHC input 'tgt' column. Default: 0",
    )
    args = ap.parse_args()

    combined_tsv = args.i
    if not combined_tsv.exists():
        raise FileNotFoundError(f"Input file not found: {combined_tsv}")

    if not BIGMHC_PREDICT.exists():
        raise FileNotFoundError(f"BigMHC predict.py not found at: {BIGMHC_PREDICT}")

    out_path = args.out or combined_tsv.with_suffix(combined_tsv.suffix + "_overall.tsv")

    bigmhc_in = REPO_ROOT / "data" / "input" / "bigmhc" / f"{combined_tsv.stem}.bigmhc_input.csv"
    n = create_bigmhc_input_csv(combined_tsv, bigmhc_in, args.default_tgt)
    if n == 0:
        raise RuntimeError(f"No peptides found in 'peptide' column of {combined_tsv}")
    print(f"Wrote BigMHC input peptides: {bigmhc_in} ({n} unique peptides)")

    python = "python"
    bigmhc_out = REPO_ROOT / "data" / "output" / "bigmhc"
    bigmhc_out.mkdir(parents=True, exist_ok=True)
    pred_csv = bigmhc_out / f"{combined_tsv.stem}.bigmhc_predictions.csv"

    cmd = [
        python,
        str(BIGMHC_PREDICT),
        f"-i={bigmhc_in}",
        f"-m={args.m}",
        f"-t={args.t}",
        f"-d={args.d}",
        f"-o={pred_csv}",
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"Wrote BigMHC predictions: {pred_csv}")

    combined = pd.read_csv(combined_tsv, sep="\t")
    scores, used_col = _read_bigmhc_output(pred_csv, args.score_col)
    merged = combined.merge(scores, on="peptide", how="left")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote merged TSV with BigMHC scores column '{used_col}': {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())