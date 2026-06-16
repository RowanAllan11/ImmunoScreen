from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.bigmhc import create_bigmhc_input_csv, _read_bigmhc_output

BIGMHC_PREDICT = REPO_ROOT / "tools" / "bigmhc" / "src" / "predict.py"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Use BigMHC to predict immunogenicity for peptides in a combined TSV and merge scores back."
    )
    ap.add_argument("--i", type=Path, required=True, help="Input combined TSV (must contain 'peptide' column).")
    ap.add_argument("--outdir", type=Path, default=REPO_ROOT / "data/output/bigmhc")
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

    combined_tsv = args.i.resolve()

    if not combined_tsv.is_file():
        raise FileNotFoundError(f"Input file not found: {combined_tsv}")
    
    run_label = combined_tsv.parent.name

    if not BIGMHC_PREDICT.exists():
        raise FileNotFoundError(f"BigMHC predict.py not found at: {BIGMHC_PREDICT}")
    
    run_dir = args.outdir / run_label
    run_dir.mkdir(parents=True, exist_ok=True)

    out_path = run_dir / "predictions_mapped.tsv"

    bigmhc_in = run_dir / "input.csv"
    n = create_bigmhc_input_csv(combined_tsv, bigmhc_in, args.default_tgt)
    if n == 0:    
        raise RuntimeError(
        f"No valid peptide–allele pairs found in: {combined_tsv}"
    )

    print(
        f"Wrote BigMHC input: {bigmhc_in} "
        f"({n:,} unique peptide–allele pairs)"
    )

    python = sys.executable
    pred_csv = run_dir / "predictions.csv"

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

    combined = pd.read_csv(
        combined_tsv,
        sep="\t",
        dtype=str,
    )

    required_combined_cols = {"allele", "peptide"}
    missing = required_combined_cols - set(combined.columns)

    if missing:
        raise ValueError(
            f"Combined TSV missing required columns: {sorted(missing)}"
        )

    scores, used_col = _read_bigmhc_output(
        pred_csv,
        args.score_col,
    )

    score_keys = ["allele", "peptide"]

    duplicate_mask = scores.duplicated(score_keys, keep=False)

    if duplicate_mask.any():
        examples = scores.loc[
            duplicate_mask,
            score_keys + [used_col],
        ].head(10)

        raise ValueError(
            "BigMHC output contains duplicate peptide–allele predictions. "
            "This would create duplicate rows during the merge.\n"
            f"Examples:\n{examples.to_string(index=False)}"
        )

    merged = combined.merge(
        scores[score_keys + [used_col]],
        on=score_keys,
        how="left",
        validate="many_to_one",
    )

    n_missing = int(merged[used_col].isna().sum())

    if n_missing:
        n_missing_pairs = (
            merged.loc[
                merged[used_col].isna(),
                score_keys,
            ]
            .drop_duplicates()
            .shape[0]
        )

        print(
            f"Warning: {n_missing:,} rows across "
            f"{n_missing_pairs:,} peptide–allele pairs "
            "did not receive a BigMHC score."
        )

    merged.to_csv(
        out_path,
        sep="\t",
        index=False,
    )

    print(
        f"Wrote mapped BigMHC output with score column "
        f"'{used_col}': {out_path}"
    )
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


"""
python scripts/run_bigmhc.py \
  --i data/output/combined/VR5_V3__k9/combined_annotated.tsv
"""