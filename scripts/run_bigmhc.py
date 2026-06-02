from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

BIGMHC_PREDICT = REPO_ROOT / "tools" / "bigmhc" / "src" / "predict.py"


def _require_python() -> str:
    exe = shutil.which("python") or shutil.which("python3")
    if not exe:
        raise RuntimeError("Python executable not found on PATH (python/python3).")
    return exe


def create_bigmhc_input_csv(combined_tsv: Path, out_csv: Path, default_mhc: str, default_tgt: int) -> int:
    """
    Create a BigMHC input CSV with columns: mhc, pep, tgt.
    Returns number of unique peptides written.
    """
    combined = pd.read_csv(combined_tsv, sep="\t")
    if "peptide" not in combined.columns:
        raise ValueError(f"Input TSV missing required column 'peptide': {combined_tsv}")

    peptides = (
        combined["peptide"]
        .astype(str)
        .str.strip()
        .replace({"": pd.NA})
        .dropna()
        .drop_duplicates()
        .tolist()
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["mhc", "pep", "tgt"])
        for pep in peptides:
            writer.writerow([default_mhc, pep, int(default_tgt)])

    return len(peptides)


def _infer_bigmhc_score_col(df: pd.DataFrame) -> str:
    bigmhc_cols = [c for c in df.columns if c.startswith("BigMHC_")]
    if not bigmhc_cols:
        raise ValueError(f"No BigMHC_* score columns found in output (cols={list(df.columns)})")
    if len(bigmhc_cols) == 1:
        return bigmhc_cols[0]
    raise ValueError(
        f"Multiple BigMHC_* score columns found {bigmhc_cols}. Please pass --score-col explicitly."
    )


def _read_bigmhc_output(pred_csv: Path, score_col: str | None) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(pred_csv)
    if "pep" not in df.columns:
        raise ValueError(f"BigMHC output missing 'pep' column: {pred_csv} (cols={list(df.columns)})")

    use_col = score_col
    if use_col is None:
        use_col = _infer_bigmhc_score_col(df)
    elif use_col not in df.columns:
        # Helpful error message
        inferred = [c for c in df.columns if c.startswith("BigMHC_")]
        raise ValueError(
            f"BigMHC output missing score column '{use_col}': {pred_csv} "
            f"(available BigMHC cols={inferred}, all cols={list(df.columns)})"
        )

    out = df[["pep", use_col]].copy()
    out.rename(columns={"pep": "peptide"}, inplace=True)
    out = out.groupby("peptide", as_index=False)[use_col].max()
    return out, use_col


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
        "--default-mhc",
        type=str,
        default="HLA-A*02:01",
        help="Placeholder allele for BigMHC input 'mhc' column. Default: HLA-A*02:01",
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

    out_path = args.out or combined_tsv.with_suffix(combined_tsv.suffix + ".bigmhc.tsv")

    bigmhc_in = REPO_ROOT / "data" / "input" / "bigmhc" / f"{combined_tsv.stem}.bigmhc_input.csv"
    n = create_bigmhc_input_csv(combined_tsv, bigmhc_in, args.default_mhc, args.default_tgt)
    if n == 0:
        raise RuntimeError(f"No peptides found in 'peptide' column of {combined_tsv}")
    print(f"Wrote BigMHC input peptides: {bigmhc_in} ({n} unique peptides)")

    python = _require_python()
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