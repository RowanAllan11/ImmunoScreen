from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

BIGMHC_PREDICT = REPO_ROOT / "tools" / "bigmhc" / "src" / "predict.py"


def create_bigmhc_input_csv(
    combined_tsv: Path,
    out_csv: Path,
    default_tgt: int = 0,
) -> int:
    """
    Create a BigMHC input CSV with columns: mhc, pep, tgt.

    Uses allele-specific peptide-MHC pairs from the combined/filter TSV:
      allele  -> mhc
      peptide -> pep

    Returns number of unique peptide-MHC pairs written.
    """
    combined = pd.read_csv(combined_tsv, sep="\t")

    required_cols = {"allele", "peptide"}
    missing = required_cols - set(combined.columns)
    if missing:
        raise ValueError(
            f"Input TSV missing required columns {sorted(missing)}: {combined_tsv}"
        )

    bigmhc_input = (
        combined[["allele", "peptide"]]
        .dropna()
        .astype(str)
        .assign(
            allele=lambda d: d["allele"].str.strip(),
            peptide=lambda d: d["peptide"].str.strip(),
        )
    )

    bigmhc_input = bigmhc_input[
        (bigmhc_input["allele"] != "") &
        (bigmhc_input["peptide"] != "")
    ]

    bigmhc_input = (
        bigmhc_input
        .drop_duplicates()
        .rename(columns={"allele": "mhc", "peptide": "pep"})
    )

    bigmhc_input["tgt"] = int(default_tgt)

    bigmhc_input = bigmhc_input[["mhc", "pep", "tgt"]]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    bigmhc_input.to_csv(out_csv, index=False)

    return len(bigmhc_input)


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
    
    if "mhc" not in df.columns:
        raise ValueError(
            f"BigMHC output missing 'mhc' column: {pred_csv} "
            f"(cols={list(df.columns)})"
        )

    out = df[["mhc", "pep", use_col]].copy()
    out.rename(
        columns={
            "mhc": "allele",
            "pep": "peptide",
        },
        inplace=True,
    )
    out = out.groupby(
        ["allele", "peptide"],
        as_index=False,
    )[use_col].max()

    return out, use_col