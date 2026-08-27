"""Shared command-line, naming, validation, and output helpers for scorers."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def parse_scoring_args(
    *,
    description: str,
    default_output_root: str,
    variant_help: str,
    wt_help: str | None = None,
) -> argparse.Namespace:
    """Build the common scoring CLI, optionally requiring a WT input."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--variant-input", type=Path, required=True, help=variant_help
    )
    if wt_help is not None:
        parser.add_argument("--wt-input", type=Path, required=True, help=wt_help)
    parser.add_argument(
        "--output-root", type=Path, default=Path(default_output_root)
    )
    parser.add_argument(
        "--run-label",
        help="Output label. Default: inferred from the variant input directory.",
    )
    return parser.parse_args()


def create_output_run_label(input_file: Path) -> str:
    """Convert a workflow directory such as VR6__k9 to VR6_K9."""
    return re.sub(
        r"__k(\d+)$",
        lambda match: f"_K{match.group(1)}",
        input_file.parent.name,
        flags=re.IGNORECASE,
    )


def clean_allele_name(allele: str) -> str:
    """Return a consistent filesystem-safe allele label."""
    value = str(allele).strip().replace("H-2", "H2")
    return re.sub(r"[^A-Za-z0-9_.-]+", "", value)


def convert_to_boolean(series: pd.Series) -> pd.Series:
    """Convert common boolean encodings; missing/unknown values become False."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False)
        .astype(bool)
    )


def count_mutations(value: object) -> int:
    """Count semicolon-separated substitutions, treating blank/WT as zero."""
    if pd.isna(value) or str(value).strip().upper() in {"", "WT"}:
        return 0
    return len([part for part in str(value).split(";") if part.strip()])


def save_allele_specific_outputs(
    allele_scores: pd.DataFrame,
    output_root: Path,
    run_label: str,
) -> list[Path]:
    """Write one variant score table per allele and return the paths."""
    output_files: list[Path] = []
    for allele, allele_df in allele_scores.groupby(
        "allele", observed=True, sort=True
    ):
        output_dir = output_root / f"{run_label}_{clean_allele_name(allele)}"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "variant_immunogenicity_scores.tsv"
        allele_df.to_csv(output_file, sep="\t", index=False)
        output_files.append(output_file)
    return output_files
