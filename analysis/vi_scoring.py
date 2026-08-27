# Variant-level immunogenicity scoring
#
# Creates one variant-level score table per allele.

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from analysis.scoring_common import (
        count_mutations,
        create_output_run_label,
        convert_to_boolean,
        parse_scoring_args,
        save_allele_specific_outputs,
    )
except ModuleNotFoundError:  # Direct execution: python analysis/vi_scoring.py
    from scoring_common import (
        count_mutations,
        create_output_run_label,
        convert_to_boolean,
        parse_scoring_args,
        save_allele_specific_outputs,
    )


NET_SCORE = "netMHCpan_EL_rank"
NET_PASS = "netMHCpan_EL_rank_pass"

FLURRY_SCORE = "MHCflurry_affinity_percentile"
FLURRY_PASS = "MHCflurry_affinity_percentile_pass"

GROUP_COLUMNS = ["allele", "variant_id"]


# ------------------------------------------------------------------
# General helper functions
# ------------------------------------------------------------------

def smallest_score(series: pd.Series) -> float:
    """
    Return the smallest non-missing score.

    Lower NetMHCpan EL ranks and MHCflurry affinity percentiles
    indicate stronger predicted presentation.
    """
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if values.empty:
        return np.nan

    return float(values.min())


def mean_top_n(
    series: pd.Series,
    n: int = 3,
    require_n: bool = False,
) -> float:
    """
    Calculate the mean of the n smallest scores.

    Parameters
    ----------
    series:
        Prediction score values.
    n:
        Number of strongest scores to average.
    require_n:
        If True, return NaN when fewer than n valid scores exist.
        If False, average however many valid scores are available,
        up to n.
    """
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if values.empty:
        return np.nan

    if require_n and len(values) < n:
        return np.nan

    return float(values.nsmallest(n).mean())


def first_non_missing(series: pd.Series):
    """Return the first non-missing value in a series."""
    non_missing = series.dropna()

    if non_missing.empty:
        return np.nan

    return non_missing.iloc[0]


# ------------------------------------------------------------------
# Prepare peptide-level data
# ------------------------------------------------------------------

def prepare_prediction_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean the peptide-level prediction dataframe.
    """
    required_columns = [
        "allele",
        "variant_id",
        "peptide_id",
        "VR_mutation",
        NET_SCORE,
        NET_PASS,
        FLURRY_SCORE,
        FLURRY_PASS,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    data = df[required_columns].copy()

    data["allele"] = (
        data["allele"]
        .astype("string")
        .str.strip()
    )

    data["variant_id"] = (
        data["variant_id"]
        .astype("string")
        .str.strip()
    )

    data["peptide_id"] = (
        data["peptide_id"]
        .astype("string")
        .str.strip()
    )

    data[NET_SCORE] = pd.to_numeric(
        data[NET_SCORE],
        errors="coerce",
    )

    data[FLURRY_SCORE] = pd.to_numeric(
        data[FLURRY_SCORE],
        errors="coerce",
    )

    data[NET_PASS] = convert_to_boolean(data[NET_PASS])
    data[FLURRY_PASS] = convert_to_boolean(data[FLURRY_PASS])

    data["VR_mutation"] = (
        data["VR_mutation"]
        .fillna("WT")
        .astype(str)
        .str.strip()
        .replace("", "WT")
    )

    # Check mutation annotation consistency within each allele/variant.
    mutation_counts = (
        data.groupby(
            GROUP_COLUMNS,
            observed=True,
        )["VR_mutation"]
        .nunique()
    )

    inconsistent_groups = mutation_counts[
        mutation_counts > 1
    ]

    if not inconsistent_groups.empty:
        examples = inconsistent_groups.head().index.tolist()

        raise ValueError(
            f"{len(inconsistent_groups)} allele/variant groups have "
            "multiple VR_mutation annotations. "
            f"Example groups: {examples}"
        )

    # The same allele-peptide-variant event should only be counted once.
    data = data.drop_duplicates(
        subset=[
            "allele",
            "variant_id",
            "peptide_id",
        ]
    ).copy()

    data["both_pass"] = (
        data[NET_PASS]
        & data[FLURRY_PASS]
    )

    data["either_pass"] = (
        data[NET_PASS]
        | data[FLURRY_PASS]
    )

    return data


# ------------------------------------------------------------------
# Allele-specific variant scores
# ------------------------------------------------------------------

def create_allele_specific_scores(
    data: pd.DataFrame,
    require_three_scores: bool = False,
) -> pd.DataFrame:
    """
    Collapse peptide-level rows into one row per allele and variant.

    Continuous score summaries are calculated across all available
    peptide scores, regardless of whether the peptide passes the
    chosen threshold.
    """
    mutation_info = (
        data.groupby(
            GROUP_COLUMNS,
            observed=True,
        )
        .agg(
            VR_mutation=("VR_mutation", "first"),
        )
        .reset_index()
    )

    mutation_info["mutation_count"] = (
        mutation_info["VR_mutation"]
        .apply(count_mutations)
        .astype(int)
    )

    count_scores = (
        data.groupby(
            GROUP_COLUMNS,
            observed=True,
        )
        .agg(
            total_epitope_count=(
                "peptide_id",
                "nunique",
            ),
            netMHCpan_passed_count=(
                NET_PASS,
                "sum",
            ),
            MHCflurry_passed_count=(
                FLURRY_PASS,
                "sum",
            ),
            both_passed_count=(
                "both_pass",
                "sum",
            ),
            either_passed_count=(
                "either_pass",
                "sum",
            ),
        )
        .reset_index()
    )

    continuous_scores = (
        data.groupby(
            GROUP_COLUMNS,
            observed=True,
        )
        .agg(
            netMHCpan_top_score=(
                NET_SCORE,
                smallest_score,
            ),
            netMHCpan_mean_top3=(
                NET_SCORE,
                lambda values: mean_top_n(
                    values,
                    n=3,
                    require_n=require_three_scores,
                ),
            ),
            MHCflurry_top_score=(
                FLURRY_SCORE,
                smallest_score,
            ),
            MHCflurry_mean_top3=(
                FLURRY_SCORE,
                lambda values: mean_top_n(
                    values,
                    n=3,
                    require_n=require_three_scores,
                ),
            ),
        )
        .reset_index()
    )

    scores = (
        mutation_info
        .merge(
            count_scores,
            on=GROUP_COLUMNS,
            how="left",
            validate="one_to_one",
        )
        .merge(
            continuous_scores,
            on=GROUP_COLUMNS,
            how="left",
            validate="one_to_one",
        )
    )

    count_columns = [
        "mutation_count",
        "total_epitope_count",
        "netMHCpan_passed_count",
        "MHCflurry_passed_count",
        "both_passed_count",
        "either_passed_count",
    ]

    scores[count_columns] = (
        scores[count_columns]
        .fillna(0)
        .astype(int)
    )

    scores = scores.sort_values(
        ["allele", "variant_id"]
    ).reset_index(drop=True)

    return scores


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> int:
    args = parse_scoring_args(
        description="Create absolute variant-level MHC-I presentation scores.",
        default_output_root="data/output/variant_immunogenicity_scores",
        variant_help="MHC-I combined_annotated.tsv produced by the workflow.",
    )
    variant_input = args.variant_input.resolve()
    run_label = args.run_label or create_output_run_label(variant_input)

    print(f"Reading predictions from: {variant_input}")

    df = pd.read_csv(
        variant_input,
        sep="\t",
        low_memory=False,
    )

    data = prepare_prediction_data(df)

    allele_scores = create_allele_specific_scores(
        data,
        # False allows variants with only one or two valid peptide
        # scores to use the available values in mean_top3.
        require_three_scores=False,
    )

    allele_output_files = save_allele_specific_outputs(
        allele_scores=allele_scores,
        output_root=args.output_root,
        run_label=run_label,
    )

    print("\nAllele-specific outputs:")

    for output_file in allele_output_files:
        print(f"  {output_file}")

    print(
        f"\nCreated scores for "
        f"{allele_scores['variant_id'].nunique():,} variants "
        f"and {allele_scores['allele'].nunique():,} alleles."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
