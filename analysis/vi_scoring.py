# Variant-level immunogenicity scoring
#
# Creates:
#   1. One variant-level score table per allele
#   2. One combined variant-level score table across all alleles
#
# Count-based combined scores represent allele-peptide presentation events.
# Continuous combined scores are calculated from allele-level summaries,
# rather than averaging every peptide prediction across all alleles.

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

INPUT_FILE = Path(
    "data/output/bigmhc/VR4__k9/predictions_mapped.tsv"
)

OUTPUT_ROOT = Path(
    "data/output/variant_immunogenicity_scores"
)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

NET_SCORE = "netMHCpan_EL_rank"
NET_PASS = "netMHCpan_EL_rank_pass"

FLURRY_SCORE = "MHCflurry_affinity_percentile"
FLURRY_PASS = "MHCflurry_affinity_percentile_pass"

GROUP_COLUMNS = ["allele", "variant_id"]


# ------------------------------------------------------------------
# Naming helpers
# ------------------------------------------------------------------

def create_output_run_label(input_file: Path) -> str:
    """
    Convert an input run directory name into a clean output label.

    Example
    -------
    VR5_V3__k9 -> VR5_V3_K9
    """
    run_name = input_file.parent.name

    run_name = re.sub(
        r"__k(\d+)$",
        lambda match: f"_K{match.group(1)}",
        run_name,
        flags=re.IGNORECASE,
    )

    return run_name


def clean_allele_name(allele: str) -> str:
    """
    Convert an allele name into a filesystem-friendly label.

    Examples
    --------
    H2-D*b -> H2-Db
    H2-D*d -> H2-Dd
    H-2-Db -> H2-Db
    """
    allele_name = str(allele).strip()

    allele_name = allele_name.replace("H-2", "H2")
    allele_name = allele_name.replace("*", "")
    allele_name = allele_name.replace(":", "")
    allele_name = allele_name.replace("/", "-")
    allele_name = allele_name.replace("\\", "-")
    allele_name = allele_name.replace(" ", "_")

    return allele_name


RUN_LABEL = create_output_run_label(INPUT_FILE)


# ------------------------------------------------------------------
# General helper functions
# ------------------------------------------------------------------

def convert_to_boolean(series: pd.Series) -> pd.Series:
    """
    Convert boolean-like values to proper True/False values.

    Handles:
    - True / False
    - "True" / "False"
    - 1 / 0

    Missing or unrecognised values are treated as False.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    converted = (
        series.astype("string")
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            }
        )
    )

    return converted.fillna(False).astype(bool)


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


def count_mutations(value: object) -> int:
    """
    Count mutations in a semicolon-separated VR_mutation string.

    Examples
    --------
    "T2L;T3H;E11A" -> 3
    "T2L"           -> 1
    "WT"            -> 0
    NaN             -> 0
    """
    if pd.isna(value):
        return 0

    value = str(value).strip()

    if not value or value.upper() == "WT":
        return 0

    mutations = [
        mutation.strip()
        for mutation in value.split(";")
        if mutation.strip()
    ]

    return len(mutations)


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

    No boolean has_pass columns are included in the output.
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
# Combined scores across alleles
# ------------------------------------------------------------------

def validate_cross_allele_mutations(
    allele_scores: pd.DataFrame,
) -> None:
    """
    Confirm that each variant has the same mutation annotation
    across every allele.
    """
    annotation_counts = (
        allele_scores.groupby(
            "variant_id",
            observed=True,
        )["VR_mutation"]
        .nunique()
    )

    inconsistent_variants = annotation_counts[
        annotation_counts > 1
    ]

    if not inconsistent_variants.empty:
        examples = inconsistent_variants.head().index.tolist()

        raise ValueError(
            f"{len(inconsistent_variants)} variants have inconsistent "
            "VR_mutation annotations across alleles. "
            f"Example variants: {examples}"
        )


def create_combined_scores(
    allele_scores: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine allele-specific variant scores into one row per variant.

    Counts
    ------
    Count features are summed across alleles.

    For example:

        combined_both_passed_count =
            H2_Dd_both_passed_count
            + H2_Db_both_passed_count

    Therefore, these counts represent allele-peptide prediction events,
    not merely unique peptide sequences.

    Continuous scores
    -----------------
    Since lower values indicate stronger predictions:

    - combined_*_top_score:
        Minimum of the allele-specific top scores.

    - *_mean_best_across_alleles:
        Mean of the best score from each allele.

    The same logic is also applied to each allele's mean-top-three
    summary.
    """
    validate_cross_allele_mutations(allele_scores)

    combined = (
        allele_scores.groupby(
            "variant_id",
            observed=True,
        )
        .agg(
            VR_mutation=(
                "VR_mutation",
                "first",
            ),
            mutation_count=(
                "mutation_count",
                "first",
            ),
            allele_count=(
                "allele",
                "nunique",
            ),

            # Counts summed across allele-peptide events
            combined_total_epitope_count=(
                "total_epitope_count",
                "sum",
            ),
            combined_netMHCpan_passed_count=(
                "netMHCpan_passed_count",
                "sum",
            ),
            combined_MHCflurry_passed_count=(
                "MHCflurry_passed_count",
                "sum",
            ),
            combined_both_passed_count=(
                "both_passed_count",
                "sum",
            ),
            combined_either_passed_count=(
                "either_passed_count",
                "sum",
            ),

            # Strongest score observed for any allele
            combined_netMHCpan_top_score=(
                "netMHCpan_top_score",
                "min",
            ),
            combined_MHCflurry_top_score=(
                "MHCflurry_top_score",
                "min",
            ),

            # Mean of each allele's strongest peptide score
            netMHCpan_mean_best_across_alleles=(
                "netMHCpan_top_score",
                "mean",
            ),
            MHCflurry_mean_best_across_alleles=(
                "MHCflurry_top_score",
                "mean",
            ),

            # Best allele-level mean-top-three score
            combined_netMHCpan_best_mean_top3=(
                "netMHCpan_mean_top3",
                "min",
            ),
            combined_MHCflurry_best_mean_top3=(
                "MHCflurry_mean_top3",
                "min",
            ),

            # Mean of the allele-specific mean-top-three summaries
            netMHCpan_mean_top3_across_alleles=(
                "netMHCpan_mean_top3",
                "mean",
            ),
            MHCflurry_mean_top3_across_alleles=(
                "MHCflurry_mean_top3",
                "mean",
            ),
        )
        .reset_index()
    )

    integer_columns = [
        "mutation_count",
        "allele_count",
        "combined_total_epitope_count",
        "combined_netMHCpan_passed_count",
        "combined_MHCflurry_passed_count",
        "combined_both_passed_count",
        "combined_either_passed_count",
    ]

    combined[integer_columns] = (
        combined[integer_columns]
        .fillna(0)
        .astype(int)
    )

    combined = combined.sort_values(
        "variant_id"
    ).reset_index(drop=True)

    return combined


# ------------------------------------------------------------------
# Save output tables
# ------------------------------------------------------------------

def save_allele_specific_outputs(
    allele_scores: pd.DataFrame,
    output_root: Path,
    run_label: str,
) -> list[Path]:
    """
    Save one output table per allele.

    Example directory:
        variant_immunogenicity_scores/VR5_V3_K9_H2-Db/
    """
    output_files = []

    for allele, allele_df in allele_scores.groupby(
        "allele",
        observed=True,
        sort=True,
    ):
        clean_allele = clean_allele_name(allele)

        allele_output_dir = (
            output_root
            / f"{run_label}_{clean_allele}"
        )

        allele_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_file = (
            allele_output_dir
            / "variant_immunogenicity_scores.tsv"
        )

        allele_df = allele_df.copy()

        allele_df.to_csv(
            output_file,
            sep="\t",
            index=False,
        )

        output_files.append(output_file)

    return output_files


def save_combined_output(
    combined_scores: pd.DataFrame,
    output_root: Path,
    run_label: str,
) -> Path:
    """
    Save the combined cross-allele table.
    """
    combined_output_dir = (
        output_root
        / f"{run_label}_combined"
    )

    combined_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        combined_output_dir
        / "variant_immunogenicity_scores.tsv"
    )

    combined_scores.to_csv(
        output_file,
        sep="\t",
        index=False,
    )

    return output_file


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> int:
    print(f"Reading predictions from: {INPUT_FILE}")

    df = pd.read_csv(
        INPUT_FILE,
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

    combined_scores = create_combined_scores(
        allele_scores
    )

    allele_output_files = save_allele_specific_outputs(
        allele_scores=allele_scores,
        output_root=OUTPUT_ROOT,
        run_label=RUN_LABEL,
    )

    combined_output_file = save_combined_output(
        combined_scores=combined_scores,
        output_root=OUTPUT_ROOT,
        run_label=RUN_LABEL,
    )

    print("\nAllele-specific outputs:")

    for output_file in allele_output_files:
        print(f"  {output_file}")

    print("\nCombined output:")
    print(f"  {combined_output_file}")

    print(
        f"\nCreated scores for "
        f"{allele_scores['variant_id'].nunique():,} variants "
        f"and {allele_scores['allele'].nunique():,} alleles."
    )

    print("\nCombined score preview:")
    print(combined_scores.head())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())