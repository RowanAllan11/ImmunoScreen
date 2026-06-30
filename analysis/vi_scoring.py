# Variant immunogenicity scoring
# Load final output dataframe from bigmhc, 
# Create immunogenicity score features for each variant
# Scores:
# - Passed epitopes for both MHCflurry and netMHCpan
# - Passed epitopes for both MHCflurry and netMHCpan individually
# - Top scoring passed epitope per variant for both MHCflurry and netMHCpan
# - Mean of top 3 scores for both MHCflurry and netMHCpan
# Analyse distributions of scores across variants

import pandas as pd
import numpy as np

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

INPUT_FILE = Path(
    "data/output/bigmhc/VR5_V3__k9/predictions_mapped.tsv"
)

OUTPUT_DIR = Path(
    "data/output/variant_immunogenicity_scores/VR5_V3__k9"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GROUP_COLUMNS = ["allele", "variant_id"]

NET_SCORE = "netMHCpan_EL_rank"
NET_PASS = "netMHCpan_EL_rank_pass"

FLURRY_SCORE = "MHCflurry_affinity_percentile"
FLURRY_PASS = "MHCflurry_affinity_percentile_pass"


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------

def convert_to_boolean(series: pd.Series) -> pd.Series:
    """
    Convert boolean-like values to proper True/False values.

    Handles:
    - True / False
    - "True" / "False"
    - 1 / 0
    """
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    return (
        series.astype(str)
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
        .fillna(False)
        .astype(bool)
    )


def smallest_score(series: pd.Series) -> float:
    """
    Return the smallest non-missing score.

    Both NetMHCpan EL rank and MHCflurry affinity percentile are
    interpreted as stronger when their values are lower.
    """
    values = pd.to_numeric(series, errors="coerce").dropna()

    if values.empty:
        return np.nan

    return values.min()


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
        If True, variants with fewer than n scores return NaN.
        If False, use however many scores are available, up to n.
    """
    values = pd.to_numeric(series, errors="coerce").dropna()

    if values.empty:
        return np.nan

    if require_n and len(values) < n:
        return np.nan

    return values.nsmallest(n).mean()


# ------------------------------------------------------------------
# Create variant-level scores
# ------------------------------------------------------------------

def count_mutations(value) -> int:
    """
    Count individual mutations in a semicolon-separated VR_mutation string.

    Examples
    --------
    "T2L;T3H;E11A;F12A;W14D" -> 5
    "T2L"                      -> 1
    "WT"                       -> 0
    NaN                        -> 0
    """
    if pd.isna(value):
        return 0

    value = str(value).strip()

    if not value or value.upper() == "WT":
        return 0

    return len(
        [
            mutation.strip()
            for mutation in value.split(";")
            if mutation.strip()
        ]
    )


def create_immunogenicity_scores(
    df: pd.DataFrame,
    require_three_passed: bool = False,
) -> pd.DataFrame:
    """
    Collapse peptide-level predictions into variant-level features.

    Features include:
    - VR mutation string
    - Number of mutations in the variant
    - Number passing NetMHCpan
    - Number passing MHCflurry
    - Number passing both tools
    - Number passing either tool
    - Strongest score from each tool across all epitopes
    - Mean of the strongest three scores from each tool across all epitopes
    """

    required_columns = [
        *GROUP_COLUMNS,
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

    # Ensure prediction scores are numeric
    data[NET_SCORE] = pd.to_numeric(
        data[NET_SCORE],
        errors="coerce",
    )

    data[FLURRY_SCORE] = pd.to_numeric(
        data[FLURRY_SCORE],
        errors="coerce",
    )

    # Ensure pass columns are actual booleans
    data[NET_PASS] = convert_to_boolean(data[NET_PASS])
    data[FLURRY_PASS] = convert_to_boolean(data[FLURRY_PASS])

    # Standardise missing mutation strings
    data["VR_mutation"] = (
        data["VR_mutation"]
        .fillna("WT")
        .astype(str)
        .str.strip()
        .replace("", "WT")
    )

    # Check that each variant/allele has only one VR_mutation annotation
    mutation_annotation_counts = (
        data.groupby(GROUP_COLUMNS, observed=True)["VR_mutation"]
        .nunique()
    )

    inconsistent_groups = mutation_annotation_counts[
        mutation_annotation_counts > 1
    ]

    if not inconsistent_groups.empty:
        raise ValueError(
            f"{len(inconsistent_groups)} variant/allele groups have more "
            "than one VR_mutation annotation."
        )

    # Avoid counting duplicate peptide rows within the same variant/allele
    data = data.drop_duplicates(
        subset=[*GROUP_COLUMNS, "peptide_id"]
    ).copy()

    # Peptide-level combined pass indicators
    data["both_pass"] = data[NET_PASS] & data[FLURRY_PASS]
    data["either_pass"] = data[NET_PASS] | data[FLURRY_PASS]

    # Variant-level mutation information
    mutation_info = (
        data.groupby(GROUP_COLUMNS, observed=True)
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

    # Count passing epitopes
    count_scores = (
        data.groupby(GROUP_COLUMNS, observed=True)
        .agg(
            total_epitope_count=("peptide_id", "nunique"),
            netMHCpan_passed_count=(NET_PASS, "sum"),
            MHCflurry_passed_count=(FLURRY_PASS, "sum"),
            both_passed_count=("both_pass", "sum"),
            either_passed_count=("either_pass", "sum"),
        )
        .reset_index()
    )

    # Score summaries across all epitopes, regardless of pass status
    all_scores = (
        data.groupby(GROUP_COLUMNS, observed=True)
        .agg(
            netMHCpan_top_score=(
                NET_SCORE,
                smallest_score,
            ),
            netMHCpan_mean_top3=(
                NET_SCORE,
                lambda x: mean_top_n(
                    x,
                    n=3,
                    require_n=require_three_passed,
                ),
            ),
            MHCflurry_top_score=(
                FLURRY_SCORE,
                smallest_score,
            ),
            MHCflurry_mean_top3=(
                FLURRY_SCORE,
                lambda x: mean_top_n(
                    x,
                    n=3,
                    require_n=require_three_passed,
                ),
            ),
        )
        .reset_index()
    )

    # Keep all variants, including variants with no passing peptides
    scores = (
        mutation_info
        .merge(
            count_scores,
            on=GROUP_COLUMNS,
            how="left",
            validate="one_to_one",
        )
        .merge(
            all_scores,
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

    # Additional binary screening features
    scores["has_netMHCpan_pass"] = (
        scores["netMHCpan_passed_count"] > 0
    )

    scores["has_MHCflurry_pass"] = (
        scores["MHCflurry_passed_count"] > 0
    )

    scores["has_both_pass"] = (
        scores["both_passed_count"] > 0
    )

    scores["has_either_pass"] = (
        scores["either_passed_count"] > 0
    )

    return scores

# ------------------------------------------------------------------
# Distribution summaries
# ------------------------------------------------------------------

def create_distribution_summary(
    scores: pd.DataFrame,
) -> pd.DataFrame:
    """Create descriptive statistics for numeric score features."""

    score_columns = [
        "total_epitope_count",
        "netMHCpan_passed_count",
        "MHCflurry_passed_count",
        "both_passed_count",
        "either_passed_count",
        "netMHCpan_top_score",
        "netMHCpan_mean_top3",
        "MHCflurry_top_score",
        "MHCflurry_mean_top3",
    ]

    available_columns = [
        column for column in score_columns
        if column in scores.columns
    ]

    summary = (
        scores[available_columns]
        .describe(
            percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
        )
        .T
        .reset_index()
        .rename(columns={"index": "feature"})
    )

    summary["missing_count"] = [
        scores[column].isna().sum()
        for column in summary["feature"]
    ]

    summary["zero_count"] = [
        scores[column].eq(0).sum()
        for column in summary["feature"]
    ]

    return summary


def plot_score_distributions(
    scores: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save one histogram for each variant-level score."""

    plot_columns = [
        "netMHCpan_passed_count",
        "MHCflurry_passed_count",
        "both_passed_count",
        "either_passed_count",
        "netMHCpan_top_score",
        "netMHCpan_mean_top3",
        "MHCflurry_top_score",
        "MHCflurry_mean_top3",
    ]

    for column in plot_columns:
        values = scores[column].dropna()

        if values.empty:
            continue

        plt.figure(figsize=(8, 5))
        plt.hist(values, bins=30, edgecolor="black")
        plt.xlabel(column)
        plt.ylabel("Number of variants")
        plt.title(f"Distribution of {column}")
        plt.tight_layout()

        output_file = output_dir / f"{column}_distribution.png"
        plt.savefig(output_file, dpi=300)
        plt.close()


# ------------------------------------------------------------------
# Run analysis
# ------------------------------------------------------------------

df = pd.read_csv(INPUT_FILE, sep="\t")

scores = create_immunogenicity_scores(
    df,
    # False means variants with only 1 or 2 passing epitopes use
    # the available scores. Set True to require exactly 3 or more.
    require_three_passed=False,
)

distribution_summary = create_distribution_summary(scores)

scores.to_csv(
    OUTPUT_DIR / "variant_immunogenicity_scores.tsv",
    sep="\t",
    index=False,
)

distribution_summary.to_csv(
    OUTPUT_DIR / "variant_score_distribution_summary.tsv",
    sep="\t",
    index=False,
)

plot_score_distributions(
    scores=scores,
    output_dir=OUTPUT_DIR,
)

print(scores.head())
print()
print(distribution_summary)

print(
    f"\nVariant-level scores saved to:\n"
    f"{OUTPUT_DIR / 'variant_immunogenicity_scores.tsv'}"
)