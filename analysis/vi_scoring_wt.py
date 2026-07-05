# Variant-level immunogenicity scoring with optional WT-relative scores
#
# Creates:
#   1. One variant-level score table per allele
#   2. One combined variant-level score table across all alleles
#
# WT-relative logic:
#   - Variant peptide rows are matched to WT peptide rows by allele + start + end + k.
#   - Do NOT match by peptide sequence, because the variant peptide sequence may differ from WT.
#   - For NetMHCpan and MHCflurry percentile/rank scores, lower values mean stronger presentation.
#       delta = variant_score - WT_score
#       negative delta = stronger than WT
#       positive delta = weaker than WT
#   - For BigMHC_EL, higher values are treated as stronger immunogenicity.
#       delta = variant_score - WT_score
#       positive delta = stronger than WT
#       negative delta = weaker than WT

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

INPUT_FILE = Path(
    "data/output/bigmhc/VR6__k9/predictions_mapped.tsv"
)

# Set to None if you do not want WT-relative columns.
# Example names:
#   data/output/bigmhc/WT_vr5__k9/predictions_mapped.tsv
#   data/output/bigmhc/WT_vr6__k9/predictions_mapped.tsv
WT_INPUT_FILE: Path | None = Path(
    "data/output/bigmhc/WT_vr6__k9/predictions_mapped.tsv"
)

OUTPUT_ROOT = Path(
    "data/output/variant_immunogenicity_scores_wt"
)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

NET_SCORE = "netMHCpan_EL_rank"
NET_PASS = "netMHCpan_EL_rank_pass"

FLURRY_SCORE = "MHCflurry_affinity_percentile"
FLURRY_PASS = "MHCflurry_affinity_percentile_pass"

BIGMHC_SCORE = "BigMHC_EL"

GROUP_COLUMNS = ["allele", "variant_id"]
WINDOW_COLUMNS = ["allele", "start", "end", "k"]


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
    values = pd.to_numeric(series, errors="coerce").dropna()

    if values.empty:
        return np.nan

    return float(values.min())


def largest_score(series: pd.Series) -> float:
    """
    Return the largest non-missing score.

    Higher BigMHC_EL values are treated as stronger immunogenicity.
    """
    values = pd.to_numeric(series, errors="coerce").dropna()

    if values.empty:
        return np.nan

    return float(values.max())


def mean_lowest_n(
    series: pd.Series,
    n: int = 3,
    require_n: bool = False,
) -> float:
    """
    Calculate the mean of the n smallest values.

    Used for NetMHCpan/MHCflurry absolute scores and their deltas,
    because lower rank/percentile values indicate stronger presentation.
    """
    values = pd.to_numeric(series, errors="coerce").dropna()

    if values.empty:
        return np.nan

    if require_n and len(values) < n:
        return np.nan

    return float(values.nsmallest(n).mean())


def mean_highest_n(
    series: pd.Series,
    n: int = 3,
    require_n: bool = False,
) -> float:
    """
    Calculate the mean of the n largest values.

    Used for BigMHC_EL absolute scores and deltas, because higher
    BigMHC_EL values are treated as stronger immunogenicity.
    """
    values = pd.to_numeric(series, errors="coerce").dropna()

    if values.empty:
        return np.nan

    if require_n and len(values) < n:
        return np.nan

    return float(values.nlargest(n).mean())


def mean_all(series: pd.Series) -> float:
    """Return the mean of all valid numeric values."""
    values = pd.to_numeric(series, errors="coerce").dropna()

    if values.empty:
        return np.nan

    return float(values.mean())


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


# ------------------------------------------------------------------
# Prepare peptide-level data
# ------------------------------------------------------------------

def prepare_prediction_data(
    df: pd.DataFrame,
    *,
    is_wt: bool = False,
) -> pd.DataFrame:
    """
    Validate and clean the peptide-level prediction dataframe.

    The function keeps start/end/k so variant peptides can be matched
    back to the equivalent WT peptide window.
    """
    required_columns = [
        "allele",
        "variant_id",
        "peptide_id",
        "start",
        "end",
        "k",
        "VR_mutation",
        NET_SCORE,
        NET_PASS,
        FLURRY_SCORE,
        FLURRY_PASS,
    ]

    optional_score_columns = []

    if BIGMHC_SCORE in df.columns:
        optional_score_columns.append(BIGMHC_SCORE)

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing_columns)
        )

    data = df[required_columns + optional_score_columns].copy()

    data["allele"] = data["allele"].astype("string").str.strip()
    data["variant_id"] = data["variant_id"].astype("string").str.strip()
    data["peptide_id"] = data["peptide_id"].astype("string").str.strip()

    for column in ["start", "end", "k"]:
        data[column] = pd.to_numeric(data[column], errors="raise").astype(int)

    data[NET_SCORE] = pd.to_numeric(data[NET_SCORE], errors="coerce")
    data[FLURRY_SCORE] = pd.to_numeric(data[FLURRY_SCORE], errors="coerce")

    if BIGMHC_SCORE in data.columns:
        data[BIGMHC_SCORE] = pd.to_numeric(data[BIGMHC_SCORE], errors="coerce")
    else:
        data[BIGMHC_SCORE] = np.nan

    data[NET_PASS] = convert_to_boolean(data[NET_PASS])
    data[FLURRY_PASS] = convert_to_boolean(data[FLURRY_PASS])

    data["VR_mutation"] = (
        data["VR_mutation"]
        .fillna("WT")
        .astype(str)
        .str.strip()
        .replace("", "WT")
    )

    # Variant file: mutation annotation should be consistent within each allele/variant.
    # WT file: the same check is harmless but mostly redundant.
    mutation_counts = (
        data.groupby(GROUP_COLUMNS, observed=True)["VR_mutation"].nunique()
    )

    inconsistent_groups = mutation_counts[mutation_counts > 1]

    if not inconsistent_groups.empty:
        examples = inconsistent_groups.head().index.tolist()

        raise ValueError(
            f"{len(inconsistent_groups)} allele/variant groups have "
            "multiple VR_mutation annotations. "
            f"Example groups: {examples}"
        )

    # The same allele-window-variant event should only be counted once.
    # Use window coordinates rather than peptide_id because WT-relative
    # matching is coordinate based.
    duplicate_subset = [
        "allele",
        "variant_id",
        "start",
        "end",
        "k",
    ]

    duplicated = data.duplicated(subset=duplicate_subset, keep=False)

    if duplicated.any():
        n_duplicates = int(duplicated.sum())
        print(
            f"Warning: {n_duplicates:,} duplicated allele/variant/window rows found. "
            "Keeping the first occurrence."
        )

    data = data.drop_duplicates(subset=duplicate_subset).copy()

    data["both_pass"] = data[NET_PASS] & data[FLURRY_PASS]
    data["either_pass"] = data[NET_PASS] | data[FLURRY_PASS]

    return data


# ------------------------------------------------------------------
# WT matching and WT-relative columns
# ------------------------------------------------------------------

def prepare_wt_reference(wt_data: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse WT data to one row per allele/window/k.

    The WT file should normally have one WT variant row per allele/window,
    but this function is defensive if duplicate WT variant IDs are present.
    """
    wt_reference = (
        wt_data.groupby(WINDOW_COLUMNS, observed=True)
        .agg(
            wt_peptide_id=("peptide_id", "first"),
            wt_variant_id=("variant_id", "first"),
            wt_netMHCpan_EL_rank=(NET_SCORE, smallest_score),
            wt_netMHCpan_EL_rank_pass=(NET_PASS, "max"),
            wt_MHCflurry_affinity_percentile=(FLURRY_SCORE, smallest_score),
            wt_MHCflurry_affinity_percentile_pass=(FLURRY_PASS, "max"),
            wt_both_pass=("both_pass", "max"),
            wt_either_pass=("either_pass", "max"),
            wt_BigMHC_EL=(BIGMHC_SCORE, largest_score),
        )
        .reset_index()
    )

    boolean_columns = [
        "wt_netMHCpan_EL_rank_pass",
        "wt_MHCflurry_affinity_percentile_pass",
        "wt_both_pass",
        "wt_either_pass",
    ]

    wt_reference[boolean_columns] = wt_reference[boolean_columns].astype(bool)

    return wt_reference


def add_wt_relative_columns(
    data: pd.DataFrame,
    wt_reference: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add WT peptide-window scores and WT-relative delta columns.
    """
    merged = data.merge(
        wt_reference,
        on=WINDOW_COLUMNS,
        how="left",
        validate="many_to_one",
        indicator="wt_merge_status",
    )

    missing_wt = (merged["wt_merge_status"] == "left_only").sum()

    if missing_wt:
        print(
            f"Warning: {missing_wt:,} variant peptide rows did not find a "
            "matched WT allele/start/end/k row. WT-relative columns will be NaN "
            "or False for these rows."
        )

    merged = merged.drop(columns=["wt_merge_status"])

    # Continuous deltas.
    # NetMHCpan/MHCflurry: negative = stronger than WT.
    merged["netMHCpan_EL_rank_delta_vs_WT"] = (
        merged[NET_SCORE] - merged["wt_netMHCpan_EL_rank"]
    )

    merged["MHCflurry_affinity_percentile_delta_vs_WT"] = (
        merged[FLURRY_SCORE] - merged["wt_MHCflurry_affinity_percentile"]
    )

    # BigMHC: positive = stronger than WT.
    merged["BigMHC_EL_delta_vs_WT"] = (
        merged[BIGMHC_SCORE] - merged["wt_BigMHC_EL"]
    )

    # Boolean pass changes.
    # Use nullable-aware false fill only for pass-change flags.
    wt_net_pass = merged["wt_netMHCpan_EL_rank_pass"].fillna(False).astype(bool)
    wt_flurry_pass = merged[
        "wt_MHCflurry_affinity_percentile_pass"
    ].fillna(False).astype(bool)
    wt_both_pass = merged["wt_both_pass"].fillna(False).astype(bool)
    wt_either_pass = merged["wt_either_pass"].fillna(False).astype(bool)

    merged["netMHCpan_new_pass_vs_WT"] = merged[NET_PASS] & ~wt_net_pass
    merged["netMHCpan_lost_pass_vs_WT"] = ~merged[NET_PASS] & wt_net_pass
    merged["netMHCpan_pass_change_vs_WT"] = (
        merged[NET_PASS].astype(int) - wt_net_pass.astype(int)
    )

    merged["MHCflurry_new_pass_vs_WT"] = merged[FLURRY_PASS] & ~wt_flurry_pass
    merged["MHCflurry_lost_pass_vs_WT"] = ~merged[FLURRY_PASS] & wt_flurry_pass
    merged["MHCflurry_pass_change_vs_WT"] = (
        merged[FLURRY_PASS].astype(int) - wt_flurry_pass.astype(int)
    )

    merged["both_new_pass_vs_WT"] = merged["both_pass"] & ~wt_both_pass
    merged["both_lost_pass_vs_WT"] = ~merged["both_pass"] & wt_both_pass
    merged["both_pass_change_vs_WT"] = (
        merged["both_pass"].astype(int) - wt_both_pass.astype(int)
    )

    merged["either_new_pass_vs_WT"] = merged["either_pass"] & ~wt_either_pass
    merged["either_lost_pass_vs_WT"] = ~merged["either_pass"] & wt_either_pass
    merged["either_pass_change_vs_WT"] = (
        merged["either_pass"].astype(int) - wt_either_pass.astype(int)
    )

    return merged


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
    peptide scores, regardless of whether the peptide passes the chosen
    threshold.
    """
    mutation_info = (
        data.groupby(GROUP_COLUMNS, observed=True)
        .agg(VR_mutation=("VR_mutation", "first"))
        .reset_index()
    )

    mutation_info["mutation_count"] = (
        mutation_info["VR_mutation"].apply(count_mutations).astype(int)
    )

    base_aggregations = dict(
        total_epitope_count=("peptide_id", "nunique"),
        netMHCpan_passed_count=(NET_PASS, "sum"),
        MHCflurry_passed_count=(FLURRY_PASS, "sum"),
        both_passed_count=("both_pass", "sum"),
        either_passed_count=("either_pass", "sum"),
    )

    wt_count_aggregations = {}

    if "wt_peptide_id" in data.columns:
        wt_count_aggregations = dict(
            wt_window_count=("wt_peptide_id", "count"),
            wt_netMHCpan_passed_count=("wt_netMHCpan_EL_rank_pass", "sum"),
            wt_MHCflurry_passed_count=(
                "wt_MHCflurry_affinity_percentile_pass",
                "sum",
            ),
            wt_both_passed_count=("wt_both_pass", "sum"),
            wt_either_passed_count=("wt_either_pass", "sum"),
            netMHCpan_new_passed_count=("netMHCpan_new_pass_vs_WT", "sum"),
            netMHCpan_lost_passed_count=("netMHCpan_lost_pass_vs_WT", "sum"),
            netMHCpan_net_pass_change=("netMHCpan_pass_change_vs_WT", "sum"),
            MHCflurry_new_passed_count=("MHCflurry_new_pass_vs_WT", "sum"),
            MHCflurry_lost_passed_count=("MHCflurry_lost_pass_vs_WT", "sum"),
            MHCflurry_net_pass_change=("MHCflurry_pass_change_vs_WT", "sum"),
            both_new_passed_count=("both_new_pass_vs_WT", "sum"),
            both_lost_passed_count=("both_lost_pass_vs_WT", "sum"),
            both_net_pass_change=("both_pass_change_vs_WT", "sum"),
            either_new_passed_count=("either_new_pass_vs_WT", "sum"),
            either_lost_passed_count=("either_lost_pass_vs_WT", "sum"),
            either_net_pass_change=("either_pass_change_vs_WT", "sum"),
        )

    count_scores = (
        data.groupby(GROUP_COLUMNS, observed=True)
        .agg(**base_aggregations, **wt_count_aggregations)
        .reset_index()
    )

    base_continuous_aggregations = dict(
        netMHCpan_top_score=(NET_SCORE, smallest_score),
        netMHCpan_mean_top3=(
            NET_SCORE,
            lambda values: mean_lowest_n(
                values,
                n=3,
                require_n=require_three_scores,
            ),
        ),
        netMHCpan_mean_all=(NET_SCORE, mean_all),
        MHCflurry_top_score=(FLURRY_SCORE, smallest_score),
        MHCflurry_mean_top3=(
            FLURRY_SCORE,
            lambda values: mean_lowest_n(
                values,
                n=3,
                require_n=require_three_scores,
            ),
        ),
        MHCflurry_mean_all=(FLURRY_SCORE, mean_all),
        BigMHC_top_score=(BIGMHC_SCORE, largest_score),
        BigMHC_mean_top3=(
            BIGMHC_SCORE,
            lambda values: mean_highest_n(
                values,
                n=3,
                require_n=require_three_scores,
            ),
        ),
        BigMHC_mean_all=(BIGMHC_SCORE, mean_all),
    )

    wt_continuous_aggregations = {}

    if "wt_peptide_id" in data.columns:
        wt_continuous_aggregations = dict(
            wt_netMHCpan_top_score=("wt_netMHCpan_EL_rank", smallest_score),
            wt_netMHCpan_mean_top3=(
                "wt_netMHCpan_EL_rank",
                lambda values: mean_lowest_n(
                    values,
                    n=3,
                    require_n=require_three_scores,
                ),
            ),
            netMHCpan_top_delta_vs_WT=(
                "netMHCpan_EL_rank_delta_vs_WT",
                smallest_score,
            ),
            netMHCpan_mean_top3_delta_vs_WT=(
                "netMHCpan_EL_rank_delta_vs_WT",
                lambda values: mean_lowest_n(
                    values,
                    n=3,
                    require_n=require_three_scores,
                ),
            ),
            netMHCpan_mean_delta_vs_WT=(
                "netMHCpan_EL_rank_delta_vs_WT",
                mean_all,
            ),
            wt_MHCflurry_top_score=(
                "wt_MHCflurry_affinity_percentile",
                smallest_score,
            ),
            wt_MHCflurry_mean_top3=(
                "wt_MHCflurry_affinity_percentile",
                lambda values: mean_lowest_n(
                    values,
                    n=3,
                    require_n=require_three_scores,
                ),
            ),
            MHCflurry_top_delta_vs_WT=(
                "MHCflurry_affinity_percentile_delta_vs_WT",
                smallest_score,
            ),
            MHCflurry_mean_top3_delta_vs_WT=(
                "MHCflurry_affinity_percentile_delta_vs_WT",
                lambda values: mean_lowest_n(
                    values,
                    n=3,
                    require_n=require_three_scores,
                ),
            ),
            MHCflurry_mean_delta_vs_WT=(
                "MHCflurry_affinity_percentile_delta_vs_WT",
                mean_all,
            ),
            wt_BigMHC_top_score=("wt_BigMHC_EL", largest_score),
            wt_BigMHC_mean_top3=(
                "wt_BigMHC_EL",
                lambda values: mean_highest_n(
                    values,
                    n=3,
                    require_n=require_three_scores,
                ),
            ),
            BigMHC_top_delta_vs_WT=("BigMHC_EL_delta_vs_WT", largest_score),
            BigMHC_mean_top3_delta_vs_WT=(
                "BigMHC_EL_delta_vs_WT",
                lambda values: mean_highest_n(
                    values,
                    n=3,
                    require_n=require_three_scores,
                ),
            ),
            BigMHC_mean_delta_vs_WT=("BigMHC_EL_delta_vs_WT", mean_all),
        )

    continuous_scores = (
        data.groupby(GROUP_COLUMNS, observed=True)
        .agg(**base_continuous_aggregations, **wt_continuous_aggregations)
        .reset_index()
    )

    scores = (
        mutation_info
        .merge(count_scores, on=GROUP_COLUMNS, how="left", validate="one_to_one")
        .merge(
            continuous_scores,
            on=GROUP_COLUMNS,
            how="left",
            validate="one_to_one",
        )
    )

    integer_columns = [
        column
        for column in scores.columns
        if column.endswith("_count")
        or column.endswith("_passed_count")
        or column.endswith("_net_pass_change")
        or column in {"mutation_count", "total_epitope_count", "wt_window_count"}
    ]

    scores[integer_columns] = scores[integer_columns].fillna(0).astype(int)

    scores = scores.sort_values(["allele", "variant_id"]).reset_index(drop=True)

    return scores


# ------------------------------------------------------------------
# Combined scores across alleles
# ------------------------------------------------------------------

def validate_cross_allele_mutations(allele_scores: pd.DataFrame) -> None:
    """
    Confirm that each variant has the same mutation annotation across every allele.
    """
    annotation_counts = allele_scores.groupby(
        "variant_id",
        observed=True,
    )["VR_mutation"].nunique()

    inconsistent_variants = annotation_counts[annotation_counts > 1]

    if not inconsistent_variants.empty:
        examples = inconsistent_variants.head().index.tolist()

        raise ValueError(
            f"{len(inconsistent_variants)} variants have inconsistent "
            "VR_mutation annotations across alleles. "
            f"Example variants: {examples}"
        )


def add_optional_agg(
    aggregations: dict,
    df: pd.DataFrame,
    output_column: str,
    source_column: str,
    agg_function,
) -> None:
    """Add an aggregation only if the source column exists."""
    if source_column in df.columns:
        aggregations[output_column] = (source_column, agg_function)


def create_combined_scores(allele_scores: pd.DataFrame) -> pd.DataFrame:
    """
    Combine allele-specific variant scores into one row per variant.

    Counts are summed across allele-peptide prediction events.
    Continuous lower-is-stronger scores use minima across alleles.
    Continuous higher-is-stronger scores use maxima across alleles.
    """
    validate_cross_allele_mutations(allele_scores)

    aggregations = dict(
        VR_mutation=("VR_mutation", "first"),
        mutation_count=("mutation_count", "first"),
        allele_count=("allele", "nunique"),
        combined_total_epitope_count=("total_epitope_count", "sum"),
        combined_netMHCpan_passed_count=("netMHCpan_passed_count", "sum"),
        combined_MHCflurry_passed_count=("MHCflurry_passed_count", "sum"),
        combined_both_passed_count=("both_passed_count", "sum"),
        combined_either_passed_count=("either_passed_count", "sum"),
        combined_netMHCpan_top_score=("netMHCpan_top_score", "min"),
        combined_MHCflurry_top_score=("MHCflurry_top_score", "min"),
        combined_BigMHC_top_score=("BigMHC_top_score", "max"),
        netMHCpan_mean_best_across_alleles=("netMHCpan_top_score", "mean"),
        MHCflurry_mean_best_across_alleles=("MHCflurry_top_score", "mean"),
        BigMHC_mean_best_across_alleles=("BigMHC_top_score", "mean"),
        combined_netMHCpan_best_mean_top3=("netMHCpan_mean_top3", "min"),
        combined_MHCflurry_best_mean_top3=("MHCflurry_mean_top3", "min"),
        combined_BigMHC_best_mean_top3=("BigMHC_mean_top3", "max"),
        netMHCpan_mean_top3_across_alleles=("netMHCpan_mean_top3", "mean"),
        MHCflurry_mean_top3_across_alleles=("MHCflurry_mean_top3", "mean"),
        BigMHC_mean_top3_across_alleles=("BigMHC_mean_top3", "mean"),
    )

    # Optional WT-relative count summaries.
    optional_sum_columns = [
        "wt_window_count",
        "wt_netMHCpan_passed_count",
        "wt_MHCflurry_passed_count",
        "wt_both_passed_count",
        "wt_either_passed_count",
        "netMHCpan_new_passed_count",
        "netMHCpan_lost_passed_count",
        "netMHCpan_net_pass_change",
        "MHCflurry_new_passed_count",
        "MHCflurry_lost_passed_count",
        "MHCflurry_net_pass_change",
        "both_new_passed_count",
        "both_lost_passed_count",
        "both_net_pass_change",
        "either_new_passed_count",
        "either_lost_passed_count",
        "either_net_pass_change",
    ]

    for column in optional_sum_columns:
        add_optional_agg(
            aggregations,
            allele_scores,
            f"combined_{column}",
            column,
            "sum",
        )

    # Optional WT-relative continuous summaries.
    optional_continuous = [
        ("combined_wt_netMHCpan_top_score", "wt_netMHCpan_top_score", "min"),
        ("combined_wt_MHCflurry_top_score", "wt_MHCflurry_top_score", "min"),
        ("combined_wt_BigMHC_top_score", "wt_BigMHC_top_score", "max"),
        (
            "combined_netMHCpan_top_delta_vs_WT",
            "netMHCpan_top_delta_vs_WT",
            "min",
        ),
        (
            "combined_MHCflurry_top_delta_vs_WT",
            "MHCflurry_top_delta_vs_WT",
            "min",
        ),
        ("combined_BigMHC_top_delta_vs_WT", "BigMHC_top_delta_vs_WT", "max"),
        (
            "combined_netMHCpan_best_mean_top3_delta_vs_WT",
            "netMHCpan_mean_top3_delta_vs_WT",
            "min",
        ),
        (
            "combined_MHCflurry_best_mean_top3_delta_vs_WT",
            "MHCflurry_mean_top3_delta_vs_WT",
            "min",
        ),
        (
            "combined_BigMHC_best_mean_top3_delta_vs_WT",
            "BigMHC_mean_top3_delta_vs_WT",
            "max",
        ),
        (
            "netMHCpan_top_delta_mean_across_alleles",
            "netMHCpan_top_delta_vs_WT",
            "mean",
        ),
        (
            "MHCflurry_top_delta_mean_across_alleles",
            "MHCflurry_top_delta_vs_WT",
            "mean",
        ),
        (
            "BigMHC_top_delta_mean_across_alleles",
            "BigMHC_top_delta_vs_WT",
            "mean",
        ),
        (
            "netMHCpan_mean_top3_delta_across_alleles",
            "netMHCpan_mean_top3_delta_vs_WT",
            "mean",
        ),
        (
            "MHCflurry_mean_top3_delta_across_alleles",
            "MHCflurry_mean_top3_delta_vs_WT",
            "mean",
        ),
        (
            "BigMHC_mean_top3_delta_across_alleles",
            "BigMHC_mean_top3_delta_vs_WT",
            "mean",
        ),
    ]

    for output_column, source_column, agg_function in optional_continuous:
        add_optional_agg(
            aggregations,
            allele_scores,
            output_column,
            source_column,
            agg_function,
        )

    combined = (
        allele_scores.groupby("variant_id", observed=True)
        .agg(**aggregations)
        .reset_index()
    )

    integer_columns = [
        column
        for column in combined.columns
        if column.endswith("_count")
        or column.endswith("_passed_count")
        or column.endswith("_net_pass_change")
        or column in {"mutation_count", "allele_count"}
    ]

    combined[integer_columns] = combined[integer_columns].fillna(0).astype(int)

    combined = combined.sort_values("variant_id").reset_index(drop=True)

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

        allele_output_dir = output_root / f"{run_label}_{clean_allele}"
        allele_output_dir.mkdir(parents=True, exist_ok=True)

        output_file = allele_output_dir / "variant_immunogenicity_scores.tsv"

        allele_df.copy().to_csv(output_file, sep="\t", index=False)

        output_files.append(output_file)

    return output_files


def save_combined_output(
    combined_scores: pd.DataFrame,
    output_root: Path,
    run_label: str,
) -> Path:
    """Save the combined cross-allele table."""
    combined_output_dir = output_root / f"{run_label}_combined"
    combined_output_dir.mkdir(parents=True, exist_ok=True)

    output_file = combined_output_dir / "variant_immunogenicity_scores.tsv"

    combined_scores.to_csv(output_file, sep="\t", index=False)

    return output_file


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> int:
    print(f"Reading variant predictions from: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE, sep="\t", low_memory=False)
    data = prepare_prediction_data(df)

    if WT_INPUT_FILE is not None:
        print(f"Reading WT predictions from: {WT_INPUT_FILE}")

        wt_df = pd.read_csv(WT_INPUT_FILE, sep="\t", low_memory=False)
        wt_data = prepare_prediction_data(wt_df, is_wt=True)
        wt_reference = prepare_wt_reference(wt_data)

        print(
            f"Prepared WT reference with {len(wt_reference):,} "
            "allele/window/k rows."
        )

        data = add_wt_relative_columns(data, wt_reference)
    else:
        print("No WT input file supplied. Creating absolute scores only.")

    allele_scores = create_allele_specific_scores(
        data,
        # False allows variants with only one or two valid peptide scores
        # to use the available values in mean_top3.
        require_three_scores=False,
    )

    combined_scores = create_combined_scores(allele_scores)

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
