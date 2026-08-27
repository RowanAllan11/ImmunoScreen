# Variant-level immunogenicity scoring with optional WT-relative scores
#
# Creates one WT-relative variant-level score table per allele.
#
# WT-relative logic:
#   - Variant peptide rows are matched to WT rows by allele + start + end + k.
#   - A window is "changed" only when its variant and matched-WT peptide
#     sequences differ.
#   - Every continuous feature uses one direction:
#       positive improvement = more immunogenic than WT
#       negative improvement = less immunogenic than WT
#   - NetMHCpan/MHCflurry improvement = WT rank/percentile - variant value.

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
except ModuleNotFoundError:  # Direct execution: python analysis/vi_scoring_wt.py
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
WINDOW_COLUMNS = ["allele", "start", "end", "k"]


# ------------------------------------------------------------------
# General helper functions
# ------------------------------------------------------------------

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


def mean_all(series: pd.Series) -> float:
    """Return the mean of all valid numeric values."""
    values = pd.to_numeric(series, errors="coerce").dropna()

    if values.empty:
        return np.nan

    return float(values.mean())


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
        "peptide",
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

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing_columns)
        )

    data = df[required_columns].copy()

    data["allele"] = data["allele"].astype("string").str.strip()
    data["peptide"] = data["peptide"].astype("string").str.strip()
    data["variant_id"] = data["variant_id"].astype("string").str.strip()
    data["peptide_id"] = data["peptide_id"].astype("string").str.strip()

    for column in ["start", "end", "k"]:
        data[column] = pd.to_numeric(data[column], errors="raise").astype(int)

    data[NET_SCORE] = pd.to_numeric(data[NET_SCORE], errors="coerce")
    data[FLURRY_SCORE] = pd.to_numeric(data[FLURRY_SCORE], errors="coerce")

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
            wt_peptide=("peptide", "first"),
            wt_variant_id=("variant_id", "first"),
            wt_netMHCpan_EL_rank=(NET_SCORE, smallest_score),
            wt_netMHCpan_EL_rank_pass=(NET_PASS, "max"),
            wt_MHCflurry_affinity_percentile=(FLURRY_SCORE, smallest_score),
            wt_MHCflurry_affinity_percentile_pass=(FLURRY_PASS, "max"),
            wt_both_pass=("both_pass", "max"),
            wt_either_pass=("either_pass", "max"),
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

    # Derive the expected denominator from the actual WT reference rather than
    # assuming a particular VR length or k-mer size. For corrected VR6 k=9 this
    # is 25 windows per allele; other VR/k configurations may differ.
    expected_windows_by_allele = (
        wt_reference.groupby("allele", observed=True)
        .size()
        .astype(int)
    )
    merged["expected_window_count"] = (
        merged["allele"].map(expected_windows_by_allele).astype("Int64")
    )

    merged["changed_window"] = (
        merged["wt_peptide"].notna()
        & merged["peptide"].notna()
        & merged["peptide"].ne(merged["wt_peptide"])
    )

    # Positive always means stronger predicted immunogenicity than WT.
    merged["netMHCpan_window_improvement"] = (
        merged["wt_netMHCpan_EL_rank"] - merged[NET_SCORE]
    )
    merged["MHCflurry_window_improvement"] = (
        merged["wt_MHCflurry_affinity_percentile"] - merged[FLURRY_SCORE]
    )
    # Identical peptide sequences represent no biological change and therefore
    # contribute an exact zero to the fixed-denominator mean.
    unchanged_matched = merged["wt_peptide"].notna() & ~merged["changed_window"]
    for prefix in ["netMHCpan", "MHCflurry"]:
        improvement = f"{prefix}_window_improvement"
        merged.loc[unchanged_matched, improvement] = 0.0

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
    Collapse peptide rows into a compact allele-by-variant feature table.

    Counts describe gained/lost threshold-passing peptide windows. Continuous
    summaries average across every matched window, with unchanged windows set
    to zero and positive values always indicating stronger immunogenicity.
    """
    mutation_info = (
        data.groupby(GROUP_COLUMNS, observed=True)
        .agg(VR_mutation=("VR_mutation", "first"))
        .reset_index()
    )

    mutation_info["mutation_count"] = (
        mutation_info["VR_mutation"].apply(count_mutations).astype(int)
    )

    if "changed_window" not in data.columns:
        raise ValueError(
            "WT-relative scoring requires a matched WT reference and changed-window columns."
        )

    aggregations = dict(
        matched_window_count=("wt_peptide_id", "count"),
        expected_window_count=("expected_window_count", "first"),
        changed_window_count=("changed_window", "sum"),
        netMHCpan_new_passed_count=("netMHCpan_new_pass_vs_WT", "sum"),
        netMHCpan_lost_passed_count=("netMHCpan_lost_pass_vs_WT", "sum"),
        netMHCpan_net_pass_change=("netMHCpan_pass_change_vs_WT", "sum"),
        MHCflurry_new_passed_count=("MHCflurry_new_pass_vs_WT", "sum"),
        MHCflurry_lost_passed_count=("MHCflurry_lost_pass_vs_WT", "sum"),
        MHCflurry_net_pass_change=("MHCflurry_pass_change_vs_WT", "sum"),
        both_new_passed_count=("both_new_pass_vs_WT", "sum"),
        both_lost_passed_count=("both_lost_pass_vs_WT", "sum"),
        both_net_pass_change=("both_pass_change_vs_WT", "sum"),
    )

    for prefix in ["netMHCpan", "MHCflurry"]:
        aggregations[f"{prefix}_mean_window_improvement"] = (
            f"{prefix}_window_improvement",
            "mean",
        )

    feature_scores = (
        data.groupby(GROUP_COLUMNS, observed=True)
        .agg(**aggregations)
        .reset_index()
    )

    scores = mutation_info.merge(
        feature_scores,
        on=GROUP_COLUMNS,
        how="left",
        validate="one_to_one",
    )

    incomplete = scores[
        scores["matched_window_count"] != scores["expected_window_count"]
    ]
    if not incomplete.empty:
        examples = incomplete[
            ["allele", "variant_id", "matched_window_count", "expected_window_count"]
        ].head().to_dict("records")
        raise ValueError(
            f"{len(incomplete):,} allele/variant groups do not have the dynamically "
            f"expected number of matched WT windows. Examples: {examples}"
        )

    integer_columns = [
        "mutation_count",
        "matched_window_count",
        "expected_window_count",
        "changed_window_count",
        "netMHCpan_new_passed_count",
        "netMHCpan_lost_passed_count",
        "netMHCpan_net_pass_change",
        "MHCflurry_new_passed_count",
        "MHCflurry_lost_passed_count",
        "MHCflurry_net_pass_change",
        "both_new_passed_count",
        "both_lost_passed_count",
        "both_net_pass_change",
    ]

    scores[integer_columns] = scores[integer_columns].fillna(0).astype(int)

    scores = scores.sort_values(["allele", "variant_id"]).reset_index(drop=True)

    return scores


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> int:
    args = parse_scoring_args(
        description="Create WT-relative variant-level MHC-I presentation scores.",
        default_output_root="data/output/variant_immunogenicity_scores_wt",
        variant_help="Variant MHC-I combined_annotated.tsv.",
        wt_help="Matching WT MHC-I combined_annotated.tsv.",
    )
    variant_input = args.variant_input.resolve()
    wt_input = args.wt_input.resolve()
    run_label = args.run_label or create_output_run_label(variant_input)

    print(f"Reading variant predictions from: {variant_input}")

    df = pd.read_csv(variant_input, sep="\t", low_memory=False)
    data = prepare_prediction_data(df)

    print(f"Reading WT predictions from: {wt_input}")
    wt_df = pd.read_csv(wt_input, sep="\t", low_memory=False)
    wt_data = prepare_prediction_data(wt_df, is_wt=True)
    wt_reference = prepare_wt_reference(wt_data)

    print(
        f"Prepared WT reference with {len(wt_reference):,} "
        "allele/window/k rows."
    )
    data = add_wt_relative_columns(data, wt_reference)

    allele_scores = create_allele_specific_scores(
        data,
        # False allows variants with only one or two valid peptide scores
        # to use the available values in mean_top3.
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
