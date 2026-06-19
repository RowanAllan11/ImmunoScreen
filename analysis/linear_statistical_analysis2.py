from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


MATCH_COLUMNS = ["allele", "start", "end", "k"]


def split_mutations(value: object) -> list[str]:
    """Split a semicolon-separated mutation annotation."""
    if pd.isna(value):
        return []

    return [
        mutation.strip()
        for mutation in str(value).split(";")
        if mutation.strip()
    ]


def safe_name(value: str) -> str:
    """Make a string safe for use in filenames."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def transform_score(series: pd.Series) -> pd.Series:
    """
    Convert a percentile/rank to an immunogenicity score.

    Higher transformed values indicate stronger predicted immunogenicity.
    """
    numeric = pd.to_numeric(series, errors="coerce")

    # Avoid log10(0).
    numeric = numeric.clip(lower=1e-10)

    return -np.log10(numeric)


def create_variant_summaries(
    df: pd.DataFrame,
    threshold: float,
    include_wt_relative: bool,
) -> pd.DataFrame:
    """
    Create variant-level summary outcomes.

    Top-three WT-relative scores are calculated using the three strongest
    absolute variant peptide scores.
    """
    group_columns = ["allele", "variant_id"]

    # Check that each variant has one consistent mutation annotation.
    mutation_counts = (
        df.groupby(group_columns)["VR_mutation"]
        .nunique(dropna=False)
    )

    inconsistent = mutation_counts[mutation_counts > 1]

    if not inconsistent.empty:
        raise ValueError(
            f"{len(inconsistent):,} allele-variant groups have inconsistent "
            "VR_mutation annotations."
        )

    # General summaries across all peptide windows.
    summaries = (
        df.groupby(group_columns, as_index=False)
        .agg(
            VR_mutation=("VR_mutation", "first"),
            peptide_count=("score", "size"),
            strongest_score=("score", "max"),
            mean_all_score=("score", "mean"),
            passing_count=("passes_threshold", "sum"),
        )
    )

    # Select the three strongest peptide windows per allele and variant.
    top_three = (
        df.sort_values(
            group_columns + ["score"],
            ascending=[True, True, False],
        )
        .groupby(group_columns, as_index=False)
        .head(3)
    )

    top_three_summary = (
        top_three.groupby(group_columns, as_index=False)
        .agg(
            mean_top3_score=("score", "mean"),
        )
    )

    summaries = summaries.merge(
        top_three_summary,
        on=group_columns,
        how="left",
        validate="one_to_one",
    )

    if include_wt_relative:
        # Largest improvement over the matched WT window.
        relative_summaries = (
            df.groupby(group_columns, as_index=False)
            .agg(
                strongest_wt_delta=("wt_delta", "max"),
                mean_all_wt_delta=("wt_delta", "mean"),
                passing_count_wt_difference=(
                    "passing_difference",
                    "sum",
                ),
            )
        )

        # Mean WT-relative difference among the variant's strongest 3 windows.
        top_three_relative = (
            top_three.groupby(group_columns, as_index=False)
            .agg(
                mean_top3_wt_delta=("wt_delta", "mean"),
            )
        )

        summaries = summaries.merge(
            relative_summaries,
            on=group_columns,
            how="left",
            validate="one_to_one",
        )

        summaries = summaries.merge(
            top_three_relative,
            on=group_columns,
            how="left",
            validate="one_to_one",
        )

    return summaries


def fit_mutation_model(
    allele_df: pd.DataFrame,
    outcome: str,
    min_prevalence: float,
    min_mutation_count: int,
) -> tuple[sm.regression.linear_model.RegressionResultsWrapper, pd.DataFrame]:
    """Fit an allele-specific OLS model with HC3 robust standard errors."""
    model_df = allele_df.dropna(subset=[outcome]).copy()

    mutation_matrix = (
        model_df["VR_mutation"]
        .apply(split_mutations)
        .str.join("|")
        .str.get_dummies(sep="|")
    )

    mutation_counts = mutation_matrix.sum(axis=0)
    mutation_prevalence = mutation_matrix.mean(axis=0)

    retained = mutation_counts.index[
        (mutation_counts >= min_mutation_count)
        & (mutation_prevalence >= min_prevalence)
    ]

    mutation_matrix = mutation_matrix[retained]

    if mutation_matrix.shape[1] == 0:
        raise ValueError(
            "No mutations passed the prevalence filters. "
            "Reduce --min-prevalence or --min-mutation-count."
        )

    # Remove any constant mutation columns.
    mutation_matrix = mutation_matrix.loc[
        :,
        mutation_matrix.nunique(axis=0) > 1,
    ]

    if mutation_matrix.shape[1] == 0:
        raise ValueError(
            "No variable mutation columns remained after filtering."
        )

    X = mutation_matrix.astype(float)
    X = sm.add_constant(X, has_constant="add")

    y = model_df[outcome].astype(float)

    model = sm.OLS(y, X).fit(cov_type="HC3")

    confidence_intervals = model.conf_int()

    results = pd.DataFrame(
        {
            "term": model.params.index,
            "coefficient": model.params.values,
            "std_error": model.bse.values,
            "t_value": model.tvalues.values,
            "p_value": model.pvalues.values,
            "ci_lower": confidence_intervals[0].values,
            "ci_upper": confidence_intervals[1].values,
        }
    )

    results["mutation_count"] = results["term"].map(
        mutation_counts
    )
    results["mutation_prevalence"] = results["term"].map(
        mutation_prevalence
    )

    # Interpretation for outcomes measured on a -log10 scale.
    #
    # A coefficient of 0.30 corresponds to:
    # 10^0.30 = approximately 2-fold improvement in percentile/rank.
    results["fold_improvement"] = 10 ** results["coefficient"]
    results["fold_improvement_ci_lower"] = 10 ** results["ci_lower"]
    results["fold_improvement_ci_upper"] = 10 ** results["ci_upper"]

    # Equivalent ratio on the original percentile/rank scale.
    # Values below 1 indicate a lower/better percentile.
    results["percentile_ratio"] = 10 ** (-results["coefficient"])

    mutation_rows = results["term"] != "const"

    results["q_value"] = np.nan

    if mutation_rows.any():
        results.loc[mutation_rows, "q_value"] = multipletests(
            results.loc[mutation_rows, "p_value"],
            method="fdr_bh",
        )[1]

    results["significant_fdr_05"] = results["q_value"] < 0.05

    results = results.sort_values(
        ["q_value", "p_value"],
        na_position="last",
    )

    return model, results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Allele-specific variant-level regression of transformed "
            "immunogenicity scores on variant mutations."
        )
    )

    parser.add_argument(
        "--variant-input",
        type=Path,
        default=Path(
            "data/output/bigmhc/VR5_V3__k9/"
            "predictions_mapped.tsv"
        ),
    )
    parser.add_argument(
        "--wt-input",
        type=Path,
        default=Path(
            "data/output/bigmhc/AAV9_WT__k9/"
            "predictions_mapped.tsv"
        ),
    )
    parser.add_argument(
        "--score-column",
        choices=[
            "netMHCpan_EL_rank",
            "MHCflurry_affinity_percentile",
        ],
        default="netMHCpan_EL_rank",
    )
    parser.add_argument(
        "--outcome",
        choices=[
            "strongest_score",
            "mean_top3_score",
            "mean_all_score",
            "passing_count",
            "strongest_wt_delta",
            "mean_top3_wt_delta",
            "mean_all_wt_delta",
            "passing_count_wt_difference",
        ],
        default="mean_top3_score",
    )
    parser.add_argument(
        "--pass-threshold",
        type=float,
        default=2.0,
        help=(
            "Raw percentile/rank threshold used for passing_count. "
            "A peptide passes when its raw score is <= this value."
        ),
    )
    parser.add_argument(
        "--include-wt-relative",
        action="store_true",
        help=(
            "Match variant and WT windows using allele, start, end and k, "
            "and calculate WT-relative outcomes."
        ),
    )
    parser.add_argument(
        "--expected-peptides",
        type=int,
        default=24,
        help="Expected peptide windows per allele and variant.",
    )
    parser.add_argument(
        "--min-prevalence",
        type=float,
        default=0.01,
        help="Minimum proportion of variants carrying a mutation.",
    )
    parser.add_argument(
        "--min-mutation-count",
        type=int,
        default=40,
        help="Minimum number of variants carrying a mutation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/output/variant_level_linear_regression"
        ),
    )

    args = parser.parse_args()

    if (
        args.outcome.endswith("wt_delta")
        or args.outcome == "passing_count_wt_difference"
    ) and not args.include_wt_relative:
        raise ValueError(
            f"Outcome '{args.outcome}' requires --include-wt-relative."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    variant_df = pd.read_csv(
        args.variant_input,
        sep="\t",
        low_memory=False,
    )

    required_columns = {
        "variant_id",
        "allele",
        "start",
        "end",
        "k",
        "VR_mutation",
        args.score_column,
    }

    missing = required_columns - set(variant_df.columns)

    if missing:
        raise ValueError(
            f"Variant input is missing columns: {sorted(missing)}"
        )

    variant_df["raw_score"] = pd.to_numeric(
        variant_df[args.score_column],
        errors="coerce",
    )

    variant_df["score"] = transform_score(
        variant_df["raw_score"]
    )

    variant_df["passes_threshold"] = (
        variant_df["raw_score"] <= args.pass_threshold
    )

    variant_df = variant_df.dropna(
        subset=[
            "variant_id",
            "allele",
            "start",
            "end",
            "k",
            "score",
        ]
    ).copy()

    if args.include_wt_relative:
        wt_df = pd.read_csv(
            args.wt_input,
            sep="\t",
            low_memory=False,
        )

        wt_required = set(MATCH_COLUMNS + [args.score_column])
        wt_missing = wt_required - set(wt_df.columns)

        if wt_missing:
            raise ValueError(
                f"WT input is missing columns: {sorted(wt_missing)}"
            )

        wt_df["wt_raw_score"] = pd.to_numeric(
            wt_df[args.score_column],
            errors="coerce",
        )

        wt_df["wt_score"] = transform_score(
            wt_df["wt_raw_score"]
        )

        wt_df["wt_passes_threshold"] = (
            wt_df["wt_raw_score"] <= args.pass_threshold
        )

        wt_df = wt_df[
            MATCH_COLUMNS
            + [
                "wt_score",
                "wt_raw_score",
                "wt_passes_threshold",
            ]
        ].dropna(
            subset=MATCH_COLUMNS + ["wt_score"]
        )

        if wt_df.duplicated(MATCH_COLUMNS).any():
            duplicates = wt_df.loc[
                wt_df.duplicated(
                    MATCH_COLUMNS,
                    keep=False,
                ),
                MATCH_COLUMNS,
            ]

            raise ValueError(
                "WT input contains duplicate rows for the matching keys "
                f"{MATCH_COLUMNS}.\n"
                f"{duplicates.head()}"
            )

        variant_df = variant_df.merge(
            wt_df,
            on=MATCH_COLUMNS,
            how="left",
            validate="many_to_one",
        )

        unmatched = variant_df["wt_score"].isna().sum()

        if unmatched:
            raise ValueError(
                f"{unmatched:,} variant rows could not be matched to a WT "
                f"window using {MATCH_COLUMNS}."
            )

        # Positive delta means stronger predicted immunogenicity than WT.
        variant_df["wt_delta"] = (
            variant_df["score"] - variant_df["wt_score"]
        )

        variant_df["passing_difference"] = (
            variant_df["passes_threshold"].astype(int)
            - variant_df["wt_passes_threshold"].astype(int)
        )

    summaries = create_variant_summaries(
        df=variant_df,
        threshold=args.pass_threshold,
        include_wt_relative=args.include_wt_relative,
    )

    summaries.to_csv(
        args.output_dir / "variant_level_summaries.tsv",
        sep="\t",
        index=False,
    )

    peptide_count_table = (
        summaries["peptide_count"]
        .value_counts()
        .sort_index()
    )

    print("\nPeptides per allele-variant group:")
    print(peptide_count_table.to_string())

    unexpected_counts = summaries[
        summaries["peptide_count"] != args.expected_peptides
    ]

    if not unexpected_counts.empty:
        print(
            f"\nWarning: {len(unexpected_counts):,} allele-variant groups "
            f"do not contain exactly {args.expected_peptides} peptides."
        )

        unexpected_counts.to_csv(
            args.output_dir / "unexpected_peptide_counts.tsv",
            sep="\t",
            index=False,
        )

    for allele, allele_df in summaries.groupby("allele"):
        allele_name = safe_name(str(allele))
        allele_output = args.output_dir / allele_name
        allele_output.mkdir(parents=True, exist_ok=True)

        allele_df.to_csv(
            allele_output / "variant_level_data.tsv",
            sep="\t",
            index=False,
        )

        model, results = fit_mutation_model(
            allele_df=allele_df,
            outcome=args.outcome,
            min_prevalence=args.min_prevalence,
            min_mutation_count=args.min_mutation_count,
        )

        results.insert(0, "allele", allele)
        results.insert(1, "outcome", args.outcome)
        results.insert(2, "score_column", args.score_column)

        results.to_csv(
            allele_output / "linear_regression_results.tsv",
            sep="\t",
            index=False,
        )

        with open(
            allele_output / "linear_regression_summary.txt",
            "w",
            encoding="utf-8",
        ) as handle:
            handle.write(model.summary().as_text())

        print(f"\nAllele: {allele}")
        print(f"Variants modelled: {int(model.nobs):,}")
        print(f"Outcome: {args.outcome}")
        print(f"R-squared: {model.rsquared:.4f}")
        print(
            "Mutation terms retained: "
            f"{len(results) - 1:,}"
        )

    print(f"\nOutputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()


"""
python analysis/linear_statistical_analysis2.py \
  --score-column netMHCpan_EL_rank \
  --outcome mean_top3_score \
  --output-dir data/output/linear_regression/VR5_V3__k9_top3_net \
"""

"""
python analysis/linear_statistical_analysis2.py \
  --score-column MHCflurry_affinity_percentile \
  --outcome mean_top3_score \
  --output-dir data/output/linear_regression/VR5_V3__k9_top3_mhcflurry
"""

"""
python analysis/linear_statistical_analysis2.py \
  --score-column netMHCpan_EL_rank \
  --outcome mean_top3_wt_delta \
  --include-wt-relative \
  --output-dir data/output/linear_regression/VR5_V3__k9_top3_net_wt_relative
"""