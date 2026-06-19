#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import sparse
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests


MUTATION_PATTERN = re.compile(r"^([A-Za-z*])(\d+)([A-Za-z*])$")


# ---------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one-mutation-at-a-time peptide-level mixed-effects models "
            "with crossed random intercepts for variant and peptide window."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Combined pipeline TSV file."
    )

    parser.add_argument(
        "--outdir",
        required=True,
        type=Path,
        help="Output directory."
    )

    parser.add_argument(
        "--outcome",
        default="MHCflurry_affinity_percentile",
        help="Outcome column to model."
    )

    parser.add_argument(
        "--transform",
        choices=["none", "log", "neglog10_percentile"],
        default="neglog10_percentile",
        help=(
            "Outcome transformation. For percentile outcomes, "
            "'neglog10_percentile' makes larger values indicate stronger binding."
        )
    )

    parser.add_argument(
        "--min-variants",
        type=int,
        default=50,
        help="Minimum number of distinct variants carrying a mutation."
    )

    parser.add_argument(
        "--min-rows",
        type=int,
        default=100,
        help="Minimum peptide-row count containing a mutation."
    )

    parser.add_argument(
        "--min-windows",
        type=int,
        default=2,
        help="Minimum number of distinct peptide windows containing a mutation."
    )

    parser.add_argument(
        "--min-controls",
        type=int,
        default=100,
        help=(
            "Minimum number of mutation-absent peptide rows among windows "
            "overlapping the mutation position."
        )
    )

    parser.add_argument(
        "--include-mutation-count",
        action="store_true",
        help=(
            "Adjust for the total number of mutations in each peptide. "
            "Best treated as a sensitivity analysis."
        )
    )

    parser.add_argument(
        "--max-mutations",
        type=int,
        default=None,
        help=(
            "Optional maximum number of mutations to model, ranked by "
            "number of distinct variants carrying them."
        )
    )

    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.7,
        help="Absolute phi-correlation threshold for highlighted mutation pairs."
    )

    parser.add_argument(
        "--maxiter",
        type=int,
        default=200,
        help="Maximum optimizer iterations per mixed model."
    )

    return parser.parse_args()


def split_mutations(value: object) -> list[str]:
    """
    Convert a semicolon-separated mutation field into a clean list.

    Handles missing values, empty strings, '.', 'WT', and similar labels.
    """
    if pd.isna(value):
        return []

    text = str(value).strip()

    if text == "" or text.lower() in {"nan", "none", "wt", ".", "na"}:
        return []

    mutations = []

    for item in text.split(";"):
        mutation = item.strip()

        if mutation and MUTATION_PATTERN.match(mutation):
            mutations.append(mutation)

    return mutations


def mutation_position(mutation: str) -> int | None:
    match = MUTATION_PATTERN.match(mutation)

    if match is None:
        return None

    return int(match.group(2))


def transform_outcome(
    values: pd.Series,
    transformation: str
) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")

    if transformation == "none":
        return values

    if transformation == "log":
        positive_values = values.where(values > 0)
        return np.log(positive_values)

    if transformation == "neglog10_percentile":
        # Works whether the percentile is represented as 0-100.
        # Dividing by 100 adds only a constant relative to -log10(percentile),
        # but gives the transformed score a clear probability interpretation.
        clipped = values.clip(lower=1e-12, upper=100)
        return -np.log10(clipped / 100.0)

    raise ValueError(f"Unknown transformation: {transformation}")


def make_window_id(df: pd.DataFrame) -> pd.Series:
    """
    A window is defined by its reference coordinates and peptide length.

    Include another sequence/library identifier here if the input combines
    multiple proteins or separately indexed variable regions.
    """
    return (
        df["start"].astype("Int64").astype(str)
        + "_"
        + df["end"].astype("Int64").astype(str)
        + "_k"
        + df["k"].astype("Int64").astype(str)
    )


# ---------------------------------------------------------------------
# Mutation support summaries
# ---------------------------------------------------------------------

def build_mutation_long_table(
    df: pd.DataFrame,
    mutation_column: str
) -> pd.DataFrame:
    """
    Create one row per original observation per mutation.
    """
    working = df[
        [
            "row_id",
            "variant_id",
            "window_id",
            "start",
            "end",
            mutation_column
        ]
    ].copy()

    working["mutation"] = working[mutation_column].map(split_mutations)

    long_df = (
        working
        .explode("mutation")
        .dropna(subset=["mutation"])
    )

    if long_df.empty:
        return long_df

    long_df["position"] = long_df["mutation"].map(mutation_position)

    return long_df.dropna(subset=["position"])


def create_mutation_support_table(
    peptide_mutation_long: pd.DataFrame
) -> pd.DataFrame:
    if peptide_mutation_long.empty:
        return pd.DataFrame()

    support = (
        peptide_mutation_long
        .groupby(["mutation", "position"], as_index=False)
        .agg(
            peptide_rows=("row_id", "nunique"),
            distinct_variants=("variant_id", "nunique"),
            distinct_windows=("window_id", "nunique")
        )
        .sort_values(
            ["distinct_variants", "peptide_rows"],
            ascending=False
        )
    )

    return support


# ---------------------------------------------------------------------
# Variant-level co-occurrence and phi correlation
# ---------------------------------------------------------------------

def build_variant_mutation_matrix(
    df: pd.DataFrame,
    retained_mutations: list[str]
) -> tuple[sparse.csr_matrix, list[str], list[str]]:
    """
    Construct a sparse variant x mutation binary matrix from VR_mutation.

    Correlation is assessed at the variant level because repeated overlapping
    peptide rows should not inflate mutation co-occurrence counts.
    """
    variant_table = (
        df[["variant_id", "VR_mutation"]]
        .drop_duplicates(subset=["variant_id"])
        .reset_index(drop=True)
    )

    mutation_to_column = {
        mutation: index
        for index, mutation in enumerate(retained_mutations)
    }

    row_indices: list[int] = []
    column_indices: list[int] = []

    for row_index, mutation_field in enumerate(variant_table["VR_mutation"]):
        present = set(split_mutations(mutation_field))

        for mutation in present:
            column_index = mutation_to_column.get(mutation)

            if column_index is not None:
                row_indices.append(row_index)
                column_indices.append(column_index)

    values = np.ones(len(row_indices), dtype=np.int8)

    matrix = sparse.csr_matrix(
        (
            values,
            (row_indices, column_indices)
        ),
        shape=(len(variant_table), len(retained_mutations)),
        dtype=np.int8
    )

    # Guarantee binary values if duplicate mutation labels occurred.
    matrix.data[:] = 1

    return (
        matrix,
        variant_table["variant_id"].astype(str).tolist(),
        retained_mutations
    )


def calculate_phi_correlations(
    matrix: sparse.csr_matrix,
    mutation_names: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate pairwise phi correlations for binary mutation indicators.

    Phi is the Pearson correlation between two binary variables.
    """
    n_variants = matrix.shape[0]
    n_mutations = matrix.shape[1]

    if n_mutations == 0:
        return pd.DataFrame(), pd.DataFrame()

    counts = np.asarray(matrix.sum(axis=0)).ravel().astype(float)
    cooccurrence = (matrix.T @ matrix).toarray().astype(float)

    correlation = np.eye(n_mutations, dtype=float)

    pair_records = []

    for i in range(n_mutations):
        for j in range(i + 1, n_mutations):
            n11 = cooccurrence[i, j]
            n10 = counts[i] - n11
            n01 = counts[j] - n11
            n00 = n_variants - n11 - n10 - n01

            denominator = np.sqrt(
                (n11 + n10)
                * (n01 + n00)
                * (n11 + n01)
                * (n10 + n00)
            )

            phi = np.nan if denominator == 0 else (
                (n11 * n00 - n10 * n01) / denominator
            )

            correlation[i, j] = phi
            correlation[j, i] = phi

            pair_records.append(
                {
                    "mutation_1": mutation_names[i],
                    "mutation_2": mutation_names[j],
                    "phi": phi,
                    "abs_phi": abs(phi) if np.isfinite(phi) else np.nan,
                    "both_present": int(n11),
                    "mutation_1_only": int(n10),
                    "mutation_2_only": int(n01),
                    "neither_present": int(n00),
                    "mutation_1_variants": int(counts[i]),
                    "mutation_2_variants": int(counts[j])
                }
            )

    correlation_df = pd.DataFrame(
        correlation,
        index=mutation_names,
        columns=mutation_names
    )

    pairs_df = (
        pd.DataFrame(pair_records)
        .sort_values("abs_phi", ascending=False)
        .reset_index(drop=True)
    )

    return correlation_df, pairs_df


# ---------------------------------------------------------------------
# Mixed-effects modelling
# ---------------------------------------------------------------------

def fit_single_mutation_model(
    df: pd.DataFrame,
    mutation: str,
    position: int,
    maxiter: int,
    include_mutation_count: bool
) -> dict:
    """
    Fit one mixed-effects model for one mutation.

    Only peptide windows spanning the mutation's reference position are used.

    Fixed effect:
        mutation_present

    Optional fixed effects:
        C(k), when more than one peptide length is present
        peptide_mutation_count

    Crossed random intercepts:
        variant_id
        window_id
    """
    subset = df.loc[
        (df["start"] <= position)
        & (df["end"] >= position)
    ].copy()

    subset["mutation_present"] = subset["peptide_mutation_list"].map(
        lambda items: int(mutation in items)
    )

    subset = subset.dropna(
        subset=[
            "model_outcome",
            "variant_id",
            "window_id",
            "mutation_present"
        ]
    )

    n_present_rows = int(subset["mutation_present"].sum())
    n_absent_rows = int(len(subset) - n_present_rows)

    present_subset = subset.loc[subset["mutation_present"] == 1]
    absent_subset = subset.loc[subset["mutation_present"] == 0]

    result = {
        "mutation": mutation,
        "position": position,
        "n_rows_analyzed": len(subset),
        "n_present_rows": n_present_rows,
        "n_absent_rows": n_absent_rows,
        "n_present_variants": present_subset["variant_id"].nunique(),
        "n_absent_variants": absent_subset["variant_id"].nunique(),
        "n_present_windows": present_subset["window_id"].nunique(),
        "n_total_windows": subset["window_id"].nunique(),
        "present_mean": present_subset["model_outcome"].mean(),
        "absent_mean": absent_subset["model_outcome"].mean(),
        "coefficient": np.nan,
        "std_error": np.nan,
        "z_value": np.nan,
        "p_value": np.nan,
        "ci_lower": np.nan,
        "ci_upper": np.nan,
        "converged": False,
        "log_likelihood": np.nan,
        "residual_variance": np.nan,
        "variant_random_variance": np.nan,
        "window_random_variance": np.nan,
        "model_formula": "",
        "status": "not_fitted",
        "warning": ""
    }

    if subset["mutation_present"].nunique() < 2:
        result["status"] = "no_mutation_contrast"
        return result

    if subset["variant_id"].nunique() < 2:
        result["status"] = "insufficient_variants"
        return result

    if subset["window_id"].nunique() < 2:
        result["status"] = "insufficient_windows"
        return result

    fixed_terms = ["mutation_present"]

    if subset["k"].nunique() > 1:
        fixed_terms.append("C(k)")

    if include_mutation_count:
        fixed_terms.append("peptide_mutation_count")

    formula = "model_outcome ~ " + " + ".join(fixed_terms)
    result["model_formula"] = formula

    # Using a single top-level group and two variance components provides
    # crossed random intercepts rather than nesting windows inside variants.
    subset["_all_group"] = 1

    variance_components = {
        "variant": "0 + C(variant_id)",
        "window": "0 + C(window_id)"
    }

    captured_warnings = []

    try:
        with warnings.catch_warnings(record=True) as warning_records:
            warnings.simplefilter("always")

            model = smf.mixedlm(
                formula=formula,
                data=subset,
                groups=subset["_all_group"],
                re_formula="0",
                vc_formula=variance_components
            )

            fitted = model.fit(
                method="lbfgs",
                reml=False,
                maxiter=maxiter,
                disp=False
            )

            captured_warnings = [
                str(warning.message)
                for warning in warning_records
            ]

        coefficient = fitted.params.get("mutation_present", np.nan)
        standard_error = fitted.bse.get("mutation_present", np.nan)

        if np.isfinite(coefficient) and np.isfinite(standard_error):
            z_value = coefficient / standard_error
            p_value = 2 * norm.sf(abs(z_value))
            ci_lower = coefficient - 1.96 * standard_error
            ci_upper = coefficient + 1.96 * standard_error
        else:
            z_value = np.nan
            p_value = np.nan
            ci_lower = np.nan
            ci_upper = np.nan

        result.update(
            {
                "coefficient": coefficient,
                "std_error": standard_error,
                "z_value": z_value,
                "p_value": p_value,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "converged": bool(fitted.converged),
                "log_likelihood": fitted.llf,
                "residual_variance": fitted.scale,
                "status": "success"
            }
        )

        # vcomp order follows insertion order of vc_formula in current
        # statsmodels versions, but preserve NaN if unavailable.
        if hasattr(fitted, "vcomp") and len(fitted.vcomp) >= 2:
            result["variant_random_variance"] = fitted.vcomp[0]
            result["window_random_variance"] = fitted.vcomp[1]

        if captured_warnings:
            result["warning"] = " | ".join(sorted(set(captured_warnings)))

    except Exception as exc:
        result["status"] = "model_error"
        result["warning"] = f"{type(exc).__name__}: {exc}"

    return result


# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {args.input}")

    required_columns = {
        "variant_id",
        "start",
        "end",
        "k",
        "VR_mutation",
        "peptide_mutation",
        args.outcome
    }

    df = pd.read_csv(
        args.input,
        sep="\t",
        low_memory=False
    )

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    df = df.copy()
    df["row_id"] = np.arange(len(df))

    for column in ["start", "end", "k"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["model_outcome"] = transform_outcome(
        df[args.outcome],
        args.transform
    )

    df = df.dropna(
        subset=[
            "variant_id",
            "start",
            "end",
            "k",
            "model_outcome"
        ]
    ).copy()

    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)
    df["k"] = df["k"].astype(int)
    df["variant_id"] = df["variant_id"].astype(str)

    df["window_id"] = make_window_id(df)

    df["peptide_mutation_list"] = df["peptide_mutation"].map(
        split_mutations
    )

    df["peptide_mutation_count"] = df[
        "peptide_mutation_list"
    ].map(len)

    print(f"Rows retained: {len(df):,}")
    print(f"Variants: {df['variant_id'].nunique():,}")
    print(f"Windows: {df['window_id'].nunique():,}")

    # -------------------------------------------------------------
    # Mutation prevalence
    # -------------------------------------------------------------

    peptide_mutation_long = build_mutation_long_table(
        df,
        mutation_column="peptide_mutation"
    )

    support = create_mutation_support_table(
        peptide_mutation_long
    )

    if support.empty:
        raise ValueError(
            "No valid mutations were found in peptide_mutation."
        )

    support["passes_prevalence_filter"] = (
        (support["distinct_variants"] >= args.min_variants)
        & (support["peptide_rows"] >= args.min_rows)
        & (support["distinct_windows"] >= args.min_windows)
    )

    support_path = args.outdir / "mutation_support.tsv"

    support.to_csv(
        support_path,
        sep="\t",
        index=False
    )

    retained = support.loc[
        support["passes_prevalence_filter"]
    ].copy()

    if args.max_mutations is not None:
        retained = retained.head(args.max_mutations)

    retained_mutations = retained["mutation"].tolist()

    print(
        f"Mutations passing prevalence filters: "
        f"{len(retained_mutations):,}"
    )

    if not retained_mutations:
        raise ValueError(
            "No mutations passed the selected prevalence filters."
        )

    # -------------------------------------------------------------
    # Variant-level mutation correlation
    # -------------------------------------------------------------

    print("Calculating variant-level mutation correlations...")

    variant_matrix, _, correlation_mutations = (
        build_variant_mutation_matrix(
            df,
            retained_mutations
        )
    )

    correlation_matrix, correlation_pairs = (
        calculate_phi_correlations(
            variant_matrix,
            correlation_mutations
        )
    )

    correlation_matrix.to_csv(
        args.outdir / "mutation_phi_correlation_matrix.tsv",
        sep="\t",
        index=True
    )

    correlation_pairs.to_csv(
        args.outdir / "mutation_phi_correlation_pairs.tsv",
        sep="\t",
        index=False
    )

    high_correlation_pairs = correlation_pairs.loc[
        correlation_pairs["abs_phi"]
        >= args.correlation_threshold
    ]

    high_correlation_pairs.to_csv(
        args.outdir / "highly_correlated_mutation_pairs.tsv",
        sep="\t",
        index=False
    )

    print(
        f"Highly correlated pairs "
        f"(|phi| >= {args.correlation_threshold}): "
        f"{len(high_correlation_pairs):,}"
    )

    # Add strongest co-occurrence partner to the support table.
    if not correlation_pairs.empty:
        partner_1 = correlation_pairs[
            ["mutation_1", "mutation_2", "phi", "abs_phi"]
        ].rename(
            columns={
                "mutation_1": "mutation",
                "mutation_2": "correlation_partner"
            }
        )

        partner_2 = correlation_pairs[
            ["mutation_2", "mutation_1", "phi", "abs_phi"]
        ].rename(
            columns={
                "mutation_2": "mutation",
                "mutation_1": "correlation_partner"
            }
        )

        partner_table = (
            pd.concat([partner_1, partner_2], ignore_index=True)
            .sort_values("abs_phi", ascending=False)
            .drop_duplicates("mutation")
            .rename(
                columns={
                    "phi": "strongest_phi",
                    "abs_phi": "strongest_abs_phi"
                }
            )
        )

        support = support.merge(
            partner_table,
            on="mutation",
            how="left"
        )

        support.to_csv(
            support_path,
            sep="\t",
            index=False
        )

    # -------------------------------------------------------------
    # Fit one mixed model per mutation
    # -------------------------------------------------------------

    model_results = []

    for model_number, row in enumerate(
        retained.itertuples(index=False),
        start=1
    ):
        mutation = row.mutation
        position = int(row.position)

        print(
            f"[{model_number}/{len(retained)}] "
            f"Fitting {mutation} at position {position}"
        )

        result = fit_single_mutation_model(
            df=df,
            mutation=mutation,
            position=position,
            maxiter=args.maxiter,
            include_mutation_count=args.include_mutation_count
        )

        if result["n_absent_rows"] < args.min_controls:
            result["status"] = "insufficient_controls"
            result["coefficient"] = np.nan
            result["std_error"] = np.nan
            result["z_value"] = np.nan
            result["p_value"] = np.nan
            result["ci_lower"] = np.nan
            result["ci_upper"] = np.nan

        model_results.append(result)

        # Save progress after every model.
        pd.DataFrame(model_results).to_csv(
            args.outdir / "univariate_mixed_model_progress.tsv",
            sep="\t",
            index=False
        )

    results = pd.DataFrame(model_results)

    # -------------------------------------------------------------
    # Multiple-testing correction
    # -------------------------------------------------------------

    results["fdr_bh"] = np.nan
    results["significant_fdr_0_05"] = False

    valid_p_values = (
        results["p_value"].notna()
        & np.isfinite(results["p_value"])
    )

    if valid_p_values.any():
        reject, adjusted_p, _, _ = multipletests(
            results.loc[valid_p_values, "p_value"],
            alpha=0.05,
            method="fdr_bh"
        )

        results.loc[valid_p_values, "fdr_bh"] = adjusted_p
        results.loc[
            valid_p_values,
            "significant_fdr_0_05"
        ] = reject

    # Add prevalence and correlation metadata.
    result_metadata_columns = [
        "mutation",
        "peptide_rows",
        "distinct_variants",
        "distinct_windows"
    ]

    if "correlation_partner" in support.columns:
        result_metadata_columns.extend(
            [
                "correlation_partner",
                "strongest_phi",
                "strongest_abs_phi"
            ]
        )

    results = results.merge(
        support[result_metadata_columns],
        on="mutation",
        how="left"
    )

    results["absolute_coefficient"] = results[
        "coefficient"
    ].abs()

    results = results.sort_values(
        [
            "significant_fdr_0_05",
            "absolute_coefficient"
        ],
        ascending=[False, False]
    )

    results.to_csv(
        args.outdir / "univariate_mixed_model_results.tsv",
        sep="\t",
        index=False
    )

    significant_results = results.loc[
        results["significant_fdr_0_05"]
    ]

    significant_results.to_csv(
        args.outdir / "significant_mutations_fdr_0.05.tsv",
        sep="\t",
        index=False
    )

    print()
    print("Analysis complete.")
    print(f"Models attempted: {len(results):,}")
    print(
        "Successful models: "
        f"{(results['status'] == 'success').sum():,}"
    )
    print(
        "FDR-significant mutations: "
        f"{results['significant_fdr_0_05'].sum():,}"
    )
    print(f"Outputs saved to: {args.outdir}")


if __name__ == "__main__":
    main()


"""
python analysis/univariate_analysis.py \
  --input data/output/bigmhc/VR5_V3__k9/predictions_mapped.tsv \
  --outdir data/output/univariate_mixed_models \
  --outcome MHCflurry_affinity_percentile \
  --transform neglog10_percentile \
  --min-variants 50 \
  --min-rows 100 \
  --min-windows 2 \
  --min-controls 100
"""

"""
python scripts/univariate_mutation_mixed_models.py \
  --input data/output/combined/combined_predictions.tsv \
  --outdir data/output/univariate_netmhcpan_models \
  --outcome netMHCpan_EL_rank \
  --transform neglog10_percentile \
  --min-variants 50 \
  --min-rows 100 \
  --min-windows 2
  """