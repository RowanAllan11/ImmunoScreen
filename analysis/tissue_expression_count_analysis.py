# Tissue expression analysis, focused on the epitope count score variables, modelling them as categorical - each separate count has its own coefficient.

# WT-relative count-category tissue expression analysis
#
# Purpose
# -------
# Analyse allele-specific WT-relative epitope count-change features as
# categorical predictors of tissue expression.
#
# This script focuses only on the four *_net_pass_change columns:
#   - netMHCpan_net_pass_change
#   - MHCflurry_net_pass_change
#   - both_net_pass_change
#   - either_net_pass_change
#
# It creates:
#   1. Descriptive frequency tables for each count value
#   2. Frequency bar plots for each count predictor
#   3. Categorical OLS models:
#        y ~ C(net_pass_change)
#        y ~ C(net_pass_change) + mutation_count
#   4. One coefficient forest plot per score type
#
# Interpretation
# --------------
# Each category coefficient is the mean tissue-expression difference
# relative to the reference count, usually count = 0.
#
# Example:
#   coefficient for count = 2 in netMHCpan_net_pass_change
#   = mean difference in y for variants with net pass change 2
#     compared with variants with net pass change 0.

from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

TISSUE_FILE = Path(
    "data/input/expression/VR4/TIS00004_brain_relative_to_virus_fold_1.csv"
)

IMMUNOGENICITY_FILE = Path(
    "data/output/variant_immunogenicity_scores_wt/VR4_K9_H2-Dd/"
    "variant_immunogenicity_scores.tsv"
)

OUTPUT_DIR = Path(
    "data/output/wt_relative_count_category_analysis/VR4_K9_H2-Dd/brain"
)

# Tissue expression outcome column after merging.
OUTCOME_COLUMN = "y"

# Robust standard errors for OLS inference.
OLS_COV_TYPE = "HC3"

# Require this many variants in a category to estimate/display its coefficient.
# Sparse categories are still reported in descriptive frequency tables but are
# excluded from categorical regression to avoid unstable estimates.
MIN_CATEGORY_N = 20

# Reference category for categorical models. If 0 is not present, the script
# uses the most frequent category instead and records this in the output.
PREFERRED_REFERENCE_COUNT = 0

COUNT_FEATURES = [
    "netMHCpan_net_pass_change",
    "MHCflurry_net_pass_change",
    "both_net_pass_change",
    "either_net_pass_change",
]

SCORE_LABELS = {
    "netMHCpan_net_pass_change": "NetMHCpan net pass change",
    "MHCflurry_net_pass_change": "MHCflurry net pass change",
    "both_net_pass_change": "Both tools net pass change",
    "either_net_pass_change": "Either tool net pass change",
}

MODEL_LABELS = {
    "unadjusted": "Unadjusted",
    "adjusted_for_mutation_count": "Adjusted for mutation count",
}


# ------------------------------------------------------------------
# Output directories
# ------------------------------------------------------------------

TABLE_DIR = OUTPUT_DIR / "tables"
PLOT_DIR = OUTPUT_DIR / "plots"
FREQUENCY_PLOT_DIR = PLOT_DIR / "net_pass_change_frequencies"
FOREST_PLOT_DIR = PLOT_DIR / "count_category_forest_plots"

for directory in [
    OUTPUT_DIR,
    TABLE_DIR,
    PLOT_DIR,
    FREQUENCY_PLOT_DIR,
    FOREST_PLOT_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------

def safe_filename(value: object) -> str:
    """Convert a string to a safe filename stem."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def require_columns(df: pd.DataFrame, columns: list[str], source_name: str) -> None:
    """Raise a clear error if required columns are missing."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"{source_name} is missing required columns: "
            + ", ".join(missing)
        )


def coerce_count_column(series: pd.Series) -> pd.Series:
    """
    Convert a count-like column to integer values where possible.

    Returns pandas nullable integer dtype so missing values can be retained.
    """
    numeric = pd.to_numeric(series, errors="coerce")

    # Allow values like 1.0 but reject non-integer values such as 1.25.
    non_missing = numeric.dropna()
    non_integer_values = non_missing.loc[~np.isclose(non_missing, np.round(non_missing))]

    if not non_integer_values.empty:
        examples = non_integer_values.head().tolist()
        raise ValueError(
            "Count column contains non-integer values. "
            f"Examples: {examples}"
        )

    return numeric.round().astype("Int64")


def load_tissue_expression(path: Path) -> pd.DataFrame:
    """Load and validate tissue-expression data."""
    tissue_df = pd.read_csv(path)
    require_columns(tissue_df, ["gene_id", OUTCOME_COLUMN], "Tissue file")

    tissue_df[OUTCOME_COLUMN] = pd.to_numeric(
        tissue_df[OUTCOME_COLUMN],
        errors="coerce",
    )

    tissue_df = tissue_df.dropna(
        subset=["gene_id", OUTCOME_COLUMN]
    ).copy()

    tissue_df["gene_id"] = tissue_df["gene_id"].astype(str).str.strip()

    # Confirm one expression value per variant.
    expression_counts = tissue_df.groupby("gene_id")[OUTCOME_COLUMN].nunique()
    conflicting_variants = expression_counts[expression_counts > 1]

    if not conflicting_variants.empty:
        raise ValueError(
            f"{len(conflicting_variants)} variants have more than one unique "
            f"{OUTCOME_COLUMN} value. Filter the tissue file before running."
        )

    tissue_variant_df = (
        tissue_df[["gene_id", OUTCOME_COLUMN]]
        .drop_duplicates(subset=["gene_id"])
        .rename(columns={"gene_id": "variant_id"})
    )

    return tissue_variant_df


def load_immunogenicity_scores(path: Path) -> pd.DataFrame:
    """Load and validate allele-specific variant immunogenicity scores."""
    scores_df = pd.read_csv(path, sep="\t")
    require_columns(scores_df, ["variant_id", "mutation_count"], "Immunogenicity file")

    scores_df["variant_id"] = scores_df["variant_id"].astype(str).str.strip()

    if scores_df["variant_id"].duplicated().any():
        duplicate_count = scores_df.loc[
            scores_df["variant_id"].duplicated(keep=False),
            "variant_id",
        ].nunique()
        raise ValueError(
            f"The immunogenicity table contains {duplicate_count} duplicate "
            "variant IDs. This script expects one allele-specific table."
        )

    available_features = [feature for feature in COUNT_FEATURES if feature in scores_df.columns]
    if not available_features:
        raise ValueError(
            "None of the expected *_net_pass_change columns were found. "
            f"Expected one or more of: {', '.join(COUNT_FEATURES)}"
        )

    scores_df["mutation_count"] = pd.to_numeric(
        scores_df["mutation_count"],
        errors="coerce",
    )

    for feature in available_features:
        scores_df[feature] = coerce_count_column(scores_df[feature])

    return scores_df


def merge_inputs(
    tissue_df: pd.DataFrame,
    scores_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge tissue expression and variant immunogenicity scores."""
    analysis_df = tissue_df.merge(
        scores_df,
        on="variant_id",
        how="inner",
        validate="one_to_one",
    )

    if analysis_df.empty:
        raise ValueError("No variants matched between tissue and immunogenicity tables.")

    return analysis_df


def create_input_summary(
    tissue_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    analysis_df: pd.DataFrame,
    available_features: list[str],
) -> pd.DataFrame:
    """Create a one-row summary of input matching."""
    n_tissue = tissue_df["variant_id"].nunique()
    n_scores = scores_df["variant_id"].nunique()
    n_matched = analysis_df["variant_id"].nunique()

    return pd.DataFrame([
        {
            "tissue_file": str(TISSUE_FILE),
            "immunogenicity_file": str(IMMUNOGENICITY_FILE),
            "output_dir": str(OUTPUT_DIR),
            "outcome_column": OUTCOME_COLUMN,
            "n_tissue_variants": n_tissue,
            "n_score_variants": n_scores,
            "n_matched_variants": n_matched,
            "tissue_variants_without_scores": n_tissue - n_matched,
            "score_variants_without_tissue": n_scores - n_matched,
            "percent_tissue_variants_matched": 100 * n_matched / n_tissue,
            "available_count_features": ";".join(available_features),
            "min_category_n_for_regression": MIN_CATEGORY_N,
            "preferred_reference_count": PREFERRED_REFERENCE_COUNT,
        }
    ])


# ------------------------------------------------------------------
# Descriptive analysis
# ------------------------------------------------------------------

def create_count_frequency_table(
    analysis_df: pd.DataFrame,
    feature: str,
) -> pd.DataFrame:
    """Create frequency table for each integer count value."""
    temp = analysis_df[["variant_id", OUTCOME_COLUMN, feature]].dropna().copy()
    temp[feature] = coerce_count_column(temp[feature]).astype(int)

    grouped = (
        temp.groupby(feature, observed=True)
        .agg(
            n_variants=("variant_id", "nunique"),
            mean_y=(OUTCOME_COLUMN, "mean"),
            median_y=(OUTCOME_COLUMN, "median"),
            sd_y=(OUTCOME_COLUMN, "std"),
            min_y=(OUTCOME_COLUMN, "min"),
            max_y=(OUTCOME_COLUMN, "max"),
        )
        .reset_index()
        .rename(columns={feature: "count_value"})
        .sort_values("count_value")
    )

    grouped["score"] = feature
    grouped["score_label"] = SCORE_LABELS.get(feature, feature)
    grouped["percent_variants"] = 100 * grouped["n_variants"] / grouped["n_variants"].sum()

    columns = [
        "score",
        "score_label",
        "count_value",
        "n_variants",
        "percent_variants",
        "mean_y",
        "median_y",
        "sd_y",
        "min_y",
        "max_y",
    ]

    return grouped[columns]


def plot_count_frequency(
    frequency_df: pd.DataFrame,
    feature: str,
    output_file: Path,
) -> None:
    """Plot frequency of each count category for a score."""
    plot_df = frequency_df.loc[frequency_df["score"] == feature].copy()
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    x_labels = plot_df["count_value"].astype(str).tolist()
    x_positions = np.arange(len(x_labels))

    ax.bar(
        x_positions,
        plot_df["n_variants"].to_numpy(),
        edgecolor="black",
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Net pass change count")
    ax.set_ylabel("Number of variants")
    ax.set_title(f"{SCORE_LABELS.get(feature, feature)} frequency")

    ax.axvline(
        x=x_labels.index("0") if "0" in x_labels else -1,
        linestyle="--",
        linewidth=1,
    )

    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------
# Categorical regression
# ------------------------------------------------------------------

def select_reference_count(model_data: pd.DataFrame, feature: str) -> int:
    """Use 0 as reference if available; otherwise use the most frequent count."""
    counts = model_data[feature].value_counts().sort_index()

    if PREFERRED_REFERENCE_COUNT in counts.index:
        return int(PREFERRED_REFERENCE_COUNT)

    return int(counts.idxmax())


def prepare_categorical_model_data(
    analysis_df: pd.DataFrame,
    feature: str,
    adjust_for_mutation_count: bool,
) -> tuple[pd.DataFrame, int, list[int], dict[int, int]]:
    """
    Prepare data for categorical OLS.

    Sparse categories are excluded from regression based on MIN_CATEGORY_N.
    """
    columns = ["variant_id", OUTCOME_COLUMN, feature]
    if adjust_for_mutation_count:
        columns.append("mutation_count")

    model_data = analysis_df[columns].dropna().copy()
    model_data[feature] = coerce_count_column(model_data[feature]).astype(int)

    category_counts = model_data[feature].value_counts().sort_index()
    retained_categories = [
        int(category)
        for category, n in category_counts.items()
        if n >= MIN_CATEGORY_N
    ]

    if len(retained_categories) < 2:
        raise ValueError(
            f"{feature} has fewer than two categories with at least "
            f"{MIN_CATEGORY_N} variants."
        )

    model_data = model_data.loc[model_data[feature].isin(retained_categories)].copy()
    reference_count = select_reference_count(model_data, feature)

    if reference_count not in retained_categories:
        reference_count = int(model_data[feature].value_counts().idxmax())

    retained_categories = sorted(retained_categories)
    category_n = model_data[feature].value_counts().to_dict()
    category_n = {int(k): int(v) for k, v in category_n.items()}

    return model_data, reference_count, retained_categories, category_n


def fit_count_category_ols(
    analysis_df: pd.DataFrame,
    feature: str,
    adjust_for_mutation_count: bool,
) -> tuple[list[dict], dict]:
    """
    Fit OLS with count value as categorical predictor.

    Model:
        y ~ C(count)
        y ~ C(count) + mutation_count

    Coefficients are returned for every non-reference count category.
    """
    model_data, reference_count, retained_categories, category_n = prepare_categorical_model_data(
        analysis_df=analysis_df,
        feature=feature,
        adjust_for_mutation_count=adjust_for_mutation_count,
    )

    # Create dummy variables with clear names that preserve the numeric count.
    category_series = model_data[feature].astype(int)
    dummy_df = pd.get_dummies(
        category_series,
        prefix="count",
        prefix_sep="_",
        dtype=float,
    )

    reference_column = f"count_{reference_count}"
    if reference_column not in dummy_df.columns:
        raise ValueError(
            f"Reference category {reference_count} for {feature} was not found "
            "after creating dummy variables."
        )

    dummy_df = dummy_df.drop(columns=[reference_column])

    # Sort dummy columns by their numeric category value.
    dummy_columns = sorted(
        dummy_df.columns,
        key=lambda value: int(value.replace("count_", "")),
    )
    dummy_df = dummy_df[dummy_columns]

    X_parts = [dummy_df]

    if adjust_for_mutation_count:
        mutation_count = pd.to_numeric(
            model_data["mutation_count"],
            errors="coerce",
        ).astype(float)
        X_parts.append(mutation_count.rename("mutation_count"))

    X = pd.concat(X_parts, axis=1)
    X = sm.add_constant(X, has_constant="add")

    y = pd.to_numeric(model_data[OUTCOME_COLUMN], errors="coerce").astype(float)

    model = sm.OLS(y, X).fit(cov_type=OLS_COV_TYPE)
    conf_int = model.conf_int()

    model_name = (
        "adjusted_for_mutation_count"
        if adjust_for_mutation_count
        else "unadjusted"
    )

    results = []

    for dummy_column in dummy_columns:
        count_value = int(dummy_column.replace("count_", ""))

        results.append(
            {
                "score": feature,
                "score_label": SCORE_LABELS.get(feature, feature),
                "model": model_name,
                "outcome": OUTCOME_COLUMN,
                "reference_count": reference_count,
                "count_value": count_value,
                "category_n": category_n.get(count_value, 0),
                "reference_n": category_n.get(reference_count, 0),
                "n_variants_model": int(model.nobs),
                "coefficient_vs_reference": model.params[dummy_column],
                "robust_std_error": model.bse[dummy_column],
                "t_value": model.tvalues[dummy_column],
                "p_value": model.pvalues[dummy_column],
                "ci_lower": conf_int.loc[dummy_column, 0],
                "ci_upper": conf_int.loc[dummy_column, 1],
                "intercept_reference_mean_estimate": model.params["const"],
                "r_squared": model.rsquared,
                "adjusted_r_squared": model.rsquared_adj,
                "aic": model.aic,
                "bic": model.bic,
                "mutation_count_coefficient": model.params.get("mutation_count", np.nan),
                "mutation_count_p_value": model.pvalues.get("mutation_count", np.nan),
                "interpretation": (
                    f"Estimated difference in {OUTCOME_COLUMN} for count "
                    f"{count_value} compared with reference count "
                    f"{reference_count}."
                ),
            }
        )

    model_summary = {
        "score": feature,
        "score_label": SCORE_LABELS.get(feature, feature),
        "model": model_name,
        "reference_count": reference_count,
        "reference_n": category_n.get(reference_count, 0),
        "retained_categories": ";".join(str(x) for x in retained_categories),
        "excluded_sparse_categories": ";".join(
            str(int(category))
            for category, n in analysis_df[feature].dropna().astype(int).value_counts().sort_index().items()
            if n < MIN_CATEGORY_N
        ),
        "n_variants_model": int(model.nobs),
        "r_squared": model.rsquared,
        "adjusted_r_squared": model.rsquared_adj,
        "aic": model.aic,
        "bic": model.bic,
        "mutation_count_coefficient": model.params.get("mutation_count", np.nan),
        "mutation_count_p_value": model.pvalues.get("mutation_count", np.nan),
    }

    return results, model_summary


# ------------------------------------------------------------------
# Forest plots
# ------------------------------------------------------------------

def plot_category_forest(
    regression_df: pd.DataFrame,
    feature: str,
    output_file: Path,
) -> None:
    """Create one forest plot for one count feature."""
    plot_df = regression_df.loc[regression_df["score"] == feature].copy()

    if plot_df.empty:
        return

    # Preserve count order, excluding the reference because it has no coefficient.
    count_values = sorted(plot_df["count_value"].unique())
    base_positions = {
        count_value: len(count_values) - 1 - index
        for index, count_value in enumerate(count_values)
    }

    model_offsets = {
        "unadjusted": 0.13,
        "adjusted_for_mutation_count": -0.13,
    }

    model_markers = {
        "unadjusted": "o",
        "adjusted_for_mutation_count": "s",
    }

    fig, ax = plt.subplots(
        figsize=(9, max(4.5, len(count_values) * 0.75))
    )

    for model_name in ["unadjusted", "adjusted_for_mutation_count"]:
        model_df = plot_df.loc[plot_df["model"] == model_name].copy()
        if model_df.empty:
            continue

        model_df = model_df.sort_values("count_value")

        y_positions = [
            base_positions[count_value] + model_offsets[model_name]
            for count_value in model_df["count_value"]
        ]

        lower_errors = model_df["coefficient_vs_reference"] - model_df["ci_lower"]
        upper_errors = model_df["ci_upper"] - model_df["coefficient_vs_reference"]

        ax.errorbar(
            model_df["coefficient_vs_reference"],
            y_positions,
            xerr=np.vstack([lower_errors, upper_errors]),
            fmt=model_markers[model_name],
            capsize=4,
            markersize=6,
            linewidth=1.3,
            label=MODEL_LABELS[model_name],
        )

    ax.axvline(0, linestyle="--", linewidth=1)

    y_ticks = [base_positions[count_value] for count_value in count_values]

    # Include n per category in y-axis label.
    category_n_map = (
        plot_df.drop_duplicates("count_value")
        .set_index("count_value")["category_n"]
        .to_dict()
    )

    y_labels = [
        f"Count {count_value} (n={category_n_map.get(count_value, 0):,})"
        for count_value in count_values
    ]

    # Get reference count from first row for the title/subtitle.
    reference_count = int(plot_df["reference_count"].iloc[0])
    reference_n = int(plot_df["reference_n"].iloc[0])

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel(
        f"Difference in {OUTCOME_COLUMN} vs count {reference_count} "
        "(95% CI)"
    )
    ax.set_title(
        f"{SCORE_LABELS.get(feature, feature)} as categorical predictor\n"
        f"Reference: count {reference_count} (n={reference_n:,})"
    )

    ax.legend(title="Model", frameon=False, loc="best")
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------
# Manifest
# ------------------------------------------------------------------

def write_manifest(paths: list[Path]) -> None:
    """Write a simple output manifest."""
    manifest_df = pd.DataFrame(
        [{"output_file": str(path)} for path in paths]
    )
    manifest_df.to_csv(
        OUTPUT_DIR / "OUTPUT_MANIFEST.tsv",
        sep="\t",
        index=False,
    )


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> int:
    print(f"Reading tissue expression from: {TISSUE_FILE}")
    tissue_df = load_tissue_expression(TISSUE_FILE)

    print(f"Reading immunogenicity scores from: {IMMUNOGENICITY_FILE}")
    scores_df = load_immunogenicity_scores(IMMUNOGENICITY_FILE)

    available_features = [
        feature for feature in COUNT_FEATURES if feature in scores_df.columns
    ]

    print("Available WT-relative count predictors:")
    for feature in available_features:
        print(f"  - {feature}")

    analysis_df = merge_inputs(tissue_df, scores_df)

    # Keep only the needed columns in the merged output.
    keep_columns = [
        "variant_id",
        OUTCOME_COLUMN,
        "mutation_count",
    ] + available_features

    if "VR_mutation" in analysis_df.columns:
        keep_columns.insert(2, "VR_mutation")

    if "allele" in analysis_df.columns:
        keep_columns.insert(1, "allele")

    analysis_subset = analysis_df[keep_columns].copy()

    written_files: list[Path] = []

    input_summary = create_input_summary(
        tissue_df=tissue_df,
        scores_df=scores_df,
        analysis_df=analysis_df,
        available_features=available_features,
    )
    input_summary_file = TABLE_DIR / "analysis_input_summary.tsv"
    input_summary.to_csv(input_summary_file, sep="\t", index=False)
    written_files.append(input_summary_file)

    merged_file = TABLE_DIR / "tissue_expression_count_change_merged.tsv"
    analysis_subset.to_csv(merged_file, sep="\t", index=False)
    written_files.append(merged_file)

    # Descriptive frequency tables and plots.
    all_frequency_tables = []

    for feature in available_features:
        frequency_df = create_count_frequency_table(analysis_df, feature)
        all_frequency_tables.append(frequency_df)

        feature_frequency_file = (
            TABLE_DIR / f"{safe_filename(feature)}_frequency_table.tsv"
        )
        frequency_df.to_csv(feature_frequency_file, sep="\t", index=False)
        written_files.append(feature_frequency_file)

        frequency_plot_file = (
            FREQUENCY_PLOT_DIR / f"{safe_filename(feature)}_frequency.png"
        )
        plot_count_frequency(
            frequency_df=frequency_df,
            feature=feature,
            output_file=frequency_plot_file,
        )
        written_files.append(frequency_plot_file)

    combined_frequency_df = pd.concat(all_frequency_tables, ignore_index=True)
    combined_frequency_file = TABLE_DIR / "net_pass_change_frequency_summary.tsv"
    combined_frequency_df.to_csv(combined_frequency_file, sep="\t", index=False)
    written_files.append(combined_frequency_file)

    # Categorical regression.
    regression_results = []
    model_summaries = []

    for feature in available_features:
        for adjust_for_mutation_count in [False, True]:
            try:
                results, model_summary = fit_count_category_ols(
                    analysis_df=analysis_df,
                    feature=feature,
                    adjust_for_mutation_count=adjust_for_mutation_count,
                )
                regression_results.extend(results)
                model_summaries.append(model_summary)
            except ValueError as error:
                print(f"Skipping {feature}: {error}")

    regression_df = pd.DataFrame(regression_results)
    model_summary_df = pd.DataFrame(model_summaries)

    regression_file = TABLE_DIR / "count_category_linear_regression_coefficients.tsv"
    regression_df.to_csv(regression_file, sep="\t", index=False)
    written_files.append(regression_file)

    model_summary_file = TABLE_DIR / "count_category_model_summary.tsv"
    model_summary_df.to_csv(model_summary_file, sep="\t", index=False)
    written_files.append(model_summary_file)

    # Separate forest plot for each of the four score types.
    for feature in available_features:
        forest_file = (
            FOREST_PLOT_DIR / f"{safe_filename(feature)}_category_forest_plot.png"
        )
        plot_category_forest(
            regression_df=regression_df,
            feature=feature,
            output_file=forest_file,
        )
        written_files.append(forest_file)

    write_manifest(written_files)

    print("\nMatched variants:")
    print(f"  Tissue variants: {tissue_df['variant_id'].nunique():,}")
    print(f"  Score variants:  {scores_df['variant_id'].nunique():,}")
    print(f"  Matched:         {analysis_df['variant_id'].nunique():,}")

    print("\nFrequency summary saved to:")
    print(f"  {combined_frequency_file}")

    print("\nCategorical regression coefficients saved to:")
    print(f"  {regression_file}")

    print("\nForest plots saved to:")
    for feature in available_features:
        print(
            "  "
            + str(
                FOREST_PLOT_DIR
                / f"{safe_filename(feature)}_category_forest_plot.png"
            )
        )

    print("\nOutput manifest:")
    print(f"  {OUTPUT_DIR / 'OUTPUT_MANIFEST.tsv'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
