# Tissue expression analysis
# DF with columns sequnce, gene_id, y, endpoint_type, source, source_alias, split
# gene_id corresponds to variant_id from bigmhc output and variant_immunogenicity_scores
# y is the expression value for the given endpoint_type and source

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, mannwhitneyu
import statsmodels.api as sm


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

TISSUE_FILE = Path("data/input/expression/spleen_relative_to_virus_fold_1.csv")

IMMUNOGENICITY_FILE = Path(
    "data/output/variant_immunogenicity_scores/VR5_V3__k9/"
    "variant_immunogenicity_scores.tsv"
)

OUTPUT_DIR = Path("data/output/tissue_expression_analysis/VR5_V3__k9/spleen")
SCATTER_DIR = OUTPUT_DIR / "immunogenicity_scatter_plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SCATTER_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# Load tissue-expression data
# ------------------------------------------------------------------

tissue_df = pd.read_csv(TISSUE_FILE)

required_tissue_columns = ["gene_id", "y"]

missing_tissue_columns = [
    column
    for column in required_tissue_columns
    if column not in tissue_df.columns
]

if missing_tissue_columns:
    raise ValueError(
        "Missing tissue-expression columns: "
        + ", ".join(missing_tissue_columns)
    )

tissue_df["y"] = pd.to_numeric(
    tissue_df["y"],
    errors="coerce",
)

tissue_df = tissue_df.dropna(
    subset=["gene_id", "y"]
).copy()

# gene_id corresponds to variant_id
tissue_df["gene_id"] = tissue_df["gene_id"].astype(str)


# ------------------------------------------------------------------
# Check that there is one expression value per variant
# ------------------------------------------------------------------

y_counts = tissue_df.groupby("gene_id")["y"].nunique()

conflicting_variants = y_counts[y_counts > 1]

if not conflicting_variants.empty:
    raise ValueError(
        f"{len(conflicting_variants)} variants have more than one unique "
        "y value. Filter the tissue dataframe by endpoint_type/source "
        "before running the correlation analysis."
    )

# Remove duplicate rows that contain the same gene_id and y
tissue_variant_df = (
    tissue_df[["gene_id", "y"]]
    .drop_duplicates(subset=["gene_id"])
    .rename(columns={"gene_id": "variant_id"})
)

print(
    f"Unique variants with tissue expression: "
    f"{tissue_variant_df['variant_id'].nunique():,}"
)


# ------------------------------------------------------------------
# Existing y distribution plot
# ------------------------------------------------------------------

plot_data = tissue_variant_df["y"].dropna()

if plot_data.empty:
    raise ValueError("The y column contains no valid numeric values.")

plt.figure(figsize=(8, 5))

plt.hist(
    plot_data,
    bins=40,
    edgecolor="black",
)

plt.axvline(
    plot_data.median(),
    linestyle="--",
    label=f"Median = {plot_data.median():.3g}",
)

plt.xlabel("Expression value (y)")
plt.ylabel("Number of variants")
plt.title("Distribution of tissue expression values")
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "y_distribution.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ------------------------------------------------------------------
# Load variant-level immunogenicity scores
# ------------------------------------------------------------------

scores_df = pd.read_csv(
    IMMUNOGENICITY_FILE,
    sep="\t",
)

if "variant_id" not in scores_df.columns:
    raise ValueError(
        "The immunogenicity dataframe does not contain variant_id."
    )

scores_df["variant_id"] = scores_df["variant_id"].astype(str)

if scores_df["variant_id"].duplicated().any():
    duplicate_count = scores_df.loc[
        scores_df["variant_id"].duplicated(keep=False),
        "variant_id",
    ].nunique()

    raise ValueError(
        f"The immunogenicity dataframe contains {duplicate_count} "
        "variant IDs more than once. This may occur if multiple alleles "
        "are present. Filter to one allele or include allele in the analysis."
    )


# ------------------------------------------------------------------
# Merge tissue expression and immunogenicity scores
# ------------------------------------------------------------------

analysis_df = tissue_variant_df.merge(
    scores_df,
    on="variant_id",
    how="inner",
    validate="one_to_one",
)

n_tissue = tissue_variant_df["variant_id"].nunique()
n_scores = scores_df["variant_id"].nunique()
n_matched = analysis_df["variant_id"].nunique()

print(f"Variants in tissue dataframe:         {n_tissue:,}")
print(f"Variants in immunogenicity dataframe: {n_scores:,}")
print(f"Variants matched between dataframes:  {n_matched:,}")
print(f"Tissue variants without scores:       {n_tissue - n_matched:,}")
print(
    f"Percentage of tissue variants matched: "
    f"{100 * n_matched / n_tissue:.2f}%"
)

analysis_df.to_csv(
    OUTPUT_DIR / "tissue_immunogenicity_merged.tsv",
    sep="\t",
    index=False,
)


# ------------------------------------------------------------------
# Select immunogenicity score features
# ------------------------------------------------------------------

immunogenicity_columns = [
    "netMHCpan_passed_count",
    "MHCflurry_passed_count",
    "both_passed_count",
    "either_passed_count",
    "netMHCpan_top_score",
    "netMHCpan_mean_top3",
    "MHCflurry_top_score",
    "MHCflurry_mean_top3",
]

immunogenicity_columns = [
    column
    for column in immunogenicity_columns
    if column in analysis_df.columns
]

if not immunogenicity_columns:
    raise ValueError(
        "No expected immunogenicity score columns were found."
    )


# ------------------------------------------------------------------
# Correlations and scatter plots
# ------------------------------------------------------------------

correlation_results = []

for score_column in immunogenicity_columns:

    correlation_data = (
        analysis_df[["variant_id", score_column, "y"]]
        .copy()
    )

    correlation_data[score_column] = pd.to_numeric(
        correlation_data[score_column],
        errors="coerce",
    )

    correlation_data = correlation_data.dropna(
        subset=[score_column, "y"]
    )

    n_complete = len(correlation_data)
    n_unique_score_values = correlation_data[score_column].nunique()
    n_unique_y_values = correlation_data["y"].nunique()

    # Correlations require at least two observations and variation
    if (
        n_complete < 3
        or n_unique_score_values < 2
        or n_unique_y_values < 2
    ):
        pearson_r = float("nan")
        pearson_p = float("nan")
        spearman_rho = float("nan")
        spearman_p = float("nan")

    else:
        pearson_result = pearsonr(
            correlation_data[score_column],
            correlation_data["y"],
        )

        spearman_result = spearmanr(
            correlation_data[score_column],
            correlation_data["y"],
        )

        pearson_r = pearson_result.statistic
        pearson_p = pearson_result.pvalue

        spearman_rho = spearman_result.statistic
        spearman_p = spearman_result.pvalue

    correlation_results.append(
        {
            "score": score_column,
            "n_variants": n_complete,
            "n_missing": len(analysis_df) - n_complete,
            "pearson_r": pearson_r,
            "pearson_p_value": pearson_p,
            "spearman_rho": spearman_rho,
            "spearman_p_value": spearman_p,
        }
    )

    # Scatter plot
    if not correlation_data.empty:
        plt.figure(figsize=(7, 5))

        plt.scatter(
            correlation_data[score_column],
            correlation_data["y"],
            alpha=0.5,
        )

        plt.xlabel(score_column)
        plt.ylabel("Tissue expression (y)")
        plt.title(f"{score_column} vs tissue expression")

        annotation = (
            f"n = {n_complete:,}\n"
            f"Pearson r = {pearson_r:.3f}\n"
            f"Spearman ρ = {spearman_rho:.3f}"
        )

        plt.text(
            0.05,
            0.95,
            annotation,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox={
                "boxstyle": "round",
                "facecolor": "white",
                "alpha": 0.8,
            },
        )

        plt.tight_layout()

        plt.savefig(
            SCATTER_DIR / f"{score_column}_vs_y.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()


# ------------------------------------------------------------------
# Save correlation results
# ------------------------------------------------------------------

correlation_df = pd.DataFrame(correlation_results)


correlation_df = correlation_df.sort_values(
    "spearman_rho",
    key=lambda x: x.abs(),
    ascending=False,
)

correlation_df.to_csv(
    OUTPUT_DIR / "immunogenicity_expression_correlations.tsv",
    sep="\t",
    index=False,
)

print("\nCorrelation results:")
print(
    correlation_df[
        [
            "score",
            "n_variants",
            "pearson_r",
            "pearson_p_value",
            "spearman_rho",
            "spearman_p_value",
        ]
    ].to_string(index=False)
)

# ------------------------------------------------------------------
# Compare bottom and top expression deciles
# ------------------------------------------------------------------

low_cutoff = analysis_df["y"].quantile(0.10)
high_cutoff = analysis_df["y"].quantile(0.90)

extreme_df = analysis_df.loc[
    (analysis_df["y"] <= low_cutoff)
    | (analysis_df["y"] >= high_cutoff)
].copy()

extreme_df["expression_group"] = np.where(
    extreme_df["y"] >= high_cutoff,
    "Top 10%",
    "Bottom 10%",
)

print(f"\nBottom-decile cutoff: y <= {low_cutoff:.4g}")
print(f"Top-decile cutoff:    y >= {high_cutoff:.4g}")

print(
    extreme_df["expression_group"]
    .value_counts()
    .to_string()
)

decile_results = []

DECILE_PLOT_DIR = OUTPUT_DIR / "top_bottom_decile_plots"
DECILE_PLOT_DIR.mkdir(parents=True, exist_ok=True)

for score_column in immunogenicity_columns:

    temp = extreme_df[
        ["variant_id", "expression_group", score_column]
    ].copy()

    temp[score_column] = pd.to_numeric(
        temp[score_column],
        errors="coerce",
    )

    temp = temp.dropna(subset=[score_column])

    bottom_values = temp.loc[
        temp["expression_group"] == "Bottom 10%",
        score_column,
    ]

    top_values = temp.loc[
        temp["expression_group"] == "Top 10%",
        score_column,
    ]

    if len(bottom_values) == 0 or len(top_values) == 0:
        continue

    test = mannwhitneyu(
        top_values,
        bottom_values,
        alternative="two-sided",
    )

    decile_results.append(
        {
            "score": score_column,
            "n_bottom_decile": len(bottom_values),
            "n_top_decile": len(top_values),
            "bottom_mean": bottom_values.mean(),
            "top_mean": top_values.mean(),
            "mean_difference_top_minus_bottom": (
                top_values.mean() - bottom_values.mean()
            ),
            "bottom_median": bottom_values.median(),
            "top_median": top_values.median(),
            "median_difference_top_minus_bottom": (
                top_values.median() - bottom_values.median()
            ),
            "mannwhitney_u": test.statistic,
            "mannwhitney_p_value": test.pvalue,
        }
    )

    # Boxplot
    plt.figure(figsize=(6, 5))

    plt.boxplot(
        [bottom_values, top_values],
        tick_labels=["Bottom 10%", "Top 10%"],
        showfliers=False,
    )

    plt.ylabel(score_column)
    plt.xlabel("Expression group")
    plt.title(f"{score_column} by expression decile")
    plt.tight_layout()

    plt.savefig(
        DECILE_PLOT_DIR / f"{score_column}_top_bottom_deciles.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


decile_results_df = pd.DataFrame(decile_results)

decile_results_df = decile_results_df.sort_values(
    "mannwhitney_p_value"
)

decile_results_df.to_csv(
    OUTPUT_DIR / "top_bottom_decile_comparison.tsv",
    sep="\t",
    index=False,
)

extreme_df.to_csv(
    OUTPUT_DIR / "top_bottom_decile_variants.tsv",
    sep="\t",
    index=False,
)

print("\nTop versus bottom decile comparison:")
print(
    decile_results_df[
        [
            "score",
            "n_bottom_decile",
            "n_top_decile",
            "bottom_median",
            "top_median",
            "median_difference_top_minus_bottom",
            "mannwhitney_p_value",
        ]
    ].to_string(index=False)
)

# ------------------------------------------------------------------
# Linear regression: unadjusted and adjusted for mutation count
# ------------------------------------------------------------------


if "mutation_count" not in analysis_df.columns:
    raise ValueError(
        "mutation_count is missing from analysis_df. "
        "Regenerate the variant immunogenicity score dataframe with "
        "VR_mutation and mutation_count included."
    )

analysis_df["mutation_count"] = pd.to_numeric(
    analysis_df["mutation_count"],
    errors="coerce",
)

analysis_df["y"] = pd.to_numeric(
    analysis_df["y"],
    errors="coerce",
)

regression_results = []


def fit_linear_model(
    data: pd.DataFrame,
    outcome: str,
    predictor: str,
    adjust_for_mutations: bool,
) -> dict:
    """
    Fit an OLS model with robust HC3 standard errors.

    Unadjusted:
        y ~ immunogenicity score

    Adjusted:
        y ~ immunogenicity score + mutation_count
    """

    model_columns = [outcome, predictor]

    if adjust_for_mutations:
        model_columns.append("mutation_count")

    model_data = (
        data[model_columns]
        .dropna()
        .copy()
    )

    if len(model_data) < 3:
        raise ValueError(
            f"Not enough complete observations for {predictor}."
        )

    if model_data[predictor].nunique() < 2:
        raise ValueError(
            f"{predictor} has fewer than two unique values."
        )

    predictors = [predictor]

    if adjust_for_mutations:
        predictors.append("mutation_count")

    X = sm.add_constant(
        model_data[predictors],
        has_constant="add",
    )

    model = sm.OLS(
        model_data[outcome],
        X,
    ).fit(cov_type="HC3")

    confidence_interval = model.conf_int().loc[predictor]

    return {
        "score": predictor,
        "model": (
            "adjusted_for_mutation_count"
            if adjust_for_mutations
            else "unadjusted"
        ),
        "n_variants": int(model.nobs),
        "coefficient": model.params[predictor],
        "robust_std_error": model.bse[predictor],
        "t_value": model.tvalues[predictor],
        "p_value": model.pvalues[predictor],
        "ci_lower": confidence_interval.iloc[0],
        "ci_upper": confidence_interval.iloc[1],
        "intercept": model.params["const"],
        "r_squared": model.rsquared,
        "adjusted_r_squared": model.rsquared_adj,
        "aic": model.aic,
        "bic": model.bic,
        "mutation_count_coefficient": (
            model.params.get("mutation_count", np.nan)
        ),
        "mutation_count_p_value": (
            model.pvalues.get("mutation_count", np.nan)
        ),
    }


for score_column in immunogenicity_columns:

    analysis_df[score_column] = pd.to_numeric(
        analysis_df[score_column],
        errors="coerce",
    )

    try:
        # Unadjusted model
        regression_results.append(
            fit_linear_model(
                data=analysis_df,
                outcome="y",
                predictor=score_column,
                adjust_for_mutations=False,
            )
        )

        # Mutation-count-adjusted model
        regression_results.append(
            fit_linear_model(
                data=analysis_df,
                outcome="y",
                predictor=score_column,
                adjust_for_mutations=True,
            )
        )

    except ValueError as error:
        print(f"Skipping {score_column}: {error}")


regression_results_df = pd.DataFrame(regression_results)

regression_results_df.to_csv(
    OUTPUT_DIR / "immunogenicity_expression_linear_regression.tsv",
    sep="\t",
    index=False,
)


# ------------------------------------------------------------------
# Put adjusted and unadjusted coefficients side by side
# ------------------------------------------------------------------

regression_comparison_df = (
    regression_results_df.pivot(
        index="score",
        columns="model",
        values=[
            "n_variants",
            "coefficient",
            "robust_std_error",
            "p_value",
            "ci_lower",
            "ci_upper",
            "r_squared",
            "adjusted_r_squared",
        ],
    )
)

regression_comparison_df.columns = [
    f"{statistic}_{model_name}"
    for statistic, model_name in regression_comparison_df.columns
]

regression_comparison_df = (
    regression_comparison_df
    .reset_index()
)

# How much the immunogenicity coefficient changes after adjustment
regression_comparison_df["coefficient_change_after_adjustment"] = (
    regression_comparison_df[
        "coefficient_adjusted_for_mutation_count"
    ]
    - regression_comparison_df[
        "coefficient_unadjusted"
    ]
)

regression_comparison_df["percent_coefficient_change"] = np.where(
    regression_comparison_df["coefficient_unadjusted"].ne(0),
    (
        regression_comparison_df[
            "coefficient_change_after_adjustment"
        ]
        / regression_comparison_df[
            "coefficient_unadjusted"
        ].abs()
        * 100
    ),
    np.nan,
)

regression_comparison_df.to_csv(
    OUTPUT_DIR / "immunogenicity_regression_comparison.tsv",
    sep="\t",
    index=False,
)


print("\nLinear regression results:")

print(
    regression_results_df[
        [
            "score",
            "model",
            "n_variants",
            "coefficient",
            "ci_lower",
            "ci_upper",
            "p_value",
            "r_squared",
        ]
    ].to_string(index=False)
)

print("\nAdjusted versus unadjusted comparison:")

print(
    regression_comparison_df[
        [
            "score",
            "coefficient_unadjusted",
            "p_value_unadjusted",
            "coefficient_adjusted_for_mutation_count",
            "p_value_adjusted_for_mutation_count",
            "coefficient_change_after_adjustment",
            "r_squared_unadjusted",
            "r_squared_adjusted_for_mutation_count",
        ]
    ].to_string(index=False)
)

# ------------------------------------------------------------------
# Forest plots of regression coefficients
# ------------------------------------------------------------------

FOREST_PLOT_DIR = OUTPUT_DIR / "regression_forest_plots"
FOREST_PLOT_DIR.mkdir(parents=True, exist_ok=True)


COUNT_FEATURES = [
    "netMHCpan_passed_count",
    "MHCflurry_passed_count",
    "both_passed_count",
    "either_passed_count",
]

CONTINUOUS_FEATURES = [
    "netMHCpan_top_score",
    "netMHCpan_mean_top3",
    "MHCflurry_top_score",
    "MHCflurry_mean_top3",
]


SCORE_LABELS = {
    "netMHCpan_passed_count": "NetMHCpan passed count",
    "MHCflurry_passed_count": "MHCflurry passed count",
    "both_passed_count": "Passed by both tools",
    "either_passed_count": "Passed by either tool",
    "netMHCpan_top_score": "NetMHCpan strongest score",
    "netMHCpan_mean_top3": "NetMHCpan mean top 3",
    "MHCflurry_top_score": "MHCflurry strongest score",
    "MHCflurry_mean_top3": "MHCflurry mean top 3",
}

MODEL_LABELS = {
    "unadjusted": "Unadjusted",
    "adjusted_for_mutation_count": "Adjusted for mutation count",
}


def create_regression_forest_plot(
    results_df: pd.DataFrame,
    features: list[str],
    title: str,
    output_file: Path,
    xlabel: str = "Regression coefficient for tissue expression (95% CI)",
) -> None:
    """
    Create a forest plot containing adjusted and unadjusted regression
    coefficients for a selected set of immunogenicity features.
    """

    required_columns = [
        "score",
        "model",
        "coefficient",
        "ci_lower",
        "ci_upper",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in results_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing regression result columns: "
            + ", ".join(missing_columns)
        )

    plot_df = results_df.loc[
        results_df["score"].isin(features)
        & results_df["model"].isin(MODEL_LABELS)
    ].copy()

    if plot_df.empty:
        raise ValueError(
            f"No regression results were available for: {features}"
        )

    plot_df["score_label"] = (
        plot_df["score"]
        .map(SCORE_LABELS)
        .fillna(plot_df["score"])
    )

    plot_df["model_label"] = (
        plot_df["model"]
        .map(MODEL_LABELS)
        .fillna(plot_df["model"])
    )

    # Preserve the requested feature order
    feature_order = [
        feature
        for feature in features
        if feature in plot_df["score"].unique()
    ]

    # Base vertical positions, with first feature shown at the top
    base_positions = {
        feature: len(feature_order) - 1 - index
        for index, feature in enumerate(feature_order)
    }

    model_offsets = {
        "unadjusted": 0.12,
        "adjusted_for_mutation_count": -0.12,
    }

    model_markers = {
        "unadjusted": "o",
        "adjusted_for_mutation_count": "s",
    }

    fig, ax = plt.subplots(
        figsize=(9, max(4.5, len(feature_order) * 1.15))
    )

    for model_name in [
        "unadjusted",
        "adjusted_for_mutation_count",
    ]:
        model_df = plot_df.loc[
            plot_df["model"] == model_name
        ].copy()

        # Keep rows in the same order as the y-axis labels
        model_df["feature_order"] = model_df["score"].map(
            {
                feature: index
                for index, feature in enumerate(feature_order)
            }
        )

        model_df = model_df.sort_values("feature_order")

        y_positions = [
            base_positions[score] + model_offsets[model_name]
            for score in model_df["score"]
        ]

        lower_errors = (
            model_df["coefficient"] - model_df["ci_lower"]
        )

        upper_errors = (
            model_df["ci_upper"] - model_df["coefficient"]
        )

        ax.errorbar(
            model_df["coefficient"],
            y_positions,
            xerr=np.vstack([lower_errors, upper_errors]),
            fmt=model_markers[model_name],
            capsize=4,
            markersize=7,
            linewidth=1.5,
            label=MODEL_LABELS[model_name],
        )

    ax.axvline(
        0,
        linestyle="--",
        linewidth=1,
    )

    y_ticks = [
        base_positions[feature]
        for feature in feature_order
    ]

    y_labels = [
        SCORE_LABELS.get(feature, feature)
        for feature in feature_order
    ]

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)

    ax.set_xlabel(xlabel)
    ax.set_title(title)

    ax.legend(
        title="Model",
        frameon=False,
        loc="best",
    )

    ax.grid(
        axis="x",
        linestyle=":",
        alpha=0.4,
    )

    fig.tight_layout()

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ------------------------------------------------------------------
# Count-feature forest plot
# ------------------------------------------------------------------

create_regression_forest_plot(
    results_df=regression_results_df,
    features=COUNT_FEATURES,
    title="Association between predicted epitope counts and tissue expression",
    output_file=(
        FOREST_PLOT_DIR
        / "epitope_count_regression_forest_plot.png"
    ),
)


# ------------------------------------------------------------------
# Continuous-score forest plot
# ------------------------------------------------------------------

create_regression_forest_plot(
    results_df=regression_results_df,
    features=CONTINUOUS_FEATURES,
    title="Association between predicted binding scores and tissue expression",
    output_file=(
        FOREST_PLOT_DIR
        / "continuous_score_regression_forest_plot.png"
    ),
)


print(
    "\nForest plots saved to:"
    f"\n{FOREST_PLOT_DIR / 'epitope_count_regression_forest_plot.png'}"
    f"\n{FOREST_PLOT_DIR / 'continuous_score_regression_forest_plot.png'}"
)