from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

RESULTS_FILE = Path("data/output/linear_regression/VR5_V3__k9_count_net_binomial/binomial_regression_results.tsv")
OUTPUT_DIR = Path("data/output/linear_regression/VR5_V3__k9_count_net_binomial/plots")

ALLELE = "H2-D*b"
OUTCOME = "passing_count"
SCORE_COLUMN = "netMHCpan_EL_rank"

N_EACH_SIDE = 10

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# Load and filter results
# ------------------------------------------------------------------

results = pd.read_csv(RESULTS_FILE, sep="\t")

subset = results.loc[
    (results["allele"] == ALLELE)
    & (results["outcome"] == OUTCOME)
    & (results["score_column"] == SCORE_COLUMN)
    & (results["term"] != "intercept")
].copy()

# Optional: restrict the plot to FDR-significant mutations
subset = subset.loc[subset["significant_fdr_05"]].copy()

subset = subset.dropna(
    subset=[
        "term",
        "coefficient",
        "ci_lower",
        "ci_upper",
    ]
)

if len(subset) < 2 * N_EACH_SIDE:
    raise ValueError(
        f"Only {len(subset)} eligible mutations were found. "
        f"At least {2 * N_EACH_SIDE} are needed."
    )


# ------------------------------------------------------------------
# Select the extreme coefficients
# ------------------------------------------------------------------

# Most negative coefficients
smallest = subset.nsmallest(N_EACH_SIDE, "coefficient").copy()
smallest["coefficient_group"] = "Most negative"

# Most positive coefficients
largest = subset.nlargest(N_EACH_SIDE, "coefficient").copy()
largest["coefficient_group"] = "Most positive"

plot_data = pd.concat(
    [smallest, largest],
    ignore_index=True,
)

# Remove a duplicate if the dataset is very small and the groups overlap
plot_data = plot_data.drop_duplicates(subset="term")

# Sorting controls the vertical order in the plot
plot_data = plot_data.sort_values(
    "coefficient",
    ascending=True,
).reset_index(drop=True)


# ------------------------------------------------------------------
# Save the selected results
# ------------------------------------------------------------------

selected_columns = [
    "allele",
    "outcome",
    "score_column",
    "term",
    "coefficient",
    "std_error",
    "t_value",
    "p_value",
    "q_value",
    "ci_lower",
    "ci_upper",
    "mutation_count",
    "mutation_prevalence",
    "fold_improvement",
    "fold_improvement_ci_lower",
    "fold_improvement_ci_upper",
    "percentile_ratio",
    "coefficient_group",
]

selected_columns = [
    column for column in selected_columns
    if column in plot_data.columns
]

output_table = (
    plot_data[selected_columns]
    .sort_values("coefficient", ascending=False)
)

table_path = OUTPUT_DIR / (
    f"{ALLELE}_{OUTCOME}_{SCORE_COLUMN}_"
    f"top_bottom_{N_EACH_SIDE}_coefficients.tsv"
).replace("*", "").replace("/", "_")

output_table.to_csv(
    table_path,
    sep="\t",
    index=False,
)


# ------------------------------------------------------------------
# Forest plot
# ------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 6))

y_positions = range(len(plot_data))

lower_error = (
    plot_data["coefficient"] - plot_data["ci_lower"]
).to_numpy()

upper_error = (
    plot_data["ci_upper"] - plot_data["coefficient"]
).to_numpy()

ax.errorbar(
    x=plot_data["coefficient"],
    y=list(y_positions),
    xerr=[lower_error, upper_error],
    fmt="o",
    capsize=4,
    markersize=6,
)

# Null-effect line
ax.axvline(
    x=0,
    linestyle="--",
    linewidth=1,
)

ax.set_yticks(list(y_positions))
ax.set_yticklabels(plot_data["term"])

ax.set_xlabel("Mutation coefficient with 95% confidence interval")
ax.set_ylabel("Mutation")

ax.set_title(
    f"Mutation coefficients\n"
    f"{ALLELE} | {OUTCOME} | {SCORE_COLUMN}"
)

ax.grid(axis="x", alpha=0.25)

plt.tight_layout()

plot_path = OUTPUT_DIR / (
    f"{ALLELE}_{OUTCOME}_{SCORE_COLUMN}_"
    f"top_bottom_{N_EACH_SIDE}_forest_plot.png"
).replace("*", "").replace("/", "_")

plt.savefig(
    plot_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(f"Saved selected coefficients to: {table_path}")
print(f"Saved forest plot to: {plot_path}")