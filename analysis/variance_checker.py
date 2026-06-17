# Variance Checker
# Statistical summaries

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

df = pd.read_csv("data/output/stats_analysis3/variant_level_data.tsv", sep="\t")


mean_count = df["passing_peptide_count"].mean()
variance_count = df["passing_peptide_count"].var()

print(f"Mean: {mean_count:.3f}")
print(f"Variance: {variance_count:.3f}")
print(f"Variance/mean: {variance_count / mean_count:.3f}")


# Count mutations per variant.
df["total_mutations"] = (
    df["VR_mutation"]
    .fillna("")
    .apply(
        lambda x: len(
            [mutation for mutation in str(x).split(";") if mutation.strip()]
        )
    )
)

# Basic summary.
print(
    df[
        [
            "passing_peptide_count",
            "total_peptide_count",
            "total_mutations",
        ]
    ].describe()
)

# Simple linear regression.
X = sm.add_constant(df["total_mutations"].astype(float))
y = df["passing_peptide_count"].astype(float)

model = sm.OLS(y, X).fit(cov_type="HC3")

print(model.summary())

# Save coefficient table with 95% confidence intervals.
ci = model.conf_int(alpha=0.05)

results = pd.DataFrame(
    {
        "term": model.params.index,
        "coefficient": model.params.values,
        "std_error": model.bse.values,
        "p_value": model.pvalues.values,
        "ci_lower_95": ci[0].values,
        "ci_upper_95": ci[1].values,
    }
)

results.to_csv(
    "data/output/linear_regression/mutation_count_regression.tsv",
    sep="\t",
    index=False,
)

# Scatter plot with fitted regression line.
plot_df = df.sort_values("total_mutations")

plt.figure(figsize=(7, 5))
plt.scatter(
    df["total_mutations"],
    df["passing_peptide_count"],
    alpha=0.3,
)

plt.plot(
    plot_df["total_mutations"],
    model.predict(
        sm.add_constant(plot_df["total_mutations"].astype(float))
    ),
)

plt.xlabel("Total mutations per variant")
plt.ylabel("Passing peptide count")
plt.title("Mutation burden versus passing peptide count")
plt.tight_layout()

plt.savefig(
    "data/output/linear_regression/"
    "mutation_count_vs_passing_epitopes.png",
    dpi=300,
)

plt.close()

print(results)