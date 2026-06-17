# Mixed-Effect Modelling for Peptide-Level Analysis

import numpy as np
import pandas as pd
import statsmodels.api as sm


INPUT_FILE = (
    "data/output/bigmhc/VR5_V3__k9/"
    "predictions_mapped.tsv"
)

MIN_PREVALENCE = 0.05
EPSILON = 1e-6


def split_mutations(value: object) -> list[str]:
    """Convert a semicolon-separated mutation string into a list."""
    if pd.isna(value):
        return []

    return [
        mutation.strip()
        for mutation in str(value).split(";")
        if mutation.strip()
    ]


df = pd.read_csv(
    INPUT_FILE,
    sep="\t",
    low_memory=False,
)

sampled_variants = (
    df["variant_id"]
    .drop_duplicates()
    .sample(n=5_000, random_state=42)
)

df = df[df["variant_id"].isin(sampled_variants)].copy()


required_columns = {
    "variant_id",
    "allele",
    "peptide_id",
    "peptide_mutation",
    "netMHCpan_EL_rank",
    "MHCflurry_affinity_percentile",
}

missing = required_columns - set(df.columns)

if missing:
    raise ValueError(
        f"Input file is missing required columns: {sorted(missing)}"
    )


# Convert the outcome to numeric and remove missing values.
df["MHCflurry_affinity_percentile"] = pd.to_numeric(
    df["MHCflurry_affinity_percentile"],
    errors="coerce",
)

df = df.dropna(
    subset=[
        "variant_id",
        "peptide_id",
        "MHCflurry_affinity_percentile",
    ]
).copy()


# Lower percentiles indicate stronger predicted binding.
# The minus sign makes larger transformed scores represent stronger binding.
df["log_affinity_score"] = -np.log10(
    df["MHCflurry_affinity_percentile"].clip(
        lower=EPSILON
    )
)


# One-hot encode peptide-level mutations.
mutation_matrix = (
    df["peptide_mutation"]
    .apply(split_mutations)
    .str.join("|")
    .str.get_dummies(sep="|")
)


# Keep mutations present in at least the specified proportion of peptide rows.
mutation_prevalence = mutation_matrix.mean(axis=0)

retained_mutations = mutation_prevalence[
    mutation_prevalence >= MIN_PREVALENCE
].index

mutation_matrix = mutation_matrix[
    retained_mutations
].astype(float)


if mutation_matrix.shape[1] == 0:
    raise ValueError(
        "No mutations passed the minimum prevalence filter. "
        "Try reducing MIN_PREVALENCE."
    )


# Add mutation columns back to the modelling dataframe.
model_df = pd.concat(
    [
        df.reset_index(drop=True),
        mutation_matrix.reset_index(drop=True),
    ],
    axis=1,
)


# Fixed effects: peptide-level mutation indicators.
X = sm.add_constant(
    model_df[list(retained_mutations)],
    has_constant="add",
)

# Continuous transformed outcome.
y = model_df["log_affinity_score"].astype(float)


# Random intercept for variant.
#
# This accounts for the fact that multiple peptide observations come
# from the same variant and are therefore not independent.
model = sm.MixedLM(
    endog=y,
    exog=X,
    groups=model_df["variant_id"],
)

result = model.fit(
    reml=False,
    method="lbfgs",
    maxiter=500,
)


print(result.summary())


# Extract fixed-effect coefficients and confidence intervals.
confidence_intervals = result.conf_int(alpha=0.05)

fixed_terms = result.fe_params.index

results = pd.DataFrame(
    {
        "term": fixed_terms,
        "coefficient": result.fe_params.loc[
            fixed_terms
        ].values,
        "std_error": result.bse.loc[
            fixed_terms
        ].values,
        "p_value": result.pvalues.loc[
            fixed_terms
        ].values,
        "ci_lower_95": confidence_intervals.loc[
            fixed_terms, 0
        ].values,
        "ci_upper_95": confidence_intervals.loc[
            fixed_terms, 1
        ].values,
    }
)


results["mutation_prevalence"] = results["term"].map(
    mutation_prevalence
)

results = results.sort_values(
    "p_value",
    ascending=True,
)


results.to_csv(
    "data/output/mixed_effects/"
    "peptide_mixed_effects_mhcflurry.tsv",
    sep="\t",
    index=False,
)


with open(
    "data/output/mixed_effects/"
    "peptide_mixed_effects_mhcflurry_summary.txt",
    "w",
    encoding="utf-8",
) as handle:
    handle.write(result.summary().as_text())


print("\nTop mutation results:")
print(results.head(20).to_string(index=False))