from sklearn.utils import resample
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

df_mut = pd.read_csv("data/output/combined/VR5_9mer_combined_netmhcpan_mhcflurry.annotated.tsv", sep="\t")

# ---------------------------
# One-hot encode mutations
# ---------------------------
mutation_dummies = (
    df_mut["mutation"]
    .fillna("")
    .replace("WT", "")
    .str.get_dummies(sep=";")
    .add_prefix("mut_")
)

df_mut = pd.concat([df_mut, mutation_dummies], axis=1)

# Predictor matrix
X_cols = mutation_dummies.columns
X = df_mut[X_cols]

# Response variable
y_col = "NetMHCpan_EL_rank"

# ---------------------------
# Bootstrap by variant
# ---------------------------
variants = df_mut["variant_id"].unique()

n_bootstraps = 100
coefs = np.zeros((n_bootstraps, len(X_cols)))

for i in range(n_bootstraps):

    sampled_variants = resample(
        variants,
        replace=True,
        n_samples=len(variants)
    )

    boot_df = pd.concat(
        [df_mut[df_mut["variant_id"] == v] for v in sampled_variants],
        ignore_index=True
    )

    X_sample = boot_df[X_cols]
    y_sample = boot_df[y_col].astype(float)

    model = LinearRegression()
    model.fit(X_sample, y_sample)

    coefs[i, :] = model.coef_

# ---------------------------
# Summarise mutation effects
# ---------------------------
coef_summary = pd.DataFrame({
    "mutation": X_cols,
    "mean_coef": coefs.mean(axis=0),
    "lower_95": np.percentile(coefs, 2.5, axis=0),
    "upper_95": np.percentile(coefs, 97.5, axis=0)
})

coef_summary = coef_summary.sort_values(
    "mean_coef",
    ascending=False
)

print(coef_summary.head(20))