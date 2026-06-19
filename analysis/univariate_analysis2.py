from pathlib import Path
import re
import warnings

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from joblib import Parallel, delayed

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = Path("data/output/bigmhc/VR5_V3__k9/predictions_mapped.tsv")
OUTPUT_FILE = Path("data/output/univariate/VR5_V3__k9/univariate_mutation_results.tsv")

OUTCOME_COLUMN = "MHCflurry_affinity_percentile"

MIN_VARIANTS = 50
MAX_ITER = 200

# True means:
# transformed score = -log10(percentile / 100)
# Higher transformed values mean stronger predicted presentation.
TRANSFORM_PERCENTILE = True


# ============================================================
# HELPERS
# ============================================================

MUTATION_PATTERN = re.compile(r"^[A-Za-z*](\d+)[A-Za-z*]$")


def split_mutations(value):
    if pd.isna(value):
        return []

    text = str(value).strip()

    if text.lower() in {"", "nan", "none", "wt", "."}:
        return []

    return [
        mutation.strip()
        for mutation in text.split(";")
        if MUTATION_PATTERN.match(mutation.strip())
    ]


def get_position(mutation):
    match = MUTATION_PATTERN.match(mutation)

    if match is None:
        return None

    return int(match.group(1))


# ============================================================
# READ DATA
# ============================================================

df = pd.read_csv(INPUT_FILE, sep="\t", low_memory=False)

required_columns = [
    "variant_id",
    "start",
    "end",
    "k",
    "peptide_mutation",
    OUTCOME_COLUMN,
]

missing = [column for column in required_columns if column not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

df["start"] = pd.to_numeric(df["start"], errors="coerce")
df["end"] = pd.to_numeric(df["end"], errors="coerce")
df["k"] = pd.to_numeric(df["k"], errors="coerce")
df["outcome"] = pd.to_numeric(df[OUTCOME_COLUMN], errors="coerce")

df = df.dropna(
    subset=[
        "variant_id",
        "start",
        "end",
        "k",
        "outcome",
    ]
).copy()

df["start"] = df["start"].astype(int)
df["end"] = df["end"].astype(int)
df["k"] = df["k"].astype(int)
df["variant_id"] = df["variant_id"].astype(str)

if TRANSFORM_PERCENTILE:
    df["outcome"] = -np.log10(
        df["outcome"].clip(lower=1e-12) / 100
    )

df["mutation_list"] = df["peptide_mutation"].apply(split_mutations)

df["window_id"] = (
    df["start"].astype(str)
    + "_"
    + df["end"].astype(str)
    + "_k"
    + df["k"].astype(str)
)


# ============================================================
# FIND COMMON MUTATIONS
# ============================================================

mutation_rows = []

for row in df[["variant_id", "mutation_list"]].itertuples(index=False):
    for mutation in row.mutation_list:
        mutation_rows.append(
            {
                "variant_id": row.variant_id,
                "mutation": mutation,
            }
        )

mutation_table = pd.DataFrame(mutation_rows)

if mutation_table.empty:
    raise ValueError("No mutations found in peptide_mutation.")

mutation_counts = (
    mutation_table
    .drop_duplicates(["variant_id", "mutation"])
    .groupby("mutation")["variant_id"]
    .nunique()
    .sort_values(ascending=False)
)

mutations_to_test = mutation_counts[
    mutation_counts >= MIN_VARIANTS
].index.tolist()

print(f"Rows: {len(df):,}")
print(f"Variants: {df['variant_id'].nunique():,}")
print(f"Mutations tested: {len(mutations_to_test):,}")


# ============================================================
# FIT ONE MODEL PER MUTATION
# ============================================================

def fit_mutation(mutation):
    position = get_position(mutation)

    subset = df[
        (df["start"] <= position)
        & (df["end"] >= position)
    ].copy()

    subset["mutation_present"] = subset["mutation_list"].apply(
        lambda mutations: int(mutation in mutations)
    )

    n_present = int(subset["mutation_present"].sum())
    n_absent = int(len(subset) - n_present)

    n_present_variants = subset.loc[
        subset["mutation_present"] == 1,
        "variant_id"
    ].nunique()

    n_absent_variants = subset.loc[
        subset["mutation_present"] == 0,
        "variant_id"
    ].nunique()

    result = {
        "mutation": mutation,
        "position": position,
        "n_rows": len(subset),
        "n_present_rows": n_present,
        "n_absent_rows": n_absent,
        "n_present_variants": n_present_variants,
        "n_absent_variants": n_absent_variants,
        "n_windows": subset["window_id"].nunique(),
        "coefficient": np.nan,
        "std_error": np.nan,
        "p_value": np.nan,
        "ci_lower": np.nan,
        "ci_upper": np.nan,
        "variant_variance": np.nan,
        "residual_variance": np.nan,
        "converged": False,
        "status": "not_fitted",
        "warning": "",
    }

    if subset["mutation_present"].nunique() < 2:
        result["status"] = "no_contrast"
        return result

    formula = "outcome ~ mutation_present + C(window_id)"

    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")

            model = smf.mixedlm(
                formula,
                data=subset,
                groups=subset["variant_id"],
                re_formula="1",
            )

            fitted = model.fit(
                reml=False,
                method="lbfgs",
                maxiter=MAX_ITER,
                disp=False,
            )

        confidence_interval = fitted.conf_int().loc["mutation_present"]

        result["coefficient"] = fitted.params["mutation_present"]
        result["std_error"] = fitted.bse["mutation_present"]
        result["p_value"] = fitted.pvalues["mutation_present"]
        result["ci_lower"] = confidence_interval.iloc[0]
        result["ci_upper"] = confidence_interval.iloc[1]
        result["residual_variance"] = fitted.scale
        result["converged"] = fitted.converged
        result["status"] = "success"

        if fitted.cov_re.shape[0] > 0:
            result["variant_variance"] = fitted.cov_re.iloc[0, 0]

        if caught_warnings:
            result["warning"] = " | ".join(
                sorted({str(w.message) for w in caught_warnings})
            )

    except Exception as error:
        result["status"] = "model_error"
        result["warning"] = f"{type(error).__name__}: {error}"

    return result

if __name__ == "__main__":

    N_JOBS = 8

    results = Parallel(
        n_jobs=N_JOBS,
        backend="loky",
        verbose=10,
    )(
        delayed(fit_mutation)(mutation)
        for mutation in mutations_to_test
    )



# ============================================================
# FDR CORRECTION
# ============================================================

    results_df = pd.DataFrame(results)

    results_df["fdr_bh"] = np.nan
    results_df["significant_fdr_0_05"] = False

    valid = results_df["p_value"].notna()

    if valid.any():
        rejected, adjusted_pvalues, _, _ = multipletests(
            results_df.loc[valid, "p_value"],
            method="fdr_bh",
            alpha=0.05,
        )

        results_df.loc[valid, "fdr_bh"] = adjusted_pvalues
        results_df.loc[valid, "significant_fdr_0_05"] = rejected

    results_df = results_df.sort_values(
        ["significant_fdr_0_05", "fdr_bh"],
        ascending=[False, True],
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(
        OUTPUT_FILE,
        sep="\t",
        index=False,
    )

    print()
    print(f"Successful models: {(results_df['status'] == 'success').sum()}")
    print(
        "FDR-significant mutations: "
        f"{results_df['significant_fdr_0_05'].sum()}"
    )
    print(f"Saved to: {OUTPUT_FILE}")