from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def split_mutations(value: object) -> list[str]:
    """Convert a semicolon-separated mutation string into a list."""
    if pd.isna(value):
        return []

    return [
        mutation.strip()
        for mutation in str(value).split(";")
        if mutation.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Variant-level linear regression of passing peptide counts on mutations."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Combined peptide-level TSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/output/linear_regression"),
    )
    parser.add_argument(
        "--min-prevalence",
        type=float,
        default=0.01,
        help="Minimum proportion of variants containing a mutation.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, sep="\t", low_memory=False)

    required_columns = {
        "variant_id",
        "allele",
        "peptide_id",
        "VR_mutation",
        "netMHCpan_EL_rank_pass",
        "MHCflurry_affinity_percentile_pass",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Input file is missing required columns: {sorted(missing)}"
        )

    # Ensure pass columns are Boolean.
    for column in [
        "netMHCpan_EL_rank_pass",
        "MHCflurry_affinity_percentile_pass",
    ]:
        df[column] = (
            df[column]
            .astype(str)
            .str.lower()
            .map({"true": True, "false": False, "1": True, "0": False})
            .fillna(False)
        )

    # A peptide passes when either predictor passes.
    df["passes_either"] = (
        df["netMHCpan_EL_rank_pass"]
        | df["MHCflurry_affinity_percentile_pass"]
    )

    # Prevent duplicate peptide rows within the same variant being counted twice.
    peptide_level = (
        df.groupby(
            ["variant_id", "allele", "peptide_id"],
            as_index=False,
        )
        .agg(
            passes_either=("passes_either", "max"),
            VR_mutation=("VR_mutation", "first"),
        )
    )

    # Outcome: number of unique peptides passing either threshold per variant.
    variant_df = (
        peptide_level.groupby("variant_id", as_index=False)
        .agg(
            passing_peptide_count=("passes_either", "sum"),
            total_peptide_count=("peptide_id", "nunique"),
            VR_mutation=("VR_mutation", "first"),
        )
    )

    # One-hot encode all mutations.
    mutation_matrix = (
        variant_df["VR_mutation"]
        .apply(split_mutations)
        .str.join("|")
        .str.get_dummies(sep="|")
    )

    # Keep mutations present in at least the requested proportion of variants.
    mutation_prevalence = mutation_matrix.mean(axis=0)

    retained_mutations = mutation_prevalence[
        mutation_prevalence >= args.min_prevalence
    ].index

    mutation_matrix = mutation_matrix[retained_mutations]

    if mutation_matrix.shape[1] == 0:
        raise ValueError(
            "No mutations passed the minimum prevalence filter. "
            "Try reducing --min-prevalence."
        )

    X = mutation_matrix.astype(float)
    X = sm.add_constant(X)

    y = variant_df["passing_peptide_count"].astype(float)

    # HC3 gives heteroskedasticity-robust standard errors.
    model = sm.OLS(y, X).fit(cov_type="HC3")

    results = pd.DataFrame(
        {
            "term": model.params.index,
            "coefficient": model.params.values,
            "std_error": model.bse.values,
            "t_value": model.tvalues.values,
            "p_value": model.pvalues.values,
            "ci_lower": model.conf_int()[0].values,
            "ci_upper": model.conf_int()[1].values,
        }
    )

    results["mutation_prevalence"] = results["term"].map(
        mutation_prevalence
    )

    results = results.sort_values(
        ["p_value", "coefficient"],
        ascending=[True, False],
    )

    variant_df.to_csv(
        args.output_dir / "variant_level_data.tsv",
        sep="\t",
        index=False,
    )

    mutation_prevalence.rename("prevalence").to_csv(
        args.output_dir / "mutation_prevalence.tsv",
        sep="\t",
        header=True,
    )

    results.to_csv(
        args.output_dir / "linear_regression_results.tsv",
        sep="\t",
        index=False,
    )

    with open(
        args.output_dir / "linear_regression_summary.txt",
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(model.summary().as_text())

    print(f"Variants: {len(variant_df):,}")
    print(f"Mutations retained: {len(retained_mutations):,}")
    print(f"R-squared: {model.rsquared:.4f}")
    print(f"Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

"""
python analysis/linear_statistical_analysis.py \
  --input data/output/bigmhc/VR5_V3__k9/predictions_mapped.tsv \
  --output-dir data/output/linear_regression \
  --min-prevalence 0.01
"""