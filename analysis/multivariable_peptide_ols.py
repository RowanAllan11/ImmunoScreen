#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


required_columns = {
    "variant_id",
    "allele",
    "peptide_id",
    "peptide_mutation",
    "netMHCpan_EL_rank",
    "MHCflurry_affinity_percentile",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a multivariable peptide-level OLS model containing all retained "
            "mutation indicators, adjusting for peptide window and clustering "
            "standard errors by variant."
        )
    )
    parser.add_argument("--input", required=True, help="Input TSV file.")
    parser.add_argument("--output", required=True, help="Output results TSV.")
    parser.add_argument(
        "--score-column",
        default="MHCflurry_affinity_percentile",
        choices=[
            "MHCflurry_affinity_percentile",
            "netMHCpan_EL_rank",
        ],
        help="Percentile/rank column to analyse.",
    )
    parser.add_argument(
        "--min-variants",
        type=int,
        default=20,
        help="Minimum number of unique variants carrying a mutation.",
    )
    parser.add_argument(
        "--window-columns",
        nargs="+",
        default=["start", "end", "k"],
        help="Columns defining the exact peptide window.",
    )
    return parser.parse_args()


def split_mutations(value):
    if pd.isna(value):
        return []

    text = str(value).strip()

    if text in {"", "WT", "None", "nan"}:
        return []

    return [mutation.strip() for mutation in text.split(";") if mutation.strip()]


def safe_column_name(mutation, index):
    return f"mutation_{index}"


def main():
    args = parse_args()

    df = pd.read_csv(args.input, sep="\t")

    missing = required_columns - set(df.columns)
    missing_window = set(args.window_columns) - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if missing_window:
        raise ValueError(f"Missing window columns: {sorted(missing_window)}")

    df = df.copy()

    df[args.score_column] = pd.to_numeric(
        df[args.score_column],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            args.score_column,
            "variant_id",
            "allele",
        ]
    )

    df = df[df[args.score_column] > 0].copy()

    # Higher values mean stronger predicted presentation.
    df["score"] = -np.log10(df[args.score_column])

    df["window_id"] = (
        df[args.window_columns]
        .astype(str)
        .agg("_".join, axis=1)
    )

    df["mutation_list"] = df["peptide_mutation"].apply(split_mutations)

    all_results = []

    for allele, allele_df in df.groupby("allele", sort=True):
        allele_df = allele_df.copy()

        all_mutations = sorted(
            {
                mutation
                for mutation_list in allele_df["mutation_list"]
                for mutation in mutation_list
            }
        )

        retained_mutations = []

        for mutation in all_mutations:
            positive_variants = allele_df.loc[
                allele_df["mutation_list"].apply(
                    lambda values: mutation in values
                ),
                "variant_id",
            ].nunique()

            if positive_variants >= args.min_variants:
                retained_mutations.append(mutation)

        if not retained_mutations:
            print(f"{allele}: no mutations passed the prevalence filter.")
            continue

        mutation_to_column = {
            mutation: safe_column_name(mutation, index)
            for index, mutation in enumerate(retained_mutations)
        }

        for mutation, column in mutation_to_column.items():
            allele_df[column] = allele_df["mutation_list"].apply(
                lambda values, m=mutation: int(m in values)
            )

        mutation_columns = list(mutation_to_column.values())

        # Remove constant mutation columns.
        mutation_columns = [
            column
            for column in mutation_columns
            if allele_df[column].nunique() > 1
        ]

        if not mutation_columns:
            print(f"{allele}: all retained mutation columns were constant.")
            continue

        formula = (
            "score ~ "
            + " + ".join(mutation_columns)
            + " + C(window_id)"
        )

        try:
            model = smf.ols(
                formula,
                data=allele_df,
            ).fit(
                cov_type="cluster",
                cov_kwds={"groups": allele_df["variant_id"]},
            )
        except Exception as exc:
            print(f"{allele}: model failed: {exc}")
            continue

        reverse_map = {
            column: mutation
            for mutation, column in mutation_to_column.items()
        }

        for column in mutation_columns:
            ci_lower, ci_upper = model.conf_int().loc[column]
            mutation = reverse_map[column]

            all_results.append(
                {
                    "allele": allele,
                    "score_column": args.score_column,
                    "mutation": mutation,
                    "coefficient": model.params[column],
                    "std_error": model.bse[column],
                    "t_value": model.tvalues[column],
                    "p_value": model.pvalues[column],
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "mutation_positive_rows": int(allele_df[column].sum()),
                    "mutation_positive_variants": allele_df.loc[
                        allele_df[column] == 1,
                        "variant_id",
                    ].nunique(),
                    "n_rows": int(model.nobs),
                    "n_variants": allele_df["variant_id"].nunique(),
                    "n_mutations_in_model": len(mutation_columns),
                    "r_squared": model.rsquared,
                    "adjusted_r_squared": model.rsquared_adj,
                }
            )

        print(
            f"{allele}: fitted {len(mutation_columns)} mutations "
            f"using {int(model.nobs)} peptide rows."
        )

    results_df = pd.DataFrame(all_results)

    if not results_df.empty:
        results_df["q_value"] = np.nan

        for allele, index in results_df.groupby("allele").groups.items():
            p_values = results_df.loc[index, "p_value"]
            valid = p_values.notna()

            if valid.any():
                valid_index = p_values.index[valid]
                results_df.loc[valid_index, "q_value"] = multipletests(
                    p_values.loc[valid_index],
                    method="fdr_bh",
                )[1]

        results_df["significant_fdr_05"] = (
            results_df["q_value"] < 0.05
        )

        results_df = results_df.sort_values(
            ["allele", "q_value", "p_value"],
            na_position="last",
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(
        output_path,
        sep="\t",
        index=False,
    )

    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()

"""
python multivariable_peptide_ols.py \
  --input data/output/bigmhc/.tsv \
  --output data/output/peptide_level/netmhcpan_multivariable_ols.tsv \
  --score-column netMHCpan_EL_rank \
  --min-variants 50
"""
