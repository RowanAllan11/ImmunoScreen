#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
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
            "Fit a multivariable peptide-level OLS model with all retained "
            "mutations, peptide-window fixed effects, and variant-clustered "
            "standard errors."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input predictions_mapped.tsv file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output results TSV file.",
    )
    parser.add_argument(
        "--score-column",
        default="MHCflurry_affinity_percentile",
        choices=[
            "MHCflurry_affinity_percentile",
            "netMHCpan_EL_rank",
        ],
        help="Percentile/rank outcome column.",
    )
    parser.add_argument(
        "--min-prevalence",
        type=float,
        default=0.005,
        help=(
            "Minimum proportion of peptide rows containing a mutation. "
            "For example, 0.005 means 0.5%% of peptide rows."
        ),
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-6,
        help="Lower clipping value before log10 transformation.",
    )
    parser.add_argument(
        "--window-columns",
        nargs="+",
        default=["start", "end", "k"],
        help="Columns defining the exact peptide window.",
    )
    parser.add_argument(
        "--sample-variants",
        type=int,
        default=None,
        help="Optional number of variants to sample before modelling.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used when sampling variants.",
    )
    return parser.parse_args()


def split_mutations(value):
    """Convert a semicolon-separated mutation string into a list."""
    if pd.isna(value):
        return []

    return [
        mutation.strip()
        for mutation in str(value).split(";")
        if mutation.strip()
    ]


def main():
    args = parse_args()

    df = pd.read_csv(
        args.input,
        sep="\t",
        low_memory=False,
    )

    missing = required_columns - set(df.columns)
    missing_window = set(args.window_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"Input file is missing required columns: {sorted(missing)}"
        )

    if missing_window:
        raise ValueError(
            f"Input file is missing window columns: {sorted(missing_window)}"
        )

    if args.sample_variants is not None:
        variants = df["variant_id"].drop_duplicates()

        if args.sample_variants > len(variants):
            raise ValueError(
                f"--sample-variants is {args.sample_variants}, but only "
                f"{len(variants)} unique variants are available."
            )

        sampled_variants = variants.sample(
            n=args.sample_variants,
            random_state=args.random_state,
        )

        df = df[df["variant_id"].isin(sampled_variants)].copy()

    df[args.score_column] = pd.to_numeric(
        df[args.score_column],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "variant_id",
            "allele",
            "peptide_id",
            args.score_column,
        ]
    ).copy()

    # Larger transformed values indicate stronger predicted presentation.
    df["log_score"] = -np.log10(
        df[args.score_column].clip(lower=args.epsilon)
    )

    # Treat WT as carrying no peptide-level mutations.
    df["peptide_mutation"] = df["peptide_mutation"].replace(
        {
            "WT": "",
            "wt": "",
        }
    )

    output_tables = []

    for allele, allele_df in df.groupby("allele", sort=True):
        allele_df = allele_df.reset_index(drop=True).copy()

        # Same prevalence logic as the supplied mixed-effects script:
        # prevalence = proportion of peptide rows containing each mutation.
        mutation_matrix = (
            allele_df["peptide_mutation"]
            .apply(split_mutations)
            .str.join("|")
            .str.get_dummies(sep="|")
        )

        mutation_matrix = mutation_matrix.drop(
            columns=["WT", "wt", ""],
            errors="ignore",
        )

        mutation_prevalence = mutation_matrix.mean(axis=0)

        retained_mutations = mutation_prevalence[
            mutation_prevalence >= args.min_prevalence
        ].index.tolist()

        if not retained_mutations:
            print(
                f"{allele}: no mutations passed prevalence "
                f"{args.min_prevalence:.6f}."
            )

            if not mutation_prevalence.empty:
                print("Highest mutation prevalences:")
                print(
                    mutation_prevalence
                    .sort_values(ascending=False)
                    .head(20)
                    .to_string()
                )
            continue

        mutation_matrix = mutation_matrix[
            retained_mutations
        ].astype(float)

        # Exact peptide-window fixed effects.
        window_id = (
            allele_df[args.window_columns]
            .astype(str)
            .agg("_".join, axis=1)
        )

        window_matrix = pd.get_dummies(
            window_id,
            prefix="window",
            drop_first=True,
            dtype=float,
        )

        X = pd.concat(
            [
                mutation_matrix.reset_index(drop=True),
                window_matrix.reset_index(drop=True),
            ],
            axis=1,
        )

        X = sm.add_constant(
            X,
            has_constant="add",
        ).astype(float)

        y = allele_df["log_score"].astype(float)

        model = sm.OLS(
            endog=y,
            exog=X,
        )

        result = model.fit(
            cov_type="cluster",
            cov_kwds={
                "groups": allele_df["variant_id"],
            },
        )

        confidence_intervals = result.conf_int(alpha=0.05)

        rows = []

        for mutation in retained_mutations:
            rows.append(
                {
                    "allele": allele,
                    "score_column": args.score_column,
                    "term": mutation,
                    "coefficient": result.params[mutation],
                    "std_error": result.bse[mutation],
                    "t_value": result.tvalues[mutation],
                    "p_value": result.pvalues[mutation],
                    "ci_lower_95": confidence_intervals.loc[mutation, 0],
                    "ci_upper_95": confidence_intervals.loc[mutation, 1],
                    "mutation_prevalence": mutation_prevalence[mutation],
                    "mutation_row_count": int(
                        mutation_matrix[mutation].sum()
                    ),
                    "n_rows": int(result.nobs),
                    "n_variants": allele_df["variant_id"].nunique(),
                    "n_mutations_in_model": len(retained_mutations),
                    "r_squared": result.rsquared,
                    "adjusted_r_squared": result.rsquared_adj,
                }
            )

        allele_results = pd.DataFrame(rows)

        allele_results["q_value"] = multipletests(
            allele_results["p_value"],
            method="fdr_bh",
        )[1]

        allele_results["significant_fdr_05"] = (
            allele_results["q_value"] < 0.05
        )

        # Because the outcome is -log10(percentile):
        # percentile ratio = mutated percentile / reference percentile.
        allele_results["percentile_ratio"] = (
            10 ** (-allele_results["coefficient"])
        )

        allele_results["percentile_percent_change"] = (
            allele_results["percentile_ratio"] - 1
        ) * 100

        allele_results["percentile_ratio_ci_lower"] = (
            10 ** (-allele_results["ci_upper_95"])
        )

        allele_results["percentile_ratio_ci_upper"] = (
            10 ** (-allele_results["ci_lower_95"])
        )

        output_tables.append(allele_results)

        print(
            f"{allele}: retained {len(retained_mutations)} mutations "
            f"from {len(allele_df):,} peptide rows."
        )

        print("Top retained mutation prevalences:")
        print(
            mutation_prevalence.loc[retained_mutations]
            .sort_values(ascending=False)
            .head(20)
            .to_string()
        )

    if not output_tables:
        raise ValueError(
            "No allele produced a fitted model. "
            "Inspect the printed mutation prevalences or reduce "
            "--min-prevalence."
        )

    results = pd.concat(
        output_tables,
        ignore_index=True,
    )

    results = results.sort_values(
        ["allele", "q_value", "p_value"],
        ascending=True,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        output_path,
        sep="\t",
        index=False,
    )

    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()

"""
python analysis/multivariable_peptide_ols.py \
  --input data/output/bigmhc/VR5_V3__k9/predictions_mapped.tsv \
  --output data/output/peptide_level/AAV9_WT__k9/netmhcpan_multivariable_ols.tsv \
  --score-column netMHCpan_EL_rank \
  --min-prevalence 0.005
"""