from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_SCORE_COLUMNS = [
    "netMHCpan_EL_rank",
    "MHCflurry_affinity_percentile",
    "BigMHC_EL",
]


def save_histogram(
    values: pd.Series,
    title: str,
    xlabel: str,
    output_path: Path,
    bins: int = 50,
) -> None:
    values = pd.to_numeric(values, errors="coerce").dropna()

    if values.empty:
        print(f"Skipping empty distribution: {xlabel}")
        return

    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins)
    plt.xlabel(xlabel)
    plt.ylabel("Number of rows")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_correlation_heatmap(
    correlation: pd.DataFrame,
    title: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(7, 6))

    image = plt.imshow(
        correlation,
        vmin=-1,
        vmax=1,
        aspect="auto",
    )

    plt.colorbar(image, label="Correlation")

    plt.xticks(
        range(len(correlation.columns)),
        correlation.columns,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        range(len(correlation.index)),
        correlation.index,
    )

    for row in range(len(correlation.index)):
        for col in range(len(correlation.columns)):
            value = correlation.iloc[row, col]

            if pd.notna(value):
                plt.text(
                    col,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                )

    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_scatterplot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    output_path: Path,
    max_points: int,
    random_seed: int,
) -> None:
    plot_df = df[[x_col, y_col]].dropna()

    if plot_df.empty:
        print(f"Skipping empty scatter plot: {x_col} vs {y_col}")
        return

    if len(plot_df) > max_points:
        plot_df = plot_df.sample(
            n=max_points,
            random_state=random_seed,
        )

    plt.figure(figsize=(7, 6))
    plt.scatter(
        plot_df[x_col],
        plot_df[y_col],
        alpha=0.25,
        s=8,
    )
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"{x_col} vs {y_col}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def mutation_frequency_table(
    variant_df: pd.DataFrame,
) -> pd.DataFrame:
    mutation_df = variant_df.copy()

    mutation_df["mutation"] = (
        mutation_df["VR_mutation"]
        .fillna("")
        .astype(str)
        .str.split(";")
    )

    mutation_df = mutation_df.explode("mutation")

    mutation_df["mutation"] = (
        mutation_df["mutation"]
        .astype(str)
        .str.strip()
    )

    mutation_df = mutation_df[
        mutation_df["mutation"].ne("")
        & mutation_df["mutation"].ne("nan")
        & mutation_df["mutation"].ne("WT")
    ]

    frequencies = (
        mutation_df.groupby("mutation")["variant_id"]
        .nunique()
        .rename("variant_count")
        .reset_index()
        .sort_values("variant_count", ascending=False)
    )

    n_variants = variant_df["variant_id"].nunique()

    if n_variants > 0:
        frequencies["variant_percentage"] = (
            frequencies["variant_count"] / n_variants * 100
        )
    else:
        frequencies["variant_percentage"] = np.nan

    return frequencies


def save_mutation_plot(
    frequency_df: pd.DataFrame,
    title: str,
    output_path: Path,
    top_n: int,
) -> None:
    plot_df = frequency_df.head(top_n).copy()

    if plot_df.empty:
        print(f"Skipping empty mutation plot: {title}")
        return

    plot_df = plot_df.sort_values(
        "variant_count",
        ascending=True,
    )

    plt.figure(figsize=(9, max(5, len(plot_df) * 0.3)))

    plt.barh(
        plot_df["mutation"],
        plot_df["variant_count"],
    )

    plt.xlabel("Number of variants")
    plt.ylabel("Mutation")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def build_score_summary(
    df: pd.DataFrame,
    score_columns: list[str],
) -> pd.DataFrame:
    rows = []

    for column in score_columns:
        values = pd.to_numeric(df[column], errors="coerce")

        rows.append(
            {
                "score": column,
                "total_rows": len(df),
                "non_missing": int(values.notna().sum()),
                "missing": int(values.isna().sum()),
                "mean": values.mean(),
                "standard_deviation": values.std(),
                "minimum": values.min(),
                "q25": values.quantile(0.25),
                "median": values.median(),
                "q75": values.quantile(0.75),
                "maximum": values.max(),
            }
        )

    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run simplified descriptive analysis of MHC and BigMHC "
            "prediction scores."
        )
    )

    parser.add_argument(
        "--i",
        type=Path,
        required=True,
        help="Input combined TSV containing predictor scores.",
    )

    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Directory for analysis outputs.",
    )

    parser.add_argument(
        "--netmhcpan-threshold",
        type=float,
        default=2.0,
        help="Maximum netMHCpan EL rank considered passing. Default: 2.",
    )

    parser.add_argument(
        "--mhcflurry-threshold",
        type=float,
        default=2.0,
        help=(
            "Maximum MHCflurry affinity percentile considered passing. "
            "Default: 2."
        ),
    )

    parser.add_argument(
        "--top-mutations",
        type=int,
        default=25,
        help="Number of mutations shown in frequency plots. Default: 25.",
    )

    parser.add_argument(
        "--scatter-max-points",
        type=int,
        default=100_000,
        help=(
            "Maximum points included in each scatter plot. "
            "Large datasets are randomly sampled. Default: 100000."
        ),
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    input_path = args.i.resolve()
    output_dir = args.outdir.resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    summary_dir = output_dir / "summary"
    correlation_dir = output_dir / "correlations"
    distribution_dir = output_dir / "distributions"
    mutation_dir = output_dir / "mutations"

    for directory in [
        summary_dir,
        correlation_dir,
        distribution_dir,
        mutation_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {input_path}")

    df = pd.read_csv(
        input_path,
        sep="\t",
        low_memory=False,
    )

    required_columns = {
        "variant_id",
        "peptide",
        "allele",
        "VR_mutation",
        "netMHCpan_EL_rank",
        "MHCflurry_affinity_percentile",
        "BigMHC_EL",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Input is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    score_columns = DEFAULT_SCORE_COLUMNS

    for column in score_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # ----------------------------------------------------------
    # Pass/fail definitions
    # ----------------------------------------------------------

    df["netmhcpan_pass"] = (
        df["netMHCpan_EL_rank"].notna()
        & (
            df["netMHCpan_EL_rank"]
            <= args.netmhcpan_threshold
        )
    )

    df["mhcflurry_pass"] = (
        df["MHCflurry_affinity_percentile"].notna()
        & (
            df["MHCflurry_affinity_percentile"]
            <= args.mhcflurry_threshold
        )
    )

    df["pass_either"] = (
        df["netmhcpan_pass"]
        | df["mhcflurry_pass"]
    )

    df["pass_both"] = (
        df["netmhcpan_pass"]
        & df["mhcflurry_pass"]
    )

    # ----------------------------------------------------------
    # Dataset summary
    # ----------------------------------------------------------

    n_rows = len(df)
    n_variants = df["variant_id"].nunique()
    n_peptides = df["peptide"].nunique()

    n_peptide_allele_pairs = (
        df[["peptide", "allele"]]
        .drop_duplicates()
        .shape[0]
    )

    dataset_summary = pd.DataFrame(
        [
            {
                "total_rows": n_rows,
                "unique_variants": n_variants,
                "unique_peptides": n_peptides,
                "unique_peptide_allele_pairs": n_peptide_allele_pairs,
                "netmhcpan_passing_rows": int(
                    df["netmhcpan_pass"].sum()
                ),
                "mhcflurry_passing_rows": int(
                    df["mhcflurry_pass"].sum()
                ),
                "passing_either_rows": int(
                    df["pass_either"].sum()
                ),
                "passing_both_rows": int(
                    df["pass_both"].sum()
                ),
                "passing_either_percentage": (
                    df["pass_either"].mean() * 100
                ),
                "passing_both_percentage": (
                    df["pass_both"].mean() * 100
                ),
            }
        ]
    )

    dataset_summary.to_csv(
        summary_dir / "dataset_summary.tsv",
        sep="\t",
        index=False,
    )

    score_summary = build_score_summary(
        df,
        score_columns,
    )

    score_summary.to_csv(
        summary_dir / "score_summary.tsv",
        sep="\t",
        index=False,
    )

    # ----------------------------------------------------------
    # Pass rates by allele
    # ----------------------------------------------------------

    allele_summary = (
        df.groupby("allele", dropna=False)
        .agg(
            total_rows=("variant_id", "size"),
            unique_variants=("variant_id", "nunique"),
            unique_peptides=("peptide", "nunique"),
            netmhcpan_passing=("netmhcpan_pass", "sum"),
            mhcflurry_passing=("mhcflurry_pass", "sum"),
            passing_either=("pass_either", "sum"),
            passing_both=("pass_both", "sum"),
        )
        .reset_index()
    )

    allele_summary["passing_either_percentage"] = (
        allele_summary["passing_either"]
        / allele_summary["total_rows"]
        * 100
    )

    allele_summary["passing_both_percentage"] = (
        allele_summary["passing_both"]
        / allele_summary["total_rows"]
        * 100
    )

    allele_summary.to_csv(
        summary_dir / "pass_rate_by_allele.tsv",
        sep="\t",
        index=False,
    )

    # ----------------------------------------------------------
    # Optional peptide-length summary
    # ----------------------------------------------------------

    length_column = None

    if "k" in df.columns:
        length_column = "k"
        df[length_column] = pd.to_numeric(
            df[length_column],
            errors="coerce",
        )

    elif "peptide_length" in df.columns:
        length_column = "peptide_length"
        df[length_column] = pd.to_numeric(
            df[length_column],
            errors="coerce",
        )

    if length_column is not None:
        length_summary = (
            df.groupby(length_column, dropna=False)
            .agg(
                total_rows=("variant_id", "size"),
                unique_variants=("variant_id", "nunique"),
                unique_peptides=("peptide", "nunique"),
                passing_either=("pass_either", "sum"),
            )
            .reset_index()
        )

        length_summary["passing_either_percentage"] = (
            length_summary["passing_either"]
            / length_summary["total_rows"]
            * 100
        )

        length_summary.to_csv(
            summary_dir / "pass_rate_by_peptide_length.tsv",
            sep="\t",
            index=False,
        )

    # ----------------------------------------------------------
    # Correlations
    # ----------------------------------------------------------

    correlation_data = df[score_columns].copy()

    pearson_correlation = correlation_data.corr(
        method="pearson"
    )

    spearman_correlation = correlation_data.corr(
        method="spearman"
    )

    pearson_correlation.to_csv(
        correlation_dir / "pearson_correlation.tsv",
        sep="\t",
    )

    spearman_correlation.to_csv(
        correlation_dir / "spearman_correlation.tsv",
        sep="\t",
    )

    save_correlation_heatmap(
        pearson_correlation,
        "Pearson correlation between prediction scores",
        correlation_dir / "pearson_correlation_heatmap.png",
    )

    save_correlation_heatmap(
        spearman_correlation,
        "Spearman correlation between prediction scores",
        correlation_dir / "spearman_correlation_heatmap.png",
    )

    score_pairs = [
        (
            "netMHCpan_EL_rank",
            "MHCflurry_affinity_percentile",
        ),
        (
            "netMHCpan_EL_rank",
            "BigMHC_EL",
        ),
        (
            "MHCflurry_affinity_percentile",
            "BigMHC_EL",
        ),
    ]

    for x_col, y_col in score_pairs:
        save_scatterplot(
            df=df,
            x_col=x_col,
            y_col=y_col,
            output_path=(
                correlation_dir
                / f"{x_col}_vs_{y_col}.png"
            ),
            max_points=args.scatter_max_points,
            random_seed=args.random_seed,
        )

    # ----------------------------------------------------------
    # Score distributions
    # ----------------------------------------------------------

    for score_column in score_columns:
        save_histogram(
            values=df[score_column],
            title=f"Distribution of {score_column}",
            xlabel=score_column,
            output_path=(
                distribution_dir
                / f"{score_column}_distribution.png"
            ),
        )

        save_histogram(
            values=df.loc[
                df["pass_either"],
                score_column,
            ],
            title=f"{score_column}: rows passing either predictor",
            xlabel=score_column,
            output_path=(
                distribution_dir
                / f"{score_column}_passed_distribution.png"
            ),
        )

        save_histogram(
            values=df.loc[
                ~df["pass_either"],
                score_column,
            ],
            title=f"{score_column}: rows not passing either predictor",
            xlabel=score_column,
            output_path=(
                distribution_dir
                / f"{score_column}_failed_distribution.png"
            ),
        )

    # ----------------------------------------------------------
    # Variant-level mutation analysis
    # ----------------------------------------------------------

    variants = (
        df.groupby("variant_id", as_index=False)
        .agg(
            VR_mutation=("VR_mutation", "first"),
            pass_either=("pass_either", "any"),
            pass_both=("pass_both", "any"),
            netmhcpan_pass=("netmhcpan_pass", "any"),
            mhcflurry_pass=("mhcflurry_pass", "any"),
        )
    )

    variant_pass_summary = pd.DataFrame(
        [
            {
                "total_variants": len(variants),
                "variants_passing_either": int(
                    variants["pass_either"].sum()
                ),
                "variants_not_passing_either": int(
                    (~variants["pass_either"]).sum()
                ),
                "variants_passing_both": int(
                    variants["pass_both"].sum()
                ),
                "variants_passing_either_percentage": (
                    variants["pass_either"].mean() * 100
                ),
            }
        ]
    )

    variant_pass_summary.to_csv(
        summary_dir / "variant_pass_summary.tsv",
        sep="\t",
        index=False,
    )

    mutation_groups = {
        "overall": variants,
        "passed": variants[variants["pass_either"]],
        "failed": variants[~variants["pass_either"]],
    }

    for group_name, group_df in mutation_groups.items():
        frequencies = mutation_frequency_table(group_df)

        frequencies.to_csv(
            mutation_dir
            / f"mutation_frequency_{group_name}.tsv",
            sep="\t",
            index=False,
        )

        save_mutation_plot(
            frequency_df=frequencies,
            title=(
                f"Most frequent mutations: {group_name} variants"
            ),
            output_path=(
                mutation_dir
                / f"mutation_frequency_{group_name}.png"
            ),
            top_n=args.top_mutations,
        )

    print(f"Analysis complete. Outputs written to: {output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


"""
python analysis/descriptive_analysis.py \
  --i data/output/bigmhc/VR5_V3__k9/predictions_mapped.tsv \
  --outdir data/output/descriptive_analysis/VR5_V3__k9
"""