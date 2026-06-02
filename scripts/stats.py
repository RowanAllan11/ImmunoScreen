#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd

"""
python scripts/stats.py \
  data/output/combined/AAV9_combined_netmhcpan_mhcflurry.variable_regions.tsv \
  -o data/output/basic_analysis
"""


SCORE_COLS = [
    "netMHCpan_EL_rank",
    "MHCflurry_affinity_percentile",
    "MHCflurry_presentation_percentile",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tsv", type=Path, help="Merged/filtered epitope TSV")
    ap.add_argument("-o", "--outdir", type=Path, default=Path("data/output/basic_analysis"))
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.tsv, sep="\t")

    # Clean numeric columns
    numeric_cols = ["start", "end", "length", "Variable_region_presence"] + SCORE_COLS
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print("\n=== BASIC SUMMARY ===")
    print(f"Rows: {len(df)}")
    print(f"Unique peptides: {df['peptide'].nunique()}")
    print(f"Unique alleles: {df['allele'].nunique()}")
    print(f"Peptide lengths: {sorted(df['length'].dropna().unique().astype(int))}")

    print("\n=== MISSINGNESS ===")
    print(df.isna().sum().sort_values(ascending=False))

    print("\n=== SCORE SUMMARY ===")
    print(df[SCORE_COLS].describe())

    print("\n=== ROWS PER ALLELE ===")
    allele_counts = df["allele"].value_counts()
    print(allele_counts)

    print("\n=== ROWS PER PEPTIDE LENGTH ===")
    length_counts = df["length"].value_counts().sort_index()
    print(length_counts)

    print("\n=== VARIABLE REGION PRESENCE ===")
    vr_counts = df["Variable_region_presence"].value_counts(dropna=False)
    print(vr_counts)

    print("\n=== VARIABLE REGION COUNTS ===")
    if "Variable_region" in df.columns:
        print(df["Variable_region"].fillna("None").replace("", "None").value_counts())

    # Model support flags
    df["has_netMHCpan"] = df["netMHCpan_EL_rank"].notna()
    df["has_MHCflurry"] = df["MHCflurry_affinity_percentile"].notna()
    df["models_supporting"] = (
        df["has_netMHCpan"].astype(int) + df["has_MHCflurry"].astype(int)
    )

    print("\n=== MODEL SUPPORT ===")
    print(df["models_supporting"].value_counts().sort_index())

    # Simple rank: lower percentile/rank = better
    df["best_percentile_like_score"] = df[
        ["netMHCpan_EL_rank", "MHCflurry_affinity_percentile"]
    ].min(axis=1, skipna=True)

    ranked = df.sort_values(
        ["models_supporting", "best_percentile_like_score", "MHCflurry_presentation_percentile"],
        ascending=[False, True, True],
        na_position="last",
    )

    top_path = args.outdir / "top_ranked_epitopes.tsv"
    ranked.to_csv(top_path, sep="\t", index=False)

    allele_summary = (
        df.groupby("allele")
        .agg(
            n_rows=("peptide", "size"),
            n_unique_peptides=("peptide", "nunique"),
            mean_netMHCpan_EL_rank=("netMHCpan_EL_rank", "mean"),
            mean_MHCflurry_affinity_percentile=("MHCflurry_affinity_percentile", "mean"),
            n_variable_region=("Variable_region_presence", "sum"),
        )
        .sort_values("n_rows", ascending=False)
    )
    allele_summary.to_csv(args.outdir / "summary_by_allele.tsv", sep="\t")

    length_summary = (
        df.groupby("length")
        .agg(
            n_rows=("peptide", "size"),
            n_unique_peptides=("peptide", "nunique"),
            mean_netMHCpan_EL_rank=("netMHCpan_EL_rank", "mean"),
            mean_MHCflurry_affinity_percentile=("MHCflurry_affinity_percentile", "mean"),
            n_variable_region=("Variable_region_presence", "sum"),
        )
        .sort_index()
    )
    length_summary.to_csv(args.outdir / "summary_by_length.tsv", sep="\t")

    vr_summary = (
        df.assign(Variable_region=df["Variable_region"].fillna("None").replace("", "None"))
        .groupby("Variable_region")
        .agg(
            n_rows=("peptide", "size"),
            n_unique_peptides=("peptide", "nunique"),
            mean_netMHCpan_EL_rank=("netMHCpan_EL_rank", "mean"),
            mean_MHCflurry_affinity_percentile=("MHCflurry_affinity_percentile", "mean"),
        )
        .sort_values("n_rows", ascending=False)
    )
    vr_summary.to_csv(args.outdir / "summary_by_variable_region.tsv", sep="\t")

    print(f"\nWrote outputs to: {args.outdir}")
    print(f"Top ranked epitopes: {top_path}")


if __name__ == "__main__":
    main()