#!/usr/bin/env python3
"""Create WT-relative NetMHCIIpan variant scores from annotated 15-mer tables.

Variant and WT windows are matched by allele, start, end and peptide length.
Rank_EL < 5% is the configured binder definition upstream. Positive continuous
change means stronger predicted presentation than WT (WT rank - mutant rank).
Unchanged peptide windows contribute exactly zero to the fixed-window mean.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from analysis.scoring_common import (
        count_mutations,
        create_output_run_label,
        convert_to_boolean,
        parse_scoring_args,
        save_allele_specific_outputs,
    )
except ModuleNotFoundError:  # Direct execution from the analysis directory
    from scoring_common import (
        count_mutations,
        create_output_run_label,
        convert_to_boolean,
        parse_scoring_args,
        save_allele_specific_outputs,
    )


WINDOW_KEYS = ["allele", "start", "end", "k"]
GROUP_KEYS = ["allele", "variant_id"]
RANK = "netMHCIIpan_EL_rank"
BINDER = "netMHCIIpan_EL_rank_binder"


def load_predictions(path: Path) -> pd.DataFrame:
    required = [
        "allele", "peptide_id", "peptide", "variant_id", "start", "end",
        "k", RANK, BINDER, "VR_mutation",
    ]
    data = pd.read_csv(path, sep="\t", usecols=required, low_memory=False)
    for column in ["start", "end", "k"]:
        data[column] = pd.to_numeric(data[column], errors="raise").astype(int)
    data[RANK] = pd.to_numeric(data[RANK], errors="coerce")
    data[BINDER] = convert_to_boolean(data[BINDER])
    data["allele"] = data["allele"].astype(str).str.strip()
    data["peptide"] = data["peptide"].astype(str).str.strip()
    data["variant_id"] = data["variant_id"].astype(str).str.strip()
    data["VR_mutation"] = data["VR_mutation"].fillna("WT").astype(str).str.strip()
    duplicate_keys = ["allele", "variant_id", "start", "end", "k"]
    if data.duplicated(duplicate_keys).any():
        raise ValueError(f"{path} contains duplicate allele/variant/window rows")
    return data


def prepare_wt_reference(wt: pd.DataFrame) -> pd.DataFrame:
    if wt.duplicated(WINDOW_KEYS).any():
        raise ValueError("WT input contains duplicate allele/window rows")
    reference = wt[WINDOW_KEYS + ["peptide_id", "peptide", RANK, BINDER]].copy()
    return reference.rename(columns={
        "peptide_id": "wt_peptide_id",
        "peptide": "wt_peptide",
        RANK: f"wt_{RANK}",
        BINDER: f"wt_{BINDER}",
    })


def add_wt_relative_columns(data: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    merged = data.merge(
        reference, on=WINDOW_KEYS, how="left", validate="many_to_one",
        indicator=True,
    )
    missing = merged["_merge"].eq("left_only")
    if missing.any():
        raise ValueError(f"{int(missing.sum()):,} variant rows lack a matched WT window")
    merged = merged.drop(columns="_merge")
    expected = reference.groupby("allele", observed=True).size()
    merged["expected_window_count"] = merged["allele"].map(expected).astype(int)
    merged["changed_window"] = merged["peptide"].ne(merged["wt_peptide"])
    merged["netMHCIIpan_window_improvement"] = (
        merged[f"wt_{RANK}"] - merged[RANK]
    )
    merged.loc[
        ~merged["changed_window"], "netMHCIIpan_window_improvement"
    ] = 0.0
    wt_binder = merged[f"wt_{BINDER}"].astype(bool)
    mutant_binder = merged[BINDER].astype(bool)
    merged["netMHCIIpan_new_binder_vs_WT"] = mutant_binder & ~wt_binder
    merged["netMHCIIpan_lost_binder_vs_WT"] = ~mutant_binder & wt_binder
    merged["netMHCIIpan_binder_change_vs_WT"] = (
        mutant_binder.astype(int) - wt_binder.astype(int)
    )
    return merged


def create_allele_scores(data: pd.DataFrame) -> pd.DataFrame:
    mutation = data.groupby(GROUP_KEYS, observed=True).agg(
        VR_mutation=("VR_mutation", "first"),
        mutation_annotation_count=("VR_mutation", "nunique"),
    ).reset_index()
    if mutation["mutation_annotation_count"].gt(1).any():
        raise ValueError("A variant has inconsistent mutation annotations")
    mutation = mutation.drop(columns="mutation_annotation_count")
    mutation["mutation_count"] = mutation["VR_mutation"].map(count_mutations)
    scores = data.groupby(GROUP_KEYS, observed=True).agg(
        matched_window_count=("wt_peptide_id", "count"),
        expected_window_count=("expected_window_count", "first"),
        changed_window_count=("changed_window", "sum"),
        netMHCIIpan_new_binder_count=("netMHCIIpan_new_binder_vs_WT", "sum"),
        netMHCIIpan_lost_binder_count=("netMHCIIpan_lost_binder_vs_WT", "sum"),
        netMHCIIpan_net_binder_change=("netMHCIIpan_binder_change_vs_WT", "sum"),
        netMHCIIpan_mean_window_improvement=("netMHCIIpan_window_improvement", "mean"),
    ).reset_index()
    scores = mutation.merge(scores, on=GROUP_KEYS, validate="one_to_one")
    incomplete = scores["matched_window_count"].ne(scores["expected_window_count"])
    if incomplete.any():
        examples = scores.loc[incomplete, GROUP_KEYS + [
            "matched_window_count", "expected_window_count"
        ]].head().to_dict("records")
        raise ValueError(f"Incomplete WT window matching: {examples}")
    integer_columns = [
        "mutation_count", "matched_window_count", "expected_window_count",
        "changed_window_count", "netMHCIIpan_new_binder_count",
        "netMHCIIpan_lost_binder_count", "netMHCIIpan_net_binder_change",
    ]
    scores[integer_columns] = scores[integer_columns].astype(int)
    return scores.sort_values(GROUP_KEYS).reset_index(drop=True)


def main() -> int:
    args = parse_scoring_args(
        description=__doc__ or "Create WT-relative MHC-II presentation scores.",
        default_output_root="data/output/variant_immunogenicity_scores_wt_mhcii",
        variant_help="Variant NetMHCIIpan predictions_mapped_annotated.tsv.",
        wt_help="Matching WT NetMHCIIpan predictions_mapped_annotated.tsv.",
    )
    variant_input = args.variant_input.resolve()
    wt_input = args.wt_input.resolve()
    run_label = args.run_label or create_output_run_label(variant_input)
    print(f"Reading variant predictions: {variant_input}")
    variants = load_predictions(variant_input)
    print(f"Reading WT predictions: {wt_input}")
    wt = load_predictions(wt_input)
    relative = add_wt_relative_columns(variants, prepare_wt_reference(wt))
    allele_scores = create_allele_scores(relative)
    args.output_root.mkdir(parents=True, exist_ok=True)

    outputs = save_allele_specific_outputs(
        allele_scores, args.output_root, run_label
    )
    print("Allele-specific outputs:")
    for output in outputs:
        print(f"  {output}")
    print(
        f"Created scores for {allele_scores['variant_id'].nunique():,} variants, "
        f"{allele_scores['allele'].nunique()} alleles, and "
        f"{int(allele_scores['expected_window_count'].max())} WT windows per allele."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
