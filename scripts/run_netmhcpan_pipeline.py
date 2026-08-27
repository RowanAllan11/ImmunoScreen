from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.netmhcpan import (
    DEFAULT_NETMHC_PATH,
    parse_netmhcpan_xls_wide_tsv,
    read_peptide_variant_map_tsv,
    read_unique_peptides_tsv,
    run_netmhcpan_for_k,
)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Run netMHCpan, add EL-rank threshold flags, "
            "and map peptides back to variants."
        )
    )

    ap.add_argument(
        "--peptides",
        type=Path,
        required=True,
        help="Path to unique_peptides.tsv.",
    )
    ap.add_argument(
        "--peptide-map",
        type=Path,
        required=True,
        help="Path to peptide_variant_map.tsv.",
    )
    ap.add_argument(
        "--alleles",
        nargs="+",
        required=True,
        help=(
            "One or more netMHCpan allele names, for example "
            "'H-2-Db H-2-Kb'."
        ),
    )
    ap.add_argument(
        "--kmers",
        type=int,
        nargs="+",
        required=True,
        help="One or more peptide lengths to predict.",
    )

    ap.add_argument(
        "--netmhcpan",
        type=Path,
        default=DEFAULT_NETMHC_PATH,
    )
    ap.add_argument(
        "--outdir",
        type=Path,
        default=REPO_ROOT / "data/output/netmhcpan",
        help="Base netMHCpan output directory.",
    )
    ap.add_argument(
        "--el-rank-threshold",
        type=float,
        default=2.0,
    )
    ap.add_argument(
        "--dedup",
        action="store_true",
        help=(
            "Remove duplicate allele-peptide-variant-position rows, "
            "retaining the lowest EL rank."
        ),
    )
    ap.add_argument(
        "--output-format",
        choices=["xls", "txt"],
        default="xls",
    )
    ap.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional arguments passed to netMHCpan.",
    )

    args = ap.parse_args()

    peptides_tsv = args.peptides.resolve()
    peptide_map_tsv = args.peptide_map.resolve()
    netmhcpan_path = args.netmhcpan.resolve()

    alleles = [
        allele.strip()
        for allele in args.alleles
        if allele.strip()
    ]

    # Preserve order while removing duplicates.
    alleles = list(dict.fromkeys(alleles))

    if not alleles:
        raise ValueError(
            "No valid alleles were supplied through --alleles."
        )

    kmers = list(dict.fromkeys(args.kmers))

    if any(k <= 0 for k in kmers):
        raise ValueError(
            "All values supplied through --kmers must be positive."
        )

    run_label = peptides_tsv.parent.name
    run_dir = args.outdir.resolve() / run_label
    raw_dir = run_dir / "raw"

    raw_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Run label: {run_label}")
    print(f"Alleles: {', '.join(alleles)}")
    print(
        "Peptide lengths: "
        + ", ".join(map(str, kmers))
    )

    # 1. Run netMHCpan separately for each peptide length.
    raw_outputs: list[Path] = []

    for k in kmers:
        out_path = run_netmhcpan_for_k(
            peptides_tsv=peptides_tsv,
            k=k,
            alleles=alleles,
            netmhcpan_path=netmhcpan_path,
            outdir=raw_dir,
            output_format=args.output_format,
            extra=list(args.extra),
        )

        raw_outputs.append(out_path)

        print(
            f"Wrote raw netMHCpan output: {out_path}"
        )

    # The current parser handles netMHCpan's tab-delimited XLS output.
    if args.output_format != "xls":
        raise NotImplementedError(
            "Parsing txt output is not currently implemented. "
            "Use --output-format xls for the mapped pipeline output."
        )

    # 2. Read peptide and variant mapping tables.
    uniq = read_unique_peptides_tsv(peptides_tsv)
    pmap = read_peptide_variant_map_tsv(
        peptide_map_tsv
    )

    # 3. Parse all netMHCpan outputs.
    prediction_frames: list[pd.DataFrame] = []

    for output_path in raw_outputs:
        frame = parse_netmhcpan_xls_wide_tsv(
            output_path
        )
        prediction_frames.append(frame)

    if not prediction_frames:
        raise RuntimeError(
            "No netMHCpan prediction tables were produced."
        )

    nm_all = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    if nm_all.empty:
        raise RuntimeError(
            "The parsed netMHCpan prediction table is empty."
        )

    # Validate that parsed output only contains requested alleles.
    requested_alleles = set(alleles)
    observed_alleles = set(
        nm_all["allele"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    unexpected_alleles = (
        observed_alleles - requested_alleles
    )

    if unexpected_alleles:
        raise ValueError(
            "netMHCpan output contained alleles that were not "
            f"requested: {sorted(unexpected_alleles)}"
        )

    missing_alleles = (
        requested_alleles - observed_alleles
    )

    if missing_alleles:
        raise ValueError(
            "No netMHCpan output was parsed for the following "
            f"requested alleles: {sorted(missing_alleles)}"
        )

    # 4. Add EL-rank threshold flag.
    nm_all["netMHCpan_EL_rank"] = pd.to_numeric(
        nm_all["netMHCpan_EL_rank"],
        errors="coerce",
    )

    nm_all["netMHCpan_EL_rank_pass"] = (
        nm_all["netMHCpan_EL_rank"].notna()
        & (
            nm_all["netMHCpan_EL_rank"]
            < float(args.el_rank_threshold)
        )
    )

    # 5. Join predictions to peptide IDs.
    merged = nm_all.merge(
        uniq,
        on=["peptide", "k"],
        how="left",
        validate="many_to_one",
    )

    n_missing_peptide_ids = int(
        merged["peptide_id"].isna().sum()
    )

    if n_missing_peptide_ids:
        raise ValueError(
            f"{n_missing_peptide_ids} netMHCpan rows could not "
            "be mapped to peptide_id. Ensure the predictions were "
            "generated from this exact unique_peptides.tsv."
        )

    # 6. Expand each peptide prediction to variant occurrences.
    out = merged.merge(
        pmap,
        on="peptide_id",
        how="left",
        validate="many_to_many",
    )

    n_missing_variants = int(
        out["variant_id"].isna().sum()
    )

    if n_missing_variants:
        raise ValueError(
            f"{n_missing_variants} rows were missing a variant "
            "mapping after joining peptide_variant_map.tsv."
        )

    out["length"] = out["k"]

    # 7. Optional deduplication.
    if args.dedup:
        dedup_keys = [
            "allele",
            "peptide_id",
            "variant_id",
            "start",
            "end",
        ]

        out = (
            out
            .sort_values(
                "netMHCpan_EL_rank",
                ascending=True,
                na_position="last",
            )
            .drop_duplicates(
                dedup_keys,
                keep="first",
            )
        )

    # 8. Choose output columns.
    base_cols = [
        "allele",
        "peptide_id",
        "peptide",
        "k",
        "variant_id",
        "start",
        "end",
        "length",
        "occurrence_count",
    ]

    excluded_map_columns = {
        "peptide_id",
        "variant_id",
        "start",
        "end",
    }

    meta_cols = [
        column
        for column in pmap.columns
        if column not in excluded_map_columns
        and column not in base_cols
    ]

    score_cols = [
        "netMHCpan_EL_score",
        "netMHCpan_EL_rank",
        "netMHCpan_EL_rank_pass",
    ]

    for optional_col in [
        "netMHCpan_BA_score",
        "netMHCpan_BA_rank",
    ]:
        if optional_col in out.columns:
            score_cols.append(optional_col)

    out_cols = (
        base_cols
        + meta_cols
        + score_cols
    )

    for column in out_cols:
        if column not in out.columns:
            out[column] = pd.NA

    out = (
        out[out_cols]
        .sort_values(
            [
                "allele",
                "netMHCpan_EL_rank",
                "variant_id",
                "peptide_id",
            ],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    # 9. Write mapped output.
    out_path = run_dir / "predictions_mapped.tsv"

    out.to_csv(
        out_path,
        sep="\t",
        index=False,
    )

    n_pass = int(
        out["netMHCpan_EL_rank_pass"]
        .fillna(False)
        .astype(bool)
        .sum()
    )
    n_total = len(out)

    print(
        f"Wrote mapped netMHCpan output: {out_path}"
    )
    print(
        "Passing EL-rank threshold: "
        f"{n_pass:,}/{n_total:,} rows "
        f"(< {args.el_rank_threshold})"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
