#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.variable_analysis import (  # noqa: E402
    annotate_epitopes_with_variability,
    compute_column_metrics,
    load_aligned_fasta,
    load_mhcflurry_epitopes_noheader,
    merge_fastas_to_multifasta,
    run_mafft,
    scatter_affinity_vs_entropy,
)

# Example Use
# python3 scripts/run_variable_analysis.py --msa-dir data/input --protein AAV9_VP1 --reference-name AAV9_VP1

def main() -> int:
    ap = argparse.ArgumentParser(description="MSA-based variability annotation for mhcflurry epitopes.")

    ap.add_argument("--protein", default="AAV9_VP1", help="Protein name as in mhcflurry_epitopes.tsv")
    ap.add_argument(
        "--epitopes-tsv",
        default=str(REPO_ROOT / "data/output/epitopes/mhcflurry_epitopes.tsv"),
        help="Path to mhcflurry_epitopes.tsv (no header)",
    )

    # Input sequences (choose one)
    ap.add_argument(
        "--msa-fasta",
        default=None,
        help="Input multi-FASTA of capsid sequences across AAV STs (unaligned).",
    )
    ap.add_argument(
        "--msa-dir",
        default=None,
        help="Directory containing many per-serotype FASTA files (e.g. data/input). If set, a multi-FASTA is built automatically.",
    )
    ap.add_argument(
        "--msa-pattern",
        default="AAV*_VP1.fasta",
        help="Glob pattern used with --msa-dir (default: AAV*_VP1.fasta)",
    )

    ap.add_argument(
        "--reference-name",
        default="AAV9_VP1",
        help="Record name to treat as reference. If using --msa-dir with name_from=stem, this is the filename stem.",
    )
    ap.add_argument(
        "--out-aln",
        default=str(REPO_ROOT / "data/output/msa/capsid.mafft.fasta"),
        help="Where to write aligned FASTA (MAFFT output).",
    )
    ap.add_argument(
        "--out-annotated-epitopes",
        default=None,
        help="Output TSV with variability columns. Default: data/output/epitopes/mhcflurry_epitopes.variability.tsv",
    )
    ap.add_argument(
        "--out-scatter",
        default=None,
        help="Output PNG for affinity vs entropy scatter. Default: data/output/plots/affinity_vs_entropy.<protein>.png",
    )
    ap.add_argument("--mafft-exe", default="mafft", help="MAFFT executable name/path")
    args = ap.parse_args()

    epitopes_path = Path(args.epitopes_tsv)
    aln_out = Path(args.out_aln)

    if not epitopes_path.exists():
        raise SystemExit(f"Missing epitopes TSV: {epitopes_path}")

    # Decide where the unaligned multi-FASTA comes from
    if args.msa_dir:
        msa_dir = Path(args.msa_dir)
        if not msa_dir.exists():
            raise SystemExit(f"--msa-dir does not exist: {msa_dir}")

        msa_in = aln_out.with_suffix(aln_out.suffix + ".input.fasta")
        merge_fastas_to_multifasta(
            msa_dir,
            msa_in,
            pattern=str(args.msa_pattern),
            name_from="stem",
        )
    else:
        if not args.msa_fasta:
            raise SystemExit("Provide either --msa-fasta or --msa-dir")
        msa_in = Path(args.msa_fasta)
        if not msa_in.exists():
            raise SystemExit(f"Missing MSA input FASTA: {msa_in}")

    # Align + compute metrics
    run_mafft(msa_in, aln_out, mafft_exe=args.mafft_exe)
    msa = load_aligned_fasta(aln_out)
    metrics = compute_column_metrics(msa)

    # Annotate epitopes + save outputs
    ep = load_mhcflurry_epitopes_noheader(epitopes_path)
    annotated = annotate_epitopes_with_variability(
        ep,
        metrics,
        msa=msa,
        reference_name=str(args.reference_name),
        protein=str(args.protein),
    )

    out_ep = (
        Path(args.out_annotated_epitopes)
        if args.out_annotated_epitopes
        else REPO_ROOT / "data/output/epitopes/mhcflurry_epitopes.variability.tsv"
    )
    out_ep.parent.mkdir(parents=True, exist_ok=True)
    annotated.to_csv(out_ep, sep="\t", index=False)
    print(f"Wrote: {out_ep}")

    out_scatter = (
        Path(args.out_scatter)
        if args.out_scatter
        else REPO_ROOT / "data/output/plots" / f"affinity_vs_entropy.{args.protein}.png"
    )
    scatter_affinity_vs_entropy(
        annotated,
        out_png=out_scatter,
        title=f"{args.protein}: affinity percentile vs mean entropy (MSA)",
    )
    print(f"Wrote: {out_scatter}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())