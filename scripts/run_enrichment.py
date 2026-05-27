#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.region_enrichment import (  # noqa: E402
    epitopes_to_binary_tracks,
    load_ranked_epitopes_tsv,
    plot_enrichment,
    plot_tracks_heatmap,
    run_epitope_enrichment,
    save_enrichment_plot,
    save_tracks_heatmap,
)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Monte Carlo enrichment analysis of epitope-covered regions across HLA tracks (binary from ranked epitope list)."
    )
    ap.add_argument(
        "--ranked-epitopes",
        default=str(REPO_ROOT / "data/output/epitopes/mhcflurry_epitopes.tsv"),
        help="Ranked epitope TSV (output of scripts/run_ranking.py)",
    )
    ap.add_argument("--protein", required=True, help="Protein name (must match 'protein' column in TSV)")
    ap.add_argument("--bin-size", type=int, default=50, help="Bin size in amino acids (default 100)")
    ap.add_argument(
        "--mode",
        choices=["proportion", "count"],
        default="proportion",
        help="Enrichment statistic per bin, averaged across allele tracks",
    )
    ap.add_argument("--iterations", type=int, default=10_000, help="Monte Carlo iterations (default 10000)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for reproducibility")
    ap.add_argument("--fdr-alpha", type=float, default=0.05, help="FDR threshold for significance (BY correction)")
    ap.add_argument(
        "--protein-length",
        type=int,
        default=None,
        help="Optional explicit protein length. If omitted, inferred from max epitope end + 1.",
    )
    ap.add_argument(
        "--alleles",
        default=None,
        help="Optional comma-separated allele subset to include (e.g. 'H2-K*b,H2-D*b'). Default: all alleles in TSV for protein.",
    )

    # Output tables
    ap.add_argument(
        "--out",
        default=None,
        help="Output TSV path. Default: data/output/enrichment/<protein>.bin<bin>.mc<iters>.tsv",
    )

    # Plotting (show + save)
    ap.add_argument("--plot", action="store_true", help="Show enrichment plot (matplotlib)")
    ap.add_argument("--plot-heatmap", action="store_true", help="Show binary-track heatmap (matplotlib)")
    ap.add_argument(
        "--save-plots",
        action="store_true",
        help="Save PNGs under data/output/enrichment/ (in addition to --plot/--plot-heatmap if used)",
    )

    args = ap.parse_args()

    ranked_path = Path(args.ranked_epitopes)
    if not ranked_path.exists():
        raise SystemExit(f"Ranked epitope TSV not found: {ranked_path}")

    allele_list = None
    if args.alleles:
        allele_list = [a.strip() for a in args.alleles.split(",") if a.strip()]
        if not allele_list:
            allele_list = None

    df = run_epitope_enrichment(
        ranked_epitopes_tsv=ranked_path,
        protein=str(args.protein),
        bin_size=int(args.bin_size),
        mode=str(args.mode),
        iterations=int(args.iterations),
        seed=int(args.seed),
        fdr_alpha=float(args.fdr_alpha),
        alleles=allele_list,
        protein_length=args.protein_length if args.protein_length is not None else None,
    )

    out_path: Optional[Path]
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = (
            REPO_ROOT
            / "data/output/enrichment"
            / f"{args.protein}.bin{int(args.bin_size)}.mc{int(args.iterations)}.tsv"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote: {out_path}")
    print(df.to_string(index=False))

    # Plot display
    if args.plot:
        plot_enrichment(df, title=f"{args.protein}: epitope enrichment (Monte Carlo, BY FDR)")

    tracks = None
    allele_order = None
    if args.plot_heatmap or args.save_plots:
        ep_df = load_ranked_epitopes_tsv(ranked_path)
        tracks, allele_order = epitopes_to_binary_tracks(
            ep_df,
            protein=str(args.protein),
            alleles=allele_list,
            protein_length=args.protein_length if args.protein_length is not None else None,
        )

    if args.plot_heatmap:
        plot_tracks_heatmap(tracks, allele_order)

    # Save PNGs under data/output/enrichment/
    if args.save_plots:
        plot_dir = REPO_ROOT / "data/output/enrichment"
        enrich_png = plot_dir / f"{args.protein}.bin{int(args.bin_size)}.mc{int(args.iterations)}.enrichment.png"
        heat_png = plot_dir / f"{args.protein}.bin{int(args.bin_size)}.mc{int(args.iterations)}.tracks.png"

        save_enrichment_plot(df, out_png=enrich_png, title=f"{args.protein}: epitope enrichment (Monte Carlo, BY FDR)")
        print(f"Wrote: {enrich_png}")

        if tracks is not None and allele_order is not None:
            save_tracks_heatmap(tracks, allele_order, out_png=heat_png, title=f"{args.protein}: binary epitope tracks")
            print(f"Wrote: {heat_png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())