#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.mhcflurry_pos_mapping import (
    coords_to_sequence,
    infer_protein_from_fragmentation_path,
    infer_protein_from_mhcflurry_path,
    load_fragmentation_coords,
    load_mhcflurry_rows,
    map_scores_to_positions_max,
    write_position_scores_tsv,
    infer_allele_from_mhcflurry_filename,
    _parse_allele_filter
)
from src.visualize_mhc_heatmap import render_heatmap_from_dir 

# How to call
# 1. Run mapping only - python3 scripts/run_mapping.py --alleles-file data/input/alleles.txt
# 2. Run mapping + plotting for all proteins - python3 scripts/run_mapping.py --alleles-file data/input/alleles.txt --plot
# 3. Run mapping + plotting for one protein - python3 scripts/run_mapping.py --alleles-file data/input/alleles.txt --plot --plot-protein AAV9_VP1

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-min", type=int, default=8)
    ap.add_argument("--k-max", type=int, default=15)
    ap.add_argument("--out-dir", default="data/output/mhcflurry_position_scores")

    # NEW: allele filtering (optional)
    ap.add_argument(
        "--alleles",
        default=None,
        help="Comma-separated allele names to include (matches filename allele part exactly). Example: 'HLA-A*02:01,HLA-A*11:01'",
    )
    ap.add_argument(
        "--alleles-file",
        default=None,
        help="Path to text file with one allele per line (lines starting with # ignored).",
    )

    # Optional plotting step (off by default)
    ap.add_argument("--plot", action="store_true", help="Generate heatmap PNG(s) from the output TSVs")
    ap.add_argument("--plot-protein", default=None, help="Protein name to plot (e.g. AAV9_VP1). If omitted, plot all proteins.")
    ap.add_argument("--plot-out", default="data/output/plots", help="Directory to write PNGs")
    ap.add_argument("--plot-xtick-step", type=int, default=50)
    ap.add_argument("--plot-vmax", type=float, default=None)

    args = ap.parse_args()
    allele_filter = _parse_allele_filter(args)

    # Load and merge fragmentation coords across k for each protein
    coords_by_protein: Dict[str, dict] = {}
    for k in range(args.k_min, args.k_max + 1):
        for fp in sorted(glob.glob(f"data/output/fragmentation/*_{k}mer.tsv")):
            protein = infer_protein_from_fragmentation_path(fp)
            coords_by_protein.setdefault(protein, {}).update(load_fragmentation_coords(fp))

    if not coords_by_protein:
        raise SystemExit(f"No fragmentation TSVs found for k={args.k_min}..{args.k_max}")

    # Precompute sequences per protein from fragmentation
    seq_by_protein: Dict[str, str] = {p: coords_to_sequence(c) for p, c in coords_by_protein.items()}

    # Accumulate max across ALL k per (protein, allele, pos_0)
    acc: Dict[Tuple[str, str, int], float] = {}

    for k in range(args.k_min, args.k_max + 1):
        for mp in sorted(glob.glob(f"data/output/mhcflurry/*_{k}mer.*.mhcflurry.tsv")):
            try:
                allele = infer_allele_from_mhcflurry_filename(mp)
            except ValueError:
                continue

            if allele_filter is not None and allele not in allele_filter:
                continue

            protein = infer_protein_from_mhcflurry_path(mp)
            coords = coords_by_protein.get(protein)
            if coords is None:
                continue

            rows = load_mhcflurry_rows(mp)
            mapped = map_scores_to_positions_max(coords, rows)

            for (allele_from_rows, pos_0), score in mapped.items():
                # Prefer allele from rows if present; but keep filename allele as key if mismatch
                use_allele = allele_from_rows or allele
                if allele_filter is not None and use_allele not in allele_filter:
                    continue

                key = (protein, use_allele, pos_0)
                prev = acc.get(key)
                if prev is None or score > prev:
                    acc[key] = score

    # Write one combined output per protein+allele (across k)
    os.makedirs(args.out_dir, exist_ok=True)
    by_protein_allele: Dict[Tuple[str, str], Dict[Tuple[str, int], float]] = {}
    for (protein, allele, pos_0), score in acc.items():
        by_protein_allele.setdefault((protein, allele), {})[(allele, pos_0)] = score

    for (protein, allele), max_by_allele_pos in sorted(by_protein_allele.items()):
        out_path = os.path.join(args.out_dir, f"{protein}.{allele}.k{args.k_min}-{args.k_max}.posmax.tsv")
        write_position_scores_tsv(out_path, protein, max_by_allele_pos, sequence=seq_by_protein.get(protein))
        print(f"Wrote: {out_path}")

    # Optional: render heatmaps from the output dir
    if args.plot:
        input_dir = Path(args.out_dir)
        plot_dir = Path(args.plot_out)

        proteins_to_plot = [args.plot_protein] if args.plot_protein else sorted(coords_by_protein.keys())
        for protein in proteins_to_plot:
            out_png = plot_dir / f"{protein}.k{args.k_min}-{args.k_max}.heatmap.png"
            render_heatmap_from_dir(
                input_dir=input_dir,
                protein=protein,
                out_png=out_png,
                xtick_step=args.plot_xtick_step,
                vmax=args.plot_vmax,
            )
            print(f"Wrote: {out_png}")


if __name__ == "__main__":
    main()