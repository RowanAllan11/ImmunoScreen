#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.variable_analysis import load_aligned_fasta, plot_peptide_msa_window 

# Example Use
''' python3 scripts/view_epitope_msa.py \
  --aln data/output/msa/capsid.mafft.fasta \
  --reference-name AAV9_VP1 \
  --peptide SSYAHSQSL \
  --flank 6 \
  --out data/output/plots/msa_peptide_SSYAHSQSL.png
'''

def main() -> int:
    ap = argparse.ArgumentParser(description="Visualise a peptide window in a capsid MSA (reference-mapped).")
    ap.add_argument(
        "--aln",
        default=str(REPO_ROOT / "data/output/msa/capsid.mafft.fasta"),
        help="Aligned FASTA (MAFFT output).",
    )
    ap.add_argument("--reference-name", default="AAV9_VP1", help="Reference record id in the aligned FASTA.")
    ap.add_argument("--peptide", required=True, help="Peptide sequence to find in the reference (exact match).")
    ap.add_argument("--flank", type=int, default=5, help="Flanking residues to show on each side (default 5).")
    ap.add_argument("--max-rows", type=int, default=30, help="Max sequences (rows) to display (default 30).")
    ap.add_argument(
        "--out",
        default=None,
        help="Output PNG path. If omitted, shows interactively.",
    )
    args = ap.parse_args()

    aln_path = Path(args.aln)
    if not aln_path.exists():
        raise SystemExit(f"Missing aligned FASTA: {aln_path}")

    msa = load_aligned_fasta(aln_path)
    out = Path(args.out) if args.out else None

    plot_peptide_msa_window(
        msa,
        peptide=str(args.peptide),
        reference_name=str(args.reference_name),
        flank=int(args.flank),
        max_rows=int(args.max_rows),
        out_png=out,
    )
    if out:
        print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())