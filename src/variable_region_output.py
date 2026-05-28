#!/usr/bin/env python3
"""Annotate combined epitope table with AAV9 VP1 variable region (VR) membership.

This script builds on the output produced by `src/combine.py`, e.g.:
  data/output/combined/AAV9_combined_netmhcpan_mhcflurry.tsv

It adds two columns:
- Variable_region_presence: 1 if the peptide overlaps any VR interval, else 0
- Variable_region: comma-separated list of VR labels overlapped (or NA)

VR intervals (approximate AAV9 VP1 residues; 1-based, inclusive):
  VR-I    ~262–269
  VR-II   ~327–333
  VR-III  ~381–389
  VR-IV   ~446–456
  VR-V    ~488–504
  VR-VI   ~528–538
  VR-VII  ~546–556
  VR-VIII ~581–594
  VR-IX   ~704–714

Overlap definition:
- Uses peptide start/end coordinates if present in the combined file.
- Coordinates are interpreted as 1-based inclusive.
- A peptide is considered to "land in" a variable region if it overlaps the VR interval.

Example:
  python src/variable_region_output.py \
    --st AAV9 \
    -i data/output/combined/AAV9_combined_netmhcpan_mhcflurry.tsv \
    -o data/output/combined
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


VR_INTERVALS: List[Tuple[str, int, int]] = [
    ("VR-I", 262, 269),
    ("VR-II", 327, 333),
    ("VR-III", 381, 389),
    ("VR-IV", 446, 456),
    ("VR-V", 488, 504),
    ("VR-VI", 528, 538),
    ("VR-VII", 546, 556),
    ("VR-VIII", 581, 594),
    ("VR-IX", 704, 714),
]


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start <= b_end and b_start <= a_end


def _annotate_row(start: Optional[float], end: Optional[float]) -> tuple[int, Optional[str]]:
    if start is None or end is None or pd.isna(start) or pd.isna(end):
        return 0, None

    try:
        s = int(start)
        e = int(end)
    except (TypeError, ValueError):
        return 0, None

    if e < s:
        s, e = e, s

    hits: List[str] = []
    for label, vr_s, vr_e in VR_INTERVALS:
        if _overlaps(s, e, vr_s, vr_e):
            hits.append(label)

    if not hits:
        return 0, None
    return 1, ",".join(hits)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--st", required=True, help="ST tag (e.g., AAV9) used in output filename")
    ap.add_argument(
        "-i",
        "--input",
        required=True,
        type=Path,
        help="Combined TSV from src/combine.py",
    )
    ap.add_argument(
        "-o",
        "--outdir",
        required=True,
        type=Path,
        help="Output directory",
    )
    ap.add_argument(
        "--start-col",
        default="start",
        help="Column name for peptide start position (1-based inclusive)",
    )
    ap.add_argument(
        "--end-col",
        default="end",
        help="Column name for peptide end position (1-based inclusive)",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.input, sep="\t")

    if args.start_col not in df.columns or args.end_col not in df.columns:
        raise ValueError(
            f"Input TSV must contain start/end columns (got start_col={args.start_col}, end_col={args.end_col}). "
            f"Available columns: {list(df.columns)}"
        )

    # Compute annotations
    pres = []
    vr = []
    for s, e in zip(df[args.start_col], df[args.end_col]):
        p, lab = _annotate_row(s, e)
        pres.append(p)
        vr.append(lab)

    df["Variable_region_presence"] = pres
    df["Variable_region"] = vr

    # Output
    args.outdir.mkdir(parents=True, exist_ok=True)
    out_path = args.outdir / f"{args.st}_combined_netmhcpan_mhcflurry.variable_regions.tsv"
    df.to_csv(out_path, sep="\t", index=False)

    print(f"Wrote {len(df)} rows to: {out_path}")


if __name__ == "__main__":
    main()
