#!/usr/bin/env python3
"""
Filter MHCflurry TSV outputs and add peptide start/end positions from fragmentation outputs.

Inputs (per k, per allele):
  data/output/mhcflurry/{ST}_VP1_{k}mer.{ALLELE}.mhcflurry.tsv

Where MHCflurry TSV has columns:
  peptide
  allele
  mhcflurry_affinity_percentile
  mhcflurry_presentation_percentile
  (and possibly others)

Start/end positions come from fragmentation TSV(s) in:
  data/output/fragmentation/
with columns including:
  peptide
  k
  start_0
  end_0_exclusive
(and/or start_1, end_1_inclusive)

Filtering:
  keep rows with mhcflurry_affinity_percentile < 2.0

Output:
  {outdir}/{ST}_MHCflurry_{kmer_range}.tsv

Columns:
  allele
  peptide
  start
  end
  length
  mhcflurry_affinity_percentile
  mhcflurry_presentation_percentile


Run it like:
python src/mhcflurry_filter.py \
  --st AAV9 \
  --kmers 8 9 10 11 12 13 14 \
  --alleles data/input/alleles_h2.txt \
  -o data/output/mhcflurry/filtered
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd


REQUIRED_MHCFLURRY_COLS = {
    "peptide",
    "allele",
    "mhcflurry_affinity_percentile",
    "mhcflurry_presentation_percentile",
}

REQUIRED_FRAG_COLS = {"peptide", "k", "start_0", "end_0_exclusive"}


def _read_alleles_file(path: Path) -> List[str]:
    alleles: List[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            alleles.append(s)
    if not alleles:
        raise ValueError(f"No alleles found in {path}")
    return alleles


def _kmer_range_label(kmers: Sequence[int]) -> str:
    ks = sorted(set(int(k) for k in kmers))
    if not ks:
        return "NA"
    if len(ks) == 1:
        return f"{ks[0]}mer"
    return f"{ks[0]}-{ks[-1]}mer"


def _find_fragmentation_tsvs(fragment_dir: Path, st: str) -> List[Path]:
    # Be permissive: accept anything with st in the name and .tsv suffix.
    # You can tighten this pattern later if your filenames are consistent.
    if not fragment_dir.exists():
        return []
    st_lower = st.lower()
    out: List[Path] = []
    for p in fragment_dir.glob("*.tsv"):
        if st_lower in p.name.lower():
            out.append(p)
    return sorted(out)


def _load_fragment_map(fragment_dir: Path, st: str, kmers: Set[int]) -> pd.DataFrame:
    """
    Returns a dataframe with unique mapping per (peptide, k):
      peptide, k, start, end   where start/end are 1-based inclusive coordinates.
    """
    tsvs = _find_fragmentation_tsvs(fragment_dir, st)
    if not tsvs:
        raise FileNotFoundError(
            f"Could not find fragmentation TSVs for ST={st} in: {fragment_dir}"
        )

    dfs: List[pd.DataFrame] = []
    for fp in tsvs:
        df = pd.read_csv(fp, sep="\t")
        missing = REQUIRED_FRAG_COLS - set(df.columns)
        if missing:
            # Not necessarily the right file; skip it.
            continue
        # Filter to relevant kmers early.
        df = df[df["k"].isin(sorted(kmers))].copy()
        if df.empty:
            continue

        # Convert to 1-based inclusive end.
        # Given: start_0 (0-based), end_0_exclusive (0-based exclusive)
        df["start"] = df["start_0"].astype(int) + 1
        df["end"] = df["end_0_exclusive"].astype(int)  # exclusive end -> inclusive 1-based == end_0_exclusive
        dfs.append(df[["peptide", "k", "start", "end"]])

    if not dfs:
        raise ValueError(
            f"Found fragmentation TSVs in {fragment_dir}, but none had required columns {sorted(REQUIRED_FRAG_COLS)} "
            f"for ST={st}"
        )

    frag = pd.concat(dfs, ignore_index=True)
    frag = frag.dropna(subset=["peptide", "k", "start", "end"])
    frag["k"] = frag["k"].astype(int)

    # Ensure uniqueness. If duplicates exist, keep the first deterministically.
    frag = frag.sort_values(["peptide", "k", "start", "end"]).drop_duplicates(
        subset=["peptide", "k"], keep="first"
    )

    return frag


def _mhcflurry_input_path(mhcflurry_dir: Path, st: str, k: int, allele: str) -> Path:
    # Matches: AAV9_VP1_8mer.H2-D*b.mhcflurry.tsv
    return mhcflurry_dir / f"{st}_VP1_{k}mer.{allele}.mhcflurry.tsv"


def _load_mhcflurry_one(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    missing = REQUIRED_MHCFLURRY_COLS - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")

    # Coerce numeric percentiles
    df["mhcflurry_affinity_percentile"] = pd.to_numeric(
        df["mhcflurry_affinity_percentile"], errors="coerce"
    )
    df["mhcflurry_presentation_percentile"] = pd.to_numeric(
        df["mhcflurry_presentation_percentile"], errors="coerce"
    )
    df["peptide"] = df["peptide"].astype(str)
    df["allele"] = df["allele"].astype(str)

    # Add k/length
    df["length"] = df["peptide"].str.len().astype(int)
    df["k"] = df["length"]

    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--st", required=True, help="ST tag to include in title (e.g., AAV9)")
    ap.add_argument(
        "--kmers",
        required=True,
        nargs="+",
        type=int,
        help="One or more k-mer lengths (e.g., --kmers 8 9 10 11)",
    )
    ap.add_argument(
        "--alleles",
        required=True,
        type=Path,
        help="Path to alleles txt file (one allele per line)",
    )
    ap.add_argument(
        "--mhcflurry-dir",
        type=Path,
        default=Path("data/output/mhcflurry"),
        help="Directory containing MHCflurry TSV files",
    )
    ap.add_argument(
        "--fragment-dir",
        type=Path,
        default=Path("data/output/fragmentation"),
        help="Directory containing fragmentation TSV files (for start/end mapping)",
    )
    ap.add_argument(
        "-o",
        "--outdir",
        required=True,
        type=Path,
        help="Output directory",
    )
    ap.add_argument(
        "--affinity-percentile-threshold",
        type=float,
        default=2.0,
        help="Keep rows with mhcflurry_affinity_percentile < threshold",
    )
    args = ap.parse_args()

    st: str = args.st
    kmers: Set[int] = set(args.kmers)
    alleles: List[str] = _read_alleles_file(args.alleles)

    # Load fragmentation mapping once
    frag_map = _load_fragment_map(args.fragment_dir, st=st, kmers=kmers)

    # Load all MHCflurry files (per k, per allele)
    dfs: List[pd.DataFrame] = []
    missing_files: List[Path] = []

    for k in sorted(kmers):
        for allele in alleles:
            fp = _mhcflurry_input_path(args.mhcflurry_dir, st=st, k=k, allele=allele)
            if not fp.exists():
                missing_files.append(fp)
                continue
            df = _load_mhcflurry_one(fp)

            # Ensure expected length matches requested k (defensive)
            df = df[df["k"] == int(k)].copy()
            if df.empty:
                continue

            dfs.append(df)

    if missing_files and not dfs:
        raise FileNotFoundError(
            "No MHCflurry input files found. Example missing path:\n"
            f"  {missing_files[0]}\n"
            "Check --mhcflurry-dir, --st, --kmers, and allele naming."
        )

    if not dfs:
        # Some files might be missing, but none loaded produced rows
        raise ValueError("Loaded 0 rows from MHCflurry TSV inputs (after basic parsing).")

    mhc = pd.concat(dfs, ignore_index=True)

    # Filter: affinity percentile < threshold
    mhc = mhc.dropna(subset=["mhcflurry_affinity_percentile"])
    mhc_filt = mhc[mhc["mhcflurry_affinity_percentile"] < float(args.affinity_percentile_threshold)].copy()

    # Join start/end from fragmentation on (peptide, k)
    merged = mhc_filt.merge(frag_map, on=["peptide", "k"], how="left", validate="many_to_one")

    # Rename/shape output
    out = merged.rename(
        columns={
            # start/end already in desired names
            "mhcflurry_affinity_percentile": "MHCflurry_affinity_percentile",
            "mhcflurry_presentation_percentile": "MHCflurry_presentation_percentile",
        }
    )

    # Recompute length (authoritative) and ensure ints
    out["length"] = out["peptide"].str.len().astype(int)

    # Keep requested columns
    out_cols = [
        "allele",
        "peptide",
        "start",
        "end",
        "length",
        "MHCflurry_affinity_percentile",
        "MHCflurry_presentation_percentile",
    ]
    for col in out_cols:
        if col not in out.columns:
            out[col] = pd.NA

    # Sorting: allele, affinity percentile ascending, presentation percentile ascending, peptide
    out = out.sort_values(
        ["allele", "MHCflurry_affinity_percentile", "MHCflurry_presentation_percentile", "peptide"],
        ascending=[True, True, True, True],
        na_position="last",
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    label = _kmer_range_label(sorted(kmers))
    out_path = args.outdir / f"{st}_MHCflurry_{label}.tsv"
    out[out_cols].to_csv(out_path, sep="\t", index=False)

    # Helpful warnings
    if missing_files:
        print(f"Warning: {len(missing_files)} expected MHCflurry files were not found.")
        print(f"First missing example: {missing_files[0]}")

    n_missing_pos = int(out["start"].isna().sum())
    if n_missing_pos:
        print(
            f"Warning: {n_missing_pos} rows missing start/end after join. "
            f"Check fragmentation outputs in {args.fragment_dir} for ST={st} and requested kmers."
        )

    print(f"Wrote {len(out)} filtered rows to: {out_path}")


if __name__ == "__main__":
    main()