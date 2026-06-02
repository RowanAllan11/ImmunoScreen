#!/usr/bin/env python3
"""
Filter MHCflurry predictions for UNIQUE peptides and propagate to variant-level rows
using the new tabular fragmentation outputs:

  - unique_peptides.tsv
  - peptide_variant_map.tsv

This script intentionally DOES NOT support the legacy per-k fragmentation files.

Typical workflow:
  1) Fragment library with scripts/run_fragmentation.py --tabular ...
  2) Run mhcflurry-predict on unique_peptides.tsv (peptide column only, allele expanded)
  3) Use this script to filter and join positions / metadata.

Input assumptions:
  - unique_peptides.tsv has columns: peptide_id, peptide, k, occurrence_count
  - peptide_variant_map.tsv has: peptide_id, variant_id, start, end, (plus metadata cols like criteria)
  - mhcflurry predictions TSV has: peptide, allele, mhcflurry_affinity_percentile, mhcflurry_presentation_percentile
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_UNIQUE_COLS = {"peptide_id", "peptide", "k", "occurrence_count"}
REQUIRED_MAP_COLS = {"peptide_id", "variant_id", "start", "end"}
REQUIRED_MHC_COLS = {
    "peptide",
    "allele",
    "mhcflurry_affinity_percentile",
    "mhcflurry_presentation_percentile",
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Filter MHCflurry predictions and join variant positions (new tabular format).")

    ap.add_argument("--unique-peptides", type=Path, required=True, help="Path to unique_peptides.tsv")
    ap.add_argument("--peptide-map", type=Path, required=True, help="Path to peptide_variant_map.tsv")
    ap.add_argument("--mhcflurry", type=Path, required=True, help="Path to mhcflurry predictions TSV")
    ap.add_argument("-o", "--out", type=Path, required=True, help="Output TSV path")

    ap.add_argument(
        "--affinity-percentile-threshold",
        type=float,
        default=2.0,
        help="Keep rows with mhcflurry_affinity_percentile < threshold (default: 2.0)",
    )
    ap.add_argument(
        "--alleles",
        type=Path,
        default=None,
        help="Optional alleles txt file (one allele per line) to subset predictions",
    )

    args = ap.parse_args()

    # ---- load unique peptides ----
    uniq = pd.read_csv(args.unique_peptides, sep="\t", dtype=str)
    missing = REQUIRED_UNIQUE_COLS - set(uniq.columns)
    if missing:
        raise ValueError(f"{args.unique_peptides} missing required columns: {sorted(missing)}")

    uniq["peptide"] = uniq["peptide"].astype(str)
    uniq["k"] = pd.to_numeric(uniq["k"], errors="coerce").astype("Int64")
    uniq["occurrence_count"] = pd.to_numeric(uniq["occurrence_count"], errors="coerce").astype("Int64")

    # Ensure unique mapping from peptide -> peptide_id. In your format, peptide_id is unique per (peptide,k),
    # but peptide strings can repeat across different k. So map by (peptide, k).
    uniq_key = uniq[["peptide_id", "peptide", "k", "occurrence_count"]].copy()

    # ---- load peptide map ----
    pmap = pd.read_csv(args.peptide_map, sep="\t", dtype=str)
    missing = REQUIRED_MAP_COLS - set(pmap.columns)
    if missing:
        raise ValueError(f"{args.peptide_map} missing required columns: {sorted(missing)}")

    # normalize numeric coords
    pmap["start"] = pd.to_numeric(pmap["start"], errors="coerce").astype("Int64")
    pmap["end"] = pd.to_numeric(pmap["end"], errors="coerce").astype("Int64")

    # ---- load mhcflurry predictions ----
    mhc = pd.read_csv(args.mhcflurry, sep="\t", dtype=str)
    missing = REQUIRED_MHC_COLS - set(mhc.columns)
    if missing:
        raise ValueError(f"{args.mhcflurry} missing required columns: {sorted(missing)}")

    mhc["peptide"] = mhc["peptide"].astype(str)
    mhc["allele"] = mhc["allele"].astype(str)
    mhc["mhcflurry_affinity_percentile"] = pd.to_numeric(mhc["mhcflurry_affinity_percentile"], errors="coerce")
    mhc["mhcflurry_presentation_percentile"] = pd.to_numeric(mhc["mhcflurry_presentation_percentile"], errors="coerce")

    # Add k from peptide length (mhcflurry returns only peptide sequence)
    mhc["length"] = mhc["peptide"].str.len().astype(int)
    mhc["k"] = mhc["length"]

    # Optional allele subset
    if args.alleles is not None:
        allele_set: set[str] = set()
        with args.alleles.open("r", encoding="utf-8") as fh:
            for line in fh:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                allele_set.add(s)
        if allele_set:
            mhc = mhc[mhc["allele"].isin(allele_set)].copy()

    # Filter by affinity percentile
    mhc = mhc.dropna(subset=["mhcflurry_affinity_percentile"])
    mhc = mhc[mhc["mhcflurry_affinity_percentile"] < float(args.affinity_percentile_threshold)].copy()

    # Join mhc predictions -> peptide_id via (peptide,k)
    merged = mhc.merge(
        uniq_key,
        on=["peptide", "k"],
        how="left",
        validate="many_to_one",
    )

    n_missing_pid = int(merged["peptide_id"].isna().sum())
    if n_missing_pid:
        raise ValueError(
            f"{n_missing_pid} filtered MHCflurry rows could not be mapped to peptide_id using "
            f"{args.unique_peptides}. Ensure you predicted peptides from that exact unique_peptides.tsv."
        )

    # Fan-out to variant occurrences (adds start/end + any metadata cols present)
    out = merged.merge(pmap, on="peptide_id", how="left", validate="many_to_many")

    # sanity: after fan-out, variant_id should exist
    if "variant_id" not in out.columns:
        raise ValueError("peptide_variant_map.tsv did not provide 'variant_id' after join (unexpected).")

    n_missing_variant = int(out["variant_id"].isna().sum())
    if n_missing_variant:
        raise ValueError(
            f"{n_missing_variant} rows missing variant_id after join. "
            "Check peptide_variant_map.tsv matches unique_peptides.tsv."
        )

    # Rename columns to your preferred capitalisation (matching your existing combined TSV style)
    out = out.rename(
        columns={
            "mhcflurry_affinity_percentile": "MHCflurry_affinity_percentile",
            "mhcflurry_presentation_percentile": "MHCflurry_presentation_percentile",
        }
    )

    # Ensure consistent ints/NA
    out["start"] = out["start"].astype("Int64")
    out["end"] = out["end"].astype("Int64")
    out["k"] = pd.to_numeric(out["k"], errors="coerce").astype("Int64")
    out["length"] = pd.to_numeric(out["length"], errors="coerce").astype("Int64")

    # Output columns (include criteria if present)
    cols = [
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
    # include all extra metadata cols from pmap (e.g. criteria) after the core cols
    extra_meta = [c for c in pmap.columns if c not in {"peptide_id", "variant_id", "start", "end"}]
    cols.extend(extra_meta)
    cols.extend(["MHCflurry_affinity_percentile", "MHCflurry_presentation_percentile"])

    # create missing cols if any (robustness)
    for c in cols:
        if c not in out.columns:
            out[c] = pd.NA

    out = out[cols].sort_values(
        ["allele", "MHCflurry_affinity_percentile", "variant_id", "peptide_id"],
        ascending=[True, True, True, True],
        na_position="last",
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, sep="\t", index=False)
    print(f"Wrote {len(out)} rows to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
python src/mhcflurry_filter.py \
  --unique-peptides data/output/fragmentation/variants_vr5_9/unique_peptides.tsv \
  --peptide-map data/output/fragmentation/variants_vr5_9/peptide_variant_map.tsv \
  --mhcflurry data/output/mhcflurry/VR5_9mer/VR5_v3_9mer.unique_peptides.mhcflurry.tsv \
  --affinity-percentile-threshold 2.0 \
  -o data/output/mhcflurry_filtered/vr5_9_MHCflurry.tsv
"""