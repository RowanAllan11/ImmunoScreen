from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from mhcflurry import filter_and_expand_mhcflurry_predictions


def main() -> int:
    ap = argparse.ArgumentParser(description="Filter mhcflurry predictions and expand to variant rows.")
    ap.add_argument("--unique-peptides", type=Path, required=True, help="Path to unique_peptides.tsv")
    ap.add_argument("--peptide-map", type=Path, required=True, help="Path to peptide_variant_map.tsv")
    ap.add_argument("--mhcflurry", type=Path, required=True, help="Path to mhcflurry predictions TSV")
    ap.add_argument("-o", "--out", type=Path, required=True, help="Output TSV path")
    ap.add_argument("--affinity-percentile-threshold", type=float, default=2.0)
    ap.add_argument("--alleles", type=Path, default=None, help="Optional alleles txt (one per line)")
    args = ap.parse_args()

    out_df = filter_and_expand_mhcflurry_predictions(
        unique_peptides_tsv=args.unique_peptides,
        peptide_map_tsv=args.peptide_map,
        mhcflurry_tsv=args.mhcflurry,
        affinity_percentile_threshold=args.affinity_percentile_threshold,
        alleles_path=args.alleles,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, sep="\t", index=False)
    print(f"Wrote {len(out_df)} rows to: {args.out}")
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