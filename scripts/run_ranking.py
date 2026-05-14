from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.epitope_ranking import rank_mhcflurry_epitopes, rank_score  # noqa: E402

# How to run:
# python3 scripts/run_ranking.py \
#   --affinity-threshold 2 \
#   --kmers 8,9,10,11,12,13,14,15 \
#   --dedupe-greedy \
#   --dedupe-min-overlap-aa 1 \
#   --out data/output/epitopes/mhcflurry_epitopes.tsv


def _parse_kmers(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="Rank peptide epitopes from MHCflurry outputs (affinity_percentile).")
    ap.add_argument("--fragment-dir", default=str(REPO_ROOT / "data/output/fragmentation"))
    ap.add_argument("--mhcflurry-dir", default=str(REPO_ROOT / "data/output/mhcflurry"))
    ap.add_argument("--out", default=str(REPO_ROOT / "data/output/epitopes/mhcflurry_epitopes.tsv"))
    ap.add_argument("--kmers", default="8,9,10,11,12,13,14,15")
    ap.add_argument(
        "--affinity-threshold",
        type=float,
        default=2.0,
        help="Keep peptides with affinity_percentile <= threshold (bottom X percent)",
    )
    ap.add_argument("--top-n-per-allele-protein", type=int, default=0)

    ap.add_argument(
        "--dedupe-greedy",
        action="store_true",
        help="Greedy non-overlap per (allele, protein): keep best peptide, drop overlapping ones",
    )
    ap.add_argument(
        "--dedupe-min-overlap-aa",
        type=int,
        default=1,
        help="Overlap (in aa) that counts as redundant. 1 means any shared residue.",
    )

    args = ap.parse_args()

    epitopes = rank_mhcflurry_epitopes(
        fragment_dir=Path(args.fragment_dir),
        mhcflurry_dir=Path(args.mhcflurry_dir),
        kmers=_parse_kmers(args.kmers),
        affinity_threshold=float(args.affinity_threshold),
        top_n_per_allele_protein=int(args.top_n_per_allele_protein),
        dedupe_greedy=bool(args.dedupe_greedy),
        dedupe_min_overlap_aa=int(args.dedupe_min_overlap_aa),
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(
            [
                "allele",
                "protein",
                "peptide",
                "start",
                "end",
                "length",
                "affinity_percentile",
                "presentation_percentile",
                "rank_score",
                "supporting_kmers",
            ]
        )
        for e in epitopes:
            pres: Optional[float] = e.presentation_percentile
            w.writerow(
                [
                    e.allele,
                    e.protein,
                    e.peptide,
                    e.start,
                    e.end,
                    e.length,
                    f"{e.affinity_percentile:.6g}",
                    "" if pres is None else f"{pres:.6g}",
                    rank_score(e),
                    e.supporting_kmers,
                ]
            )

    print(f"Wrote: {out_path} ({len(epitopes)} epitopes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())