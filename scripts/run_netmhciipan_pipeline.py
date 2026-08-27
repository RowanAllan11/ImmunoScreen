#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.netmhciipan import (
    DEFAULT_NETMHCII_PATH,
    expand_predictions,
    parse_netmhciipan_xls,
    read_peptide_map,
    read_unique_peptides,
    run_netmhciipan,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run NetMHCIIpan, classify EL-rank binders, and map predictions to variants."
    )
    parser.add_argument("--peptides", type=Path, required=True)
    parser.add_argument("--peptide-map", type=Path, required=True)
    parser.add_argument("--alleles", nargs="+", required=True)
    parser.add_argument("--netmhciipan", type=Path, default=DEFAULT_NETMHCII_PATH)
    parser.add_argument("--el-rank-binder-threshold", type=float, default=5.0)
    parser.add_argument(
        "--outdir", type=Path, default=REPO_ROOT / "data/output/netmhciipan"
    )
    args = parser.parse_args()

    unique = read_unique_peptides(args.peptides.resolve())
    peptide_map = read_peptide_map(args.peptide_map.resolve())
    alleles = list(dict.fromkeys(a.strip() for a in args.alleles if a.strip()))
    run_label = args.peptides.resolve().parent.name
    run_dir = args.outdir.resolve() / run_label
    raw_dir = run_dir / "raw"
    raw_path = raw_dir / f"{run_label}.netmhciipan.xls"
    mapped_path = run_dir / "predictions_mapped.tsv"

    run_netmhciipan(
        unique["peptide"].drop_duplicates().tolist(),
        alleles,
        args.netmhciipan.resolve(),
        raw_path,
    )
    predictions = parse_netmhciipan_xls(raw_path)
    observed = set(predictions["allele"])
    if observed != set(alleles):
        raise ValueError(
            f"Requested alleles {sorted(alleles)}, observed {sorted(observed)}"
        )
    mapped = expand_predictions(
        predictions, unique, peptide_map, args.el_rank_binder_threshold
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    mapped.to_csv(mapped_path, sep="\t", index=False)
    binders = int(mapped["netMHCIIpan_EL_rank_binder"].sum())
    print(f"Wrote raw NetMHCIIpan output: {raw_path}")
    print(f"Wrote mapped NetMHCIIpan output: {mapped_path} ({len(mapped):,} rows)")
    print(
        f"EL-rank binders: {binders:,}/{len(mapped):,} rows "
        f"(< {args.el_rank_binder_threshold:g}%)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
