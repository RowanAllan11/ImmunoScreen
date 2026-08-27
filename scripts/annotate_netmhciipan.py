#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.mutation_label import attach_mutation_labels


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add VR sequence and mutation annotations to mapped NetMHCIIpan predictions."
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--library-id-column", required=True)
    parser.add_argument("--sequence-column", required=True)
    parser.add_argument("--variable-region-start", type=int, required=True)
    parser.add_argument("--variable-region-end", type=int, required=True)
    parser.add_argument("--wild-type-variable-region", required=True)
    parser.add_argument(
        "--outdir", type=Path,
        default=REPO_ROOT / "data/output/netmhciipan_annotated",
    )
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions, sep="\t", low_memory=False)
    annotated = attach_mutation_labels(
        predictions,
        args.library,
        variant_id_col="variant_id",
        library_id_col=args.library_id_column,
        seq_col=args.sequence_column,
        var_start=args.variable_region_start,
        var_end=args.variable_region_end,
        wt_vr=args.wild_type_variable_region,
    )
    run_label = args.predictions.parent.name
    output_dir = args.outdir / run_label
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "predictions_mapped_annotated.tsv"
    annotated.to_csv(output_path, sep="\t", index=False)
    print(f"Wrote annotated NetMHCIIpan output: {output_path} ({len(annotated):,} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
