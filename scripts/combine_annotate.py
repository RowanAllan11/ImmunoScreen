#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.combine_predictions import combine_predictions
from src.mutation_label import attach_mutation_labels


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Combine NetMHCpan + MHCflurry outputs and add VR mutation labels."
    )

    ap.add_argument("--netmhcpan-file", type=Path, required=True)
    ap.add_argument("--mhcflurry-file", type=Path, required=True)
    ap.add_argument("--i", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, default=REPO_ROOT / "data/output/combined")

    ap.add_argument("--variant-id-col", default="variant_id")
    ap.add_argument("--library-id-col", default=None)
    ap.add_argument("--seq-col", default=None)
    ap.add_argument("--var-start", type=int, default=None)
    ap.add_argument("--var-end", type=int, default=None)
    ap.add_argument("--wt-vr", default=None)

    args = ap.parse_args()

    net_run_label = args.netmhcpan_file.parent.name
    mhc_run_label = args.mhcflurry_file.parent.name

    if net_run_label != mhc_run_label:
        raise ValueError(
            "NetMHCpan and MHCflurry inputs appear to come from different runs: "
            f"{net_run_label!r} versus {mhc_run_label!r}"
        )

    run_label = net_run_label


    combined = combine_predictions(
        netmhcpan_file=args.netmhcpan_file,
        mhcflurry_file=args.mhcflurry_file,
    )

    annotated = attach_mutation_labels(
        combined,
        args.i,
        variant_id_col=args.variant_id_col,
        library_id_col=args.library_id_col,
        seq_col=args.seq_col,
        var_start=args.var_start,
        var_end=args.var_end,
        wt_vr=args.wt_vr,
    )

    run_out_dir = args.outdir / run_label
    run_out_dir.mkdir(parents=True, exist_ok=True)

    out_path = run_out_dir / "combined_annotated.tsv"

    annotated.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote: {out_path} ({len(annotated)} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
