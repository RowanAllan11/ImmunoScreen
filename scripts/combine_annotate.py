#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from combine_predictions import combine_predictions
from mutation_label import attach_mutation_labels


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Combine NetMHCpan + MHCflurry outputs and add VR mutation labels."
    )

    ap.add_argument("--st", required=True)
    ap.add_argument("--netmhcpan-file", type=Path, required=True)
    ap.add_argument("--mhcflurry-file", type=Path, required=True)
    ap.add_argument("--library-csv", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)

    ap.add_argument("--variant-id-col", default="variant_id")
    ap.add_argument("--library-id-col", default="Geneid")
    ap.add_argument("--seq-col", default="twist_seq_prot")
    ap.add_argument("--var-start", type=int, default=8)
    ap.add_argument("--var-end", type=int, default=24)
    ap.add_argument("--wt-vr", default=None)

    args = ap.parse_args()

    combined = combine_predictions(
        netmhcpan_file=args.netmhcpan_file,
        mhcflurry_file=args.mhcflurry_file,
    )

    annotated = attach_mutation_labels(
        combined,
        args.library_csv,
        variant_id_col=args.variant_id_col,
        library_id_col=args.library_id_col,
        seq_col=args.seq_col,
        var_start=args.var_start,
        var_end=args.var_end,
        wt_vr=args.wt_vr,
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    out_path = args.outdir / f"{args.st}_combined_annotated.tsv"

    annotated.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote: {out_path} ({len(annotated)} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())