from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.fragmentation import iter_fasta_inputs, iter_table_inputs, fragment_to_tables


def _parse_kmers(values: list[str]) -> list[int]:
    out: list[int] = []
    for v in values:
        out.append(int(v))
    if not out:
        raise ValueError("No kmers specified.")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Fragment protein sequences into overlapping k-mers.")
    ap.add_argument("--input-type", choices=["fasta", "csv"], default="fasta", help="Input type. Default: fasta")
    ap.add_argument("--i", type=Path, required=False, default=None, help="Input path")
    ap.add_argument("--out-dir", type=Path, default=Path("data/output/fragmentation"), help="Output directory")

    ap.add_argument(
        "--kmers",
        nargs="+",
        default=None,
        help="k-mer sizes, e.g. --kmers 8 9 10 ...",
    )

    # CSV/TSV-mode options
    ap.add_argument("--id-col", default="Geneid", help="ID column for csv/tsv input. Default: Geneid")
    ap.add_argument(
        "--sequence-col",
        default="twist_seq_prot",
        help="Sequence column for csv/tsv. Default: twist_seq_prot",
    )
    ap.add_argument(
        "--metadata-cols",
        nargs="*",
        default=[],
        help="Metadata columns to carry through (space-separated), e.g. --metadata-cols criteria",
    )
    ap.add_argument("--chunksize", type=int, default=50_000, help="Chunk size for csv/tsv streaming. Default: 50000")
    ap.add_argument(
        "--sep",
        type=str,
        default=None,
        help="Optional separator override for table input (default: inferred from extension)",
    )

    ap.add_argument(
        "--var-only",
        action="store_true",
        help="Keep only peptides overlapping/contained within the variable region window (requires --var-start/--var-end).",
    )
    ap.add_argument("--var-start", type=int, default=None, help="Variable region start (1-based, inclusive)")
    ap.add_argument("--var-end", type=int, default=None, help="Variable region end (1-based, inclusive)")
    ap.add_argument(
        "--var-mode",
        choices=["overlap", "contained"],
        default="overlap",
        help="Variable region filter mode. overlap=any intersection, contained=fully inside. Default: overlap",
    )

    args = ap.parse_args()
    kmers = _parse_kmers(args.kmers)

    if args.input_type == "fasta":
        inputs = iter_fasta_inputs(args.i)
        metadata_cols = []
    else:
        inputs = iter_table_inputs(
            args.i,
            id_col=args.id_col,
            sequence_col=args.sequence_col,
            metadata_cols=args.metadata_cols,
            chunksize=args.chunksize,
            sep=args.sep,
        )
        metadata_cols = list(args.metadata_cols)

    outputs = fragment_to_tables(
        inputs,
        kmers=kmers,
        out_dir=args.out_dir,
        metadata_cols=metadata_cols,
        var_only=args.var_only,
        var_start=args.var_start,
        var_end=args.var_end,
        var_mode=args.var_mode,
    )
    for k, p in outputs.items():
        print(f"wrote {k}: {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


"""
python scripts/run_fragmentation.py \
  --input-type csv \
  --i data/input/libraries/VR5_v3_final_library_detailed.csv \
  --metadata-cols criteria \
  --var-only --var-start 8 --var-end 24 --var-mode overlap \
  --out-dir data/output/fragmentation/variants_vr5_9 \
  --kmers 9
"""