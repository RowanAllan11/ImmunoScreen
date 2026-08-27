from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.fragmentation import iter_fasta_inputs, iter_table_inputs, fragment_to_tables, _filter_variable_region_stops
from src.naming import make_run_label


def _parse_kmers(values: list[str]) -> list[int]:
    out: list[int] = []
    for v in values:
        out.append(int(v))
    if not out:
        raise ValueError("No kmers specified.")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Fragment protein sequences into overlapping k-mers.")
    ap.add_argument("--input-type", choices=["fasta", "csv"], default="csv", help="Input type. Default: csv")

    ap.add_argument(
        "--tag",
        required=True,
        help="Initial run tag used in output naming, e.g. AAV9, VR4, VR6.",
    )

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
        default=["criteria"],
        help="Metadata columns to carry through (space-separated), e.g. --metadata-cols criteria",
    )
    ap.add_argument("--chunksize", type=int, default=50_000, help="Chunk size for csv/tsv streaming. Default: 50000")
    ap.add_argument(
        "--sep",
        type=str,
        default=",",
        help="Optional separator override for table input (default: is comma).",
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

    run_label = make_run_label(args.tag, kmers)
    run_out_dir = args.out_dir / run_label

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
        
    stop_codon_stats = {"filtered": 0}

    inputs = _filter_variable_region_stops(
        inputs,
        var_start=args.var_start,
        var_end=args.var_end,
        stats=stop_codon_stats,
    )

    outputs = fragment_to_tables(
        inputs,
        kmers=kmers,
        out_dir=run_out_dir,
        metadata_cols=metadata_cols,
        var_only=args.var_only,
        var_start=args.var_start,
        var_end=args.var_end,
        var_mode=args.var_mode,
    )
    
    if stop_codon_stats["filtered"] > 0:
        print(
            f"Filtered {stop_codon_stats['filtered']:,} sequence(s) "
            f"containing '*' within variable region "
            f"{args.var_start}-{args.var_end}."
        )

    for k, p in outputs.items():
        print(f"wrote {k}: {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
