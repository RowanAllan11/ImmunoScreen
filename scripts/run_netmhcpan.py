from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Ensure project and src/ are importable
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from src.netmhcpan import run_netmhcpan_for_k, DEFAULT_NETMHC_PATH


def main() -> int:
    ap = argparse.ArgumentParser(description="Run netMHCpan on unique_peptides.tsv")
    ap.add_argument("--peptides", type=Path, required=True, help="unique_peptides.tsv (columns: peptide, k, ...)")
    ap.add_argument("--kmers", type=int, nargs="+", required=True, help="k-mer lengths to run")
    ap.add_argument("--alleles", type=Path, required=True, help="alleles txt (one per line)")
    ap.add_argument("--netmhcpan", type=Path, default=DEFAULT_NETMHC_PATH, help="path to netMHCpan binary/wrapper")
    ap.add_argument("--outdir", type=Path, default=REPO_ROOT / "data" / "output" / "netmhcpan")
    ap.add_argument("--output-format", choices=["xls", "txt"], default="xls")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="extra args forwarded to netMHCpan")

    args = ap.parse_args()

    for k in args.kmers:
        out = run_netmhcpan_for_k(
            peptides_tsv=args.peptides,
            k=int(k),
            alleles_path=args.alleles,
            netmhcpan_path=args.netmhcpan,
            outdir=args.outdir,
            output_format=args.output_format,
            extra=list(args.extra),
        )
        print(f"Wrote: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())