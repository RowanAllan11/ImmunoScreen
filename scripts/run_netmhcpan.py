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

    peptides_tsv = args.peptides.resolve()
    alleles_path = args.alleles.resolve()

    if not peptides_tsv.is_file():
        raise FileNotFoundError(
            f"unique_peptides.tsv not found: {peptides_tsv}"
        )

    if peptides_tsv.name != "unique_peptides.tsv":
        raise ValueError(
            "Expected an input file named 'unique_peptides.tsv', "
            f"but received: {peptides_tsv.name}"
        )

    if not alleles_path.is_file():
        raise FileNotFoundError(
            f"Allele file not found: {alleles_path}"
        )

    kmers = sorted(set(int(k) for k in args.kmers))

    if not kmers:
        raise ValueError("No k-mer lengths were provided.")

    if any(k <= 0 for k in kmers):
        raise ValueError(
            f"All k-mer lengths must be positive: {kmers}"
        )

    run_label = peptides_tsv.parent.name
    run_out_dir = args.out_dir / run_label
    run_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run label: {run_label}")
    print(f"NetMHCpan output directory: {run_out_dir}")

    for k in kmers:
        out = run_netmhcpan_for_k(
            peptides_tsv=peptides_tsv,
            k=k,
            alleles_path=alleles_path,
            netmhcpan_path=args.netmhcpan,
            outdir=run_out_dir,
            output_format=args.output_format,
            extra=list(args.extra),
        )

        print(f"Wrote k={k}: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())