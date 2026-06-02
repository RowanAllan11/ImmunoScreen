import argparse
import os
import subprocess
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
NETMHCPAN_WRAPPER = REPO_ROOT / "tools" / "netMHCpan-4.2" / "netMHCpan"


def read_lines(path: Path) -> list[str]:
    lines = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            lines.append(s)
    return lines


def write_peptides_temp(peptides: list[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for p in peptides:
            f.write(p + "\n")


def read_unique_peptides(peptides_tsv: Path, *, kmer: int) -> list[str]:
    """
    Read peptides of a given k from tabular fragmentation output peptides TSV.

    Expected columns: peptide_id, peptide, k, occurrence_count
    """
    df = pd.read_csv(peptides_tsv, sep="\t", dtype=str)
    for col in ("peptide", "k"):
        if col not in df.columns:
            raise ValueError(f"{peptides_tsv} missing required column '{col}' (cols={list(df.columns)})")

    df["k"] = pd.to_numeric(df["k"], errors="coerce")
    df = df[df["k"] == int(kmer)]
    peptides = (
        df["peptide"]
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .drop_duplicates()
        .tolist()
    )
    return peptides


def run_one(
    *,
    aav: str,
    kmer: int,
    alleles_path: Path,
    fragments_path: Path | None,
    peptides_tsv: Path | None,
    outdir: Path,
    netmhcpan_path: Path,
    extra: list[str],
    output_format: str,
) -> Path:
    # read peptides from new or legacy source
    if peptides_tsv is not None:
        if not peptides_tsv.exists():
            raise FileNotFoundError(f"peptides TSV not found: {peptides_tsv}")
        peptides = read_unique_peptides(peptides_tsv, kmer=int(kmer))
        if not peptides:
            raise ValueError(f"No peptides found for k={kmer} in {peptides_tsv}")
    else:
        # legacy fallback: fragments_path or inferred txt
        if fragments_path is None:
            fragments_path = REPO_ROOT / "data" / "output" / "fragmentation" / f"{aav}_{kmer}mer.txt"

        if not fragments_path.exists():
            raise FileNotFoundError(f"Fragments file not found: {fragments_path}")

        peptides = read_lines(fragments_path)
        if not peptides:
            raise ValueError(f"No peptides found in {fragments_path}")

    if not alleles_path.exists():
        raise FileNotFoundError(f"Alleles file not found: {alleles_path}")

    if not netmhcpan_path.exists():
        raise FileNotFoundError(f"netMHCpan wrapper not found: {netmhcpan_path}")

    alleles = read_lines(alleles_path)
    if not alleles:
        raise ValueError(f"No alleles found in {alleles_path}")

    allele_arg = ",".join(alleles)

    # output paths
    outdir.mkdir(parents=True, exist_ok=True)
    out_prefix = outdir / f"{aav}_{kmer}mer"
    tmp_peptides = out_prefix.with_suffix(".peptides.tmp.txt")

    if output_format == "xls":
        out_path = out_prefix.with_suffix(".netmhcpan.xls")
    elif output_format == "txt":
        out_path = out_prefix.with_suffix(".netmhcpan.txt")
    else:
        raise ValueError(f"Unsupported output_format: {output_format}")

    write_peptides_temp(peptides, tmp_peptides)

    cmd = [
        str(netmhcpan_path),
        "-p",
        "-a",
        allele_arg,
        "-f",
        str(tmp_peptides),
    ]

    if output_format == "xls":
        cmd += ["-xls", "-xlsfile", str(out_path)]

    cmd += extra

    env = os.environ.copy()
    env.setdefault("TMPDIR", "/tmp")

    with out_path.open("w", encoding="utf-8") as out_f:
        proc = subprocess.run(cmd, stdout=out_f, stderr=subprocess.PIPE, text=True, env=env)

    if proc.returncode != 0:
        raise RuntimeError(
            "netMHCpan failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Exit code: {proc.returncode}\n"
            f"Stderr:\n{proc.stderr}"
        )

    try:
        tmp_peptides.unlink()
    except OSError:
        pass

    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Run NetMHCpan on fragment peptides / peptides TSV.")
    ap.add_argument(
        "--aav",
        default=None,
        help="Output naming prefix (legacy assumed AAV9_VP1). If omitted, inferred from input.",
    )
    ap.add_argument(
        "--kmers",
        type=int,
        nargs="+",
        default=[11],
        help="One or more k-mer lengths (e.g. --kmers 8 9 10 11). Default: 11",
    )
    ap.add_argument("--alleles", required=True, type=Path, help="Path to allele list .txt (one allele per line)")

    # NEW: tabular fragmentation input (renamed to --peptides)
    ap.add_argument(
        "--peptides",
        type=Path,
        default=None,
        help="Path to unique_peptides.tsv from tabular fragmentation (columns: peptide_id, peptide, k, occurrence_count).",
    )

    # legacy optional explicit fragments file
    ap.add_argument(
        "--fragments",
        type=Path,
        default=None,
        help="Legacy: explicit fragments txt file path (single file). If provided, --kmers must be a single value.",
    )

    ap.add_argument(
        "--outdir",
        type=Path,
        default=REPO_ROOT / "data" / "output" / "netmhcpan",
        help="Output directory (default: data/output/netmhcpan)",
    )
    ap.add_argument(
        "--netmhcpan",
        type=Path,
        default=NETMHCPAN_WRAPPER,
        help="Path to netMHCpan wrapper script (default: tools/netMHCpan-4.2/netMHCpan)",
    )
    ap.add_argument(
        "--output-format",
        choices=["xls", "txt"],
        default="xls",
        help="Output format to write (default: xls).",
    )
    ap.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args forwarded to netMHCpan (put after --extra ...)",
    )

    args = ap.parse_args()

    if args.fragments is not None and len(args.kmers) != 1:
        raise ValueError("--fragments can only be used with a single kmer.")

    if args.peptides is not None and args.fragments is not None:
        raise ValueError("Use only one of --peptides or --fragments (not both).")

    # choose prefix name
    if args.aav is not None:
        prefix = args.aav
    elif args.peptides is not None:
        prefix = Path(args.peptides).parent.name
    else:
        raise ValueError("Provide --aav (legacy) or --peptides (new mode).")

    for k in args.kmers:
        out_path = run_one(
            aav=prefix,
            kmer=int(k),
            alleles_path=args.alleles,
            fragments_path=args.fragments,
            peptides_tsv=args.peptides,
            outdir=args.outdir,
            netmhcpan_path=args.netmhcpan,
            extra=args.extra,
            output_format=args.output_format,
        )
        print(f"Wrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
python -m src.netmhcpan \
  --peptides data/output/fragmentation/variants_vr5_9/unique_peptides.tsv \
  --kmers 9 \
  --alleles data/input/alleles/netmhcpan/allele_single.txt \
  --outdir data/output/netmhcpan/VR5_9mer
"""