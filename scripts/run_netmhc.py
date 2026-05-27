from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.netmhc import (  # noqa: E402
    discover_fragment_tsvs,
    ensure_netmhcpan_configured,
    resolve_netmhcpan_exe,
    run_netmhcpan,
    write_peptides_file,
)


def main() -> int:
    fragment_dir = REPO_ROOT / "data/output/fragmentation"
    out_dir = REPO_ROOT / "data/output/netmhcpan"
    out_dir.mkdir(parents=True, exist_ok=True)

    # NetMHCpan install sits next to pipeline directory in your workspace layout
    # final_project/
    #   aav-tcr-epitope-pipeline/
    #   netMHCpan-4.2/
    netmhcpan_dir = REPO_ROOT.parents[0] / "netMHCpan-4.2"
    tmpdir = netmhcpan_dir / "tmp"

    kmers = (8, 9, 10, 11, 12, 13, 14)

    # NetMHCpan allele format is typically like HLA-A01:01 (no '*')
    alleles = [
        "H-2-Db",
        "H-2-Kb",
        "H-2-Dd",
        "H-2-Dq",
        "H-2-Kd",
        "H-2-Kk",
        "H-2-Ld",
        "H-2-Lq",
        "H-2-Kq",
        "HLA-A01:01",
        "HLA-A02:01",
        "HLA-A03:01",
        "HLA-A11:01",
        "HLA-B07:02",
        "HLA-B08:01",
        "HLA-C07:01",
        "HLA-C07:02",
    ]

    ensure_netmhcpan_configured(netmhcpan_dir, tmpdir=tmpdir)
    netmhcpan_exe = resolve_netmhcpan_exe(netmhcpan_dir)

    inputs = discover_fragment_tsvs(fragment_dir, kmers)
    if not inputs:
        raise FileNotFoundError(
            f"No fragmentation TSVs found in {fragment_dir} for kmers {kmers}. "
            "Run scripts/run_fragmentation.py first."
        )

    for frag_tsv in inputs:
        pep_file = out_dir / f"{frag_tsv.stem}.peptides.txt"
        n = write_peptides_file(frag_tsv, pep_file)
        if n == 0:
            print(f"Skipping empty input for {frag_tsv}")
            continue

        for allele in alleles:
            out_path = out_dir / f"{frag_tsv.stem}.{allele}.netmhcpan.txt"

            print(f"Running NetMHCpan: {frag_tsv.name} allele={allele} peptides={n}")
            run_netmhcpan(
                netmhcpan_exe=netmhcpan_exe,
                peptides_file=pep_file,
                allele=allele,
                out_path=out_path,
                netmhcpan_rdir=netmhcpan_dir,  # helps avoid NMHOME issues
                tmpdir=tmpdir,
                extra_args=["-s"],  # optional: sort output (see netMHCpan.1)
            )
            print(f"Wrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())