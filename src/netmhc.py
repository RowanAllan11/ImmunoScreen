from __future__ import annotations

import csv
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional


def require_file(path: Path, what: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{what} not found: {path}")
    return path


def discover_fragment_tsvs(fragment_dir: Path, kmers: Iterable[int]) -> List[Path]:
    files: List[Path] = []
    for k in kmers:
        files.extend(sorted(fragment_dir.glob(f"*_{k}mer.tsv")))
    return files

# query
def write_peptides_file(fragment_tsv: Path, out_pep: Path) -> int:
    """
    NetMHCpan can take peptides via '-p <file>' where file is typically one peptide per line.
    Input is our fragmentation TSV that must contain a 'peptide' column.
    Returns number of peptides written.
    """
    n = 0
    with fragment_tsv.open("r", encoding="utf-8", newline="") as fin, out_pep.open(
        "w", encoding="utf-8", newline="\n"
    ) as fout:
        reader = csv.DictReader(fin, delimiter="\t")
        if not reader.fieldnames or "peptide" not in reader.fieldnames:
            raise ValueError(f"Input TSV missing 'peptide' column: {fragment_tsv}")

        for row in reader:
            pep = (row.get("peptide") or "").strip()
            if not pep:
                continue
            fout.write(pep + "\n")
            n += 1
    return n


def resolve_netmhcpan_exe(netmhcpan_dir: Path) -> str:
    """
    Prefer calling the wrapper script inside the distribution (tcsh script),
    otherwise fall back to PATH resolution.
    """
    wrapper = netmhcpan_dir / "netMHCpan"
    if wrapper.exists():
        return str(wrapper)

    exe = shutil.which("netMHCpan")
    if exe:
        return exe

    raise RuntimeError(
        "Could not find netMHCpan executable. "
        f"Tried: {wrapper} and PATH('netMHCpan')."
    )


def ensure_netmhcpan_configured(netmhcpan_dir: Path, tmpdir: Optional[Path] = None) -> None:
    """
    NetMHCpan's wrapper script expects NMHOME to be customized (per readme).
    We'll not edit it automatically; we just sanity-check it and give a clear error.

    You can also pass TMPDIR via environment at runtime.
    """
    wrapper = netmhcpan_dir / "netMHCpan"
    require_file(wrapper, "netMHCpan wrapper script")

    # Ensure tmpdir exists if provided
    if tmpdir is not None:
        tmpdir.mkdir(parents=True, exist_ok=True)


def run_netmhcpan(
    *,
    netmhcpan_exe: str,
    peptides_file: Path,
    allele: str,
    out_path: Path,
    netmhcpan_rdir: Optional[Path] = None,
    tmpdir: Optional[Path] = None,
    extra_args: Optional[List[str]] = None,
) -> None:
    """
    Run NetMHCpan and capture stdout to a file.
    Minimal args: '-p <file>' (peptides) and '-a <allele>' (allele).
    We add '-BA' (binding affinity) to get IC50-style output.

    If netmhcpan_rdir is set, we add '-rdir <dir>' which points to install root.
    """
    cmd: List[str] = [netmhcpan_exe]

    if netmhcpan_rdir is not None:
        cmd += ["-rdir", str(netmhcpan_rdir)]

    cmd += ["-p", str(peptides_file), "-a", allele, "-BA"]

    if extra_args:
        cmd += list(extra_args)

    env = os.environ.copy()
    if tmpdir is not None:
        env["TMPDIR"] = str(tmpdir)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # NetMHCpan prints results to stdout, so redirect to file
    with out_path.open("w", encoding="utf-8", newline="\n") as fout:
        subprocess.run(cmd, check=True, stdout=fout, stderr=subprocess.PIPE, text=True, env=env)
        # Note: stderr is captured; if you want it printed, remove stderr=PIPE.