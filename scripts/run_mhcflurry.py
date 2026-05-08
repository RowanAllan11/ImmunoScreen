from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _require_exe(name: str) -> str:
    exe = shutil.which(name)
    if not exe:
        raise RuntimeError(
            f"Required executable not found on PATH: {name}\n"
            "Activate your conda env on the HPC (the one with mhcflurry installed)."
        )
    return exe


def _discover_peptide_txts(fragment_dir: Path, kmers: Iterable[int]) -> List[Path]:
    files: List[Path] = []
    for k in kmers:
        files.extend(sorted(fragment_dir.glob(f"*_{k}mer.txt")))
    # de-dup while keeping order
    seen = set()
    out = []
    for p in files:
        rp = str(p.resolve()) 
        if rp not in seen:
            seen.add(rp)
            out.append(p)
    return out


def main() -> int:
    # Adjust these for your HPC run
    fragment_dir = REPO_ROOT / "data/output/fragmentation"
    out_dir = REPO_ROOT / "data/output/mhcflurry"
    out_dir.mkdir(parents=True, exist_ok=True)

    kmers = (8, 9, 10, 11, 13)

    # IMPORTANT: set your allele(s) here (space-separated list)
    # Example: ["HLA-A*02:01", "HLA-B*07:02"]
    alleles = ["H2-D*b", "H2-D*d"]

    mhcflurry_predict = _require_exe("mhcflurry-predict")

    peptide_files = _discover_peptide_txts(fragment_dir, kmers)
    if not peptide_files:
        raise FileNotFoundError(
            f"No peptide txt files found in {fragment_dir} for kmers {kmers}. "
            "Run scripts/run_fragmentation.py first."
        )

    for pep_path in peptide_files:
        # Example input: AAV9_VP1_15mer.txt -> output: AAV9_VP1_15mer.mhcflurry.tsv
        out_path = out_dir / f"{pep_path.stem}.mhcflurry.tsv"

        cmd = [
            mhcflurry_predict,
            "--peptides",
            str(pep_path),
            "--alleles",
            *alleles,
            "--out",
            str(out_path),
        ]

        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print(f"Wrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())