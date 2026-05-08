from __future__ import annotations

import csv
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
        raise RuntimeError(f"Required executable not found on PATH: {name}")
    return exe


def _discover_fragment_tsvs(fragment_dir: Path, kmers: Iterable[int]) -> List[Path]:
    files: List[Path] = []
    for k in kmers:
        files.extend(sorted(fragment_dir.glob(f"*_{k}mer.tsv")))
    return files


def _write_peptide_allele_input(fragment_tsv: Path, allele: str, out_csv: Path) -> int:
    """
    Create a mhcflurry *CSV* input with columns: peptide, allele
    from a fragmentation TSV (must contain a 'peptide' column).
    Returns number of rows written.
    """
    n = 0
    with fragment_tsv.open("r", encoding="utf-8", newline="") as fin, out_csv.open(
        "w", encoding="utf-8", newline=""
    ) as fout:
        reader = csv.DictReader(fin, delimiter="\t")
        if not reader.fieldnames or "peptide" not in reader.fieldnames:
            raise ValueError(f"Input TSV missing 'peptide' column: {fragment_tsv}")

        writer = csv.DictWriter(
            fout, fieldnames=["peptide", "allele"], delimiter=",", lineterminator="\n"
        )
        writer.writeheader()

        for row in reader:
            pep = (row.get("peptide") or "").strip()
            if not pep:
                continue
            writer.writerow({"peptide": pep, "allele": allele})
            n += 1
    return n


def main() -> int:
    fragment_dir = REPO_ROOT / "data/output/fragmentation"
    out_dir = REPO_ROOT / "data/output/mhcflurry"
    out_dir.mkdir(parents=True, exist_ok=True)

    kmers = (8, 9, 10, 11)

    # NOTE: mhcflurry is mainly HLA; mouse H-2 alleles may not be supported.
    alleles = ["H2-D*b", "H2-D*d"]

    mhcflurry_predict = _require_exe("mhcflurry-predict")

    inputs = _discover_fragment_tsvs(fragment_dir, kmers)
    if not inputs:
        raise FileNotFoundError(
            f"No fragmentation TSVs found in {fragment_dir} for kmers {kmers}. "
            "Run scripts/run_fragmentation.py first."
        )

    for frag_tsv in inputs:
        for allele in alleles:
            mhc_in = out_dir / f"{frag_tsv.stem}.{allele}.input.csv"
            out_path = out_dir / f"{frag_tsv.stem}.{allele}.mhcflurry.tsv"

            n = _write_peptide_allele_input(frag_tsv, allele, mhc_in)
            if n == 0:
                print(f"Skipping empty input for {frag_tsv} allele={allele}")
                continue

            cmd = [
                mhcflurry_predict,
                str(mhc_in),  # positional input file
                "--allele-column",
                "allele",
                "--peptide-column",
                "peptide",
                "--out",
                str(out_path),
                "--output-delimiter",
                "\t",
                "--no-flanking",
            ]

            print(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            print(f"Wrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())