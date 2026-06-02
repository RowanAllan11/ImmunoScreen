from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _read_alleles_file(path: Path) -> list[str]:
    alleles: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            alleles.append(s)
    return alleles


def _require_exe(name: str) -> str:
    exe = shutil.which(name)
    if not exe:
        raise RuntimeError(f"Required executable not found on PATH: {name}")
    return exe


def _write_unique_peptides_allele_input(unique_peptides_tsv: Path, alleles: list[str], out_csv: Path) -> int:
    """
    Create mhcflurry input CSV (peptide, allele) from unique_peptides.tsv.

    Expected columns:
      peptide_id, peptide, k, occurrence_count

    Returns number of (peptide, allele) rows written.
    """
    if not alleles:
        return 0

    df = pd.read_csv(unique_peptides_tsv, sep="\t", dtype=str)
    if "peptide" not in df.columns:
        raise ValueError(f"unique_peptides.tsv missing 'peptide' column: {unique_peptides_tsv}")

    peptides = df["peptide"].astype(str).str.strip()
    peptides = peptides[peptides != ""].drop_duplicates().tolist()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_csv.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=["peptide", "allele"], delimiter=",", lineterminator="\n")
        writer.writeheader()
        for pep in peptides:
            for allele in alleles:
                writer.writerow({"peptide": pep, "allele": allele})
                n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description="Run mhcflurry-predict on fragmented peptides.")
    ap.add_argument(
        "--unique-peptides",
        type=Path,
        required=True,
        help="Path to unique_peptides.tsv from scripts/run_fragmentation.py --tabular",
    )
    ap.add_argument("--alleles", type=Path, required=True, help="Alleles .txt (one allele per line)")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "data/output/mhcflurry",
        help="Output directory (default: data/output/mhcflurry)",
    )
    ap.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Output tag/prefix for filenames (default: parent directory name of unique-peptides)",
    )
    args = ap.parse_args()

    unique_peptides_tsv = args.unique_peptides
    if not unique_peptides_tsv.exists():
        raise FileNotFoundError(f"unique_peptides.tsv not found: {unique_peptides_tsv}")

    alleles = _read_alleles_file(args.alleles)
    mhcflurry_predict = _require_exe("mhcflurry-predict")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = args.tag or unique_peptides_tsv.parent.name
    mhc_in = out_dir / f"{tag}.unique_peptides.input.csv"
    out_path = out_dir / f"{tag}.unique_peptides.mhcflurry.tsv"

    n = _write_unique_peptides_allele_input(unique_peptides_tsv, alleles, mhc_in)
    if n == 0:
        raise RuntimeError(f"0 (peptide, allele) rows written from: {unique_peptides_tsv}")

    cmd = [
        mhcflurry_predict,
        str(mhc_in),
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

"""
python scripts/run_mhcflurry.py \
  --unique-peptides data/output/fragmentation/variants_vr5_9/unique_peptides.tsv \
  --alleles data/input/alleles/mhcflurry/alleles_h2.txt \
  --out-dir data/output/mhcflurry/VR5_9mer \
  --tag VR5_v3_9mer
"""