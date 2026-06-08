from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NETMHC_PATH = REPO_ROOT / "tools" / "netMHCpan-4.2" / "netMHCpan"


# Reads alleles from a text file (one allele per line), uses for netMHCpan -a argument
def _read_alleles(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.strip().startswith("#")]


def read_unique_peptides(peptides_tsv: Path, *, kmer: int) -> List[str]:
    df = pd.read_csv(peptides_tsv, sep="\t", dtype=str)
    if "peptide" not in df.columns or "k" not in df.columns:
        raise ValueError(f"{peptides_tsv} must contain 'peptide' and 'k' columns")
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

# Writes a list of peptides to a temporary text file, one peptide per line, for netMHCpan input
def _write_temp_peptides(peptides: List[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for p in peptides:
            fh.write(p + "\n")


def run_netmhcpan_for_k(
    *,
    peptides_tsv: Path,
    k: int,
    alleles_path: Path,
    netmhcpan_path: Path,
    outdir: Path,
    output_format: str = "xls",
    extra: List[str] = (),
) -> Path:
    if not peptides_tsv.exists():
        raise FileNotFoundError(peptides_tsv)
    if not alleles_path.exists():
        raise FileNotFoundError(alleles_path)
    if not netmhcpan_path.exists():
        raise FileNotFoundError(netmhcpan_path)

    peptides = read_unique_peptides(peptides_tsv, kmer=int(k))
    if not peptides:
        raise ValueError(f"No peptides found for k={k}")

    alleles = _read_alleles(alleles_path)
    if not alleles:
        raise ValueError("No alleles found")

    allele_arg = ",".join(alleles)

    outdir.mkdir(parents=True, exist_ok=True)
    prefix = Path(peptides_tsv).parent.name
    out_prefix = outdir / f"{prefix}_{k}mer"
    tmp_peptides = out_prefix.with_suffix(".peptides.tmp.txt")

    if output_format == "xls":
        out_path = out_prefix.with_suffix(".netmhcpan.xls")
    elif output_format == "txt":
        out_path = out_prefix.with_suffix(".netmhcpan.txt")
    else:
        raise ValueError("output_format must be 'xls' or 'txt'")

    _write_temp_peptides(peptides, tmp_peptides)

    cmd = [str(netmhcpan_path), "-p", "-a", allele_arg, "-f", str(tmp_peptides)]
    if output_format == "xls":
        cmd += ["-xls", "-xlsfile", str(out_path)]
    cmd += list(extra or ())

    env = os.environ.copy()
    env.setdefault("TMPDIR", "/tmp")

    # write output (stdout) directly to out_path for txt mode; xls flag handles writing when requested
    if output_format == "txt":
        with out_path.open("w", encoding="utf-8") as out_f:
            proc = subprocess.run(cmd, stdout=out_f, stderr=subprocess.PIPE, text=True, env=env)
    else:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)

    if proc.returncode != 0:
        raise RuntimeError(f"netMHCpan failed (returncode={proc.returncode}):\n{proc.stderr}")

    try:
        tmp_peptides.unlink()
    except OSError:
        pass

    return out_path


def parse_netmhcpan_xls_wide_tsv(path: Path) -> pd.DataFrame:
    """
    Parse NetMHCpan -xls output (tab-delimited "wide" table) into tidy/long df with:
      peptide, k, allele, netMHCpan_EL_score, netMHCpan_EL_rank, (optional BA_score, BA_rank)
    """
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("Pos\tPeptide\tID\t"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"Could not find NetMHCpan xls header in: {path}")

    allele_line = lines[header_idx - 1]
    allele_cells = allele_line.split("\t")

    col_header = lines[header_idx].split("\t")
    base_cols = ["Pos", "Peptide", "ID"]
    if col_header[: len(base_cols)] != base_cols:
        raise ValueError(
            "Unexpected NetMHCpan xls columns. "
            f"Expected prefix {base_cols}, got {col_header[:len(base_cols)]}"
        )

    header_tail = col_header[len(base_cols) :]
    has_ba = "BA_score" in header_tail

    per_allele_fields = ["core", "icore", "EL_score", "EL_rank"]
    if has_ba:
        per_allele_fields += ["BA_score", "BA_rank"]
    block_w = len(per_allele_fields)

    allele_names: list[str] = []
    for c in allele_cells:
        c = c.strip()
        if not c:
            continue
        if "-" in c or c.startswith("HLA"):
            allele_names.append(c)
    if not allele_names:
        raise ValueError(f"Could not infer allele names from line: {allele_line}")

    records: list[dict] = []
    for line in lines[header_idx + 1 :]:
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < len(base_cols) + block_w:
            continue

        pep = parts[1]
        k = len(pep)

        tail = parts[len(base_cols) :]
        for ai, allele in enumerate(allele_names):
            start = ai * block_w
            end = start + block_w
            if end > len(tail):
                break

            blk = tail[start:end]
            el_score = pd.to_numeric(blk[2], errors="coerce")
            el_rank = pd.to_numeric(blk[3], errors="coerce")
            ba_score = pd.NA
            ba_rank = pd.NA
            if has_ba:
                ba_score = pd.to_numeric(blk[4], errors="coerce")
                ba_rank = pd.to_numeric(blk[5], errors="coerce")

            rec = {
                "peptide": pep,
                "k": int(k),
                "allele": allele,
                "netMHCpan_EL_score": float(el_score) if pd.notna(el_score) else pd.NA,
                "netMHCpan_EL_rank": float(el_rank) if pd.notna(el_rank) else pd.NA,
            }
            if has_ba:
                rec["netMHCpan_BA_score"] = float(ba_score) if pd.notna(ba_score) else pd.NA
                rec["netMHCpan_BA_rank"] = float(ba_rank) if pd.notna(ba_rank) else pd.NA

            records.append(rec)

    return pd.DataFrame.from_records(records)


def read_unique_peptides_tsv(path: Path) -> pd.DataFrame:
    """
    Read unique_peptides.tsv:
      peptide_id, peptide, k, occurrence_count
    """
    df = pd.read_csv(path, sep="\t", dtype=str)
    required = {"peptide_id", "peptide", "k", "occurrence_count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")

    df = df.copy()
    df["peptide"] = df["peptide"].astype(str)
    df["k"] = pd.to_numeric(df["k"], errors="coerce").astype("Int64")
    df["occurrence_count"] = pd.to_numeric(df["occurrence_count"], errors="coerce").astype("Int64")
    return df[["peptide_id", "peptide", "k", "occurrence_count"]]

def read_peptide_variant_map_tsv(path: Path) -> pd.DataFrame:
    """
    Read peptide_variant_map.tsv:
      peptide_id, variant_id, start, end, + metadata columns (e.g. criteria)
    """
    df = pd.read_csv(path, sep="\t", dtype=str)
    required = {"peptide_id", "variant_id", "start", "end"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")

    df = df.copy()
    df["start"] = pd.to_numeric(df["start"], errors="coerce").astype("Int64")
    df["end"] = pd.to_numeric(df["end"], errors="coerce").astype("Int64")
    return df

def infer_kmer_range_from_df(df: pd.DataFrame) -> str:
    mn = int(pd.to_numeric(df["k"], errors="coerce").min())
    mx = int(pd.to_numeric(df["k"], errors="coerce").max())
    return f"{mn}-{mx}mer" if mn != mx else f"{mn}mer"