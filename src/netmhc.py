#!/usr/bin/env python3
"""
Parse NetMHCpan -xls/-BA "wide" TSV output (multi-allele blocks) and filter hits.

Input:  NetMHCpan .tsv produced with -xls -BA (like data/output/netmhcpan/*_NetMHCpan.tsv)
Output: Tidy TSV with one row per (peptide, allele) passing EL_rank threshold.

Columns:
- allele
- peptide
- length
- netMHCpan_EL_rank
- netMHCpan_BA_score
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


BASE_COLS = ["Pos", "Peptide", "ID"]


def _is_comment_or_blank(line: str) -> bool:
    s = line.strip()
    return (not s) or s.startswith("#") or s.startswith("//")


def _split_tsv(line: str) -> List[str]:
    return line.rstrip("\n").split("\t")


def parse_netmhcpan_xls_wide_tsv(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    header_idx: Optional[int] = None
    for i, line in enumerate(lines):
        if _is_comment_or_blank(line):
            continue
        if line.startswith("Pos\tPeptide\tID\t"):
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(
            f"Could not find header row starting with 'Pos\\tPeptide\\tID\\t' in {path}"
        )

    header = _split_tsv(lines[header_idx])

    base_end = len(BASE_COLS)
    trailing = header[base_end:]

    ave_nb = []
    if len(trailing) >= 2 and trailing[-2:] == ["Ave", "NB"]:
        ave_nb = ["Ave", "NB"]
        trailing = trailing[:-2]

    if len(trailing) % 6 != 0:
        raise ValueError(
            f"Unexpected number of non-base columns in header. "
            f"Expected multiples of 6 (plus optional Ave/NB), got {len(trailing)} for {path}"
        )

    n_alleles = len(trailing) // 6

    allele_names: List[str] = []
    if header_idx - 1 >= 0:
        allele_line = lines[header_idx - 1].rstrip("\n")
        parts = [p.strip() for p in allele_line.split("\t") if p.strip()]
        if len(parts) == n_alleles:
            allele_names = parts

    if not allele_names:
        allele_names = [f"allele_{i+1}" for i in range(n_alleles)]

    allele_block_cols = ["core", "icore", "EL_score", "EL_rank", "BA_score", "BA_rank"]

    records: List[Dict] = []
    for line in lines[header_idx + 1 :]:
        if _is_comment_or_blank(line):
            continue
        row = _split_tsv(line)
        if len(row) < base_end:
            continue

        expected_len = base_end + (n_alleles * 6) + (2 if ave_nb else 0)
        if len(row) < expected_len:
            row = row + [""] * (expected_len - len(row))

        pos = row[0]
        pep = row[1]

        if not str(pos).isdigit():
            continue

        length = len(pep)

        offset = base_end
        for ai, allele in enumerate(allele_names):
            block = row[offset + ai * 6 : offset + (ai + 1) * 6]
            data = dict(zip(allele_block_cols, block))

            try:
                el_rank = float(data["EL_rank"]) if data["EL_rank"] != "" else None
            except ValueError:
                el_rank = None

            try:
                ba_score = float(data["BA_score"]) if data["BA_score"] != "" else None
            except ValueError:
                ba_score = None

            records.append(
                {
                    "Pos": int(pos),
                    "peptide": pep,
                    "length": length,
                    "allele": allele,
                    "netMHCpan_EL_rank": el_rank,
                    "netMHCpan_BA_score": ba_score,
                }
            )

    return pd.DataFrame.from_records(records)


def infer_kmer_range_from_df(df: pd.DataFrame) -> str:
    mn = int(df["length"].min())
    mx = int(df["length"].max())
    return f"{mn}-{mx}mer" if mn != mx else f"{mn}mer"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, type=Path, help="NetMHCpan -xls/-BA TSV")
    ap.add_argument("-o", "--outdir", required=True, type=Path, help="Output directory")
    ap.add_argument("--st", required=True, help="ST tag to include in title (e.g., AAV9)")
    ap.add_argument("--el-rank-threshold", type=float, default=2.0, help="Keep rows with EL_rank < threshold")
    ap.add_argument("--dedup", action="store_true", help="Deduplicate identical (allele, peptide) keeping best EL_rank")
    args = ap.parse_args()

    df = parse_netmhcpan_xls_wide_tsv(args.input)

    df_filt = df.dropna(subset=["netMHCpan_EL_rank"])
    df_filt = df_filt[df_filt["netMHCpan_EL_rank"] < args.el_rank_threshold].copy()

    if args.dedup and not df_filt.empty:
        df_filt = (
            df_filt.sort_values(["allele", "peptide", "netMHCpan_EL_rank"], ascending=[True, True, True])
            .drop_duplicates(subset=["allele", "peptide"], keep="first")
        )

    df_filt = df_filt.sort_values(
        ["allele", "netMHCpan_EL_rank", "netMHCpan_BA_score", "peptide"],
        ascending=[True, True, False, True],
        na_position="last",
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    kmer_range = infer_kmer_range_from_df(df)

    # Title: ST + number of kmers (we use observed range)
    out_path = args.outdir / f"{args.st}_netMHCpan_{kmer_range}.tsv"

    out_cols = ["allele", "peptide", "length", "netMHCpan_EL_rank", "netMHCpan_BA_score"]
    df_filt[out_cols].to_csv(out_path, sep="\t", index=False)

    print(f"Wrote {len(df_filt)} filtered rows to: {out_path}")


if __name__ == "__main__":
    main()


"""
python src/netmhcpan_filt.py \
  -i data/output/netmhcpan/2187821_NetMHCpan.tsv \
  -o data/output/netmhcpan/filtered \
  --st AAV9
"""