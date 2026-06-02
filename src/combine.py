#!/usr/bin/env python3
"""
Combine filtered NetMHCpan and MHCflurry outputs.
Was effective for FASTA input


Behavior:
- Merge on (peptide, allele) after allele normalization
- Keep ALL rows from either tool (OUTER JOIN)
- Tool-specific columns are NA when missing

Why allele normalization:
- Some outputs use e.g. "H-2-Db" while others use "H2-D*b"
- We normalize allele strings to a canonical form so matching works

Output:
  {outdir}/{ST}_combined_netmhcpan_mhcflurry.tsv

Includes a single 'length' column computed from peptide.

python src/combine.py \
  --st VR5_9mer \
  --netmhcpan-file data/output/netmhcpan_filtered/VR5_9mer_netMHCpan.tsv \
  --mhcflurry-file data/output/mhcflurry_filtered/vr5_9_MHCflurry.tsv \
    --outdir data/output/combined

"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional

import pandas as pd


def _pick_single_tsv(dirpath: Path, st: str) -> Path:
    if not dirpath.exists():
        raise FileNotFoundError(f"Directory not found: {dirpath}")

    tsvs = sorted(dirpath.glob("*.tsv"))
    if not tsvs:
        raise FileNotFoundError(f"No .tsv files found in: {dirpath}")

    st_lower = st.lower()
    matches = [p for p in tsvs if st_lower in p.name.lower()]
    if not matches:
        raise FileNotFoundError(f"No .tsv files in {dirpath} matching ST={st}")

    if len(matches) > 1:
        msg = "\n".join(str(p) for p in matches[:50])
        raise ValueError(
            f"Multiple TSVs match ST={st} in {dirpath}. Please pass an explicit file path.\n"
            f"Matches (up to 50):\n{msg}"
        )

    return matches[0]


def _load_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def _ensure_cols(df: pd.DataFrame, required: List[str], label: str, path: Path) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{label} input {path} missing required columns: {sorted(missing)}")


def _dedup_keep_first(df: pd.DataFrame, keys: List[str], label: str) -> pd.DataFrame:
    if df.empty:
        return df
    before = len(df)
    df2 = df.drop_duplicates(subset=keys, keep="first")
    after = len(df2)
    if after < before:
        print(f"Warning: {label} had {before - after} duplicate rows on {keys}; keeping the first per key.")
    return df2


_allele_re_mouse = re.compile(r"^(H)(?:-)?(2)(?:-)?([DKL])(?:\*?)([A-Za-z0-9]+)$")


def normalize_allele(a: str) -> str:
    """
    Normalize common mouse allele spellings to a canonical form:
      H-2-Db  -> H2-D*b
      H2-Db   -> H2-D*b
      H2-D*b  -> H2-D*b
      H-2-Kd  -> H2-K*d
    For other alleles (e.g. HLA-A*02:01), we return trimmed original.
    """
    if a is None:
        return a
    s = str(a).strip()
    if not s:
        return s

    # Fast path: already like H2-D*b / H2-K*d / etc
    # Keep as-is aside from whitespace.
    if s.startswith("H2-") and "*" in s:
        return s

    # Remove separators to match pattern
    compact = s.replace("_", "").replace(" ", "")
    compact = compact.replace("H-2-", "H2-").replace("H-2", "H2")
    compact = compact.replace("H2-", "H2")  # make it H2Db style for regex
    compact = compact.replace("*", "")

    m = _allele_re_mouse.match(compact)
    if m:
        h, two, locus, suffix = m.groups()
        return f"{h}{two}-{locus}*{suffix}"

    return s  # leave unknown formats unchanged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--st", required=True, help="ST tag (e.g., AAV9)")

    ap.add_argument(
        "--netmhcpan-dir",
        type=Path,
        default=Path("data/output/netmhcpan/filtered"),
        help="Directory containing filtered NetMHCpan TSV(s)",
    )
    ap.add_argument(
        "--mhcflurry-dir",
        type=Path,
        default=Path("data/output/mhcflurry/filtered"),
        help="Directory containing filtered MHCflurry TSV(s)",
    )
    ap.add_argument("--netmhcpan-file", type=Path, default=None)
    ap.add_argument("--mhcflurry-file", type=Path, default=None)

    ap.add_argument(
        "--dedup",
        action="store_true",
        help="Deduplicate within each input on (peptide, allele) before merging.",
    )
    ap.add_argument("-o", "--outdir", required=True, type=Path, help="Output directory")
    args = ap.parse_args()

    st = args.st

    net_path = args.netmhcpan_file or _pick_single_tsv(args.netmhcpan_dir, st=st)
    mhc_path = args.mhcflurry_file or _pick_single_tsv(args.mhcflurry_dir, st=st)

    net = _load_tsv(net_path)
    mhc = _load_tsv(mhc_path)

    on = ["peptide", "allele"]
    _ensure_cols(net, on, "NetMHCpan", net_path)
    _ensure_cols(mhc, on, "MHCflurry", mhc_path)

    # Normalize types and allele formatting
    net["peptide"] = net["peptide"].astype(str)
    mhc["peptide"] = mhc["peptide"].astype(str)

    net["allele"] = net["allele"].astype(str).map(normalize_allele)
    mhc["allele"] = mhc["allele"].astype(str).map(normalize_allele)

    # Ensure a single length column derived from peptide (consistent)
    net["length"] = net["peptide"].str.len().astype(int)
    mhc["length"] = mhc["peptide"].str.len().astype(int)

    # If either input already has start/end, keep them; otherwise they'll be NA after merge.
    # Avoid potential length collisions by dropping tool-specific length columns later.

    if args.dedup:
        net = _dedup_keep_first(net, keys=on, label="NetMHCpan")
        mhc = _dedup_keep_first(mhc, keys=on, label="MHCflurry")

    combined = net.merge(
        mhc,
        on=on,
        how="outer",
        validate="many_to_many",
        suffixes=("_netmhcpan", "_mhcflurry"),
    )

    # Unify length to a single column:
    # - if merge created length_netmhcpan/length_mhcflurry, prefer non-null, then fallback to peptide length
    if "length_netmhcpan" in combined.columns or "length_mhcflurry" in combined.columns:
        combined["length"] = combined.get("length_mhcflurry")
        if "length_netmhcpan" in combined.columns:
            combined["length"] = combined["length"].fillna(combined["length_netmhcpan"])
        # if still missing, compute
        combined["length"] = combined["length"].fillna(combined["peptide"].astype(str).str.len())

        # drop the tool-specific length columns
        drop_cols = [c for c in ["length_netmhcpan", "length_mhcflurry"] if c in combined.columns]
        combined = combined.drop(columns=drop_cols)
    else:
        combined["length"] = combined["length"].fillna(combined["peptide"].astype(str).str.len())

    # Prefer start/end from MHCflurry if present, else NetMHCpan (or vice versa)
    # If the merge created start_netmhcpan/start_mhcflurry, unify them.
    for col in ["start", "end"]:
        c1 = f"{col}_mhcflurry"
        c2 = f"{col}_netmhcpan"
        if c1 in combined.columns or c2 in combined.columns:
            combined[col] = combined.get(c1)
            if c2 in combined.columns:
                combined[col] = combined[col].fillna(combined[c2])
            drop = [c for c in [c1, c2] if c in combined.columns]
            combined = combined.drop(columns=drop)

    # Column order
    preferred: List[str] = []
    for c in ["allele", "peptide", "start", "end", "length"]:
        if c in combined.columns:
            preferred.append(c)
    remaining = [c for c in combined.columns if c not in preferred]
    combined = combined[preferred + remaining]

    # Sort
    combined = combined.sort_values(["allele", "peptide"], ascending=True, na_position="last")

    args.outdir.mkdir(parents=True, exist_ok=True)
    out_path = args.outdir / f"{st}_combined.tsv"
    combined.to_csv(out_path, sep="\t", index=False)

    print(f"NetMHCpan input: {net_path}")
    print(f"MHCflurry input: {mhc_path}")
    print(f"Wrote {len(combined)} combined rows to: {out_path}")


if __name__ == "__main__":
    main()