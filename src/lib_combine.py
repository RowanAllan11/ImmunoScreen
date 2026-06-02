#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

"""
python src/lib_combine.py \
  --st VR5_9mer \
  --netmhcpan-file data/output/netmhcpan_filtered/VR5_9mer_netMHCpan.tsv \
  --mhcflurry-file data/output/mhcflurry_filtered/vr5_9_MHCflurry.tsv \
  --outdir data/output/combined

"""
def normalize_allele(a: str) -> str:
    """
    Normalize common mouse allele spellings.
    Example: "H-2-Db" -> "H2-D*b"
    Leave unknown formats unchanged.
    """
    s = (a or "").strip()
    if not s:
        return s
    # netMHCpan often: H-2-Db / H-2-Kb
    if s.startswith("H-2-") and len(s) >= 6:
        # H-2-Db -> H2-D*b
        rest = s[4:]  # "Db"
        if len(rest) >= 2:
            return f"H2-{rest[0]}*{rest[1:]}"
    return s


MERGE_KEYS = ["allele", "peptide_id", "variant_id", "start", "end", "k"]


def _load_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep="\t", dtype=str)


def _assert_has_cols(df: pd.DataFrame, cols: list[str], path: Path) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")


def _assert_unique_on_keys(df: pd.DataFrame, keys: list[str], label: str) -> None:
    dups = df.duplicated(keys).sum()
    if int(dups) != 0:
        examples = df[df.duplicated(keys, keep=False)].head(10)[keys]
        raise ValueError(
            f"{label} has {int(dups)} duplicate rows on merge keys {keys}. "
            f"This would create merge explosion.\nExamples:\n{examples.to_string(index=False)}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="Combine filtered NetMHCpan + MHCflurry outputs (new tabular format).")
    ap.add_argument("--st", required=True, help="Tag used for output filename")
    ap.add_argument("--netmhcpan-file", type=Path, required=True, help="Filtered netMHCpan TSV")
    ap.add_argument("--mhcflurry-file", type=Path, required=True, help="Filtered MHCflurry TSV")
    ap.add_argument("--outdir", type=Path, required=True, help="Output directory")

    args = ap.parse_args()

    net_path = args.netmhcpan_file
    mhc_path = args.mhcflurry_file

    net = _load_tsv(net_path)
    mhc = _load_tsv(mhc_path)

    _assert_has_cols(net, MERGE_KEYS, net_path)
    _assert_has_cols(mhc, MERGE_KEYS, mhc_path)

    # normalize allele + numeric-ish join cols to consistent string form
    net = net.copy()
    mhc = mhc.copy()

    net["allele"] = net["allele"].astype(str).map(normalize_allele)
    mhc["allele"] = mhc["allele"].astype(str).map(normalize_allele)

    for c in ["start", "end", "k"]:
        net[c] = pd.to_numeric(net[c], errors="coerce").astype("Int64").astype(str)
        mhc[c] = pd.to_numeric(mhc[c], errors="coerce").astype("Int64").astype(str)

    # guardrails: ensure 1:1 merge
    _assert_unique_on_keys(net, MERGE_KEYS, "NetMHCpan")
    _assert_unique_on_keys(mhc, MERGE_KEYS, "MHCflurry")

    combined = net.merge(
        mhc,
        on=MERGE_KEYS,
        how="outer",
        validate="one_to_one",
        suffixes=("_netmhcpan", "_mhcflurry"),
    )

    # prefer single copies of common columns (peptide, length, occurrence_count, criteria)
    # if both present, keep netmhcpan version unless missing.
    for col in ["peptide", "length", "occurrence_count", "criteria"]:
        net_col = f"{col}_netmhcpan"
        mhc_col = f"{col}_mhcflurry"
        if net_col in combined.columns and mhc_col in combined.columns:
            combined[col] = combined[net_col].where(combined[net_col].notna(), combined[mhc_col])
            combined = combined.drop(columns=[net_col, mhc_col])
        # if only one exists, leave as-is

    # nice column order
    front = MERGE_KEYS + ["peptide", "length", "occurrence_count", "criteria"]
    front = [c for c in front if c in combined.columns]
    rest = [c for c in combined.columns if c not in front]
    combined = combined[front + rest]

    args.outdir.mkdir(parents=True, exist_ok=True)
    out_path = args.outdir / f"{args.st}_combined_netmhcpan_mhcflurry.tsv"
    combined.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote: {out_path} ({len(combined)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())