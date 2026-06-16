#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


MERGE_KEYS = ["allele", "peptide", "variant_id", "start", "end", "k"]

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


def _prepare_for_merge(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["allele"] = df["allele"].astype(str).map(normalize_allele)

    for c in ["start", "end", "k"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64").astype(str)

    return df


def combine_predictions(netmhcpan_file: Path, mhcflurry_file: Path) -> pd.DataFrame:
    net = _load_tsv(netmhcpan_file)
    mhc = _load_tsv(mhcflurry_file)

    _assert_has_cols(net, MERGE_KEYS, netmhcpan_file)
    _assert_has_cols(mhc, MERGE_KEYS, mhcflurry_file)

    net = _prepare_for_merge(net)
    mhc = _prepare_for_merge(mhc)

    _assert_unique_on_keys(net, MERGE_KEYS, "NetMHCpan")
    _assert_unique_on_keys(mhc, MERGE_KEYS, "MHCflurry")

    combined = net.merge(
        mhc,
        on=MERGE_KEYS,
        how="outer",
        validate="one_to_one",
        suffixes=("_netmhcpan", "_mhcflurry"),
        indicator=True
    )

    print("Merge result:")
    print(combined["_merge"].value_counts())
    combined = combined.drop(columns="_merge")

    for col in ["peptide_id", "length", "occurrence_count", "criteria"]:
        net_col = f"{col}_netmhcpan"
        mhc_col = f"{col}_mhcflurry"

        if net_col in combined.columns and mhc_col in combined.columns:
            combined[col] = combined[net_col].where(
                combined[net_col].notna(),
                combined[mhc_col],
            )
            combined = combined.drop(columns=[net_col, mhc_col])

    front = MERGE_KEYS + ["peptide_id", "length", "occurrence_count", "criteria"]
    front = [c for c in front if c in combined.columns]
    rest = [c for c in combined.columns if c not in front]

    return combined[front + rest]