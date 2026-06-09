from __future__ import annotations


import csv
import subprocess
from pathlib import Path

import pandas as pd

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

def run_mhcflurry_predict(
    mhcflurry_predict: str | Path,
    mhc_in: Path,
    out_path: Path,
    *,
    output_delimiter: str = "\t",
    no_flanking: bool = True,
    extra: list[str] | None = None,
) -> Path:
    """
    Run mhcflurry-predict with the given input CSV and write TSV results to out_path.
    Returns out_path on success.
    """
    cmd = [
        str(mhcflurry_predict),
        str(mhc_in),
        "--allele-column",
        "allele",
        "--peptide-column",
        "peptide",
        "--out",
        str(out_path),
        "--output-delimiter",
        output_delimiter,
    ]
    if no_flanking:
        cmd.append("--no-flanking")
    if extra:
        cmd += list(extra)

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return out_path

from src.netmhcpan import read_unique_peptides_tsv, read_peptide_variant_map_tsv


def filter_and_expand_mhcflurry_predictions(
    *,
    unique_peptides_tsv: Path,
    peptide_map_tsv: Path,
    mhcflurry_tsv: Path,
    affinity_percentile_threshold: float = 2.0,
    alleles_path: Path | None = None,
) -> pd.DataFrame:
    """
    Load mhcflurry TSV, filter by affinity_percentile_threshold, map to peptide_id and expand to variant rows.
    Returns final tidy dataframe ready to write.
    """
    uniq = read_unique_peptides_tsv(unique_peptides_tsv)
    pmap = read_peptide_variant_map_tsv(peptide_map_tsv)

    # load mhcflurry output
    mhc = pd.read_csv(mhcflurry_tsv, sep="\t", dtype=str)
    required = {
        "peptide",
        "allele",
        "mhcflurry_affinity_percentile",
        "mhcflurry_presentation_percentile",
    }
    missing = required - set(mhc.columns)
    if missing:
        raise ValueError(f"{mhcflurry_tsv} missing required columns: {sorted(missing)}")

    mhc["peptide"] = mhc["peptide"].astype(str)
    mhc["allele"] = mhc["allele"].astype(str)
    mhc["mhcflurry_affinity_percentile"] = pd.to_numeric(
        mhc["mhcflurry_affinity_percentile"], errors="coerce"
    )
    mhc["mhcflurry_presentation_percentile"] = pd.to_numeric(
        mhc["mhcflurry_presentation_percentile"], errors="coerce"
    )

    # add length/k
    mhc["length"] = mhc["peptide"].str.len().astype("Int64")
    mhc["k"] = mhc["length"]

    # optional allele subset
    if alleles_path is not None:
        allele_set: set[str] = set()
        with alleles_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                allele_set.add(s)
        if allele_set:
            mhc = mhc[mhc["allele"].isin(allele_set)].copy()

    # filter by affinity percentile
    mhc = mhc.dropna(subset=["mhcflurry_affinity_percentile"])
    mhc = mhc[mhc["mhcflurry_affinity_percentile"] < float(affinity_percentile_threshold)].copy()

    # join -> peptide_id via (peptide,k)
    uniq_key = uniq[["peptide_id", "peptide", "k", "occurrence_count"]].copy()
    merged = mhc.merge(uniq_key, on=["peptide", "k"], how="left", validate="many_to_one")
    n_missing_pid = int(merged["peptide_id"].isna().sum())
    if n_missing_pid:
        raise ValueError(
            f"{n_missing_pid} filtered MHCflurry rows could not be mapped to peptide_id using "
            f"{unique_peptides_tsv}. Ensure you predicted peptides from that exact unique_peptides.tsv."
        )

    # fan-out to variant occurrences
    out = merged.merge(pmap, on="peptide_id", how="left", validate="many_to_many")
    if "variant_id" not in out.columns:
        raise ValueError("peptide_variant_map.tsv did not provide 'variant_id' after join (unexpected).")
    n_missing_variant = int(out["variant_id"].isna().sum())
    if n_missing_variant:
        raise ValueError(
            f"{n_missing_variant} rows missing variant_id after join. Check peptide_variant_map.tsv matches unique_peptides.tsv."
        )

    # rename and normalise types
    out = out.rename(
        columns={
            "mhcflurry_affinity_percentile": "MHCflurry_affinity_percentile",
            "mhcflurry_presentation_percentile": "MHCflurry_presentation_percentile",
        }
    )
    out["start"] = out["start"].astype("Int64")
    out["end"] = out["end"].astype("Int64")
    out["k"] = pd.to_numeric(out["k"], errors="coerce").astype("Int64")
    out["length"] = pd.to_numeric(out["length"], errors="coerce").astype("Int64")

    # build final output columns
    cols = [
        "allele",
        "peptide_id",
        "peptide",
        "k",
        "variant_id",
        "start",
        "end",
        "length",
        "occurrence_count",
    ]
    extra_meta = [c for c in pmap.columns if c not in {"peptide_id", "variant_id", "start", "end"}]
    cols.extend(extra_meta)
    cols.extend(["MHCflurry_affinity_percentile", "MHCflurry_presentation_percentile"])

    for c in cols:
        if c not in out.columns:
            out[c] = pd.NA

    out = out[cols].sort_values(
        ["allele", "MHCflurry_affinity_percentile", "variant_id", "peptide_id"],
        ascending=[True, True, True, True],
        na_position="last",
    )

    return out
