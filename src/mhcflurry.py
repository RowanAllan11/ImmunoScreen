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
    alleles: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load MHCflurry predictions, add threshold-pass flags, map predictions
    to peptide IDs, and expand them to variant-level rows.

    Parameters
    ----------
    unique_peptides_tsv
        Deduplicated peptide table produced during fragmentation.
    peptide_map_tsv
        Mapping between peptide IDs and variant occurrences.
    mhcflurry_tsv
        Raw MHCflurry prediction output.
    affinity_percentile_threshold
        Maximum affinity percentile considered passing.
    alleles
        Optional allele subset. Usually the same list used to generate
        the MHCflurry input table.
    """
    uniq = read_unique_peptides_tsv(unique_peptides_tsv)
    pmap = read_peptide_variant_map_tsv(peptide_map_tsv)

    mhc = pd.read_csv(mhcflurry_tsv, sep="\t", dtype=str)

    required = {
        "peptide",
        "allele",
        "mhcflurry_affinity_percentile",
        "mhcflurry_presentation_percentile",
    }
    missing = required - set(mhc.columns)

    if missing:
        raise ValueError(
            f"{mhcflurry_tsv} missing required columns: "
            f"{sorted(missing)}"
        )

    mhc["peptide"] = mhc["peptide"].astype(str).str.strip()
    mhc["allele"] = mhc["allele"].astype(str).str.strip()

    mhc["mhcflurry_affinity_percentile"] = pd.to_numeric(
        mhc["mhcflurry_affinity_percentile"],
        errors="coerce",
    )
    mhc["mhcflurry_presentation_percentile"] = pd.to_numeric(
        mhc["mhcflurry_presentation_percentile"],
        errors="coerce",
    )

    mhc["length"] = mhc["peptide"].str.len().astype("Int64")
    mhc["k"] = mhc["length"]

    # Optional validation/subsetting against the configured allele list.
    if alleles is not None:
        allele_set = {
            str(allele).strip()
            for allele in alleles
            if str(allele).strip()
        }

        if not allele_set:
            raise ValueError(
                "The supplied allele list contains no valid allele names."
            )

        unexpected_alleles = set(mhc["allele"].dropna()) - allele_set
        if unexpected_alleles:
            raise ValueError(
                "MHCflurry output contains alleles that were not requested: "
                f"{sorted(unexpected_alleles)}"
            )

        mhc = mhc[mhc["allele"].isin(allele_set)].copy()

    mhc["MHCflurry_affinity_percentile_pass"] = (
        mhc["mhcflurry_affinity_percentile"].notna()
        & (
            mhc["mhcflurry_affinity_percentile"]
            < float(affinity_percentile_threshold)
        )
    )

    uniq_key = uniq[
        ["peptide_id", "peptide", "k", "occurrence_count"]
    ].copy()

    merged = mhc.merge(
        uniq_key,
        on=["peptide", "k"],
        how="left",
        validate="many_to_one",
    )

    n_missing_pid = int(merged["peptide_id"].isna().sum())
    if n_missing_pid:
        raise ValueError(
            f"{n_missing_pid} MHCflurry rows could not be mapped to a "
            f"peptide_id using {unique_peptides_tsv}. Ensure MHCflurry "
            "was run on that exact unique-peptide table."
        )

    out = merged.merge(
        pmap,
        on="peptide_id",
        how="left",
        validate="many_to_many",
    )

    if "variant_id" not in out.columns:
        raise ValueError(
            "peptide_variant_map.tsv did not provide 'variant_id' "
            "after joining."
        )

    n_missing_variant = int(out["variant_id"].isna().sum())
    if n_missing_variant:
        raise ValueError(
            f"{n_missing_variant} rows are missing variant_id after "
            "joining. Check that peptide_variant_map.tsv matches "
            "unique_peptides.tsv."
        )

    out = out.rename(
        columns={
            "mhcflurry_affinity_percentile":
                "MHCflurry_affinity_percentile",
            "mhcflurry_presentation_percentile":
                "MHCflurry_presentation_percentile",
        }
    )

    out["start"] = pd.to_numeric(
        out["start"],
        errors="coerce",
    ).astype("Int64")

    out["end"] = pd.to_numeric(
        out["end"],
        errors="coerce",
    ).astype("Int64")

    out["k"] = pd.to_numeric(
        out["k"],
        errors="coerce",
    ).astype("Int64")

    out["length"] = pd.to_numeric(
        out["length"],
        errors="coerce",
    ).astype("Int64")

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

    map_key_columns = {
        "peptide_id",
        "variant_id",
        "start",
        "end",
    }

    extra_meta = [
        column
        for column in pmap.columns
        if column not in map_key_columns
        and column not in cols
    ]

    cols.extend(extra_meta)

    cols.extend(
        [
            "MHCflurry_affinity_percentile",
            "MHCflurry_presentation_percentile",
            "MHCflurry_affinity_percentile_pass",
        ]
    )

    for column in cols:
        if column not in out.columns:
            out[column] = pd.NA

    out = out[cols].sort_values(
        [
            "allele",
            "MHCflurry_affinity_percentile",
            "variant_id",
            "peptide_id",
        ],
        ascending=[True, True, True, True],
        na_position="last",
    )

    return out