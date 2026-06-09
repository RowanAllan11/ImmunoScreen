from __future__ import annotations
from pathlib import Path
from typing import Optional
import re

import pandas as pd

def mutation_string(vr_seq: str, wt_vr_seq: str) -> str:
    """
    Return mutations in format S1A, where:
      S = wild-type residue
      1 = position within VR, 1-indexed
      A = variant residue
    If sequences equal -> "WT".
    """
    if vr_seq is None or wt_vr_seq is None:
        return pd.NA
    mutations = []
    for i, (wt, var) in enumerate(zip(wt_vr_seq, vr_seq), start=1):
        if wt != var:
            mutations.append(f"{wt}{i}{var}")
    return "WT" if not mutations else ";".join(mutations)


def peptide_mutation_string(
    mutation: str,
    peptide_start: int,
    peptide_end: int,
    *,
    var_start: int = 8,
) -> str:
    """
    Keep only VR mutations that overlap the peptide coordinates.

    mutation positions are VR-relative, e.g. S1L.
    peptide_start/end are full protein coordinates, 1-based inclusive.
    var_start is the full protein coordinate where the VR begins.
    """
    if pd.isna(mutation) or mutation == "WT":
        return "WT"

    kept = []

    for mut in str(mutation).split(";"):
        mut = mut.strip()
        match = re.match(r"^([A-Z*])(\d+)([A-Z*])$", mut)
        if not match:
            continue

        vr_pos = int(match.group(2))
        protein_pos = var_start + vr_pos - 1

        if peptide_start <= protein_pos <= peptide_end:
            kept.append(mut)

    return "WT" if not kept else ";".join(kept)


def _extract_vr_series(lib_df: pd.DataFrame, seq_col: str, var_start: int, var_end: int) -> pd.Series:
    # var_start/var_end are 1-based inclusive (matching CLI --var-start/--var-end)
    start0 = max(0, var_start - 1)
    # pandas .str.slice is end-exclusive so var_end is fine
    return lib_df[seq_col].astype(str).str.slice(start0, var_end)



def attach_mutation_labels(
    results_df: pd.DataFrame,
    library_csv: Path,
    *,
    variant_id_col: str = "variant_id",
    library_id_col: str = "Geneid",
    seq_col: str = "twist_seq_prot",
    var_start: int = 8,
    var_end: int = 24,
    wt_vr: Optional[str] = None,
) -> pd.DataFrame:
    """
    Return a copy of results_df with two new columns added:
      VR_sequence : substring of seq_col from library (var_start..var_end, 1-based inclusive)
      mutation    : string like 'S1A;T3G' or 'WT' if identical to wt_vr
    - If wt_vr is None it is inferred as the modal VR_sequence in the library.
    """
    lib_df = pd.read_csv(library_csv, sep=",", dtype=str)
    if library_id_col not in lib_df.columns:
        raise ValueError(f"{library_csv} missing id column: {library_id_col}")
    if seq_col not in lib_df.columns:
        raise ValueError(f"{library_csv} missing sequence column: {seq_col}")

    lib_df = lib_df.copy()
    lib_df["VR_sequence"] = _extract_vr_series(lib_df, seq_col, var_start, var_end)

    if wt_vr is None:
        modes = lib_df["VR_sequence"].mode()
        if modes.empty:
            raise ValueError("Could not infer WT VR sequence from library; pass wt_vr explicitly.")
        wt_vr_inferred = modes.iloc[0]
    else:
        wt_vr_inferred = wt_vr

    # build mapping
    lib_map = lib_df[[library_id_col, "VR_sequence"]].drop_duplicates().rename(columns={library_id_col: variant_id_col})
    lib_map["mutation"] = lib_map["VR_sequence"].apply(lambda s: mutation_string(s, wt_vr_inferred))

    # merge (left join so we keep all rows from results_df)
    out = results_df.merge(lib_map, on=variant_id_col, how="left", validate="many_to_one")

    out = out.rename(columns={"mutation": "VR_mutation"})

    out["peptide_mutation"] = out.apply(
        lambda row: peptide_mutation_string(
            row["VR_mutation"],
            int(row["start"]),
            int(row["end"]),
            var_start=var_start,
        ),
        axis=1,
    )
    
    return out
