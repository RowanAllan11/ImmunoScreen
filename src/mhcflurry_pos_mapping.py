from __future__ import annotations

import csv
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class PeptideCoord:
    peptide: str
    start_0: int
    end_0_exclusive: int
    protein_length: Optional[int] = None


def read_tsv_dicts(path: str) -> Iterable[dict]:
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row:
                yield row


def load_fragmentation_coords(path: str) -> Dict[str, PeptideCoord]:
    rows = list(read_tsv_dicts(path))
    if not rows:
        raise ValueError(f"No rows in fragmentation TSV: {path}")

    required = {"peptide", "start_0", "end_0_exclusive"}
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)} in fragmentation TSV: {path}")

    coords: Dict[str, PeptideCoord] = {}
    for r in rows:
        pep = r["peptide"]
        if pep in coords:
            continue
        coords[pep] = PeptideCoord(
            peptide=pep,
            start_0=int(float(r["start_0"])),
            end_0_exclusive=int(float(r["end_0_exclusive"])),
            protein_length=int(float(r["protein_length"])) if r.get("protein_length") else None,
        )
    return coords


def load_mhcflurry_rows(path: str) -> List[dict]:
    rows = list(read_tsv_dicts(path))
    if not rows:
        raise ValueError(f"No rows in mhcflurry TSV: {path}")

    required = {"peptide", "allele", "mhcflurry_presentation_percentile"}
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)} in mhcflurry TSV: {path}")

    return rows


def mhcflurry_row_to_fields(row: dict) -> Tuple[str, str, float]:
    pep = row["peptide"]
    allele = row["allele"]
    pct = float(row["mhcflurry_presentation_percentile"])
    return pep, allele, pct


def percentile_to_strength(pct: float) -> float:
    """
    Log-transform presentation percentile into an intuitive "higher=stronger" score.

    score = -log10(percentile / 100)

    Examples:
      pct=10  -> 1
      pct=1   -> 2
      pct=0.1 -> 3
    """
    p = max(float(pct), 1e-300) / 100.0
    return -math.log10(p)


def infer_protein_from_fragmentation_path(path: str) -> str:
    base = os.path.basename(path)
    m = re.match(r"^(?P<protein>.+?)_\d+mer\.tsv$", base)
    return m.group("protein") if m else base.replace(".tsv", "")


def infer_protein_from_mhcflurry_path(path: str) -> str:
    base = os.path.basename(path)
    m = re.match(r"^(?P<protein>.+?)_\d+mer\..+?\.mhcflurry\.tsv$", base)
    return m.group("protein") if m else base.replace(".mhcflurry.tsv", "")


def map_scores_to_positions_max(
    coords_by_peptide: Dict[str, PeptideCoord],
    mhcflurry_rows: List[dict],
) -> Dict[Tuple[str, int], float]:
    """
    Returns dict (allele, pos_0) -> max log score across overlapping peptides.
    """
    out: Dict[Tuple[str, int], float] = {}

    for row in mhcflurry_rows:
        pep, allele, pct = mhcflurry_row_to_fields(row)
        coord = coords_by_peptide.get(pep)
        if coord is None:
            continue

        score = percentile_to_strength(pct)

        for pos_0 in range(coord.start_0, coord.end_0_exclusive):
            k = (allele, pos_0)
            prev = out.get(k)
            if prev is None or score > prev:
                out[k] = score

    return out


def coords_to_sequence(coords_by_peptide: Dict[str, PeptideCoord]) -> str:
    protein_length = None
    for c in coords_by_peptide.values():
        if c.protein_length is not None:
            protein_length = c.protein_length
            break
    if protein_length is None:
        protein_length = max(c.end_0_exclusive for c in coords_by_peptide.values())

    seq = ["?"] * protein_length
    for pep, c in coords_by_peptide.items():
        for i, aa in enumerate(pep):
            pos = c.start_0 + i
            if 0 <= pos < protein_length:
                if seq[pos] == "?" or seq[pos] == aa:
                    seq[pos] = aa
    return "".join(seq)


def write_position_scores_tsv(
    out_path: str,
    protein: str,
    max_by_allele_pos: Dict[Tuple[str, int], float],
    sequence: Optional[str] = None,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["protein", "allele", "pos_0", "aa", "max_log_score"])
        for (allele, pos_0), score in sorted(max_by_allele_pos.items(), key=lambda x: (x[0][0], x[0][1])):
            aa = "."
            if sequence is not None and 0 <= pos_0 < len(sequence):
                aa = sequence[pos_0]
            w.writerow([protein, allele, pos_0, aa, f"{score:.6g}"])