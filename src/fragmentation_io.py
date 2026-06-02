from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional, Sequence, Union

import pandas as pd


@dataclass(frozen=True)
class FragmentationInput:
    sequence_id: str
    sequence: str
    metadata: Dict[str, str]


def _detect_sep(path: Path) -> str:
    suf = path.suffix.lower()
    if suf == ".tsv":
        return "\t"
    if suf == ".csv":
        return ","
    return "\t"


def iter_fasta_inputs(path: Union[str, Path]) -> Iterator[FragmentationInput]:
    """
    Stream FASTA records into FragmentationInput.
    Uses existing read_fasta from src.fragment to preserve behaviour.
    """
    from src.fragment import read_fasta  # local import to avoid cycles

    path = Path(path)
    for record_id, record_desc, seq in read_fasta(path):
        yield FragmentationInput(
            sequence_id=str(record_id),
            sequence=str(seq),
            metadata={"record_desc": str(record_desc)},
        )


def iter_table_inputs(
    path: Union[str, Path],
    *,
    id_col: str,
    sequence_col: str,
    metadata_cols: Optional[Sequence[str]] = None,
    chunksize: int = 50_000,
    sep: Optional[str] = None,
) -> Iterator[FragmentationInput]:
    """
    Stream TSV/CSV rows into FragmentationInput using pandas chunked read.
    Designed for 40k+ variants.
    """
    path = Path(path)
    if sep is None:
        sep = _detect_sep(path)

    usecols = [id_col, sequence_col]
    if metadata_cols:
        for c in metadata_cols:
            if c not in usecols:
                usecols.append(c)

    for chunk in pd.read_csv(path, sep=sep, dtype=str, usecols=usecols, chunksize=chunksize):
        chunk = chunk.fillna("")
        for _, row in chunk.iterrows():
            sid = str(row[id_col]).strip()
            seq = str(row[sequence_col]).strip().replace(" ", "")
            if not sid or not seq:
                continue
            md: Dict[str, str] = {}
            if metadata_cols:
                for c in metadata_cols:
                    md[c] = str(row.get(c, "")).strip()
            yield FragmentationInput(sequence_id=sid, sequence=seq, metadata=md)