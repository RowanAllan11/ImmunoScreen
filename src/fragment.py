from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple, Union


def read_fasta(path: Union[str, Path]) -> List[Tuple[str, str, str]]:
    """
    Minimal FASTA reader.

    Returns a list of tuples: (record_id, description, sequence)
      - record_id: first token after '>'
      - description: full header line (without leading '>')
      - sequence: concatenated sequence lines (whitespace removed)
    """
    path = Path(path)
    out: List[Tuple[str, str, str]] = []

    header: str | None = None
    seq_chunks: List[str] = []

    def flush() -> None:
        nonlocal header, seq_chunks
        if header is None:
            return
        desc = header
        rid = header.split()[0] if header.split() else header
        seq = "".join(seq_chunks).replace(" ", "").strip()
        out.append((rid, desc, seq))
        header = None
        seq_chunks = []

    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                header = line[1:].strip()
                seq_chunks = []
            else:
                seq_chunks.append(line)

    flush()
    return out


def fragment_sequence(seq: str, k: int) -> Iterable[Tuple[int, str]]:
    """Yield (start_0_based, peptide) for every contiguous k-mer in seq."""
    if k <= 0:
        raise ValueError("k must be a positive integer")
    seq = (seq or "").strip()
    n = len(seq)
    if n < k:
        return
    for start in range(0, n - k + 1):
        yield start, seq[start : start + k]


def sanitize_basename(name: str) -> str:
    # Keep filenames simple/portable
    cleaned = []
    for ch in name:
        if ch.isalnum() or ch in ("_", "-", "."):
            cleaned.append(ch)
        else:
            cleaned.append("_")
    s = "".join(cleaned).strip("_")
    return s or "sample"


def write_fragment_outputs(
    fasta_path: Union[str, Path],
    k: int,
    out_dir: Union[str, Path] = ".",
    out_prefix: str | None = None,
) -> Tuple[Path, Path]:
    """
    From input FASTA, write:
      - {prefix}_{k}mer.tsv : rich metadata master record
      - {prefix}_{k}mer.txt : plain peptide list (one peptide per line)

    Returns: (tsv_path, txt_path)
    """
    fasta_path = Path(fasta_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if out_prefix is None:
        out_prefix = sanitize_basename(fasta_path.stem)
    else:
        out_prefix = sanitize_basename(out_prefix)

    tsv_path = out_dir / f"{out_prefix}_{k}mer.tsv"
    txt_path = out_dir / f"{out_prefix}_{k}mer.txt"

    records = read_fasta(fasta_path)

    header = [
        "peptide_id",
        "peptide",
        "k",
        "record_id",
        "record_desc",
        "protein_length",
        "start_0",
        "end_0_exclusive",
        "start_1",
        "end_1_inclusive",
    ]

    with tsv_path.open("w", encoding="utf-8") as tsv, txt_path.open(
        "w", encoding="utf-8"
    ) as txt:
        tsv.write("\t".join(header) + "\n")

        for record_id, record_desc, seq in records:
            protein_len = len(seq)
            row_idx = 0
            for start_0, pep in fragment_sequence(seq, k):
                end_0_excl = start_0 + k
                start_1 = start_0 + 1
                end_1_incl = start_1 + k - 1

                peptide_id = f"{record_id}|k{k}|{start_1}-{end_1_incl}|{row_idx}"
                row_idx += 1

                tsv.write(
                    "\t".join(
                        [
                            peptide_id,
                            pep,
                            str(k),
                            record_id,
                            record_desc,
                            str(protein_len),
                            str(start_0),
                            str(end_0_excl),
                            str(start_1),
                            str(end_1_incl),
                        ]
                    )
                    + "\n"
                )
                txt.write(pep + "\n")

    return tsv_path, txt_path