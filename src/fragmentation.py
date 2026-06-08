from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional, Sequence, Union, List, Tuple, Iterable

import pandas as pd

"""
Two types of input sources for fragmentation:
1. FASTA file e.g. AAV1_VP1.fasta
2. Tabular file of AAV variant libraries e.g. VR5_v3_final_library_detailed.tsv with columns: Geneid,twist_seq,twist_seq_prot,criteria,evo_score_label,evo_score
"""

def read_fasta(path: Union[str, Path]) -> List[Tuple[str, str, str]]:
    """
    Minimal FASTA reader.

    Returns a list of tuples: (record_id, description, sequence)
      - record_id: first token after '>'
      - description: full header line (without leading '>')
      - sequence: concatenated sequence lines (whitespace removed)
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    out: List[Tuple[str, str, str]] = []
    for entry in (e.strip() for e in text.split(">") if e.strip()):
        lines = entry.splitlines()
        header = lines[0].strip()
        rid = header.split(None, 1)[0] if header else header
        seq = "".join(l.strip().replace(" ", "") for l in lines[1:])
        out.append((rid, header, seq))
    return out

def fragment_sequence(seq: str, k: int) -> Iterator[Tuple[int, str]]:
    """
    Yield (start_index, peptide) sliding windows of length k with step 1.
    """
    if k <= 0:
        raise ValueError("k must be > 0")
    n = len(seq)
    # no fragments if sequence shorter than k
    for i in range(0, n - k + 1):
        yield i, seq[i : i + k]


@dataclass(frozen=True)
class FragmentationInput:
    sequence_id: str
    sequence: str
    metadata: Dict[str, str]


def iter_fasta_inputs(path: Union[str, Path]) -> Iterator[FragmentationInput]:
    """
    Stream FASTA records into FragmentationInput.
    """

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
    """
    path = Path(path)

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


def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _open_db(db_path: Path) -> sqlite3.Connection:
    _ensure_parent(db_path)
    con = sqlite3.connect(str(db_path))
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA temp_store=MEMORY;")
    return con


def _passes_var_filter(
    start1: int,
    end1: int,
    *,
    var_only: bool,
    var_start: int | None,
    var_end: int | None,
    var_mode: str,
) -> bool:
    if not var_only:
        return True
    if var_start is None or var_end is None:
        raise ValueError("var_only=True requires var_start and var_end.")
    if var_start > var_end:
        raise ValueError(f"Invalid variable region: var_start ({var_start}) > var_end ({var_end})")

    if var_mode == "overlap":
        # intervals intersect
        return not (end1 < var_start or start1 > var_end)
    if var_mode == "contained":
        # fully within variable region
        return start1 >= var_start and end1 <= var_end

    raise ValueError(f"Unknown var_mode: {var_mode} (expected 'overlap' or 'contained')")


def fragment_to_tables(
    inputs: Iterable[FragmentationInput],
    *,
    kmers: Sequence[int],
    out_dir: Path,
    metadata_cols: Sequence[str] = (),
    commit_every: int = 200_000,
    var_only: bool = False,
    var_start: int | None = None,
    var_end: int | None = None,
    var_mode: str = "overlap",
) -> Dict[str, Path]:
    """
    Write:
      - all_fragments.tsv
      - unique_peptides.tsv
      - peptide_variant_map.tsv

    Coordinates: 1-based inclusive start/end.

    Variable-region filter:
      - var_only=True keeps only peptides that overlap (default) or are contained within
        [var_start, var_end] (1-based inclusive).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_fragments_path = out_dir / "all_fragments.tsv"
    unique_peptides_path = out_dir / "unique_peptides.tsv"
    peptide_variant_map_path = out_dir / "peptide_variant_map.tsv"
    db_path = out_dir / ".fragment_dedup.sqlite"

    con = _open_db(db_path)
    cur = con.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS peptides (
            peptide_id TEXT PRIMARY KEY,
            peptide TEXT NOT NULL,
            k INTEGER NOT NULL,
            occurrence_count INTEGER NOT NULL,
            UNIQUE(peptide, k)
        )
        """
    )

    md_cols_sql = ", ".join([f"{c} TEXT" for c in metadata_cols])
    if md_cols_sql:
        md_cols_sql = ", " + md_cols_sql

    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS pep_map (
            peptide_id TEXT NOT NULL,
            variant_id TEXT NOT NULL,
            start INTEGER NOT NULL,
            end INTEGER NOT NULL
            {md_cols_sql}
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_pep_map_peptide_id ON pep_map(peptide_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_pep_map_variant_id ON pep_map(variant_id)")
    con.commit()

    pep_counter = 0

    def get_or_create_peptide_id(peptide: str, k: int) -> str:
        nonlocal pep_counter
        row = cur.execute(
            "SELECT peptide_id FROM peptides WHERE peptide = ? AND k = ?",
            (peptide, int(k)),
        ).fetchone()
        if row:
            cur.execute(
                "UPDATE peptides SET occurrence_count = occurrence_count + 1 WHERE peptide = ? AND k = ?",
                (peptide, int(k)),
            )
            return str(row[0])

        pep_counter += 1
        pid = f"pep_{pep_counter:06d}"
        cur.execute(
            "INSERT INTO peptides(peptide_id, peptide, k, occurrence_count) VALUES (?, ?, ?, 1)",
            (pid, peptide, int(k)),
        )
        return pid

    _ensure_parent(all_fragments_path)
    with all_fragments_path.open("w", encoding="utf-8") as af:
        af.write("\t".join(["variant_id", *metadata_cols, "k", "peptide", "start", "end"]) + "\n")

        n_rows = 0
        for inp in inputs:
            vid = inp.sequence_id
            seq = inp.sequence
            md = {c: inp.metadata.get(c, "") for c in metadata_cols}

            for k in kmers:
                k = int(k)
                for start0, pep in fragment_sequence(seq, k):
                    start1 = start0 + 1
                    end1 = start1 + k - 1

                    if not _passes_var_filter(
                        start1,
                        end1,
                        var_only=var_only,
                        var_start=var_start,
                        var_end=var_end,
                        var_mode=var_mode,
                    ):
                        continue

                    af.write(
                        "\t".join(
                            [vid]
                            + [md[c] for c in metadata_cols]
                            + [str(k), pep, str(start1), str(end1)]
                        )
                        + "\n"
                    )

                    pid = get_or_create_peptide_id(pep, k)

                    map_values: List[object] = [pid, vid, int(start1), int(end1)]
                    for c in metadata_cols:
                        map_values.append(md[c])

                    placeholders = ",".join(["?"] * len(map_values))
                    cur.execute(f"INSERT INTO pep_map VALUES ({placeholders})", map_values)

                    n_rows += 1
                    if n_rows % commit_every == 0:
                        con.commit()

        con.commit()

    _ensure_parent(unique_peptides_path)
    with unique_peptides_path.open("w", encoding="utf-8") as up:
        up.write("\t".join(["peptide_id", "peptide", "k", "occurrence_count"]) + "\n")
        for pid, pep, k, oc in con.execute(
            "SELECT peptide_id, peptide, k, occurrence_count FROM peptides ORDER BY peptide_id"
        ):
            up.write(f"{pid}\t{pep}\t{k}\t{oc}\n")

    _ensure_parent(peptide_variant_map_path)
    with peptide_variant_map_path.open("w", encoding="utf-8") as pm:
        pm.write("\t".join(["peptide_id", "variant_id", "start", "end", *metadata_cols]) + "\n")
        cols = ["peptide_id", "variant_id", "start", "end"] + list(metadata_cols)
        for row in con.execute(f"SELECT {', '.join(cols)} FROM pep_map ORDER BY peptide_id"):
            pm.write("\t".join([str(x) if x is not None else "" for x in row]) + "\n")

    con.close()

    return {
        "all_fragments": all_fragments_path,
        "unique_peptides": unique_peptides_path,
        "peptide_variant_map": peptide_variant_map_path,
    }