from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd


AA_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")
AA_SET = set(AA_ALPHABET)


@dataclass(frozen=True)
class MSA:
    """
    Multiple sequence alignment representation.

    Attributes:
        names: list of sequence identifiers.
        seqs: list of aligned sequences (same length), uppercase, may contain '-'.
    """

    names: List[str]
    seqs: List[str]

    @property
    def n_seqs(self) -> int:
        return len(self.seqs)

    @property
    def aln_len(self) -> int:
        return len(self.seqs[0]) if self.seqs else 0


def read_first_fasta_record(path: Path) -> Tuple[str, str]:
    """
    Read the first FASTA record from a file.

    Returns:
        (record_id, sequence)
    """
    recs = read_fasta(path)
    if not recs:
        raise ValueError(f"No FASTA records found in {path}")
    # `read_fasta` returns a dict {name: seq}; take the first item
    name = next(iter(recs.keys()))
    seq = recs[name]
    return name, seq


def merge_fastas_to_multifasta(
    input_dir: Path,
    out_fasta: Path,
    *,
    pattern: str = "*.fasta",
    name_from: str = "stem",
    include: Optional[Sequence[str]] = None,
    exclude: Optional[Sequence[str]] = None,
) -> Path:
    """
    Merge many FASTA files in a directory into a single multi-FASTA.

    Designed for current layout: one FASTA per serotype 

    Record naming:
      - name_from="stem": use filename stem as record id (e.g. 'AAV9_VP1' from 'AAV9_VP1.fasta')
      - name_from="header": use the first FASTA header token (up to first whitespace)

    Args:
        input_dir: directory containing FASTA files.
        out_fasta: output multi-FASTA path.
        pattern: glob pattern for selecting FASTA files.
        name_from: "stem" or "header".
        include: optional list of allowed stems (only applies to stem filtering).
        exclude: optional list of stems to skip.

    Returns:
        out_fasta
    """
    fasta_files = sorted(input_dir.glob(pattern))
    if include is not None:
        inc = set(include)
        fasta_files = [p for p in fasta_files if p.stem in inc]
    if exclude is not None:
        exc = set(exclude)
        fasta_files = [p for p in fasta_files if p.stem not in exc]

    if not fasta_files:
        raise FileNotFoundError(f"No FASTA files found in {input_dir} matching pattern '{pattern}'")

    records: Dict[str, str] = {}
    for fp in fasta_files:
        if name_from == "stem":
            rec_name = fp.stem
            _hdr, seq = read_first_fasta_record(fp)
        elif name_from == "header":
            rec_name, seq = read_first_fasta_record(fp)
        else:
            raise ValueError("name_from must be 'stem' or 'header'")

        if rec_name in records:
            raise ValueError(f"Duplicate record name '{rec_name}' from file: {fp}")
        records[rec_name] = seq

    write_fasta(records, out_fasta)
    return out_fasta


def read_fasta(path: Path) -> Dict[str, str]:
    """Read a FASTA (unaligned) file into {name: sequence}."""
    name: Optional[str] = None
    seq_parts: List[str] = []
    out: Dict[str, str] = {}

    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    out[name] = "".join(seq_parts).replace(" ", "").upper()
                name = line[1:].split()[0]
                seq_parts = []
            else:
                seq_parts.append(line)
        if name is not None:
            out[name] = "".join(seq_parts).replace(" ", "").upper()

    if not out:
        raise ValueError(f"No FASTA records found in {path}")
    return out


def write_fasta(records: Dict[str, str], path: Path) -> None:
    """Write {name: seq} FASTA."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for name, seq in records.items():
            f.write(f">{name}\n")
            for i in range(0, len(seq), 80):
                f.write(seq[i : i + 80] + "\n")


def run_mafft(input_fasta: Path, output_fasta: Path, *, mafft_exe: str = "mafft") -> Path:
    """
    Run MAFFT to produce an aligned FASTA.

    Requires: mafft available on PATH.

    Args:
        input_fasta: path to multi-FASTA (unaligned).
        output_fasta: path to write aligned FASTA.

    Returns:
        output_fasta
    """
    output_fasta.parent.mkdir(parents=True, exist_ok=True)
    cmd = [mafft_exe, "--auto", str(input_fasta)]
    try:
        res = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError as e:
        raise RuntimeError(f"MAFFT executable not found: '{mafft_exe}'. Is it installed/loaded?") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"MAFFT failed.\nSTDERR:\n{e.stderr}") from e

    output_fasta.write_text(res.stdout)
    return output_fasta


def load_aligned_fasta(path: Path) -> MSA:
    """Load an aligned FASTA into an MSA object, checking equal lengths."""
    records = read_fasta(path)
    names = list(records.keys())
    seqs = [records[n] for n in names]
    if len({len(s) for s in seqs}) != 1:
        lens = sorted({len(s) for s in seqs})
        raise ValueError(f"Aligned FASTA does not have equal-length sequences: lengths={lens}")
    return MSA(names=names, seqs=seqs)


def shannon_entropy_from_counts(counts: np.ndarray) -> float:
    """
    Shannon entropy in bits from counts over categories (nonnegative).
    Ignores zero-count categories.
    """
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    p = counts[counts > 0].astype(float) / total
    return float(-(p * np.log2(p)).sum())


def compute_column_metrics(msa: MSA) -> pd.DataFrame:
    """
    Compute per-column variability metrics:
      - entropy over amino acids (excluding gaps and non-standard)
      - percent identity (fraction of non-gap residues matching the modal AA)
      - gap_fraction (fraction of sequences with '-')

    Returns:
        DataFrame with columns: aln_col, entropy, pid, gap_fraction
    """
    aln_len = msa.aln_len
    n = msa.n_seqs
    if aln_len == 0 or n == 0:
        raise ValueError("Empty MSA")

    # matrix shape (n_seqs, aln_len)
    mat = np.array([list(s) for s in msa.seqs], dtype="U1")

    gap_fraction = (mat == "-").mean(axis=0)

    entropy = np.zeros(aln_len, dtype=float)
    pid = np.zeros(aln_len, dtype=float)

    for j in range(aln_len):
        col = mat[:, j]
        nongap = col[col != "-"]
        if nongap.size == 0:
            entropy[j] = 0.0
            pid[j] = 0.0
            continue

        # Keep standard AAs only
        aa = np.array([c for c in nongap if c in AA_SET], dtype="U1")
        if aa.size == 0:
            entropy[j] = 0.0
            pid[j] = 0.0
            continue

        # counts in fixed AA alphabet:
        counts = np.array([(aa == a).sum() for a in AA_ALPHABET], dtype=float)
        entropy[j] = shannon_entropy_from_counts(counts)

        # percent identity to modal AA among aa
        modal = counts.max()
        pid[j] = float(modal / aa.size)

    df = pd.DataFrame(
        {
            "aln_col": np.arange(aln_len, dtype=int),
            "entropy": entropy,
            "pid": pid,
            "gap_fraction": gap_fraction.astype(float),
        }
    )
    return df


def reference_residue_to_aln_col(msa: MSA, *, reference_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build mapping between reference residue positions (0-based, ungapped) and alignment columns.

    Returns:
        ref_pos_to_col: length Lref array, maps ref residue index -> aln_col
        col_to_ref_pos: length aln_len array, maps aln_col -> ref residue index or -1 if gap in ref
    """
    if reference_name not in msa.names:
        raise ValueError(f"Reference '{reference_name}' not found in MSA. Available: {msa.names[:10]}...")

    ref_seq = msa.seqs[msa.names.index(reference_name)]
    aln_len = len(ref_seq)

    col_to_ref = np.full(aln_len, -1, dtype=int)
    ref_pos_to_col: List[int] = []

    ref_pos = 0
    for col, ch in enumerate(ref_seq):
        if ch == "-":
            continue
        col_to_ref[col] = ref_pos
        ref_pos_to_col.append(col)
        ref_pos += 1

    return np.array(ref_pos_to_col, dtype=int), col_to_ref


def load_mhcflurry_epitopes_noheader(path: Path) -> pd.DataFrame:
    """
    Load mhcflurry_epitopes.tsv into a DataFrame.


    Required fields:
      allele, protein, peptide, start, end
    At least one of:
      affinity_percentile / presentation_percentile / rank_score (for plotting later)
    """
    # Try headered TSV first (your file is headered)
    df0 = pd.read_csv(path, sep="\t", dtype=str)
    required = {"allele", "protein", "peptide", "start", "end"}
    if required.issubset(df0.columns):
        df = df0.copy()
        df["start"] = df["start"].astype(int)
        df["end"] = df["end"].astype(int)

        # These may or may not exist depending on how you wrote the TSV
        for col in ["affinity_percentile", "presentation_percentile", "rank_score"]:
            if col in df.columns:
                df[col] = df[col].astype(float)

        return df

    # Fallback: no-header legacy format
    df = pd.read_csv(path, sep="\t", header=None, dtype=str)
    if df.shape[1] < 5:
        raise ValueError(f"Unexpected epitopes TSV format: {path} has {df.shape[1]} columns")

    df = df.rename(
        columns={
            0: "allele",
            1: "protein",
            2: "peptide",
            3: "start",
            4: "end",
            6: "affinity_percentile",
        }
    )
    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)
    if "affinity_percentile" in df.columns:
        df["affinity_percentile"] = df["affinity_percentile"].astype(float)
    return df


def annotate_epitopes_with_variability(
    epitopes: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    msa: MSA,
    reference_name: str,
    protein: str,
) -> pd.DataFrame:
    """
    Add variability columns to epitopes for a given protein:
      - mean_entropy, max_entropy, mean_pid, gap_fraction (mean gap fraction over region)
      - score = mean_entropy * (-log10(affinity_percentile))
        (higher score = strong binder in variable region)

    Notes:
      - score is only computed when 'affinity_percentile' exists and > 0.
      - returned rows are ranked by score descending (NaNs last).
    """
    d = epitopes[epitopes["protein"] == protein].copy()
    if d.empty:
        raise ValueError(f"No epitopes for protein='{protein}' to annotate")

    ref_pos_to_col, _col_to_ref = reference_residue_to_aln_col(msa, reference_name=reference_name)

    ent = metrics.set_index("aln_col")["entropy"]
    pid = metrics.set_index("aln_col")["pid"]
    gap = metrics.set_index("aln_col")["gap_fraction"]

    mean_entropy: List[float] = []
    max_entropy: List[float] = []
    mean_pid: List[float] = []
    mean_gap: List[float] = []

    for row in d.itertuples(index=False):
        a = int(getattr(row, "start"))
        b = int(getattr(row, "end"))
        if b < a:
            a, b = b, a

        if a < 0 or b >= ref_pos_to_col.size:
            mean_entropy.append(float("nan"))
            max_entropy.append(float("nan"))
            mean_pid.append(float("nan"))
            mean_gap.append(float("nan"))
            continue

        cols = ref_pos_to_col[a : b + 1]
        e = ent.loc[cols].to_numpy(dtype=float)
        p = pid.loc[cols].to_numpy(dtype=float)
        g = gap.loc[cols].to_numpy(dtype=float)

        mean_entropy.append(float(np.nanmean(e)))
        max_entropy.append(float(np.nanmax(e)))
        mean_pid.append(float(np.nanmean(p)))
        mean_gap.append(float(np.nanmean(g)))

    d["mean_entropy"] = mean_entropy
    d["max_entropy"] = max_entropy
    d["mean_pid"] = mean_pid
    d["gap_fraction"] = mean_gap

    # Compute combined score and rank
    if "affinity_percentile" in d.columns:
        ap = d["affinity_percentile"].astype(float).to_numpy()
        ap = np.where(np.isfinite(ap) & (ap > 0), ap, np.nan)
        strength = -np.log10(ap)  # higher = stronger binder
        d["score"] = d["mean_entropy"].astype(float).to_numpy() * strength
    else:
        d["score"] = np.nan

    d = d.sort_values("score", ascending=False, na_position="last").reset_index(drop=True)
    d["rank_by_score"] = np.arange(1, len(d) + 1, dtype=int)
    return d


def scatter_affinity_vs_entropy(
    df: pd.DataFrame,
    *,
    out_png: Optional[Path] = None,
    title: str = "Affinity percentile vs entropy",
    label_top_n: int = 2,
) -> None:
    """
    Scatter: affinity_percentile (x) vs mean_entropy (y).

    Requirements from user:
      - x axis: 0.00 on the RIGHT and 2.00 on the LEFT (i.e., invert x-axis)
      - label only top 2 ranked epitopes (by 'score' if present, else by mean_entropy)
    """
    import matplotlib.pyplot as plt

    if "affinity_percentile" not in df.columns:
        raise ValueError("Expected column 'affinity_percentile' for scatter plot")
    if "mean_entropy" not in df.columns:
        raise ValueError("Expected column 'mean_entropy' for scatter plot")

    d = df.copy()
    d["affinity_percentile"] = d["affinity_percentile"].astype(float)

    x = d["affinity_percentile"].to_numpy(dtype=float)
    y = d["mean_entropy"].to_numpy(dtype=float)

    plt.figure(figsize=(6.5, 5))
    plt.scatter(x, y, s=18, alpha=0.6)

    # Invert x-axis: 0 on right, larger on left
    plt.gca().invert_xaxis()

    # Label top N by score (preferred)
    if "score" in d.columns and d["score"].notna().any():
        top = d.sort_values("score", ascending=False).head(label_top_n)
    else:
        top = d.sort_values("mean_entropy", ascending=False).head(label_top_n)

    for _, r in top.iterrows():
        lab = f"{r.get('peptide','')}"
        if "allele" in d.columns:
            lab = f"{lab} ({r.get('allele','')})"
        plt.annotate(
            lab,
            (float(r["affinity_percentile"]), float(r["mean_entropy"])),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
            alpha=0.95,
        )

    plt.xlabel("Affinity percentile (lower = stronger)")
    plt.ylabel("Mean Shannon entropy")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    if out_png is not None:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=300)
        plt.close()
    else:
        plt.show()


def _find_peptide_in_reference(ungapped_ref_seq: str, peptide: str) -> int:
    pep = peptide.strip().upper()
    if not pep:
        raise ValueError("Empty peptide")
    i = ungapped_ref_seq.find(pep)
    return i  # -1 if not found


def plot_peptide_msa_window(
    msa: MSA,
    *,
    peptide: str,
    reference_name: str,
    flank: int = 5,
    max_rows: int = 30,
    out_png: Optional[Path] = None,
    title: Optional[str] = None,
) -> Tuple[int, int]:
    """
    Visualise an MSA window around a peptide (reference-mapped).


    Returns:
        (ref_start, ref_end) in reference residue coordinates (0-based, inclusive).
    """
    import matplotlib.pyplot as plt

    if reference_name not in msa.names:
        raise ValueError(f"Reference '{reference_name}' not found in MSA")

    # --- locate peptide in UNGAPPED reference sequence ---
    ref_aln_full = msa.seqs[msa.names.index(reference_name)]
    ref_ungapped = ref_aln_full.replace("-", "")
    pep = peptide.strip().upper()
    hit = _find_peptide_in_reference(ref_ungapped, pep)
    if hit < 0:
        raise ValueError(f"Peptide not found in reference sequence: {pep}")

    ref_start = hit
    ref_end = hit + len(pep) - 1

    # --- mapping between ref positions and alignment columns ---
    ref_pos_to_col, col_to_ref = reference_residue_to_aln_col(msa, reference_name=reference_name)

    pep_col0 = int(ref_pos_to_col[ref_start])
    pep_col1 = int(ref_pos_to_col[ref_end])

    # Define window in ALIGNMENT columns (insertion-safe)
    col0 = max(0, pep_col0 - flank)
    col1 = min(msa.aln_len - 1, pep_col1 + flank)
    aln_cols = np.arange(col0, col1 + 1, dtype=int)
    width = int(col1 - col0 + 1)

    # X tick labels: reference residue index where the reference is not a gap; blank otherwise
    xlabels = []
    for c in aln_cols:
        rp = int(col_to_ref[c])
        xlabels.append("" if rp < 0 else str(rp))

    # --- Put reference first, then truncate rows ---
    names = msa.names[:]
    seqs = msa.seqs[:]
    ref_idx = names.index(reference_name)
    if ref_idx != 0:
        ref_name = names[ref_idx]
        ref_seq = seqs[ref_idx]
        names.pop(ref_idx)
        seqs.pop(ref_idx)
        names = [ref_name] + names
        seqs = [ref_seq] + seqs

    if len(seqs) > max_rows:
        names = names[:max_rows]
        seqs = seqs[:max_rows]

    # Slice sequences to the alignment window
    block = [s[col0 : col1 + 1] for s in seqs]
    mat = np.array([list(s) for s in block], dtype="U1")
    nseq, _ = mat.shape

    # Reference row slice (aligned coordinates)
    ref_row = np.array(list(ref_aln_full[col0 : col1 + 1]), dtype="U1")
    if ref_row.size != width:
        raise RuntimeError(f"Reference slice width mismatch: ref_row={ref_row.size}, window={width}")

    # Entropy (bits) for the shown columns
    metrics = compute_column_metrics(msa).set_index("aln_col")
    ent = metrics.loc[aln_cols, "entropy"].to_numpy(dtype=float)

    # Color: match-to-reference blue, mismatch gray, gap white
    rgb = np.ones((nseq, width, 3), dtype=float)
    for i in range(nseq):
        for j in range(width):
            c = mat[i, j]
            if c == "-":
                rgb[i, j] = (1.0, 1.0, 1.0)
            elif c == ref_row[j]:
                rgb[i, j] = (0.22, 0.50, 0.95)
            else:
                rgb[i, j] = (0.88, 0.88, 0.88)

    # Figure size tuned for legible letters
    fig_w = max(10, width * 0.45)
    fig_h = max(5, nseq * 0.35 + 1.7)
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(2, 1, height_ratios=[6, 1.3], hspace=0.25)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(rgb, aspect="auto", interpolation="nearest")

    # Overlay letters
    fs = 11 if width <= 25 else 9 if width <= 40 else 7
    for i in range(nseq):
        for j in range(width):
            ch = mat[i, j]
            if ch == "-":
                continue
            is_match = (ch == ref_row[j])
            color = "white" if is_match else "black"
            ax.text(j, i, ch, ha="center", va="center", fontsize=fs, color=color)

    ax.set_yticks(np.arange(nseq))
    ax.set_yticklabels(names, fontsize=9)

    # IMPORTANT: no x tick labels on heatmap axis (prevents overlay with entropy plot)
    ax.set_xticks([])
    ax.set_xlabel("")

    ax.set_title(title or f"{reference_name}: {pep}  (ref {ref_start}-{ref_end})")

    # Box peptide region in ALIGNMENT columns (accurate under insertions)
    pep_off0 = pep_col0 - col0
    pep_off1 = pep_col1 - col0
    ax.add_patch(
        plt.Rectangle(
            (pep_off0 - 0.5, -0.5),
            (pep_off1 - pep_off0 + 1),
            nseq,
            fill=False,
            edgecolor="red",
            linewidth=2,
        )
    )

    # Entropy track (bits) on its own axis; show x ticks/labels here
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax)
    ax2.bar(np.arange(width), ent, color="#5aa85a")
    ax2.set_ylabel("Entropy\n(bits)", fontsize=9)
    ax2.set_ylim(0, max(0.5, float(np.nanmax(ent)) * 1.15))
    ax2.grid(True, axis="y", alpha=0.25)

    ax2.set_xticks(np.arange(width))
    ax2.set_xticklabels(xlabels, rotation=0, fontsize=9)
    ax2.set_xlabel("Reference residue index (0-based); blank = gap in reference", fontsize=9)

    plt.tight_layout()

    if out_png is not None:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=350)
        plt.close()
    else:
        plt.show()

    return ref_start, ref_end