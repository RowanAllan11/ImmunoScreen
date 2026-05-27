from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

EnrichmentMode = Literal["proportion", "count"]


@dataclass(frozen=True)
class EpitopeCall:
    """One epitope call mapped to a protein with 0-based inclusive endpoints."""
    protein: str
    allele: str
    start_0: int
    end_0_inclusive: int


def load_ranked_epitopes_tsv(path: Path) -> pd.DataFrame:
    """
    Load the ranked epitope TSV created by scripts/run_ranking.py.

    Required columns:
      - allele, protein, start, end
    Optional columns are ignored.
    """
    df = pd.read_csv(path, sep="\t")
    required = {"allele", "protein", "start", "end"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Ranked epitope TSV missing columns: {sorted(missing)}")

    # Ensure ints
    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)
    return df


def epitopes_to_binary_tracks(
    epitopes_df: pd.DataFrame,
    *,
    protein: str,
    alleles: Optional[Sequence[str]] = None,
    protein_length: Optional[int] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Convert epitope intervals into one binary track per allele.

    Semantics:
      - If an allele has at least one epitope covering a residue -> 1
      - Otherwise -> 0

    Args:
        epitopes_df: DataFrame with columns allele, protein, start, end (end inclusive).
        protein: protein name to subset.
        alleles: optional subset of alleles to include (order preserved).
        protein_length: if None, inferred as (max end + 1) from provided calls.

    Returns:
        tracks: uint8 array (K, L)
        allele_order: list of allele names (length K)
    """
    d = epitopes_df[epitopes_df["protein"] == protein].copy()
    if d.empty:
        raise ValueError(f"No epitopes for protein='{protein}'")

    if alleles is not None:
        allele_order = list(alleles)
        d = d[d["allele"].isin(allele_order)].copy()
        if d.empty:
            raise ValueError(f"No epitopes for protein='{protein}' after allele filter")
    else:
        allele_order = sorted(d["allele"].unique().tolist())

    if protein_length is None:
        protein_length = int(d["end"].max()) + 1
    if protein_length <= 0:
        raise ValueError("protein_length must be > 0")

    tracks = np.zeros((len(allele_order), protein_length), dtype=np.uint8)

    # Vectorized-ish fill using slicing per interval (interval count is manageable)
    for i, allele in enumerate(allele_order):
        da = d[d["allele"] == allele]
        for row in da.itertuples(index=False):
            a = int(row.start)
            b_incl = int(row.end)
            if b_incl < 0 or a >= protein_length:
                continue
            a = max(0, a)
            b_excl = min(protein_length, b_incl + 1)
            if b_excl > a:
                tracks[i, a:b_excl] = 1

    return tracks, allele_order


def bin_edges(seq_len: int, bin_size: int) -> List[Tuple[int, int]]:
    """Half-open bins covering [0, seq_len)."""
    if seq_len <= 0:
        raise ValueError("seq_len must be > 0")
    if bin_size <= 0:
        raise ValueError("bin_size must be > 0")
    return [(i, min(seq_len, i + bin_size)) for i in range(0, seq_len, bin_size)]


def enrichment_per_bin(track: np.ndarray, *, bin_size: int, mode: EnrichmentMode) -> np.ndarray:
    """
    Enrichment per bin for one binary track.

    - proportion: mean of {0,1} in bin
    - count: sum of {0,1} in bin
    """
    x = np.asarray(track, dtype=np.uint8)
    edges = bin_edges(x.size, bin_size)
    out = np.empty(len(edges), dtype=float)
    for j, (a, b) in enumerate(edges):
        seg = x[a:b]
        if mode == "proportion":
            out[j] = float(seg.mean()) if seg.size else 0.0
        elif mode == "count":
            out[j] = float(seg.sum())
        else:
            raise ValueError(f"Unknown mode: {mode}")
    return out


def observed_enrichment(tracks: np.ndarray, *, bin_size: int, mode: EnrichmentMode) -> np.ndarray:
    """Average per-bin enrichment across K tracks."""
    bt = np.asarray(tracks, dtype=np.uint8)
    if bt.ndim != 2 or bt.shape[0] == 0:
        raise ValueError("tracks must be (K, L) with K>=1")
    per = np.vstack([enrichment_per_bin(bt[i], bin_size=bin_size, mode=mode) for i in range(bt.shape[0])])
    return per.mean(axis=0)


def _run_lengths(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (values, lengths) run-length encoding for a 1D uint8 array."""
    a = np.asarray(x, dtype=np.uint8)
    if a.ndim != 1:
        raise ValueError("x must be 1D")
    if a.size == 0:
        return np.array([], dtype=np.uint8), np.array([], dtype=int)

    changes = np.flatnonzero(a[1:] != a[:-1]) + 1
    starts = np.r_[0, changes]
    ends = np.r_[changes, a.size]
    lengths = (ends - starts).astype(int)
    values = a[starts].astype(np.uint8)
    return values, lengths


def shuffle_segments_and_gaps(x: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
    """
    Shuffle 1-run lengths independently from 0-run lengths; reconstruct alternating runs
    (starting with the original first value), preserving total length.
    """
    a = np.asarray(x, dtype=np.uint8)
    vals, lens = _run_lengths(a)
    if lens.size == 0:
        return a.copy()

    seg = lens[vals == 1].astype(int)
    gap = lens[vals == 0].astype(int)
    rng.shuffle(seg)
    rng.shuffle(gap)

    starts_with_one = int(vals[0]) == 1
    out = np.empty(a.size, dtype=np.uint8)

    i_seg = 0
    i_gap = 0
    pos = 0

    for run_idx in range(int(lens.size)):
        want_one = starts_with_one if (run_idx % 2 == 0) else (not starts_with_one)
        if want_one:
            run_len = int(seg[i_seg]) if i_seg < seg.size else int(gap[i_gap])
            val = 1 if i_seg < seg.size else 0
            i_seg += 1 if val == 1 else 0
            i_gap += 1 if val == 0 else 0
        else:
            run_len = int(gap[i_gap]) if i_gap < gap.size else int(seg[i_seg])
            val = 0 if i_gap < gap.size else 1
            i_gap += 1 if val == 0 else 0
            i_seg += 1 if val == 1 else 0

        out[pos : pos + run_len] = val
        pos += run_len

    # Safety (should match exactly)
    if pos != a.size:
        out = out[: a.size]
        if out.size < a.size:
            out = np.pad(out, (0, a.size - out.size), constant_values=0)

    return out


def monte_carlo_null(
    tracks: np.ndarray,
    *,
    bin_size: int,
    mode: EnrichmentMode,
    iterations: int,
    seed: int,
) -> np.ndarray:
    """
    Generate null distribution for the *mean across tracks* enrichment per bin.

    Returns:
        null_stats: (iterations, n_bins)
    """
    bt = np.asarray(tracks, dtype=np.uint8)
    k, l = bt.shape
    n_bins = len(bin_edges(l, bin_size))
    rng = np.random.default_rng(seed)

    null_stats = np.empty((iterations, n_bins), dtype=float)
    for it in range(iterations):
        per_track = np.empty((k, n_bins), dtype=float)
        for i in range(k):
            shuf = shuffle_segments_and_gaps(bt[i], rng=rng)
            per_track[i] = enrichment_per_bin(shuf, bin_size=bin_size, mode=mode)
        null_stats[it] = per_track.mean(axis=0)
    return null_stats


def pvalues_ge(observed: np.ndarray, null_stats: np.ndarray) -> np.ndarray:
    """
    One-sided Monte Carlo p-values: p = P(null >= observed), add-one corrected.

    p = (count(null >= obs) + 1) / (iterations + 1)
    """
    obs = np.asarray(observed, dtype=float)
    null = np.asarray(null_stats, dtype=float)
    if null.ndim != 2 or obs.ndim != 1 or null.shape[1] != obs.shape[0]:
        raise ValueError("Shape mismatch between observed and null_stats")
    ge = (null >= obs[None, :]).sum(axis=0)
    iters = null.shape[0]
    return (ge + 1.0) / (iters + 1.0)


def fdr_by(pvals: np.ndarray, *, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Benjamini–Yekutieli correction; returns (qvals, significant)."""
    p = np.asarray(pvals, dtype=float)
    reject, qvals, _, _ = multipletests(p, alpha=alpha, method="fdr_by")
    return qvals, reject.astype(bool)


def run_epitope_enrichment(
    *,
    ranked_epitopes_tsv: Path,
    protein: str,
    bin_size: int = 100,
    mode: EnrichmentMode = "proportion",
    iterations: int = 10_000,
    seed: int = 0,
    fdr_alpha: float = 0.05,
    alleles: Optional[Sequence[str]] = None,
    protein_length: Optional[int] = None,
) -> pd.DataFrame:
    """
    End-to-end enrichment pipeline using epitope intervals -> binary tracks.

    Args:
        ranked_epitopes_tsv: path to data/output/epitopes/mhcflurry_epitopes.tsv
        protein: protein to analyse (e.g. "AAV9_VP1")
        bin_size: aa bin width
        mode: "proportion" or "count"
        iterations: Monte Carlo iterations
        seed: RNG seed
        fdr_alpha: significance threshold for q-values
        alleles: optional allele subset (K tracks)
        protein_length: optional explicit length. If None inferred from max end.

    Returns:
        DataFrame with:
          bin_index, bin_start_0, bin_end_0_exclusive,
          observed_enrichment, p_value, q_value, significant
    """
    df = load_ranked_epitopes_tsv(ranked_epitopes_tsv)
    tracks, allele_order = epitopes_to_binary_tracks(
        df,
        protein=protein,
        alleles=alleles,
        protein_length=protein_length,
    )

    obs = observed_enrichment(tracks, bin_size=bin_size, mode=mode)
    null = monte_carlo_null(tracks, bin_size=bin_size, mode=mode, iterations=iterations, seed=seed)
    p = pvalues_ge(obs, null)
    q, sig = fdr_by(p, alpha=fdr_alpha)

    edges = bin_edges(tracks.shape[1], bin_size)
    out = pd.DataFrame(
        {
            "bin_index": np.arange(len(edges), dtype=int),
            "bin_start_0": [a for a, _b in edges],
            "bin_end_0_exclusive": [b for _a, b in edges],
            "observed_enrichment": obs,
            "p_value": p,
            "q_value": q,
            "significant": sig,
        }
    )
    out.attrs["alleles"] = allele_order
    return out


def plot_enrichment(df: pd.DataFrame, *, title: str = "Epitope enrichment across bins") -> None:
    """Line plot with significant bins highlighted."""
    import matplotlib.pyplot as plt

    x = df["bin_index"].to_numpy()
    y = df["observed_enrichment"].to_numpy()
    sig = df["significant"].to_numpy(dtype=bool)

    plt.figure(figsize=(12, 4))
    plt.plot(x, y, marker="o", linewidth=1.5, label="Observed enrichment")
    if sig.any():
        plt.scatter(x[sig], y[sig], color="red", zorder=3, label="FDR<0.05")
    plt.xlabel("Bin index")
    plt.ylabel("Enrichment" + (" (proportion)" if True else ""))
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_tracks_heatmap(binary_tracks: np.ndarray, allele_order: Sequence[str]) -> None:
    """Optional heatmap of binary allele tracks."""
    import matplotlib.pyplot as plt

    mat = np.asarray(binary_tracks, dtype=float)
    plt.figure(figsize=(12, max(3, 0.4 * mat.shape[0])))
    plt.imshow(mat, aspect="auto", interpolation="nearest", cmap="Greys", vmin=0.0, vmax=1.0)
    plt.yticks(np.arange(mat.shape[0]), list(allele_order))
    plt.xlabel("AA position (0-based)")
    plt.ylabel("Allele track")
    plt.title("Binary epitope tracks (1=covered by any epitope)")
    plt.colorbar(label="epitope")
    plt.tight_layout()
    plt.show()


def save_enrichment_plot(df: pd.DataFrame, *, out_png: Path, title: str) -> Path:
    """
    Save enrichment line plot (with significant bins highlighted) to a PNG.
    """
    import matplotlib.pyplot as plt

    x = df["bin_index"].to_numpy()
    y = df["observed_enrichment"].to_numpy()
    sig = df["significant"].to_numpy(dtype=bool)

    plt.figure(figsize=(12, 4))
    plt.plot(x, y, marker="o", linewidth=1.5, label="Observed enrichment")
    if sig.any():
        plt.scatter(x[sig], y[sig], color="red", zorder=3, label="FDR<0.05")
    plt.xlabel("Bin index")
    plt.ylabel("Enrichment")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close()
    return out_png


def save_tracks_heatmap(binary_tracks: np.ndarray, allele_order: Sequence[str], *, out_png: Path, title: str) -> Path:
    """
    Save binary-track heatmap to a PNG.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    mat = np.asarray(binary_tracks, dtype=float)
    plt.figure(figsize=(12, max(3, 0.4 * mat.shape[0])))
    plt.imshow(mat, aspect="auto", interpolation="nearest", cmap="Greys", vmin=0.0, vmax=1.0)
    plt.yticks(np.arange(mat.shape[0]), list(allele_order))
    plt.xlabel("AA position (0-based)")
    plt.ylabel("Allele track")
    plt.title(title)
    plt.colorbar(label="epitope (1=yes)")
    plt.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close()
    return out_png