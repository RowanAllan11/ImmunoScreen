from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_position_score_tsvs(input_dir: Path) -> pd.DataFrame:
    tsv_files = sorted(input_dir.glob("*.tsv"))
    if not tsv_files:
        raise FileNotFoundError(f"No .tsv files found in: {input_dir}")

    dfs = [pd.read_csv(p, sep="\t") for p in tsv_files]
    df = pd.concat(dfs, ignore_index=True)

    required = ["protein", "allele", "pos_0", "aa", "max_log_score"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in concatenated TSVs: {missing}")

    return df


def make_heatmap_matrix(
    df: pd.DataFrame,
    protein: str,
    alleles: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    d = df[df["protein"] == protein].copy()
    if d.empty:
        raise ValueError(f"No rows for protein='{protein}' in the provided TSVs")

    if alleles is not None:
        d = d[d["allele"].isin(list(alleles))].copy()
        if d.empty:
            raise ValueError(f"No rows for protein='{protein}' after filtering alleles={alleles}")

    d["position"] = d["pos_0"] + 1  # 1-based for plotting

    heatmap_df = d.pivot_table(
        index="allele",
        columns="position",
        values="max_log_score",
        aggfunc="max",
        fill_value=0.0,
    )

    heatmap_df = heatmap_df.sort_index()
    heatmap_df = heatmap_df.reindex(sorted(heatmap_df.columns), axis=1)
    return heatmap_df


def plot_heatmap(
    heatmap_df: pd.DataFrame,
    protein: str,
    out_png: Path,
    *,
    cmap: str = "Blues",
    vmin: float = 0.0,
    vmax: Optional[float] = None,
    xtick_step: int = 50,
    title: Optional[str] = None,
) -> None:
    n_alleles, seq_len = heatmap_df.shape

    fig_height = max(4, n_alleles * 0.8)
    fig_width = max(14, seq_len / 50)

    plt.figure(figsize=(fig_width, fig_height))

    if vmax is None:
        vmax = max(4.0, float(np.nanmax(heatmap_df.values)))

    im = plt.imshow(
        heatmap_df.values,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    plt.yticks(
        ticks=np.arange(n_alleles),
        labels=heatmap_df.index,
        fontsize=12,
    )

    # x ticks every xtick_step
    cols = heatmap_df.columns.to_list()
    tick_idx = list(range(0, len(cols), max(1, int(xtick_step))))
    plt.xticks(
        ticks=tick_idx,
        labels=[cols[i] for i in tick_idx],
        fontsize=11,
    )

    plt.xlabel("Amino acid position (1-based)", fontsize=14)
    plt.ylabel("Allele", fontsize=14)

    if title is None:
        title = f"{protein} predicted MHC-I presentation landscape"

    plt.title(title, fontsize=18, pad=15)

    cbar = plt.colorbar(im, fraction=0.025, pad=0.03)
    cbar.set_label("Max log Affinity score", fontsize=13)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def render_heatmap_from_dir(
    input_dir: Path,
    protein: str,
    out_png: Path,
    *,
    alleles: Optional[Sequence[str]] = None,
    xtick_step: int = 50,
    vmax: Optional[float] = None,
) -> Path:
    df = load_position_score_tsvs(input_dir)
    hm = make_heatmap_matrix(df, protein=protein, alleles=alleles)
    plot_heatmap(hm, protein=protein, out_png=out_png, xtick_step=xtick_step, vmax=vmax)
    return out_png