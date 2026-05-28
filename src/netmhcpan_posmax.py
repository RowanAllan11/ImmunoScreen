#!/usr/bin/env python3
"""Generate NetMHCpan/NetMHCIIpan per-position max score TSVs + optional heatmap.

This integrates with pipeline outputs:
- NetMHCpan (MHC-I) wide TSV (produced with -xls -BA):
    data/output/netmhcpan/*_NetMHCpan.tsv
  Header structure:
    (optional allele line)
    Pos  Peptide  ID  (then per-allele 6 cols: core icore EL_score EL_rank BA_score BA_rank)  Ave NB

- NetMHCIIpan (MHC-II) wide TSV:
    data/output/netmhcpan/*_NetMHCIIpan.tsv
  Header structure:
    (optional allele line)
    Pos  Peptide  ID  Target  (then per-allele 3 cols: Core Score Rank)  Ave NB

- Fragmentation TSV(s) providing peptide coordinates:
    data/output/fragmentation/*.tsv
  Required cols: peptide, k, start_0, end_0_exclusive

Outputs:
- Per-allele TSVs compatible with src/visualize_mhc_heatmap.py:
    protein, allele, pos_0, aa, max_log_score
- Optional heatmap PNG based on concatenated TSVs

Score used:
- For both MHC-I and MHC-II, we convert a percentile Rank (lower=better) into strength (higher=better):
    strength = -log10(Rank/100)
  * For MHC-I, Rank is EL_rank
  * For MHC-II, Rank is Rank

Examples:

MHC-I:
  python src/netmhcpan_posmax.py \
    --mode mhci \
    --st AAV9 --protein VP1 \
    --input-tsv data/output/netmhcpan/2187821_NetMHCpan.tsv \
    --fragment-dir data/output/fragmentation \
    --out-dir data/output/netmhcpan/position_scores \
    --plot --plot-out data/output/plots

MHC-II:
  python src/netmhcpan_posmax.py \
    --mode mhcii \
    --st AAV9 --protein VP1 \
    --input-tsv data/output/netmhcpan/500747_NetMHCIIpan.tsv \
    --fragment-dir data/output/fragmentation \
    --out-dir data/output/netmhciipan/position_scores \
    --plot --plot-out data/output/plots
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_COLS_MHCI = ["Pos", "Peptide", "ID"]
BASE_COLS_MHCII = ["Pos", "Peptide", "ID", "Target"]
REQUIRED_FRAG_COLS = {"peptide", "k", "start_0", "end_0_exclusive"}


def _percentile_to_strength(pct: float) -> float:
    p = max(float(pct), 1e-300) / 100.0
    return -math.log10(p)


def _is_comment_or_blank(line: str) -> bool:
    s = line.strip()
    return (not s) or s.startswith("#") or s.startswith("//")


def _split_tsv(line: str) -> List[str]:
    return line.rstrip("\n").split("\t")


def _find_header_idx(lines: List[str], startswith: str) -> int:
    for i, line in enumerate(lines):
        if _is_comment_or_blank(line):
            continue
        if line.startswith(startswith):
            return i
    raise ValueError(f"Could not find header row starting with {startswith!r}")


def _parse_allele_names_from_previous_line(lines: List[str], header_idx: int, n_alleles: int) -> List[str]:
    """
    NetMHCpan/IIpan wide TSVs often have an allele-name line right above the column header line.
    That line typically has sparse tabs; we keep non-empty fields and require it to match n_alleles.
    """
    if header_idx - 1 < 0:
        return []
    allele_line = lines[header_idx - 1].rstrip("\n")
    parts = [p.strip() for p in allele_line.split("\t") if p.strip()]
    if len(parts) == n_alleles:
        return parts
    return []


def parse_netmhcpan_xls_wide_tsv(path: Path) -> pd.DataFrame:
    """Parse NetMHCpan (MHC-I) wide TSV with 6 columns per allele."""
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    header_idx = _find_header_idx(lines, "Pos\tPeptide\tID\t")
    header = _split_tsv(lines[header_idx])

    base_end = len(BASE_COLS_MHCI)
    trailing = header[base_end:]

    if trailing[-2:] == ["Ave", "NB"]:
        trailing = trailing[:-2]

    if len(trailing) % 6 != 0:
        raise ValueError(
            f"Unexpected number of non-base columns in MHC-I header. "
            f"Expected multiples of 6 (core/icore/EL_score/EL_rank/BA_score/BA_rank per allele) plus optional Ave/NB; "
            f"got {len(trailing)} for {path}"
        )

    n_alleles = len(trailing) // 6
    allele_names = _parse_allele_names_from_previous_line(lines, header_idx, n_alleles) or [
        f"allele_{i+1}" for i in range(n_alleles)
    ]

    allele_block_cols = ["core", "icore", "EL_score", "EL_rank", "BA_score", "BA_rank"]

    records: List[Dict] = []
    for line in lines[header_idx + 1 :]:
        if _is_comment_or_blank(line):
            continue
        row = _split_tsv(line)
        if len(row) < base_end:
            continue

        expected_len = base_end + (n_alleles * 6) + 2  # allow Ave/NB padding if present
        if len(row) < expected_len:
            row = row + [""] * (expected_len - len(row))

        pos = row[0]
        pep = row[1]
        if not str(pos).isdigit():
            continue

        length = len(pep)
        offset = base_end
        for ai, allele in enumerate(allele_names):
            block = row[offset + ai * 6 : offset + (ai + 1) * 6]
            data = dict(zip(allele_block_cols, block))

            try:
                el_rank = float(data["EL_rank"]) if data["EL_rank"] != "" else None
            except ValueError:
                el_rank = None

            records.append(
                {
                    "peptide": pep,
                    "length": length,
                    "allele": allele,
                    "netMHCpan_EL_rank": el_rank,
                }
            )

    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise ValueError(f"Parsed 0 rows from NetMHCpan TSV: {path}")
    return df


def parse_netmhciipan_wide_tsv(path: Path) -> pd.DataFrame:
    """Parse NetMHCIIpan (MHC-II) wide TSV with 3 columns per allele (Core/Score/Rank)."""
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    header_idx = _find_header_idx(lines, "Pos\tPeptide\tID\t")
    header = _split_tsv(lines[header_idx])

    base_end = len(BASE_COLS_MHCII)
    if len(header) < base_end:
        raise ValueError(f"Header too short for MHC-II layout in {path}: {header}")

    trailing = header[base_end:]

    has_ave_nb = False
    if len(trailing) >= 2 and trailing[-2:] == ["Ave", "NB"]:
        trailing = trailing[:-2]
        has_ave_nb = True

    if len(trailing) % 3 != 0:
        raise ValueError(
            f"Unexpected number of non-base columns in MHC-II header. "
            f"Expected multiples of 3 (Core/Score/Rank per allele) plus optional Ave/NB; got {len(trailing)} for {path}"
        )

    n_alleles = len(trailing) // 3
    allele_names = _parse_allele_names_from_previous_line(lines, header_idx, n_alleles) or [
        f"allele_{i+1}" for i in range(n_alleles)
    ]

    allele_block_cols = ["core", "score", "rank"]

    records: List[Dict] = []
    for line in lines[header_idx + 1 :]:
        if _is_comment_or_blank(line):
            continue
        row = _split_tsv(line)
        if len(row) < base_end:
            continue

        expected_len = base_end + (n_alleles * 3) + (2 if has_ave_nb else 0)
        if len(row) < expected_len:
            row = row + [""] * (expected_len - len(row))

        pos = row[0]
        pep = row[1]
        if not str(pos).isdigit():
            continue

        length = len(pep)
        offset = base_end
        for ai, allele in enumerate(allele_names):
            block = row[offset + ai * 3 : offset + (ai + 1) * 3]
            data = dict(zip(allele_block_cols, block))

            try:
                rank = float(data["rank"]) if data["rank"] != "" else None
            except ValueError:
                rank = None

            records.append(
                {
                    "peptide": pep,
                    "length": length,
                    "allele": allele,
                    # keep downstream column name unchanged; it holds a percentile rank in both modes
                    "netMHCpan_EL_rank": rank,
                }
            )

    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise ValueError(f"Parsed 0 rows from NetMHCIIpan TSV: {path}")
    return df


def _find_fragmentation_tsvs(fragment_dir: Path, st: str) -> List[Path]:
    if not fragment_dir.exists():
        return []
    st_lower = st.lower()
    return sorted([p for p in fragment_dir.glob("*.tsv") if st_lower in p.name.lower()])


def load_fragment_map(fragment_dir: Path, st: str, kmers: Optional[set[int]] = None) -> pd.DataFrame:
    tsvs = _find_fragmentation_tsvs(fragment_dir, st)
    if not tsvs:
        raise FileNotFoundError(f"Could not find fragmentation TSVs for ST={st} in: {fragment_dir}")

    dfs: List[pd.DataFrame] = []
    for fp in tsvs:
        df = pd.read_csv(fp, sep="\t")
        if not REQUIRED_FRAG_COLS.issubset(df.columns):
            continue

        df = df.copy()
        df["peptide"] = df["peptide"].astype(str)
        df["k"] = pd.to_numeric(df["k"], errors="coerce")
        df["start_0"] = pd.to_numeric(df["start_0"], errors="coerce")
        df["end_0_exclusive"] = pd.to_numeric(df["end_0_exclusive"], errors="coerce")
        df = df.dropna(subset=["peptide", "k", "start_0", "end_0_exclusive"]).copy()
        df["k"] = df["k"].astype(int)
        df["start_0"] = df["start_0"].astype(int)
        df["end_0_exclusive"] = df["end_0_exclusive"].astype(int)

        if kmers is not None:
            df = df[df["k"].isin(sorted(kmers))].copy()
        if df.empty:
            continue

        dfs.append(df[["peptide", "k", "start_0", "end_0_exclusive"]])

    if not dfs:
        raise ValueError(
            f"Found fragmentation TSVs in {fragment_dir}, but none had required columns {sorted(REQUIRED_FRAG_COLS)} for ST={st}"
        )

    frag = pd.concat(dfs, ignore_index=True)
    frag = frag.sort_values(["peptide", "k", "start_0", "end_0_exclusive"]).drop_duplicates(
        subset=["peptide", "k"], keep="first"
    )
    return frag


def map_to_positions_max(net_df: pd.DataFrame, frag_map: pd.DataFrame) -> Dict[Tuple[str, int], float]:
    d = net_df.copy()
    d["peptide"] = d["peptide"].astype(str)
    d["allele"] = d["allele"].astype(str)
    d["k"] = d["peptide"].str.len().astype(int)
    d["netMHCpan_EL_rank"] = pd.to_numeric(d["netMHCpan_EL_rank"], errors="coerce")

    merged = d.merge(frag_map, on=["peptide", "k"], how="left", validate="many_to_one")
    merged = merged.dropna(subset=["netMHCpan_EL_rank", "start_0", "end_0_exclusive"]).copy()
    merged["start_0"] = merged["start_0"].astype(int)
    merged["end_0_exclusive"] = merged["end_0_exclusive"].astype(int)

    out: Dict[Tuple[str, int], float] = {}
    for r in merged.itertuples(index=False):
        allele = str(getattr(r, "allele"))
        start_0 = int(getattr(r, "start_0"))
        end_0_excl = int(getattr(r, "end_0_exclusive"))
        score = _percentile_to_strength(float(getattr(r, "netMHCpan_EL_rank")))

        for pos_0 in range(start_0, end_0_excl):
            k = (allele, pos_0)
            prev = out.get(k)
            if prev is None or score > prev:
                out[k] = score

    return out


def write_position_scores_tsv(out_path: Path, protein: str, max_by_allele_pos: Dict[Tuple[str, int], float]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["protein", "allele", "pos_0", "aa", "max_log_score"])
        for (allele, pos_0), score in sorted(max_by_allele_pos.items(), key=lambda x: (x[0][0], x[0][1])):
            w.writerow([protein, allele, pos_0, ".", f"{score:.6g}"])


def _load_position_score_tsvs(input_dir: Path) -> pd.DataFrame:
    tsv_files = sorted(input_dir.glob("*.tsv"))
    if not tsv_files:
        raise FileNotFoundError(f"No .tsv files found in: {input_dir}")
    dfs = [pd.read_csv(p, sep="\t") for p in tsv_files]
    return pd.concat(dfs, ignore_index=True)


def render_heatmap_from_dir(
    input_dir: Path,
    protein: str,
    out_png: Path,
    *,
    xtick_step: int = 50,
    title: Optional[str] = None,
    cbar_label: str = "Max -log10(Rank/100)",
) -> Path:
    df = _load_position_score_tsvs(input_dir)
    d = df[df["protein"] == protein].copy()
    if d.empty:
        raise ValueError(f"No rows for protein='{protein}' in {input_dir}")

    d["position"] = d["pos_0"] + 1
    hm = d.pivot_table(
        index="allele",
        columns="position",
        values="max_log_score",
        aggfunc="max",
        fill_value=0.0,
    ).sort_index()
    hm = hm.reindex(sorted(hm.columns), axis=1)

    n_alleles, seq_len = hm.shape
    fig_height = max(4, n_alleles * 0.8)
    fig_width = max(14, seq_len / 50)

    plt.figure(figsize=(fig_width, fig_height))
    vmax = max(4.0, float(np.nanmax(hm.values))) if hm.size else 4.0

    im = plt.imshow(hm.values, aspect="auto", interpolation="nearest", cmap="Blues", vmin=0.0, vmax=vmax)
    plt.yticks(ticks=np.arange(n_alleles), labels=hm.index, fontsize=12)

    cols = hm.columns.to_list()
    tick_idx = list(range(0, len(cols), max(1, int(xtick_step))))
    plt.xticks(ticks=tick_idx, labels=[cols[i] for i in tick_idx], fontsize=11)

    plt.xlabel("Amino acid position (1-based)", fontsize=14)
    plt.ylabel("Allele", fontsize=14)
    if title is None:
        title = f"{protein} NetMHC mapped Rank to positions"
    plt.title(title, fontsize=16, pad=12)

    cbar = plt.colorbar(im, fraction=0.025, pad=0.03)
    cbar.set_label(cbar_label, fontsize=12)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    return out_png


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=["mhci", "mhcii"],
        default="mhci",
        help="Input TSV layout: mhci=NetMHCpan (6 cols/allele), mhcii=NetMHCIIpan (3 cols/allele).",
    )
    ap.add_argument(
        "--input-tsv",
        type=Path,
        required=True,
        help="Path to NetMHCpan/NetMHCIIpan wide TSV (e.g. data/output/netmhcpan/*_NetMHCpan.tsv)",
    )
    ap.add_argument("--st", required=True, help="ST tag (e.g., AAV9) used to locate fragmentation TSVs")
    ap.add_argument("--protein", default="VP1", help="Protein label (e.g., VP1) used for output labelling")
    ap.add_argument(
        "--fragment-dir",
        type=Path,
        default=Path("data/output/fragmentation"),
        help="Directory containing fragmentation TSV(s)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for per-allele position-score TSVs",
    )
    ap.add_argument("--plot", action="store_true", help="Also generate a heatmap PNG")
    ap.add_argument(
        "--plot-out",
        type=Path,
        default=Path("data/output/plots"),
        help="Output directory for heatmap PNGs",
    )
    ap.add_argument("--plot-xtick-step", type=int, default=50)

    args = ap.parse_args()

    if args.mode == "mhcii":
        net_df = parse_netmhciipan_wide_tsv(args.input_tsv)
    else:
        net_df = parse_netmhcpan_xls_wide_tsv(args.input_tsv)

    kmers = set(pd.to_numeric(net_df["length"], errors="coerce").dropna().astype(int).unique().tolist())
    frag_map = load_fragment_map(args.fragment_dir, st=args.st, kmers=kmers)

    mapped = map_to_positions_max(net_df, frag_map)

    by_allele: Dict[str, Dict[Tuple[str, int], float]] = defaultdict(dict)
    for (allele, pos_0), score in mapped.items():
        by_allele[allele][(allele, pos_0)] = score

    protein_label = f"{args.st}_{args.protein}"

    suffix = "netmhciipan.Rank.posmax" if args.mode == "mhcii" else "netmhcpan.EL_rank.posmax"

    for allele, d in sorted(by_allele.items()):
        safe_allele = allele.replace("/", "_")
        out_path = args.out_dir / f"{protein_label}.{safe_allele}.{suffix}.tsv"
        write_position_scores_tsv(out_path, protein=protein_label, max_by_allele_pos=d)
        print(f"Wrote: {out_path}")

    if args.plot:
        out_png = args.plot_out / f"{protein_label}.{suffix}.heatmap.png"
        title = f"{protein_label} NetMHCIIpan Rank mapped to positions" if args.mode == "mhcii" else f"{protein_label} NetMHCpan EL_rank mapped to positions"
        render_heatmap_from_dir(
            args.out_dir,
            protein=protein_label,
            out_png=out_png,
            xtick_step=args.plot_xtick_step,
            title=title,
            cbar_label="Max -log10(Rank/100)" if args.mode == "mhcii" else "Max -log10(EL_rank/100)",
        )
        print(f"Wrote: {out_png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())