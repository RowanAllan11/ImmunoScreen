#!/usr/bin/env python3
"""
Mutation-immunogenicity modelling for WT-relative AAV variant scores.

Input:
  A variant_immunogenicity_scores.tsv file, or a directory containing multiple
  variant_immunogenicity_scores.tsv files, for example:

  data/output/variant_immunogenicity_scores_wt/VR4_K9_H2-Db/variant_immunogenicity_scores.tsv

Model:
  For each WT-relative count score:
      score ~ one-hot encoded recurrent mutations

Outputs:
  - mutation frequencies
  - coefficient tables with bootstrap mean/sd/CI/sign stability
  - forest plots of top positive and negative mutations
  - long WT-position x substitution coefficient heatmaps (PNG + PDF)
  - ranked mutation tables for each score (TSV + readable HTML)
  - side-by-side and top-positive/top-negative score summaries
  - coefficient correlation plots across scores and, when multiple inputs are used, alleles
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle


DEFAULT_OUTCOMES = [
    "netMHCpan_net_pass_change",
    "MHCflurry_net_pass_change",
    "both_net_pass_change",
    "either_net_pass_change",
]

MUT_RE = re.compile(r"^([A-Za-z\*\-])([0-9]+)([A-Za-z\*\-])$")

# Same amino-acid order as the reference substitution heatmap.
HEATMAP_AA_ORDER = list("CDEKRHNQAGSTVMLIFYWP")


def safe_name(x: str) -> str:
    """Make a string safe for filenames."""
    x = str(x).replace("*", "star")
    x = re.sub(r"[^A-Za-z0-9_.-]+", "_", x)
    return x.strip("_") or "unknown"


def split_mutations(value: object) -> list[str]:
    """Parse VR_mutation strings like 'I2A;Q7Y'."""
    if pd.isna(value):
        return []
    s = str(value).strip()
    if s == "" or s.upper() in {"WT", "NONE", "NAN"}:
        return []
    return [m.strip() for m in s.split(";") if m.strip() and m.strip().upper() != "WT"]


def parse_mutation(mutation: str, position_offset: int = 0) -> dict[str, object]:
    """Parse mutation label T1F into WT residue, relative position and mutant residue."""
    match = MUT_RE.match(str(mutation))
    if not match:
        return {
            "mutation": mutation,
            "wt_aa": np.nan,
            "position_relative": np.nan,
            "position_absolute": np.nan,
            "mut_aa": np.nan,
        }
    wt_aa, pos, mut_aa = match.groups()
    pos_int = int(pos)
    return {
        "mutation": mutation,
        "wt_aa": wt_aa,
        "position_relative": pos_int,
        "position_absolute": pos_int + position_offset,
        "mut_aa": mut_aa,
    }


def find_score_files(input_path: Path) -> list[Path]:
    """Return one or more variant_immunogenicity_scores.tsv files."""
    if input_path.is_file():
        return [input_path]
    files = sorted(input_path.rglob("variant_immunogenicity_scores.tsv"))
    if not files:
        raise FileNotFoundError(f"No variant_immunogenicity_scores.tsv files found under {input_path}")
    return files


def load_scores(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    required = {"variant_id", "VR_mutation"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    # These files should be variant-level already. If duplicates exist, keep the first
    # because duplicated variants would inflate the effective sample size.
    before = len(df)
    df = df.drop_duplicates(subset=["variant_id"]).copy()
    if len(df) < before:
        print(f"[WARN] {path}: dropped {before - len(df)} duplicated variant_id rows")
    return df


def make_mutation_matrix(df: pd.DataFrame, min_count: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    mutation_lists = df["VR_mutation"].apply(split_mutations)
    all_mutations = sorted({m for muts in mutation_lists for m in muts})
    if not all_mutations:
        raise ValueError("No mutations found in VR_mutation column")

    X_all = pd.DataFrame(0, index=df.index, columns=all_mutations, dtype=np.int8)
    for idx, muts in mutation_lists.items():
        if muts:
            X_all.loc[idx, list(set(muts))] = 1

    freq = X_all.sum(axis=0).sort_values(ascending=False)
    keep = freq[freq >= min_count].index.tolist()
    if not keep:
        raise ValueError(
            f"No mutations passed --min-mutation-count {min_count}. "
            f"Try lowering the threshold. Max observed count was {int(freq.max())}."
        )

    X = X_all[keep].copy()
    mutation_freq = freq.rename("n_variants").reset_index().rename(columns={"index": "mutation"})
    mutation_freq["used_in_model"] = mutation_freq["mutation"].isin(keep)
    return X, mutation_freq, freq.loc[keep]


def fit_ols(X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """OLS with intercept using numpy least squares."""
    X_design = np.column_stack([np.ones(X.shape[0]), X])
    beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    return float(beta[0]), beta[1:]


def bootstrap_coefficients(
    X_df: pd.DataFrame,
    y: pd.Series,
    n_boot: int,
    seed: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Fit full OLS and bootstrap coefficients by resampling variants."""
    mask = y.notna()
    y_clean = y.loc[mask].astype(float).to_numpy()
    X_clean = X_df.loc[mask].astype(float).to_numpy()

    if X_clean.shape[0] <= X_clean.shape[1]:
        print(
            f"[WARN] n rows ({X_clean.shape[0]}) <= p predictors ({X_clean.shape[1]}). "
            "OLS coefficients may be unstable."
        )

    intercept, full_coef = fit_ols(X_clean, y_clean)

    rng = np.random.default_rng(seed)
    boot = np.full((n_boot, X_clean.shape[1]), np.nan, dtype=float)
    n = X_clean.shape[0]
    for b in range(n_boot):
        sample_idx = rng.integers(0, n, size=n)
        _, boot_coef = fit_ols(X_clean[sample_idx, :], y_clean[sample_idx])
        boot[b, :] = boot_coef

    return intercept, full_coef, boot


def coefficient_summary(
    mutations: list[str],
    outcome: str,
    intercept: float,
    full_coef: np.ndarray,
    boot: np.ndarray,
    mutation_counts: pd.Series,
    position_offset: int,
) -> pd.DataFrame:
    rows = []
    for j, mutation in enumerate(mutations):
        vals = boot[:, j]
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            mean_coef = sd_coef = ci_low = ci_high = sign_stability = np.nan
        else:
            mean_coef = float(np.mean(vals))
            sd_coef = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
            ci_low, ci_high = np.percentile(vals, [2.5, 97.5])
            if mean_coef == 0:
                sign_stability = np.nan
            elif mean_coef > 0:
                sign_stability = float(np.mean(vals > 0))
            else:
                sign_stability = float(np.mean(vals < 0))

        parsed = parse_mutation(mutation, position_offset=position_offset)
        rows.append(
            {
                "outcome": outcome,
                "mutation": mutation,
                "n_variants_with_mutation": int(mutation_counts.get(mutation, 0)),
                "intercept": intercept,
                "full_data_coef": float(full_coef[j]),
                "boot_mean_coef": mean_coef,
                "boot_sd_coef": sd_coef,
                "boot_ci_low": float(ci_low),
                "boot_ci_high": float(ci_high),
                "boot_sign_stability": sign_stability,
                **parsed,
            }
        )
    return pd.DataFrame(rows)


def plot_forest(coef_df: pd.DataFrame, outpath: Path, top_n: int) -> None:
    """Forest plot of top positive and negative coefficients for one outcome."""
    df = coef_df.dropna(subset=["boot_mean_coef", "boot_ci_low", "boot_ci_high"]).copy()
    if df.empty:
        return
    bottom = df.nsmallest(top_n, "boot_mean_coef")
    top = df.nlargest(top_n, "boot_mean_coef")
    show = pd.concat([bottom, top], axis=0).drop_duplicates(subset=["mutation"])
    show = show.sort_values("boot_mean_coef")

    y_pos = np.arange(len(show))
    x = show["boot_mean_coef"].to_numpy()
    xerr_low = x - show["boot_ci_low"].to_numpy()
    xerr_high = show["boot_ci_high"].to_numpy() - x

    height = max(4, 0.45 * len(show) + 1.5)
    fig, ax = plt.subplots(figsize=(8, height))
    ax.errorbar(x, y_pos, xerr=[xerr_low, xerr_high], fmt="o", capsize=3)
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(show["mutation"].tolist())
    ax.set_xlabel("Bootstrap mean coefficient")
    ax.set_title(f"Top and bottom mutation coefficients: {coef_df['outcome'].iloc[0]}")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def _normalise_wt_sequence(sequence: Optional[str]) -> Optional[str]:
    """Remove whitespace from an optional WT sequence and validate its symbols."""
    if sequence is None:
        return None
    cleaned = re.sub(r"\s+", "", str(sequence)).upper()
    if not cleaned:
        return None
    invalid = sorted(set(cleaned) - set(HEATMAP_AA_ORDER) - {"*", "-", "X"})
    if invalid:
        raise ValueError(
            "--wt-sequence contains unsupported symbols: " + ", ".join(invalid)
        )
    return cleaned


def _mode_or_unknown(values: pd.Series) -> str:
    """Return the most frequent one-letter residue, or '?' when unavailable."""
    cleaned = values.dropna().astype(str).str.upper().str[0]
    cleaned = cleaned[cleaned.isin(HEATMAP_AA_ORDER)]
    if cleaned.empty:
        return "?"
    return str(cleaned.mode().iloc[0])


def _format_wt_sequence_caption(
    wt_sequence: str,
    start_position: int,
    line_width: int = 60,
) -> str:
    """Format a long WT sequence into position-labelled blocks."""
    if not wt_sequence:
        return ""
    lines = []
    for offset in range(0, len(wt_sequence), line_width):
        block = wt_sequence[offset : offset + line_width]
        block_start = start_position + offset
        block_end = block_start + len(block) - 1
        grouped = " ".join(
            block[i : i + 10] for i in range(0, len(block), 10)
        )
        lines.append(f"{block_start:>5}  {grouped}  {block_end:<5}")
    return "WT sequence:\n" + "\n".join(lines)


def plot_position_substitution_heatmap(
    coef_df: pd.DataFrame,
    outpath: Path,
    position_offset: int,
    wt_sequence: Optional[str] = None,
    wt_sequence_start: int = 1,
    row_height: float = 0.13,
    dpi: int = 250,
    fixed_vmax: Optional[float] = None,
    min_sign_stability: float = 0.80,
) -> None:
    """
    Plot a substitution-effect heatmap

    Rows are WT sequence positions and columns are the 20 possible mutant amino
    acids. Mutations at the same position are therefore grouped on one row.

    Cell interpretation:
      - white: mutation was not modelled / not present
      - grey: coefficient was estimated but bootstrap sign stability is below
        ``min_sign_stability``
      - green -> yellow -> red: stable negative, near-zero and positive effects
      - outlined white: the native WT amino acid at that position

    A PNG and a vector PDF are written. The WT sequence is shown in the row
    labels and repeated as a position-labelled caption below the heatmap.
    """
    required = {
        "boot_mean_coef",
        "boot_sign_stability",
        "position_relative",
        "wt_aa",
        "mut_aa",
    }
    missing = sorted(required - set(coef_df.columns))
    if missing:
        print(f"[WARN] Heatmap skipped; missing columns: {missing}")
        return

    df = coef_df.dropna(
        subset=["boot_mean_coef", "position_relative", "mut_aa"]
    ).copy()
    if df.empty:
        return

    df["position_relative"] = pd.to_numeric(
        df["position_relative"], errors="coerce"
    )
    df = df.dropna(subset=["position_relative"])
    df["position_relative"] = df["position_relative"].astype(int)
    df["mut_aa"] = df["mut_aa"].astype(str).str.upper().str[0]
    df = df[df["mut_aa"].isin(HEATMAP_AA_ORDER)].copy()
    if df.empty:
        return

    sequence = _normalise_wt_sequence(wt_sequence)

    inferred_wt = (
        df.groupby("position_relative", sort=True)["wt_aa"]
        .apply(_mode_or_unknown)
        .to_dict()
    )

    if sequence is not None:
        relative_positions = list(
            range(wt_sequence_start, wt_sequence_start + len(sequence))
        )
        wt_by_position = {
            position: sequence[position - wt_sequence_start]
            for position in relative_positions
        }
    else:
        # Only show positions represented in the coefficient table. Supplying
        # --wt-sequence is the way to force unmutated positions into the plot.
        relative_positions = sorted(df["position_relative"].unique().tolist())
        wt_by_position = {
            position: inferred_wt.get(position, "?")
            for position in relative_positions
        }
        sequence = "".join(wt_by_position[position] for position in relative_positions)

    position_to_row = {
        position: row for row, position in enumerate(relative_positions)
    }
    aa_to_column = {
        amino_acid: column
        for column, amino_acid in enumerate(HEATMAP_AA_ORDER)
    }

    n_rows = len(relative_positions)
    n_columns = len(HEATMAP_AA_ORDER)
    coefficients = np.full((n_rows, n_columns), np.nan, dtype=float)
    stability = np.full((n_rows, n_columns), np.nan, dtype=float)

    grouped = (
        df.groupby(["position_relative", "mut_aa"], as_index=False)
        .agg(
            boot_mean_coef=("boot_mean_coef", "mean"),
            boot_sign_stability=("boot_sign_stability", "mean"),
        )
    )

    for row in grouped.itertuples(index=False):
        if row.position_relative not in position_to_row:
            continue
        matrix_row = position_to_row[row.position_relative]
        matrix_column = aa_to_column[row.mut_aa]
        coefficients[matrix_row, matrix_column] = float(row.boot_mean_coef)
        stability[matrix_row, matrix_column] = float(row.boot_sign_stability)

    finite_coefficients = coefficients[np.isfinite(coefficients)]
    if finite_coefficients.size == 0:
        return

    if fixed_vmax is None:
        # A robust range prevents one extreme coefficient from flattening the
        # colour contrast for every other mutation.
        vmax = float(np.nanpercentile(np.abs(finite_coefficients), 98))
        observed_max = float(np.nanmax(np.abs(finite_coefficients)))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = observed_max
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
    else:
        vmax = abs(float(fixed_vmax))
        if vmax == 0:
            raise ValueError("--heatmap-vmax must be greater than zero")

    effect_cmap = mcolors.LinearSegmentedColormap.from_list(
        "mutation_effect",
        ["#079700", "#ffff00", "#ff5a6b"],
        N=401,
    )
    effect_norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    caption_start = wt_sequence_start if wt_sequence is not None else relative_positions[0]
    caption = _format_wt_sequence_caption(sequence, caption_start)
    caption_lines = caption.count("\n") + 1 if caption else 0
    bottom_inches = 0.50 + 0.14 * caption_lines

    figure_width = 6.4
    figure_height = max(6.0, 2.1 + row_height * n_rows + bottom_inches)
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))

    # Leave room below for the full sequence and to the right for the colourbar.
    bottom_fraction = min(0.28, bottom_inches / figure_height)
    fig.subplots_adjust(
        left=0.19,
        right=0.81,
        top=0.965,
        bottom=bottom_fraction,
    )

    ax.set_facecolor("white")
    masked = np.ma.masked_invalid(np.clip(coefficients, -vmax, vmax))
    image = ax.imshow(
        masked,
        origin="upper",
        aspect="auto",
        interpolation="none",
        cmap=effect_cmap,
        norm=effect_norm,
    )

    # Grey-out coefficients whose bootstrap sign is not sufficiently stable.
    uncertain = np.isfinite(coefficients) & (
        ~np.isfinite(stability) | (stability < min_sign_stability)
    )
    uncertain_rgba = np.zeros((n_rows, n_columns, 4), dtype=float)
    uncertain_rgba[uncertain] = mcolors.to_rgba("0.82")
    ax.imshow(
        uncertain_rgba,
        origin="upper",
        aspect="auto",
        interpolation="none",
    )

    # Native amino acids are kept white and outlined, matching the reference.
    for row_index, relative_position in enumerate(relative_positions):
        wt_amino_acid = wt_by_position.get(relative_position, "?")
        native_column = aa_to_column.get(wt_amino_acid)
        if native_column is None:
            continue
        ax.add_patch(
            Rectangle(
                (native_column - 0.5, row_index - 0.5),
                1,
                1,
                facecolor="white",
                edgecolor="black",
                linewidth=0.45,
                zorder=4,
            )
        )

    display_positions = [
        position + position_offset for position in relative_positions
    ]
    y_labels = [
        f"{wt_by_position.get(relative_position, '?')}{display_position}"
        for relative_position, display_position in zip(
            relative_positions, display_positions
        )
    ]

    ax.set_xticks(np.arange(n_columns))
    ax.set_xticklabels(HEATMAP_AA_ORDER, fontsize=7)
    ax.tick_params(
        axis="x",
        top=True,
        bottom=True,
        labeltop=True,
        labelbottom=True,
        length=2.5,
        pad=2,
    )

    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(y_labels, fontsize=5.2)
    ax.tick_params(axis="y", length=2, pad=2)

    ax.set_xlim(-0.5, n_columns - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_xlabel("Mutant amino acid", fontsize=8, labelpad=7)
    ax.set_ylabel(
        "WT residue and capsid position"
        if position_offset != 0
        else "WT residue and VR-relative position",
        fontsize=8,
    )

    outcome = str(coef_df["outcome"].iloc[0])
    ax.set_title(
        f"Mutation coefficient heatmap: {outcome}\n"
        f"Grey = bootstrap sign stability < {min_sign_stability:.0%}; "
        "outlined white = native residue",
        fontsize=9,
        pad=24,
    )

    # Fine grid lines help visually group substitutions within each position.
    ax.set_xticks(np.arange(-0.5, n_columns, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", linewidth=0.12, alpha=0.22)
    ax.tick_params(which="minor", bottom=False, left=False)

    colourbar_axis = fig.add_axes([0.845, bottom_fraction, 0.026, 0.28])
    colourbar = fig.colorbar(image, cax=colourbar_axis)
    colourbar.set_label("Bootstrap mean coefficient", fontsize=7)
    colourbar.ax.tick_params(labelsize=7)

    if caption:
        fig.text(
            0.19,
            0.012,
            caption,
            ha="left",
            va="bottom",
            family="monospace",
            fontsize=6.2,
        )

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(outpath.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def rank_mutations(coef_df: pd.DataFrame) -> pd.DataFrame:
    """Create a readable, fully ranked coefficient table for one score."""
    ranked = coef_df.copy()
    ranked["absolute_coefficient"] = ranked["boot_mean_coef"].abs()
    ranked["ci_excludes_zero"] = (
        (ranked["boot_ci_low"] > 0) | (ranked["boot_ci_high"] < 0)
    )
    ranked["direction"] = np.select(
        [ranked["boot_mean_coef"] > 0, ranked["boot_mean_coef"] < 0],
        ["increases score", "decreases score"],
        default="near zero",
    )
    ranked["rank_by_absolute_effect"] = ranked["absolute_coefficient"].rank(
        method="min", ascending=False
    ).astype("Int64")

    positive = ranked["boot_mean_coef"].where(ranked["boot_mean_coef"] > 0)
    negative = ranked["boot_mean_coef"].where(ranked["boot_mean_coef"] < 0)
    ranked["rank_most_positive"] = positive.rank(
        method="min", ascending=False
    ).astype("Int64")
    ranked["rank_most_negative"] = negative.rank(
        method="min", ascending=True
    ).astype("Int64")
    ranked["sign_stability_percent"] = 100 * ranked["boot_sign_stability"]
    ranked["coefficient_95ci"] = ranked.apply(
        lambda row: (
            f"{row['boot_mean_coef']:.4g} "
            f"[{row['boot_ci_low']:.4g}, {row['boot_ci_high']:.4g}]"
        )
        if pd.notna(row["boot_mean_coef"])
        and pd.notna(row["boot_ci_low"])
        and pd.notna(row["boot_ci_high"])
        else "",
        axis=1,
    )

    ranked = ranked.sort_values(
        ["absolute_coefficient", "boot_sign_stability"],
        ascending=[False, False],
        na_position="last",
    )

    preferred_columns = [
        "outcome",
        "rank_by_absolute_effect",
        "rank_most_positive",
        "rank_most_negative",
        "mutation",
        "wt_aa",
        "position_relative",
        "position_absolute",
        "mut_aa",
        "direction",
        "boot_mean_coef",
        "absolute_coefficient",
        "boot_ci_low",
        "boot_ci_high",
        "coefficient_95ci",
        "ci_excludes_zero",
        "boot_sign_stability",
        "sign_stability_percent",
        "n_variants_with_mutation",
        "full_data_coef",
        "run_label",
        "allele",
        "source_file",
    ]
    return ranked[[column for column in preferred_columns if column in ranked.columns]]


def write_readable_html_table(
    table: pd.DataFrame,
    outpath: Path,
    title: str,
) -> None:
    """Write a standalone scrollable HTML table with fixed headers."""
    display = table.copy()
    numeric_formats = {
        "boot_mean_coef": "{:.4g}",
        "absolute_coefficient": "{:.4g}",
        "boot_ci_low": "{:.4g}",
        "boot_ci_high": "{:.4g}",
        "boot_sign_stability": "{:.3f}",
        "sign_stability_percent": "{:.1f}",
        "full_data_coef": "{:.4g}",
    }
    for column, number_format in numeric_formats.items():
        if column in display.columns:
            display[column] = display[column].map(
                lambda value: number_format.format(value)
                if pd.notna(value)
                else ""
            )

    html_table = display.to_html(
        index=False,
        escape=True,
        classes="ranking-table",
        border=0,
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
h1 {{ font-size: 20px; margin-bottom: 8px; }}
p {{ color: #555; max-width: 1000px; }}
.table-wrap {{ overflow: auto; max-height: 88vh; border: 1px solid #ccc; }}
table {{ border-collapse: collapse; width: max-content; min-width: 100%; font-size: 12px; }}
thead th {{ position: sticky; top: 0; background: #f2f2f2; z-index: 2; }}
th, td {{ border-bottom: 1px solid #ddd; padding: 6px 8px; text-align: right; white-space: nowrap; }}
th:nth-child(1), td:nth-child(1), th:nth-child(5), td:nth-child(5),
th:nth-child(10), td:nth-child(10) {{ text-align: left; }}
tbody tr:nth-child(even) {{ background: #fafafa; }}
tbody tr:hover {{ background: #fff4cc; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p>Rank 1 indicates the largest effect in the named direction. The absolute-effect rank ignores coefficient sign.</p>
<div class="table-wrap">{html_table}</div>
</body>
</html>
"""
    outpath.write_text(html, encoding="utf-8")


def build_cross_score_ranking_table(ranked_long: pd.DataFrame) -> pd.DataFrame:
    """Create one mutation-per-row table for side-by-side score comparison."""
    if ranked_long.empty:
        return pd.DataFrame()

    metadata_columns = [
        column
        for column in [
            "mutation",
            "wt_aa",
            "position_relative",
            "position_absolute",
            "mut_aa",
            "n_variants_with_mutation",
        ]
        if column in ranked_long.columns
    ]
    metadata = ranked_long[metadata_columns].drop_duplicates(subset=["mutation"])

    value_columns = [
        "boot_mean_coef",
        "rank_by_absolute_effect",
        "rank_most_positive",
        "rank_most_negative",
        "coefficient_95ci",
        "sign_stability_percent",
        "ci_excludes_zero",
    ]

    result = metadata.set_index("mutation")
    for outcome in ranked_long["outcome"].dropna().unique():
        outcome_data = ranked_long[ranked_long["outcome"] == outcome].copy()
        outcome_data = outcome_data.drop_duplicates(subset=["mutation"]).set_index("mutation")
        prefix = safe_name(outcome)
        selected = outcome_data[
            [column for column in value_columns if column in outcome_data.columns]
        ].rename(columns=lambda column: f"{prefix}__{column}")
        result = result.join(selected, how="outer")

    rank_columns = [
        column for column in result.columns if column.endswith("__rank_by_absolute_effect")
    ]
    if rank_columns:
        result["best_absolute_rank_across_scores"] = result[rank_columns].min(axis=1)
        result = result.sort_values(
            ["best_absolute_rank_across_scores"], na_position="last"
        )

    return result.reset_index()


def build_top_mutation_summary(
    ranked_long: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    """Return the strongest positive and negative mutations for each score."""
    rows = []
    for outcome, group in ranked_long.groupby("outcome", sort=False):
        positive = group[group["boot_mean_coef"] > 0].nsmallest(
            top_n, "rank_most_positive"
        )
        negative = group[group["boot_mean_coef"] < 0].nsmallest(
            top_n, "rank_most_negative"
        )
        for direction, subset, rank_column in [
            ("most positive", positive, "rank_most_positive"),
            ("most negative", negative, "rank_most_negative"),
        ]:
            if subset.empty:
                continue
            selected = subset.copy()
            selected.insert(1, "ranking_direction", direction)
            selected.insert(2, "direction_rank", selected[rank_column])
            rows.append(selected)

    if not rows:
        return pd.DataFrame()
    summary = pd.concat(rows, ignore_index=True)
    keep = [
        "outcome",
        "ranking_direction",
        "direction_rank",
        "mutation",
        "position_relative",
        "position_absolute",
        "boot_mean_coef",
        "coefficient_95ci",
        "sign_stability_percent",
        "ci_excludes_zero",
        "n_variants_with_mutation",
    ]
    return summary[[column for column in keep if column in summary.columns]]


def write_ranking_outputs(
    ranked_long: pd.DataFrame,
    outdir: Path,
    summary_top_n: int,
) -> None:
    """Write long, side-by-side and top-mutation ranking summaries."""
    if ranked_long.empty:
        return

    ranked_long.to_csv(
        outdir / "ranked_mutations_all_scores.tsv",
        sep="\t",
        index=False,
    )
    write_readable_html_table(
        ranked_long,
        outdir / "ranked_mutations_all_scores.html",
        "Ranked mutations across all scores",
    )

    cross_score = build_cross_score_ranking_table(ranked_long)
    if not cross_score.empty:
        cross_score.to_csv(
            outdir / "mutation_rankings_side_by_side.tsv",
            sep="\t",
            index=False,
        )
        write_readable_html_table(
            cross_score,
            outdir / "mutation_rankings_side_by_side.html",
            "Mutation coefficient rankings side by side",
        )

    top_summary = build_top_mutation_summary(ranked_long, summary_top_n)
    if not top_summary.empty:
        top_summary.to_csv(
            outdir / "top_ranked_mutations_by_score.tsv",
            sep="\t",
            index=False,
        )
        write_readable_html_table(
            top_summary,
            outdir / "top_ranked_mutations_by_score.html",
            f"Top {summary_top_n} positive and negative mutations per score",
        )


def plot_correlation_matrix(matrix: pd.DataFrame, outpath: Path, title: str) -> Optional[pd.DataFrame]:
    """Plot a correlation heatmap using matplotlib."""
    matrix = matrix.dropna(axis=1, how="all")
    if matrix.shape[1] < 2:
        return None
    corr = matrix.corr(method="pearson", min_periods=3)
    data = corr.to_numpy(dtype=float)

    fig_w = max(6, 0.7 * corr.shape[1] + 2)
    fig_h = max(5, 0.6 * corr.shape[0] + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(data, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(corr.shape[1]))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(corr.shape[0]))
    ax.set_yticklabels(corr.index)
    ax.set_title(title)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            val = data[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Pearson r")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    return corr


def plot_allele_scatter(combined: pd.DataFrame, outdir: Path) -> None:
    """If exactly two alleles are present, plot coefficient scatter by outcome."""
    labels = sorted(combined["run_label"].dropna().unique())
    if len(labels) != 2:
        return

    for outcome in sorted(combined["outcome"].dropna().unique()):
        sub = combined[combined["outcome"] == outcome]
        wide = sub.pivot_table(index="mutation", columns="run_label", values="boot_mean_coef", aggfunc="mean")
        wide = wide.dropna()
        if len(wide) < 3:
            continue
        x = wide[labels[0]].to_numpy()
        y = wide[labels[1]].to_numpy()
        r = np.corrcoef(x, y)[0, 1] if len(wide) >= 3 else np.nan

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(x, y, alpha=0.65)
        ax.axhline(0, linestyle="--", linewidth=1)
        ax.axvline(0, linestyle="--", linewidth=1)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_title(f"Allele coefficient comparison: {outcome}\nr = {r:.2f}")
        fig.tight_layout()
        fig.savefig(outdir / f"allele_scatter_{safe_name(outcome)}.png", dpi=300)
        plt.close(fig)


def write_collinearity_report(X: pd.DataFrame, outpath: Path, threshold: float = 0.8) -> None:
    """Write mutation pairs with high absolute correlation."""
    if X.shape[1] < 2:
        pd.DataFrame(columns=["mutation_1", "mutation_2", "pearson_r"]).to_csv(outpath, sep="\t", index=False)
        return

    corr = X.astype(float).corr()
    rows = []
    cols = list(corr.columns)

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if pd.notna(r) and abs(r) >= threshold:
                rows.append({"mutation_1": cols[i], "mutation_2": cols[j], "pearson_r": r})

    if not rows:
        pd.DataFrame(columns=["mutation_1", "mutation_2", "pearson_r"]).to_csv(outpath, sep="\t", index=False)
        return

    pd.DataFrame(rows).sort_values(
        "pearson_r",
        key=lambda s: s.abs(),
        ascending=False
    ).to_csv(outpath, sep="\t", index=False)


def analyse_one_file(
    path: Path,
    outdir: Path,
    outcomes: list[str],
    min_count: int,
    n_boot: int,
    seed: int,
    top_n: int,
    position_offset: int,
    corr_threshold: float,
    wt_sequence: Optional[str],
    wt_sequence_start: int,
    heatmap_row_height: float,
    heatmap_dpi: int,
    heatmap_vmax: Optional[float],
    heatmap_min_sign_stability: float,
    summary_top_n: int,
) -> pd.DataFrame:
    print(f"[INFO] Analysing {path}")
    df = load_scores(path)
    run_label = path.parent.name if path.name == "variant_immunogenicity_scores.tsv" else path.stem
    if "allele" in df.columns and df["allele"].nunique(dropna=True) == 1:
        allele_value = str(df["allele"].dropna().iloc[0])
    elif "allele" in df.columns:
        allele_value = "multiple_alleles"
    else:
        allele_value = "unknown_allele"

    run_out = outdir / safe_name(run_label)
    run_out.mkdir(parents=True, exist_ok=True)

    X, mutation_freq, kept_counts = make_mutation_matrix(df, min_count=min_count)
    mutation_freq.to_csv(run_out / "mutation_frequencies.tsv", sep="\t", index=False)
    write_collinearity_report(X, run_out / "high_correlation_mutation_pairs.tsv", threshold=corr_threshold)

    available_outcomes = [o for o in outcomes if o in df.columns]
    missing = sorted(set(outcomes) - set(available_outcomes))
    if missing:
        print(f"[WARN] {path}: missing outcomes skipped: {missing}")
    if not available_outcomes:
        raise ValueError(f"None of the requested outcomes were present in {path}")

    all_coef = []
    all_ranked = []
    for outcome_idx, outcome in enumerate(available_outcomes):
        y = pd.to_numeric(df[outcome], errors="coerce")
        if y.notna().sum() < 10:
            print(f"[WARN] {path}: skipping {outcome}; fewer than 10 non-missing values")
            continue
        intercept, full_coef, boot = bootstrap_coefficients(
            X, y, n_boot=n_boot, seed=seed + outcome_idx
        )
        coef = coefficient_summary(
            mutations=list(X.columns),
            outcome=outcome,
            intercept=intercept,
            full_coef=full_coef,
            boot=boot,
            mutation_counts=kept_counts,
            position_offset=position_offset,
        )
        coef.insert(0, "source_file", str(path))
        coef.insert(1, "run_label", run_label)
        coef.insert(2, "allele", allele_value)
        all_coef.append(coef)

        coefficient_path = run_out / f"coefficients_{safe_name(outcome)}.tsv"
        coef.to_csv(coefficient_path, sep="\t", index=False)
        plot_forest(
            coef,
            run_out / f"forest_top_bottom_{safe_name(outcome)}.png",
            top_n=top_n,
        )
        plot_position_substitution_heatmap(
            coef,
            run_out / f"position_substitution_heatmap_{safe_name(outcome)}.png",
            position_offset=position_offset,
            wt_sequence=wt_sequence,
            wt_sequence_start=wt_sequence_start,
            row_height=heatmap_row_height,
            dpi=heatmap_dpi,
            fixed_vmax=heatmap_vmax,
            min_sign_stability=heatmap_min_sign_stability,
        )

        ranked = rank_mutations(coef)
        all_ranked.append(ranked)
        ranked.to_csv(
            run_out / f"ranked_mutations_{safe_name(outcome)}.tsv",
            sep="\t",
            index=False,
        )
        write_readable_html_table(
            ranked,
            run_out / f"ranked_mutations_{safe_name(outcome)}.html",
            f"Ranked mutation coefficients: {outcome}",
        )

    if not all_coef:
        raise ValueError(f"No requested outcomes could be modelled for {path}")

    combined = pd.concat(all_coef, ignore_index=True)
    combined.to_csv(run_out / "all_coefficients.tsv", sep="\t", index=False)

    ranked_long = pd.concat(all_ranked, ignore_index=True)
    write_ranking_outputs(
        ranked_long=ranked_long,
        outdir=run_out,
        summary_top_n=summary_top_n,
    )

    # Within-run comparison across score types.
    wide_scores = combined.pivot_table(
        index="mutation", columns="outcome", values="boot_mean_coef", aggfunc="mean"
    )
    corr = plot_correlation_matrix(
        wide_scores,
        run_out / "coefficient_correlation_across_scores.png",
        title=f"Coefficient correlation across scores: {run_label}",
    )
    if corr is not None:
        corr.to_csv(run_out / "coefficient_correlation_across_scores.tsv", sep="\t")

    return combined


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Simple bootstrapped LR of WT-relative immunogenicity count scores on one-hot mutations."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input variant_immunogenicity_scores.tsv file, or directory containing such files.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("data/output/mutation_immunogenicity_models"),
        help="Output directory.",
    )
    parser.add_argument(
        "--outcomes",
        nargs="+",
        default=DEFAULT_OUTCOMES,
        help="Outcome columns to model. Defaults to the four WT-relative count scores.",
    )
    parser.add_argument(
        "--min-mutation-count",
        type=int,
        default=20,
        help="Only include mutations observed in at least this many variants.",
    )
    parser.add_argument(
        "--bootstraps",
        type=int,
        default=100,
        help="Number of bootstrap resamples.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Top and bottom N mutations for forest plots.",
    )
    parser.add_argument(
        "--summary-top-n",
        type=int,
        default=20,
        help="Top positive and negative mutations per score in the readable summary.",
    )
    parser.add_argument(
        "--position-offset",
        type=int,
        default=0,
        help="Add this to VR-relative mutation position. For VR4 T1F -> capsid 450, use --position-offset 449.",
    )
    parser.add_argument(
        "--wt-sequence",
        type=str,
        default=None,
        help=(
            "Actual WT amino-acid sequence for the plotted region. When supplied, "
            "all sequence positions are retained in the heatmap and the sequence "
            "is printed below it. Without this option, WT residues are inferred "
            "from mutation labels."
        ),
    )
    parser.add_argument(
        "--wt-sequence-start",
        type=int,
        default=1,
        help="VR-relative position represented by the first character of --wt-sequence.",
    )
    parser.add_argument(
        "--heatmap-row-height",
        type=float,
        default=0.13,
        help="Figure height in inches allocated per sequence position. Increase for very dense labels.",
    )
    parser.add_argument(
        "--heatmap-dpi",
        type=int,
        default=250,
        help="PNG resolution for the long substitution heatmaps.",
    )
    parser.add_argument(
        "--heatmap-vmax",
        type=float,
        default=None,
        help=(
            "Optional common absolute colour limit for all score heatmaps. "
            "For example, 1 gives a scale from -1 to +1. By default, each score "
            "uses its robust 98th-percentile absolute coefficient."
        ),
    )
    parser.add_argument(
        "--heatmap-min-sign-stability",
        type=float,
        default=0.80,
        help="Coefficients below this bootstrap sign stability are shown in grey.",
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.8,
        help="Absolute mutation co-occurrence correlation threshold to report.",
    )
    args = parser.parse_args()

    if args.bootstraps < 1:
        parser.error("--bootstraps must be at least 1")
    if args.min_mutation_count < 1:
        parser.error("--min-mutation-count must be at least 1")
    if args.summary_top_n < 1:
        parser.error("--summary-top-n must be at least 1")
    if args.heatmap_row_height <= 0:
        parser.error("--heatmap-row-height must be greater than zero")
    if args.heatmap_dpi < 72:
        parser.error("--heatmap-dpi must be at least 72")
    if not 0 <= args.heatmap_min_sign_stability <= 1:
        parser.error("--heatmap-min-sign-stability must be between 0 and 1")

    wt_sequence = _normalise_wt_sequence(args.wt_sequence)

    args.outdir.mkdir(parents=True, exist_ok=True)
    files = find_score_files(args.input)
    print(f"[INFO] Found {len(files)} score file(s)")
    if wt_sequence is not None and len(files) > 1:
        print(
            "[WARN] The same --wt-sequence will be used for every input file. "
            "Run files separately if they represent different variable regions."
        )

    all_results = []
    for i, path in enumerate(files):
        result = analyse_one_file(
            path=path,
            outdir=args.outdir,
            outcomes=args.outcomes,
            min_count=args.min_mutation_count,
            n_boot=args.bootstraps,
            seed=args.seed + i * 1000,
            top_n=args.top_n,
            position_offset=args.position_offset,
            corr_threshold=args.corr_threshold,
            wt_sequence=wt_sequence,
            wt_sequence_start=args.wt_sequence_start,
            heatmap_row_height=args.heatmap_row_height,
            heatmap_dpi=args.heatmap_dpi,
            heatmap_vmax=args.heatmap_vmax,
            heatmap_min_sign_stability=args.heatmap_min_sign_stability,
            summary_top_n=args.summary_top_n,
        )
        all_results.append(result)

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(args.outdir / "all_runs_all_coefficients.tsv", sep="\t", index=False)

    # A complete long ranking across every run/allele and score.
    global_ranked = []
    for (_, _), group in combined.groupby(["run_label", "outcome"], sort=False):
        global_ranked.append(rank_mutations(group))
    if global_ranked:
        global_ranked_table = pd.concat(global_ranked, ignore_index=True)
        global_ranked_table.to_csv(
            args.outdir / "all_runs_ranked_mutations.tsv",
            sep="\t",
            index=False,
        )
        write_readable_html_table(
            global_ranked_table,
            args.outdir / "all_runs_ranked_mutations.html",
            "Ranked mutation coefficients across all runs",
        )

    # Cross-run / cross-allele comparison.
    wide_all = combined.pivot_table(
        index="mutation",
        columns=["run_label", "outcome"],
        values="boot_mean_coef",
        aggfunc="mean",
    )
    if not wide_all.empty:
        wide_all.columns = [f"{safe_name(a)}__{safe_name(b)}" for a, b in wide_all.columns]
        corr = plot_correlation_matrix(
            wide_all,
            args.outdir / "coefficient_correlation_across_runs_and_scores.png",
            title="Coefficient correlation across runs, alleles and scores",
        )
        if corr is not None:
            corr.to_csv(args.outdir / "coefficient_correlation_across_runs_and_scores.tsv", sep="\t")

    plot_allele_scatter(combined, args.outdir)
    print(f"[INFO] Done. Results written to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


"""
Example:

python analysis/mutation_immunogenicity_lr.py \
  --input data/output/variant_immunogenicity_scores_wt/VR4_K9_H2-Db/variant_immunogenicity_scores.tsv \
  --outdir data/output/mutation_immunogenicity_models/VR4_K9_H2-Db \
  --min-mutation-count 50 \
  --bootstraps 2 \
  --top-n 5 \
  --summary-top-n 20 \
  --position-offset 449 \
  --wt-sequence TINGSGQNQQTLKFSVAGPSN
"""