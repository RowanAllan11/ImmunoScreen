#!/usr/bin/env python3
"""Model mutation and physicochemical effects on allele-specific VR scores.

For every allele and requested score, this command fits

    score ~ recurrent mutation indicators

using a multivariable one-hot encoded OLS model with variant bootstrap
confidence intervals. It then compares mutation coefficients across alleles
and fits, for each score, the coefficient-level model

    median mutation effect across alleles ~ physicochemical changes + VR position.

The WT variable-region sequence is supplied at runtime
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from Bio.SeqUtils import IsoelectricPoint as IP
from matplotlib.patches import Rectangle


AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
MUTATION_PATTERN = re.compile(r"^([A-Z*])(\d+)([A-Z*])$")
HYDROPHOBICITY = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5,
    "M": 1.9, "A": 1.8, "G": -0.4, "T": -0.7, "S": -0.8,
    "W": -0.9, "Y": -1.3, "P": -1.6, "H": -3.2, "E": -3.5,
    "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5,
}
VOLUME = {
    "A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5,
    "Q": 143.8, "E": 138.4, "G": 60.1, "H": 153.2, "I": 166.7,
    "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9, "P": 112.7,
    "S": 89.0, "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0,
}
DEFAULT_OUTCOMES = [
    "netMHCpan_net_pass_change",
    "MHCflurry_net_pass_change",
    "netMHCpan_mean_window_improvement",
    "MHCflurry_mean_window_improvement",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        required=True,
        help=(
            "Allele-specific variant_immunogenicity_scores.tsv files or "
            "directories containing them. Combined-score directories are ignored."
        ),
    )
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--outcomes",
        nargs="+",
        default=DEFAULT_OUTCOMES,
        help="Score columns to model independently.",
    )
    parser.add_argument(
        "--wt-sequence",
        required=True,
        help="WT amino-acid sequence of the designated variable region.",
    )
    parser.add_argument(
        "--position-offset",
        type=int,
        default=0,
        help=(
            "Value added to VR-relative mutation positions for plot labels. "
            "For a VR beginning at absolute position N, use N - 1."
        ),
    )
    parser.add_argument("--min-mutation-count", type=int, default=20)
    parser.add_argument("--bootstraps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-sign-stability", type=float, default=0.8)
    args = parser.parse_args()
    if args.min_mutation_count < 1:
        parser.error("--min-mutation-count must be positive")
    if args.bootstraps < 1:
        parser.error("--bootstraps must be positive")
    if not 0 <= args.min_sign_stability <= 1:
        parser.error("--min-sign-stability must be between 0 and 1")
    args.wt_sequence = re.sub(r"\s+", "", args.wt_sequence).upper()
    invalid = sorted(set(args.wt_sequence) - set(AA_ORDER))
    if not args.wt_sequence or invalid:
        parser.error(
            "--wt-sequence must contain standard amino acids only"
            + (f"; invalid: {invalid}" if invalid else "")
        )
    return args


def safe_name(value: object) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).replace("*", ""))
    return cleaned.strip("_") or "unknown"


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def discover_score_files(inputs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in inputs:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(path.rglob("variant_immunogenicity_scores.tsv"))
        else:
            raise FileNotFoundError(path)
    unique = sorted({path.resolve() for path in files})
    unique = [path for path in unique if not path.parent.name.endswith("_combined")]
    if not unique:
        raise FileNotFoundError("No allele-specific score tables were found")
    return unique


def split_mutations(value: object) -> list[str]:
    if pd.isna(value) or str(value).strip().upper() in {"", "WT", "NAN"}:
        return []
    return [part.strip() for part in str(value).split(";") if part.strip()]


def parse_mutation(
    mutation: str, wt_sequence: str, position_offset: int
) -> dict[str, object]:
    match = MUTATION_PATTERN.fullmatch(str(mutation).upper())
    if not match:
        raise ValueError(f"Invalid mutation label: {mutation!r}")
    labelled_wt, position_text, mutant = match.groups()
    position = int(position_text)
    if not 1 <= position <= len(wt_sequence):
        raise ValueError(
            f"Mutation {mutation} is outside the supplied {len(wt_sequence)}-residue WT sequence"
        )
    expected_wt = wt_sequence[position - 1]
    if labelled_wt != expected_wt:
        raise ValueError(
            f"Mutation {mutation} conflicts with --wt-sequence: position {position} "
            f"is {expected_wt}, not {labelled_wt}"
        )
    if mutant not in AA_ORDER:
        raise ValueError(f"Mutation {mutation} has a non-standard mutant residue")
    return {
        "mutation": mutation,
        "wt_aa": labelled_wt,
        "mutant_aa": mutant,
        "position_relative": position,
        "position_absolute": position + position_offset,
    }


def side_chain_charge(amino_acid: str, ph: float = 7.4) -> float:
    if amino_acid in IP.positive_pKs:
        pka = IP.positive_pKs[amino_acid]
        return float(10**pka / (10**pka + 10**ph))
    if amino_acid in IP.negative_pKs:
        pka = IP.negative_pKs[amino_acid]
        return float(-(10**ph / (10**pka + 10**ph)))
    return 0.0


def mutation_properties(parsed: dict[str, object]) -> dict[str, float]:
    wt = str(parsed["wt_aa"])
    mutant = str(parsed["mutant_aa"])
    return {
        "hydrophobicity_change": HYDROPHOBICITY[mutant] - HYDROPHOBICITY[wt],
        "volume_change": VOLUME[mutant] - VOLUME[wt],
        "charge_change": side_chain_charge(mutant) - side_chain_charge(wt),
    }


def load_allele_scores(path: Path, outcomes: list[str]) -> tuple[str, pd.DataFrame]:
    data = pd.read_csv(path, sep="\t", low_memory=False)
    required = {"allele", "variant_id", "VR_mutation"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    alleles = data["allele"].dropna().astype(str).str.strip().unique()
    if len(alleles) != 1:
        raise ValueError(f"{path} must contain exactly one allele; found {list(alleles)}")
    available = [outcome for outcome in outcomes if outcome in data.columns]
    if not available:
        raise ValueError(f"{path} contains none of the requested outcomes: {outcomes}")
    data = data.drop_duplicates("variant_id").copy()
    data["VR_mutation"] = data["VR_mutation"].fillna("WT").astype(str)
    for outcome in available:
        data[outcome] = pd.to_numeric(data[outcome], errors="coerce")
    return str(alleles[0]), data


def mutation_design(
    data: pd.DataFrame, min_count: int
) -> tuple[pd.DataFrame, pd.Series]:
    mutation_lists = data["VR_mutation"].map(split_mutations)
    mutations = sorted({mutation for values in mutation_lists for mutation in values})
    if not mutations:
        raise ValueError("No mutations were found")
    design = pd.DataFrame(0, index=data.index, columns=mutations, dtype=np.int8)
    for index, values in mutation_lists.items():
        design.loc[index, list(set(values))] = 1
    counts = design.sum().sort_values(ascending=False)
    retained = counts[counts >= min_count].index
    if retained.empty:
        raise ValueError(
            f"No mutations passed --min-mutation-count {min_count}; maximum count was {int(counts.max())}"
        )
    return design.loc[:, retained].copy(), counts.loc[retained]


def fit_mutation_model(
    data: pd.DataFrame,
    design: pd.DataFrame,
    counts: pd.Series,
    allele: str,
    outcome: str,
    bootstraps: int,
    seed: int,
    wt_sequence: str,
    position_offset: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    response = data[outcome]
    valid = response.notna()
    x = design.loc[valid].astype(float).to_numpy()
    y = response.loc[valid].astype(float).to_numpy()
    if len(y) <= 1:
        raise ValueError(f"{allele}/{outcome} has fewer than two numeric observations")
    matrix = np.column_stack([np.ones(len(x)), x])
    fitted, *_ = np.linalg.lstsq(matrix, y, rcond=None)
    predictions = matrix @ fitted
    residual_sum_squares = float(np.sum((y - predictions) ** 2))
    total_sum_squares = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1 - residual_sum_squares / total_sum_squares if total_sum_squares else np.nan

    rng = np.random.default_rng(seed)
    bootstrap_values = np.empty((bootstraps, design.shape[1]), dtype=float)
    for index in range(bootstraps):
        selected = rng.integers(0, len(y), len(y))
        estimate, *_ = np.linalg.lstsq(matrix[selected], y[selected], rcond=None)
        bootstrap_values[index] = estimate[1:]

    rows: list[dict[str, object]] = []
    for column_index, mutation in enumerate(design.columns):
        values = bootstrap_values[:, column_index]
        mean = float(values.mean())
        parsed = parse_mutation(mutation, wt_sequence, position_offset)
        stability = float(np.mean(values > 0 if mean >= 0 else values < 0))
        rows.append({
            "allele": allele,
            "outcome": outcome,
            **parsed,
            **mutation_properties(parsed),
            "n_variants_with_mutation": int(counts[mutation]),
            "full_data_coefficient": float(fitted[column_index + 1]),
            "bootstrap_mean_coefficient": mean,
            "ci_low": float(np.percentile(values, 2.5)),
            "ci_high": float(np.percentile(values, 97.5)),
            "sign_stability": stability,
        })
    model = {
        "allele": allele,
        "outcome": outcome,
        "n_variants": len(y),
        "n_mutations": design.shape[1],
        "rank": int(np.linalg.matrix_rank(matrix)),
        "design_columns": matrix.shape[1],
        "r_squared": r_squared,
    }
    return pd.DataFrame(rows), model


def plot_mutation_heatmap(
    coefficients: pd.DataFrame,
    wt_sequence: str,
    min_sign_stability: float,
    path: Path,
) -> None:
    effects = np.full((len(AA_ORDER), len(wt_sequence)), np.nan)
    stable = np.zeros_like(effects, dtype=bool)
    for row in coefficients.itertuples(index=False):
        y = AA_ORDER.index(row.mutant_aa)
        x = int(row.position_relative) - 1
        effects[y, x] = row.bootstrap_mean_coefficient
        stable[y, x] = row.sign_stability >= min_sign_stability
    finite = np.abs(effects[np.isfinite(effects)])
    limit = float(np.percentile(finite, 98)) if finite.size else 1.0
    if not np.isfinite(limit) or limit == 0:
        limit = 1.0
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "mutation_effect", ["#2166ac", "#ffffff", "#b2182b"]
    )
    fig, ax = plt.subplots(figsize=(max(7, 0.55 * len(wt_sequence) + 2), 8))
    image = ax.imshow(
        np.ma.masked_invalid(effects), aspect="auto", cmap=cmap,
        norm=mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
    )
    uncertain = np.isfinite(effects) & ~stable
    overlay = np.zeros((*effects.shape, 4))
    overlay[uncertain] = mcolors.to_rgba("0.72")
    ax.imshow(overlay, aspect="auto")
    for position, wt in enumerate(wt_sequence):
        ax.add_patch(Rectangle(
            (position - 0.5, AA_ORDER.index(wt) - 0.5), 1, 1,
            facecolor="white", edgecolor="black", linewidth=0.6,
        ))
    ax.set_xticks(
        range(len(wt_sequence)),
        [f"{aa}{index}" for index, aa in enumerate(wt_sequence, start=1)],
        rotation=45,
        ha="right",
    )
    ax.set_yticks(range(len(AA_ORDER)), AA_ORDER)
    ax.set_xlabel("WT residue and VR-relative position")
    ax.set_ylabel("Mutant amino acid")
    allele = coefficients["allele"].iloc[0]
    outcome = coefficients["outcome"].iloc[0]
    ax.set_title(f"{allele} | {outcome}\nGrey: sign stability < {min_sign_stability:.0%}")
    fig.colorbar(image, ax=ax, label="Mutation coefficient")
    save_figure(fig, path)


def plot_correlation_heatmap(matrix: pd.DataFrame, title: str, path: Path) -> None:
    correlation = matrix.corr(method="pearson", min_periods=3)
    fig, ax = plt.subplots(figsize=(max(5, 0.8 * len(correlation) + 2), 5))
    image = ax.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(correlation)), correlation.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(correlation)), correlation.index)
    for y in range(len(correlation)):
        for x in range(len(correlation)):
            value = correlation.iloc[y, x]
            if pd.notna(value):
                ax.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="Pearson correlation")
    save_figure(fig, path)
    correlation.to_csv(path.with_suffix(".tsv"), sep="\t")


def allele_comparisons(coefficients: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    summaries: list[pd.DataFrame] = []
    for outcome, group in coefficients.groupby("outcome", sort=True):
        wide = group.pivot_table(
            index="mutation", columns="allele",
            values="bootstrap_mean_coefficient", aggfunc="mean",
        )
        if wide.shape[1] >= 2:
            plot_correlation_heatmap(
                wide,
                f"Allele correlation | {outcome}",
                outdir / "plots" / "allele_correlations" / f"{safe_name(outcome)}.png",
            )
        summary = group.groupby("mutation", as_index=False).agg(
            median_allele_coefficient=("bootstrap_mean_coefficient", "median"),
            q1=("bootstrap_mean_coefficient", lambda values: values.quantile(0.25)),
            q3=("bootstrap_mean_coefficient", lambda values: values.quantile(0.75)),
            allele_count=("allele", "nunique"),
            negative_alleles=(
                "bootstrap_mean_coefficient", lambda values: int((values < 0).sum())
            ),
            zero_alleles=(
                "bootstrap_mean_coefficient", lambda values: int((values == 0).sum())
            ),
            positive_alleles=(
                "bootstrap_mean_coefficient", lambda values: int((values > 0).sum())
            ),
        )
        summary["allele_effect_iqr"] = summary["q3"] - summary["q1"]
        summary["alleles_directionally_agreeing"] = summary[
            ["negative_alleles", "positive_alleles"]
        ].max(axis=1)
        summary["allele_directional_agreement"] = (
            summary["alleles_directionally_agreeing"] / summary["allele_count"]
        )
        summary.insert(0, "outcome", outcome)
        summaries.append(summary)
        if summary["allele_count"].max() >= 2:
            fig, ax = plt.subplots(figsize=(7, 5.5))
            points = ax.scatter(
                summary["median_allele_coefficient"], summary["allele_effect_iqr"],
                c=summary["allele_directional_agreement"], cmap="viridis",
                vmin=0.5, vmax=1.0, alpha=0.8, edgecolor="white", linewidth=0.35,
            )
            label_rows = summary.nlargest(min(5, len(summary)), "allele_effect_iqr")
            for row in label_rows.itertuples(index=False):
                ax.annotate(
                    row.mutation,
                    (row.median_allele_coefficient, row.allele_effect_iqr),
                    xytext=(4, 4), textcoords="offset points", fontsize=8,
                )
            ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
            ax.set_xlabel("Median mutation coefficient across alleles")
            ax.set_ylabel("Mutation coefficient IQR across alleles")
            ax.set_title(f"Allele-effect IQR | {outcome}")
            colorbar = fig.colorbar(points, ax=ax)
            colorbar.set_label("Allele directional agreement")
            colorbar.set_ticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            save_figure(
                fig, outdir / "plots" / "allele_iqr" / f"{safe_name(outcome)}.png"
            )
    result = pd.concat(summaries, ignore_index=True)
    result.to_csv(outdir / "tables" / "allele_effect_iqr.tsv", sep="\t", index=False)
    return result


def aggregate_mutation_effects(
    coefficients: pd.DataFrame, min_sign_stability: float
) -> pd.DataFrame:
    """Collapse allele-specific estimates to one row per mutation and score."""
    keys = [
        "outcome", "mutation", "wt_aa", "mutant_aa", "position_relative",
        "position_absolute", "hydrophobicity_change", "volume_change", "charge_change",
    ]
    data = coefficients.groupby(keys, as_index=False, dropna=False).agg(
        median_allele_coefficient=("bootstrap_mean_coefficient", "median"),
        allele_effect_q1=("bootstrap_mean_coefficient", lambda values: values.quantile(0.25)),
        allele_effect_q3=("bootstrap_mean_coefficient", lambda values: values.quantile(0.75)),
        alleles_evaluated=("allele", "nunique"),
        bootstrap_supported_alleles=(
            "sign_stability", lambda values: int((values >= min_sign_stability).sum())
        ),
    )
    data["allele_effect_iqr"] = data["allele_effect_q3"] - data["allele_effect_q1"]
    data["bootstrap_support_fraction"] = (
        data["bootstrap_supported_alleles"] / data["alleles_evaluated"]
    )
    return data


def physicochemical_design(
    group: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, dict[str, list[str]], list[int]]:
    continuous = ["hydrophobicity_change", "volume_change", "charge_change"]
    data = group.dropna(subset=["median_allele_coefficient", *continuous]).copy()
    design = pd.DataFrame(index=data.index)
    for column in continuous:
        design[column] = data[column].astype(float)
    positions = sorted(data["position_relative"].astype(int).unique())
    position_dummies = pd.get_dummies(
        data["position_relative"].astype(int), prefix="position", drop_first=True,
        dtype=float,
    )
    design = pd.concat([design, position_dummies], axis=1)
    design = sm.add_constant(design.astype(float), has_constant="add")
    groups = {
        "hydrophobicity": ["hydrophobicity_change"],
        "volume": ["volume_change"],
        "charge": ["charge_change"],
        "position": list(position_dummies.columns),
    }
    return design, data["median_allele_coefficient"].astype(float), groups, positions


def plot_partial_r_squared(table: pd.DataFrame, output_dir: Path) -> None:
    predictors = ["Hydrophobicity", "Charge", "Residue volume", "Position"]
    for outcome, group in table.groupby("outcome", sort=True):
        values = group.set_index("predictor")["partial_r_squared"].reindex(predictors)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(predictors, values, color=["#4477aa", "#66ccee", "#228833", "#cc6677"])
        ax.set_ylabel("Partial R²")
        ax.set_xlabel("Predictor")
        ax.set_title(f"Unique physicochemical and positional associations\n{outcome}")
        ax.tick_params(axis="x", rotation=25)
        ax.set_ylim(bottom=0)
        save_figure(fig, output_dir / f"{safe_name(outcome)}__partial_r_squared.png")


def plot_adjusted_positions(table: pd.DataFrame, path: Path) -> None:
    outcomes = list(dict.fromkeys(table["outcome"]))
    columns = 2 if len(outcomes) > 1 else 1
    rows = int(np.ceil(len(outcomes) / columns))
    fig, axes = plt.subplots(
        rows, columns, figsize=(7.2 * columns, 4.4 * rows), squeeze=False, sharex=True
    )
    for ax, outcome in zip(axes.flat, outcomes):
        data = table[table["outcome"].eq(outcome)].sort_values("position_absolute")
        effect = data["adjusted_position_effect"].to_numpy()
        ax.errorbar(
            data["position_absolute"], effect,
            yerr=np.vstack([effect - data["ci_low"], data["ci_high"] - effect]),
            fmt="o-", capsize=3, color="#356daf",
        )
        ax.axhline(0, color="0.45", linestyle="--", linewidth=0.8)
        ax.set_title(outcome)
        ax.set_ylabel("Adjusted position effect")
        ax.tick_params(axis="x", rotation=45)
    for ax in axes.flat[len(outcomes):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        if ax.get_visible():
            ax.set_xlabel(
                "VR-relative position"
                if table["position_absolute"].equals(table["position_relative"])
                else "Absolute position"
            )
    fig.suptitle("Adjusted positional effects across the designated VR")
    save_figure(fig, path)


def fit_physicochemical_models(
    coefficients: pd.DataFrame, outdir: Path, min_sign_stability: float
) -> None:
    model_data = aggregate_mutation_effects(coefficients, min_sign_stability)
    model_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    partial_rows: list[dict[str, object]] = []
    position_rows: list[dict[str, object]] = []
    for outcome, group in model_data.groupby("outcome", sort=True):
        design, response, predictor_groups, positions = physicochemical_design(group)
        if len(response) <= design.shape[1]:
            raise ValueError(
                f"{outcome}: physicochemical model has {len(response)} rows but "
                f"{design.shape[1]} parameters; provide more modelled mutations/alleles"
            )
        fit = sm.OLS(response, design).fit()
        model_rows.append({
            "outcome": outcome, "n_mutations": int(fit.nobs),
            "n_positions": len(positions),
            "r_squared": float(fit.rsquared),
            "adjusted_r_squared": float(fit.rsquared_adj),
            "aic": float(fit.aic), "bic": float(fit.bic),
        })
        for term in design.columns:
            coefficient_rows.append({
                "outcome": outcome, "term": term,
                "coefficient": float(fit.params[term]),
                "standard_error": float(fit.bse[term]),
                "p_value": float(fit.pvalues[term]),
                "ci_low": float(fit.conf_int().loc[term, 0]),
                "ci_high": float(fit.conf_int().loc[term, 1]),
            })
        full_sse = float(np.sum(fit.resid**2))
        predictor_labels = {
            "hydrophobicity": "Hydrophobicity", "charge": "Charge",
            "volume": "Residue volume", "position": "Position",
        }
        for predictor, columns in predictor_groups.items():
            if not columns:
                continue
            reduced = sm.OLS(response, design.drop(columns=columns)).fit()
            reduced_sse = float(np.sum(reduced.resid**2))
            partial = (reduced_sse - full_sse) / reduced_sse if reduced_sse else np.nan
            partial_rows.append({
                "outcome": outcome, "predictor": predictor_labels[predictor],
                "partial_r_squared": partial,
                "full_model_sse": full_sse, "reduced_model_sse": reduced_sse,
                "degrees_of_freedom": len(columns),
            })
        covariance = fit.cov_params()
        position_contrasts: dict[int, pd.Series] = {}
        for position in positions:
            contrast = pd.Series(0.0, index=design.columns)
            contrast["const"] = 1.0
            position_column = f"position_{position}"
            if position_column in contrast.index:
                contrast[position_column] = 1.0
            position_contrasts[position] = contrast
        mean_contrast = sum(position_contrasts.values()) / len(position_contrasts)
        for position, raw_contrast in position_contrasts.items():
            contrast = raw_contrast - mean_contrast
            estimate = float(contrast @ fit.params)
            variance = float(contrast @ covariance @ contrast)
            standard_error = np.sqrt(max(variance, 0))
            position_rows.append({
                "outcome": outcome,
                "position_relative": position,
                "position_absolute": int(
                    group.loc[group["position_relative"].eq(position), "position_absolute"].iloc[0]
                ),
                "adjusted_position_effect": estimate,
                "ci_low": estimate - 1.96 * standard_error,
                "ci_high": estimate + 1.96 * standard_error,
            })

    tables = outdir / "tables" / "physicochemical_models"
    plots = outdir / "plots" / "physicochemical_models"
    tables.mkdir(parents=True, exist_ok=True)
    model_table = pd.DataFrame(model_rows)
    coefficient_table = pd.DataFrame(coefficient_rows)
    partial_table = pd.DataFrame(partial_rows)
    position_table = pd.DataFrame(position_rows)
    model_data.to_csv(tables / "physicochemical_model_data.tsv", sep="\t", index=False)
    model_table.to_csv(tables / "model_summary.tsv", sep="\t", index=False)
    coefficient_table.to_csv(tables / "coefficient_summary.tsv", sep="\t", index=False)
    partial_table.to_csv(tables / "partial_r_squared.tsv", sep="\t", index=False)
    position_table.to_csv(tables / "adjusted_position_effects.tsv", sep="\t", index=False)

    plot_partial_r_squared(partial_table, plots)
    plot_adjusted_positions(position_table, plots / "adjusted_position_effects.png")


def main() -> int:
    args = parse_args()
    files = discover_score_files(args.inputs)
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "tables").mkdir(parents=True, exist_ok=True)
    all_coefficients: list[pd.DataFrame] = []
    model_rows: list[dict[str, object]] = []
    seen_alleles: set[str] = set()
    for file_index, path in enumerate(files):
        allele, data = load_allele_scores(path, args.outcomes)
        if allele in seen_alleles:
            raise ValueError(
                f"Allele {allele} occurs in multiple input files; supply one table per allele"
            )
        seen_alleles.add(allele)
        design, counts = mutation_design(data, args.min_mutation_count)
        available_outcomes = [outcome for outcome in args.outcomes if outcome in data.columns]
        for outcome_index, outcome in enumerate(available_outcomes):
            coefficients, model = fit_mutation_model(
                data=data, design=design, counts=counts, allele=allele,
                outcome=outcome, bootstraps=args.bootstraps,
                seed=args.seed + file_index * 1000 + outcome_index,
                wt_sequence=args.wt_sequence,
                position_offset=args.position_offset,
            )
            all_coefficients.append(coefficients)
            model_rows.append(model)
            allele_dir = args.outdir / "tables" / "mutation_models" / safe_name(allele)
            allele_dir.mkdir(parents=True, exist_ok=True)
            coefficients.to_csv(
                allele_dir / f"{safe_name(outcome)}__coefficients.tsv",
                sep="\t", index=False,
            )
            plot_mutation_heatmap(
                coefficients, args.wt_sequence, args.min_sign_stability,
                args.outdir / "plots" / "mutation_heatmaps" / safe_name(allele)
                / f"{safe_name(outcome)}.png",
            )
    combined = pd.concat(all_coefficients, ignore_index=True)
    combined.to_csv(
        args.outdir / "tables" / "all_mutation_coefficients.tsv",
        sep="\t", index=False,
    )
    pd.DataFrame(model_rows).to_csv(
        args.outdir / "tables" / "mutation_model_fit.tsv", sep="\t", index=False
    )
    allele_comparisons(combined, args.outdir)
    fit_physicochemical_models(combined, args.outdir, args.min_sign_stability)
    print(
        f"Complete: {len(seen_alleles)} alleles, "
        f"{combined['outcome'].nunique()} scores -> {args.outdir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
