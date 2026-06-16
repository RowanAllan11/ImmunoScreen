from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

RUN_LABEL = "VR5_V3__k9"

# Use this after BigMHC:
INPUT_TSV = Path(
    f"data/output/bigmhc/{RUN_LABEL}/predictions_mapped.tsv"
)

# Alternatively, use the combined table before BigMHC:
# INPUT_TSV = Path(
#     f"data/output/combined/{RUN_LABEL}/combined_annotated.tsv"
# )

OUTDIR = Path(
    f"data/output/descriptive_analysis/{RUN_LABEL}"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

PEPTIDE_COL = "peptide"
ALLELE_COL = "allele"
VARIANT_COL = "variant_id"
START_COL = "start"
END_COL = "end"
K_COL = "k"
MUTATION_COL = "peptide_mutation"

NET_PASS_COL = "netMHCpan_EL_rank_pass"
MHC_PASS_COL = "MHCflurry_affinity_percentile_pass"

NET_SCORE_COLS = [
    "netMHCpan_EL_score",
    "netMHCpan_EL_rank"
]

MHCFLURRY_SCORE_COLS = [
    "MHCflurry_affinity_percentile",
    "MHCflurry_presentation_percentile",
]

VR_START = 8
VR_END = 24

MAX_SCATTER_POINTS = 100_000
RANDOM_SEED = 42


# ============================================================
# HELPERS
# ============================================================

def parse_boolean_column(series: pd.Series) -> pd.Series:
    """
    Parse common Boolean representations safely.

    Returns a pandas nullable Boolean series.
    """
    true_values = {"true", "1", "yes", "y"}
    false_values = {"false", "0", "no", "n"}

    cleaned = (
        series
        .astype("string")
        .str.strip()
        .str.lower()
    )

    result = pd.Series(
        pd.NA,
        index=series.index,
        dtype="boolean",
    )

    result.loc[cleaned.isin(true_values)] = True
    result.loc[cleaned.isin(false_values)] = False

    return result


def safe_filename(value: str) -> str:
    """
    Convert a string to a filesystem-safe filename component.
    """
    return (
        str(value)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(">", "")
        .replace("<", "")
        .replace(":", "_")
    )


def count_unique_peptide_allele_pairs(
    data: pd.DataFrame,
) -> int:
    """
    Count unique allele-peptide prediction units.
    """
    if data.empty:
        return 0

    return (
        data[[ALLELE_COL, PEPTIDE_COL]]
        .drop_duplicates()
        .shape[0]
    )


def count_unique_mapped_occurrences(
    data: pd.DataFrame,
) -> int:
    """
    Count unique peptide-allele-variant-position occurrences.
    """
    keys = [
        ALLELE_COL,
        PEPTIDE_COL,
        VARIANT_COL,
        START_COL,
        END_COL,
        K_COL,
    ]

    available_keys = [
        col for col in keys
        if col in data.columns
    ]

    if not available_keys or data.empty:
        return 0

    return data[available_keys].drop_duplicates().shape[0]


def calculate_position_overlap(
    data: pd.DataFrame,
    *,
    deduplicate: bool,
) -> pd.DataFrame:
    """
    Count the number of rows or unique epitopes overlapping
    each residue position.

    When deduplicate=True, each unique:
        allele + peptide + start + end
    contributes once.
    """
    working = data.dropna(
        subset=[START_COL, END_COL, PEPTIDE_COL]
    ).copy()

    if deduplicate:
        working = working.drop_duplicates(
            subset=[
                ALLELE_COL,
                PEPTIDE_COL,
                START_COL,
                END_COL,
            ]
        )

    position_counts: dict[int, int] = {}

    for row in working.itertuples(index=False):
        start = int(getattr(row, START_COL))
        end = int(getattr(row, END_COL))

        for position in range(start, end + 1):
            position_counts[position] = (
                position_counts.get(position, 0) + 1
            )

    if not position_counts:
        return pd.DataFrame(
            columns=["position", "overlap_count"]
        )

    return (
        pd.DataFrame(
            {
                "position": list(position_counts.keys()),
                "overlap_count": list(position_counts.values()),
            }
        )
        .sort_values("position")
        .reset_index(drop=True)
    )


def save_position_plot(
    position_df: pd.DataFrame,
    *,
    output_path: Path,
    title: str,
    ylabel: str,
) -> None:
    """
    Save a bar plot of overlap counts by residue position.
    """
    if position_df.empty:
        return

    plt.figure(figsize=(12, 5))
    plt.bar(
        position_df["position"],
        position_df["overlap_count"],
    )
    plt.xlabel("Position")
    plt.ylabel(ylabel)
    plt.title(title)

    positions = position_df["position"].astype(int)
    plt.xticks(positions)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def parse_peptide_mutations(
    row: pd.Series,
    mutation_pattern: re.Pattern[str],
) -> list[dict[str, object]]:
    """
    Parse mutation strings such as:
        N8G
        S1L;T3N;N8G
        WT

    Mutation positions are assumed to be relative to the
    variable region.
    """
    mutation_string = row.get(MUTATION_COL, np.nan)

    if pd.isna(mutation_string):
        return []

    mutation_string = str(mutation_string).strip()

    if not mutation_string:
        return []

    if mutation_string.upper() == "WT":
        return []

    mutations: list[dict[str, object]] = []

    for match in mutation_pattern.finditer(mutation_string):
        ref_aa = match.group(1)
        vr_relative_position = int(match.group(2))
        alt_aa = match.group(3)

        absolute_position = (
            VR_START + vr_relative_position - 1
        )

        if (
            absolute_position < VR_START
            or absolute_position > VR_END
        ):
            continue

        mutations.append(
            {
                "variant_id": row.get(VARIANT_COL, np.nan),
                "allele": row.get(ALLELE_COL, np.nan),
                "peptide": row.get(PEPTIDE_COL, np.nan),
                "peptide_start": row.get(START_COL, np.nan),
                "peptide_end": row.get(END_COL, np.nan),
                "mutation": (
                    f"{ref_aa}"
                    f"{vr_relative_position}"
                    f"{alt_aa}"
                ),
                "ref_aa": ref_aa,
                "alt_aa": alt_aa,
                "vr_relative_position": (
                    vr_relative_position
                ),
                "absolute_position": absolute_position,
                "passes_netmhcpan": row.get(
                    "passes_netmhcpan",
                    pd.NA,
                ),
                "passes_mhcflurry": row.get(
                    "passes_mhcflurry",
                    pd.NA,
                ),
                "passes_both": row.get(
                    "passes_both",
                    pd.NA,
                ),
                "passes_either": row.get(
                    "passes_either",
                    pd.NA,
                ),
            }
        )

    return mutations


def write_population_position_outputs(
    analysis_sets: dict[str, pd.DataFrame],
) -> None:
    """
    Write mapped and unique position-overlap summaries
    for each analysis population.
    """
    position_dir = OUTDIR / "position_analysis"
    position_dir.mkdir(exist_ok=True)

    population_titles = {
        "all_candidates": "All evaluated candidates",
        "netmhcpan_pass": "Candidates passing NetMHCpan",
        "mhcflurry_pass": "Candidates passing MHCflurry",
        "both_pass": "Candidates passing both predictors",
        "either_pass": "Candidates passing either predictor",
        "neither_pass": "Candidates passing neither predictor",
    }

    for population, subset in analysis_sets.items():
        label = population_titles.get(
            population,
            population,
        )

        mapped_overlap = calculate_position_overlap(
            subset,
            deduplicate=False,
        )

        unique_overlap = calculate_position_overlap(
            subset,
            deduplicate=True,
        )

        mapped_overlap.to_csv(
            position_dir
            / f"{population}_mapped_overlap_by_position.tsv",
            sep="\t",
            index=False,
        )

        unique_overlap.to_csv(
            position_dir
            / f"{population}_unique_overlap_by_position.tsv",
            sep="\t",
            index=False,
        )

        save_position_plot(
            mapped_overlap,
            output_path=(
                position_dir
                / f"{population}_mapped_overlap_by_position.png"
            ),
            title=(
                f"{label}: mapped candidate occurrences "
                "across positions"
            ),
            ylabel="Number of mapped candidate rows",
        )

        save_position_plot(
            unique_overlap,
            output_path=(
                position_dir
                / f"{population}_unique_overlap_by_position.png"
            ),
            title=(
                f"{label}: unique peptide–allele candidates "
                "across positions"
            ),
            ylabel="Number of unique overlapping candidates",
        )


# ============================================================
# LOAD DATA
# ============================================================

if not INPUT_TSV.is_file():
    raise FileNotFoundError(
        f"Input TSV not found: {INPUT_TSV}"
    )

df = pd.read_csv(
    INPUT_TSV,
    sep="\t",
    low_memory=False,
)

df.columns = (
    df.columns
    .str.strip()
    .str.replace(" ", "_")
    .str.replace(">", "", regex=False)
)

print("Loaded dataframe:")
print(f"  Rows: {len(df):,}")
print(f"  Columns: {len(df.columns):,}")
print(df.columns.tolist())


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_cols = {
    PEPTIDE_COL,
    ALLELE_COL,
    START_COL,
    END_COL,
    NET_PASS_COL,
    MHC_PASS_COL,
}

missing_required = required_cols - set(df.columns)

if missing_required:
    raise ValueError(
        "Input table is missing required columns: "
        f"{sorted(missing_required)}"
    )


# ============================================================
# BASIC CLEANING
# ============================================================

df[START_COL] = pd.to_numeric(
    df[START_COL],
    errors="coerce",
)

df[END_COL] = pd.to_numeric(
    df[END_COL],
    errors="coerce",
)

if K_COL in df.columns:
    df[K_COL] = pd.to_numeric(
        df[K_COL],
        errors="coerce",
    )

df[PEPTIDE_COL] = (
    df[PEPTIDE_COL]
    .astype("string")
    .str.strip()
)

df[ALLELE_COL] = (
    df[ALLELE_COL]
    .astype("string")
    .str.strip()
)

if VARIANT_COL in df.columns:
    df[VARIANT_COL] = (
        df[VARIANT_COL]
        .astype("string")
        .str.strip()
    )

df[NET_PASS_COL] = parse_boolean_column(
    df[NET_PASS_COL]
)

df[MHC_PASS_COL] = parse_boolean_column(
    df[MHC_PASS_COL]
)

missing_pass_annotations = df[
    [NET_PASS_COL, MHC_PASS_COL]
].isna().sum()

if missing_pass_annotations.any():
    raise ValueError(
        "Missing or unrecognised pass annotations detected:\n"
        f"{missing_pass_annotations.to_string()}"
    )

df_pos = df.dropna(
    subset=[
        START_COL,
        END_COL,
        PEPTIDE_COL,
        ALLELE_COL,
    ]
).copy()

df_pos[START_COL] = (
    df_pos[START_COL]
    .astype(int)
)

df_pos[END_COL] = (
    df_pos[END_COL]
    .astype(int)
)

if K_COL in df_pos.columns:
    df_pos[K_COL] = (
        df_pos[K_COL]
        .astype("Int64")
    )

if MUTATION_COL in df_pos.columns:
    df_pos[MUTATION_COL] = (
        df_pos[MUTATION_COL]
        .astype("string")
        .str.strip()
        .replace(
            {
                "": pd.NA,
                "nan": pd.NA,
                "NaN": pd.NA,
            }
        )
    )
else:
    print(
        f"Warning: {MUTATION_COL} not found. "
        "Mutation analyses will be empty."
    )
    df_pos[MUTATION_COL] = pd.NA


# ============================================================
# DEFINE PASS POPULATIONS
# ============================================================

df_pos["passes_netmhcpan"] = (
    df_pos[NET_PASS_COL].astype("boolean")
)

df_pos["passes_mhcflurry"] = (
    df_pos[MHC_PASS_COL].astype("boolean")
)

df_pos["passes_both"] = (
    df_pos["passes_netmhcpan"]
    & df_pos["passes_mhcflurry"]
)

df_pos["passes_either"] = (
    df_pos["passes_netmhcpan"]
    | df_pos["passes_mhcflurry"]
)

df_pos["passes_neither"] = (
    ~df_pos["passes_netmhcpan"]
    & ~df_pos["passes_mhcflurry"]
)

analysis_sets = {
    "all_candidates": df_pos,
    "netmhcpan_pass": (
        df_pos[df_pos["passes_netmhcpan"]].copy()
    ),
    "mhcflurry_pass": (
        df_pos[df_pos["passes_mhcflurry"]].copy()
    ),
    "both_pass": (
        df_pos[df_pos["passes_both"]].copy()
    ),
    "either_pass": (
        df_pos[df_pos["passes_either"]].copy()
    ),
    "neither_pass": (
        df_pos[df_pos["passes_neither"]].copy()
    ),
}


# ============================================================
# 1. POPULATION SUMMARY
# ============================================================

summary_records: list[dict[str, object]] = []

for population, subset in analysis_sets.items():
    summary_records.append(
        {
            "population": population,
            "mapped_rows": len(subset),
            "unique_variants": (
                subset[VARIANT_COL].nunique()
                if VARIANT_COL in subset.columns
                else np.nan
            ),
            "unique_peptides": (
                subset[PEPTIDE_COL].nunique()
            ),
            "unique_peptide_allele_pairs": (
                count_unique_peptide_allele_pairs(
                    subset
                )
            ),
            "unique_mapped_occurrences": (
                count_unique_mapped_occurrences(
                    subset
                )
            ),
            "unique_alleles": (
                subset[ALLELE_COL].nunique()
            ),
            "percent_of_all_rows": (
                100 * len(subset) / len(df_pos)
                if len(df_pos)
                else np.nan
            ),
        }
    )

summary_df = pd.DataFrame(summary_records)

summary_df.to_csv(
    OUTDIR / "prediction_population_summary.tsv",
    sep="\t",
    index=False,
)

print("\nPrediction population summary:")
print(summary_df.to_string(index=False))


# ============================================================
# 2. PASS AGREEMENT BETWEEN PREDICTORS
# ============================================================

row_agreement = pd.crosstab(
    df_pos["passes_netmhcpan"],
    df_pos["passes_mhcflurry"],
    rownames=["NetMHCpan_pass"],
    colnames=["MHCflurry_pass"],
    dropna=False,
)

row_agreement.to_csv(
    OUTDIR / "predictor_pass_agreement_rows.tsv",
    sep="\t",
)

pair_pass = (
    df_pos[
        [
            ALLELE_COL,
            PEPTIDE_COL,
            "passes_netmhcpan",
            "passes_mhcflurry",
        ]
    ]
    .drop_duplicates()
)

pair_agreement = pd.crosstab(
    pair_pass["passes_netmhcpan"],
    pair_pass["passes_mhcflurry"],
    rownames=["NetMHCpan_pass"],
    colnames=["MHCflurry_pass"],
    dropna=False,
)

pair_agreement.to_csv(
    OUTDIR
    / "predictor_pass_agreement_unique_pairs.tsv",
    sep="\t",
)

print("\nPass agreement across mapped rows:")
print(row_agreement)

print("\nPass agreement across unique peptide–allele pairs:")
print(pair_agreement)


# ============================================================
# 3. PASS RATES BY ALLELE
# ============================================================

allele_summary = (
    df_pos
    .groupby(ALLELE_COL, dropna=False)
    .agg(
        mapped_rows=(PEPTIDE_COL, "size"),
        unique_peptides=(PEPTIDE_COL, "nunique"),
        netmhcpan_pass_rate=(
            "passes_netmhcpan",
            "mean",
        ),
        mhcflurry_pass_rate=(
            "passes_mhcflurry",
            "mean",
        ),
        both_pass_rate=(
            "passes_both",
            "mean",
        ),
        either_pass_rate=(
            "passes_either",
            "mean",
        ),
    )
    .reset_index()
)

for col in [
    "netmhcpan_pass_rate",
    "mhcflurry_pass_rate",
    "both_pass_rate",
    "either_pass_rate",
]:
    allele_summary[col] = (
        100 * allele_summary[col]
    )

allele_summary.to_csv(
    OUTDIR / "pass_rates_by_allele.tsv",
    sep="\t",
    index=False,
)


# ============================================================
# 4. POSITIONAL DISTRIBUTIONS
# ============================================================

write_population_position_outputs(
    analysis_sets
)


# ============================================================
# 5. PARSE PEPTIDE MUTATIONS
# ============================================================

mutation_pattern = re.compile(
    r"([A-Z])(\d+)([A-Z])"
)

mutation_records: list[dict[str, object]] = []

for _, row in df_pos.iterrows():
    mutation_records.extend(
        parse_peptide_mutations(
            row,
            mutation_pattern,
        )
    )

mut_df = pd.DataFrame(mutation_records)

mutation_dir = OUTDIR / "mutation_analysis"
mutation_dir.mkdir(exist_ok=True)

if mut_df.empty:
    print("\nNo peptide mutations detected.")
else:
    mut_df.to_csv(
        mutation_dir / "parsed_peptide_mutations.tsv",
        sep="\t",
        index=False,
    )

    print("\nParsed mutation table:")
    print(mut_df.head().to_string(index=False))


# ============================================================
# 6. VARIANT-LEVEL MUTATION FREQUENCIES
# ============================================================

if not mut_df.empty:
    variant_mutations = (
        mut_df
        .drop_duplicates(
            subset=[
                VARIANT_COL,
                "mutation",
            ]
        )
    )

    variant_mutation_position_freq = (
        variant_mutations
        .groupby("absolute_position")
        .size()
        .reset_index(
            name="variant_mutation_count"
        )
        .sort_values("absolute_position")
    )

    variant_mutation_position_freq.to_csv(
        mutation_dir
        / "variant_mutation_frequency_by_position.tsv",
        sep="\t",
        index=False,
    )

    plt.figure(figsize=(12, 5))
    plt.bar(
        variant_mutation_position_freq[
            "absolute_position"
        ],
        variant_mutation_position_freq[
            "variant_mutation_count"
        ],
    )
    plt.xlabel("Absolute position")
    plt.ylabel("Number of variants")
    plt.title(
        "Variant-level mutation frequency across positions"
    )
    plt.xticks(
        variant_mutation_position_freq[
            "absolute_position"
        ].astype(int)
    )
    plt.tight_layout()
    plt.savefig(
        mutation_dir
        / "variant_mutation_frequency_by_position.png",
        dpi=300,
    )
    plt.close()

    variant_mutation_freq = (
        variant_mutations
        .groupby("mutation")
        .size()
        .reset_index(name="variant_count")
        .sort_values(
            "variant_count",
            ascending=False,
        )
    )

    variant_mutation_freq.to_csv(
        mutation_dir
        / "variant_mutation_frequency.tsv",
        sep="\t",
        index=False,
    )

    top_n = 30
    top_variant_mutations = (
        variant_mutation_freq.head(top_n)
    )

    plt.figure(figsize=(10, 8))
    plt.barh(
        top_variant_mutations[
            "mutation"
        ][::-1],
        top_variant_mutations[
            "variant_count"
        ][::-1],
    )
    plt.xlabel("Number of variants")
    plt.ylabel("Mutation")
    plt.title(
        f"Top {top_n} variant-level mutations"
    )
    plt.tight_layout()
    plt.savefig(
        mutation_dir
        / "top_variant_mutation_frequencies.png",
        dpi=300,
    )
    plt.close()


# ============================================================
# 7. EPITOPE-ASSOCIATED MUTATION FREQUENCIES
# ============================================================

if not mut_df.empty:
    epitope_mutations = (
        mut_df
        .drop_duplicates(
            subset=[
                VARIANT_COL,
                ALLELE_COL,
                PEPTIDE_COL,
                "peptide_start",
                "peptide_end",
                "mutation",
            ]
        )
    )

    epitope_mutation_freq = (
        epitope_mutations
        .groupby("mutation")
        .size()
        .reset_index(
            name=(
                "peptide_allele_occurrence_count"
            )
        )
        .sort_values(
            "peptide_allele_occurrence_count",
            ascending=False,
        )
    )

    epitope_mutation_freq.to_csv(
        mutation_dir
        / "epitope_associated_mutation_frequency.tsv",
        sep="\t",
        index=False,
    )

    mutation_population_records = []

    mutation_population_masks = {
        "all_candidates": pd.Series(
            True,
            index=epitope_mutations.index,
        ),
        "netmhcpan_pass": (
            epitope_mutations[
                "passes_netmhcpan"
            ].fillna(False)
        ),
        "mhcflurry_pass": (
            epitope_mutations[
                "passes_mhcflurry"
            ].fillna(False)
        ),
        "both_pass": (
            epitope_mutations[
                "passes_both"
            ].fillna(False)
        ),
        "either_pass": (
            epitope_mutations[
                "passes_either"
            ].fillna(False)
        ),
    }

    for population, mask in (
        mutation_population_masks.items()
    ):
        subset = epitope_mutations.loc[mask]

        grouped = (
            subset
            .groupby("mutation")
            .size()
            .reset_index(name="count")
        )

        grouped["population"] = population
        mutation_population_records.append(grouped)

    mutation_population_df = pd.concat(
        mutation_population_records,
        ignore_index=True,
    )

    mutation_population_df.to_csv(
        mutation_dir
        / "mutation_frequency_by_prediction_population.tsv",
        sep="\t",
        index=False,
    )


# ============================================================
# 8. SELECT CONTINUOUS PREDICTOR COLUMNS
# ============================================================

mhcflurry_cols = [
    col for col in MHCFLURRY_SCORE_COLS
    if col in df_pos.columns
]

netmhcpan_cols = [
    col for col in NET_SCORE_COLS
    if col in df_pos.columns
]

print("\nContinuous MHCflurry columns:")
print(mhcflurry_cols)

print("\nContinuous NetMHCpan columns:")
print(netmhcpan_cols)

for col in mhcflurry_cols + netmhcpan_cols:
    df_pos[col] = pd.to_numeric(
        df_pos[col],
        errors="coerce",
    )


# ============================================================
# 9. UNIQUE PEPTIDE–ALLELE CORRELATION DATA
# ============================================================

correlation_columns = (
    [ALLELE_COL, PEPTIDE_COL]
    + mhcflurry_cols
    + netmhcpan_cols
)

correlation_data = (
    df_pos[correlation_columns]
    .drop_duplicates(
        subset=[
            ALLELE_COL,
            PEPTIDE_COL,
        ]
    )
)

print(
    "\nUnique peptide–allele pairs used for "
    f"correlation: {len(correlation_data):,}"
)


# ============================================================
# 10. PREDICTOR CORRELATIONS
# ============================================================

correlation_records: list[dict[str, object]] = []

for m_col in mhcflurry_cols:
    for n_col in netmhcpan_cols:
        temp = correlation_data[
            [m_col, n_col]
        ].dropna()

        n_complete = len(temp)

        if n_complete < 3:
            pearson_r = np.nan
            spearman_r = np.nan
        else:
            pearson_r = temp[m_col].corr(
                temp[n_col],
                method="pearson",
            )

            spearman_r = temp[m_col].corr(
                temp[n_col],
                method="spearman",
            )

        correlation_records.append(
            {
                "mhcflurry_column": m_col,
                "netmhcpan_column": n_col,
                "n_complete_unique_pairs": n_complete,
                "pearson_r": pearson_r,
                "spearman_r": spearman_r,
            }
        )

correlation_df = pd.DataFrame(
    correlation_records
)

correlation_df.to_csv(
    OUTDIR
    / "unique_pair_mhcflurry_netmhcpan_correlations.tsv",
    sep="\t",
    index=False,
)

print("\nMHCflurry versus NetMHCpan correlations:")
print(correlation_df.to_string(index=False))


# ============================================================
# 11. CORRELATION HEATMAP
# ============================================================

if (
    not correlation_df.empty
    and mhcflurry_cols
    and netmhcpan_cols
):
    heatmap_data = correlation_df.pivot(
        index="mhcflurry_column",
        columns="netmhcpan_column",
        values="spearman_r",
    )

    plt.figure(
        figsize=(
            1.8 * len(netmhcpan_cols) + 4,
            1.2 * len(mhcflurry_cols) + 3,
        )
    )

    plt.imshow(
        heatmap_data,
        aspect="auto",
        vmin=-1,
        vmax=1,
    )

    plt.xticks(
        ticks=np.arange(
            len(heatmap_data.columns)
        ),
        labels=heatmap_data.columns,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        ticks=np.arange(
            len(heatmap_data.index)
        ),
        labels=heatmap_data.index,
    )

    plt.colorbar(
        label="Spearman correlation"
    )

    plt.title(
        "MHCflurry versus NetMHCpan correlations\n"
        "Unique peptide–allele pairs"
    )

    for i in range(heatmap_data.shape[0]):
        for j in range(
            heatmap_data.shape[1]
        ):
            value = heatmap_data.iloc[i, j]

            if pd.notna(value):
                plt.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                )

    plt.tight_layout()
    plt.savefig(
        OUTDIR
        / "unique_pair_mhcflurry_netmhcpan_spearman_heatmap.png",
        dpi=300,
    )
    plt.close()


# ============================================================
# 12. SCATTER PLOTS
# ============================================================

scatter_dir = (
    OUTDIR
    / "mhcflurry_netmhcpan_scatterplots"
)
scatter_dir.mkdir(exist_ok=True)

for m_col in mhcflurry_cols:
    for n_col in netmhcpan_cols:
        temp = correlation_data[
            [m_col, n_col]
        ].dropna()

        if len(temp) < 3:
            continue

        if len(temp) > MAX_SCATTER_POINTS:
            plot_data = temp.sample(
                MAX_SCATTER_POINTS,
                random_state=RANDOM_SEED,
            )
        else:
            plot_data = temp

        plt.figure(figsize=(6, 5))
        plt.scatter(
            plot_data[m_col],
            plot_data[n_col],
            alpha=0.25,
            s=8,
        )
        plt.xlabel(m_col)
        plt.ylabel(n_col)
        plt.title(
            f"{m_col} vs {n_col}\n"
            f"Unique peptide–allele pairs "
            f"(plotted n={len(plot_data):,})"
        )
        plt.tight_layout()

        safe_m = safe_filename(m_col)
        safe_n = safe_filename(n_col)

        plt.savefig(
            scatter_dir
            / f"{safe_m}_vs_{safe_n}.png",
            dpi=300,
        )
        plt.close()


# ============================================================
# 13. SCORE DISTRIBUTIONS BY PASS STATUS
# ============================================================

score_distribution_dir = (
    OUTDIR / "score_distributions"
)
score_distribution_dir.mkdir(exist_ok=True)

score_pass_pairs = [
    (
        "netMHCpan_EL_rank",
        "passes_netmhcpan",
    ),
    (
        "MHCflurry_affinity_percentile",
        "passes_mhcflurry",
    ),
]

for score_col, pass_col in score_pass_pairs:
    if score_col not in df_pos.columns:
        continue

    score_data = (
        df_pos[
            [
                ALLELE_COL,
                PEPTIDE_COL,
                score_col,
                pass_col,
            ]
        ]
        .drop_duplicates(
            subset=[
                ALLELE_COL,
                PEPTIDE_COL,
            ]
        )
        .dropna(
            subset=[score_col, pass_col]
        )
    )

    score_data.to_csv(
        score_distribution_dir
        / f"{safe_filename(score_col)}_by_pass_status.tsv",
        sep="\t",
        index=False,
    )

    passed = score_data.loc[
        score_data[pass_col],
        score_col,
    ]

    failed = score_data.loc[
        ~score_data[pass_col],
        score_col,
    ]

    plt.figure(figsize=(8, 5))

    if not failed.empty:
        plt.hist(
            failed,
            bins=50,
            alpha=0.5,
            label="Failed",
        )

    if not passed.empty:
        plt.hist(
            passed,
            bins=50,
            alpha=0.5,
            label="Passed",
        )

    plt.xlabel(score_col)
    plt.ylabel("Unique peptide–allele pairs")
    plt.title(
        f"{score_col} distribution by pass status"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        score_distribution_dir
        / f"{safe_filename(score_col)}_by_pass_status.png",
        dpi=300,
    )
    plt.close()


print(
    f"\nDone. Outputs saved to: {OUTDIR}"
)