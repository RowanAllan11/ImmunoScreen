from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# CONFIG
# -----------------------------

INPUT_TSV = Path("data/output/combined/VR5_v3_9mer_combined_netmhcpan_mhcflurry_overall.tsv")
OUTDIR = Path("data/output/descriptive_analysis")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Change these if your column names differ
PEPTIDE_COL = "peptide"
START_COL = "start"
END_COL = "end"
MUTATION_COL = "peptide_mutation"


# -----------------------------
# LOAD DATA
# -----------------------------

df = pd.read_csv(INPUT_TSV, sep="\t")

# Clean column names
df.columns = (
    df.columns
    .str.strip()
    .str.replace(" ", "_")
    .str.replace(">", "", regex=False)
)

print("Loaded dataframe:")
print(df.shape)
print(df.columns.tolist())


# -----------------------------
# BASIC CLEANING
# -----------------------------

# Make sure position columns are numeric
df[START_COL] = pd.to_numeric(df[START_COL], errors="coerce")
df[END_COL] = pd.to_numeric(df[END_COL], errors="coerce")

# Drop rows missing essential positional information
df_pos = df.dropna(subset=[START_COL, END_COL, PEPTIDE_COL]).copy()

df_pos[START_COL] = df_pos[START_COL].astype(int)
df_pos[END_COL] = df_pos[END_COL].astype(int)

# Standardise peptide mutation column
if MUTATION_COL in df_pos.columns:
    df_pos[MUTATION_COL] = (
        df_pos[MUTATION_COL]
        .astype(str)
        .str.strip()
        .replace({"nan": np.nan, "": np.nan})
    )
else:
    print(f"Warning: {MUTATION_COL} not found.")
    df_pos[MUTATION_COL] = np.nan


# -----------------------------
# 1. NUMBER OF UNIQUE PEPTIDES
# -----------------------------

n_unique_peptides = df_pos[PEPTIDE_COL].nunique()
n_unique_peptide_allele_pairs = df_pos[["allele", PEPTIDE_COL]].drop_duplicates().shape[0]

summary = pd.DataFrame({
    "metric": [
        "total_rows",
        "unique_variants",
        "unique_peptides",
        "unique_peptide_allele_pairs",
        "unique_alleles"
    ],
    "value": [
        len(df_pos),
        df_pos["variant_id"].nunique() if "variant_id" in df_pos.columns else np.nan,
        n_unique_peptides,
        n_unique_peptide_allele_pairs,
        df_pos["allele"].nunique() if "allele" in df_pos.columns else np.nan
    ]
})

summary.to_csv(OUTDIR / "basic_summary.tsv", sep="\t", index=False)

print("\nBasic summary:")
print(summary)


# -----------------------------
# 2. DISTRIBUTION OF EPITOPES ACROSS POSITIONS
# -----------------------------
# This counts how many epitope rows overlap each residue position.
# If the same peptide appears in many variants, it contributes multiple times.
# This is useful for seeing recurrent epitope burden across the region.

position_counts = {}

for _, row in df_pos.iterrows():
    start = int(row[START_COL])
    end = int(row[END_COL])

    for pos in range(start, end + 1):
        position_counts[pos] = position_counts.get(pos, 0) + 1

epitope_position_df = (
    pd.DataFrame({
        "position": list(position_counts.keys()),
        "epitope_overlap_count": list(position_counts.values())
    })
    .sort_values("position")
)

epitope_position_df.to_csv(
    OUTDIR / "epitope_overlap_by_position.tsv",
    sep="\t",
    index=False
)

plt.figure(figsize=(12, 5))
plt.bar(
    epitope_position_df["position"],
    epitope_position_df["epitope_overlap_count"]
)
plt.xlabel("Position")
plt.ylabel("Number of overlapping epitope rows")
plt.title("Distribution of predicted epitopes across positions")
plt.tight_layout()
plt.savefig(OUTDIR / "epitope_overlap_by_position.png", dpi=300)
plt.close()


# -----------------------------
# 3. DISTRIBUTION OF UNIQUE EPITOPES ACROSS POSITIONS
# -----------------------------
# This avoids overcounting the same peptide duplicated across variants.
# Unit: unique allele + peptide + start + end.

unique_epitopes = df_pos.drop_duplicates(
    subset=["allele", PEPTIDE_COL, START_COL, END_COL]
).copy()

unique_position_counts = {}

for _, row in unique_epitopes.iterrows():
    start = int(row[START_COL])
    end = int(row[END_COL])

    for pos in range(start, end + 1):
        unique_position_counts[pos] = unique_position_counts.get(pos, 0) + 1

unique_epitope_position_df = (
    pd.DataFrame({
        "position": list(unique_position_counts.keys()),
        "unique_epitope_overlap_count": list(unique_position_counts.values())
    })
    .sort_values("position")
)

unique_epitope_position_df.to_csv(
    OUTDIR / "unique_epitope_overlap_by_position.tsv",
    sep="\t",
    index=False
)

plt.figure(figsize=(12, 5))
plt.bar(
    unique_epitope_position_df["position"],
    unique_epitope_position_df["unique_epitope_overlap_count"]
)
plt.xlabel("Position")
plt.ylabel("Number of unique overlapping epitopes")
plt.title("Distribution of unique predicted epitopes across positions")
plt.tight_layout()
plt.savefig(OUTDIR / "unique_epitope_overlap_by_position.png", dpi=300)
plt.close()


# -----------------------------
# 4. PARSE PEPTIDE MUTATIONS
# -----------------------------
# Handles mutation strings like:
#   N8G
#   S1L;T3N;N8G
#   WT
#
# IMPORTANT:
# peptide_mutation positions are relative to the variable region (VR),
# not relative to the peptide start.
#
# VR region covers positions 8-24 in the 38 aa sequence.
# Therefore:
#   V1F  -> absolute position 8
#   N8G  -> absolute position 15
#   A17T -> absolute position 24

VR_START = 8
VR_END = 24

mutation_pattern = re.compile(r"([A-Z])(\d+)([A-Z])")


def parse_peptide_mutations(row):
    mutation_string = row[MUTATION_COL]

    if pd.isna(mutation_string):
        return []

    mutation_string = str(mutation_string).strip()

    if mutation_string.upper() == "WT":
        return []

    mutations = []

    for match in mutation_pattern.finditer(mutation_string):
        ref_aa = match.group(1)
        vr_relative_pos = int(match.group(2))
        alt_aa = match.group(3)

        absolute_pos = VR_START + vr_relative_pos - 1

        # Safety check: keep only mutations that map inside VR
        if absolute_pos < VR_START or absolute_pos > VR_END:
            continue

        mutations.append({
            "variant_id": row.get("variant_id", np.nan),
            "allele": row.get("allele", np.nan),
            "peptide": row.get(PEPTIDE_COL, np.nan),
            "peptide_start": row[START_COL],
            "peptide_end": row[END_COL],
            "mutation": f"{ref_aa}{vr_relative_pos}{alt_aa}",
            "ref_aa": ref_aa,
            "alt_aa": alt_aa,
            "vr_relative_position": vr_relative_pos,
            "absolute_position": absolute_pos
        })

    return mutations

mutation_records = []

for _, row in df_pos.iterrows():
    mutation_records.extend(parse_peptide_mutations(row))

mut_df = pd.DataFrame(mutation_records)

if mut_df.empty:
    print("\nNo peptide mutations detected.")
else:
    mut_df.to_csv(OUTDIR / "parsed_peptide_mutations.tsv", sep="\t", index=False)

    print("\nParsed mutation table:")
    print(mut_df.head())

# -----------------------------
# 5. FREQUENCY OF MUTATIONS ACROSS POSITIONS
# -----------------------------

if not mut_df.empty:
    mutation_position_freq = (
        mut_df.groupby("absolute_position")
        .size()
        .reset_index(name="mutation_count")
        .sort_values("absolute_position")
    )

    mutation_position_freq.to_csv(
        OUTDIR / "mutation_frequency_by_position.tsv",
        sep="\t",
        index=False
    )

    plt.figure(figsize=(12, 5))
    plt.bar(
        mutation_position_freq["absolute_position"],
        mutation_position_freq["mutation_count"]
    )
    plt.xlabel("Absolute position")
    plt.ylabel("Mutation count")
    plt.title("Frequency of peptide mutations across positions")

    positions = mutation_position_freq["absolute_position"].astype(int)
    plt.xticks(positions)

    plt.tight_layout()
    plt.savefig(OUTDIR / "mutation_frequency_by_position.png", dpi=300)
    plt.close()


# -----------------------------
# 6. FREQUENCY OF EACH MUTATION
# -----------------------------

if not mut_df.empty:
    mutation_freq = (
        mut_df.groupby("mutation")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    mutation_freq.to_csv(
        OUTDIR / "mutation_frequency.tsv",
        sep="\t",
        index=False
    )

    print("\nTop mutations:")
    print(mutation_freq.head(20))

    top_n = 30
    top_mutations = mutation_freq.head(top_n)

    plt.figure(figsize=(10, 8))
    plt.barh(
        top_mutations["mutation"][::-1],
        top_mutations["count"][::-1]
    )
    plt.xlabel("Count")
    plt.ylabel("Mutation")
    plt.title(f"Top {top_n} most frequent peptide mutations")
    plt.tight_layout()
    plt.savefig(OUTDIR / "top_mutation_frequencies.png", dpi=300)
    plt.close()


# -----------------------------
# 7. CORRELATION BETWEEN MHCFLURRY AND NETMHCPAN OUTPUT COLUMNS
# -----------------------------

mhcflurry_cols = [
    col for col in df_pos.columns
    if "mhcflurry" in col.lower()
]

netmhcpan_cols = [
    col for col in df_pos.columns
    if "netmhcpan" in col.lower()
]

print("\nDetected MHCflurry columns:")
print(mhcflurry_cols)

print("\nDetected netMHCpan columns:")
print(netmhcpan_cols)

# Convert detected predictor columns to numeric
for col in mhcflurry_cols + netmhcpan_cols:
    df_pos[col] = pd.to_numeric(df_pos[col], errors="coerce")

correlation_records = []

for m_col in mhcflurry_cols:
    for n_col in netmhcpan_cols:
        temp = df_pos[[m_col, n_col]].dropna()

        if len(temp) < 3:
            pearson_r = np.nan
            spearman_r = np.nan
            n = len(temp)
        else:
            pearson_r = temp[m_col].corr(temp[n_col], method="pearson")
            spearman_r = temp[m_col].corr(temp[n_col], method="spearman")
            n = len(temp)

        correlation_records.append({
            "mhcflurry_column": m_col,
            "netmhcpan_column": n_col,
            "n_complete_rows": n,
            "pearson_r": pearson_r,
            "spearman_r": spearman_r
        })

correlation_df = pd.DataFrame(correlation_records)

correlation_df.to_csv(
    OUTDIR / "mhcflurry_netmhcpan_correlations.tsv",
    sep="\t",
    index=False
)

print("\nMHCflurry vs netMHCpan correlations:")
print(correlation_df)


# -----------------------------
# 8. CORRELATION HEATMAP
# -----------------------------

if not correlation_df.empty:
    heatmap_data = correlation_df.pivot(
        index="mhcflurry_column",
        columns="netmhcpan_column",
        values="spearman_r"
    )

    plt.figure(figsize=(1.8 * len(netmhcpan_cols) + 4, 1.2 * len(mhcflurry_cols) + 3))
    plt.imshow(heatmap_data, aspect="auto")

    plt.xticks(
        ticks=np.arange(len(heatmap_data.columns)),
        labels=heatmap_data.columns,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        ticks=np.arange(len(heatmap_data.index)),
        labels=heatmap_data.index
    )

    plt.colorbar(label="Spearman correlation")
    plt.title("MHCflurry vs netMHCpan correlations")

    for i in range(heatmap_data.shape[0]):
        for j in range(heatmap_data.shape[1]):
            value = heatmap_data.iloc[i, j]
            if pd.notna(value):
                plt.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center"
                )

    plt.tight_layout()
    plt.savefig(OUTDIR / "mhcflurry_netmhcpan_spearman_heatmap.png", dpi=300)
    plt.close()


# -----------------------------
# 9. OPTIONAL SCATTER PLOTS FOR EVERY COMBINATION
# -----------------------------

scatter_dir = OUTDIR / "mhcflurry_netmhcpan_scatterplots"
scatter_dir.mkdir(exist_ok=True)

for m_col in mhcflurry_cols:
    for n_col in netmhcpan_cols:
        temp = df_pos[[m_col, n_col]].dropna()

        if len(temp) < 3:
            continue

        plt.figure(figsize=(6, 5))
        plt.scatter(temp[m_col], temp[n_col], alpha=0.4)
        plt.xlabel(m_col)
        plt.ylabel(n_col)
        plt.title(f"{m_col} vs {n_col}")
        plt.tight_layout()

        safe_m = m_col.replace("/", "_").replace(" ", "_")
        safe_n = n_col.replace("/", "_").replace(" ", "_")

        plt.savefig(scatter_dir / f"{safe_m}_vs_{safe_n}.png", dpi=300)
        plt.close()


print(f"\nDone. Outputs saved to: {OUTDIR}")