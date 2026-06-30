from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

OUTPUT_FILE = Path(
    "data/output/linear_regression/VR5_V3__k9_count_net_wt_relative/hydrophobicity_changes.png"
)

# Mutations to display
SELECTED_MUTATIONS = [
    "F12P",
    "F12R",
    "F12W",
    "F12D",
    "F12C",
    "V4I",
    "V4L",
    "S1F",
    "V4F",
    "S1I"
]

# Kyte–Doolittle hydrophobicity scale
HYDROPHOBICITY = {
    "I": 4.5,
    "V": 4.2,
    "L": 3.8,
    "F": 2.8,
    "C": 2.5,
    "M": 1.9,
    "A": 1.8,
    "G": -0.4,
    "T": -0.7,
    "S": -0.8,
    "W": -0.9,
    "Y": -1.3,
    "P": -1.6,
    "H": -3.2,
    "E": -3.5,
    "Q": -3.5,
    "D": -3.5,
    "N": -3.5,
    "K": -3.9,
    "R": -4.5,
}


# ------------------------------------------------------------
# Mutation parsing
# ------------------------------------------------------------

def parse_mutation(mutation: str) -> dict:
    """
    Parse a mutation such as T2L into:
    wild-type residue, position and mutant residue.
    """
    match = re.fullmatch(r"([A-Z])(\d+)([A-Z])", mutation)

    if match is None:
        raise ValueError(f"Invalid mutation format: {mutation}")

    wild_type, position, mutant = match.groups()

    if wild_type not in HYDROPHOBICITY:
        raise ValueError(f"Unknown wild-type amino acid: {wild_type}")

    if mutant not in HYDROPHOBICITY:
        raise ValueError(f"Unknown mutant amino acid: {mutant}")

    wild_type_value = HYDROPHOBICITY[wild_type]
    mutant_value = HYDROPHOBICITY[mutant]

    return {
        "mutation": mutation,
        "position": int(position),
        "wild_type": wild_type,
        "mutant": mutant,
        "wild_type_hydrophobicity": wild_type_value,
        "mutant_hydrophobicity": mutant_value,
        "hydrophobicity_change": mutant_value - wild_type_value,
    }


plot_data = pd.DataFrame(
    [parse_mutation(mutation) for mutation in SELECTED_MUTATIONS]
)

# Retain the order given in SELECTED_MUTATIONS
plot_data["mutation"] = pd.Categorical(
    plot_data["mutation"],
    categories=SELECTED_MUTATIONS,
    ordered=True,
)

plot_data = plot_data.sort_values("mutation")


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 5.5))

colours = [
    "tab:blue" if value > 0 else "tab:orange"
    for value in plot_data["hydrophobicity_change"]
]

bars = ax.bar(
    plot_data["mutation"],
    plot_data["hydrophobicity_change"],
    color=colours,
    edgecolor="black",
    linewidth=0.7,
)

ax.axhline(0, color="black", linewidth=1)

ax.set_xlabel("Amino-acid mutation")
ax.set_ylabel("Change in hydrophobicity\n(mutant − wild type)")
ax.set_title("Hydrophobicity changes caused by selected mutations")

# Add values above or below each bar
for bar, value in zip(bars, plot_data["hydrophobicity_change"]):
    vertical_alignment = "bottom" if value >= 0 else "top"
    offset = 0.12 if value >= 0 else -0.12

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + offset,
        f"{value:+.1f}",
        ha="center",
        va=vertical_alignment,
        fontsize=9,
    )

# Add explanatory labels
ax.text(
    1.01,
    0.95,
    "More hydrophobic",
    transform=ax.transAxes,
    ha="left",
    va="top",
    color="tab:blue",
)

ax.text(
    1.01,
    0.05,
    "Less hydrophobic",
    transform=ax.transAxes,
    ha="left",
    va="bottom",
    color="tab:orange",
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
plt.show()

print(plot_data.to_string(index=False))
print(f"\nPlot saved to: {OUTPUT_FILE}")