from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NETMHCII_PATH = REPO_ROOT / "tools" / "netMHCIIpan-4.3" / "netMHCIIpan"


def read_unique_peptides(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, sep="\t", dtype=str)
    required = {"peptide_id", "peptide", "k", "occurrence_count"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    data["k"] = pd.to_numeric(data["k"], errors="raise").astype(int)
    data["occurrence_count"] = pd.to_numeric(
        data["occurrence_count"], errors="raise"
    ).astype(int)
    return data


def read_peptide_map(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, sep="\t", dtype=str)
    required = {"peptide_id", "variant_id", "start", "end"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    data["start"] = pd.to_numeric(data["start"], errors="raise").astype(int)
    data["end"] = pd.to_numeric(data["end"], errors="raise").astype(int)
    return data


def run_netmhciipan(
    peptides: list[str],
    alleles: list[str],
    executable: Path,
    output_path: Path,
) -> None:
    if not executable.is_file():
        raise FileNotFoundError(f"NetMHCIIpan executable not found: {executable}")
    if not peptides:
        raise ValueError("No peptides were supplied to NetMHCIIpan")
    if not alleles:
        raise ValueError("No alleles were supplied to NetMHCIIpan")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    peptide_path = output_path.with_suffix(".peptides.tmp.txt")
    peptide_path.write_text("\n".join(peptides) + "\n", encoding="utf-8")
    command = [
        str(executable), "-inptype", "1", "-a", ",".join(alleles),
        "-f", str(peptide_path), "-xls", "-xlsfile", str(output_path),
    ]
    environment = os.environ.copy()
    environment.setdefault("TMPDIR", "/tmp")
    print("Running:", " ".join(command))
    try:
        process = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=environment,
        )
        if process.returncode != 0:
            raise RuntimeError(
                f"NetMHCIIpan failed ({process.returncode}):\n{process.stderr}"
            )
    finally:
        peptide_path.unlink(missing_ok=True)
    if not output_path.is_file():
        raise RuntimeError(f"NetMHCIIpan did not create {output_path}")


def parse_netmhciipan_xls(path: Path) -> pd.DataFrame:
    """Convert NetMHCIIpan's wide tab-delimited XLS output to tidy rows."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header_index = next(
        (index for index, line in enumerate(lines) if line.startswith("Pos\tPeptide\tID\tTarget\t")),
        None,
    )
    if header_index is None or header_index == 0:
        raise ValueError(f"Could not find the NetMHCIIpan XLS header in {path}")

    columns = lines[header_index].split("\t")
    expected = ["Pos", "Peptide", "ID", "Target"]
    if columns[:4] != expected:
        raise ValueError(f"Unexpected NetMHCIIpan header prefix: {columns[:4]}")
    allele_cells = lines[header_index - 1].split("\t")[4:]
    alleles = [cell.strip() for cell in allele_cells if cell.strip()]
    if not alleles:
        raise ValueError(f"Could not infer allele names from {path}")

    fields = ["Core", "Inverted", "Score_EL", "Rank_EL"]
    records: list[dict[str, object]] = []
    for line in lines[header_index + 1:]:
        if not line.strip() or line.startswith("#"):
            continue
        values = line.split("\t")
        if len(values) < 4 + len(alleles) * len(fields):
            continue
        peptide = values[1].strip()
        tail = values[4:]
        for allele_index, allele in enumerate(alleles):
            offset = allele_index * len(fields)
            block = tail[offset:offset + len(fields)]
            records.append({
                "peptide": peptide,
                "allele": allele,
                "netMHCIIpan_core": block[0],
                "netMHCIIpan_inverted": pd.to_numeric(block[1], errors="coerce"),
                "netMHCIIpan_EL_score": pd.to_numeric(block[2], errors="coerce"),
                "netMHCIIpan_EL_rank": pd.to_numeric(block[3], errors="coerce"),
            })
    result = pd.DataFrame(records)
    if result.empty:
        raise ValueError(f"No prediction rows were parsed from {path}")
    return result


def expand_predictions(
    predictions: pd.DataFrame,
    unique_peptides: pd.DataFrame,
    peptide_map: pd.DataFrame,
    binder_rank_threshold: float,
) -> pd.DataFrame:
    """Map peptide-level predictions back to all source variants."""
    if binder_rank_threshold <= 0:
        raise ValueError("The binder rank threshold must be positive")
    prediction_keys = ["allele", "peptide"]
    if predictions.duplicated(prediction_keys).any():
        raise ValueError("NetMHCIIpan returned duplicate allele-peptide predictions")
    peptide_predictions = unique_peptides.merge(
        predictions, on="peptide", how="inner", validate="one_to_many"
    )
    if peptide_predictions["peptide_id"].nunique() != unique_peptides["peptide_id"].nunique():
        raise ValueError("Not every unique peptide received a NetMHCIIpan prediction")
    expanded = peptide_predictions.merge(
        peptide_map, on="peptide_id", how="inner", validate="many_to_many"
    )
    expanded["netMHCIIpan_EL_rank_binder"] = (
        expanded["netMHCIIpan_EL_rank"].notna()
        & (expanded["netMHCIIpan_EL_rank"] < float(binder_rank_threshold))
    )
    leading = [
        "allele", "peptide_id", "peptide", "k", "variant_id", "start", "end",
        "occurrence_count",
    ]
    remaining = [column for column in expanded.columns if column not in leading]
    return expanded[leading + remaining].sort_values(
        ["allele", "variant_id", "start", "end"]
    ).reset_index(drop=True)
