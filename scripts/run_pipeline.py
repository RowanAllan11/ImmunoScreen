from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import (
    get_alleles,
    get_run_label,
    load_config,
)


def run_command(command: list[str]) -> None:
    print("\nRunning:")
    print(" ".join(command))
    subprocess.run(command, check=True)


def run_fragmentation(
    config: dict[str, Any],
    run_label: str,
) -> None:
    settings = config["fragmentation"]

    if not settings.get("enabled", True):
        print("Skipping fragmentation.")
        return

    command = [
        sys.executable,
        "scripts/run_fragmentation.py",
        "--tag",
        config["run"]["tag"],
        "--input-type",
        settings["input_type"],
        "--i",
        settings["input"],
        "--sequence-col",
        settings["sequence_column"],
        "--id-col",
        settings["variant_id_column"],
        "--kmers",
        *map(str, settings["kmers"]),
    ]

    variable_region = settings.get("variable_region", {})

    if variable_region.get("enabled", False):
        command.extend(
            [
                "--var-only",
                "--var-start",
                str(variable_region["start"]),
                "--var-end",
                str(variable_region["end"]),
                "--var-mode",
                variable_region.get("mode", "overlap"),
            ]
        )

    metadata_columns = settings.get("metadata_columns", [])

    if metadata_columns:
        command.extend(
            ["--metadata-cols", *metadata_columns]
        )

    run_command(command)


def run_mhcflurry(
    config: dict[str, Any],
    run_label: str,
) -> None:
    settings = config["mhcflurry"]

    if not settings.get("enabled", True):
        print("Skipping MHCflurry.")
        return

    output_root = Path(config["run"]["output_root"])
    fragment_dir = output_root / "fragmentation" / run_label

    command = [
        sys.executable,
        "scripts/run_mhcflurry_pipeline.py",
        "--peptides",
        str(fragment_dir / "unique_peptides.tsv"),
        "--peptide-map",
        str(fragment_dir / "peptide_variant_map.tsv"),
        "--alleles",
        *get_alleles(config, "mhcflurry"),
        "--affinity-percentile-threshold",
        str(settings["affinity_percentile_threshold"]),
        "--outdir",
        str(output_root / "mhcflurry"),
    ]

    run_command(command)


def run_netmhcpan(
    config: dict[str, Any],
    run_label: str,
) -> None:
    settings = config["netmhcpan"]

    if not settings.get("enabled", True):
        print("Skipping netMHCpan.")
        return

    output_root = Path(config["run"]["output_root"])
    fragment_dir = output_root / "fragmentation" / run_label

    command = [
        sys.executable,
        "-m",
        "scripts.run_netmhcpan_pipeline",
        "--peptides",
        str(fragment_dir / "unique_peptides.tsv"),
        "--peptide-map",
        str(fragment_dir / "peptide_variant_map.tsv"),
        "--alleles",
        *get_alleles(config, "netmhcpan"),
        "--kmers",
        *map(str, config["fragmentation"]["kmers"]),
        "--el-rank-threshold",
        str(settings["el_rank_threshold"]),
        "--outdir",
        str(output_root / "netmhcpan"),
    ]

    if settings.get("deduplicate", True):
        command.append("--dedup")

    run_command(command)


def run_combination(
    config: dict[str, Any],
    run_label: str,
) -> None:
    settings = config["combine"]

    if not settings.get("enabled", True):
        print("Skipping combination and annotation.")
        return

    output_root = Path(config["run"]["output_root"])

    command = [
        sys.executable,
        "scripts/combine_annotate.py",
        "--netmhcpan-file",
        str(
            output_root
            / "netmhcpan"
            / run_label
            / "predictions_mapped.tsv"
        ),
        "--mhcflurry-file",
        str(
            output_root
            / "mhcflurry"
            / run_label
            / "predictions_mapped.tsv"
        ),
        "--variant-id-col",
        settings["variant_id_column"],
        "--seq-col",
        settings["sequence_column"],
        "--i",
        settings["library"],
        "--var-start",
        str(settings["variable_region_start"]),
        "--var-end",
        str(settings["variable_region_end"]),
        "--wt-vr",
        settings["wild_type_variable_region"],
        "--outdir",
        str(output_root / "combined"),
    ]

    run_command(command)


def run_bigmhc(
    config: dict[str, Any],
    run_label: str,
) -> None:
    settings = config["bigmhc"]

    if not settings.get("enabled", True):
        print("Skipping BigMHC.")
        return

    output_root = Path(config["run"]["output_root"])

    command = [
        sys.executable,
        "scripts/run_bigmhc.py",
        "--i",
        str(
            output_root
            / "combined"
            / run_label
            / "combined_annotated.tsv"
        ),
        "--m",
        settings["model"],
        "--t",
        str(settings["threads"]),
        "--d",
        settings["device"],
        "--outdir",
        str(output_root / "bigmhc"),
    ]

    run_command(command)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the complete AAV epitope pipeline."
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Pipeline YAML configuration file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_label = get_run_label(config)

    print(f"Run label: {run_label}")
    print(
        "MHCflurry alleles:",
        ", ".join(get_alleles(config, "mhcflurry")),
    )
    print(
        "netMHCpan alleles:",
        ", ".join(get_alleles(config, "netmhcpan")),
    )

    run_fragmentation(config, run_label)
    run_mhcflurry(config, run_label)
    run_netmhcpan(config, run_label)
    run_combination(config, run_label)
    run_bigmhc(config, run_label)

    print(f"\nPipeline complete: {run_label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Command to run the pipeline:
python scripts/run_pipeline.py --config config/vr5_v3_k9.yaml
"""