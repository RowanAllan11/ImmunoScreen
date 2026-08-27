#!/usr/bin/env python3
"""Run fragmentation, NetMHCIIpan prediction, and mutation annotation."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config_mhcii import (
    get_netmhciipan_alleles,
    get_run_label,
    load_mhcii_config,
)


def run(command: list[str]) -> None:
    print("\nRunning:")
    print(" ".join(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_mhcii_config(args.config)
    run_label = get_run_label(config)
    output_root = Path(config["run"].get("output_root", "data/output"))
    fragmentation = config["fragmentation"]
    variable_region = fragmentation.get("variable_region", {})

    if fragmentation.get("enabled", True):
        command = [
            sys.executable, "scripts/run_fragmentation.py",
            "--tag", config["run"]["tag"],
            "--input-type", fragmentation["input_type"],
            "--i", fragmentation["input"],
            "--sequence-col", fragmentation["sequence_column"],
            "--id-col", fragmentation["variant_id_column"],
            "--kmers", *map(str, fragmentation["kmers"]),
            "--out-dir", str(output_root / "fragmentation"),
        ]
        metadata = fragmentation.get("metadata_columns", [])
        if metadata:
            command.extend(["--metadata-cols", *metadata])
        if variable_region.get("enabled", False):
            command.extend([
                "--var-only", "--var-start", str(variable_region["start"]),
                "--var-end", str(variable_region["end"]),
                "--var-mode", variable_region.get("mode", "overlap"),
            ])
        run(command)

    fragment_dir = output_root / "fragmentation" / run_label
    settings = config["netmhciipan"]
    if settings.get("enabled", True):
        run([
            sys.executable, "scripts/run_netmhciipan_pipeline.py",
            "--peptides", str(fragment_dir / "unique_peptides.tsv"),
            "--peptide-map", str(fragment_dir / "peptide_variant_map.tsv"),
            "--alleles", *get_netmhciipan_alleles(config),
            "--netmhciipan", settings.get(
                "executable", "tools/netMHCIIpan-4.3/netMHCIIpan"
            ),
            "--el-rank-binder-threshold",
            str(settings["el_rank_binder_threshold"]),
            "--outdir", str(output_root / "netmhciipan"),
        ])

    annotate = config["annotate"]
    if annotate.get("enabled", True):
        predictions = output_root / "netmhciipan" / run_label / "predictions_mapped.tsv"
        run([
            sys.executable, "scripts/annotate_netmhciipan.py",
            "--predictions", str(predictions),
            "--library", annotate["library"],
            "--library-id-column", annotate["library_id_column"],
            "--sequence-column", annotate["sequence_column"],
            "--variable-region-start", str(annotate["variable_region_start"]),
            "--variable-region-end", str(annotate["variable_region_end"]),
            "--wild-type-variable-region", annotate["wild_type_variable_region"],
            "--outdir", str(output_root / "netmhciipan_annotated"),
        ])

    print(f"\nNetMHCIIpan pipeline complete: {run_label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
