from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when the pipeline configuration is invalid."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate a pipeline YAML configuration."""

    config_path = Path(path).expanduser().resolve()

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file does not exist: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ConfigError("The YAML root must be a mapping.")

    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required_sections = [
        "run",
        "fragmentation",
        "mhc",
        "mhcflurry",
        "netmhcpan",
        "combine",
    ]

    missing = [
        section
        for section in required_sections
        if section not in config
    ]

    if missing:
        raise ConfigError(
            "Missing required configuration sections: "
            + ", ".join(missing)
        )

    run = config["run"]
    if not run.get("tag"):
        raise ConfigError("run.tag must be specified.")
    if not run.get("output_root"):
        raise ConfigError("run.output_root must be specified.")

    fragmentation = config["fragmentation"]
    kmers = fragmentation.get("kmers")

    if not isinstance(kmers, list) or not kmers:
        raise ConfigError(
            "fragmentation.kmers must be a non-empty list."
        )

    if any(not isinstance(k, int) or k < 1 for k in kmers):
        raise ConfigError(
            "Every fragmentation.kmers value must be a positive integer."
        )

    required_fragmentation_fields = {
        "input_type",
        "input",
        "sequence_column",
        "variant_id_column",
    }
    missing_fragmentation_fields = (
        required_fragmentation_fields - fragmentation.keys()
    )
    if missing_fragmentation_fields:
        raise ConfigError(
            "fragmentation is missing: "
            + ", ".join(sorted(missing_fragmentation_fields))
        )

    alleles = config["mhc"].get("alleles")

    if not isinstance(alleles, list) or not alleles:
        raise ConfigError(
            "mhc.alleles must contain at least one allele."
        )

    required_allele_fields = {"name", "mhcflurry", "netmhcpan"}

    for index, allele in enumerate(alleles):
        if not isinstance(allele, dict):
            raise ConfigError(
                f"mhc.alleles[{index}] must be a mapping."
            )

        missing_fields = required_allele_fields - allele.keys()

        if missing_fields:
            raise ConfigError(
                f"mhc.alleles[{index}] is missing: "
                + ", ".join(sorted(missing_fields))
            )

    combine = config["combine"]
    required_combine_fields = {
        "sequence_column",
        "library_id_column",
        "library",
        "variable_region_start",
        "variable_region_end",
        "wild_type_variable_region",
    }
    missing_combine_fields = required_combine_fields - combine.keys()
    if missing_combine_fields:
        raise ConfigError(
            "combine is missing: "
            + ", ".join(sorted(missing_combine_fields))
        )

    if "affinity_percentile_threshold" not in config["mhcflurry"]:
        raise ConfigError(
            "mhcflurry.affinity_percentile_threshold must be specified."
        )

    netmhcpan = config["netmhcpan"]
    for field in ["executable", "el_rank_threshold"]:
        if field not in netmhcpan:
            raise ConfigError(f"netmhcpan.{field} must be specified.")


def get_alleles(
    config: dict[str, Any],
    tool: str,
) -> list[str]:
    """Return allele names in the format required by a predictor."""

    valid_tools = {"mhcflurry", "netmhcpan"}

    if tool not in valid_tools:
        raise ValueError(
            f"Unknown MHC tool '{tool}'. "
            f"Expected one of: {sorted(valid_tools)}"
        )

    return [
        allele[tool]
        for allele in config["mhc"]["alleles"]
    ]


def get_run_label(config: dict[str, Any]) -> str:
    """Construct a run label without including the allele names."""

    tag = config["run"]["tag"]
    kmers = config["fragmentation"]["kmers"]

    if len(kmers) == 1:
        kmer_label = f"k{kmers[0]}"
    else:
        kmer_label = "k" + "-".join(map(str, kmers))

    return f"{tag}__{kmer_label}"
