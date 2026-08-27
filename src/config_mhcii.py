from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class MHCIIConfigError(ValueError):
    pass


def load_mhcii_config(path: Path) -> dict[str, Any]:
    with path.expanduser().resolve().open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = {"run", "fragmentation", "mhc", "netmhciipan", "annotate"}
    missing = required - set(config or {})
    if missing:
        raise MHCIIConfigError(f"Missing configuration sections: {sorted(missing)}")
    kmers = config["fragmentation"].get("kmers")
    if not isinstance(kmers, list) or not kmers or any(not isinstance(k, int) or k < 1 for k in kmers):
        raise MHCIIConfigError("fragmentation.kmers must contain positive integers")
    alleles = config["mhc"].get("alleles")
    if not isinstance(alleles, list) or not alleles:
        raise MHCIIConfigError("mhc.alleles must be a non-empty list")
    for index, allele in enumerate(alleles):
        if not isinstance(allele, dict) or not allele.get("netmhciipan"):
            raise MHCIIConfigError(f"mhc.alleles[{index}] requires netmhciipan")
    threshold = config["netmhciipan"].get("el_rank_binder_threshold")
    if not isinstance(threshold, (int, float)) or threshold <= 0:
        raise MHCIIConfigError("netmhciipan.el_rank_binder_threshold must be positive")
    return config


def get_run_label(config: dict[str, Any]) -> str:
    kmers = config["fragmentation"]["kmers"]
    kmer_label = f"k{kmers[0]}" if len(kmers) == 1 else "k" + "-".join(map(str, kmers))
    return f"{config['run']['tag']}__{kmer_label}"


def get_netmhciipan_alleles(config: dict[str, Any]) -> list[str]:
    return [allele["netmhciipan"] for allele in config["mhc"]["alleles"]]
