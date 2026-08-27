from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from scripts import run_mhcii_pipeline, run_pipeline
from src.config import ConfigError, get_run_label, load_config, validate_config
from src.config_mhcii import load_mhcii_config


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "name",
    ["vr4_k9.yaml", "vr6_k9.yaml", "wt_vr4_k9.yaml", "wt_vr6_k9.yaml"],
)
def test_retained_mhci_configs_are_valid(name: str) -> None:
    config = load_config(REPO_ROOT / "configs" / name)
    assert get_run_label(config).endswith("__k9")
    assert config["netmhcpan"]["executable"] == "tools/netMHCpan-4.2/netMHCpan"


@pytest.mark.parametrize(
    "name", ["vr6_k15_netmhciipan.yaml", "wt_vr6_k15_netmhciipan.yaml"]
)
def test_retained_mhcii_configs_are_valid(name: str) -> None:
    config = load_mhcii_config(REPO_ROOT / "configs" / name)
    assert config["fragmentation"]["kmers"] == [15]


def test_mhci_config_rejects_missing_downstream_field() -> None:
    config = load_config(REPO_ROOT / "configs" / "vr6_k9.yaml")
    broken = deepcopy(config)
    del broken["combine"]["sequence_column"]
    with pytest.raises(ConfigError, match="sequence_column"):
        validate_config(broken)


def test_mhci_orchestrator_builds_all_stage_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(run_pipeline, "run_command", commands.append)
    monkeypatch.setattr(
        "sys.argv",
        ["run_pipeline.py", "--config", str(REPO_ROOT / "configs/vr4_k9.yaml")],
    )
    assert run_pipeline.main() == 0
    assert len(commands) == 4
    assert commands[0][1].endswith("scripts/run_fragmentation.py")
    assert commands[1][1].endswith("scripts/run_mhcflurry_pipeline.py")
    assert commands[2][1:3] == ["-m", "scripts.run_netmhcpan_pipeline"]
    assert commands[3][1].endswith("scripts/combine_annotate.py")
    assert commands[0][commands[0].index("--out-dir") + 1] == "data/output/fragmentation"
    assert commands[2][commands[2].index("--netmhcpan") + 1] == (
        "tools/netMHCpan-4.2/netMHCpan"
    )


def test_mhcii_orchestrator_builds_all_stage_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(run_mhcii_pipeline, "run", commands.append)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_mhcii_pipeline.py",
            "--config",
            str(REPO_ROOT / "configs/vr6_k15_netmhciipan.yaml"),
        ],
    )
    assert run_mhcii_pipeline.main() == 0
    assert len(commands) == 3
    assert commands[0][1].endswith("scripts/run_fragmentation.py")
    assert commands[1][1].endswith("scripts/run_netmhciipan_pipeline.py")
    assert commands[2][1].endswith("scripts/annotate_netmhciipan.py")

