from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from analysis import vi_scoring, vi_scoring_wt, vi_scoring_wt_mhcii


def write_mhci_pair(tmp_path: Path, run_name: str = "VR4__k9") -> tuple[Path, Path]:
    variant_dir = tmp_path / run_name
    wt_dir = tmp_path / "WT_vr4__k9"
    variant_dir.mkdir()
    wt_dir.mkdir()
    common = {
        "allele": ["H2-D*b", "H2-D*b"],
        "peptide_id": ["p1", "p2"],
        "start": [1, 2],
        "end": [9, 10],
        "k": [9, 9],
        "netMHCpan_EL_rank": [1.0, 3.0],
        "netMHCpan_EL_rank_pass": [True, False],
        "MHCflurry_affinity_percentile": [1.5, 4.0],
        "MHCflurry_affinity_percentile_pass": [True, False],
    }
    wt = pd.DataFrame(
        common
        | {
            "peptide": ["AAAAAAAAA", "AAAAAAAAA"],
            "variant_id": ["wt", "wt"],
            "VR_mutation": ["WT", "WT"],
        }
    )
    variant = pd.DataFrame(
        common
        | {
            "peptide": ["VAAAAAAAA", "AAAAAAAAA"],
            "variant_id": ["v1", "v1"],
            "VR_mutation": ["A1V", "A1V"],
            "netMHCpan_EL_rank": [0.5, 3.0],
            "MHCflurry_affinity_percentile": [0.5, 4.0],
        }
    )
    variant_path = variant_dir / "combined_annotated.tsv"
    wt_path = wt_dir / "combined_annotated.tsv"
    variant.to_csv(variant_path, sep="\t", index=False)
    wt.to_csv(wt_path, sep="\t", index=False)
    return variant_path, wt_path


def write_mhcii_pair(tmp_path: Path) -> tuple[Path, Path]:
    variant_dir = tmp_path / "VR6__k15"
    wt_dir = tmp_path / "WT_vr6__k15"
    variant_dir.mkdir()
    wt_dir.mkdir()
    common = {
        "allele": ["H-2-IAb", "H-2-IAb"],
        "peptide_id": ["p1", "p2"],
        "start": [1, 2],
        "end": [15, 16],
        "k": [15, 15],
        "netMHCIIpan_EL_rank": [4.0, 7.0],
        "netMHCIIpan_EL_rank_binder": [True, False],
    }
    wt = pd.DataFrame(
        common
        | {
            "peptide": ["A" * 15, "A" * 15],
            "variant_id": ["wt", "wt"],
            "VR_mutation": ["WT", "WT"],
        }
    )
    variant = pd.DataFrame(
        common
        | {
            "peptide": ["V" + "A" * 14, "A" * 15],
            "variant_id": ["v1", "v1"],
            "VR_mutation": ["A1V", "A1V"],
            "netMHCIIpan_EL_rank": [2.0, 7.0],
        }
    )
    variant_path = variant_dir / "predictions_mapped_annotated.tsv"
    wt_path = wt_dir / "predictions_mapped_annotated.tsv"
    variant.to_csv(variant_path, sep="\t", index=False)
    wt.to_csv(wt_path, sep="\t", index=False)
    return variant_path, wt_path


def test_absolute_mhci_cli_infers_vr4_label(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    variant, _ = write_mhci_pair(tmp_path)
    output = tmp_path / "absolute"
    monkeypatch.setattr(
        "sys.argv",
        ["vi_scoring.py", "--variant-input", str(variant), "--output-root", str(output)],
    )
    assert vi_scoring.main() == 0
    result = output / "VR4_K9_H2-Db" / "variant_immunogenicity_scores.tsv"
    assert result.is_file()
    assert pd.read_csv(result, sep="\t")["variant_id"].tolist() == ["v1"]
    assert not (output / "VR4_K9_combined").exists()


def test_wt_relative_mhci_cli_accepts_vr4_outputs(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    variant, wt = write_mhci_pair(tmp_path)
    output = tmp_path / "relative"
    monkeypatch.setattr(
        "sys.argv",
        [
            "vi_scoring_wt.py", "--variant-input", str(variant),
            "--wt-input", str(wt), "--output-root", str(output),
        ],
    )
    assert vi_scoring_wt.main() == 0
    result = output / "VR4_K9_H2-Db" / "variant_immunogenicity_scores.tsv"
    scored = pd.read_csv(result, sep="\t")
    assert scored.loc[0, "matched_window_count"] == 2
    assert scored.loc[0, "netMHCpan_mean_window_improvement"] == pytest.approx(0.25)
    assert not (output / "VR4_K9_combined").exists()


def test_wt_relative_mhcii_cli_infers_vr6_label(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    variant, wt = write_mhcii_pair(tmp_path)
    output = tmp_path / "mhcii"
    monkeypatch.setattr(
        "sys.argv",
        [
            "vi_scoring_wt_mhcii.py", "--variant-input", str(variant),
            "--wt-input", str(wt), "--output-root", str(output),
        ],
    )
    assert vi_scoring_wt_mhcii.main() == 0
    result = output / "VR6_K15_H2-IAb" / "variant_immunogenicity_scores.tsv"
    scored = pd.read_csv(result, sep="\t")
    assert scored.loc[0, "matched_window_count"] == 2
    assert scored.loc[0, "netMHCIIpan_mean_window_improvement"] == pytest.approx(1.0)
    assert not (output / "VR6_K15_combined").exists()
