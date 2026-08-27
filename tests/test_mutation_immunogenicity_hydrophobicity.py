from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT = Path(__file__).parents[1] / "analysis" / "mutation_immunogenicity_hydrophobicity_wt.py"


def load_module():
    spec = importlib.util.spec_from_file_location("mutation_framework", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_score_table(path: Path, allele: str, allele_index: int) -> None:
    rng = np.random.default_rng(100 + allele_index)
    substitutions = {
        1: ["A1V", "A1G", "A1S"], 2: ["C2A", "C2V", "C2S"],
        3: ["D3A", "D3V", "D3S"], 4: ["E4A", "E4V", "E4S"],
    }
    mutation_order = [mutation for choices in substitutions.values() for mutation in choices]
    effects = {mutation: (index - 5.5) / 8 for index, mutation in enumerate(mutation_order)}
    rows = []
    for variant_index in range(180):
        mutations = []
        for choices in substitutions.values():
            if rng.random() < 0.72:
                mutations.append(str(rng.choice(choices)))
        signal = sum(effects[mutation] for mutation in mutations)
        rows.append({
            "variant_id": f"variant_{variant_index}",
            "VR_mutation": ";".join(mutations) or "WT",
            "allele": allele,
            "score_a": signal + 0.08 * allele_index + rng.normal(0, 0.08),
            "score_b": -0.55 * signal + 0.04 * allele_index + rng.normal(0, 0.08),
        })
    path.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def test_framework_runs_all_requested_outputs_from_cli(tmp_path, monkeypatch):
    module = load_module()
    alleles = ["H2-D*b", "H2-D*d", "H2-K*b"]
    paths = []
    for index, allele in enumerate(alleles):
        path = tmp_path / "scores" / f"allele_{index}" / "variant_immunogenicity_scores.tsv"
        write_score_table(path, allele, index)
        paths.append(path)

    output = tmp_path / "analysis"
    monkeypatch.setattr(sys, "argv", [
        str(SCRIPT), "--inputs", *(str(path) for path in paths),
        "--outdir", str(output), "--outcomes", "score_a", "score_b",
        "--wt-sequence", "ACDE", "--position-offset", "100",
        "--min-mutation-count", "8", "--bootstraps", "8", "--seed", "7",
    ])
    assert module.main() == 0

    expected = [
        "tables/all_mutation_coefficients.tsv",
        "tables/mutation_model_fit.tsv",
        "tables/allele_effect_iqr.tsv",
        "tables/physicochemical_models/model_summary.tsv",
        "tables/physicochemical_models/coefficient_summary.tsv",
        "tables/physicochemical_models/partial_r_squared.tsv",
        "tables/physicochemical_models/adjusted_position_effects.tsv",
        "tables/physicochemical_models/physicochemical_model_data.tsv",
        "plots/mutation_heatmaps/H2-Db/score_a.png",
        "plots/allele_correlations/score_a.png",
        "plots/allele_iqr/score_a.png",
        "plots/physicochemical_models/score_a__partial_r_squared.png",
        "plots/physicochemical_models/score_b__partial_r_squared.png",
        "plots/physicochemical_models/adjusted_position_effects.png",
    ]
    for relative_path in expected:
        assert (output / relative_path).is_file(), relative_path

    coefficients = pd.read_csv(output / "tables/all_mutation_coefficients.tsv", sep="\t")
    assert set(coefficients["allele"]) == set(alleles)
    assert set(coefficients["outcome"]) == {"score_a", "score_b"}
    assert set(coefficients["position_absolute"]) == {101, 102, 103, 104}
    assert coefficients.groupby(["allele", "outcome"]).size().eq(12).all()

    model_data = pd.read_csv(
        output / "tables/physicochemical_models/physicochemical_model_data.tsv", sep="\t"
    )
    assert len(model_data) == 24
    assert "median_allele_coefficient" in model_data
    model_terms = pd.read_csv(
        output / "tables/physicochemical_models/coefficient_summary.tsv", sep="\t"
    )["term"]
    assert not model_terms.str.startswith("allele_").any()
    iqr = pd.read_csv(output / "tables/allele_effect_iqr.tsv", sep="\t")
    assert iqr["allele_directional_agreement"].between(0.5, 1.0).all()


def test_mutation_labels_must_match_supplied_wt_sequence():
    module = load_module()
    with pytest.raises(ValueError, match="conflicts with --wt-sequence"):
        module.parse_mutation("V1A", "ACDE", 0)
