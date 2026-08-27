from __future__ import annotations

import pandas as pd
import pytest

from src.fragmentation import FragmentationInput, fragment_sequence, fragment_to_tables


def test_fragment_sequence_boundaries() -> None:
    assert list(fragment_sequence("ABCDE", 3)) == [(0, "ABC"), (1, "BCD"), (2, "CDE")]
    assert list(fragment_sequence("AB", 3)) == []
    with pytest.raises(ValueError, match="k must be"):
        list(fragment_sequence("ABC", 0))


def test_fragmentation_stress_and_rerun_is_idempotent(tmp_path) -> None:
    count = 50

    def inputs():
        for index in range(count):
            yield FragmentationInput(
                sequence_id=f"v{index}",
                sequence="ABCDEFGHIJ",
                metadata={"group": "x", 'quoted"name': "y"},
            )

    output = tmp_path / "fragments"
    for _ in range(2):
        paths = fragment_to_tables(
            inputs(),
            kmers=[3, 4],
            out_dir=output,
            metadata_cols=["group", 'quoted"name'],
            commit_every=7,
        )
        all_fragments = pd.read_csv(paths["all_fragments"], sep="\t")
        unique = pd.read_csv(paths["unique_peptides"], sep="\t")
        mapping = pd.read_csv(paths["peptide_variant_map"], sep="\t")
        expected = count * ((10 - 3 + 1) + (10 - 4 + 1))
        assert len(all_fragments) == expected
        assert len(mapping) == expected
        assert int(unique["occurrence_count"].sum()) == expected
        assert mapping["group"].eq("x").all()
        assert mapping['quoted"name'].eq("y").all()


def test_variable_region_filtering(tmp_path) -> None:
    paths = fragment_to_tables(
        [FragmentationInput("v1", "ABCDEFG", {})],
        kmers=[3],
        out_dir=tmp_path,
        var_only=True,
        var_start=4,
        var_end=4,
        var_mode="overlap",
    )
    data = pd.read_csv(paths["all_fragments"], sep="\t")
    assert data[["start", "end"]].values.tolist() == [[2, 4], [3, 5], [4, 6]]
