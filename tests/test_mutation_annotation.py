from __future__ import annotations

import pandas as pd

from src.mutation_label import attach_mutation_labels, mutation_string, peptide_mutation_string


def test_mutation_helpers() -> None:
    assert mutation_string("ABXD", "ABCD") == "C3X"
    assert mutation_string("ABCD", "ABCD") == "WT"
    assert peptide_mutation_string("A1V;C3X", 3, 4, var_start=2) == "C3X"


def test_attach_mutation_labels(tmp_path) -> None:
    library = tmp_path / "library.csv"
    pd.DataFrame(
        {"gene_id": ["wt", "mut"], "sequence": ["XXABCDYY", "XXABXDYY"]}
    ).to_csv(library, index=False)
    predictions = pd.DataFrame(
        {"variant_id": ["wt", "mut"], "start": [1, 3], "end": [4, 5]}
    )
    result = attach_mutation_labels(
        predictions,
        library,
        library_id_col="gene_id",
        seq_col="sequence",
        var_start=3,
        var_end=6,
        wt_vr="ABCD",
    )
    assert result["VR_mutation"].tolist() == ["WT", "C3X"]
    assert result["peptide_mutation"].tolist() == ["WT", "C3X"]

