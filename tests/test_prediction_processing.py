from __future__ import annotations

import pandas as pd
import pytest

from src.combine_predictions import combine_predictions
from src.mhcflurry import filter_and_expand_mhcflurry_predictions
from src.netmhciipan import expand_predictions, parse_netmhciipan_xls
from src.netmhcpan import parse_netmhcpan_xls_wide_tsv


def test_parse_netmhcpan_xls(tmp_path) -> None:
    path = tmp_path / "netmhcpan.xls"
    path.write_text(
        "\t\t\tH-2-Db\n"
        "Pos\tPeptide\tID\tcore\ticore\tEL_score\tEL_rank\n"
        "1\tABCDEFGHI\tpep_1\tABCDEFGHI\tABCDEFGHI\t0.8\t1.2\n"
    )
    result = parse_netmhcpan_xls_wide_tsv(path)
    assert result.loc[0, "allele"] == "H-2-Db"
    assert result.loc[0, "k"] == 9
    assert result.loc[0, "netMHCpan_EL_rank"] == pytest.approx(1.2)


def test_parse_and_expand_netmhciipan(tmp_path) -> None:
    path = tmp_path / "netmhciipan.xls"
    path.write_text(
        "\t\t\t\tH-2-IAb\n"
        "Pos\tPeptide\tID\tTarget\tCore\tInverted\tScore_EL\tRank_EL\n"
        "1\tABCDEFGHIJKLMNO\tpep_1\tPEPLIST\tDEFGHIJKL\t0\t0.7\t4.2\n"
    )
    predictions = parse_netmhciipan_xls(path)
    unique = pd.DataFrame(
        {"peptide_id": ["pep_1"], "peptide": ["ABCDEFGHIJKLMNO"], "k": [15], "occurrence_count": [2]}
    )
    mapping = pd.DataFrame(
        {"peptide_id": ["pep_1", "pep_1"], "variant_id": ["v1", "v2"], "start": [1, 2], "end": [15, 16]}
    )
    expanded = expand_predictions(predictions, unique, mapping, 5.0)
    assert len(expanded) == 2
    assert expanded["netMHCIIpan_EL_rank_binder"].all()


def test_mhcflurry_mapping_and_class_i_combination(tmp_path) -> None:
    unique_path = tmp_path / "unique.tsv"
    map_path = tmp_path / "map.tsv"
    raw_path = tmp_path / "mhcflurry.tsv"
    net_path = tmp_path / "net.tsv"
    mhc_path = tmp_path / "mhc.tsv"

    pd.DataFrame(
        {"peptide_id": ["p1"], "peptide": ["ABCDEFGHI"], "k": [9], "occurrence_count": [1]}
    ).to_csv(unique_path, sep="\t", index=False)
    pd.DataFrame(
        {"peptide_id": ["p1"], "variant_id": ["v1"], "start": [1], "end": [9], "criteria": ["keep"]}
    ).to_csv(map_path, sep="\t", index=False)
    pd.DataFrame(
        {
            "peptide": ["ABCDEFGHI"], "allele": ["H2-D*b"],
            "mhcflurry_affinity_percentile": [1.0],
            "mhcflurry_presentation_percentile": [0.5],
        }
    ).to_csv(raw_path, sep="\t", index=False)
    mhc = filter_and_expand_mhcflurry_predictions(
        unique_peptides_tsv=unique_path,
        peptide_map_tsv=map_path,
        mhcflurry_tsv=raw_path,
        alleles=["H2-D*b"],
    )
    assert mhc["MHCflurry_affinity_percentile_pass"].all()
    mhc.to_csv(mhc_path, sep="\t", index=False)

    pd.DataFrame(
        {
            "allele": ["H-2-Db"], "peptide": ["ABCDEFGHI"], "variant_id": ["v1"],
            "start": [1], "end": [9], "k": [9], "peptide_id": ["p1"],
            "length": [9], "occurrence_count": [1], "criteria": ["keep"],
            "netMHCpan_EL_rank": [1.1], "netMHCpan_EL_rank_pass": [True],
        }
    ).to_csv(net_path, sep="\t", index=False)
    combined = combine_predictions(net_path, mhc_path)
    assert len(combined) == 1
    assert combined.loc[0, "allele"] == "H2-D*b"
    assert combined.loc[0, "criteria"] == "keep"


def test_combination_rejects_duplicate_prediction_keys(tmp_path) -> None:
    base = {
        "allele": ["H2-D*b", "H2-D*b"], "peptide": ["ABCDEFGHI"] * 2,
        "variant_id": ["v1"] * 2, "start": [1] * 2, "end": [9] * 2, "k": [9] * 2,
    }
    net = tmp_path / "net.tsv"
    mhc = tmp_path / "mhc.tsv"
    pd.DataFrame(base).to_csv(net, sep="\t", index=False)
    pd.DataFrame({key: [value[0]] for key, value in base.items()}).to_csv(mhc, sep="\t", index=False)
    with pytest.raises(ValueError, match="duplicate"):
        combine_predictions(net, mhc)

