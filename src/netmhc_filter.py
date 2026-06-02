from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_netmhcpan_xls_wide_tsv(path: Path) -> pd.DataFrame:
    """
    Parse NetMHCpan -xls output (tab-delimited "wide" table) into tidy/long df with:
      peptide, k, allele, netMHCpan_EL_score, netMHCpan_EL_rank, (optional BA_score, BA_rank)
    """
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("Pos\tPeptide\tID\t"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"Could not find NetMHCpan xls header in: {path}")

    allele_line = lines[header_idx - 1]
    allele_cells = allele_line.split("\t")

    col_header = lines[header_idx].split("\t")
    base_cols = ["Pos", "Peptide", "ID"]
    if col_header[: len(base_cols)] != base_cols:
        raise ValueError(
            "Unexpected NetMHCpan xls columns. "
            f"Expected prefix {base_cols}, got {col_header[:len(base_cols)]}"
        )

    header_tail = col_header[len(base_cols) :]
    has_ba = "BA_score" in header_tail

    per_allele_fields = ["core", "icore", "EL_score", "EL_rank"]
    if has_ba:
        per_allele_fields += ["BA_score", "BA_rank"]
    block_w = len(per_allele_fields)

    allele_names: list[str] = []
    for c in allele_cells:
        c = c.strip()
        if not c:
            continue
        if "-" in c or c.startswith("HLA"):
            allele_names.append(c)
    if not allele_names:
        raise ValueError(f"Could not infer allele names from line: {allele_line}")

    records: list[dict] = []
    for line in lines[header_idx + 1 :]:
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < len(base_cols) + block_w:
            continue

        pep = parts[1]
        k = len(pep)

        tail = parts[len(base_cols) :]
        for ai, allele in enumerate(allele_names):
            start = ai * block_w
            end = start + block_w
            if end > len(tail):
                break

            blk = tail[start:end]
            el_score = pd.to_numeric(blk[2], errors="coerce")
            el_rank = pd.to_numeric(blk[3], errors="coerce")
            ba_score = pd.NA
            ba_rank = pd.NA
            if has_ba:
                ba_score = pd.to_numeric(blk[4], errors="coerce")
                ba_rank = pd.to_numeric(blk[5], errors="coerce")

            rec = {
                "peptide": pep,
                "k": int(k),
                "allele": allele,
                "netMHCpan_EL_score": float(el_score) if pd.notna(el_score) else pd.NA,
                "netMHCpan_EL_rank": float(el_rank) if pd.notna(el_rank) else pd.NA,
            }
            if has_ba:
                rec["netMHCpan_BA_score"] = float(ba_score) if pd.notna(ba_score) else pd.NA
                rec["netMHCpan_BA_rank"] = float(ba_rank) if pd.notna(ba_rank) else pd.NA

            records.append(rec)

    return pd.DataFrame.from_records(records)


def read_unique_peptides_tsv(path: Path) -> pd.DataFrame:
    """
    Read unique_peptides.tsv:
      peptide_id, peptide, k, occurrence_count
    """
    df = pd.read_csv(path, sep="\t", dtype=str)
    required = {"peptide_id", "peptide", "k", "occurrence_count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")

    df = df.copy()
    df["peptide"] = df["peptide"].astype(str)
    df["k"] = pd.to_numeric(df["k"], errors="coerce").astype("Int64")
    df["occurrence_count"] = pd.to_numeric(df["occurrence_count"], errors="coerce").astype("Int64")
    return df[["peptide_id", "peptide", "k", "occurrence_count"]]


def read_peptide_variant_map_tsv(path: Path) -> pd.DataFrame:
    """
    Read peptide_variant_map.tsv:
      peptide_id, variant_id, start, end, + metadata columns (e.g. criteria)
    """
    df = pd.read_csv(path, sep="\t", dtype=str)
    required = {"peptide_id", "variant_id", "start", "end"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")

    df = df.copy()
    df["start"] = pd.to_numeric(df["start"], errors="coerce").astype("Int64")
    df["end"] = pd.to_numeric(df["end"], errors="coerce").astype("Int64")
    return df


def infer_kmer_range_from_df(df: pd.DataFrame) -> str:
    mn = int(pd.to_numeric(df["k"], errors="coerce").min())
    mx = int(pd.to_numeric(df["k"], errors="coerce").max())
    return f"{mn}-{mx}mer" if mn != mx else f"{mn}mer"


def main() -> int:
    ap = argparse.ArgumentParser(description="Filter netMHCpan outputs and join variant positions (new tabular fragmentation).")

    ap.add_argument(
        "--netmhcpan-xls",
        type=Path,
        nargs="+",
        required=True,
        help="One or more netMHCpan .xls (tab-delimited) outputs (can be multiple k-mer runs).",
    )
    ap.add_argument("--peptides", type=Path, required=True, help="Path to unique_peptides.tsv")
    ap.add_argument("--peptide-map", type=Path, required=True, help="Path to peptide_variant_map.tsv")

    ap.add_argument("-o", "--outdir", type=Path, required=True, help="Output directory")
    ap.add_argument("--st", required=True, help="ST tag for output filename stem (e.g. VR5)")
    ap.add_argument(
        "--combined-name",
        default=None,
        help="Optional explicit filename stem for combined output (no extension).",
    )

    ap.add_argument("--el-rank-threshold", type=float, default=2.0, help="Keep rows with EL_rank < threshold")
    ap.add_argument("--dedup", action="store_true", help="Deduplicate identical (allele, peptide_id, variant_id, start, end) keeping best rank")

    args = ap.parse_args()

    uniq = read_unique_peptides_tsv(args.peptides)
    pmap = read_peptide_variant_map_tsv(args.peptide_map)

    dfs: list[pd.DataFrame] = []
    for xls_path in args.netmhcpan_xls:
        nm = parse_netmhcpan_xls_wide_tsv(xls_path)
        dfs.append(nm)

    nm_all = pd.concat(dfs, ignore_index=True)

    # Filter first (reduces fan-out)
    nm_all["netMHCpan_EL_rank"] = pd.to_numeric(nm_all["netMHCpan_EL_rank"], errors="coerce")
    nm_all = nm_all.dropna(subset=["netMHCpan_EL_rank"])
    nm_all = nm_all[nm_all["netMHCpan_EL_rank"] < float(args.el_rank_threshold)].copy()

    # Map (peptide,k) -> peptide_id, occurrence_count
    merged = nm_all.merge(
        uniq,
        on=["peptide", "k"],
        how="left",
        validate="many_to_one",
    )
    n_missing = int(merged["peptide_id"].isna().sum())
    if n_missing:
        raise ValueError(
            f"{n_missing} filtered netMHCpan rows could not be mapped to peptide_id via (peptide,k). "
            "Ensure the netMHCpan input peptides came from this exact unique_peptides.tsv."
        )

    # Fan-out to variant occurrences with positional + metadata
    out = merged.merge(pmap, on="peptide_id", how="left", validate="many_to_many")

    n_missing_var = int(out["variant_id"].isna().sum())
    if n_missing_var:
        raise ValueError(
            f"{n_missing_var} rows missing variant mapping after join. "
            "Check peptide_variant_map.tsv matches unique_peptides.tsv."
        )

    out["length"] = out["k"]

    # Optional dedup
    if args.dedup:
        dedup_keys = ["allele", "peptide_id", "variant_id", "start", "end"]
        out = out.sort_values(["netMHCpan_EL_rank"], ascending=True).drop_duplicates(dedup_keys, keep="first")

    # Output columns similar to mhcflurry output you showed.
    # criteria is included if present in peptide map.
    base_cols = [
        "allele",
        "peptide_id",
        "peptide",
        "k",
        "variant_id",
        "start",
        "end",
        "length",
        "occurrence_count",
    ]
    meta_cols = [c for c in pmap.columns if c not in {"peptide_id", "variant_id", "start", "end"}]
    score_cols = ["netMHCpan_EL_score", "netMHCpan_EL_rank"]
    if "netMHCpan_BA_score" in out.columns:
        score_cols.append("netMHCpan_BA_score")
    if "netMHCpan_BA_rank" in out.columns:
        score_cols.append("netMHCpan_BA_rank")

    out_cols = base_cols + meta_cols + score_cols
    for c in out_cols:
        if c not in out.columns:
            out[c] = pd.NA

    out = out[out_cols].sort_values(["allele", "netMHCpan_EL_rank", "variant_id", "peptide_id"]).reset_index(drop=True)

    kmer_tag = infer_kmer_range_from_df(out)
    title = args.combined_name or f"{args.st}_{kmer_tag}_netMHCpan"

    args.outdir.mkdir(parents=True, exist_ok=True)
    out_path = args.outdir / f"{title}.tsv"
    out.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote: {out_path} ({len(out)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


"""
python -m src.netmhc_filter \
  --netmhcpan-xls data/output/netmhcpan/VR5_9mer/variants_vr5_9_9mer.netmhcpan.xls \
  --peptides data/output/fragmentation/variants_vr5_9/unique_peptides.tsv \
  --peptide-map data/output/fragmentation/variants_vr5_9/peptide_variant_map.tsv \
  --el-rank-threshold 2.0 \
  --st VR5 \
  -o data/output/netmhcpan_filtered
"""