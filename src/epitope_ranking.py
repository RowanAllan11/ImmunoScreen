from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .mhcflurry_pos_mapping import load_fragmentation_coords, read_tsv_dicts


@dataclass(frozen=True)
class RankedEpitope:
    allele: str
    protein: str
    peptide: str
    start: int  # 0-based
    end: int  # inclusive, 0-based
    length: int
    affinity_percentile: float
    presentation_percentile: Optional[float]
    supporting_kmers: int  # number of peptides (any k) overlapping this epitope span for same allele+protein


def discover_mhcflurry_tsvs(mhcflurry_dir: Path, kmers: Iterable[int]) -> List[Path]:
    files: List[Path] = []
    for k in kmers:
        files.extend(sorted(mhcflurry_dir.glob(f"*_{k}mer.*.mhcflurry.tsv")))
    return files


def infer_fragment_path_from_mhcflurry_path(mhcflurry_path: Path, fragment_dir: Path) -> Path:
    m = re.match(r"^(?P<fragstem>.+?_\d+mer)\..+?\.mhcflurry\.tsv$", mhcflurry_path.name)
    if not m:
        raise ValueError(f"Unexpected mhcflurry output filename: {mhcflurry_path.name}")
    return fragment_dir / f"{m.group('fragstem')}.tsv"


def infer_protein_from_fragmentation_path(path: Path) -> str:
    m = re.match(r"^(?P<protein>.+?)_\d+mer\.tsv$", path.name)
    return m.group("protein") if m else path.stem


def infer_k_from_fragmentation_path(path: Path) -> Optional[int]:
    m = re.match(r"^.+?_(?P<k>\d+)mer\.tsv$", path.name)
    return int(m.group("k")) if m else None


def _safe_float(x: object) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _pick_pct(row: dict, keys: List[str]) -> Optional[float]:
    for k in keys:
        if k in row:
            v = _safe_float(row.get(k))
            if v is not None:
                return v
    return None


def _overlap_len(a_start: int, a_end_incl: int, b_start: int, b_end_incl: int) -> int:
    lo = max(a_start, b_start)
    hi = min(a_end_incl, b_end_incl)
    return max(0, hi - lo + 1)


def greedy_nonoverlap(
    epitopes: List[RankedEpitope],
    *,
    min_overlap_aa: int = 1,
) -> List[RankedEpitope]:
    if min_overlap_aa < 1:
        raise ValueError("min_overlap_aa must be >= 1")

    kept: List[RankedEpitope] = []
    kept_by_group: Dict[Tuple[str, str], List[RankedEpitope]] = {}

    for e in epitopes:
        g = (e.allele, e.protein)
        prev = kept_by_group.get(g, [])
        ok = True
        for k in prev:
            if _overlap_len(e.start, e.end, k.start, k.end) >= min_overlap_aa:
                ok = False
                break
        if ok:
            kept.append(e)
            prev.append(e)
            kept_by_group[g] = prev

    return kept


def _rank_score_bucket(affinity_percentile: float) -> int:
    """
    Bucket affinity_percentile into 1..5, where 1 is strongest.
      1: <= 0.1
      2: <= 0.5
      3: <= 1
      4: <= 2
      5: <= 5
    Caller typically filters to <=2 anyway, so you will mostly see 1..4.
    """
    p = float(affinity_percentile)
    if p <= 0.1:
        return 1
    if p <= 0.5:
        return 2
    if p <= 1.0:
        return 3
    if p <= 2.0:
        return 4
    return 5


def rank_mhcflurry_epitopes(
    *,
    fragment_dir: Path,
    mhcflurry_dir: Path,
    kmers: Iterable[int],
    affinity_threshold: float,
    top_n_per_allele_protein: int = 0,
    dedupe_greedy: bool = False,
    dedupe_min_overlap_aa: int = 1,
) -> List[RankedEpitope]:
    kmers = tuple(int(k) for k in kmers)
    mhc_paths = discover_mhcflurry_tsvs(mhcflurry_dir, kmers)
    if not mhc_paths:
        raise FileNotFoundError(f"No mhcflurry TSVs found in {mhcflurry_dir} for kmers={kmers}")

    coords_cache: Dict[Path, Dict[str, object]] = {}

    affinity_keys = ["mhcflurry_affinity_percentile", "affinity_percentile"]
    presentation_keys = ["mhcflurry_presentation_percentile", "presentation_percentile"]

    # Collect candidates per (allele, protein) so we can compute supporting_kmers within the same group.
    candidates_by_group: Dict[Tuple[str, str], List[RankedEpitope]] = {}

    for mhc_tsv in mhc_paths:
        frag_tsv = infer_fragment_path_from_mhcflurry_path(mhc_tsv, fragment_dir)
        if frag_tsv not in coords_cache:
            if not frag_tsv.exists():
                raise FileNotFoundError(f"Missing fragmentation TSV for {mhc_tsv.name}: expected {frag_tsv}")
            coords_cache[frag_tsv] = load_fragmentation_coords(str(frag_tsv))

        coords_by_pep = coords_cache[frag_tsv]
        protein = infer_protein_from_fragmentation_path(frag_tsv)
        k_from_name = infer_k_from_fragmentation_path(frag_tsv)

        for row in read_tsv_dicts(str(mhc_tsv)):
            pep = (row.get("peptide") or "").strip()
            allele = (row.get("allele") or "").strip()
            if not pep or not allele:
                continue

            affinity_pct = _pick_pct(row, affinity_keys)
            if affinity_pct is None:
                continue
            if float(affinity_pct) > float(affinity_threshold):
                continue

            presentation_pct = _pick_pct(row, presentation_keys)

            coord = coords_by_pep.get(pep)
            if coord is None:
                continue

            start_0 = int(getattr(coord, "start_0"))
            end_0_excl = int(getattr(coord, "end_0_exclusive"))
            end_0_inclusive = end_0_excl - 1
            length = len(pep) if k_from_name is None else int(k_from_name)

            g = (allele, protein)
            candidates_by_group.setdefault(g, []).append(
                RankedEpitope(
                    allele=allele,
                    protein=protein,
                    peptide=pep,
                    start=start_0,
                    end=end_0_inclusive,
                    length=length,
                    affinity_percentile=float(affinity_pct),
                    presentation_percentile=float(presentation_pct) if presentation_pct is not None else None,
                    supporting_kmers=0,  # filled later
                )
            )

    # Flatten and sort by best affinity overall (NOT by allele), as requested.
    all_candidates: List[RankedEpitope] = []
    for g, eps in candidates_by_group.items():
        eps.sort(key=lambda e: (e.affinity_percentile, e.start, e.length, e.peptide))
        all_candidates.extend(eps)

    all_candidates.sort(key=lambda e: (e.affinity_percentile, e.protein, e.allele, e.start, e.length, e.peptide))

    # Apply greedy de-dupe within each (allele, protein), but keep global ordering afterwards.
    if dedupe_greedy:
        kept_by_group: Dict[Tuple[str, str], List[RankedEpitope]] = {}
        for g, eps in candidates_by_group.items():
            eps_sorted = sorted(eps, key=lambda e: (e.affinity_percentile, e.start, e.length, e.peptide))
            kept_by_group[g] = greedy_nonoverlap(eps_sorted, min_overlap_aa=int(dedupe_min_overlap_aa))

        kept: List[RankedEpitope] = []
        for g, kept_list in kept_by_group.items():
            # compute supporting kmers for each kept epitope by overlap within the SAME group
            group_all = candidates_by_group[g]
            for e in kept_list:
                support = 0
                for c in group_all:
                    if _overlap_len(e.start, e.end, c.start, c.end) > 0:
                        support += 1
                kept.append(
                    RankedEpitope(
                        allele=e.allele,
                        protein=e.protein,
                        peptide=e.peptide,
                        start=e.start,
                        end=e.end,
                        length=e.length,
                        affinity_percentile=e.affinity_percentile,
                        presentation_percentile=e.presentation_percentile,
                        supporting_kmers=support,
                    )
                )

        # Global sort by affinity (NOT by allele)
        kept.sort(key=lambda e: (e.affinity_percentile, e.protein, e.allele, e.start, e.length, e.peptide))
        out = kept
    else:
        # If not deduping, still compute supporting_kmers for each row (how many candidates overlap it in same group)
        out: List[RankedEpitope] = []
        for (allele, protein), eps in candidates_by_group.items():
            for e in eps:
                support = 0
                for c in eps:
                    if _overlap_len(e.start, e.end, c.start, c.end) > 0:
                        support += 1
                out.append(
                    RankedEpitope(
                        allele=e.allele,
                        protein=e.protein,
                        peptide=e.peptide,
                        start=e.start,
                        end=e.end,
                        length=e.length,
                        affinity_percentile=e.affinity_percentile,
                        presentation_percentile=e.presentation_percentile,
                        supporting_kmers=support,
                    )
                )
        out.sort(key=lambda e: (e.affinity_percentile, e.protein, e.allele, e.start, e.length, e.peptide))

    if top_n_per_allele_protein and top_n_per_allele_protein > 0:
        trimmed: List[RankedEpitope] = []
        counts: Dict[Tuple[str, str], int] = {}
        for e in out:
            k = (e.allele, e.protein)
            counts[k] = counts.get(k, 0) + 1
            if counts[k] <= int(top_n_per_allele_protein):
                trimmed.append(e)
        out = trimmed

    return out


def rank_score(epitope: RankedEpitope) -> int:
    return _rank_score_bucket(epitope.affinity_percentile)