from pathlib import Path
from typing import Sequence


def clean_label(value: str) -> str:
    """
    Make a user-provided label safe for filenames.
    """
    return (
        str(value)
        .strip()
        .replace(" ", "_")
        .replace("/", "-")
        .replace("\\", "-")
    )


def format_kmer_label(kmers: Sequence[int]) -> str:
    """
    Convert kmers into a compact label.

    Examples:
      [9] -> k9
      [8, 9, 10, 11, 12, 13, 14, 15] -> k8-15
      [8, 9, 11] -> k8_9_11
    """
    kmers = sorted(set(int(k) for k in kmers))

    if not kmers:
        raise ValueError("No kmers provided.")

    if len(kmers) == 1:
        return f"k{kmers[0]}"

    expected = list(range(kmers[0], kmers[-1] + 1))
    if kmers == expected:
        return f"k{kmers[0]}-{kmers[-1]}"

    return "k" + "_".join(str(k) for k in kmers)


def make_run_label(tag: str, kmers: Sequence[int]) -> str:
    """
    Standard run label used across the pipeline.

    Example:
      VR6__k9
      VR6__k8-15
    """
    return f"{clean_label(tag)}__{format_kmer_label(kmers)}"
