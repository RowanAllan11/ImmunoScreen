from __future__ import annotations
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.fragment import write_fragment_outputs


def main() -> int:
    # Repo-relative paths (run from repo root)
    fasta_path = Path("data/input/AAV9_VP1.fasta")
    out_dir = Path("data/output/fragmentation")
    out_prefix = "AAV9_VP1"  # or None to use fasta stem automatically

    for k in (8, 9, 10, 11, 13, 15, 17):
        tsv_path, txt_path = write_fragment_outputs(
            fasta_path=fasta_path,
            k=k,
            out_dir=out_dir,
            out_prefix=out_prefix,
        )
        print(f"[k={k}] wrote {tsv_path}")
        print(f"[k={k}] wrote {txt_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())