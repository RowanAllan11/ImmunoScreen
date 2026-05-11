# AAV TCR Epitope Identification Pipeline

This project identifies candidate immunogenic regions on AAV capsid proteins (e.g. VP1) by:
1) fragmenting protein sequences into overlapping k-mers,  
2) scoring peptide presentation with **MHCFlurry**,  
3) mapping peptide-level presentation scores back onto **per-residue positions**, and  
4) visualising the per-position landscape as a heatmap.

---

## Repository structure

```text
aav-tcr-epitope-pipeline/
├── src/
│   ├── fragment.py                 # FASTA IO + k-mer fragmentation + writing fragmentation TSVs
│   ├── mhcflurry_pos_mapping.py    # read TSVs, -log10 percentile scoring, peptide->position mapping
│   └── visualize_mhc_heatmap.py    # render heatmaps from position-score TSV outputs
├── scripts/
│   └── run_mapping.py              # main runner: merge k-mers, map scores, plot
├── data/
│   ├── input/
│   │   └── README.md               # place input FASTA(s) here
│   └── output/
│       ├── fragmentation/          # fragmentation TSV outputs (one per protein per k)
│       ├── mhcflurry/              # MHCFlurry prediction TSVs per protein/k/allele
│       ├── mhcflurry_position_scores/  # per-position mapped outputs (combined k-range)
│       └── plots/                  # heatmap PNGs 
├── environment.yml                 # conda env
└── README.md
```

---

## Setup

### 1) Create/activate the conda environment (need to add matplotlib!)

```bash
mamba env create -f environment.yml
mamba activate AAV
```
### 2) Install MHCflurry

```bash
pip install mhcflurry
export MHCFLURRY_DATA_DIR="/set/direc/"
mhcflurry-downloads fetch
```

---

## Inputs and outputs

### Fragmentation TSVs

Fragmentation files live under:

- `data/output/fragmentation/*.tsv`

They include columns like:

- `peptide`, `k`, `start_0`, `end_0_exclusive`, `protein_length`, etc.

### MHCFlurry TSVs

MHCFlurry prediction files live under:

- `data/output/mhcflurry/*.mhcflurry.tsv`

Expected header:

- `peptide`
- `allele`
- `mhcflurry_affinity`
- `mhcflurry_affinity_percentile`
- `mhcflurry_processing_score`
- `mhcflurry_presentation_score`
- `mhcflurry_presentation_percentile`

### Position-mapped outputs

Outputs from mapping are written to:

- `data/output/mhcflurry_position_scores/`

Each file is **one protein + one allele**, with k-mers combined across a range (default 8–11), e.g.:

- `AAV9_VP1.H2-D*b.k8-11.posmax.tsv`

Columns:

- `protein`
- `allele`
- `pos_0` (0-based residue index)
- `aa` (amino acid at that position)
- `max_log_score` (max across overlapping peptides, higher = stronger)

---

## Scoring logic

MHCFlurry provides `mhcflurry_presentation_percentile`, where **lower is better**.

This pipeline converts that percentile into an intuitive score:

```python
score = -log10(mhcflurry_presentation_percentile / 100)
```

Examples:

| percentile | score |
|---:|---:|
| 10  | 1 |
| 1   | 2 |
| 0.1 | 3 |

When mapping to positions, for each residue we take the **maximum** score across all peptides (and across all k-mer sizes in the chosen range) that overlap that residue.

---

## Running the mapping and plotting

```bash
python3 scripts/run_mapping.py
```

By default it:
- combines **k = 8..11**
- reads:
  - `data/output/fragmentation/*_{k}mer.tsv`
  - `data/output/mhcflurry/*_{k}mer.*.mhcflurry.tsv`
- writes:
  - `data/output/mhcflurry_position_scores/*.posmax.tsv`

### Also generate heatmap PNGs

Plot all proteins:

```bash
python3 scripts/run_mapping.py --plot
```

Make sure you run it from the repository root!