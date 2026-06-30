## Running the Pipeline at Individual Steps

The pipeline consists of five main stages:

1. fragmentation of input peptide into overlapping kmers
2. running mhcflurry and filtering epitopes
3. running netmhcpan and filtering epitopes
4. combine predictions and annotate variants with mutation labels
5. running bigmhc and adding immunogenicity scores

### 1) Fragment Input Sequences

This step supports both csv (library input) and fasta:

| Input Type | Description |
| --- | --- |
| `fasta` | Fragment one or more protein sequences from FASTA files. |
| `csv` | Fragment protein sequences stored in a library CSV containing one sequence per variant. |

Example library input function:

```bash
python scripts/run_fragmentation.py \
  --tag VR5_V3 \
  --input-type csv \
  --i data/input/libraries/VR5_v3_final_library_detailed.csv \
  --var-only \
  --var-start 8 \
  --var-end 24 \
  --kmers 9
```

#### Important Parameters

| Parameter | Description |
| --- | --- |
| `--var-only` | Restrict fragmentation to the variable region rather than the full protein sequence. |
| `--var-start` | Start position of the variable region (1-based, inclusive). |
| `--var-end` | End position of the variable region (1-based, inclusive). |
| `--var-mode overlap` | Generate peptides that overlap the specified variable region. |
| `--kmers` | Peptide length(s) to generate. Multiple values may be supplied (e.g. `8 9 10 11`). |
| `--metadata-cols` | Metadata columns to retain in the output files. |

For the example above, only peptides overlapping positions 8--24 of each variant sequence are generated.

### 2) Run MHCflurry Predictions

This step predicts MHC-I peptide presentation using MHCflurry:

Example:

```bash
python scripts/run_mhcflurry_pipeline.py \
  --peptides data/output/fragmentation/VR5_V3__k9/unique_peptides.tsv \
  --peptide-map data/output/fragmentation/VR5_V3__k9/peptide_variant_map.tsv \
  --alleles H-2-Db H-2-Kb \
  --affinity-percentile-threshold 2.0
```

#### Important Parameters

| Parameter | Description |
|------------|-------------|
| `--peptides` | Deduplicated peptide set generated during fragmentation. |
| `--peptide-map` | Maps predicted peptides back to individual variants. |
| `--alleles` | Text file containing one MHCflurry-supported allele per line. |
| `--affinity-percentile-threshold` | Maximum affinity percentile rank retained in filtered outputs. Lower values indicate stronger predicted binding. |
| `--outdir` | Output directory. |

### 3) Run netMHCpan Predictions

This step predicts MHC-I peptide presentation using netMHCpan:

Example:

```bash
python -m scripts.run_netmhcpan_pipeline \
  --peptides data/output/fragmentation/VR5_V3__k9/unique_peptides.tsv \
  --peptide-map data/output/fragmentation/VR5_V3__k9/peptide_variant_map.tsv \
  --alleles H-2-Db H-2-Kb \
  --kmers 9 \
  --el-rank-threshold 2.0 \
  --dedup
```

#### Important Parameters

| Parameter | Description |
|------------|-------------|
| `--peptides` | Deduplicated peptide set generated during fragmentation. |
| `--peptide-map` | Maps each unique peptide back to the variants in which it occurs. |
| `--alleles` | Text file containing one NetMHCpan-supported allele per line. |
| `--kmers` | Peptide length(s) to evaluate. |
| `--el-rank-threshold` | Maximum EL rank retained in filtered outputs. Lower values indicate stronger predicted presentation. |
| `--dedup` | Run predictions only on unique peptides to reduce runtime and output size. |
| `--outdir` | Output directory. |

### 4) Combine Predictions and Annotate Variants

Merge MHCflurry and netMHCpan predictions using shared peptide and variant identifiers. Mutations labels are tehn added by comparing each variant sequence against the wild-type variable region.

Example:

```bash
python scripts/combine_annotate.py \
  --netmhcpan-file data/output/netmhcpan/VR5_V3__k9/predictions_mapped.tsv \
  --mhcflurry-file data/output/mhcflurry/VR5_V3__k9/predictions_mapped.tsv \
  --i data/input/libraries/VR5_v3_final_library_detailed.csv \
  --var-start 8 \
  --var-end 24 \
  --wt-vr STTVTQNNNSEFAWPGA
```

#### Important Parameters

| Parameter | Description |
|------------|-------------|
| `--netmhcpan-file` | Filtered NetMHCpan TSV file. |
| `--mhcflurry-file` | Filtered MHCflurry TSV file. |
| `--i` | Original variant library CSV used to recover variant sequences. |
| `--outdir` | Directory where the combined annotated output will be written. |
| `--var-start` | Variable region start position, 1-based inclusive. Default: `8`. |
| `--var-end` | Variable region end position, 1-based inclusive. Default: `24`. |
| `--wt-vr` | Optional wild-type variable-region sequence used for mutation labelling. |

### 5) Run bigMHC

This step uses BigMHC to predict immunogenicity scores for peptides in the combined and annotated prediction table.

The script takes the combined TSV produced in the previous step, extracts the unique peptide sequences, creates a BigMHC-compatible input CSV, runs BigMHC, and then merges the resulting scores back into the original table.

Example:

```bash
python scripts/run_bigmhc.py \
  --i data/output/combined/VR5_V3__k9/combined_annotated.tsv \
  --m el \
  --t 2 \
  --d cpu \
```

#### Output

The final output is a TSV containing the original combined prediction table plus the selected BigMHC score column:

```text
data/output/bigmhc/
└── .tsv
```

---

## Typical Workflow

```text
Input sequences
      ↓
Fragmentation
      ↓
Unique peptide set + peptide-variant map
      ↓
MHCflurry prediction      NetMHCpan prediction
      ↓                         ↓
Filtered MHCflurry TSV     Filtered NetMHCpan TSV
      ↓                         ↓
      Combined peptide-MHC prediction table
      ↓
Variable-region mutation annotation
      ↓
BigMHC immunogenicity scoring
      ↓
Final candidate epitope table
      ↓
Analysis ......
```