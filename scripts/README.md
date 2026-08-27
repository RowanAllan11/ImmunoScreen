# Running individual MHC-I pipeline stages

The supported MHC-I workflow has four stages. Run the complete configured
workflow with `python scripts/run_pipeline.py --config <config.yaml>`.

## 1. Fragment input sequences

```bash
python scripts/run_fragmentation.py \
  --tag VR6 --input-type csv \
  --i data/input/libraries/VR6/VR6_v1_library.csv \
  --sequence-col aa_sequence --id-col gene_id \
  --var-only --var-start 526 --var-end 542 --var-mode overlap \
  --kmers 9
```

## 2. Run MHCflurry

```bash
python scripts/run_mhcflurry_pipeline.py \
  --peptides data/output/fragmentation/VR6__k9/unique_peptides.tsv \
  --peptide-map data/output/fragmentation/VR6__k9/peptide_variant_map.tsv \
  --alleles H2-D*b H2-K*b \
  --affinity-percentile-threshold 2.0
```

## 3. Run netMHCpan

```bash
python -m scripts.run_netmhcpan_pipeline \
  --peptides data/output/fragmentation/VR6__k9/unique_peptides.tsv \
  --peptide-map data/output/fragmentation/VR6__k9/peptide_variant_map.tsv \
  --alleles H-2-Db H-2-Kb \
  --kmers 9 --el-rank-threshold 2.0 --dedup
```

## 4. Combine and annotate

```bash
python scripts/combine_annotate.py \
  --netmhcpan-file data/output/netmhcpan/VR6__k9/predictions_mapped.tsv \
  --mhcflurry-file data/output/mhcflurry/VR6__k9/predictions_mapped.tsv \
  --i data/input/libraries/VR6/VR6_v1_library.csv \
  --library-id-col gene_id --seq-col aa_sequence \
  --var-start 526 --var-end 542 --wt-vr SHKEGEDRFFPLSGSLI
```

The final output is
`data/output/combined/<run-label>/combined_annotated.tsv`.

## Separate MHC-II workflow

NetMHCIIpan is intentionally maintained as a separate pipeline:

```bash
python scripts/run_mhcii_pipeline.py \
  --config configs/vr6_k15_netmhciipan.yaml
```
