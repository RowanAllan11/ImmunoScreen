# ImmunoScreen

ImmunoScreen predicts peptide presentation from AAV protein libraries. It
contains two separate workflows:

- **MHC-I:** fragmentation → MHCflurry → netMHCpan → combine and annotate
- **MHC-II:** fragmentation → netMHCIIpan → annotate

Pipeline behavior is controlled through the YAML files in `configs/`. These
files let you change the input library and column names, fragmentation k-mer
size, variable-region coordinates and overlap mode, MHC alleles, predictor
executable paths, binding-rank thresholds, output location, and which pipeline
stages are enabled.

The included configurations cover the VR4 and VR6 libraries and their
corresponding wild-type sequences.

## Overview

![ImmunoScreen MHC-I and MHC-II workflow](docs/immunoscreen.drawio.png)

## Repository layout

```text
configs/       VR4, VR6, and wild-type pipeline configurations
src/           Reusable pipeline implementation
scripts/       Pipeline entry points and individual-stage commands
tests/         Tests that do not require the external DTU predictors
data/input/    Local input libraries
data/output/   Generated pipeline outputs
tools/         Local netMHCpan and netMHCIIpan installations
```

## Requirements

- Conda or Mamba
- netMHCpan 4.2 for the MHC-I workflow
- netMHCIIpan 4.3 for the MHC-II workflow

The DTU predictors are licensed separately and must be downloaded and installed
by each user.

## Environment setup

Create and activate the project environment:

```bash
mamba env create -f environment.yml
mamba activate AAV
```

Download the MHCflurry model data:

```bash
mhcflurry-downloads fetch
```

Install the DTU predictors at the paths used by the included configurations:

```text
tools/netMHCpan-4.2/netMHCpan
tools/netMHCIIpan-4.3/netMHCIIpan
```

Change the corresponding `executable` value in the YAML configuration.

## Input libraries

Place the library files at the paths referenced by the configurations:

```text
data/input/libraries/VR4/VR4_v1_library.csv
data/input/libraries/VR6/VR6_v1_library.csv
data/input/libraries/WT/AAV9_WT.csv
```

## Run the MHC-I pipeline

Example: Run VR6 and its wild-type reference:

```bash
python scripts/run_pipeline.py --config configs/vr6_k9.yaml
python scripts/run_pipeline.py --config configs/wt_vr6_k9.yaml
```

### MHC-I outputs

For a VR6 run, the main outputs are:

```text
data/output/
├── fragmentation/VR6__k9/
│   ├── all_fragments.tsv
│   ├── unique_peptides.tsv
│   └── peptide_variant_map.tsv
├── mhcflurry/VR6__k9/
│   ├── input.csv
│   ├── predictions.tsv
│   └── predictions_mapped.tsv
├── netmhcpan/VR6__k9/
│   ├── raw/
│   └── predictions_mapped.tsv
└── combined/VR6__k9/
    └── combined_annotated.tsv
```

Important final-table fields include peptide and variant identifiers, window
coordinates, allele, predictor scores, threshold-pass flags, `VR_sequence`,
`VR_mutation`, and `peptide_mutation`.

## Run the MHC-II pipeline

Run VR6 and its wild-type reference:

```bash
python scripts/run_mhcii_pipeline.py --config configs/vr6_k15_netmhciipan.yaml
python scripts/run_mhcii_pipeline.py --config configs/wt_vr6_k15_netmhciipan.yaml
```

### MHC-II outputs

For a VR6 run, the main outputs are:

```text
data/output/
├── fragmentation/VR6__k15/
│   ├── all_fragments.tsv
│   ├── unique_peptides.tsv
│   └── peptide_variant_map.tsv
├── netmhciipan/VR6__k15/
│   ├── raw/
│   └── predictions_mapped.tsv
└── netmhciipan_annotated/VR6__k15/
    └── predictions_mapped_annotated.tsv
```

The mapped table includes the predicted binding core, EL score, EL rank, the
configured EL-rank binder flag, variant mapping, and mutation annotations.

## Variant scoring and mutation analysis

The commands in `analysis/` turn workflow predictions into allele-specific,
variant-level presentation scores. They support absolute MHC-I scoring,
WT-relative MHC-I scoring and WT-relative MHC-II scoring.

The mutation framework then fits multivariable one-hot mutation models for each
allele and score, compares mutation effects across alleles, and evaluates their
association with hydrophobicity, side-chain charge, residue volume and VR
position.

See the [analysis guide](analysis/README.md) for input requirements, usage
examples, configurable outcomes and the generated tables and plots.

## Configuration controls

Each YAML file defines:

- the input library and its ID/sequence columns;
- peptide length and variable-region coordinates;
- alleles in each predictor's required notation;
- predictor executable paths and rank thresholds;
- output root; and
- the wild-type variable-region sequence used for annotation.

Individual stages can be skipped by setting their `enabled` value to `false`.
When skipping an upstream stage, its expected output must already exist at the
configured output location.

## Citations

If you use ImmunoScreen, cite the prediction methods used in your analysis:

- O'Donnell, T. J., Rubinsteyn, A. and Laserson, U. (2020). MHCflurry 2.0:
  Improved pan-allele prediction of MHC class I-presented peptides by
  incorporating antigen processing. *Cell Systems*, 11, 42–48.e7.
- Nilsson, J. B., Greenbaum, J., Peters, B. and Nielsen, M. (2025).
  NetMHCpan-4.2: Improved prediction of CD8+ epitopes by use of transfer
  learning and structural features. *Frontiers in Immunology*, 16, 1616113.
  [doi:10.3389/fimmu.2025.1616113](https://doi.org/10.3389/fimmu.2025.1616113)

NetMHCpan and NetMHCIIpan are separately licensed DTU software and are not
distributed with this repository.
