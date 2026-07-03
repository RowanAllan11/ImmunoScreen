# AAV TCR Epitope Identification Pipeline

This pipeline identifies canditate immunogenic regions and mutations on input peptides sequence (currently for MHC-I, soon to be Class II.....) by:

1) fragmenting sequences into overlapping k-mers,  
2) scoring peptide presentation with **MHCFlurry** & **netMHCpan**,  
3) scoring immunogenicity with **bigMHC**  
4) statisitical analysis of variant specific mutations and tissue expression modelling

---

## Repository structure

```text
aav-tcr-epitope-pipeline/
├── configs/
│   ├── (library specific) .yaml
├── src/
│   ├── fragmentation.py                 
│   ├── mhcflurry.py    
│   ├── netmhcpan.py.py
│   ├── combine_predictions.py
│   ├── mutation_label.py
│   └── bigmhc.py
├── scripts/
│   ├── run_fragmentation.py
│   ├── run_netmhcpan_pipeline.py
│   ├── run_mhcflurry_pipeline.py
│   ├── combine_annotation.py
│   └── run_bigmhc.py
├── analysis/
│   └── in progess .........        
├── data/
│   ├── input/
│   │   ├── libraries/              # place input VR libraries here
│   │   ├── expression/             # place tissue expression datasets here
│   │   └── README.md               
│   └── output/
│       ├── fragmentation/          # fragmentation TSV outputs - all, unique and mapping information
│       ├── mhcflurry/              # mhcflurry prediction TSVs
│       ├── mhcflurry_filtered/     # mhcflurry epitopes which passed minimum threshold
│       ├── netmhcpan/              # netmhcpan predictions
│       ├── netmhcpan_filtered/     # netmhcpan epitopes which passed minimum threshold 
│       └── combined                # final output dataframe 
├── environment.yml            
└── README.md
```

---

## Setup

### 1) Create/activate the conda environment

```bash
mamba env create -f environment.yml
mamba activate AAV
```
### 2) Install Immunogenicity Tools

mhcFlurry:

```bash
pip install mhcflurry
export MHCFLURRY_DATA_DIR="/set/direc/"
mhcflurry-downloads fetch
```

bigMHC:

```bash
mkdir tools
cd tools
git clone https://github.com/karchinlab/bigmhc.git
```

Install netMHCpan-4.2 from website:

https://services.healthtech.dtu.dk/services/NetMHCpan-4.2/

---

## Running the Pipeline

The pipeline consists of five main stages:

1. fragmentation of input peptide into overlapping kmers
2. running mhcflurry and filtering epitopes
3. running netmhcpan and filtering epitopes
4. combine predictions and annotate variants with mutation labels
5. running bigmhc and adding immunogenicity scores

### Configuration

Each variant library is controlled by a YAML configuration file stored in ```configs/```

The configuration defines:

- the input library
- kmer lengths
- variable-region coordinates
- MHC alleles
- tool-specifc settings
- which stages of the pipeline to run

### Running the pipeline

Run the complete pipeline using:

```bash
python scripts/run_pipeline.py --config configs/vr6_k9.yaml
```

To run individual stages refer to README in ```scripts/```

#### Output

The final output is a TSV containing the original combined prediction table plus the selected BigMHC score column:

```text
data/output/bigmhc/.../
└── predictions_mapped.tsv
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