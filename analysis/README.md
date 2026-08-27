# ImmunoScreen analysis commands

Run these commands from the repository root after activating the environment:

```bash
mamba activate AAV
```

The scoring commands convert peptide/window-level workflow predictions into one
variant-level table per allele. The mutation analysis consumes those
allele-specific tables.

## Variant immunogenicity scoring

All three scoring commands use the same basic interface:

```text
--variant-input  Workflow prediction table for the variant library
--wt-input       Matching WT table (WT-relative scorers only)
--output-root    Parent output directory (optional)
--run-label      Name used for output directories (optional)
```

When `--run-label` is omitted, it is inferred from the input directory. For
example, `VR6__k9` becomes `VR6_K9`. Output is always allele-specific:

```text
OUTPUT_ROOT/RUN_LABEL_ALLELE/variant_immunogenicity_scores.tsv
```

### Absolute MHC-I scores

Use `vi_scoring.py` when scores should describe each variant without comparison
to WT:

```bash
python analysis/vi_scoring.py \
  --variant-input data/output/combined/VR6__k9/combined_annotated.tsv
```

Default output root:

```text
data/output/variant_immunogenicity_scores/
```

The input must be the MHC-I `combined_annotated.tsv` produced after combining
NetMHCpan and MHCflurry predictions.

### WT-relative MHC-I scores

Use `vi_scoring_wt.py` for mutation effects relative to matched WT peptide
windows. Variant and WT runs must use the same VR boundaries, k-mer length and
alleles.

VR6 example:

```bash
python analysis/vi_scoring_wt.py \
  --variant-input data/output/combined/VR6__k9/combined_annotated.tsv \
  --wt-input data/output/combined/WT_vr6__k9/combined_annotated.tsv
```

Default output root:

```text
data/output/variant_immunogenicity_scores_wt/
```

Positive WT-relative continuous values mean stronger predicted presentation
than WT. The main mutation-analysis outcomes produced by this scorer are:

- `netMHCpan_net_pass_change`
- `MHCflurry_net_pass_change`
- `netMHCpan_mean_window_improvement`
- `MHCflurry_mean_window_improvement`

### WT-relative MHC-II scores

Use `vi_scoring_wt_mhcii.py` with the annotated NetMHCIIpan tables:

```bash
python analysis/vi_scoring_wt_mhcii.py \
  --variant-input data/output/netmhciipan_annotated/VR6__k15/predictions_mapped_annotated.tsv \
  --wt-input data/output/netmhciipan_annotated/WT_vr6__k15/predictions_mapped_annotated.tsv
```

Default output root:

```text
data/output/variant_immunogenicity_scores_wt_mhcii/
```

The principal MHC-II outcomes are:

- `netMHCIIpan_net_binder_change`
- `netMHCIIpan_mean_window_improvement`

## Mutation immunogenicity and physicochemical analysis

`mutation_immunogenicity_hydrophobicity_wt.py` performs the complete mutation
analysis for one designated VR. Every input must represent that same VR, with
one `variant_immunogenicity_scores.tsv` file per allele.

For each allele and selected score it first fits the multivariable model:

```text
variant score ~ one-hot indicators for all retained mutations
```

It bootstraps variants to estimate mutation coefficients, confidence intervals and sign stability. It then calculates allele correlations, mutation heatmaps and the median-effect versus allele-IQR scatter plot.

For each score, the physicochemical model uses one row per mutation:

```text
median mutation coefficient across alleles
    ~ hydrophobicity change
    + charge change
    + residue-volume change
    + categorical VR position
```

### VR6 MHC-I example

VR6 spans absolute positions 526–542, so its position offset is 525:

```bash
python analysis/mutation_immunogenicity_hydrophobicity_wt.py \
  --inputs data/output/variant_immunogenicity_scores_wt/VR6_K9_H2-* \
  --outdir data/output/mutation_analysis/VR6_K9 \
  --wt-sequence SHKEGEDRFFPLSGSLI \
  --position-offset 525
```

The four MHC-I WT-relative outcomes listed above are used by default.

### MHC-II example

MHC-II score columns must be selected explicitly because the CLI defaults are
the four MHC-I outcomes:

```bash
python analysis/mutation_immunogenicity_hydrophobicity_wt.py \
  --inputs data/output/variant_immunogenicity_scores_wt_mhcii/VR6_K15_H2-* \
  --outdir data/output/mutation_analysis/VR6_K15_MHCII \
  --wt-sequence SHKEGEDRFFPLSGSLI \
  --position-offset 525 \
  --outcomes \
    netMHCIIpan_net_binder_change \
    netMHCIIpan_mean_window_improvement
```

### Analysis options

- `--outcomes`: score columns modelled independently.
- `--min-mutation-count`: minimum number of variants carrying a mutation;
  default `20`.
- `--bootstraps`: number of variant-bootstrap replicates; default `500`.
- `--seed`: random seed for reproducible bootstrap results; default `42`.
- `--min-sign-stability`: bootstrap sign-stability threshold used for heatmap
  annotation and support summaries; default `0.8`.
- `--position-offset`: value added to a VR-relative mutation position. For a VR
  beginning at absolute position `N`, use `N - 1`.

### Output structure

The analysis writes TSV tables under `OUTDIR/tables/` and both PNG and PDF
plots under `OUTDIR/plots/`. Principal outputs include:

- Allele-specific mutation coefficients and model-fit summaries.
- Mutation heatmaps for every allele and score.
- Cross-allele coefficient-correlation heatmaps.
- Median mutation-effect versus allele-IQR scatter plots.
- Physicochemical model data, coefficient summaries and partial R² values.
- A separate vertical partial-R² plot for each score.
- A multi-panel adjusted positional-effects plot across scores.

For the complete CLI reference at any time:

```bash
python analysis/vi_scoring.py --help
python analysis/vi_scoring_wt.py --help
python analysis/vi_scoring_wt_mhcii.py --help
python analysis/mutation_immunogenicity_hydrophobicity_wt.py --help
```
