# Input libraries

Input sequence libraries are deliberately excluded from version control. Place
local CSV files beneath `data/input/libraries/` and make their paths and column
names match the selected YAML configuration.

## Included configuration schemas

The supplied VR4 and VR6 variant configurations expect:

```text
data/input/libraries/VR4/VR4_v1_library.csv
data/input/libraries/VR6/VR6_v1_library.csv
```

| Column | Required | Description |
| --- | --- | --- |
| `gene_id` | Yes | Unique variant identifier. |
| `aa_sequence` | Yes | Full amino-acid sequence used for fragmentation. |
| `criteria` | Yes for the supplied configs | Metadata copied into fragmentation outputs. |

The supplied WT configurations expect
`data/input/libraries/WT/AAV9_WT.csv` with these columns:

| Column | Required | Description |
| --- | --- | --- |
| `Geneid` | Yes | WT record identifier. |
| `twist_seq_prot` | Yes | Full WT amino-acid sequence. |
| `criteria` | Yes for the supplied configs | Metadata copied into fragmentation outputs. |

## Input rules

- Identifiers should be non-empty and unique within a library.
- Sequences should contain amino-acid letters without gaps or stop codons in
  the analysed variable region.
- Sequences must be long enough to cover the 1-based variable-region
  coordinates configured in YAML.
- Blank identifier or sequence values are skipped during fragmentation.
- Additional metadata columns are allowed. To retain one in the fragmentation
  outputs, add its name under `fragmentation.metadata_columns` in the config.

Custom column names are supported by changing
`fragmentation.variant_id_column`, `fragmentation.sequence_column`,
`combine.library_id_column` and `combine.sequence_column` in the corresponding
MHC-I config. For MHC-II, change the equivalent fields under `fragmentation`
and `annotate`.

Do not commit restricted or unpublished sequence libraries. The repository's
`.gitignore` excludes files beneath `data/input/`, except for this guide.
