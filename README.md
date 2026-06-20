# Raman Spectroscopy Analysis Pipeline

Python pipeline for extracellular-vesicle Raman spectroscopy across three independent tracks:

| Track | Entry point | Main output |
|---|---|---|
| EDA | `uv run scripts/run_eda.py` | `Results_control_vs_disease/` or `Results_intra_disease/` |
| Spectral plots and QC | `uv run scripts/run_spectral_plots.py` | `Output_plot/qc/`, `Output_plot/spectral_plots/` |

## Data Model

Canonical hierarchy:

```text
Subject -> sample_XX -> measurement -> range {1300, 2850} -> pixels
```

## Data Sources

The analysis stack now supports three explicit source levels:

- `raw_text`: original Renishaw `.txt` spectra plus `data/clinical_data.csv`
- `parsed`: materialized metadata and spectra before scientific preprocessing
- `preprocessed`: materialized metadata and spectra after the preprocessing pipeline

Default source: `preprocessed`.

Generated stores use:

- Parquet for canonical tables
- NPZ for dense NumPy arrays
- JSON for manifests
- CSV only as optional inspection mirrors

Materialized layout:

```text
data/
  raw/
  clinical_data.csv
  parsed/
    manifest.json
    file_index.parquet
    clinical_subjects.parquet
    area_metadata.parquet
    spectra_1300.npz
    spectra_2850.npz
    area_summaries_1300.npz
    area_summaries_2850.npz
    whole_area_store.npz
    cohort/
      control_vs_disease/
      disease_only/
  preprocessed/
    ...
```

The ingestion source of truth is:

- `src/metadata/parser.py`
- `src/io/file_crawler.py`
- `src/io/txt_reader.py`

Acquisition-mode tokens (`SS`, `FP`) in source filenames are matched for backward compatibility but are not returned or persisted as metadata.

## Setup

Requirements:

- Python >= 3.12
- Dependencies declared in `pyproject.toml`
- `uv` recommended

Install:

```bash
uv pip install -e .
```



Default data roots:

- `data/`: dataset

To manage generated stores:

```bash
uv run scripts/manage_data_store.py build --source parsed
uv run scripts/manage_data_store.py build --source preprocessed
uv run scripts/manage_data_store.py validate --source parsed
uv run scripts/manage_data_store.py validate --source preprocessed
uv run scripts/manage_data_store.py info --source preprocessed
uv run scripts/manage_data_store.py clean --source parsed
```

## Run The Pipelines

EDA:

```bash
uv run scripts/run_eda.py --range 1300
uv run scripts/run_eda.py --range 2850
uv run scripts/run_eda.py --range both
uv run scripts/run_eda.py --range 1300 --max-subjects 10
```

When `--range both` is selected, the EDA runner executes the ordinary PCA path
and then the subject-level Biological PCA layer once. Canonical Biological PCA
outputs are written into the range-local result folders:

- `01_fingerprint_1300/06_biological_pca/`
- `02_ch_stretch_2850/06_biological_pca/`
- `03_whole_spectrum_1300_plus_2850/04_biological_pca/`

The root-level `biological_pca/` directory is retained as a compatibility and
navigation index; there is no separate Biological PCA command.

Spectral comparison and QC plots:

```bash
uv run scripts/run_spectral_plots.py
uv run scripts/run_spectral_plots.py --max-subjects 20
uv run scripts/run_spectral_plots.py --outdir Output_plot
```

The figures written under `Output_plot/spectral_plots/` are post-preprocessing waveforms produced after the Raman preprocessing pipeline, not raw spectra.



`doc/Analysis_*` directories are generated artifacts and are not part of the maintained documentation set.

## Disease-Only vs. Disease-vs-Control EDA

Two separate config files select the analysis mode. Each writes to its own output root so results never overwrite each other.

**Disease vs. control** (default grouping, intra-disease correlation off):

```bash
uv run scripts/run_eda.py --config config/eda_disease_control.yaml --range 1300
uv run scripts/run_eda.py --config config/eda_disease_control.yaml --range 2850
uv run scripts/run_eda.py --config config/eda_disease_control.yaml --range both
```

Output root: `Results_control_vs_disease/`

**Disease-only** (intra-disease Spearman clinical correlations enabled; requires `data/clinical_data_clean.csv`):

```bash
uv run scripts/run_eda.py --config config/eda_disease_only.yaml --range 1300
uv run scripts/run_eda.py --config config/eda_disease_only.yaml --range 2850
uv run scripts/run_eda.py --config config/eda_disease_only.yaml --range both
```

Output root: `Results_intra_disease/`

To run both modes back-to-back for all ranges:

```bash
for CFG in config/eda_disease_control.yaml config/eda_disease_only.yaml; do
  for RANGE in 1300 2850 both; do
    uv run scripts/run_eda.py --config $CFG --range $RANGE
  done
done
```

## Configuration

`config/settings.yaml` is the shared base config. Track configs use `_extends: settings.yaml` and are the single source of truth for track-owned sections:

- `config/eda.yaml`
- `config/plots.yaml`

Current defaults to note:

- `data.source`: active source, one of `raw_text`, `parsed`, `preprocessed`
- `data.auto_build`: auto-build missing derived stores
- `data.require_valid_manifest`: enforce manifest validation
- `data.stale_manifest_policy`: default `error`; stale stores do not rebuild silently
- `data.write_csv_mirrors`: optional human-readable table copies beside Parquet
- Use `--source ...` on supported scripts to override the config for a run

- During startup, analysis prints the active data source, path, formats, manifest hash, counts, and whether the source was loaded or built.
- Normal analysis no longer writes `Results_*/00_cohort/`; cohort tables are generated into `data/parsed/cohort/` and `data/preprocessed/cohort/`.
- Each analysis output root now includes `run_manifest.json`, `data_manifest_snapshot.json`, and `config_snapshot.yaml`.

- `config/eda.yaml` owns `run.ranges`: `["1300", "2850", "both"]`.
- EDA preprocessing order is: dead/anomalous screen → cosmic-ray detection → optional cosmic-ray suppression → spectral quality filter → edge trim → baseline correction → isolated spike repair → optional smoothing → vector normalization.
- Cosmic-ray detection always runs; suppression is controlled by `preprocessing.cosmic_ray.enabled`.
- Additive capabilities (all off by default): region weighting/exclusion, pixel-cluster interpretation, QC-cluster analysis, region-influence robustness, figure gating via `analysis.figures`, and ML latent-structure reporting.
- `output.keep_preprocessing_artifacts` controls whether per-area preprocessing artifacts are kept in the active EDA root under `processed/preprocessing_artifacts`.
- `output.keep_whole_spectra_cache` controls whether the paired-area whole-spectrum cache is saved in the active EDA root under `processed/whole_spectra`.



## Scientific Caveats

- Inter-subject variance can reflect biology, SEC fraction, concentration, column batch or age, lipoprotein contamination, and sample volume.
- Sex imbalance can produce pooled effects that do not hold in sex-stratified results.
- QC flags indicate reliability concerns, not automatic exclusion rules.
- EDA is exploratory. Do not interpret unsupervised structure as diagnostic classification evidence; this repository does not perform diagnostic classification or prediction.
