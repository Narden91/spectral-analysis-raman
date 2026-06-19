# Raman Spectroscopy Analysis Pipeline

Python pipeline for extracellular-vesicle Raman spectroscopy across three independent tracks:

| Track | Entry point | Main output |
|---|---|---|
| EDA | `uv run scripts/run_eda.py` | `output_EDA/` |
| Spectral plots and QC | `uv run scripts/run_spectral_plots.py` | `Output_plot/qc/`, `Output_plot/spectral_plots/` |
| ML | `uv run scripts/run_ml_pipeline.py` or `uv run -m src.main` | `output_ML/` |

All ML splitting is subject-level. Measurements, slides, maps, and pixels may become training or analysis items only after the subject split is fixed.

## Data Model

Canonical hierarchy:

```text
Subject -> sample_XX -> measurement -> range {1300, 2850} -> pixels
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

Prepare data:

```bash
uv run -m src.data_preparation.clean_raw_data --force
uv run -m src.data_preparation.reorganize
```

Default data roots:

- `data_raw/`: original Renishaw exports
- `data_cleaned/`: cleaned copy and curated clinical files
- `data/`: reorganized canonical dataset

## Run The Pipelines

EDA:

```bash
uv run scripts/run_eda.py --range 1300
uv run scripts/run_eda.py --range 2850
uv run scripts/run_eda.py --range both
uv run scripts/run_eda.py --range 1300 --max-subjects 10
```

Spectral comparison and QC plots:

```bash
uv run scripts/run_spectral_plots.py
uv run scripts/run_spectral_plots.py --max-subjects 20
uv run scripts/run_spectral_plots.py --outdir Output_plot
```

The figures written under `Output_plot/spectral_plots/` are post-preprocessing waveforms produced after the Raman preprocessing pipeline, not raw spectra.

ML:

```bash
uv run scripts/run_ml_pipeline.py
uv run scripts/run_ml_pipeline.py --config config/ml.yaml
uv run scripts/run_ml_pipeline.py --max-subjects 20
uv run scripts/run_ml_pipeline.py --force-reprocess
```


Analyze saved outputs:

```bash
uv run scripts/analyze_ml_results.py --input output_ML --output doc/Analysis_ML --no-pdf
```

`doc/Analysis_*` directories are generated artifacts and are not part of the maintained documentation set.

## Disease-Only vs. Disease-vs-Control EDA

Two separate config files select the analysis mode. Each writes to its own output root so results never overwrite each other.

**Disease vs. control** (default grouping, intra-disease correlation off):

```bash
uv run scripts/run_eda.py --config config/eda_disease_control.yaml --range 1300
uv run scripts/run_eda.py --config config/eda_disease_control.yaml --range 2850
uv run scripts/run_eda.py --config config/eda_disease_control.yaml --range both
```

Output root: `output_EDA_disease_vs_control/`

**Disease-only** (intra-disease Spearman clinical correlations enabled; requires `data/clinical_data_clean.csv`):

```bash
uv run scripts/run_eda.py --config config/eda_disease_only.yaml --range 1300
uv run scripts/run_eda.py --config config/eda_disease_only.yaml --range 2850
uv run scripts/run_eda.py --config config/eda_disease_only.yaml --range both
```

Output root: `output_EDA_disease_only/`

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
- `config/ml.yaml`
- `config/plots.yaml`

Current defaults to note:

- `config/eda.yaml` owns `run.ranges`: `["1300", "2850", "both"]`.
- EDA advanced phases are enabled in the current configs; disable `analysis.advanced.pixel_umap`, `subject_umap`, or `cluster_interpretation` for faster runs.
- `config/ml.yaml` is the publication-oriented ML config. It uses balanced accuracy for method selection, keeps a holdout lockbox, and leaves holdout scoring disabled unless explicitly enabled.
- EDA preprocessing order is: dead/anomalous screen → cosmic-ray detection → optional cosmic-ray suppression → spectral quality filter → edge trim → baseline correction → isolated spike repair → optional smoothing → vector normalization.
- Cosmic-ray detection always runs; suppression is controlled by `preprocessing.cosmic_ray.enabled`.
- Additive capabilities (all off by default): region weighting/exclusion, pixel-cluster interpretation, QC-cluster analysis, region-influence robustness, figure gating via `analysis.figures`, and ML latent-structure reporting.
- ML applies the fold-local per-wavenumber spectral scaler only to raw-channel learned transforms such as spectral PCA. Engineered Raman features are computed from physically preprocessed, unscaled subject-mean spectra, then scaled fold-locally at the feature level inside CV/holdout.
- `output.keep_preprocessing_artifacts` controls whether per-area preprocessing artifacts are kept in `output_EDA/processed/preprocessing_artifacts`.
- `output.keep_whole_spectra_cache` controls whether the paired-area whole-spectrum cache is saved in `output_EDA/processed/whole_spectra`.

## Maintained Docs

- `doc/eda_pipeline.md`: EDA workflow and outputs
- `doc/eda_methods.md`: EDA method reference (PCA, UMAP, HDBSCAN, QC, robustness, MFA, CCA)
- `doc/plots_command.md`: plot command reference and output map
- `plots.md`: spectral plot catalog and output layout
- `doc/ml_pipeline.md`: ML workflow
- `doc/whole_spectra_concatenation.md`: whole-spectrum fusion notes
- `doc/clinical_mismatch_handling.md`: clinical/data-label reconciliation
- `doc/bands_frequencies.md`: curated Raman band windows and caveats

## Scientific Caveats

- Inter-subject variance can reflect biology, SEC fraction, concentration, column batch or age, lipoprotein contamination, and sample volume.
- Sex imbalance can produce pooled effects that do not hold in sex-stratified results.
- QC flags indicate reliability concerns, not automatic exclusion rules.
- EDA is exploratory. Do not interpret unsupervised structure as diagnostic classification evidence; use the ML track for supervised questions.
