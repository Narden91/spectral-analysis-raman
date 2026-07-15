# Raman Spectroscopy Analysis Pipeline

Reproducible Python workflows for extracellular-vesicle Raman spectra. The
pipeline materializes immutable raw acquisitions, produces a QC-auditable
analysis basis, and runs exploratory EDA, subject-level Biological PCA,
control-vs-disease modelling, and publication-style spectral plots.

All scientific outputs are exploratory. They are not diagnostic, predictive,
causal, biomarker, or clinical-utility evidence.

## What the pipeline does

1. Ingests Renishaw text exports and clinical metadata.
2. Preserves each acquisition's native calibrated wavenumber axis.
3. Applies deterministic pixel QC and preprocessing.
4. Interpolates only after edge trimming, onto pre-registered grids with no
   extrapolation.
5. Aggregates to the subject level for primary Biological PCA and downstream
   classification.
6. Writes manifests, configuration snapshots, QC tables, and figures for
   reproducibility.

## Requirements and installation

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/) (recommended)
- Raw Raman text exports and `data/clinical_data.csv`

```bash
git clone <repository-url>
cd spectral-analysis
uv sync
```

## Raw-data contract

Place source files under `data/raw/` in the repository's existing hierarchy.
The filename parser extracts subject, slide, spectral range, acquisition mode,
and measurement number. Each file must contain one strictly increasing
wavenumber axis and its pixel spectra.

```text
data/
├── raw/
│   └── <cohort>/<sex>/<subject>/sample_<id>/<measurement>.txt
└── clinical_data.csv
```

`clinical_data.csv` must include a subject identifier compatible with the raw
file index. Keep raw files read-only once a campaign begins: their signatures
are recorded in the parsed-store manifest.

## End-to-end production workflow

Run these commands from the repository root. They are the supported path from
raw text files to results.

```bash
# 1. Create / update the environment.
uv sync

# 2. Build immutable parsed data from raw text.
uv run scripts/manage_data_store.py build --source parsed --force

# 3. Build the QC-filtered, baseline-corrected, normalized analysis store.
uv run scripts/manage_data_store.py build --source preprocessed --force

# 4. Verify every generated artifact and every raw input signature.
uv run scripts/manage_data_store.py validate --source parsed --strict
uv run scripts/manage_data_store.py validate --source preprocessed --strict

# 5. Run the selected scientific workflow.
uv run scripts/run_eda.py --config config/eda_disease_control.yaml --range both

# 6. Generate aggregate and QC spectral plots.
uv run scripts/run_spectral_plots.py --config config/plots.yaml --source preprocessed
```

The materializer writes to a temporary sibling directory and replaces a target
store only after its writer completes. Do not use `clean` during a campaign.
Use `--force` only when raw data, preprocessing settings, or materialization
logic have changed and a deliberate rebuild is required.

### Fast development only

These commands reduce resampling or subject count and must not be used for
final scientific outputs.

```bash
uv run scripts/run_eda.py --config config/eda_disease_fast.yaml --range both --max-subjects 8
uv run scripts/run_eda.py --config config/eda_disease_control_fast.yaml --range both
```

## Data stores and preprocessing

`data/parsed/` contains raw spectra, coordinates, metadata, native axes, and
input signatures. It is not a globally aligned analysis matrix.

`data/preprocessed/` contains the only canonical aligned matrices used by
analysis:

```text
data/preprocessed/
├── measurements.parquet
├── subjects.parquet
├── areas.parquet
├── qc.parquet
├── normalized_1300.npy
├── normalized_2850.npy
├── pre_normalization_1300.npy
├── pre_normalization_2850.npy
├── wavenumbers_1300.npy
├── wavenumbers_2850.npy
└── manifest.json
```

The production sequence is:

```text
native raw spectra
→ dead/saturated-pixel QC
→ narrow positive cosmic-spike detection and conservative suppression
→ edge trim
→ linear interpolation to a frozen grid (no extrapolation)
→ baseline correction
→ quality gate
→ optional smoothing
→ normalization
```

The registered analysis grids are 733–1788 cm⁻¹ (fingerprint) and
2450–3197 cm⁻¹ (CH stretch), at 1 cm⁻¹ spacing. They were selected from native
axis coverage after the configured edge trim and must not be guessed or changed
for a campaign without a documented calibration decision.

Fingerprint spectra use SNV normalization; CH-stretch spectra use area
normalization. Signed baseline-corrected residuals and negative SNV values are
expected. Use `pre_normalization_*.npy` only for the dedicated relative
band-integral workflow; PCA and ML consume normalized spectra.

## Analysis commands

```bash
# Fingerprint EDA only.
uv run scripts/run_eda.py --config config/eda_disease_control.yaml --range 1300

# CH-stretch EDA only.
uv run scripts/run_eda.py --config config/eda_disease_control.yaml --range 2850

# Fused case-control workflow: EDA, Biological PCA, and exploratory ML.
uv run scripts/run_eda.py --config config/eda_disease_control.yaml --range both

# Disease-only exploratory clinical workflow.
uv run scripts/run_eda.py --config config/eda_disease_only.yaml --range both

# Generate only plot images or only plot data exports.
uv run scripts/run_spectral_plots.py --no-export-data
uv run scripts/run_spectral_plots.py --no-plots
```

`--analysis-mode {disease_vs_control,disease_only}` overrides the mode in the
configuration. `--source preprocessed` is the normal analysis source. Parsed
data is for native-axis audit and cannot be treated as a globally aligned
matrix.

## Outputs

| Workflow | Output root |
|---|---|
| Case-control EDA, Biological PCA, exploratory ML | `Results_control_vs_disease/` |
| Disease-only clinical/cross-region analysis | `Results_intra_disease/` |
| Spectral plotting and QC | `Output_plot/` |

Each EDA run records provenance in `00_run_metadata/`, including the resolved
configuration and data-manifest snapshot. ML output is subject-level only;
nested CV tuning remains inside outer training folds, permutation tests include
a sex-stratified null, and importance summaries are derived across outer folds.

Aggregate spectral plots use the curated 21 Raman windows. Their coloured bands
indicate broad biochemical context (nucleic acids, proteins, lipids,
carotenoids, or mixed), not molecule-specific abundance or diagnostic identity.

## Assumptions and interpretation limits

- One subject can contribute several samples, spots, and pixels; pixels are not
  independent biological replicates.
- Technical variation (sample preparation, concentration, SEC fraction,
  column state, and acquisition conditions) can contribute to spectral
  differences.
- Clinical metadata are observational and may be incomplete; no causal
  interpretation is supported.
- The control-vs-disease ML layer is single-cohort exploratory work with no
  external validation.
- Rebuild and strict-validate stores before every final analysis campaign.

## Verification

```bash
# Focused materialization, preprocessing, and plot checks.
uv run pytest tests/test_data_store.py tests/test_preprocessing_pipeline.py tests/test_spectral_summary_plots.py -q

# ML checks.
uv run pytest tests/test_control_vs_disease_ml.py tests/test_control_vs_disease_ml_report.py -q

# Full test suite before release.
uv run pytest -q

# Static import/bytecode check.
uv run python -m compileall -q src scripts
```

## Project map

| Path | Purpose |
|---|---|
| `config/` | YAML configuration and fast development overlays |
| `scripts/manage_data_store.py` | Store build, validation, information, and deliberate cleanup |
| `scripts/run_eda.py` | Main EDA, Biological PCA, and ML entry point |
| `scripts/run_spectral_plots.py` | Spectral plots, QC plots, and CSV export |
| `src/data_preparation/` | Materialization, manifests, validation, and loaders |
| `src/preprocessing/` | QC and preprocessing implementation |
| `src/analysis/` | EDA, PCA, statistics, and exploratory modelling |
| `src/visualization/` | Figures and plot styling |

For contributor rules and implementation details, read [CLAUDE.md](CLAUDE.md).
