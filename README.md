# Raman Spectroscopy Analysis Pipeline

A professional, high-performance Python pipeline for extracellular-vesicle (EV) Raman spectroscopy analysis. This repository implements data ingestion, robust multi-stage preprocessing, statistical group comparisons, clinical correlations, a strict subject-level **Biological PCA** interpretation layer, and an exploratory **control-vs-disease ML classification** layer. Every output — including the ML layer — is exploratory scientific analysis: no diagnostic, biomarker, causal, or clinical-utility claim is made anywhere in this repository.

---

## 🚀 Key Features

*   **Robust Preprocessing**: Custom multi-stage pipeline including cosmic-ray removal, arPLS/AsLS/SNIP baseline correction, Savitzky-Golay smoothing, edge-trimming, and region-specific normalization.
*   **Hierarchical Data Model**: Natively handles nested structures: $\text{Subject} \rightarrow \text{Sample (Slide)} \rightarrow \text{Spot (Area)} \rightarrow \text{Pixels}$.
*   **Subject-Level Biological PCA**: Aggregates spectral data at the subject level to avoid pseudoreplication, projecting fingerprint (1300 cm⁻¹), CH-stretch (2850 cm⁻¹), and concatenated whole-spectrum spaces.
*   **Exploratory Control-vs-Disease ML Classification**: A separate, read-only downstream layer over the Biological PCA subject feature matrix. Nested cross-validation, label-permutation significance (unrestricted and sex-stratified), bootstrap confidence intervals, and measured (not assumed) confounder handling. Exploratory only — no diagnostic, biomarker, causal, or clinical-utility claim.
*   **Clinical Correlation**: Outputs are generated in `outputs/eda/intra_disease/`:
- `01_clinical_correlation_matrix/`: Broad spectral feature correlations with clinical endpoints.
- `02_longitudinal_moca/`: Associations specifically targeting ΔMoCA.
- `03_disease_stratification/`: Spectral profiles across disease severity sub-groups.
- `04_minimal_panel/`: ElasticNet feature reduction attempts.
- `05_confirmatory_panel/`: compatibility path for provenance-aware fixed post hoc panel analyses; current-cohort results remain exploratory.
- `06_spectral_interpretability/`: (Tier-2) Tools for spectral shape extraction, Feature Coupling (regimes), NMF decomposition, and entropy descriptors.
*   **Rich Visualizations**: Generates publication-ready scree plots, score pairgrids, loading summaries, group-mean waveforms, and individual QC grids.
*   **Data Integrity & Validation**: Fully manifest-driven data storage using Parquet for tabular metadata, NPZ for dense arrays, and JSON for schema/config snapshots.

---

## 📋 Pipeline Tracks

The analysis pipeline is organized into four operational tracks:

| Track | Entry Point / Script | Primary Outputs |
| :--- | :--- | :--- |
| **1. Data Store Management** | `uv run scripts/manage_data_store.py` | `data/parsed/` and `data/preprocessed/` stores |
| **2. Exploratory Data Analysis** | `uv run scripts/run_eda.py` | `Results_control_vs_disease/` (incl. Biological PCA and ML classification) or `Results_intra_disease/` |
| **3. Spectral Plotting & QC** | `uv run scripts/run_spectral_plots.py` | `Output_plot/qc/` and `Output_plot/spectral_plots/` |
| **4. GPU Readiness Probe** (optional) | `uv run scripts/check_gpu_readiness.py` | CuPy/CUDA equivalence check — does not change the pipeline |

---

## 🗂️ Data Model & Directory Layout

### Hierarchical Data Structure
Spectroscopic measurements are modeled using a strict physical hierarchy:
```text
Subject (e.g. S01)
 └── Sample/Slide (e.g. sample_01)
      └── Spot/Area (e.g. area_1)
           └── Pixels (e.g. 5x5 grid = 25 spectra)
```

### Materialized Data Store Layout
To separate crawler ingestion from downstream scientific analysis, data stores are version-controlled and cached under `data/`:

```text
data/
├── raw/                         # Raw Renishaw .txt exports
├── clinical_data.csv            # Original subject clinical metadata
├── parsed/                      # Ingested, validated raw metadata & spectra
│   ├── manifest.json            # Read/write integrity manifest
│   ├── file_index.parquet       # Indexed raw txt paths & parameters
│   ├── clinical_subjects.parquet# Parsed subject clinical fields
│   ├── area_metadata.parquet    # Slide and spot spatial coordinates
│   ├── spectra_1300.npz         # Dense raw spectra in Fingerprint region
│   ├── spectra_2850.npz         # Dense raw spectra in CH-Stretch region
│   └── cohort/                  # Materialized cohort subset tables
│       ├── control_vs_disease/  # Subject listings for case-control
│       └── disease_only/        # Subject listings for intra-disease
└── preprocessed/                # Post-preprocessing metadata & spectra
    └── ...                      # Mirror structure of parsed/ (preprocessed)
```

---

## 🛠️ Setup & Installation

### Prerequisites
*   Python **>= 3.12**
*   Project dependencies are declared in `pyproject.toml`.
*   [uv](https://github.com/astral-sh/uv) is highly recommended for fast environment setup and execution.

### Installation
Clone the repository and install the pipeline in editable mode:

```bash
# Clone the repository
git clone https://github.com/Narden91/spectral-analysis-raman.git
cd spectral-analysis-raman

# Create virtual environment and install dependencies
uv venv
uv pip install -e .
```

---

## ⚙️ Running the Pipeline

### 1. Building the Data Stores
Before running analyses, crawl the raw text data to build and validate the structured binary caches:

```bash
# Build the parsed data cache
uv run scripts/manage_data_store.py build --source parsed

# Apply the preprocessing pipeline and cache the preprocessed cache
uv run scripts/manage_data_store.py build --source preprocessed

# Force a rebuild even if a fresh store already exists
uv run scripts/manage_data_store.py build --source preprocessed --force

# Validate cache integrity and check config matches
uv run scripts/manage_data_store.py validate --source preprocessed

# Display information on the cached data stores
uv run scripts/manage_data_store.py info --source preprocessed

# Remove a materialized store (parsed or preprocessed)
uv run scripts/manage_data_store.py clean --source preprocessed

# Preview actions without mutating any store (works with any subcommand)
uv run scripts/manage_data_store.py --dry-run build --source preprocessed

# Point at a non-default config file
uv run scripts/manage_data_store.py --config config/eda_disease_control.yaml info --source preprocessed
```

`build`, `validate`, `info`, and `clean` all require `--source {parsed,preprocessed}`. `--dry-run` and `--config` are top-level flags placed before the subcommand.

### 2. Running the Exploratory Data Analysis (EDA)
The EDA script executes statistical checks, principal component analyses, and (in Case vs. Control mode) the ML classification layer. EDA mode selection is explicit and authoritative through either `analysis.mode` in config or `--analysis-mode` on the CLI. `scripts/run_eda.py` is the **only** entry point for Biological PCA and the control-vs-disease ML layer.

#### CLI Flags
| Flag | Description |
| :--- | :--- |
| `--range {1300,2850,both}` | Spectral range to analyze. Omit to use `run.ranges` from config. `both` also runs the fused whole-spectrum analysis, Biological PCA, and (in Case vs. Control mode) the ML layer. |
| `--config PATH` | Config YAML to use (overrides `config/eda.yaml`). Supports `_extends:` inheritance; relative paths resolve from the project root. |
| `--analysis-mode {disease_vs_control,disease_only}` | Explicit EDA analysis family. Overrides `analysis.mode` from config. |
| `--max-subjects N` | Limit subjects per group — quick smoke testing only. Below ~10/class the ML layer's minimum-subjects guard will fail it by design. |
| `--skip-preprocess` | Use raw (unpreprocessed) spectra instead of the preprocessed store. |
| `--skip-biological-pca` | Skip the subject-level Biological PCA layer (and, transitively, the ML layer, which consumes its output). |
| `--source {raw_text,parsed,preprocessed}` | Override `data.source`. Defaults to `preprocessed`; `raw_text` crawls `data/raw` directly (avoid in normal analysis, per data-source rules). |
| `--verbose` | Show detailed INFO logs in the terminal. Full detail is always written to `run.log` regardless of this flag. |

#### Mode A: Case vs. Control (Default case)
Compares healthy controls against disease subjects. Runs Biological PCA, then the control-vs-disease ML classification layer, when `--range both` is used.
```bash
# Analyze fingerprint region
uv run scripts/run_eda.py --config config/eda_disease_control.yaml --range 1300

# Analyze CH-stretch region
uv run scripts/run_eda.py --config config/eda_disease_control.yaml --range 2850

# Fused analysis (combines fingerprint & CH-stretch + Subject Biological PCA + ML classification)
uv run scripts/run_eda.py --config config/eda_disease_control.yaml --range both
```
*   **Output Directory**: `Results_control_vs_disease/` (ML classification outputs land in `Results_control_vs_disease/07_ml_classification/`)

#### Mode B: Disease-Only Cohort (Clinical Correlations)
Restricts analysis to disease subjects to correlate spectral bands with clinical parameters (requires `data/clinical_data.csv`).
```bash
uv run scripts/run_eda.py --analysis-mode disease_only --range both

# Equivalent wrapper-based invocation
uv run scripts/run_eda.py --config config/eda_disease_only.yaml --range both
```
*   **Output Directory**: `Results_intra_disease/`

Notes:
*   `--analysis-mode` overrides `analysis.mode` from config.
*   `config/eda_disease_control.yaml` and `config/eda_disease_only.yaml` are thin wrappers over `config/eda.yaml`.
*   `config/eda_disease_fast.yaml` is a **development-only** overlay on `eda_disease_only.yaml` that lowers permutation/bootstrap counts and CV cost so disease-only cross-region phases finish quickly. Do **not** use it for final scientific outputs.
*   `config/eda_disease_control_fast.yaml` is the equivalent **development-only** overlay for the Case vs. Control ML layer — it lowers nested-CV repeats, permutation counts, and bootstrap counts. The ML layer needs the full cohort; `--max-subjects` below ~10/class fails its minimum-subjects guard. Do **not** use it for manuscript numbers.
*   Output root selection is derived from the resolved EDA mode, not from `data.subject_group` or wrapper-specific output paths.

#### Fast iteration

For day-to-day development after Biological PCA, prefer the fast profiles:

```bash
# Disease-only cross-region phases (reduced permutations/CV)
uv run scripts/run_eda.py --config config/eda_disease_fast.yaml --range both --max-subjects 8

# Case vs. Control ML layer (reduced permutations/CV/bootstrap; full cohort required)
uv run scripts/run_eda.py --config config/eda_disease_control_fast.yaml --range both

# Smoke test across both ranges with a tiny subject cap (full resampling settings)
uv run scripts/run_eda.py --range both --max-subjects 8
```

Production runs should use `config/eda_disease_only.yaml` or `config/eda_disease_control.yaml` (full resampling counts from `config/eda.yaml`) — these produce the manuscript numbers, the fast overlays do not.

After each EDA run, inspect per-phase wall time at:

```text
Results_intra_disease/00_run_metadata/disease_only_cross_region_phase_timing.csv
```

Resampling-heavy steps parallelize across CPU cores via `n_jobs` in config (default `-1` in `config/eda.yaml`). Set `n_jobs: 1` to force the serial path. GPU acceleration is **not** part of the analysis pipeline; optional CUDA readiness can be checked with:

```bash
uv run scripts/check_gpu_readiness.py

# Custom seed, numerical tolerance, and result output path
uv run scripts/check_gpu_readiness.py --seed 42 --tolerance 1e-6 --output gpu_check.json
```

This script does not change the analysis pipeline — it only verifies CuPy/CUDA availability and CPU-vs-GPU numerical equivalence for future GPU-backed code paths.

### 3. Spectral Plotting & QC Visualization
Generate post-preprocessing waveforms, individual scan layouts for diagnostic audit, and optionally export the underlying data as CSVs.

| Flag | Description |
| :--- | :--- |
| `--config PATH` | Config YAML to use (overrides `config/plots.yaml`). |
| `--max-subjects N` | Limit subjects per cohort. |
| `--outdir DIR` | Override the base output directory. |
| `--source {raw_text,parsed,preprocessed}` | Override `data.source`. |
| `--plots` / `--no-plots` | Generate plot images (default: on). |
| `--export-data` / `--no-export-data` | Export CSV data files (default: on). |

```bash
# Generate group spectral plots and export CSV data (default behavior)
uv run scripts/run_spectral_plots.py

# Restrict to a smaller test cohort size
uv run scripts/run_spectral_plots.py --max-subjects 20 --outdir Output_plot

# Generate plots only (skip CSV export)
uv run scripts/run_spectral_plots.py --no-export-data

# Export CSV data only (skip plot generation)
uv run scripts/run_spectral_plots.py --no-plots

# Use a non-default config and data source
uv run scripts/run_spectral_plots.py --config config/plots.yaml --source preprocessed
```

The exported CSV data will be saved in a directory parallel to the plot output directory (e.g., `output_plot_data/` when the plot directory is `Output_plot/`).

---

## 🧪 Subject-Level Biological PCA Rules

To prevent **pseudoreplication** (treating multiple pixels from the same subject as independent statistical observations), the pipeline enforces a strict subject-level aggregation step prior to primary PCA projection.

1.  **Model Layouts**:
    *   `PCA_1330`: Fingerprint region (1300 cm⁻¹ range).
    *   `PCA_2850`: CH-stretch region (2850 cm⁻¹ range).
    *   `PCA_whole`: MFA-scaled concatenation of fingerprint and CH-stretch medians; unscaled PCA is sensitivity-only.
2.  **Naming Convention**: Loading columns and indices are prefixed with their region (e.g. `1300__1240.0` or `2850__2840.0`) to avoid coordinate collisions.
3.  **Components**: Exactly 3 components (PC1, PC2, PC3) are computed and exported.
4.  **Figures Generated**:
    *   `scores_pc1_pc2.png`, `scores_pc1_pc3.png`, `scores_pc2_pc3.png`
    *   `scores_pc_pairgrid.png`
    *   `scree_plot.png`
    *   `loadings_pc1_pc2_pc3.png`, `loading_regions_pc1_pc2_pc3.png`

---

## 🤖 Control-vs-Disease ML Classification (Case vs. Control Mode Only)

`disease_vs_control` mode runs an exploratory, subject-level supervised classification layer after Biological PCA (`src/analysis/control_vs_disease_ml.py`, figures/report in `src/analysis/control_vs_disease_ml_report.py`). It is a **separate, read-only downstream consumer** of the Biological PCA subject feature matrix — no ML claim attaches to Biological PCA outputs, and this layer never mutates Biological PCA state.

*   **Models**: PLS-DA, L1-logistic (`class_weight='balanced'`), Random Forest (`class_weight='balanced_subsample'`) — each on two feature representations (`whole` fused spectrum, `band_panel` 21-band biochemical panel).
*   **Positive class**: Disease (`is_control == False` → 1), documented in every output artifact.
*   **Rigor**: Nested cross-validation (`RepeatedStratifiedKFold`, in-fold scaling/tuning/thresholding — never on the outer test fold), full-pipeline label-permutation significance (unrestricted **and** sex-stratified nulls — the sex-stratified p-value is the primary inferential result), out-of-bag bootstrap 95% confidence intervals, BH-FDR correction across the model family.
*   **Confounding is measured, not assumed**: this cohort's sex distribution is significantly associated with class (controls are predominantly female, disease more balanced). `confounder_analysis.csv` reports a female-only sensitivity refit and a sex-only baseline classifier per model; a spectral finding must survive the sex-stratified permutation null and beat the sex-only baseline. Age is **not modeled anywhere in this layer** — no age data exists for controls in this cohort and no proxy exists in the repository; this is a permanent limitation.
*   **Claims discipline**: exploratory only. No diagnostic, biomarker, causal, or clinical-utility claim. Single-cohort, single-site, no external validation. `ml_report.md` states this explicitly at the top of every run.
*   **Output**: `Results_control_vs_disease/07_ml_classification/`, including per-model metric tables, ROC/PR/confusion-matrix and permutation-null figures, `confounder_analysis.csv`, and `ml_report.md`.
*   **Config**: `analysis.control_vs_disease_ml` in `config/eda.yaml`; fast dev overlay `config/eda_disease_control_fast.yaml` (iteration only, not manuscript numbers — see [Fast iteration](#fast-iteration)).

```bash
# Production run (full resampling — manuscript numbers)
uv run scripts/run_eda.py --config config/eda_disease_control.yaml --range both

# Fast dev iteration (reduced permutations/CV/bootstrap; full cohort still required)
uv run scripts/run_eda.py --config config/eda_disease_control_fast.yaml --range both
```

---

## ✅ Testing

The full suite covers preprocessing, statistics, Biological PCA, the disease-only cross-region pipeline, and the control-vs-disease ML layer.

```bash
# Run the full test suite
uv run pytest -v

# Run a specific test file
uv run pytest tests/test_biological_pca.py -v

# Run Biological PCA tests only
uv run pytest tests/test_biological_pca.py

# Run control-vs-disease ML tests only (core pipeline + report generation)
uv run pytest tests/test_control_vs_disease_ml.py tests/test_control_vs_disease_ml_report.py

# Run with a keyword filter (matches test/file names)
uv run pytest -k "clinical or metadata"

# Stop at the first failure, show local variables on error
uv run pytest -x -l
```

---

## 🔬 Scientific Caveats & Guidance

*   **No Diagnostic Claims**: The pipeline is strictly intended for exploratory data analysis (EDA), including the supervised ML classification layer. No output — unsupervised or supervised — may be interpreted as diagnostic, predictive, causal, or biomarker classification evidence.
*   **Variance Confounders**: Keep in mind that variance in EV spectral signals may be influenced by technical factors (SEC fraction, sample concentration, column age, and sample volume) in addition to genuine biology.
*   **Stratification Guardrails**: Severe imbalances in sex distribution between cohorts can produce misleading pooled effects. Stratify findings where possible; the ML layer's sex-stratified permutation null and female-only sensitivity refit exist specifically to guard against this.
*   **Single-Cohort Scope**: All results — Biological PCA and ML alike — are single-cohort, single-site, with no external validation. Findings do not generalize beyond this dataset without independent replication.
