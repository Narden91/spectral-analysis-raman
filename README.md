# Raman Spectroscopy Analysis Pipeline

A professional, high-performance Python pipeline for extracellular-vesicle (EV) Raman spectroscopy analysis. This repository implements data ingestion, robust multi-stage preprocessing, statistical group comparisons, clinical correlations, and a strict subject-level **Biological PCA** interpretation layer.

---

## 🚀 Key Features

*   **Robust Preprocessing**: Custom multi-stage pipeline including cosmic-ray removal, arPLS/AsLS/SNIP baseline correction, Savitzky-Golay smoothing, edge-trimming, and region-specific normalization.
*   **Hierarchical Data Model**: Natively handles nested structures: $\text{Subject} \rightarrow \text{Sample (Slide)} \rightarrow \text{Spot (Area)} \rightarrow \text{Pixels}$.
*   **Subject-Level Biological PCA**: Aggregates spectral data at the subject level to avoid pseudoreplication, projecting fingerprint (1300 cm⁻¹), CH-stretch (2850 cm⁻¹), and concatenated whole-spectrum spaces.
*   **Clinical Correlation**: Outputs are generated in `outputs/eda/intra_disease/`:
- `01_clinical_correlation_matrix/`: Broad spectral feature correlations with clinical endpoints.
- `02_longitudinal_moca/`: Associations specifically targeting ΔMoCA.
- `03_disease_stratification/`: Spectral profiles across disease severity sub-groups.
- `04_minimal_panel/`: ElasticNet feature reduction attempts.
- `05_confirmatory_panel/`: (Tier-1) Robust statistics and permutation testing on pre-registered features.
- `06_spectral_interpretability/`: (Tier-2) Tools for spectral shape extraction, Feature Coupling (regimes), NMF decomposition, and entropy descriptors.
*   **Rich Visualizations**: Generates publication-ready scree plots, score pairgrids, loading summaries, group-mean waveforms, and individual QC grids.
*   **Data Integrity & Validation**: Fully manifest-driven data storage using Parquet for tabular metadata, NPZ for dense arrays, and JSON for schema/config snapshots.

---

## 📋 Pipeline Tracks

The analysis pipeline is organized into three primary operational tracks:

| Track | Entry Point / Script | Primary Outputs |
| :--- | :--- | :--- |
| **1. Data Store Management** | `uv run scripts/manage_data_store.py` | `data/parsed/` and `data/preprocessed/` stores |
| **2. Exploratory Data Analysis** | `uv run scripts/run_eda.py` | `Results_control_vs_disease/` or `Results_intra_disease/` |
| **3. Spectral Plotting & QC** | `uv run scripts/run_spectral_plots.py` | `Output_plot/qc/` and `Output_plot/spectral_plots/` |

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

# Validate cache integrity and check config matches
uv run scripts/manage_data_store.py validate --source preprocessed

# Display information on the cached data stores
uv run scripts/manage_data_store.py info --source preprocessed
```

### 2. Running the Exploratory Data Analysis (EDA)
The EDA script executes statistical checks and principal component analyses. EDA mode selection is explicit and authoritative through either `analysis.mode` in config or `--analysis-mode` on the CLI.

#### Mode A: Case vs. Control (Default case)
Compares healthy controls against disease subjects.
```bash
# Analyze fingerprint region
uv run scripts/run_eda.py --config config/eda_disease_control.yaml --range 1300

# Analyze CH-stretch region
uv run scripts/run_eda.py --config config/eda_disease_control.yaml --range 2850

# Fused analysis (combines fingerprint & CH-stretch + Subject Biological PCA)
uv run scripts/run_eda.py --config config/eda_disease_control.yaml --range both
```
*   **Output Directory**: `Results_control_vs_disease/`

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
*   `config/eda_disease_fast.yaml` is a **development-only** overlay on `eda_disease_only.yaml` that lowers permutation/bootstrap counts and CV cost so cross-region phases finish quickly. Do **not** use it for final scientific outputs.
*   Output root selection is derived from the resolved EDA mode, not from `data.subject_group` or wrapper-specific output paths.

#### Fast iteration (disease-only cross-region phases)

For day-to-day development after Biological PCA, prefer the fast profile:

```bash
uv run scripts/run_eda.py --config config/eda_disease_fast.yaml --range both --max-subjects 8
```

Production disease-only runs should use `config/eda_disease_only.yaml` (full resampling counts from `config/eda.yaml`).

After each EDA run, inspect per-phase wall time at:

```text
Results_intra_disease/00_run_metadata/disease_only_cross_region_phase_timing.csv
```

Resampling-heavy steps parallelize across CPU cores via `n_jobs` in config (default `-1` in `config/eda.yaml`). Set `n_jobs: 1` to force the serial path. GPU acceleration is **not** part of the analysis pipeline; optional CUDA readiness can be checked with:

```bash
uv run scripts/check_gpu_readiness.py
```

### 3. Spectral Plotting & QC Visualization
Generate post-preprocessing waveforms and individual scan layouts for diagnostic audit:

```bash
# Generate group spectral plots
uv run scripts/run_spectral_plots.py

# Restrict plots to a smaller test cohort size
uv run scripts/run_spectral_plots.py --max-subjects 20 --outdir Output_plot
```

---

## 🧪 Subject-Level Biological PCA Rules

To prevent **pseudoreplication** (treating multiple pixels from the same subject as independent statistical observations), the pipeline enforces a strict subject-level aggregation step prior to primary PCA projection.

1.  **Model Layouts**:
    *   `PCA_1330`: Fingerprint region (1300 cm⁻¹ range).
    *   `PCA_2850`: CH-stretch region (2850 cm⁻¹ range).
    *   `PCA_whole`: Unweighted concatenation of fingerprint and CH-stretch medians.
2.  **Naming Convention**: Loading columns and indices are prefixed with their region (e.g. `1300__1240.0` or `2850__2840.0`) to avoid coordinate collisions.
3.  **Components**: Exactly 3 components (PC1, PC2, PC3) are computed and exported.
4.  **Figures Generated**:
    *   `scores_pc1_pc2.png`, `scores_pc1_pc3.png`, `scores_pc2_pc3.png`
    *   `scores_pc_pairgrid.png`
    *   `scree_plot.png`
    *   `loadings_pc1_pc2_pc3.png`, `loading_regions_pc1_pc2_pc3.png`

---

## 🔬 Scientific Caveats & Guidance

*   **No Diagnostic Claims**: The pipeline is strictly intended for unsupervised exploratory data analysis (EDA). The outputs must not be interpreted as diagnostic, predictive, or biomarker classification evidence.
*   **Variance Confounders**: Keep in mind that variance in EV spectral signals may be influenced by technical factors (SEC fraction, sample concentration, column age, and sample volume) in addition to genuine biology.
*   **Stratification Guardrails**: Severe imbalances in sex distribution between cohorts can produce misleading pooled effects. Stratify findings where possible.
