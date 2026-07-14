# Repository Guidance: Raman Spectral Analysis

This repository contains a Python Raman spectroscopy analysis pipeline for extracellular-vesicle data. The maintained workflow covers preprocessing/QC, exploratory data analysis, visualization, a strict subject-level Biological PCA layer, and an exploratory control-vs-disease ML classification layer (§7). Treat all outputs — including the ML layer — as exploratory scientific analysis: no diagnostic, biomarker, causal, or clinical-utility claim is made anywhere in this repository.

## 1. Quick Start & Entry Points

- `uv run scripts/run_eda.py`: Main EDA entry point and the **only** entry point for Biological PCA and the control-vs-disease ML layer (§7).
- `uv run scripts/run_spectral_plots.py`: Spectral comparison, QC plotting, and CSV data export.

### Common Commands
- Smoke test (full resampling): `uv run scripts/run_eda.py --range both --max-subjects 8`
- Fast disease-only iteration (reduced permutations/CV; not for final outputs): `uv run scripts/run_eda.py --config config/eda_disease_fast.yaml --range both --max-subjects 8`
- Fast disease-vs-control ML iteration (reduced permutations/CV/bootstrap; not for final outputs — the ML layer needs the full cohort, `--max-subjects` below ~10/class will fail its minimum-subjects guard): `uv run scripts/run_eda.py --config config/eda_disease_control_fast.yaml --range both`
- Run all tests: `uv run pytest -v`
- Run biological PCA tests: `uv run pytest tests/test_biological_pca.py`
- Run control-vs-disease ML tests: `uv run pytest tests/test_control_vs_disease_ml.py tests/test_control_vs_disease_ml_report.py`
- Optional GPU readiness probe (does not change pipeline): `uv run scripts/check_gpu_readiness.py`

## 2. Core Architecture

The codebase follows a modular structure:
- `src/io/`: file indexing and Renishaw text loading.
- `src/metadata/`: filename metadata parsing.
- `src/data_preparation/`: parsed and preprocessed store materialization, validation, loading, and cohorts.
- `src/preprocessing/`: Raman preprocessing and QC.
- `src/clinical/`: clinical endpoint definitions and mappings.
- `src/fusion/`: cross-range and area-level fusion helpers.
- `src/analysis/`: EDA, PCA, whole-spectra analysis, Biological PCA, and the control-vs-disease ML classification layer (§7).
- `src/visualization/`: plotting and report figures.
- `src/reporting/`: candidate region provenance and manuscript reports.

Configuration is YAML-based. Track configs use `_extends`; `config/eda.yaml` owns EDA, Biological PCA, and control-vs-disease ML settings.

## 2b. Preprocessing & Normalization (`src/preprocessing/pipeline.py`)

Single implementation, orchestrated by `run_preprocessing`. Order: QC metrics → dead-pixel screen → cosmic-ray suppression → edge trim → **arPLS baseline correction** (`corrected = row - baseline`) → SNR/area quality gate → isolated-spike repair → optional Savitzky-Golay smoothing → optional interpolation → **normalization (last)**.

- **Negative values after preprocessing are expected, not a bug.** Two intentional sources: (1) arPLS baseline subtraction with `quality_filter.baseline_clip.enabled: false` (`config/settings.yaml`) — signed residuals are deliberately preserved, standard Raman practice; (2) SNV normalization on the fingerprint region mean-centers the spectrum, making roughly half the channels negative by construction. `PCA_whole` additionally mean-centers each block before joint PCA (`src/analysis/joint_dr.py`).
- **Normalization differs by region, intentionally**: fingerprint (1300) = **SNV** (signed, standardized), CH-stretch (2850) = **area** (unsigned scale). Config: `preprocessing.ranges.{fingerprint,ch_stretch}.normalization.method` in `config/settings.yaml`. `PCA_whole` reconciles the scale mismatch via MFA-style block centering + division by `sqrt(first eigenvalue)` per block (`src/analysis/joint_dr.py`) before joint PCA — see §6.
- **Band integration must not use signed SNV data.** The canonical preprocessed store persists `pre_normalization_{1300,2850}.npy` beside `normalized_{1300,2850}.npy`, both from the same QC/baseline pass. It is an arbitrary-unit, baseline-corrected basis for relative band ratios—not an absolute intensity claim. Phase 8 alone reads this companion basis; PCA and ML use normalized spectra.
- Data-store schema is versioned (`src/data_preparation/manifest.py::SCHEMA_VERSION`). A schema mismatch fails loudly; rebuild explicitly with `uv run scripts/manage_data_store.py build --source preprocessed --force`.

## 3. Data Source Rules

The repository supports two analysis source levels: `parsed` and `preprocessed` (default). Raw text is ingestion-only.

- **Load safely**: Normal analysis must load through `src/data_preparation/store.py` and `src/data_preparation/loaders.py`.
- **No raw crawling**: `data/raw` is used only by the explicit parsed-store ingestion command.
- **No silent fallbacks**: Do not silently fall back across sources or silently rebuild stale generated stores during analysis.
- **Formats**: Use Parquet for canonical tables, memory-mappable NPY for dense arrays, and JSON for manifests. No duplicate NPZ/CSV mirrors are materialized.
- **Artifacts**: Result directories must include `run_manifest.json`, `data_manifest_snapshot.json`, and `config_snapshot.yaml`.

## 4. Output Routing

Outputs are routed through the resolved EDA analysis mode (`--analysis-mode` or `analysis.mode` in config).
- **Disease vs Control**: `Results_control_vs_disease/`
- **Disease Only**: `Results_intra_disease/`

## 5. Disease-Only Pipeline

The `disease_only` pipeline mode executes a comprehensive suite of cross-region summarizations and modelling. Descriptive/association phases output to `Results_intra_disease/04_cross_region_summary/` as one flat, ascending-numbered sequence (`CROSS_REGION_PHASE_DIRS` in `src/__init__.py`). Predictive/classifier phases output to a separate top-level `Results_intra_disease/classification/` folder (`CLASSIFICATION_PHASE_DIRS`, resolved via `get_classification_phase_dir(phase_key)`) — kept apart so a reader never mistakes an exploratory classifier's output for a descriptive finding. All phase code resolves its output directory through these resolver functions — never construct these paths by string concatenation.

**`04_cross_region_summary/` (descriptive/association, flat 01-10):**
- **Phase 1 (Subject Features)** / **Phase 2 (Biological PCA Summary)**: subject-level feature exports and Biological PCA summary — shared inputs to every downstream phase. Phase 1 also writes `subject_feature_matrix_whole_bc.parquet`, the baseline-corrected/un-normalized companion consumed only by Phase 8 (§2b).
- **Phases 3-5**: Base clinical correlation matrices, longitudinal delta MoCA testing, and disease-internal subject stratification. Each also emits a `*_candidate_regions.csv` byproduct consumed only by Phase 6 — not a standalone result.
- **Phase 6 (Candidate Regions)**: Unifies hits from phases 1-5 into a consolidated, cluster-corrected table (`candidate_regions_cluster_corrected.csv`). This is the single canonical candidate-region table.
- **Phase 7 (Covariate Models)**: Fits covariate-adjusted models (age, sex) on Phase 6 candidate regions and logs persistence — a confound-check, not a predictive model, so it stays in this sequence rather than `classification/`.
- **Phase 8 (Biochemical Ratios)**: Computes targeted relative band ratios using the baseline-corrected, pre-normalization `_bc` feature view (§2b), not the SNV/area feature matrix. These are arbitrary-unit signal measures, not absolute molecular intensities.
- **Phase 9 (Integrated Candidate Regions)**: Merges univariate hits, covariate persistence, biological PCA loading overlap, PLS VIP coverage, and biochemical interpretation into a single priority-ranked table (`High`/`Medium`/`Low`) written to `09_integrated_candidates/integrated_candidate_regions.csv`.
- **Phase 10 (Publication Report)**: Generates a conservative markdown report and summary CSV that classifies findings as robust, exploratory, unstable, or confounded. Outputs `10_publication_report/candidate_region_report.md`, `candidate_region_summary.csv`, and `candidate_region_figures/`.

**`classification/` (predictive/classifier phases, unnumbered — each a self-contained analysis, not a sequence):**
- **PLS1 Regression** (`pls_regression/`): Supervised continuous modelling of endpoints with permutation testing and VIP selection.
- **PLS-L1DA** (`splsda_exploratory_classifier/`): Supervised binary classifier, gated strictly by PLS1 Regression significance.
- **Minimal Candidate Panel** (`minimal_panel/`): Selects 5–10 spectral peaks/ratios per endpoint using nested cross-validated ElasticNet / L1-penalized logistic regression with full-pipeline permutation testing and bootstrap selection-stability summaries. Candidate-derived modeling is skipped when no cluster-supported region passes its gate.
- **Disease Subgroups** (`disease_subgroups/`): Exploratory agglomerative Ward clustering evaluated via Gap statistic and bootstrap Adjusted Rand Index.

Phases that cannot run at the current sample size (below `min_subjects`) report their status in `00_run_metadata/phase_status.csv` instead of writing header-only CSVs — see §10.

## 6. Biological PCA Rules

Biological PCA is subject-level only and is the authoritative PCA layer per region. Area-level EDA PCA remains but is demoted to exploratory/QC use only.

- **Models**: `PCA_1330`, `PCA_2850`, `PCA_whole`.
- **Multiblock scaling**: canonical `PCA_whole` centers each block and divides it by the square root of its first eigenvalue before joint PCA. Unscaled concatenation is sensitivity-only.
- **Identity**: Whole-spectrum columns and loadings must preserve block identity (e.g., `1300__1240.0`).
- **Interpretation**: Keep it conservative. Use integrated loading-region language. **Do not** make causal, diagnostic, abundance, predictive, or biomarker claims.
- **Components**: Exactly 3 components (PC1, PC2, PC3) are reported.

Outputs are written under the active root and range-local folders (e.g., `Results_intra_disease/01_fingerprint_1300/06_biological_pca/`). *Do not generate a root-level `biological_pca/` directory.*

## 7. Control-vs-Disease ML Classification (Disease-vs-Control Mode Only)

`disease_vs_control` mode runs an exploratory, subject-level supervised classification layer (`src/analysis/control_vs_disease_ml.py`, figures/report in `src/analysis/control_vs_disease_ml_report.py`), firing after Biological PCA in `scripts/run_eda.py`. Outputs to `Results_control_vs_disease/07_ml_classification/`. This is a **separate, read-only downstream consumer** of the Biological PCA subject feature matrix — per §6, no ML claim may attach to Biological PCA outputs, and this layer never mutates Biological PCA state.

- **Models**: PLS-DA, L1-logistic (`class_weight='balanced'`), Random Forest (`class_weight='balanced_subsample'`) — each on two feature representations (`whole` fused spectrum, `band_panel` 21-band biochemical panel).
- **Positive class**: Disease (`is_control == False` → 1), documented in every output artifact.
- **Rigor**: Nested CV (`RepeatedStratifiedKFold`, in-fold scaling/tuning/thresholding — never on the outer test fold), full-pipeline label-permutation significance (unrestricted **and** sex-stratified nulls — the sex-stratified p-value is the primary inferential result), out-of-bag bootstrap 95% CIs, BH-FDR across the model family.
- **Confounding is measured, not assumed**: this cohort's sex distribution is significantly associated with class (χ²≈13.9, p≈0.0002 — controls are ~80% female, disease ~46% female). `confounder_analysis.csv` reports a female-only sensitivity refit and a sex-only baseline classifier per model; a spectral finding must survive the sex-stratified permutation null and beat the sex-only baseline. Age is **not modeled anywhere in this layer** — no age data exists for controls in this cohort and no proxy exists in the repository; this is a permanent limitation, not a gap left open for a future fix.
- **Claims discipline**: exploratory only. No diagnostic, biomarker, causal, or clinical-utility claim. Single-cohort, single-site, no external validation. `ml_report.md` states this explicitly at the top of every run.
- **Config**: `analysis.control_vs_disease_ml` in `config/eda.yaml`; fast dev overlay `config/eda_disease_control_fast.yaml` (iteration only, not manuscript numbers).
- Full scientific-rigor rationale, verified data facts, and phase-by-phase implementation status: `plan_ml.md`.

## 8. Visualization Conventions

- `_TARGET_LABELS` maps raw clinical column names to readable labels. Always use `_target_label()` for axes and legends.
- In `plot_correlation_spectrum` and `plot_intra_disease_correlation_overview`, the legend is placed below the x-axis to prevent collision.
- `plot_intra_disease_top_regions` scales dynamically based on bar count and adds value labels to bar ends.

## 9. Cross-Region Performance (Disease-Only)

Cross-region phases (`run_disease_only_cross_region_pipeline` in `src/analysis/disease_only_pipeline.py`) are resampling-heavy. The default path stays **CPU-compatible and deterministic**.

- **Fast dev profile**: `config/eda_disease_fast.yaml` extends `eda_disease_only.yaml` with lower `n_permutations`, bootstrap counts, and CV splits. Use for iteration only—not manuscript/final science.
- **Parallel CPU**: `n_jobs` in `config/eda.yaml` (default `-1`) controls joblib workers for cluster permutation, PLS/sPLS-DA permutation tests, fixed-panel resampling, and minimal panel. Set `n_jobs: 1` for serial debugging.
- **Phase timing**: Each run writes `00_run_metadata/disease_only_cross_region_phase_timing.csv` under the active EDA root. Inspect before optimizing further.
- **Feature-matrix cache**: `src/analysis/clinical_feature_associations.py` caches whole-spectrum Parquet reads in-process (path + mtime/size); callers still get copied frames.
- **GPU**: No GPU solvers in the pipeline. `scripts/check_gpu_readiness.py` is an optional CuPy equivalence gate only. Do not introduce GPU backends without validated CPU parity.

## 10. General Agent Rules

- **Use Existing Tools**: Prefer repo-local helpers, statistics functions (e.g., `src.statistics.cluster_permutation`), and established config patterns over new entry points or redundant logic.
- **Version Control**: Keep generated artifacts out of version control.
- **Non-Destructive Work**: Do not delete unrelated files from result directories; overwrite only deterministic output files.
- **Skip-manifest, not stub files**: When a disease-only cross-region phase's entire input is empty (e.g. zero candidate regions survived an upstream phase, or no target reaches its configured `min_subjects`), the phase must not write header-only CSVs. It returns a status via `src/analysis/phase_status.py::PhaseStatus` instead; the orchestrator aggregates these into `00_run_metadata/phase_status.csv`. This does **not** apply to legitimate per-row status columns inside a populated table (e.g. a per-feature `insufficient_n` row in `biochemical_ratio_associations.csv`) — those must keep writing normally.
