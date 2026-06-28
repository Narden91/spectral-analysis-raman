# Repository Guidance: Raman Spectral Analysis

This repository contains a Python Raman spectroscopy analysis pipeline for extracellular-vesicle data. The maintained workflow covers preprocessing/QC, exploratory data analysis, visualization, and a strict subject-level Biological PCA layer. Treat all outputs as exploratory scientific analysis, not diagnosis, prediction, or supervised classification.

## 1. Quick Start & Entry Points

- `uv run scripts/run_eda.py`: Main EDA entry point and the **only** entry point for Biological PCA.
- `uv run scripts/run_spectral_plots.py`: Spectral comparison and QC plotting.

### Common Commands
- Smoke test (full resampling): `uv run scripts/run_eda.py --range both --max-subjects 8`
- Fast disease-only iteration (reduced permutations/CV; not for final outputs): `uv run scripts/run_eda.py --config config/eda_disease_fast.yaml --range both --max-subjects 8`
- Run all tests: `uv run pytest -v`
- Run biological PCA tests: `uv run pytest tests/test_biological_pca.py`
- Optional GPU readiness probe (does not change pipeline): `uv run scripts/check_gpu_readiness.py`

## 2. Core Architecture

The codebase follows a modular structure:
- `src/io/`: file indexing and Renishaw text loading.
- `src/metadata/`: filename metadata parsing.
- `src/data_preparation/`: parsed and preprocessed store materialization, validation, loading, and cohorts.
- `src/preprocessing/`: Raman preprocessing and QC.
- `src/aggregation/`: subject and area aggregation helpers.
- `src/fusion/`: cross-range and area-level fusion helpers.
- `src/analysis/`: EDA, PCA, whole-spectra analysis, and Biological PCA.
- `src/visualization/`: plotting and report figures.

Configuration is YAML-based. Track configs use `_extends`; `config/eda.yaml` owns EDA and Biological PCA settings.

## 3. Data Source Rules

The repository supports three explicit source levels: `raw_text`, `parsed`, and `preprocessed` (default).

- **Load safely**: Normal analysis must load through `src/data_preparation/store.py` and `src/data_preparation/loaders.py`.
- **No raw crawling**: Do not crawl `data/raw` during normal analysis unless `raw_text` is explicitly selected.
- **No silent fallbacks**: Do not silently fall back across sources or silently rebuild stale generated stores during analysis.
- **Formats**: Use Parquet for canonical tables, NPZ for dense arrays, and JSON for manifests. CSV mirrors are optional inspection artifacts only.
- **Artifacts**: Result directories must include `run_manifest.json`, `data_manifest_snapshot.json`, and `config_snapshot.yaml`.

## 4. Output Routing

Outputs are routed through the resolved EDA analysis mode (`--analysis-mode` or `analysis.mode` in config).
- **Disease vs Control**: `Results_control_vs_disease/`
- **Disease Only**: `Results_intra_disease/`

## 5. Disease-Only Pipeline (14 Phases)

The `disease_only` pipeline mode executes a comprehensive suite of cross-region summarizations and modelling, outputting to `Results_intra_disease/04_cross_region_summary/`.

- **Phases 1-5**: Base clinical correlation matrices, longitudinal delta MoCA testing, and disease-internal subject stratification.
- **Phase 6 (Candidate Regions)**: Unifies hits from phases 1-5 into a consolidated, cluster-corrected table (`candidate_regions_cluster_corrected.csv`).
- **Phase 7 (Covariate Models)**: Fits covariate-adjusted models (age, sex) on Phase 6 candidate regions and logs persistence.
- **Phase 8 (PLS1 Regression)**: Supervised continuous modelling of endpoints with permutation testing and VIP selection.
- **Phase 9 (PLS-L1DA)**: Supervised binary classifier, gated strictly by Phase 8 significance.
- **Phase 10 (Biochemical Ratios)**: Computes targeted band-intensity ratios via trapezoidal integration and assesses Spearman correlation with clinical endpoints.
- **Phase 11 (Integrated Candidate Regions)**: Merges univariate hits, covariate persistence, biological PCA loading overlap, PLS VIP coverage, and biochemical interpretation into a single priority-ranked table (`High`/`Medium`/`Low`) written to `11_integrated_candidates/integrated_candidate_regions.csv`.
- **Phase 12 (Disease Subgroups)**: Exploratory agglomerative Ward clustering evaluated via Gap statistic and bootstrap Adjusted Rand Index.
- **Phase 13 (Minimal Candidate Panel)**: Selects 5–10 spectral peaks/ratios per endpoint using nested cross-validated ElasticNet / L1-penalized logistic regression with permutation testing and bootstrap confidence intervals. Outputs `13_minimal_panel/minimal_candidate_panel.csv`, `minimal_panel_cv_performance.csv`, and `minimal_panel_coefficients.csv`.
- **Phase 14 (Publication Report)**: Generates a conservative markdown report and summary CSV that classifies findings as robust, exploratory, unstable, or confounded. Outputs `14_publication_report/candidate_region_report.md`, `candidate_region_summary.csv`, and `candidate_region_figures/`.

## 6. Biological PCA Rules

Biological PCA is subject-level only and is the authoritative PCA layer per region. Area-level EDA PCA remains but is demoted to exploratory/QC use only.

- **Models**: `PCA_1330`, `PCA_2850`, `PCA_whole`.
- **Concatenation**: `PCA_whole` is an unweighted concatenation of the full median spectra.
- **Identity**: Whole-spectrum columns and loadings must preserve block identity (e.g., `1300__1240.0`).
- **Interpretation**: Keep it conservative. Use integrated loading-region language. **Do not** make causal, diagnostic, abundance, predictive, or biomarker claims.
- **Components**: Exactly 3 components (PC1, PC2, PC3) are reported.

Outputs are written under the active root and range-local folders (e.g., `Results_intra_disease/01_fingerprint_1300/06_biological_pca/`). *Do not generate a root-level `biological_pca/` directory.*

## 7. Visualization Conventions

- `_TARGET_LABELS` maps raw clinical column names to readable labels. Always use `_target_label()` for axes and legends.
- In `plot_correlation_spectrum` and `plot_intra_disease_correlation_overview`, the legend is placed below the x-axis to prevent collision.
- `plot_intra_disease_top_regions` scales dynamically based on bar count and adds value labels to bar ends.

## 8. Cross-Region Performance (Disease-Only)

Cross-region phases (`run_disease_only_cross_region_pipeline` in `src/analysis/disease_only_pipeline.py`) are resampling-heavy. The default path stays **CPU-compatible and deterministic**.

- **Fast dev profile**: `config/eda_disease_fast.yaml` extends `eda_disease_only.yaml` with lower `n_permutations`, bootstrap counts, and CV splits. Use for iteration only—not manuscript/final science.
- **Parallel CPU**: `n_jobs` in `config/eda.yaml` (default `-1`) controls joblib workers for cluster permutation, PLS/sPLS-DA permutation tests, confirmatory-panel resampling, and minimal panel. Set `n_jobs: 1` for serial debugging.
- **Phase timing**: Each run writes `00_run_metadata/disease_only_cross_region_phase_timing.csv` under the active EDA root. Inspect before optimizing further.
- **Feature-matrix cache**: `src/analysis/clinical_feature_associations.py` caches whole-spectrum Parquet reads in-process (path + mtime/size); callers still get copied frames.
- **GPU**: No GPU solvers in the pipeline. `scripts/check_gpu_readiness.py` is an optional CuPy equivalence gate only. Do not introduce GPU backends without validated CPU parity.

## 9. General Agent Rules

- **Use Existing Tools**: Prefer repo-local helpers, statistics functions (e.g., `src.statistics.cluster_permutation`), and established config patterns over new entry points or redundant logic.
- **Version Control**: Keep generated artifacts out of version control.
- **Non-Destructive Work**: Do not delete unrelated files from result directories; overwrite only deterministic output files.
