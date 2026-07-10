# Spectral Analysis Preprocessing Pipeline

This document outlines the end-to-end preprocessing pipeline for Raman spectral analysis within the codebase, detailing the sequence of operations, the underlying biological and methodological rationale, and the resulting data structure.

## 1. Core Motivation

The core philosophy of this pipeline is: **"Get the preprocessing right first — it dominates everything downstream."**
Raman spectra from biological tissues are highly heterogeneous and contain multiple confounding artifacts:
- Sharp, isolated intensity spikes from cosmic rays hitting the CCD.
- Broad, varying fluorescent background signals.
- Sensor noise that obscures underlying biochemical signals.
- Scattering effects that modulate intensity irrespective of concentration.

Because these artifacts differ systematically between spectral regions—especially between the fingerprint region (400-1800 cm⁻¹) and the high-wavenumber CH-stretch region (2800-3000 cm⁻¹)—they **cannot** be treated with a single global pipeline. They are processed separately using tailored algorithms.

## 2. Data Sources, Structure & Aggregation Strategy

The repository now separates ingestion from analysis through three explicit
source levels:

- `raw_text`: crawl `data/raw` and `data/clinical_data.csv`
- `parsed`: materialized raw metadata and spectra
- `preprocessed`: materialized post-pipeline spectra and cohort tables

`parsed` replaces the older idea of an "unpreprocessed" output folder. The goal
is to crawl raw text once, validate it, and reuse deterministic materialized
stores across analyses.

Canonical generated assets:

- `file_index.parquet`
- `clinical_subjects.parquet`
- `area_metadata.parquet`
- `spectra_1300.npz`, `spectra_2850.npz`
- `area_summaries_1300.npz`, `area_summaries_2850.npz`
- `whole_area_store.npz`
- `manifest.json`
- `cohort/control_vs_disease/*.parquet`
- `cohort/disease_only/*.parquet`

The data exhibits an intrinsically nested hierarchical structure:
`Subject -> Slide -> Spot (Area) -> Pixels`

### Aggregation: The Hybrid Approach

A critical decision in Raman analysis is how to aggregate pixels. 
- **Averaging everything to one subject mean** destroys intra-subject heterogeneity.
- **Keeping all pixels** inflates statistical significance (pseudoreplication) and makes clustering (like PCA) computationally heavy and noisy.

This pipeline utilizes a **Hybrid Aggregation** strategy:
1. We compute the **median spectrum** for each physical spot (area). The median is statistically robust against pixel-level outliers.
2. We **keep the spots separate** in the downstream EDA matrix (PCA).
3. This yields approximately ~3 points per subject, preserving physical heterogeneity, reducing pixel-level noise, and preventing the "overweighting" of larger physical spots.

## 3. Pipeline Stages

The pipeline acts independently on raw pixel arrays, ensuring no pixel bleeds artifacts into its neighbors during early stages. The orchestrator sequentially applies the following steps:

> **Authoritative production order:** initial dead/saturation screening; cosmic handling; edge trim; baseline correction; post-baseline SNR QC; spike repair; smoothing; normalization. Any older numbered text below that places spectral-quality filtering before baseline correction is superseded by this order. Batch-relative area-percentile rejection is disabled, and no absolute area floor is active without an independent calibration set.

### Stage 1: Quality Control & Dead/Anomalous Pixel Screening
- **Method**: Standard deviation screening and saturation check.
- **Rationale**: Pixels with very low variability (dead pixels, $\text{std} < 1.0$) or detector saturation ($> 65000$) are excluded early to avoid injecting unphysical signals.

### Stage 2: Cosmic Ray Detection and Suppression
- **Method**: Spatial/Neighbor Median Detection (Z-score based) with Local Median fallback when spatial coordinates are unavailable.
- **Rationale**: Cosmic rays hit the CCD detector as sharp, highly localized intensity spikes. Comparing each pixel's spectrum to its physical neighbors on the grid map identifies these anomalies robustly so they can be repaired (replaced with local medians) or flagged for exclusion.

### Stage 5: Post-baseline Spectral Quality Filtering
- **Method**: Deterministic post-baseline SNR gating before smoothing; integrated signal area is diagnostic only.
- **Rationale**: SNR describes Raman residual signal rather than fluorescence background. Batch-relative percentile rejection is disabled; `min_integrated_signal` remains unset pending independent calibration.

### Stage 3: Edge Trimming
- **Method**: Spectral boundary trimming (10 channels or 15.0 cm⁻¹ alternative).
- **Rationale**: Speeds and margins of the spectrometer CCD arrays are prone to border noise and artifact corruption. Trimming the ends ensures subsequent baseline fits are stable.

### Stage 4: Baseline Correction
- **Method**: Asymmetrically Reweighted Penalized Least Squares (arPLS), Asymmetric Least Squares (AsLS), or Statistics-sensitive Non-linear Iterative Peak-clipping (SNIP).
- **Rationale**: Biological fluorescence creates a large, slow-varying baseline. arPLS estimates this background smoothly without polynomial oscillations. Signed residuals are preserved; `negative_fraction` and `negative_area_fraction` are descriptive QC metrics, not rejection rules.

### Stage 6: Isolated Spike Repair
- **Method**: Whitaker-Hayes Z-score check on discrete differences.
- **Rationale**: Sharp, single-channel instrument spikes are flagged and replaced by local neighborhood means if the overall corrected fraction is small ($< 2\%$).

### Stage 7: Smoothing
- **Method**: Savitzky-Golay filter (Window: 7-9 channels, Polynomial Order: 2).
- **Rationale**: Minimizes high-frequency sensor noise. Restricting the window prevents "oversmoothing" that would artificially merge separate adjacent peaks.

### Stage 8: Interpolation
- **Method**: Resampling to a uniform grid (default 1.0 cm⁻¹ spacing) using cubic or linear interpolation.
- **Rationale**: Combines spectra onto a shared wavenumber coordinate system for aggregations and joint matrix operations.

### Stage 9: Outlier Detection
- **Method**: Total-area outlier check using Median Absolute Deviation (MAD > 3.0).
- **Rationale**: Identifies anomalous scans as warnings for downstream inspection rather than immediate hard exclusions.

### Stage 10: Normalization (Region-Specific)
Different spectral regions carry distinct physical meaning, requiring customized normalization strategies:
- **Fingerprint Region (1300 cm⁻¹)**: Standard Normal Variate (SNV) or L2 Vector Normalization. This removes multiplicative scatter from sample surface roughness while preserving relative peak heights.
- **High-Wavenumber CH-Stretch Region (2850 cm⁻¹)**: Area Normalization (Integration to 1.0). Frames the region as a proportional density metric, aligning with biomass/lipid-to-protein literature conventions.

## 4. Execution Flow

1. Raw `.txt` hypercubes are loaded when building the `parsed` store.
2. Spectra are split into the two regions.
3. The `src/preprocessing/pipeline.py` orchestrator sequentially applies the stages (QC -> Baseline -> Smooth -> Normalize) configured via `config/settings.yaml` when building the `preprocessed` store.
4. Artifacts and outliers are masked.
5. The pixel spectra are collapsed to area medians via `src/fusion/area_summary.py`.
6. Cohort tables are materialized under the active generated store.
7. EDA, spectral plots, and Biological PCA load from the selected source instead of regenerating cohort tables under `Results_*/00_cohort`.

## 5. Manifest And Validation

Each generated store writes a `manifest.json` with:

- store type (`parsed` or `preprocessed`)
- source path and source manifest hash
- input file signatures
- generated file signatures
- preprocessing and materialization config hashes
- record counts

Derived stores are validated before use. Stale manifests fail by default and
are rebuilt only through an explicit force build command.
