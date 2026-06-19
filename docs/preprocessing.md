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

## 2. Data Structure & Aggregation Strategy

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

The pipeline acts independently on raw pixel arrays, ensuring no pixel bleeds artifacts into its neighbors during early stages.

### Stage 1: Quality Control & Cosmic Ray Removal
- **Method**: Spatial Median Detection (Z-score based) or local outlier factor (LOF) on neighboring pixels.
- **Rationale**: Cosmic rays are highly localized in both spectral width and spatial dimensions. By comparing a pixel to its physical neighbors on the 5x5 map, cosmic rays can be robustly identified and repaired (or the pixel excluded) without distorting sharp natural Raman peaks.

### Stage 2: Baseline Correction
- **Method**: Asymmetric Least Squares (AsLS) or Asymmetrically Reweighted Penalized Least Squares (arPLS).
- **Rationale**: Biological fluorescence manifests as a broad, varying baseline. Polynomial fitting often over-fits or introduces oscillations (Runge's phenomenon), particularly in the narrow CH-stretch region. Penalty-based methods (arPLS) estimate the baseline cleanly without high-order polynomial distortion.

### Stage 3: Smoothing
- **Method**: Savitzky-Golay filter (Window: 7-9, Polynomial Order: 2).
- **Rationale**: Mitigates high-frequency sensor noise. We restrict the window size to prevent "oversmoothing," which would artificially merge adjacent biochemical peaks (e.g., Amide I and lipid bands) and degrade the biological resolution.

### Stage 4: Normalization (Region-Specific)

Different regions carry different physical meaning, requiring disparate normalization strategies.

#### Fingerprint Region (1300 cm⁻¹ approx)
- **Method**: Standard Normal Variate (SNV) or Vector (L2) Normalization.
- **Rationale**: The fingerprint region contains a complex superposition of proteins, lipids, and nucleic acids. SNV removes multiplicative scatter effects (e.g., sample surface roughness) while preserving relative peak heights.

#### High-Wavenumber CH-Stretch Region (2850 cm⁻¹ approx)
- **Method**: Area Normalization (Integration to 1).
- **Rationale**: The CH-stretch band (dominated by lipids and proteins) acts as an excellent internal standard for total biomass. Area normalization explicitly frames this region as a proportional density metric, aligning perfectly with standard literature practices for CH-band analysis.

## 4. Execution Flow

1. Raw `.txt` hypercubes are loaded.
2. Spectra are split into the two regions.
3. The `src/preprocessing/pipeline.py` orchestrator sequentially applies the stages (QC -> Baseline -> Smooth -> Normalize) configured via `config/settings.yaml`.
4. Artifacts and outliers are masked.
5. The pixel spectra are collapsed to area medians via `src/fusion/area_summary.py`.
6. PCA and EDA plots are generated dynamically from these aggregated spot medians.
