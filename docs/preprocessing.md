# Preprocessing and Data Materialization

Build analysis-ready data from raw files:

```bash
uv run scripts/manage_data_store.py prepare
```

The command builds both stores and checks their contents before reporting
success. Progress bars show ingestion and preprocessing.

## Identity and hierarchy

```text
subject → sample → spot/area → pixel
```

The enclosing raw subject folder is authoritative. A legacy filename prefix is
retained as `filename_subject_id` provenance only and cannot override it.

## Stores

`data/parsed/` is immutable ingestion state. It contains native raw pixel
spectra, coordinates, source signatures, and one calibrated axis per
measurement in `native_wavenumbers_{1300,2850}.npy`. It is not a globally
aligned analysis matrix.

`data/preprocessed/` is the canonical analysis source. It contains:

```text
normalized_{1300,2850}.npy
pre_normalization_{1300,2850}.npy
wavenumbers_{1300,2850}.npy
measurements.parquet, subjects.parquet, areas.parquet, qc.parquet
manifest.json and cohort/
```

Primary EDA, Biological PCA, and ML use normalized data. The pre-normalization
basis is reserved for the dedicated relative band-integral workflow.

## Processing sequence

```text
native pixels → dead/saturation screen → conservative positive cosmic handling
→ 15 cm⁻¹ edge trim → frozen-grid linear interpolation (no extrapolation)
→ baseline correction → deterministic quality checks → optional smoothing
→ region-specific normalization
```

| Region | Frozen 1 cm⁻¹ grid | Normalization |
|---|---|---|
| Fingerprint | 734–1788 cm⁻¹ | SNV |
| CH stretch | 2451–3197 cm⁻¹ | Trapezoidal area |

The bounds are supported by every retained native axis after trimming. Any
attempted extrapolation is a hard error. Signed baseline residuals and negative
SNV values are valid and are not clipped to zero.

`run_preprocessing(..., cfg=...)` uses the supplied configuration throughout
QC and preprocessing. Omitting `cfg` uses the module defaults. Range-specific
baseline settings override the global defaults; the production default method
is arPLS. Materialization passes the resolved run configuration and records it
in the store manifest.

Generated interpolation grids stay within the native axis bounds, including
when the requested spacing does not divide the interval exactly.

## Rebuilds and checks

`prepare` reuses valid stores. After a deliberate raw-data, configuration, or
materialization change, perform a full rebuild:

```bash
uv run scripts/manage_data_store.py prepare --force
```

Staging occurs outside `data/`; only a complete store is atomically promoted.
The command checks source signatures, generated artifacts, axes, shapes,
metadata relations, and manifest provenance before it reports success.
