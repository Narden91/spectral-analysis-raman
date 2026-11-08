# Repository Guidance: Raman Spectral Analysis

This is the authoritative contributor guide for the EV Raman analysis
repository. Keep the pipeline reproducible, scientifically conservative, and
safe for an active experimental campaign.

## Operating principles

- Treat every scientific output as exploratory. Do not introduce diagnostic,
  biomarker, causal, predictive, or clinical-utility claims.
- Preserve the hierarchy `subject → sample → spot/area → pixel`. Primary
  inference and Biological PCA are subject-level; pixels are technical
  observations, not independent subjects.
- Use `scripts/run_eda.py` as the sole EDA, Biological PCA, and case-control ML
  entry point. Do not add parallel orchestration scripts.
- Use `scripts/manage_data_store.py` for all materialized-store operations.
  Analysis must not crawl `data/raw` or silently rebuild a stale store.
- Preserve unrelated worktree changes and generated campaign outputs.

## Canonical storage contract

`parsed` is immutable ingestion state. It stores native per-measurement axes
in `native_wavenumbers_{1300,2850}.npy`, raw pixel spectra, coordinates, and
source signatures. The subject directory in the raw hierarchy is authoritative;
a legacy filename prefix is provenance only. Do not interpolate parsed spectra
or expose them as a single globally aligned matrix.

`preprocessed` is the canonical analysis source. It stores aligned normalized
and pre-normalization matrices, fixed wavenumber axes, QC provenance, and
materialized cohort tables. Read it through `src.data_preparation.store` and
`src.data_preparation.loaders`, or the typed `IngestedRamanDataset` /
`AnalysisRamanDataset` wrappers in `src.data_preparation.dataset`
(`open_ingested_dataset`, `open_analysis_dataset`, `prepare_dataset`) — these
never build, rebuild, or fall back silently.

`loaders.py` caches Parquet tables, memmap handles, and derived area/whole-area
summaries per process, keyed by data-source identity (`source_path` +
`manifest_hash`), so a forced rebuild invalidates the cache automatically.
This removes the repeated `read_parquet`/`np.load`/`nanmedian` passes that
previously ran once per caller (Biological PCA, `intra_disease`,
`spectral_loader`) instead of once per dataset. Cache scope is table handles
and small derived summaries only — never densified pixel-level matrices. Call
`clear_loader_caches()` after any out-of-band store mutation.

The manifest schema is an internal compatibility guard. Do not weaken its
validation or add compatibility fallbacks. If materialization changes, rebuild
both stores deliberately and strict-validate them:

```bash
uv run scripts/manage_data_store.py prepare
```

## Preprocessing contract

The implementation is `src/preprocessing/pipeline.py::run_preprocessing`.
The production order is:

```text
QC metrics → dead-pixel screen → cosmic detection/suppression → edge trim
→ linear frozen-grid interpolation without extrapolation → baseline correction
→ quality gate → optional smoothing → normalization
```

- The frozen grids are 734–1788 cm⁻¹ and 2451–3197 cm⁻¹ at 1 cm⁻¹ spacing.
  They were established from native-axis coverage after trimming. Do not derive
  new bounds from a subset, and do not permit extrapolation.
- Cosmic correction is restricted to narrow positive artifacts. Never restore
  a second derivative-based spike-repair pass in the production path.
- Negative baseline residuals and SNV values are valid. Do not clip them to
  zero.
- Fingerprint uses SNV; CH stretch uses area normalization. Biological PCA and
  ML use normalized data. Relative band-integral work uses the paired
  pre-normalization basis only.

## Scientific and modelling rules

- `PCA_1330`, `PCA_2850`, and `PCA_whole` are subject-level interpretations.
  Preserve `spectral_block__raman_shift` identity for fused features.
- The case-control ML layer is read-only downstream of Biological PCA. All
  tuning, scaling, and threshold selection must remain inside outer training
  folds. Permutations must reuse one fixed repeat of the locked outer-fold plan
  for both the observed and every permuted score; repeated CV remains descriptive.
  Persist resumable resampling checkpoints keyed by configuration, data, and
  feature-matrix identity. Report only
  outer-fold feature-importance summaries; forest importances use held-out-fold
  permutation scoring.
- Permutation tests with one reference distribution compared against one observed
  statistic per call (`control_vs_disease_ml`, `pls_regression`, `splsda`,
  `minimal_panel`, `robust_contrasts`) use adaptive early stopping via
  `src.statistics.sequential_permutation`: Clopper-Pearson curtailment (Besag
  & Clifford 1991) on the Monte Carlo p-value, never stopping before
  `permutation_min` draws or before the configured `permutation_check_interval`
  boundary, and never exceeding `permutation_max`
  (`n_permutations` if unset). Always report the actual draws used, not the
  configured ceiling. This does not apply to `src.statistics.cluster_permutation`'s
  shared family-wise max-statistic reference distribution (compared against many simultaneous
  cluster p-values per spectral block) — that case is not curtailed and keeps
  its fixed `n_permutations` draw count.
  Simulated under a plausible outcome mixture (mostly clear non-significant,
  some clear signal, a minority borderline near α), clear-cut cases resolve at
  the `permutation_min` floor (~80% fewer draws than the 1000-draw ceiling);
  genuinely borderline cases near α=0.05 save little to nothing, because the
  Clopper-Pearson interval cannot resolve there and correctly keeps drawing —
  this is the statistically required behavior, not a shortfall. Weighted
  average across that mixture: ~72% fewer permutation draws. Applied to
  `control_vs_disease_ml`, whose permutation testing is ~90% of its total
  outer-fold-fit cost, this is a ~65% reduction in that module's total
  compute. Both figures are simulation estimates for planning, not measured
  wall-clock times — actual savings depend on the true, unknown tail
  probabilities of the specific dataset and hypothesis being tested.
- The curated 21 Raman windows are centrally registered in
  `src.analysis.raman_bands`. Keep configuration, analysis, and aggregate plot
  styling aligned with that registry. Assignments are biochemical context, not
  direct molecular quantitation.
- Keep fast configs for development only. Final scientific runs use the base
  `config/eda.yaml` settings through the appropriate wrapper.

## Output and code rules

- Route EDA outputs through the resolved analysis mode:
  `Results_control_vs_disease/` or `Results_intra_disease/`.
- Route spectral plots to `Output_plot/`. Coloured Raman category bands belong
  on aggregate spectral plots, not individual trace plots.
- Preserve run provenance: generated result roots require configuration and
  data-manifest snapshots.
- Prefer existing helpers and config patterns. Keep functions small, typed,
  deterministic, and side-effect explicit.
- Use Parquet for tables, memory-mappable NPY for dense spectral arrays, and
  JSON for manifests. Do not create duplicate CSV/NPZ stores.

## Standard commands

```bash
# Stores
uv run scripts/manage_data_store.py prepare

# Production workflows
uv run scripts/run_eda.py --config config/eda_disease_control.yaml --range both
uv run scripts/run_eda.py --config config/eda_disease_control.yaml --range both --resume-ml
uv run scripts/run_eda.py --config config/eda_disease_only.yaml --range both
uv run scripts/run_spectral_plots.py --config config/plots.yaml

# Verification
uv run pytest -q
uv run python -m compileall -q src scripts
```

## Before handoff

1. Run focused tests for every changed subsystem.
2. Run `git diff --check`.
3. State whether `uv run scripts/manage_data_store.py prepare --force` is required.
4. Do not claim a campaign is ready if strict store validation or required
   scientific safeguards are incomplete.
