# Manuscript ML experiment runbook

Updated 20 November 2026 after calibration revision and pre-evaluation readiness upgrades. The active specification is [docs/research_design.md](../docs/research_design.md); the manuscript is [ml_paper/main.tex](../ml_paper/main.tex). Mathematical routines live in [src/analysis/paired_lipid/](../src/analysis/paired_lipid/), with settings in [config/manuscript_ml.yaml](../config/manuscript_ml.yaml). Each CLI accepts `--config` and records `run_provenance_<stage>.json`.

## Recorded status and evidence limits

The [9 November verification record](../AGENT_LEARNINGS.md#2026-11-09-verified-facts-about-the-manuscript-ml-pipeline-do-not-re-test) reports a successful real cohort join: 92 participants, 88 paired, 4 single-sample and 180 eligible samples. Numerical checks used real spectra, covariates, labels and folds with seeded dummy outcomes. No cohort lipid result follows from those checks. The 10 November revision reviewed code and documentation; it did not rerun cohort analyses or simulation campaigns.

Machine readiness does not establish measurement readiness. Source lipid units remain assumed, assay method and blood-draw correspondence need confirmation, and independently repeated preparation/acquisition stages remain undocumented. Keep these limits visible even when `ready_for_cohort_evaluation` is true.

## Runtime and execution budget

The following measurements were recorded on Python 3.12 with an NVIDIA GeForce RTX 5080 Laptop GPU (15.9 GiB VRAM).

| Workload | Recorded result | Limit on interpretation |
|---|---|---|
| Real spectra with dummy outcomes; GPU exact fp64 + CheFSI (20 Nov 2026); 2,000 bootstrap draws | 10.19 s; 2.83 GiB peak pool; 0 fallbacks | Single cohort bootstrap (CheFSI active on 20/20 real divisions, max diff 2.2e-14 vs sequential refits); not a synthetic calibration replicate. |
| Synthetic calibration replicate; 16-core CPU parallel LAPACK `dsyevr` (20 Nov 2026); 2,000 draws; `compact_grid: false` (1 cm^-1 cohort grid); `n=92`, `n_test=25000` (5,000 timing baseline) | 8.66 s/rep (138.6 s for 16 reps at `n_test=5000`; fused external PLS keeps `n_test=25000` overhead small) | Default auto-routed backend for multi-replicate calibration (`use_gpu=None` with >= 4 CPU cores): ~4x faster than serial GPU (~2.4 h per 1,000-rep scenario). |
| Synthetic calibration replicate; GPU exact fp64 (20 Nov 2026); 2,000 draws; `compact_grid: false` (1 cm^-1 cohort grid); `n=92`, `n_test=5000` | 34.4 s/rep; 2.16 GiB peak pool | Flat noise tail beyond 3 synthetic factors bypasses GPU CheFSI and routes to serial active-sample `eigh` (~9.6 h per 1,000-rep scenario); omit `--use-gpu` on calibration unless fewer than 4 CPU cores are available. |

Use exact fp64 eigensolvers for reported bootstrap results. Earlier mixed-precision checks changed draws by up to 3e-2 against the sequential procedure. The recorded fast-versus-sequential checks cover CPU and exact-fp64 GPU (including CheFSI on real spectra); repeat only what changed inputs or code affect.

Before a long campaign, time one intended development replicate in a separate output directory and record grid, participant count, candidate settings, bootstrap count, device, runtime and memory. Obtain approval if even that check is expected to exceed five minutes. Estimate the complete campaign, including scenario evaluations and calibration target scoring, from measured costs. Do not extrapolate the real-data bootstrap timing as the whole campaign cost. Any experiment expected to exceed five minutes requires separate approval.

## 1. Input and measurement readiness

- Verify the versioned standalone [clinical master](../data/clinical_master.csv), [dictionary](../data/clinical_master_dictionary.csv) and [source notes](../data/clinical_master_notes.md).
- Confirm identity review exclusions and the 92/88/4/180 eligibility counts against the recorded input identity. Recompute only if inputs or relevant code changed.
- Confirm lipid units, direct versus calculated LDL-C, statin reference time, collection correspondence and the stages independently repeated. Missing records restrict interpretation; do not infer answers from populated clinical columns.
- Archive grids, covariates, inclusive band windows, aggregation rules, seeds and data fingerprints before endpoint evaluation. Acquisition modes FP and SS are pooled as equivalent; the numeric sample label is an unverified nuisance proxy, not established physical order.
- Schema v5 (`SCHEMA_VERSION = 5`) requires materialized `data/parsed` and `data/preprocessed` stores referencing `data/clinical_master.csv` and `data/clinical_master_dictionary.csv`. If `audit_readiness.py` reports a schema-version or store-validation failure, rebuild with `uv run python scripts/manage_data_store.py prepare --force` and verify `ready_for_cohort_evaluation: true` before running any cohort stage.

The readiness audit command is:

```powershell
uv run --group gpu python manuscript_ml/audit_readiness.py --output-dir Results_manuscript_ml/readiness_review
```

The audit checks current clinical identities and real-store join counts without fitting models or running a synthetic timing benchmark. Computational runtime is marked unmeasured until a separately authorized real-data check records it. Inspect `cohort_join`, expected-count checks, data identity and `limitations_table.csv`; the real-cohort resolver enforces the identity and join-count checks, while the clinical metadata requirements remain separate.

## 2. Simulation development and coverage validation

Known-factor simulations answer statistical questions. They do not replace the investigator-required numerical verification on real inputs with dummy outcomes. Existing verification facts are in the evidence record; routine reruns of unchanged checks are unnecessary.

The development smoke command is:

```powershell
uv run --group gpu python manuscript_ml/run_simulation_calibration.py --quick --output-dir Results_manuscript_ml/synthetic_smoke
```

This uses development seeds and reduced settings. Small-sample interval failures are reported, not discarded. It cannot establish coverage. The configured performance suite includes linear signal, no-association, nonlinear covariate proxy, A/B shifts, label drift, formula-linked anchors, correlated anchors and map dropout. Inspect estimates and recovery without requiring any one finite realization to have a preferred sign.

**Multi-scenario calibration status and 20 Nov 2026 revision to centered studentized bootstrap-$t$:** `evaluate_scenario_calibration_suite` (`--calibrate-all-scenarios`) calibrates both `ci_95_delta_mse` (vs covariate-only ridge) and `ci_95_vs_comparator` (vs inner-selected strongest comparator) across all 17 configured scenarios (10 core/legacy plus 7 stress regimes covering high-dimensional technical nuisance, strong shared preparation, correlated/unequal A/B sample errors, nonlinear spectral signal, correlated within-sample maps, and covariate missingness) and supports the pre-specified MDE signal-strength curve (`--run-mde-curve`, outputting `simulation_mde_curve.csv`). A 20 Nov 2026 development check showed that raw percentile bootstrap intervals undercovered `nonlinear_proxy_linear_covariates` (`83.3–85.4%`, right-tail misses) and overcovered `ci_95_vs_comparator` (`100%`, `1.88–2.65×` SD ratio). The locked default estimator is now the centered full-procedure studentized participant-multiplicity bootstrap-$t$ (`--interval-method studentized_t`, default in `config/manuscript_ml.yaml`, with raw percentile intervals also retained in primary output). On the updated 5-scenario development benchmark (`n=92`, `n_single_sample=4`, `R=96`, `B=250`, `n_test=25,000`, `compact_grid=True`), centered studentized-$t$ achieved `96.9–97.9%` primary coverage on linear-signal, no-association, A/B-shift, and high-dimensional nuisance (`92.7%` / 89 of 96 at `R=96` on `nonlinear_proxy_linear_covariates`, rising to `96.1%` / 246 of 256, Wilson lower `0.930`, `0.0%` failures in a follow-up `R=256` development check under both `n_single_sample=4` and `n_single_sample=2`), `97.9–100.0%` comparator coverage (`1.20–1.57×` SD ratio), `0.0%` failures, and `target_delta_vs_comparator = +0.0732` on `high_dim_technical_nuisance`; the remaining 6 stress regimes (`R=48`, `B=200`, `n_test=10,000`) likewise had `0.0%` dataset failures (including `covariate_missingness` at `97.9%` primary / `93.8%` comparator coverage). Raising `calibration.n_test_participants` to `25000` cuts `target_test_cohort_se` from ~`0.019–0.022` to ~`0.008–0.010`.

The validation-seed multi-scenario cohort-grid invocation (using default parallel CPU routing, centered studentized-$t$ intervals, `n_test_participants = 25000`, and resumable checkpoints), after separate runtime approval, is:

```powershell
uv run --group gpu python manuscript_ml/run_simulation_calibration.py --scenario-reps 1000 --calibration-reps 1000 --bootstraps 2000 --participants 92 --n-test-participants 25000 --interval-method studentized_t --cohort-grid --seed-set validation --calibrate-all-scenarios --run-mde-curve --resume --output-dir Results_manuscript_ml/validation_all_scenarios
```

Cluster submission uses one script per experiment under [slurm_jobs/](../slurm_jobs). Configuration is in [slurm_jobs/README.md](../slurm_jobs/README.md). The checkbox command list is [slurm_jobs/runbook_hpc.md](../slurm_jobs/runbook_hpc.md). Names follow pipeline order and the config scenario list (`02_simulation_01_linear_biological_signal.sbatch` through `02_simulation_18_mde_signal_curve.sbatch`, then the cohort scripts). Submit each stage with its helper, `bash slurm_jobs/submit_NN_*.sh` (simulations: `submit_02_simulation.sh`; each helper runs preflight first). Each scenario writes `Results_manuscript_ml/validation/<scenario>/` so the shared CSV names do not collide; `uv run python slurm_jobs/merge_simulation_reports.py` concatenates them afterwards. The simulation helper never submits cohort scripts. Cohort helpers (`submit_03_*`, `submit_04_*`, `submit_cohort.sh`) require `--approve`, the separate authorization step.

Omit `--calibrate-all-scenarios` for single-setting calibration, or pass `--scenarios <comma-separated-names>` (which preserves canonical scenario seed offsets) and `--skip-scenario-suite` to run or resume calibration per scenario batch. `--cohort-grid` overrides `simulation.compact_grid: true` from the checked-in config and records `compact_grid: false` in `run_provenance_simulation_calibration.json`. `run_all.py` supports the same simulation flags (`--cohort-grid`, `--participants`, `--n-test-participants`, `--interval-method`, `--scenarios`, `--skip-scenario-suite`, `--run-mde-curve`, `--resume`, `--checkpoint-chunk-size`, `--batch-size`).

For each core scenario, report unconditional coverage with failures counted as noncoverage, coverage among successful intervals, failure rate, bias, width, two-way target-estimation SEs (`target_monte_carlo_se`, `target_test_cohort_se`, `target_total_se`), `target_sensitivity_min_coverage`, `detection_rate`, and Monte Carlo uncertainty for both primary and comparator intervals. The provisional thresholds are Wilson 95% lower bound at least 0.925 and failed dataset intervals below 1%. At least 48 perfect repetitions are needed even to attain that gate; 48 is not a replacement for the planned 1,000 per core scenario.

The target is the mean prediction-risk difference of the replicate outer-fold models on one shared independent test cohort of 25,000 simulated participants (`calibration.n_test_participants: 25000`, with matched single-sample proportion). `target_monte_carlo_se` captures training-replicate variation conditional on that test cohort, `target_test_cohort_se` captures finite test-cohort variation, and `target_total_se` combines both via the crossed two-way random-effects variance estimator. Because the coverage indicators still share one estimated target across replicates, exact binomial interpretation of the Wilson summary remains approximate; inspect `target_sensitivity_min_coverage` alongside `wilson_95_lower`. A `gate_is_confirmatory` flag identifies validation seeds only.

## 3. Primary cohort evaluation

Cohort endpoint evaluation follows the archived input/measurement decisions and a decision about inferential versus descriptive reporting. The commands below are operational examples, not new authorization to analyze outcomes.

Point estimate:

```powershell
uv run --group gpu python manuscript_ml/run_primary_evaluation.py --cohort --approve-cohort-evaluation --bootstraps 0 --output-dir Results_manuscript_ml/cohort_primary_point
```

Complete-procedure bootstrap:

```powershell
uv run --group gpu python manuscript_ml/run_primary_evaluation.py --cohort --approve-cohort-evaluation --bootstraps 2000 --output-dir Results_manuscript_ml/cohort_primary_bootstrap
```

Use separate output directories to retain the original point run. Runtime on the intended full configuration must be checked against the approval limit.

Inspect these artifacts:

| Artifact | Required interpretation |
|---|---|
| `eligibility_and_split_manifest.json` | Private eligibility and split identity, centered studentized-$t$ and percentile bootstrap intervals (primary, vs comparator, strongest vs covariate), distinct-participant counts, Maritz–Jarrett quantile endpoint MC SEs, and failure status. |
| `bootstrap_draws.npz` | Raw bootstrap draw arrays and per-draw out-of-fold loss-difference SDs when `--bootstraps > 0`. |
| `nested_held_out_predictions.csv` | Private participant predictions; do not publish identifiers or individual clinical values. |
| `comparator_errors_and_differences.csv` | Report each model, comparison baseline, role, error difference, relative reduction, RMSE, MAE and descriptive calibration. |
| `fold_selected_settings.csv` | Check training-only selection, inner-validation MSEs, `tr_T`/`tr_W`/`top_rhos_json`, inner division ranks/dropped covariates, and invalid candidate reasons; never select a comparator using outer errors. |
| `baseline_repeatability_report.csv` | Fold-specific paired counts, full-prediction versus spectral-contribution ICCs (for paired, fixed-band, PCA, PLS, and inner-selected strongest comparator), ANOVA mean squares, spectral contribution SD and variance share, RMSD, discriminability and retrieval. |
| `subspace_principal_angles.csv` | Dimensions and individual principal angles (`angle_1_deg`..`angle_3_deg`) for both `paired_linear_ridge` and `pca_ridge` in the shared equal-block, equal-channel metric. |
| `fold_spectral_weights.csv` | Canonical 1,802-channel fold-specific effective spectral regression weights mapped back through PCA and MFA scaling with locked 21-band tags. |

Keep covariate-only-minus-paired, inner-selected-comparator-minus-paired, and covariate-only-minus-inner-selected-comparator differences distinct. Bootstrap ranges remain descriptive until calibration requirements are met. Do not pool fold-specific ICCs, interpret agreement as particle identity, or treat random participant splits as external validation.

## 4. Sensitivity and secondary endpoints

After the primary evaluation is authorized and locked:

```powershell
uv run --group gpu python manuscript_ml/run_sensitivity_and_secondary.py --cohort --approve-cohort-evaluation --output-dir Results_manuscript_ml/cohort_sensitivity
```

Pass `--bootstraps 2000` when bootstrap uncertainty intervals are also requested for the covariate-preset and subset-mode rows in `covariate_and_subset_sensitivity.csv`. Report all covariate and subset settings (including `mse_covariate`, `mse_paired`, comparator MSEs, `strongest_comparator_name`, and `attenuation_vs_primary_delta_mse`) and all locked additional partitions in `outer_partition_seed_sensitivity.csv`, without choosing the best partition. The synthetic generator's subset summaries do not independently reproduce real map-level QC selection.

Report `secondary_endpoints_summary.csv` for HDL-C, triglycerides and platelets on the primary population and outer membership, with complete inner selection repeated for each target and missing targets given zero weight. Keep secondary comparisons and `anchor_associations_continuous.csv` (2,000 resamples, explicit `target_role`, and per-fold partial-correlation summaries) descriptive. Formula-linked lipid anchors do not supply independent biological confirmation; clinical platelet count does not measure residual platelet material. Categorical attribution requires a separately frozen and validated rule.

## 5. Manuscript artifact generation, checks and interpretation

- Build deterministic LaTeX table fragments and diagnostic figures from authorized stage outputs via `uv run --group gpu python manuscript_ml/build_manuscript_artifacts.py --input-dir <results_dir> --output-dir <artifacts_dir>` (the builder refuses non-cohort outputs unless `--allow-synthetic-preview` is passed, which watermarks every preview table and figure).

- Keep pending fields pending until authorized study artifacts exist. Engineering checks with dummy outcomes cannot fill prediction, repeatability or coverage tables.
- A supported primary improvement requires a positive estimate and an adequately calibrated interval excluding zero. An interval spanning zero is inconclusive.
- A separate methodological claim needs a reproducible advantage over the established comparator selected in inner validation. Similar point estimates do not establish equivalence.
- Describe relative normalized spectral shape under the measured protocol. Do not infer EV specificity, particle concentration, brain origin, causal effects or clinical utility.
- Retain failed analyses and limitations. An unsupported primary result does not authorize switching outcomes or preprocessing to rescue the claim.
- Build the LaTeX draft and check citations, table references, wording and links. Keep participant artifacts private; publication and release require separate approval.


Cohort stages now default to real data and still require `--approve-cohort-evaluation`. Use `--synthetic` only for explicitly labelled development work in a separate output directory. The cohort `run_all.py` workflow runs simulation calibration only with `--include-simulation-calibration`; the dedicated calibration script remains an explicit development tool.
