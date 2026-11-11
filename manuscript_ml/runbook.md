# Manuscript ML experiment runbook

Updated 10 November 2026 after source review. The active specification is [docs/research_design.md](../docs/research_design.md); the manuscript is [ml_paper/main.tex](../ml_paper/main.tex). Mathematical routines live in [src/analysis/paired_lipid/](../src/analysis/paired_lipid/), with settings in [config/manuscript_ml.yaml](../config/manuscript_ml.yaml). Each CLI accepts `--config` and records `run_provenance_<stage>.json`.

## Recorded status and evidence limits

The [9 November verification record](../AGENT_LEARNINGS.md#2026-11-09-verified-facts-about-the-manuscript-ml-pipeline-do-not-re-test) reports a successful real cohort join: 92 participants, 88 paired, 4 single-sample and 180 eligible samples. Numerical checks used real spectra, covariates, labels and folds with seeded dummy outcomes. No cohort lipid result follows from those checks. The 10 November revision reviewed code and documentation; it did not rerun cohort analyses or simulation campaigns.

Machine readiness does not establish measurement readiness. Source lipid units remain assumed, assay method and blood-draw correspondence need confirmation, and independently repeated preparation/acquisition stages remain undocumented. Keep these limits visible even when `ready_for_cohort_evaluation` is true.

## Runtime and execution budget

The following measurements are carried forward from the dated evidence record, not remeasured here. The recorded environment used Python 3.12 and an NVIDIA GeForce RTX 5080 Laptop GPU.

| Workload | Recorded result | Limit on interpretation |
|---|---|---|
| Real spectra with dummy outcomes; GPU exact fp64; 2,000 bootstrap draws; batch 2048 | 38.5 s; 3.79 GiB peak pool | Bootstrap timing, not a full simulation replicate or campaign. |
| Same check; batch 1024 | 40.2 s; 1.95 GiB | Recorded alternative; no new batch-size setting is required. |
| Same check; batch 512 | 48.7 s; 0.98 GiB | Recorded alternative. |
| Calibration campaign | Not currently timed | The former approximately 21 h estimate predates optimization. Full-grid calibration and multi-scenario coverage cost remain unmeasured. |

Use exact fp64 eigensolvers for reported bootstrap results. Earlier mixed-precision checks changed draws by up to 3e-2 against the sequential procedure. The recorded fast-versus-sequential checks already cover CPU and exact-fp64 GPU; repeat only what changed inputs or code affect.

Before a long campaign, time one intended development replicate in a separate output directory and record grid, participant count, candidate settings, bootstrap count, device, runtime and memory. Obtain approval if even that check is expected to exceed five minutes. Estimate the complete campaign, including scenario evaluations and calibration target scoring, from measured costs. Do not extrapolate the real-data bootstrap timing as the whole campaign cost. Any experiment expected to exceed five minutes requires separate approval.

## 1. Input and measurement readiness

- Verify the versioned standalone [clinical master](../data/clinical_master.csv), [dictionary](../data/clinical_master_dictionary.csv) and [source notes](../data/clinical_master_notes.md).
- Confirm identity review exclusions and the 92/88/4/180 eligibility counts against the recorded input identity. Recompute only if inputs or relevant code changed.
- Confirm lipid units, direct versus calculated LDL-C, statin reference time, collection correspondence and the stages independently repeated. Missing records restrict interpretation; do not infer answers from populated clinical columns.
- Archive grids, covariates, inclusive band windows, aggregation rules, seeds and data fingerprints before endpoint evaluation. Acquisition modes FP and SS are pooled as equivalent; the numeric sample label is an unverified nuisance proxy, not established physical order.
- Do not rebuild the store merely to address deprecated `data/clinical_data.csv`. Strict store validation can flag that old source even though paired-lipid evaluation reads the standalone master. Investigate any other validation failure on its own evidence.

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

**Current implementation gap:** `run_synthetic_scenario_suite` reports prediction/recovery summaries across the configured scenarios, while `evaluate_bootstrap_calibration` evaluates intervals for just one configured signal-strength setting. Increasing `--scenario-reps` does not calibrate intervals for every scenario. The design still requires a frozen core set and at least 1,000 validation datasets per core scenario. That work is pending.

The current validation-seed invocation, after separate runtime approval, is:

```powershell
uv run --group gpu python manuscript_ml/run_simulation_calibration.py --scenario-reps 1000 --calibration-reps 1000 --bootstraps 2000 --participants 92 --use-gpu --seed-set validation --output-dir Results_manuscript_ml/validation_single_setting
```

This command runs the scenario performance suite plus **one-setting coverage validation**, not the entire design's coverage campaign. It inherits `simulation.compact_grid: true` from the checked-in config. A final cohort-grid calibration requires a separately archived config with that field set to `false`; confirm timing and complete scenario support before treating any command as the final campaign.

For each core scenario, report unconditional coverage with failures counted as noncoverage, coverage among successful intervals, failure rate, bias, width, target-estimation error and Monte Carlo uncertainty. The provisional thresholds are Wilson 95% lower bound at least 0.925 and failed dataset intervals below 1%. At least 48 perfect repetitions are needed even to attain that gate; 48 is not a replacement for the planned 1,000 per core scenario.

The target is the mean prediction-risk difference of the replicate outer-fold models on one shared independent test cohort of 5,000 simulated participants. `target_monte_carlo_se` describes variation across training replicates conditional on that cohort; it omits finite-test-cohort uncertainty. The shared estimated target also limits exact binomial interpretation of the Wilson coverage summary. Quantify target-estimation error separately before inferential use. A `gate_is_confirmatory` flag identifies validation seeds only; even a passing provisional flag does not resolve these limitations or missing scenario coverage.

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
| `eligibility_and_split_manifest.json` | Private eligibility and split identity, bootstrap point estimates, interval endpoints and failure status. |
| `nested_held_out_predictions.csv` | Private participant predictions; do not publish identifiers or individual clinical values. |
| `comparator_errors_and_differences.csv` | Report each model, comparison baseline, role, error difference, relative reduction, RMSE, MAE and descriptive calibration. |
| `fold_selected_settings.csv` | Check training-only selection and all selected settings; never select a comparator using outer errors. |
| `baseline_repeatability_report.csv` | Fold-specific paired counts and full-prediction versus spectral-contribution agreement, RMSD, discriminability and retrieval. |
| `subspace_principal_angles.csv` | Dimensions and angles in the shared equal-block, equal-channel metric. |

Keep covariate-only-minus-paired and inner-selected-comparator-minus-paired differences distinct. Bootstrap ranges remain descriptive until calibration requirements are met. Do not pool fold-specific ICCs, interpret agreement as particle identity, or treat random participant splits as external validation.

## 4. Sensitivity and secondary endpoints

After the primary evaluation is authorized and locked:

```powershell
uv run --group gpu python manuscript_ml/run_sensitivity_and_secondary.py --cohort --approve-cohort-evaluation --output-dir Results_manuscript_ml/cohort_sensitivity
```

Report all covariate and subset settings in `covariate_and_subset_sensitivity.csv` and all locked additional partitions in `outer_partition_seed_sensitivity.csv`, without choosing the best partition. The synthetic generator's subset summaries do not independently reproduce real map-level QC selection.

Report `secondary_endpoints_summary.csv` for HDL-C, triglycerides and platelets on the primary population and outer membership, with complete inner selection repeated for each target and missing targets given zero weight. Keep secondary comparisons and `anchor_associations_continuous.csv` descriptive. Formula-linked lipid anchors do not supply independent biological confirmation; clinical platelet count does not measure residual platelet material. Categorical attribution requires a separately frozen and validated rule.

## 5. Manuscript checks and interpretation

- Keep pending fields pending until authorized study artifacts exist. Engineering checks with dummy outcomes cannot fill prediction, repeatability or coverage tables.
- A supported primary improvement requires a positive estimate and an adequately calibrated interval excluding zero. An interval spanning zero is inconclusive.
- A separate methodological claim needs a reproducible advantage over the established comparator selected in inner validation. Similar point estimates do not establish equivalence.
- Describe relative normalized spectral shape under the measured protocol. Do not infer EV specificity, particle concentration, brain origin, causal effects or clinical utility.
- Retain failed analyses and limitations. An unsupported primary result does not authorize switching outcomes or preprocessing to rescue the claim.
- Build the LaTeX draft and check citations, table references, wording and links. Keep participant artifacts private; publication and release require separate approval.


Cohort stages now default to real data and still require `--approve-cohort-evaluation`. Use `--synthetic` only for explicitly labelled development work in a separate output directory. The cohort `run_all.py` workflow runs simulation calibration only with `--include-simulation-calibration`; the dedicated calibration script remains an explicit development tool.
