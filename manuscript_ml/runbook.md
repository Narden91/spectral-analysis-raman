# Manuscript ML Experiment Runbook

Operational checklist for executing the study design specified in [docs/research_design.md](../docs/research_design.md).
All mathematical and statistical routines are implemented in the [src/analysis/paired_lipid/](../src/analysis/paired_lipid/) package (`cohort`, `models`, `batched`, `nested_cv`, `simulation`) and run by the CLI scripts in `manuscript_ml/`.
All experiment settings (grids, seeds, readiness expectation, synthetic scenarios) live in [config/manuscript_ml.yaml](../config/manuscript_ml.yaml); every CLI accepts `--config` and writes `run_provenance_<stage>.json` next to its outputs.

---

## Measured Runtime Summary & Terminal Commands

Local workspace: Python 3.12, NVIDIA GeForce RTX 5080 Laptop GPU via `cupy-cuda13x`, 16-core CPU.
Measured on 8 November 2026 (n = 92 synthetic, compact grid, 200 bootstrap draws): GPU exact fp64 7.6 s, CPU 12.7 s, GPU mixed precision 1.9 s.
The bootstrap uses exact fp64 eigensolvers by default. Mixed precision (`exact_fp64_eigh=False`) changed 200/200 draws by up to 3e-2 against the sequential procedure, so it is not used for reported intervals.
Rows marked *estimate* are extrapolated, not measured.

| Phase / Step | Mode | Estimated Wall-Clock Time | Terminal Command |
|---|---|---:|---|
| **Phase 0: Unit & Synthetic Test Suite** | Verification (12 tests incl. batched-vs-sequential on CPU and GPU) | **~60 s** (measured) | `uv run --group gpu pytest tests/test_paired_lipid_representation.py -p no:cacheprovider --basetemp=.pytest_tmp_runbook` |
| **Phase 0: Store Validation** | Preprocessed store check | **~2–5 s** | `uv run python scripts/manage_data_store.py validate --source preprocessed --strict` |
| **Phase 1: Readiness & Budget Audit** | Clinical + GPU benchmark | **~4–6 s** | `uv run --group gpu python manuscript_ml/audit_readiness.py --output-dir Results_manuscript_ml` |
| **Phase 2A: Synthetic Smoke Suite** | Quick (`--quick`, compact grid) | **~3–5 s** | `uv run --group gpu python manuscript_ml/run_simulation_calibration.py --quick --output-dir Results_manuscript_ml/synthetic_smoke` |
| **Phase 2B: Standard Synthetic Suite** | Default (10 scenarios $\times$ 3 participant counts $\times 10$ reps + $20 \times 100$ cal, development seeds) | not measured | `uv run --group gpu python manuscript_ml/run_simulation_calibration.py --output-dir Results_manuscript_ml/simulation_standard` |
| **Phase 2C: Full Calibration Campaign** | 1,000 cal reps $\times$ 2,000 bootstraps, **validation seeds** | *estimate* ~21 h calibration (1,000 $\times$ ~76 s exact-fp64 GPU) plus scenario suite | `uv run --group gpu python manuscript_ml/run_simulation_calibration.py --scenario-reps 1000 --calibration-reps 1000 --bootstraps 2000 --participants 92 --use-gpu --seed-set validation --output-dir Results_manuscript_ml/simulation_full` |
| **Phase 3A: Synthetic Primary Dry Run** | Compact grid + 20 bootstraps | **~2–3 s** | `uv run --group gpu python manuscript_ml/run_primary_evaluation.py --bootstraps 20 --compact-grid --output-dir Results_manuscript_ml/synthetic_primary` |
| **Phase 3B: Cohort Primary Point Estimate** | Real cohort, 0 bootstraps | **~1–2 s** | `uv run --group gpu python manuscript_ml/run_primary_evaluation.py --cohort --approve-cohort-evaluation --bootstraps 0 --output-dir Results_manuscript_ml/cohort_primary` |
| **Phase 3C: Cohort Primary + 2,000 Bootstraps** | Real cohort, 2,000 refits on RTX 5080 | *estimate* ~1.5 min (compact grid); full grid not measured | `uv run --group gpu python manuscript_ml/run_primary_evaluation.py --cohort --approve-cohort-evaluation --bootstraps 2000 --output-dir Results_manuscript_ml/cohort_primary` |
| **Phase 4A: Synthetic Sensitivity Dry Run** | Compact grid (15 nested evaluations) | **~2–4 s** | `uv run --group gpu python manuscript_ml/run_sensitivity_and_secondary.py --compact-grid --output-dir Results_manuscript_ml/synthetic_sensitivity` |
| **Phase 4B: Cohort Sensitivity & Secondary** | Real cohort, full grid (15 nested evaluations) | **~4–8 s** | `uv run --group gpu python manuscript_ml/run_sensitivity_and_secondary.py --cohort --approve-cohort-evaluation --output-dir Results_manuscript_ml/cohort_sensitivity` |
| **All-in-One Quick Smoke Pipeline** | Phases 1–4 synthetic quick | **~8–12 s** | `uv run --group gpu python manuscript_ml/run_all.py --quick --bootstraps 4 --output-dir Results_manuscript_ml/quick_all` |

---

## Phase 0 — Pre-Flight & Environment Verification

- [ ] Confirm the workspace virtual environment and test suite pass (**~45–75 s**):
  ```powershell
  uv run pytest tests/test_paired_lipid_representation.py -p no:cacheprovider --basetemp=.pytest_tmp_runbook
  Remove-Item -Recurse -Force .pytest_tmp_runbook -ErrorAction SilentlyContinue
  ```
- [ ] Verify that standalone clinical master artifacts are current ([data/clinical_master.csv](../data/clinical_master.csv), [data/clinical_master_dictionary.csv](../data/clinical_master_dictionary.csv), and [data/clinical_master_notes.md](../data/clinical_master_notes.md)).
- [ ] Confirm that the canonical preprocessed spectral store (`data/preprocessed/manifest.json`) is materialized and validated before any cohort run (**~2–5 s**):
  ```powershell
  uv run python scripts/manage_data_store.py validate --source preprocessed --strict
  ```

---

## Phase 1 — Readiness Audit & Computational Budget Gate (~4–6 s)

Run [manuscript_ml/audit_readiness.py](audit_readiness.py) to verify linkage counts, lock hyperparameter grids, generate the source-linked limitations table, and benchmark single-refit runtime and memory:

```powershell
uv run python manuscript_ml/audit_readiness.py --output-dir Results_manuscript_ml
```

- [ ] **Measurement readiness**: Confirm in `Results_manuscript_ml/readiness_manifest.json` that `stroke_with_spectra == 97`, `stroke_identity_verified == 92` (5 excluded for `SEX_FOLDER_CONFLICT` / identity review), and laboratory units/assay definitions (`ldl_mmol_l`, `hdl_mmol_l`, `triglycerides_mmol_l`, `plt_10e9_l`) are documented.
- [ ] **MISEV2023 repeat hierarchy**: Document in the study manifest which preparation and acquisition stages (isolation, deposition, substrate, session) were independently repeated across technical sample A and sample B.
- [ ] **Input & comparator lock**: Confirm that `pca_dimensions = [5, 10, 15]`, `retained_directions = [1, 2, 3]`, `lambda_grid = [0.001, 0.01, 0.1, 1.0, 10.0]`, `ridge_alpha_grid = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]`, `pls_component_grid = [1, 2, 3, 4, 5]`, `primary_seed = 42`, and `sensitivity_seeds = [101, 202, 303]` are archived in `readiness_manifest.json`.
- [ ] **Computational readiness**: Inspect `estimated_2000_bootstrap_minutes` in `readiness_manifest.json`. Obtain explicit approval before launching any campaign expected to exceed 5 minutes.
- [ ] **Linkage expectation**: Confirm `identity_verified_matches_expected == true` (expected count set in `config/manuscript_ml.yaml`).
- [ ] **Limitations table**: Inspect `Results_manuscript_ml/limitations_table.csv` to confirm that retrospective adaptation, unconfirmed assay/fasting timing, MISEV2023 co-isolate constraints, and absence of control lipid values are recorded.

---

## Phase 2 — Known-Factor Synthetic Simulations & Uncertainty Calibration

Run [manuscript_ml/run_simulation_calibration.py](run_simulation_calibration.py) to evaluate method recovery, failure modes, and bootstrap interval coverage on synthetic data before cohort evaluation.

### 2A. Fast Verification Smoke Run (~8–12 s)
- [ ] Execute the quick synthetic verification suite:
  ```powershell
  uv run python manuscript_ml/run_simulation_calibration.py --quick --output-dir Results_manuscript_ml/synthetic_smoke
  ```
- [ ] Verify in `simulation_scenarios_summary.csv` that:
  - [ ] `linear_biological_signal` yields positive `mean_delta_mse_paired` and high spectral `ICC(A,1)`.
  - [ ] `no_association_reference` yields `mean_delta_mse_paired` near zero.
  - [ ] `nonlinear_proxy_linear_covariates` vs `nonlinear_proxy_quadratic_covariates` demonstrates that adding quadratic age terms removes artificial spectral proxy gain when spectra proxy $\text{age}^2$.

### 2B. Standard Development Calibration Run (development seeds; not confirmatory)
- [ ] Execute the default development simulation and calibration check (10 scenario repetitions + 20 calibration datasets $\times$ 100 bootstraps):
  ```powershell
  uv run --group gpu python manuscript_ml/run_simulation_calibration.py --use-gpu --output-dir Results_manuscript_ml/simulation_standard
  ```

### 2C. Full Calibration Campaign (estimate ~21 h on RTX 5080; ask for approval first)
- [ ] Execute the confirmatory calibration on validation seeds (disjoint from the development seeds used in 2A/2B):
  ```powershell
  uv run --group gpu python manuscript_ml/run_simulation_calibration.py --scenario-reps 1000 --calibration-reps 1000 --bootstraps 2000 --participants 92 --use-gpu --seed-set validation --output-dir Results_manuscript_ml/simulation_full
  ```
- [ ] Coverage target: `target_delta_mse` is the replicate mean of each replicate's outer-fold models scored on one independent synthetic test cohort (n = 5,000); check `target_monte_carlo_se` is small against `mean_interval_width`.
- [ ] **Uncertainty readiness gate**: Check `simulation_calibration_report.csv` (`gate_is_confirmatory` must be `True`):
  - [ ] Verify whether `passes_provisional_gate == True` (`wilson_95_lower >= 0.925` for nominal 95% unconditional coverage and `dataset_failure_rate < 0.01`).
  - [ ] If calibration passes, bootstrap intervals may be reported inferentially for the primary endpoint.
  - [ ] If calibration fails, either revise and re-validate the interval procedure on synthetic development data before cohort evaluation, or restrict cohort reporting to point estimates and descriptive resampling dispersion.

---

## Phase 3 — Primary Held-Out LDL-C Evaluation & Repeatability Report

Run [manuscript_ml/run_primary_evaluation.py](run_primary_evaluation.py). By default, the script runs on synthetic data; evaluating on the stroke cohort requires passing `--cohort --approve-cohort-evaluation` after Phases 1–2 are checked.

### 3A. Synthetic End-to-End Dry Run (~4–6 s)
- [ ] Verify all primary output schemas on synthetic data:
  ```powershell
  uv run python manuscript_ml/run_primary_evaluation.py --bootstraps 20 --compact-grid --output-dir Results_manuscript_ml/synthetic_primary
  ```

### 3B. Locked Stroke Cohort Point-Estimate & Repeatability Evaluation (~2–4 s)
- [ ] Run the locked $5 \times 3$ nested evaluation on the verified stroke cohort without bootstrap refits:
  ```powershell
  uv run python manuscript_ml/run_primary_evaluation.py --cohort --approve-cohort-evaluation --bootstraps 0 --output-dir Results_manuscript_ml/cohort_primary
  ```

### 3C. Locked Stroke Cohort Full 2,000-Bootstrap Evaluation (~8–12 s on RTX 5080)
- [ ] Run the locked $5 \times 3$ nested evaluation with 2,000 multiplicity bootstrap refits:
  ```powershell
  uv run --group gpu python manuscript_ml/run_primary_evaluation.py --cohort --approve-cohort-evaluation --bootstraps 2000 --output-dir Results_manuscript_ml/cohort_primary
  ```
- [ ] Verify generated artifacts in `Results_manuscript_ml/cohort_primary/`:
  - [ ] `eligibility_and_split_manifest.json`: Verify eligible participant count, paired vs single-sample counts, and sample exclusions (kept private; never publish participant IDs).
  - [ ] `comparator_errors_and_differences.csv`: Inspect $\widehat\Delta$, relative MSE reduction $\widehat\Delta/\mathrm{MSE}_C$, RMSE, MAE, descriptive calibration intercept/slope for `paired_linear_ridge`, `fixed_band_ridge`, `pca_ridge`, `pls_regression`, `band_paired_ridge`, and `paired_linear_ridge_vs_strongest_comparator`.
  - [ ] `baseline_repeatability_report.csv`: Compare full-prediction $\mathrm{ICC}(A,1)$ and RMSD against isolated spectral-contribution ($s_{is} = \widehat\beta_Z^\top z_{is}$) $\mathrm{ICC}(A,1)$ and RMSD within each outer fold, alongside Bridgeford discriminability and bidirectional ($A \leftrightarrow B$) retrieval accuracy.
  - [ ] `subspace_principal_angles.csv`: Inspect pairwise principal angles (in degrees) between outer-fold learned subspaces mapped to the canonical equal-block, equal-channel metric.

---

## Phase 4 — Sensitivity Analyses, Secondary Endpoints & Exploratory Attribution

Run [manuscript_ml/run_sensitivity_and_secondary.py](run_sensitivity_and_secondary.py) to execute all prespecified covariate, subset, outer-seed, and secondary-anchor evaluations.

- [ ] Execute sensitivity and secondary analyses on synthetic data (**~4–6 s**):
  ```powershell
  uv run python manuscript_ml/run_sensitivity_and_secondary.py --compact-grid --output-dir Results_manuscript_ml/synthetic_sensitivity
  ```
- [ ] Execute sensitivity and secondary analyses on the verified stroke cohort (**~30–40 s**):
  ```powershell
  uv run python manuscript_ml/run_sensitivity_and_secondary.py --cohort --approve-cohort-evaluation --output-dir Results_manuscript_ml/cohort_sensitivity
  ```
- [ ] Inspect all four deliverables:
  - [ ] `covariate_and_subset_sensitivity.csv`: Compare `primary`, `no_label_trend`, `quadratic_label_trend`, `quadratic_age_and_label`, and `pre_event_risk_factors`, plus `all_eligible`, `paired_only`, `single_sample`, and `three_map_complete` (3 maps per block).
  - [ ] `outer_partition_seed_sensitivity.csv`: Inspect dispersion across the primary seed (`42`) and the three sensitivity seeds (`101`, `202`, `303`). Never select the best seed as the primary result.
  - [ ] `secondary_endpoints_summary.csv`: Inspect held-out prediction results for `hdl_mmol_l`, `triglycerides_mmol_l`, and `plt_10e9_l` (each tuned with complete inner selection on its own target).
  - [ ] `anchor_associations_continuous.csv`: Inspect continuous partial Pearson and Spearman associations and resample sign stability across anchors. Keep attribution descriptive (`continuous_exploratory_summary_only`) unless a separate categorical decision rule is frozen and validated.

---

## Phase 5 — Interpretation & Reporting Guardrails

- [ ] **Primary claim check**: Report a supported predictive improvement only if $\widehat\Delta > 0$ and a calibrated 95% interval excludes zero. If the interval spans zero, report the finding as inconclusive (not evidence of equivalence or absence).
- [ ] **Methodological claim check**: Claim a methodological advantage for `paired_linear_ridge` only if it reproducibly outperforms the strongest established comparator selected inside inner validation (`paired_linear_ridge_vs_strongest_comparator`). If a simpler comparator (`fixed_band_ridge`, `pca_ridge`, or `pls_regression`) performs comparably, retain the simpler method.
- [ ] **Biological boundary check**: Ensure manuscript prose states that held-out LDL-C prediction from SNV/area-normalized EV-enriched plasma Raman spectra reflects relative spectral shape under the observed protocol, and does **not** establish EV-specific particle identity, lipoprotein purity, brain origin, or diagnostic utility.
- [ ] **Privacy check**: Keep `eligibility_and_split_manifest.json` and `nested_held_out_predictions.csv` local and private; export only aggregate summaries and synthetic fixtures for public release.
