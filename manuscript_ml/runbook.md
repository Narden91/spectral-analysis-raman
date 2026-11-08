# Manuscript ML Experiment Runbook

Operational checklist for executing the study design specified in [docs/research_design.md](../docs/research_design.md).
All mathematical and statistical routines are implemented in [src/analysis/paired_lipid_representation.py](../src/analysis/paired_lipid_representation.py) and orchestrated via the CLI scripts in `manuscript_ml/`.

---

## Phase 0 — Pre-Flight & Environment Verification

- [ ] Confirm the workspace virtual environment and test suite pass:
  ```powershell
  uv run pytest tests/test_paired_lipid_representation.py -p no:cacheprovider --basetemp=.pytest_tmp_runbook
  Remove-Item -Recurse -Force .pytest_tmp_runbook -ErrorAction SilentlyContinue
  ```
- [ ] Verify that standalone clinical master artifacts are current ([data/clinical_master.csv](../data/clinical_master.csv), [data/clinical_master_dictionary.csv](../data/clinical_master_dictionary.csv), and [data/clinical_master_notes.md](../data/clinical_master_notes.md)).
- [ ] Confirm that the canonical preprocessed spectral store (`data/preprocessed/manifest.json`) is materialized and validated before any cohort run:
  ```powershell
  uv run python scripts/manage_data_store.py validate --source preprocessed --strict
  ```

---

## Phase 1 — Readiness Audit & Computational Budget Gate

Run [manuscript_ml/audit_readiness.py](audit_readiness.py) to verify linkage counts, lock hyperparameter grids, generate the source-linked limitations table, and benchmark single-refit runtime and memory:

```powershell
uv run python manuscript_ml/audit_readiness.py --output-dir Results_manuscript_ml
```

- [ ] **Measurement readiness**: Confirm in `Results_manuscript_ml/readiness_manifest.json` that `stroke_with_spectra == 97`, `stroke_identity_verified == 92` (5 excluded for `SEX_FOLDER_CONFLICT` / identity review), and laboratory units/assay definitions (`ldl_mmol_l`, `hdl_mmol_l`, `triglycerides_mmol_l`, `plt_10e9_l`) are documented.
- [ ] **MISEV2023 repeat hierarchy**: Document in the study manifest which preparation and acquisition stages (isolation, deposition, substrate, session) were independently repeated across technical sample A and sample B.
- [ ] **Input & comparator lock**: Confirm that `pca_dimensions = [5, 10, 15]`, `retained_directions = [1, 2, 3]`, `lambda_grid = [0.001, 0.01, 0.1, 1.0, 10.0]`, `ridge_alpha_grid = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]`, `pls_component_grid = [1, 2, 3, 4, 5]`, `primary_seed = 42`, and `sensitivity_seeds = [101, 202, 303]` are archived in `readiness_manifest.json`.
- [ ] **Computational readiness**: Inspect `estimated_2000_bootstrap_minutes` in `readiness_manifest.json`. Obtain explicit approval before launching any campaign expected to exceed 5 minutes.
- [ ] **Limitations table**: Inspect `Results_manuscript_ml/limitations_table.csv` to confirm that retrospective adaptation, unconfirmed assay/fasting timing, MISEV2023 co-isolate constraints, and absence of control lipid values are recorded.

---

## Phase 2 — Known-Factor Synthetic Simulations & Uncertainty Calibration

Run [manuscript_ml/run_simulation_calibration.py](run_simulation_calibration.py) to evaluate method recovery, failure modes, and bootstrap interval coverage on synthetic data before cohort evaluation.

### 2A. Fast Verification Smoke Run
- [ ] Execute the quick synthetic verification suite:
  ```powershell
  uv run python manuscript_ml/run_simulation_calibration.py --quick --output-dir Results_manuscript_ml/synthetic_smoke
  ```
- [ ] Verify in `simulation_scenarios_summary.csv` that:
  - [ ] `linear_biological_signal` yields positive `mean_delta_mse_paired` and high spectral `ICC(A,1)`.
  - [ ] `no_association_reference` yields `mean_delta_mse_paired` near zero.
  - [ ] `nonlinear_proxy_linear_covariates` vs `nonlinear_proxy_quadratic_covariates` demonstrates that adding quadratic age terms removes artificial spectral proxy gain when spectra proxy $\text{age}^2$.

### 2B. Full Calibration Campaign (Requires Campaign Approval if > 5 Minutes)
- [ ] Execute the full simulation calibration campaign (planned $\ge 1,000$ repetitions per core scenario):
  ```powershell
  uv run python manuscript_ml/run_simulation_calibration.py --scenario-reps 1000 --calibration-reps 1000 --bootstraps 2000 --participants 92 --output-dir Results_manuscript_ml/simulation_full
  ```
- [ ] **Uncertainty readiness gate**: Check `simulation_calibration_report.csv`:
  - [ ] Verify whether `passes_provisional_gate == True` (`wilson_95_lower >= 0.925` for nominal 95% unconditional coverage and `dataset_failure_rate < 0.01`).
  - [ ] If calibration passes, bootstrap intervals may be reported inferentially for the primary endpoint.
  - [ ] If calibration fails, either revise and re-validate the interval procedure on synthetic development data before cohort evaluation, or restrict cohort reporting to point estimates and descriptive resampling dispersion.

---

## Phase 3 — Primary Held-Out LDL-C Evaluation & Repeatability Report

Run [manuscript_ml/run_primary_evaluation.py](run_primary_evaluation.py). By default, the script runs on synthetic data; evaluating on the stroke cohort requires passing `--cohort --approve-cohort-evaluation` after Phases 1–2 are checked.

### 3A. Synthetic End-to-End Dry Run
- [ ] Verify all primary output schemas on synthetic data:
  ```powershell
  uv run python manuscript_ml/run_primary_evaluation.py --bootstraps 20 --compact-grid --output-dir Results_manuscript_ml/synthetic_primary
  ```

### 3B. Locked Stroke Cohort Primary Evaluation
- [ ] Run the locked $5 \times 3$ nested evaluation on the verified stroke cohort:
  ```powershell
  uv run python manuscript_ml/run_primary_evaluation.py --cohort --approve-cohort-evaluation --bootstraps 2000 --output-dir Results_manuscript_ml/cohort_primary
  ```
- [ ] Verify generated artifacts in `Results_manuscript_ml/cohort_primary/`:
  - [ ] `eligibility_and_split_manifest.json`: Verify eligible participant count, paired vs single-sample counts, and sample exclusions (kept private; never publish participant IDs).
  - [ ] `comparator_errors_and_differences.csv`: Inspect $\widehat\Delta$, relative MSE reduction $\widehat\Delta/\mathrm{MSE}_C$, RMSE, MAE, descriptive calibration intercept/slope for `paired_linear_ridge`, `fixed_band_ridge`, `pca_ridge`, `pls_regression`, `band_paired_ridge`, and `paired_linear_ridge_vs_strongest_comparator`.
  - [ ] `baseline_repeatability_report.csv`: Compare full-prediction $\mathrm{ICC}(A,1)$ and RMSD against isolated spectral-contribution ($s_{is} = \widehat\beta_Z^\top z_{is}$) $\mathrm{ICC}(A,1)$ and RMSD within each outer fold, alongside Bridgeford discriminability and bidirectional ($A \leftrightarrow B$) retrieval accuracy.
  - [ ] `subspace_principal_angles.csv`: Inspect pairwise principal angles (in degrees) between outer-fold learned subspaces mapped to the canonical equal-block, equal-channel metric.

---

## Phase 4 — Sensitivity Analyses, Secondary Endpoints & Exploratory Attribution

Run [manuscript_ml/run_sensitivity_and_secondary.py](run_sensitivity_and_secondary.py) to execute all prespecified covariate, subset, outer-seed, and secondary-anchor evaluations.

- [ ] Execute sensitivity and secondary analyses on synthetic data (dry run):
  ```powershell
  uv run python manuscript_ml/run_sensitivity_and_secondary.py --compact-grid --output-dir Results_manuscript_ml/synthetic_sensitivity
  ```
- [ ] Execute sensitivity and secondary analyses on the verified stroke cohort:
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

