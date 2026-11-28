# HPC runbook

Checklist for the manuscript ML campaign on Roihu. Run every command from the
repository root on `roihu-cpu`. Tick a box only when the expected files exist.
Background and troubleshooting: [README.md](README.md).

## 1. Setup

- [ ] Export the scheduler settings (or add them to `~/.bashrc`)

  ```bash
  export SBATCH_ACCOUNT=project_2019570 SBATCH_PARTITION=small
  ```

- [ ] Budget is sufficient: `csc-projects` (CPU BU remaining)
- [ ] Preflight passes

  ```bash
  bash slurm_jobs/preflight.sh --scope simulation --skip-audit --require-scheduler
  ```

## 2. Simulation

- [ ] Submit (skips jobs already running or finished; rerun after timeouts)

  ```bash
  bash slurm_jobs/submit_02_simulation.sh --dry-run
  bash slurm_jobs/submit_02_simulation.sh          # all; or: ... 05 11  (selected)
  ```

- [ ] Monitor: `squeue --me` and `tail -f Results_manuscript_ml/slurm_logs/<name>-<jobid>.out`

Expected files: `validation/<scenario>/simulation_scenarios_summary.csv` and
`simulation_calibration_report.csv`; for 18,
`validation/mde_signal_curve/simulation_mde_curve.csv`.

- [ ] 01 `linear_biological_signal`
- [ ] 02 `no_association_reference`
- [ ] 03 `nonlinear_proxy_linear_covariates`
- [ ] 04 `nonlinear_proxy_quadratic_covariates`
- [ ] 05 `systematic_ab_shift_and_label_drift`
- [ ] 06 `formula_linked_friedewald_ldl`
- [ ] 07 `anchor_correlation_none`
- [ ] 08 `anchor_correlation_high`
- [ ] 09 `qc_map_dropout`
- [ ] 10 `ablation_no_technical_covariates`
- [ ] 11 `high_dim_technical_nuisance`
- [ ] 12 `strong_shared_preparation`
- [ ] 13 `correlated_ab_sample_errors`
- [ ] 14 `unequal_ab_sample_errors`
- [ ] 15 `nonlinear_spectral_signal`
- [ ] 16 `correlated_within_sample_maps`
- [ ] 17 `covariate_missingness`
- [ ] 18 MDE curve

- [ ] After the first jobs finish, check cost and efficiency: `seff <jobid>`
  (expect ≥80% CPU efficiency, billed on CPU, ~14 BU per scenario)
- [ ] Merge reports → `Results_manuscript_ml/validation_merged/`

  ```bash
  uv run python slurm_jobs/merge_simulation_reports.py
  ```

## 3. Cohort

Do not start until the store is rebuilt and outcome analysis is authorized.
The current audit is **not ready** (stale preprocessed store).

- [ ] Rebuild and strict-validate the store: `uv run scripts/manage_data_store.py prepare`
- [ ] Cohort preflight reports `ready_for_cohort_evaluation=true`

  ```bash
  bash slurm_jobs/preflight.sh --scope cohort --audit --require-scheduler
  ```

- [ ] Submit (audit first, outcome jobs run `afterok`)

  ```bash
  bash slurm_jobs/submit_cohort.sh --approve --dry-run
  bash slurm_jobs/submit_cohort.sh --approve
  ```

  Per stage instead: `submit_01_readiness.sh`, then
  `submit_03_cohort_primary.sh --approve` and
  `submit_04_cohort_sensitivity.sh --approve`.

- [ ] Readiness audit → `readiness_review/`
- [ ] Primary point estimate → `cohort_primary_point/`
- [ ] Primary bootstrap interval → `cohort_primary_bootstrap/`
- [ ] Sensitivity and secondary endpoints → `cohort_sensitivity/`
