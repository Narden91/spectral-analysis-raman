# HPC runbook

Track the manuscript ML cluster jobs. Run every command from the repository root. Slurm copies the batch script, so `sbatch` must be started here or `env.sh` is not found. Mark a box only after that job's output directory contains its expected files.

Details and future configuration: [README.md](README.md).

## Setup

- [ ] Export the scheduler settings in this shell

```bash
export MS_ML_PARTITION=small
export MS_ML_ACCOUNT=project_2019570
```

Each script emails `emanuele.nardone@uef.fi` on failure, time limit, and finish.

Heavy jobs request 128 CPUs and 512G. Changing a script does not resize a job that is already running. Cancel it and submit the same script again; `--resume` continues its output directory.

- [ ] Simulation preflight passes

```bash
bash slurm_jobs/preflight.sh --scope simulation --skip-audit --require-scheduler
```

`small` allows the 48-hour jobs. `medium` does not.

## Simulation

Submit all 18 jobs:

```bash
bash slurm_jobs/submit_simulation.sh
```

Or submit one experiment:

```bash
sbatch --partition="$MS_ML_PARTITION" --account="$MS_ML_ACCOUNT" --chdir="$PWD" slurm_jobs/<script>.sbatch
```

Resubmit the same script to resume. Do not change the output directory.

- [ ] 01 `linear_biological_signal` — `sbatch slurm_jobs/02_simulation_01_linear_biological_signal.sbatch`
- [ ] 02 `no_association_reference` — `sbatch slurm_jobs/02_simulation_02_no_association_reference.sbatch`
- [ ] 03 `nonlinear_proxy_linear_covariates` — `sbatch slurm_jobs/02_simulation_03_nonlinear_proxy_linear_covariates.sbatch`
- [ ] 04 `nonlinear_proxy_quadratic_covariates` — `sbatch slurm_jobs/02_simulation_04_nonlinear_proxy_quadratic_covariates.sbatch`
- [ ] 05 `systematic_ab_shift_and_label_drift` — `sbatch slurm_jobs/02_simulation_05_systematic_ab_shift_and_label_drift.sbatch`
- [ ] 06 `formula_linked_friedewald_ldl` — `sbatch slurm_jobs/02_simulation_06_formula_linked_friedewald_ldl.sbatch`
- [ ] 07 `anchor_correlation_none` — `sbatch slurm_jobs/02_simulation_07_anchor_correlation_none.sbatch`
- [ ] 08 `anchor_correlation_high` — `sbatch slurm_jobs/02_simulation_08_anchor_correlation_high.sbatch`
- [ ] 09 `qc_map_dropout` — `sbatch slurm_jobs/02_simulation_09_qc_map_dropout.sbatch`
- [ ] 10 `ablation_no_technical_covariates` — `sbatch slurm_jobs/02_simulation_10_ablation_no_technical_covariates.sbatch`
- [ ] 11 `high_dim_technical_nuisance` — `sbatch slurm_jobs/02_simulation_11_high_dim_technical_nuisance.sbatch`
- [ ] 12 `strong_shared_preparation` — `sbatch slurm_jobs/02_simulation_12_strong_shared_preparation.sbatch`
- [ ] 13 `correlated_ab_sample_errors` — `sbatch slurm_jobs/02_simulation_13_correlated_ab_sample_errors.sbatch`
- [ ] 14 `unequal_ab_sample_errors` — `sbatch slurm_jobs/02_simulation_14_unequal_ab_sample_errors.sbatch`
- [ ] 15 `nonlinear_spectral_signal` — `sbatch slurm_jobs/02_simulation_15_nonlinear_spectral_signal.sbatch`
- [ ] 16 `correlated_within_sample_maps` — `sbatch slurm_jobs/02_simulation_16_correlated_within_sample_maps.sbatch`
- [ ] 17 `covariate_missingness` — `sbatch slurm_jobs/02_simulation_17_covariate_missingness.sbatch`
- [ ] 18 MDE curve — `sbatch slurm_jobs/02_simulation_18_mde_signal_curve.sbatch`

Expected files for 01–17: `Results_manuscript_ml/validation/<scenario>/simulation_scenarios_summary.csv` and `simulation_calibration_report.csv`. For 18: `Results_manuscript_ml/validation/mde_signal_curve/simulation_mde_curve.csv`.

```bash
squeue -u "$USER" -n sim-01-linear_biological_signal,sim-18-mde_signal_curve
```

- [ ] Merge completed scenario reports

```bash
uv run python slurm_jobs/merge_simulation_reports.py
```

Output: `Results_manuscript_ml/validation_merged/`.

## Cohort

Do not start this section until the store is ready and outcome analysis is authorized. The current audit is not ready: the preprocessed store is stale.

- [ ] Rebuild and verify the preprocessed store
- [ ] Cohort preflight reports `ready_for_cohort_evaluation=true`

```bash
bash slurm_jobs/preflight.sh --scope cohort --audit --require-scheduler
```

- [ ] Submit the authorized cohort jobs

```bash
bash slurm_jobs/submit_cohort.sh --approve
```

- [ ] Readiness audit — `sbatch slurm_jobs/01_readiness_audit.sbatch`
- [ ] Primary point estimate — `sbatch slurm_jobs/03_cohort_primary_point_estimate.sbatch`
- [ ] Primary bootstrap interval — `sbatch slurm_jobs/03_cohort_primary_bootstrap_interval.sbatch`
- [ ] Sensitivity and secondary endpoints — `sbatch slurm_jobs/04_cohort_sensitivity_and_secondary_endpoints.sbatch`

Outputs: `Results_manuscript_ml/readiness_review/`, `cohort_primary_point/`, `cohort_primary_bootstrap/`, and `cohort_sensitivity/`.