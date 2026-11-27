# Manuscript ML Slurm jobs

Command checklist: [runbook_hpc.md](runbook_hpc.md).

One script per experiment. Do not submit `manuscript_ml/run_all.py`. These jobs encode the locked 1,000-replicate validation settings in [manuscript_ml/runbook.md](../manuscript_ml/runbook.md). Creating or submitting the scripts does not authorize cohort outcome analysis.

Submit from the repository root. Slurm runs a copy of the batch script, so the jobs find `env.sh` through `SLURM_SUBMIT_DIR`. Submitting from another directory makes that lookup fail.

## Configure

Set these in the same shell before any submit command. `--require-scheduler` fails if either is empty. That was the `MS_ML_PARTITION is empty` failure.

```bash
export MS_ML_PARTITION=small
export MS_ML_ACCOUNT=project_2019570
```

Optional:

| Variable | Default | Use |
|---|---|---|
| `MS_ML_MAIL_USER` | `emanuele.nardone@uef.fi` in each script | Override only if mail should go elsewhere |
| `MS_ML_UV` | `uv` | Alternate `uv` executable |
| `MS_ML_UV_EXTRAS` | empty | Leave empty. Jobs pass `--no-gpu` |
| `MS_ML_CONFIG` | `config/manuscript_ml.yaml` | Do not point at a different science config for reported runs |
| `MS_ML_RESULTS_ROOT` | `Results_manuscript_ml` | Only for a scratch redirect |

`MS_ML_UV_EXTRAS="--group gpu"` installs CuPy and is slower for calibration. Do not set it on a CPU partition.

On this cluster, `small` is the correct partition: one node, up to 3 days, and it accepts the 16-CPU / 64G / 48-hour requests. `medium` rejects them because its limit is 36 hours (`1-12:00:00`). Use `longrun` only if a resubmit is not enough; it requires the `longrun` QoS.

## Check

```bash
bash slurm_jobs/preflight.sh --scope simulation --skip-audit --require-scheduler
```

This checks the script inventory against `config/manuscript_ml.yaml`, the runners, `uv`, clinical files, and the preprocessed store. It does not fit models and does not submit jobs.

| Flag | Meaning |
|---|---|
| `--scope simulation` | Synthetic jobs only. Skips the cohort audit unless `--audit` is also passed |
| `--scope cohort` | Cohort files and, with `--audit`, `ready_for_cohort_evaluation` |
| `--skip-audit` | Do not load the real cohort |
| `--require-scheduler` | Fail if `sbatch`, `MS_ML_PARTITION`, or `MS_ML_ACCOUNT` is missing |

A passing simulation preflight does not mean the cohort store is ready.

## Submit simulation

```bash
bash slurm_jobs/submit_simulation.sh
```

This submits 18 jobs: scenarios 01–17 plus the MDE curve. It does not submit readiness or cohort jobs.

Each scenario writes only to `Results_manuscript_ml/validation/<scenario>/`. The MDE job writes to `Results_manuscript_ml/validation/mde_signal_curve/`. A shared directory would overwrite `simulation_calibration_report.csv`.

Logs: `Results_manuscript_ml/slurm_logs/<job-name>-<jobid>.out`.

If a job hits the wall time, submit the same script again. `--resume` continues from that directory's checkpoints. Do not change the output directory between attempts.

After the scenario jobs finish:

```bash
uv run python slurm_jobs/merge_simulation_reports.py
```

Merged tables go to `Results_manuscript_ml/validation_merged/`. The MDE directory's single-setting calibration report is not included. Add `--allow-partial` only to inspect an incomplete campaign.

## Submit cohort

Do this only after the readiness gates in the runbook are accepted.

```bash
bash slurm_jobs/preflight.sh --scope cohort --audit --require-scheduler
bash slurm_jobs/submit_cohort.sh --approve
```

`--approve` is required. The helper submits the readiness audit, then the three outcome jobs with `afterok` on that audit. Outcome commands still pass `--approve-cohort-evaluation`.

| Script | Output |
|---|---|
| `01_readiness_audit.sbatch` | `Results_manuscript_ml/readiness_review/` |
| `03_cohort_primary_point_estimate.sbatch` | `Results_manuscript_ml/cohort_primary_point/` |
| `03_cohort_primary_bootstrap_interval.sbatch` | `Results_manuscript_ml/cohort_primary_bootstrap/` |
| `04_cohort_sensitivity_and_secondary_endpoints.sbatch` | `Results_manuscript_ml/cohort_sensitivity/` |

The current checkout is not cohort-ready. The audit reports a stale preprocessed store: generated artifact signature mismatches, and the store is stale relative to the clinical sources and its parsed source. Rebuild and verify the store before cohort submission. Simulation jobs do not use that gate.

## Scripts

Names follow pipeline order. The two-digit index is the config order in `simulation.scenarios`; that order is also the canonical calibration seed offset.

| Script | What it runs |
|---|---|
| `01_readiness_audit.sbatch` | `audit_readiness.py` |
| `02_simulation_01_…` through `02_simulation_17_…` | One configured scenario, suite plus calibration |
| `02_simulation_18_mde_signal_curve.sbatch` | MDE curve only |
| `03_cohort_primary_point_estimate.sbatch` | Primary point estimate, 0 bootstraps |
| `03_cohort_primary_bootstrap_interval.sbatch` | Primary bootstrap, 2,000 draws |
| `04_cohort_sensitivity_and_secondary_endpoints.sbatch` | All presets, subsets, seeds, and secondary endpoints |

Locked simulation flags live in [env.sh](env.sh), not copied into every script: 1,000 scenario reps, 1,000 calibration reps, 2,000 bootstraps, 92 participants, 25,000 test participants, `studentized_t`, cohort grid, validation seeds, `--no-gpu`, `--resume`. Scenario jobs also pass `--calibrate-all-scenarios` with one `--scenarios` name so the seed offset stays canonical. The MDE job does not pass `--calibrate-all-scenarios`.

Resources in the `.sbatch` files:

| Jobs | CPUs | Memory | Wall time |
|---|---|---|---|
| Readiness | 2 | 16G | 30 minutes |
| Primary point estimate | 8 | 32G | 4 hours |
| Simulation, MDE, primary bootstrap, sensitivity | 16 | 64G | 48 hours, except primary bootstrap at 12 hours |

`n_jobs` is `$SLURM_CPUS_PER_TASK`. BLAS threads are pinned to 1 so workers do not oversubscribe the node.

## Change the configuration

- New or renamed scenario: edit `config/manuscript_ml.yaml`, add or rename the matching `02_simulation_NN_<scenario>.sbatch`, and keep the index equal to the config position. Then run `uv run python slurm_jobs/check_job_inventory.py`. Reordering scenarios changes seed offsets; do not renumber without treating that as a new campaign.
- Partition, account, or mail: export the variables above. Do not hard-code them into the `.sbatch` files.
- Wall time, CPUs, or memory: edit the `#SBATCH` lines in the affected scripts. Keep simulation requests at or below the partition limit.
- Science settings: change only the locked block in `env.sh`, then rerun the inventory check. Do not give one scenario a different budget.
- Another cluster: export `MS_ML_PARTITION` and `MS_ML_ACCOUNT` for that site. Confirm its maximum wall time before submitting the 48-hour scripts.

Direct `sbatch slurm_jobs/<script>.sbatch` from the repository root also works, but it uses the site default partition and account unless you pass `--partition` and `--account` yourself. Prefer the submit helpers.