# Manuscript ML Slurm jobs

One `.sbatch` script per experiment for the Roihu cluster (CSC). The jobs
encode the locked 1,000-replicate validation settings from
[manuscript_ml/runbook.md](../manuscript_ml/runbook.md). Do not submit
`manuscript_ml/run_all.py`.

- Step-by-step checklist: [runbook_hpc.md](runbook_hpc.md)
- General Roihu commands (queue, budget, storage): [roihu_cheatsheet.md](roihu_cheatsheet.md)

Creating or submitting these scripts does **not** authorize cohort outcome
analysis.

## Quick start (simulation)

Run everything from the repository root on `roihu-cpu`.

```bash
export SBATCH_ACCOUNT=project_2019570 SBATCH_PARTITION=small
bash slurm_jobs/submit_02_simulation.sh --dry-run  # preview
bash slurm_jobs/submit_02_simulation.sh            # scenarios 01-17 + MDE curve
# ...after all jobs finish:
uv run python slurm_jobs/merge_simulation_reports.py
```

## Launchers: one per stage number

| Stage | Launcher | Submits |
|---|---|---|
| 01 | `submit_01_readiness.sh` | `01_readiness_audit` (always reruns) |
| 02 | `submit_02_simulation.sh [NN ...]` | all `02_simulation_*`, or only the given scenario numbers |
| 03 | `submit_03_cohort_primary.sh --approve` | both `03_cohort_primary_*` |
| 04 | `submit_04_cohort_sensitivity.sh --approve` | `04_cohort_sensitivity_and_secondary_endpoints` |
| 01→03+04 | `submit_cohort.sh --approve` | audit, then 03 and 04 `afterok` on it |

Each launcher runs the matching preflight first, then:

- skips any job whose name is already queued or running, so two jobs never
  share a checkpoint;
- skips jobs whose output already holds `run_provenance_*.json` (written
  last by every runner); `--force` reruns them;
- `--dry-run` prints what would be submitted and submits nothing.

Cohort launchers 03/04 wait (`afterok`) for a readiness job that is still
queued or running. Shared logic lives in [submit_lib.sh](submit_lib.sh).

`SBATCH_ACCOUNT` and `SBATCH_PARTITION` are read by plain `sbatch` and by the
helpers here (`MS_ML_ACCOUNT`/`MS_ML_PARTITION` override them). Without an
account, Roihu rejects the job with `AssocMaxSubmitJobLimit`.

Why the repository root: Slurm runs a spool copy of the script, so each job
finds `env.sh` through `$SLURM_SUBMIT_DIR/slurm_jobs/`, and the relative log
path `Results_manuscript_ml/slurm_logs/` must already exist. The helpers
create it; a bare `sbatch` from a fresh clone does not.

## Jobs and outputs

The two-digit simulation index is the scenario's position in
`simulation.scenarios` and doubles as its calibration seed offset.

| Script | Runs | Output under `Results_manuscript_ml/` |
|---|---|---|
| `01_readiness_audit` | `audit_readiness.py` | `readiness_review/` |
| `02_simulation_01…17_<scenario>` | One scenario: suite + calibration | `validation/<scenario>/` |
| `02_simulation_18_mde_signal_curve` | MDE curve only | `validation/mde_signal_curve/` |
| `03_cohort_primary_point_estimate` | Primary estimate, 0 bootstraps | `cohort_primary_point/` |
| `03_cohort_primary_bootstrap_interval` | Primary bootstrap, 2,000 draws | `cohort_primary_bootstrap/` |
| `04_cohort_sensitivity_and_secondary_endpoints` | Presets, subsets, seeds, secondary endpoints | `cohort_sensitivity/` |

Logs: `Results_manuscript_ml/slurm_logs/<job-name>-<jobid>.{out,err}`.

Each scenario has its own directory because the runner always writes the same
file names. The merge script combines them into `validation_merged/`; it
skips the MDE directory's single-setting calibration report. Use
`--allow-partial` only to inspect an unfinished campaign.

The locked flags (replicate counts, bootstraps, participants, interval method,
seed set, `--no-gpu`, `--resume`) live once in [env.sh](env.sh), not in each
script.

## Resources and billing

| Jobs | CPUs | Memory | Wall time |
|---|---|---|---|
| Readiness audit | 2 | 16G | 0:30 |
| Simulations 01–17 | 32 | 32G | 4:00 |
| Simulation 18 (MDE, 5 × 1,000 reps) | 32 | 32G | 8:00 |
| Primary point estimate | 16 | 30G | 4:00 |
| Primary bootstrap | 128 | 240G | 12:00 |
| Sensitivity | 128 | 240G | 48:00 |

On `small`, Roihu bills **max(0.75 BU × cores, 0.375 BU × GiB) per hour** of
elapsed time on what is *allocated*, used or not. Keep memory ≤ 2 GiB/core.

**Why 32 cores for simulations.** Measured on scenario 01: one calibration
rep is single-threaded and takes ~57 CPU-seconds, so a scenario is ~16
core-hours of work, and each worker peaks at ~170 MB. Reps run in
checkpoint chunks, and every chunk ends waiting for its slowest rep while the
other cores idle. With few reps per core (1,000 reps on 128 cores is under 8
each), that idle tail is a large share of the bill. At 32 cores with
8-round chunks the expected efficiency is ~85–90% against ~55–60% at 128
cores: ~35 min and ~14 BU per scenario instead of ~13 min and ~20 BU.
Smaller requests also start sooner on a busy `small` partition. The whole
simulation campaign should cost roughly 300–400 BU.

Wall-time limits are ~6× the measured need and are ceilings, not
expectations. A heavier scenario that times out loses at most one chunk;
resubmit it. Check real cost with `seff <jobid>` after the first jobs finish,
and re-size from that.

`n_jobs` equals `$SLURM_CPUS_PER_TASK`; the checkpoint chunk is 8 × `n_jobs`
reps; BLAS is pinned to one thread per worker. Neither `n_jobs` nor the chunk
size affects results (each rep has its own seed) or checkpoint compatibility,
so cores can be changed between resubmissions.

Cohort job sizes are not measured yet; size them from `seff` on the first run.

Partition choice: `small` (shared nodes, ≤3 days, 384 cores and 745 GiB or
1.5 TiB per node). `medium`/`large` allocate and bill whole nodes and cap at
36 hours. `longrun` (≤10 days) is limited to 192 cores per user.

## Resume and failures

Every simulation command passes `--resume`. If jobs time out or are
cancelled, rerun `submit_02_simulation.sh` (or name the scenarios, e.g.
`submit_02_simulation.sh 11 16`): running and finished jobs are skipped, and
the rest continue from their checkpoints. Do not change the output directory
between attempts. Editing a script does not resize
a job that is already queued or running.

| Symptom | Cause |
|---|---|
| `env.sh: No such file or directory` in `.err` | Submitted from outside the repository root |
| `AssocMaxSubmitJobLimit` at submit | No account; export `SBATCH_ACCOUNT` |
| Job fails instantly with no log | Log directory missing; use the submit helper |
| `OUT_OF_MEMORY` | Raise `--mem` or lower `--cpus-per-task` in that script |

## Cohort jobs

Only after the store is rebuilt and the readiness gates in the runbook are
accepted:

```bash
bash slurm_jobs/submit_cohort.sh --approve --dry-run
bash slurm_jobs/submit_cohort.sh --approve     # or submit_01, then submit_03 / submit_04
```

`--approve` is mandatory for 03, 04, and the combined launcher; their
preflight runs the readiness audit and requires `ready_for_cohort_evaluation`.
The outcome commands still pass `--approve-cohort-evaluation`.

The current checkout is **not** cohort-ready: the readiness audit reports a
stale preprocessed store. Simulations do not depend on that gate.

## Preflight flags

| Flag | Meaning |
|---|---|
| `--scope simulation` | Synthetic jobs only; skips the cohort audit unless `--audit` is given |
| `--scope cohort` | Cohort files and, with `--audit`, `ready_for_cohort_evaluation` |
| `--skip-audit` | Do not load the real cohort |
| `--require-scheduler` | Fail if `sbatch`, partition, or account is missing |

Preflight checks scripts against `config/manuscript_ml.yaml`, the runners,
`uv`, clinical files, and the store. It fits nothing and submits nothing.

## Changing the setup

- **Scenario added or renamed:** edit `config/manuscript_ml.yaml`, add or
  rename `02_simulation_NN_<scenario>.sbatch` with `NN` equal to its config
  position, then run `uv run python slurm_jobs/check_job_inventory.py`.
  Reordering changes seed offsets and is a new campaign.
- **Resources:** edit the `#SBATCH` lines. Keep memory ≤ 2 GiB/core unless a
  job needs more.
- **Science settings:** edit only the locked block in `env.sh`; every scenario
  shares one budget.
- **Account, partition, mail:** environment variables (`SBATCH_ACCOUNT`,
  `SBATCH_PARTITION`, `MS_ML_MAIL_USER`), never hard-coded.
- **Optional:** `MS_ML_UV` (alternate `uv`), `MS_ML_RESULTS_ROOT` (scratch
  redirect), `MS_ML_CONFIG` (do not change for reported runs). Leave
  `MS_ML_UV_EXTRAS` empty; the jobs are CPU-only.
