#!/usr/bin/env bash
# Stage 03: cohort primary LDL-C point estimate and bootstrap interval.
# Requires --approve: creating these scripts is not authorization. Runs the
# cohort preflight with the readiness audit first, and waits (afterok) for a
# stage-01 readiness job if one is still queued or running.
#
#   export SBATCH_ACCOUNT=project_2019570 SBATCH_PARTITION=small
#   bash slurm_jobs/submit_03_cohort_primary.sh --approve [--dry-run] [--force]

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/submit_lib.sh"
ms_ml_parse_submit_args "bash slurm_jobs/submit_03_cohort_primary.sh --approve [--dry-run] [--force]" "$@"
ms_ml_require_approval

bash "${MS_ML_JOB_DIR}/preflight.sh" --scope cohort --audit --require-scheduler
mapfile -t scripts < <(ms_ml_stage_scripts 03)
dependency=$(ms_ml_readiness_dependency)
ms_ml_submit_scripts ${dependency:+"${dependency}"} "${scripts[@]}"
