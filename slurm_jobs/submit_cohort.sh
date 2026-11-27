#!/usr/bin/env bash
# All cohort stages at once: 01 readiness audit, then 03 and 04 with
# --dependency=afterok on that audit. Equivalent to running submit_01, then
# submit_03 and submit_04, with one preflight.
#
#   export SBATCH_ACCOUNT=project_2019570 SBATCH_PARTITION=small
#   bash slurm_jobs/submit_cohort.sh --approve [--dry-run] [--force]

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/submit_lib.sh"
ms_ml_parse_submit_args "bash slurm_jobs/submit_cohort.sh --approve [--dry-run] [--force]" "$@"
ms_ml_require_approval

bash "${MS_ML_JOB_DIR}/preflight.sh" --scope cohort --audit --require-scheduler
ms_ml_submit_scripts 01_readiness_audit.sbatch
mapfile -t outcome < <(ms_ml_stage_scripts 03; ms_ml_stage_scripts 04)
dependency=$(ms_ml_readiness_dependency)
if [[ -z "${dependency}" && "${MS_ML_DRY_RUN}" -eq 1 ]]; then
  dependency="--dependency=afterok:<readiness job>"
fi
ms_ml_submit_scripts ${dependency:+"${dependency}"} "${outcome[@]}"
