#!/usr/bin/env bash
# Stage 01: cohort readiness audit (fits no outcome models; always reruns,
# because it must reflect the current store).
#
#   export SBATCH_ACCOUNT=project_2019570 SBATCH_PARTITION=small
#   bash slurm_jobs/submit_01_readiness.sh [--dry-run]

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/submit_lib.sh"
ms_ml_parse_submit_args "bash slurm_jobs/submit_01_readiness.sh [--dry-run]" "$@"

bash "${MS_ML_JOB_DIR}/preflight.sh" --scope cohort --skip-audit --require-scheduler
mapfile -t scripts < <(ms_ml_stage_scripts 01)
ms_ml_submit_scripts "${scripts[@]}"
