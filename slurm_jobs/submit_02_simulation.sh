#!/usr/bin/env bash
# Stage 02: synthetic validation, one job per scenario (01-17) plus the MDE
# curve (18). Never submits readiness or cohort jobs.
#
#   export SBATCH_ACCOUNT=project_2019570 SBATCH_PARTITION=small
#   bash slurm_jobs/submit_02_simulation.sh --dry-run   # preview
#   bash slurm_jobs/submit_02_simulation.sh             # all 02_simulation_*.sbatch
#   bash slurm_jobs/submit_02_simulation.sh 05 11       # only scenarios 05 and 11
#
# Safe to rerun after timeouts: running and finished jobs are skipped, the
# rest resume from their checkpoints. --force reruns finished ones.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/submit_lib.sh"
ms_ml_parse_submit_args "bash slurm_jobs/submit_02_simulation.sh [--dry-run] [--force] [NN ...]" "$@"

bash "${MS_ML_JOB_DIR}/preflight.sh" --scope simulation --skip-audit --require-scheduler
mapfile -t scripts < <(ms_ml_stage_scripts 02)
if [[ "${#scripts[@]}" -eq 0 ]]; then
  echo "FAIL: no matching 02_simulation_*.sbatch scripts" >&2
  exit 1
fi
echo "Locked 1,000-replicate validation jobs, not a smoke test."
ms_ml_submit_scripts "${scripts[@]}"
