#!/usr/bin/env bash
# Submit one Slurm job per synthetic scenario, plus the MDE curve job.
# Does not submit readiness or cohort outcome jobs.
#
#   export MS_ML_PARTITION=cpu
#   export MS_ML_ACCOUNT=project_account
#   bash slurm_jobs/submit_simulation.sh
#
# Resubmitting a script continues that experiment: each command passes --resume
# and writes only to its own output directory.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

bash "${SCRIPT_DIR}/preflight.sh" --scope simulation --skip-audit --require-scheduler

mapfile -t scripts < <(find "${SCRIPT_DIR}" -maxdepth 1 -name '02_simulation_*.sbatch' -printf '%f\n' | sort)
if [[ "${#scripts[@]}" -eq 0 ]]; then
  echo "FAIL: no 02_simulation_*.sbatch scripts found" >&2
  exit 1
fi

mkdir -p "${MS_ML_LOG_DIR}"
cd "${MS_ML_REPO_ROOT}"
echo "Submitting ${#scripts[@]} simulation jobs from ${MS_ML_REPO_ROOT}"
echo "These are the locked 1,000-replicate validation jobs, not a smoke test."
for script_name in "${scripts[@]}"; do
  job_id=$(ms_ml_sbatch --parsable "${SCRIPT_DIR}/${script_name}")
  echo "submitted ${script_name} as job ${job_id}"
done
