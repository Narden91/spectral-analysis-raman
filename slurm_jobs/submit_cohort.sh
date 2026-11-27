#!/usr/bin/env bash
# Submit readiness, then the three cohort outcome jobs after that audit succeeds.
# Creating these scripts is not authorization. Pass --approve only after the
# runbook readiness gates have been accepted.
#
#   export MS_ML_PARTITION=cpu
#   export MS_ML_ACCOUNT=project_account
#   bash slurm_jobs/submit_cohort.sh --approve

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

APPROVE=0
for arg in "$@"; do
  case "${arg}" in
    --approve) APPROVE=1 ;;
    -h|--help)
      echo "Usage: bash slurm_jobs/submit_cohort.sh --approve"
      exit 0
      ;;
    *)
      echo "Unknown argument: ${arg}" >&2
      exit 2
      ;;
  esac
done

if [[ "${APPROVE}" -ne 1 ]]; then
  echo "Refusing to submit cohort jobs without --approve." >&2
  echo "That flag is the authorization for --approve-cohort-evaluation. It is not implied by creating these scripts." >&2
  exit 2
fi

bash "${SCRIPT_DIR}/preflight.sh" --scope cohort --audit --require-scheduler
mkdir -p "${MS_ML_LOG_DIR}"
cd "${MS_ML_REPO_ROOT}"

readiness_id=$(ms_ml_sbatch --parsable "${SCRIPT_DIR}/01_readiness_audit.sbatch")
echo "submitted 01_readiness_audit.sbatch as job ${readiness_id}"

for script_name in \
  03_cohort_primary_point_estimate.sbatch \
  03_cohort_primary_bootstrap_interval.sbatch \
  04_cohort_sensitivity_and_secondary_endpoints.sbatch
do
  job_id=$(ms_ml_sbatch --parsable --dependency="afterok:${readiness_id}" "${SCRIPT_DIR}/${script_name}")
  echo "submitted ${script_name} as job ${job_id} afterok:${readiness_id}"
done
