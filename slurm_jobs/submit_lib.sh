#!/usr/bin/env bash
# Shared submission logic for the submit_NN_*.sh helpers. Source; do not run.
#
# Every helper accepts:
#   --dry-run   print what would be submitted, submit nothing
#   --force     also resubmit jobs whose output is already complete
#   --approve   required by cohort outcome stages (03, 04)
#   NN ...      optional two-digit numbers selecting individual scripts
#               within a stage (e.g. submit_02_simulation.sh 05 11)
#
# A job is skipped when a job with the same name is already queued or running
# (two jobs must never share a checkpoint), or when its output directory holds
# run_provenance_*.json, which the runners write last.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Source slurm_jobs/submit_lib.sh; do not execute it." >&2
  exit 1
fi

SUBMIT_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SUBMIT_LIB_DIR}/env.sh"

MS_ML_DRY_RUN=0
MS_ML_FORCE=0
MS_ML_APPROVE=0
MS_ML_SELECT=()

ms_ml_parse_submit_args() {
  local usage="$1"
  shift
  local arg
  for arg in "$@"; do
    case "${arg}" in
      --dry-run) MS_ML_DRY_RUN=1 ;;
      --force) MS_ML_FORCE=1 ;;
      --approve) MS_ML_APPROVE=1 ;;
      [0-9][0-9]) MS_ML_SELECT+=("${arg}") ;;
      -h|--help) echo "Usage: ${usage}"; exit 0 ;;
      *) echo "Unknown argument: ${arg}" >&2; echo "Usage: ${usage}" >&2; exit 2 ;;
    esac
  done
}

ms_ml_require_approval() {
  if [[ "${MS_ML_APPROVE}" -ne 1 ]]; then
    echo "Refusing to submit cohort outcome jobs without --approve." >&2
    echo "That flag is the authorization for --approve-cohort-evaluation. It is not implied by creating these scripts." >&2
    exit 2
  fi
}

# Output directory a script writes to; empty means "always run" (readiness gate).
ms_ml_script_output_dir() {
  local name="${1%.sbatch}"
  case "${name}" in
    01_readiness_audit) echo "" ;;
    02_simulation_[0-9][0-9]_*) echo "${MS_ML_VALIDATION_ROOT}/${name#02_simulation_[0-9][0-9]_}" ;;
    03_cohort_primary_point_estimate) echo "${MS_ML_RESULTS_ROOT}/cohort_primary_point" ;;
    03_cohort_primary_bootstrap_interval) echo "${MS_ML_RESULTS_ROOT}/cohort_primary_bootstrap" ;;
    04_cohort_sensitivity_and_secondary_endpoints) echo "${MS_ML_RESULTS_ROOT}/cohort_sensitivity" ;;
    *) echo "Unknown job script: $1" >&2; return 1 ;;
  esac
}

ms_ml_job_name() {
  sed -n 's/^#SBATCH --job-name=//p' "${MS_ML_JOB_DIR}/$1"
}

# Job id of a queued/running job with this name, if any.
ms_ml_active_job_id() {
  squeue --me -h -n "$1" -o '%i' 2>/dev/null | head -n 1
}

# Scripts for a stage, e.g. "02", filtered by MS_ML_SELECT for 02_simulation_NN_.
ms_ml_stage_scripts() {
  local stage="$1" script sel keep
  while IFS= read -r script; do
    if [[ "${#MS_ML_SELECT[@]}" -gt 0 ]]; then
      keep=0
      for sel in "${MS_ML_SELECT[@]}"; do
        [[ "${script}" == "${stage}"_*_"${sel}"_* ]] && keep=1
      done
      [[ "${keep}" -eq 1 ]] || continue
    fi
    echo "${script}"
  done < <(find "${MS_ML_JOB_DIR}" -maxdepth 1 -name "${stage}_*.sbatch" -printf '%f\n' | sort)
}

# Submit scripts with the skip rules. Optional first argument: --dependency=...
# Prints "submitted <script> <jobid>" lines; sets MS_ML_SUBMITTED_COUNT.
ms_ml_submit_scripts() {
  local dependency=""
  if [[ "${1:-}" == --dependency=* ]]; then
    dependency="$1"
    shift
  fi
  local script job_name active out_dir job_id
  local -a extra=()
  [[ -n "${dependency}" ]] && extra+=("${dependency}")
  MS_ML_SUBMITTED_COUNT=0
  mkdir -p "${MS_ML_LOG_DIR}"
  cd "${MS_ML_REPO_ROOT}"
  for script in "$@"; do
    job_name=$(ms_ml_job_name "${script}")
    active=$(ms_ml_active_job_id "${job_name}")
    if [[ -n "${active}" ]]; then
      echo "skip ${script}: ${job_name} is already queued or running (job ${active})"
      continue
    fi
    out_dir=$(ms_ml_script_output_dir "${script}")
    if [[ "${MS_ML_FORCE}" -eq 0 && -n "${out_dir}" ]] \
      && compgen -G "${out_dir}/run_provenance_*.json" > /dev/null; then
      echo "skip ${script}: finished (${out_dir#"${MS_ML_REPO_ROOT}/"}); --force to rerun"
      continue
    fi
    if [[ "${MS_ML_DRY_RUN}" -eq 1 ]]; then
      echo "would submit ${script} ${dependency}"
    else
      job_id=$(ms_ml_sbatch --parsable "${extra[@]}" "${MS_ML_JOB_DIR}/${script}")
      echo "submitted ${script} as job ${job_id} ${dependency}"
    fi
    MS_ML_SUBMITTED_COUNT=$((MS_ML_SUBMITTED_COUNT + 1))
  done
  if [[ "${MS_ML_DRY_RUN}" -eq 1 ]]; then
    echo "${MS_ML_SUBMITTED_COUNT} job(s) would be submitted."
  else
    echo "${MS_ML_SUBMITTED_COUNT} job(s) submitted."
  fi
}

# Cohort outcome jobs wait for a readiness audit that is still queued/running.
ms_ml_readiness_dependency() {
  local audit_id
  audit_id=$(ms_ml_active_job_id "$(ms_ml_job_name 01_readiness_audit.sbatch)")
  [[ -n "${audit_id}" ]] && echo "--dependency=afterok:${audit_id}"
  return 0
}
