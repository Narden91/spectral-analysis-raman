#!/usr/bin/env bash
# Shared environment for manuscript ML cluster jobs.
# Source this file; do not execute it.
#
# Set before submit. MS_ML_PARTITION/MS_ML_ACCOUNT default to the standard
# SBATCH_PARTITION/SBATCH_ACCOUNT variables, which plain sbatch also reads:
#   export SBATCH_ACCOUNT=project_2019570 SBATCH_PARTITION=small
# Optional overrides:
#   export MS_ML_PARTITION=cpu
#   export MS_ML_ACCOUNT=project_account
#   export MS_ML_MAIL_USER=name@university.edu
#   export MS_ML_UV=uv
#   export MS_ML_UV_EXTRAS="--group gpu"   # only on a node where the CUDA extra is intended
#
# Jobs force --no-gpu. Do not set MS_ML_UV_EXTRAS on a CPU partition: the gpu
# dependency group installs CuPy and is not required for these scripts.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Source slurm_jobs/env.sh; do not execute it." >&2
  exit 1
fi

if [[ -n "${MS_ML_ENV_LOADED:-}" ]]; then
  return 0
fi
MS_ML_ENV_LOADED=1

MS_ML_JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MS_ML_REPO_ROOT="$(cd "${MS_ML_JOB_DIR}/.." && pwd)"

MS_ML_CONFIG="${MS_ML_CONFIG:-${MS_ML_REPO_ROOT}/config/manuscript_ml.yaml}"
MS_ML_UV="${MS_ML_UV:-uv}"
MS_ML_PARTITION="${MS_ML_PARTITION:-${SBATCH_PARTITION:-}}"
MS_ML_ACCOUNT="${MS_ML_ACCOUNT:-${SBATCH_ACCOUNT:-}}"
MS_ML_MAIL_USER="${MS_ML_MAIL_USER:-}"
MS_ML_RESULTS_ROOT="${MS_ML_RESULTS_ROOT:-${MS_ML_REPO_ROOT}/Results_manuscript_ml}"
MS_ML_LOG_DIR="${MS_ML_LOG_DIR:-${MS_ML_RESULTS_ROOT}/slurm_logs}"
MS_ML_VALIDATION_ROOT="${MS_ML_VALIDATION_ROOT:-${MS_ML_RESULTS_ROOT}/validation}"

ms_ml_enter_repo() {
  cd "${MS_ML_REPO_ROOT}"
  mkdir -p "${MS_ML_LOG_DIR}" "${MS_ML_VALIDATION_ROOT}"
  # One BLAS thread per joblib worker. n_jobs is the allocated CPU count.
  export OMP_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1
  export VECLIB_MAXIMUM_THREADS=1
  export PYTHONUNBUFFERED=1
}

ms_ml_n_jobs() {
  if [[ -n "${SLURM_CPUS_PER_TASK:-}" && "${SLURM_CPUS_PER_TASK}" != "0" ]]; then
    echo "${SLURM_CPUS_PER_TASK}"
  else
    echo 1
  fi
}

# Reps per checkpoint flush. Workers pull reps dynamically inside a chunk and
# wait for the slowest one at its end, so a chunk of 8 rounds keeps the barrier
# idle time small while losing at most ~8 rep-times of work on a kill.
# Chunking does not change results: each rep has its own fixed seed.
ms_ml_checkpoint_chunk() {
  echo $(( $(ms_ml_n_jobs) * 8 ))
}

ms_ml_uv() {
  local -a extras=()
  if [[ -n "${MS_ML_UV_EXTRAS:-}" ]]; then
    # User-supplied extras such as "--group gpu".
    read -r -a extras <<< "${MS_ML_UV_EXTRAS}"
  fi
  "${MS_ML_UV}" run "${extras[@]}" "$@"
}

ms_ml_sbatch() {
  local -a extra=()
  if [[ -n "${MS_ML_PARTITION}" ]]; then
    extra+=(--partition="${MS_ML_PARTITION}")
  fi
  if [[ -n "${MS_ML_ACCOUNT}" ]]; then
    extra+=(--account="${MS_ML_ACCOUNT}")
  fi
  if [[ -n "${MS_ML_MAIL_USER}" ]]; then
    extra+=(--mail-user="${MS_ML_MAIL_USER}" --mail-type=FAIL,END,TIME_LIMIT)
  fi
  extra+=(--chdir="${MS_ML_REPO_ROOT}")
  sbatch "${extra[@]}" "$@"
}

# Locked validation budget from manuscript_ml/runbook.md. One scenario per
# output directory: the runner always writes the same CSV names.
ms_ml_run_simulation_scenario() {
  local scenario="$1"
  local n_jobs output_dir
  n_jobs="$(ms_ml_n_jobs)"
  output_dir="${MS_ML_VALIDATION_ROOT}/${scenario}"
  mkdir -p "${output_dir}"
  echo "experiment=simulation scenario=${scenario} output_dir=${output_dir} n_jobs=${n_jobs} host=$(hostname) start=$(date -Is)"
  ms_ml_uv python manuscript_ml/run_simulation_calibration.py \
    --config "${MS_ML_CONFIG}" \
    --output-dir "${output_dir}" \
    --scenarios "${scenario}" \
    --calibrate-all-scenarios \
    --scenario-reps 1000 \
    --calibration-reps 1000 \
    --bootstraps 2000 \
    --participants 92 \
    --n-test-participants 25000 \
    --interval-method studentized_t \
    --cohort-grid \
    --seed-set validation \
    --no-gpu \
    --n-jobs "${n_jobs}" \
    --resume \
    --checkpoint-chunk-size "$(ms_ml_checkpoint_chunk)"
}

# The runner has no skip-calibration flag. This job also writes a single-setting
# simulation_calibration_report.csv in its own directory. Do not merge that file
# with the per-scenario calibration reports.
ms_ml_run_mde_signal_curve() {
  local n_jobs output_dir
  n_jobs="$(ms_ml_n_jobs)"
  output_dir="${MS_ML_VALIDATION_ROOT}/mde_signal_curve"
  mkdir -p "${output_dir}"
  echo "experiment=mde_signal_curve output_dir=${output_dir} n_jobs=${n_jobs} host=$(hostname) start=$(date -Is)"
  ms_ml_uv python manuscript_ml/run_simulation_calibration.py \
    --config "${MS_ML_CONFIG}" \
    --output-dir "${output_dir}" \
    --skip-scenario-suite \
    --run-mde-curve \
    --scenario-reps 1000 \
    --calibration-reps 1000 \
    --bootstraps 2000 \
    --participants 92 \
    --n-test-participants 25000 \
    --interval-method studentized_t \
    --cohort-grid \
    --seed-set validation \
    --no-gpu \
    --n-jobs "${n_jobs}" \
    --resume \
    --checkpoint-chunk-size "$(ms_ml_checkpoint_chunk)"
}

ms_ml_run_readiness_audit() {
  local output_dir="${MS_ML_RESULTS_ROOT}/readiness_review"
  mkdir -p "${output_dir}"
  echo "experiment=readiness_audit output_dir=${output_dir} host=$(hostname) start=$(date -Is)"
  ms_ml_uv python manuscript_ml/audit_readiness.py \
    --config "${MS_ML_CONFIG}" \
    --planned-bootstraps 2000 \
    --planned-sim-reps 1000 \
    --output-dir "${output_dir}"
}

ms_ml_run_cohort_primary_point_estimate() {
  local n_jobs output_dir
  n_jobs="$(ms_ml_n_jobs)"
  output_dir="${MS_ML_RESULTS_ROOT}/cohort_primary_point"
  mkdir -p "${output_dir}"
  echo "experiment=cohort_primary_point_estimate output_dir=${output_dir} n_jobs=${n_jobs} host=$(hostname) start=$(date -Is)"
  ms_ml_uv python manuscript_ml/run_primary_evaluation.py \
    --config "${MS_ML_CONFIG}" \
    --cohort \
    --approve-cohort-evaluation \
    --bootstraps 0 \
    --no-gpu \
    --n-jobs "${n_jobs}" \
    --interval-method studentized_t \
    --output-dir "${output_dir}"
}

ms_ml_run_cohort_primary_bootstrap_interval() {
  local n_jobs output_dir
  n_jobs="$(ms_ml_n_jobs)"
  output_dir="${MS_ML_RESULTS_ROOT}/cohort_primary_bootstrap"
  mkdir -p "${output_dir}"
  echo "experiment=cohort_primary_bootstrap_interval output_dir=${output_dir} n_jobs=${n_jobs} host=$(hostname) start=$(date -Is)"
  ms_ml_uv python manuscript_ml/run_primary_evaluation.py \
    --config "${MS_ML_CONFIG}" \
    --cohort \
    --approve-cohort-evaluation \
    --bootstraps 2000 \
    --no-gpu \
    --n-jobs "${n_jobs}" \
    --interval-method studentized_t \
    --output-dir "${output_dir}"
}

ms_ml_run_cohort_sensitivity_and_secondary_endpoints() {
  local n_jobs output_dir
  n_jobs="$(ms_ml_n_jobs)"
  output_dir="${MS_ML_RESULTS_ROOT}/cohort_sensitivity"
  mkdir -p "${output_dir}"
  echo "experiment=cohort_sensitivity_and_secondary_endpoints output_dir=${output_dir} n_jobs=${n_jobs} host=$(hostname) start=$(date -Is)"
  ms_ml_uv python manuscript_ml/run_sensitivity_and_secondary.py \
    --config "${MS_ML_CONFIG}" \
    --cohort \
    --approve-cohort-evaluation \
    --bootstraps 2000 \
    --no-gpu \
    --n-jobs "${n_jobs}" \
    --interval-method studentized_t \
    --output-dir "${output_dir}"
}
