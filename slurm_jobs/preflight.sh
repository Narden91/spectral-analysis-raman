#!/usr/bin/env bash
# Check that the manuscript ML cluster jobs can be submitted from this checkout.
#
#   bash slurm_jobs/preflight.sh
#   bash slurm_jobs/preflight.sh --scope simulation --skip-audit
#   bash slurm_jobs/preflight.sh --scope cohort --audit --require-scheduler
#
# Missing sbatch, partition, or account is a warning unless --require-scheduler
# is set. Cohort outcome jobs are not authorized by a successful preflight.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

SCOPE="all"
RUN_AUDIT=1
REQUIRE_SCHEDULER=0
AUDIT_EXPLICIT=0

usage() {
  cat <<'EOF'
Usage: bash slurm_jobs/preflight.sh [--scope all|simulation|cohort] [--audit|--skip-audit] [--require-scheduler]

Checks the locked script inventory, runner entry points, uv, clinical files,
and the preprocessed store. --audit also runs manuscript_ml/audit_readiness.py
and requires ready_for_cohort_evaluation. --scope simulation skips that audit
unless --audit is explicit.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scope)
      SCOPE="$2"
      shift 2
      ;;
    --audit)
      RUN_AUDIT=1
      AUDIT_EXPLICIT=1
      shift
      ;;
    --skip-audit)
      RUN_AUDIT=0
      shift
      ;;
    --require-scheduler)
      REQUIRE_SCHEDULER=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${SCOPE}" != "all" && "${SCOPE}" != "simulation" && "${SCOPE}" != "cohort" ]]; then
  echo "FAIL: --scope must be all, simulation, or cohort" >&2
  exit 2
fi
if [[ "${SCOPE}" == "simulation" && "${AUDIT_EXPLICIT}" -eq 0 ]]; then
  RUN_AUDIT=0
fi

failures=0
warnings=0

ok() { echo "OK: $*"; }
warn() { echo "WARN: $*" >&2; warnings=$((warnings + 1)); }
fail() { echo "FAIL: $*" >&2; failures=$((failures + 1)); }

require_file() {
  if [[ -f "$1" ]]; then
    ok "found ${1#"${MS_ML_REPO_ROOT}/"}"
  else
    fail "missing $1"
  fi
}

echo "Manuscript ML preflight"
echo "repo=${MS_ML_REPO_ROOT}"
echo "scope=${SCOPE} audit=${RUN_AUDIT} require_scheduler=${REQUIRE_SCHEDULER}"

require_file "${MS_ML_CONFIG}"
require_file "${MS_ML_REPO_ROOT}/uv.lock"
require_file "${MS_ML_REPO_ROOT}/pyproject.toml"
require_file "${MS_ML_REPO_ROOT}/manuscript_ml/audit_readiness.py"
require_file "${MS_ML_REPO_ROOT}/manuscript_ml/run_simulation_calibration.py"
require_file "${MS_ML_REPO_ROOT}/manuscript_ml/run_primary_evaluation.py"
require_file "${MS_ML_REPO_ROOT}/manuscript_ml/run_sensitivity_and_secondary.py"
require_file "${MS_ML_REPO_ROOT}/data/clinical_master.csv"
require_file "${MS_ML_REPO_ROOT}/data/clinical_master_dictionary.csv"
require_file "${MS_ML_REPO_ROOT}/data/clinical_master_notes.md"
require_file "${MS_ML_REPO_ROOT}/data/preprocessed/manifest.json"
require_file "${MS_ML_REPO_ROOT}/data/preprocessed/measurements.parquet"
require_file "${MS_ML_REPO_ROOT}/data/preprocessed/normalized_1300.npy"
require_file "${MS_ML_REPO_ROOT}/data/preprocessed/normalized_2850.npy"
require_file "${MS_ML_REPO_ROOT}/data/parsed/manifest.json"

if command -v "${MS_ML_UV}" >/dev/null 2>&1; then
  ok "uv executable: $(command -v "${MS_ML_UV}")"
else
  fail "uv executable not found (${MS_ML_UV})"
fi

if [[ -d "${MS_ML_REPO_ROOT}/.venv" ]]; then
  ok "project virtualenv exists"
else
  warn "no .venv yet; the first uv run will create it and may download packages"
fi

if [[ "${failures}" -eq 0 ]]; then
  if ms_ml_enter_repo && ms_ml_uv python "${SCRIPT_DIR}/check_job_inventory.py"; then
    ok "script inventory matches config/manuscript_ml.yaml"
  else
    fail "script inventory does not match config/manuscript_ml.yaml"
  fi
fi

if [[ "${failures}" -eq 0 ]]; then
  if ms_ml_uv python -c 'import sys; from manuscript_ml import load_manuscript_config; load_manuscript_config("config/manuscript_ml.yaml"); assert sys.version_info >= (3, 12), sys.version'; then
    ok "uv can import manuscript_ml and the locked config"
  else
    fail "uv run python could not import manuscript_ml or load the config"
  fi
fi

if command -v sbatch >/dev/null 2>&1; then
  ok "sbatch executable: $(command -v sbatch)"
else
  if [[ "${REQUIRE_SCHEDULER}" -eq 1 ]]; then
    fail "sbatch is not on PATH"
  else
    warn "sbatch is not on PATH; inventory can be checked here, submission must be done on a cluster login node"
  fi
fi

if [[ -n "${MS_ML_PARTITION}" ]]; then
  ok "MS_ML_PARTITION=${MS_ML_PARTITION}"
elif [[ "${REQUIRE_SCHEDULER}" -eq 1 ]]; then
  fail "MS_ML_PARTITION is empty"
else
  warn "MS_ML_PARTITION is empty; sbatch will use the site default partition"
fi

if [[ -n "${MS_ML_ACCOUNT}" ]]; then
  ok "MS_ML_ACCOUNT=${MS_ML_ACCOUNT}"
elif [[ "${REQUIRE_SCHEDULER}" -eq 1 ]]; then
  fail "MS_ML_ACCOUNT is empty"
else
  warn "MS_ML_ACCOUNT is empty; set it before submit if the site requires an account"
fi

if [[ -n "${MS_ML_UV_EXTRAS:-}" ]]; then
  warn "MS_ML_UV_EXTRAS=${MS_ML_UV_EXTRAS}; these CPU jobs pass --no-gpu and do not need the gpu dependency group"
fi

mkdir -p "${MS_ML_LOG_DIR}"
ok "log directory ${MS_ML_LOG_DIR#"${MS_ML_REPO_ROOT}/"}"

if [[ "${RUN_AUDIT}" -eq 1 && "${SCOPE}" != "simulation" ]]; then
  if [[ "${failures}" -ne 0 ]]; then
    fail "skipping readiness audit because earlier checks failed"
  else
    audit_dir="${MS_ML_RESULTS_ROOT}/readiness_review"
    echo "Running readiness audit into ${audit_dir}"
    if ms_ml_run_readiness_audit; then
      if ms_ml_uv python "${SCRIPT_DIR}/check_job_inventory.py" --readiness-manifest "${audit_dir}/readiness_manifest.json"; then
        ok "readiness audit reports ready_for_cohort_evaluation=true"
      else
        fail "readiness audit did not report ready_for_cohort_evaluation=true"
      fi
    else
      fail "readiness audit command failed"
    fi
  fi
elif [[ "${SCOPE}" != "simulation" ]]; then
  warn "cohort readiness audit was not run; pass --audit before submitting cohort jobs"
fi

echo
if [[ "${failures}" -ne 0 ]]; then
  echo "Preflight failed: ${failures} failure(s), ${warnings} warning(s)." >&2
  exit 1
fi
echo "Preflight passed with ${warnings} warning(s)."
echo "Passing preflight does not authorize cohort outcome evaluation."
exit 0
