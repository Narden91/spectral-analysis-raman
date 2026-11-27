#!/usr/bin/env bash
# Manuscript ML simulation experiment 09: qc_map_dropout. Config order is the canonical seed offset.
# Submit from the repository root after bash slurm_jobs/preflight.sh.
# Partition and account are applied by the submit helper from MS_ML_PARTITION and MS_ML_ACCOUNT.
#SBATCH --job-name=sim-09-qc_map_dropout
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=Results_manuscript_ml/slurm_logs/%x-%j.out
#SBATCH --error=Results_manuscript_ml/slurm_logs/%x-%j.err
#SBATCH --mail-type=FAIL,END,TIME_LIMIT
#SBATCH --mail-user=emanuele.nardone@uef.fi

set -euo pipefail
# Slurm executes a spool copy of this file, so BASH_SOURCE is not slurm_jobs/.
# Submit from the repository root so SLURM_SUBMIT_DIR contains slurm_jobs/env.sh.
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/slurm_jobs/env.sh" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}/slurm_jobs"
else
  SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
fi
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"
ms_ml_enter_repo
ms_ml_run_simulation_scenario "qc_map_dropout"
