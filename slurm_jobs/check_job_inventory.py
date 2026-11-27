"""Check that Slurm scripts match the locked manuscript ML experiments.

The simulation runner writes fixed CSV names, so each configured scenario must
have its own script and output directory. Reordering ``simulation.scenarios``
changes canonical seed offsets and these filenames together.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

JOB_DIR = Path(__file__).resolve().parent
REPO_ROOT = JOB_DIR.parent
CONFIG_PATH = REPO_ROOT / "config" / "manuscript_ml.yaml"

READINESS_SCRIPT = "01_readiness_audit.sbatch"
POINT_SCRIPT = "03_cohort_primary_point_estimate.sbatch"
BOOTSTRAP_SCRIPT = "03_cohort_primary_bootstrap_interval.sbatch"
SENSITIVITY_SCRIPT = "04_cohort_sensitivity_and_secondary_endpoints.sbatch"

LOCKED_SCENARIO_FLAGS = (
    "--calibrate-all-scenarios",
    "--scenario-reps 1000",
    "--calibration-reps 1000",
    "--bootstraps 2000",
    "--participants 92",
    "--n-test-participants 25000",
    "--interval-method studentized_t",
    "--cohort-grid",
    "--seed-set validation",
    "--no-gpu",
    "--resume",
    "--checkpoint-chunk-size 25",
)
MDE_FLAGS = ("--skip-scenario-suite", "--run-mde-curve")
COHORT_OUTPUTS = {
    POINT_SCRIPT: "cohort_primary_point",
    BOOTSTRAP_SCRIPT: "cohort_primary_bootstrap",
    SENSITIVITY_SCRIPT: "cohort_sensitivity",
}
READINESS_OUTPUT = "readiness_review"
SCENARIO_NAME_RE = re.compile(r"^[a-z0-9_]+$")
SBATCH_NAME_RE = re.compile(r"^#SBATCH --job-name=(\S+)\s*$", re.MULTILINE)


def simulation_script_name(index: int, scenario: str) -> str:
    """1-based config order, zero-padded so directory listings follow the protocol."""
    return f"02_simulation_{index:02d}_{scenario}.sbatch"


def mde_script_name(n_scenarios: int) -> str:
    return f"02_simulation_{n_scenarios + 1:02d}_mde_signal_curve.sbatch"


def load_scenarios(config_path: Path = CONFIG_PATH) -> list[str]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    scenarios = raw["simulation"]["scenarios"]
    names = [str(spec["scenario"]) for spec in scenarios]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate scenario names in {config_path}: {names}")
    return names


def _function_body(source: str, function_name: str) -> str:
    marker = f"{function_name}() {{"
    start = source.find(marker)
    if start < 0:
        raise ValueError(f"Missing function {function_name}()")
    end = source.find("\n}\n", start)
    if end < 0:
        raise ValueError(f"Unclosed function {function_name}()")
    return source[start:end]


def validate_inventory(job_dir: Path = JOB_DIR, config_path: Path = CONFIG_PATH) -> list[str]:
    """Return human-readable errors. An empty list means the inventory matches."""
    errors: list[str] = []
    try:
        scenarios = load_scenarios(config_path)
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        return [f"Could not read simulation scenarios from {config_path}: {exc}"]

    for name in scenarios:
        if SCENARIO_NAME_RE.fullmatch(name) is None:
            errors.append(f"Scenario name is not a safe script token: {name!r}")

    expected_sim = {
        simulation_script_name(index, name): name for index, name in enumerate(scenarios, start=1)
    }
    mde_name = mde_script_name(len(scenarios))
    present_sim = sorted(path.name for path in job_dir.glob("02_simulation_*.sbatch"))
    expected_names = sorted([*expected_sim, mde_name])
    if present_sim != expected_names:
        missing = sorted(set(expected_names) - set(present_sim))
        extra = sorted(set(present_sim) - set(expected_names))
        errors.append(f"Simulation script inventory mismatch. missing={missing} extra={extra}")

    env_path = job_dir / "env.sh"
    if not env_path.is_file():
        return [*errors, f"Missing {env_path}"]
    env_text = env_path.read_text(encoding="utf-8")
    try:
        scenario_body = _function_body(env_text, "ms_ml_run_simulation_scenario")
        mde_body = _function_body(env_text, "ms_ml_run_mde_signal_curve")
    except ValueError as exc:
        errors.append(str(exc))
        scenario_body = ""
        mde_body = ""

    for flag in LOCKED_SCENARIO_FLAGS:
        if flag not in scenario_body:
            errors.append(f"ms_ml_run_simulation_scenario is missing locked flag {flag}")
        if flag not in mde_body and flag != "--calibrate-all-scenarios":
            errors.append(f"ms_ml_run_mde_signal_curve is missing locked flag {flag}")
    if "--calibrate-all-scenarios" in mde_body:
        errors.append("MDE job must not pass --calibrate-all-scenarios; that would rerun every scenario")
    if "--scenarios" in mde_body:
        errors.append("MDE job must not pass --scenarios")
    for flag in MDE_FLAGS:
        if flag not in mde_body:
            errors.append(f"ms_ml_run_mde_signal_curve is missing {flag}")
    if 'output_dir="${MS_ML_VALIDATION_ROOT}/${scenario}"' not in scenario_body:
        errors.append("Scenario jobs must write to validation/<scenario>, not a shared directory")
    if 'output_dir="${MS_ML_VALIDATION_ROOT}/mde_signal_curve"' not in mde_body:
        errors.append("MDE job must write to validation/mde_signal_curve")

    job_names: dict[str, str] = {}
    for script_name, scenario in expected_sim.items():
        errors.extend(
            _check_sbatch(
                job_dir / script_name,
                job_names,
                required=(f'ms_ml_run_simulation_scenario "{scenario}"',),
            )
        )
    errors.extend(
        _check_sbatch(
            job_dir / mde_name,
            job_names,
            required=("ms_ml_run_mde_signal_curve",),
            forbidden=tuple(f'"{name}"' for name in scenarios),
        )
    )

    cohort_calls = {
        READINESS_SCRIPT: ("ms_ml_run_readiness_audit",),
        POINT_SCRIPT: ("ms_ml_run_cohort_primary_point_estimate",),
        BOOTSTRAP_SCRIPT: ("ms_ml_run_cohort_primary_bootstrap_interval",),
        SENSITIVITY_SCRIPT: ("ms_ml_run_cohort_sensitivity_and_secondary_endpoints",),
    }
    for script_name, calls in cohort_calls.items():
        errors.extend(_check_sbatch(job_dir / script_name, job_names, required=calls))

    for function_name, output_name in (
        ("ms_ml_run_readiness_audit", READINESS_OUTPUT),
        ("ms_ml_run_cohort_primary_point_estimate", COHORT_OUTPUTS[POINT_SCRIPT]),
        ("ms_ml_run_cohort_primary_bootstrap_interval", COHORT_OUTPUTS[BOOTSTRAP_SCRIPT]),
        ("ms_ml_run_cohort_sensitivity_and_secondary_endpoints", COHORT_OUTPUTS[SENSITIVITY_SCRIPT]),
    ):
        try:
            body = _function_body(env_text, function_name)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if f'"${{MS_ML_RESULTS_ROOT}}/{output_name}"' not in body:
            errors.append(f"{function_name} must write Results_manuscript_ml/{output_name}")
        if function_name != "ms_ml_run_readiness_audit" and "--approve-cohort-evaluation" not in body:
            errors.append(f"{function_name} is missing --approve-cohort-evaluation")
        if function_name != "ms_ml_run_readiness_audit" and "--no-gpu" not in body:
            errors.append(f"{function_name} is missing --no-gpu")

    submit_sim_path = job_dir / "submit_simulation.sh"
    submit_sim = submit_sim_path.read_text(encoding="utf-8") if submit_sim_path.is_file() else ""
    if "02_simulation_*.sbatch" not in submit_sim:
        errors.append("submit_simulation.sh must submit only 02_simulation_*.sbatch")
    if "03_cohort_" in submit_sim or "04_cohort_" in submit_sim or "01_readiness" in submit_sim:
        errors.append("submit_simulation.sh must not submit readiness or cohort jobs")
    submit_cohort_path = job_dir / "submit_cohort.sh"
    submit_cohort = submit_cohort_path.read_text(encoding="utf-8") if submit_cohort_path.is_file() else ""
    if "--approve" not in submit_cohort:
        errors.append("submit_cohort.sh must require an explicit --approve flag")
    return errors


def _check_sbatch(
    path: Path,
    job_names: dict[str, str],
    *,
    required: tuple[str, ...],
    forbidden: tuple[str, ...] = (),
) -> list[str]:
    errors: list[str] = []
    if not path.is_file():
        return [f"Missing script {path.name}"]
    text = path.read_text(encoding="utf-8")
    if not text.startswith("#!/usr/bin/env bash\n"):
        errors.append(f"{path.name} must start with a bash shebang")
    if "set -euo pipefail" not in text:
        errors.append(f"{path.name} must set -euo pipefail")
    if 'source "${SCRIPT_DIR}/env.sh"' not in text:
        errors.append(f"{path.name} must source env.sh")
    if "SLURM_SUBMIT_DIR" not in text:
        errors.append(f"{path.name} must locate env.sh from SLURM_SUBMIT_DIR; Slurm copies the script to the spool")
    match = SBATCH_NAME_RE.search(text)
    if match is None:
        errors.append(f"{path.name} is missing #SBATCH --job-name")
    else:
        job_name = match.group(1)
        if job_name in job_names:
            errors.append(f"Duplicate Slurm job name {job_name} in {path.name} and {job_names[job_name]}")
        job_names[job_name] = path.name
    for item in required:
        if item not in text:
            errors.append(f"{path.name} must call {item}")
    for item in forbidden:
        if item in text:
            errors.append(f"{path.name} must not reference scenario {item}")
    directives = (
        "--cpus-per-task=",
        "--mem=",
        "--time=",
        "--output=Results_manuscript_ml/slurm_logs/%x-%j.out",
        "--error=Results_manuscript_ml/slurm_logs/%x-%j.err",
        "--mail-type=FAIL,END,TIME_LIMIT",
        "--mail-user=emanuele.nardone@uef.fi",
    )
    for directive in directives:
        if directive not in text:
            errors.append(f"{path.name} is missing Slurm directive containing {directive}")
    return errors


def cohort_readiness_errors(manifest_path: Path) -> list[str]:
    """Return errors when an audit manifest is not ready for cohort evaluation."""
    if not manifest_path.is_file():
        return [f"Missing readiness manifest: {manifest_path}"]
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    measurement = payload.get("measurement_readiness", {})
    join = measurement.get("cohort_join", {})
    ready = measurement.get("ready_for_cohort_evaluation")
    status = join.get("status")
    error = join.get("error")
    print(f"ready_for_cohort_evaluation={ready}")
    print(f"cohort_join_status={status}")
    if error:
        print(f"cohort_join_error={error}")
    if ready is True:
        return []
    detail = f"cohort_join_status={status}"
    if error:
        detail = f"{detail}; {error}"
    return [f"cohort evaluation is not ready ({detail}); simulation jobs do not use this gate"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check manuscript ML Slurm script inventory.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--job-dir", type=Path, default=JOB_DIR)
    parser.add_argument(
        "--readiness-manifest",
        type=Path,
        default=None,
        help="If set, also require this audit manifest to be ready for cohort evaluation.",
    )
    args = parser.parse_args(argv)
    if args.readiness_manifest is not None and args.job_dir == JOB_DIR and args.config == CONFIG_PATH:
        # Manifest-only checks are used by preflight after the inventory has already passed.
        errors = cohort_readiness_errors(args.readiness_manifest)
        if errors:
            for error in errors:
                print(f"FAIL: {error}", file=sys.stderr)
            return 1
        print("OK: readiness manifest allows cohort evaluation")
        return 0
    errors = validate_inventory(args.job_dir, args.config)
    if args.readiness_manifest is not None:
        errors.extend(cohort_readiness_errors(args.readiness_manifest))
    if errors:
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        return 1
    scenarios = load_scenarios(args.config)
    print(f"OK: {len(scenarios)} scenario scripts, MDE script, and cohort scripts match {args.config.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
