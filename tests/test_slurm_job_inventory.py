"""Inventory and merge checks for the manuscript ML Slurm scripts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from slurm_jobs.check_job_inventory import cohort_readiness_errors, validate_inventory
from slurm_jobs.merge_simulation_reports import merge_simulation_reports

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_slurm_scripts_match_locked_config() -> None:
    errors = validate_inventory()
    assert errors == []


@pytest.mark.parametrize(
    "helper",
    ["submit_cohort.sh", "submit_03_cohort_primary.sh", "submit_04_cohort_sensitivity.sh"],
)
def test_cohort_submit_helpers_refuse_without_approval(helper: str) -> None:
    result = subprocess.run(
        ["bash", f"slurm_jobs/{helper}", "--dry-run"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "--approve" in result.stderr


def test_merge_requires_every_scenario_unless_partial(tmp_path: Path) -> None:
    validation = tmp_path / "validation"
    scenario_dir = validation / "linear_biological_signal"
    scenario_dir.mkdir(parents=True)
    pd.DataFrame({"scenario": ["linear_biological_signal"], "coverage": [0.95]}).to_csv(
        scenario_dir / "simulation_calibration_report.csv", index=False
    )
    pd.DataFrame({"scenario": ["linear_biological_signal"], "n_participants": [92]}).to_csv(
        scenario_dir / "simulation_scenarios_summary.csv", index=False
    )
    mde_dir = validation / "mde_signal_curve"
    mde_dir.mkdir()
    pd.DataFrame({"signal_strength": [0.0, 1.0]}).to_csv(mde_dir / "simulation_mde_curve.csv", index=False)
    # A single-setting calibration file in the MDE directory must not enter the scenario table.
    pd.DataFrame({"scenario": ["single_setting"], "coverage": [0.5]}).to_csv(
        mde_dir / "simulation_calibration_report.csv", index=False
    )

    try:
        merge_simulation_reports(validation, tmp_path / "merged")
    except FileNotFoundError as exc:
        assert "Missing scenarios" in str(exc)
    else:
        raise AssertionError("partial merge should fail closed")

    outputs = merge_simulation_reports(validation, tmp_path / "merged", allow_partial=True)
    calibration = pd.read_csv(outputs["simulation_calibration_report"])
    assert list(calibration["scenario"]) == ["linear_biological_signal"]
    assert "single_setting" not in set(calibration["scenario"])
    mde = pd.read_csv(outputs["simulation_mde_curve"])
    assert list(mde["signal_strength"]) == [0.0, 1.0]


def test_merge_rejects_mismatched_scenario_column(tmp_path: Path) -> None:
    validation = tmp_path / "validation"
    scenario_dir = validation / "linear_biological_signal"
    scenario_dir.mkdir(parents=True)
    pd.DataFrame({"scenario": ["other_scenario"], "coverage": [0.95]}).to_csv(
        scenario_dir / "simulation_calibration_report.csv", index=False
    )
    pd.DataFrame({"scenario": ["linear_biological_signal"], "n_participants": [92]}).to_csv(
        scenario_dir / "simulation_scenarios_summary.csv", index=False
    )
    try:
        merge_simulation_reports(validation, tmp_path / "merged", allow_partial=True)
    except ValueError as exc:
        assert "scenario column" in str(exc)
    else:
        raise AssertionError("mismatched scenario column should fail")


def test_readiness_manifest_reports_join_failure(tmp_path: Path) -> None:
    manifest = tmp_path / "readiness_manifest.json"
    manifest.write_text(
        '{"measurement_readiness": {"ready_for_cohort_evaluation": false, '
        '"cohort_join": {"status": "failed", "error": "Store is stale"}}}',
        encoding="utf-8",
    )
    errors = cohort_readiness_errors(manifest)
    assert errors
    assert "not ready" in errors[0]


def test_check_job_inventory_cli() -> None:
    result = subprocess.run(
        [sys.executable, "slurm_jobs/check_job_inventory.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "17 scenario scripts" in result.stdout
