"""Concatenate per-scenario simulation reports after the cluster jobs finish.

The MDE directory is copied separately. Its single-setting calibration report is
not part of the scenario calibration table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slurm_jobs.check_job_inventory import CONFIG_PATH, load_scenarios  # noqa: E402

DEFAULT_VALIDATION_ROOT = REPO_ROOT / "Results_manuscript_ml" / "validation"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Results_manuscript_ml" / "validation_merged"


def _read_scenario_csv(path: Path, scenario: str, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label} for {scenario}: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Empty {label} for {scenario}: {path}")
    if "scenario" not in frame.columns:
        frame.insert(0, "scenario", scenario)
    else:
        observed = set(frame["scenario"].astype(str))
        if observed != {scenario}:
            raise ValueError(f"{path} scenario column is {sorted(observed)}, expected {{{scenario}}}")
    frame.insert(0, "source_dir", str(path.parent))
    return frame


def merge_simulation_reports(
    validation_root: Path,
    output_dir: Path,
    *,
    config_path: Path = CONFIG_PATH,
    allow_partial: bool = False,
) -> dict[str, Path]:
    scenarios = load_scenarios(config_path)
    missing: list[str] = []
    calibration_frames: list[pd.DataFrame] = []
    suite_frames: list[pd.DataFrame] = []
    sources: list[dict[str, str]] = []

    for scenario in scenarios:
        scenario_dir = validation_root / scenario
        calibration_path = scenario_dir / "simulation_calibration_report.csv"
        suite_path = scenario_dir / "simulation_scenarios_summary.csv"
        if not calibration_path.is_file() or not suite_path.is_file():
            missing.append(scenario)
            continue
        calibration_frames.append(_read_scenario_csv(calibration_path, scenario, "calibration report"))
        suite_frames.append(_read_scenario_csv(suite_path, scenario, "scenario suite"))
        provenance = scenario_dir / "run_provenance_simulation_calibration.json"
        sources.append(
            {
                "scenario": scenario,
                "directory": str(scenario_dir),
                "provenance": str(provenance) if provenance.is_file() else "",
            }
        )

    if missing and not allow_partial:
        raise FileNotFoundError(
            "Refusing to merge a partial campaign. Missing scenarios: " + ", ".join(missing)
        )
    if not calibration_frames:
        raise FileNotFoundError(f"No completed scenario reports under {validation_root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    calibration = pd.concat(calibration_frames, ignore_index=True)
    suite = pd.concat(suite_frames, ignore_index=True)
    calibration_path = output_dir / "simulation_calibration_report.csv"
    suite_path = output_dir / "simulation_scenarios_summary.csv"
    calibration.to_csv(calibration_path, index=False)
    suite.to_csv(suite_path, index=False)

    mde_source = validation_root / "mde_signal_curve" / "simulation_mde_curve.csv"
    mde_target = output_dir / "simulation_mde_curve.csv"
    if mde_source.is_file():
        mde_target.write_bytes(mde_source.read_bytes())
    elif mde_target.exists():
        mde_target.unlink()

    manifest = {
        "validation_root": str(validation_root),
        "n_scenarios_expected": len(scenarios),
        "n_scenarios_merged": len(sources),
        "missing_scenarios": missing,
        "sources": sources,
        "mde_curve_present": mde_source.is_file(),
        "note": (
            "Concatenation of independently submitted scenario jobs. "
            "Per-scenario provenance remains in each source directory. "
            "The MDE directory's single-setting calibration report is excluded."
        ),
    }
    manifest_path = output_dir / "merge_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "simulation_calibration_report": calibration_path,
        "simulation_scenarios_summary": suite_path,
        "merge_manifest": manifest_path,
        **({"simulation_mde_curve": mde_target} if mde_source.is_file() else {}),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Merge per-scenario manuscript ML simulation reports.")
    parser.add_argument("--validation-root", type=Path, default=DEFAULT_VALIDATION_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Merge completed scenarios and list the missing ones.",
    )
    args = parser.parse_args(argv)
    try:
        outputs = merge_simulation_reports(
            args.validation_root,
            args.output_dir,
            config_path=args.config,
            allow_partial=args.allow_partial,
        )
    except (OSError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    for key, path in outputs.items():
        print(f"wrote {key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
