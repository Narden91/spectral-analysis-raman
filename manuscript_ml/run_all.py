"""Unified orchestrator for all manuscript_ml experiment stages."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manuscript_ml import ensure_output_dir
from manuscript_ml.audit_readiness import run_readiness_audit
from manuscript_ml.run_primary_evaluation import run_primary_evaluation
from manuscript_ml.run_sensitivity_and_secondary import run_sensitivity_and_secondary
from manuscript_ml.run_simulation_calibration import run_simulation_and_calibration


def run_all_pipeline(
    *,
    use_synthetic: bool = False,
    approve_cohort_evaluation: bool = False,
    quick: bool = False,
    output_dir: str | Path | None = None,
    n_bootstraps: int = 0,
    config_path: str | Path | None = None,
    include_simulation_calibration: bool = False,
) -> dict[str, Path]:
    """Run readiness and cohort analyses, with calibration only by explicit request."""
    if not use_synthetic and not approve_cohort_evaluation:
        raise PermissionError("Cohort evaluation requires explicit '--approve-cohort-evaluation'")
    target_dir = ensure_output_dir(output_dir)
    all_outputs: dict[str, Path] = {}

    all_outputs.update(run_readiness_audit(output_dir=target_dir, config_path=config_path))
    if use_synthetic or include_simulation_calibration:
        all_outputs.update(run_simulation_and_calibration(output_dir=target_dir, quick=quick, config_path=config_path))
    all_outputs.update(
        run_primary_evaluation(
            use_synthetic=use_synthetic,
            approve_cohort_evaluation=approve_cohort_evaluation,
            output_dir=target_dir,
            n_bootstraps=n_bootstraps,
            compact_grid=quick,
            config_path=config_path,
        )
    )
    all_outputs.update(
        run_sensitivity_and_secondary(
            use_synthetic=use_synthetic,
            approve_cohort_evaluation=approve_cohort_evaluation,
            output_dir=target_dir,
            compact_grid=quick,
            config_path=config_path,
        )
    )
    return all_outputs


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all manuscript_ml stages.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--cohort", action="store_true", help="Evaluate on the real stroke cohort (default).")
    mode.add_argument("--synthetic", action="store_true", help="Explicit development simulation; not cohort evidence.")
    parser.add_argument("--approve-cohort-evaluation", action="store_true")
    parser.add_argument("--include-simulation-calibration", action="store_true", help="Run explicitly labelled simulation calibration separately from cohort evidence.")
    parser.add_argument("--quick", action="store_true", help="Use compact grids and quick repetition counts.")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--bootstraps", type=int, default=0)
    parser.add_argument("--config", type=str, default=None, help="Experiment settings YAML.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    outputs = run_all_pipeline(
        use_synthetic=args.synthetic,
        approve_cohort_evaluation=args.approve_cohort_evaluation,
        quick=args.quick,
        output_dir=args.output_dir,
        n_bootstraps=args.bootstraps,
        config_path=args.config,
        include_simulation_calibration=args.include_simulation_calibration,
    )
    for key, path in outputs.items():
        print(f"[run_all] written artifact {key}: {path}")


if __name__ == "__main__":
    main()

