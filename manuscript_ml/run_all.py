"""Unified orchestrator for all manuscript_ml experiment stages."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manuscript_ml import (  # noqa: E402
    add_cohort_stage_args,
    ensure_output_dir,
    print_stage_outputs,
    require_cohort_approval,
)
from manuscript_ml.audit_readiness import run_readiness_audit  # noqa: E402
from manuscript_ml.run_primary_evaluation import run_primary_evaluation  # noqa: E402
from manuscript_ml.run_sensitivity_and_secondary import run_sensitivity_and_secondary  # noqa: E402
from manuscript_ml.run_simulation_calibration import run_simulation_and_calibration  # noqa: E402


def run_all_pipeline(
    *,
    use_synthetic: bool = False,
    approve_cohort_evaluation: bool = False,
    clinical_master_path: str | Path | None = None,
    quick: bool = False,
    compact_grid: bool = False,
    output_dir: str | Path | None = None,
    n_bootstraps: int = 0,
    config_path: str | Path | None = None,
    include_simulation_calibration: bool = False,
) -> dict[str, Path]:
    """Run readiness and cohort analyses, with calibration only by explicit request."""
    require_cohort_approval(
        use_synthetic=use_synthetic,
        approve_cohort_evaluation=approve_cohort_evaluation,
    )
    target_dir = ensure_output_dir(output_dir)
    eff_compact = bool(quick or compact_grid)
    clinical_kw = {"clinical_master_path": clinical_master_path} if clinical_master_path is not None else {}
    all_outputs: dict[str, Path] = {}

    all_outputs.update(run_readiness_audit(output_dir=target_dir, config_path=config_path, **clinical_kw))
    if use_synthetic or include_simulation_calibration:
        sim_kw = {"n_bootstraps": n_bootstraps} if n_bootstraps > 0 else {}
        all_outputs.update(
            run_simulation_and_calibration(
                output_dir=target_dir,
                quick=quick,
                config_path=config_path,
                **sim_kw,
            )
        )
    all_outputs.update(
        run_primary_evaluation(
            use_synthetic=use_synthetic,
            approve_cohort_evaluation=approve_cohort_evaluation,
            output_dir=target_dir,
            n_bootstraps=n_bootstraps,
            compact_grid=eff_compact,
            config_path=config_path,
            **clinical_kw,
        )
    )
    all_outputs.update(
        run_sensitivity_and_secondary(
            use_synthetic=use_synthetic,
            approve_cohort_evaluation=approve_cohort_evaluation,
            output_dir=target_dir,
            compact_grid=eff_compact,
            config_path=config_path,
            **clinical_kw,
        )
    )
    return all_outputs


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all manuscript_ml stages.")
    add_cohort_stage_args(parser)
    parser.set_defaults(clinical_master=None)
    parser.add_argument(
        "--include-simulation-calibration",
        action="store_true",
        help="Run explicitly labelled simulation calibration separately from cohort evidence.",
    )
    parser.add_argument("--quick", action="store_true", help="Use compact grids and quick repetition counts.")
    parser.add_argument("--bootstraps", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    outputs = run_all_pipeline(
        use_synthetic=args.synthetic,
        approve_cohort_evaluation=args.approve_cohort_evaluation,
        clinical_master_path=args.clinical_master,
        quick=args.quick,
        compact_grid=args.compact_grid,
        output_dir=args.output_dir,
        n_bootstraps=args.bootstraps,
        config_path=args.config,
        include_simulation_calibration=args.include_simulation_calibration,
    )
    print_stage_outputs("run_all", outputs)


if __name__ == "__main__":
    main()
