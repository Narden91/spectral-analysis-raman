"""Unified orchestrator for all manuscript_ml experiment stages."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

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
from src.analysis.paired_lipid.nested_cv import INTERVAL_METHODS, IntervalMethod  # noqa: E402
from src.analysis.paired_lipid.simulation import SeedSet  # noqa: E402


def run_all_pipeline(
    *,
    use_synthetic: bool = False,
    approve_cohort_evaluation: bool = False,
    clinical_master_path: str | Path | None = None,
    quick: bool = False,
    compact_grid: bool | None = None,
    output_dir: str | Path | None = None,
    n_bootstraps: int = 0,
    sensitivity_bootstraps: int = 0,
    config_path: str | Path | None = None,
    include_simulation_calibration: bool = False,
    n_scenario_reps: int = 10,
    n_calibration_reps: int = 20,
    n_participants: int = 92,
    n_test_participants: int | None = None,
    scenario_names: Sequence[str] | None = None,
    skip_scenario_suite: bool = False,
    run_mde_curve: bool = False,
    resume: bool = False,
    checkpoint_chunk_size: int | None = None,
    seed_set: SeedSet = "development",
    calibrate_all_scenarios: bool = False,
    n_jobs: int = -1,
    use_gpu: bool | None = None,
    batch_size: int | None = None,
    interval_method: IntervalMethod | None = None,
) -> dict[str, Path]:
    """Run readiness and cohort analyses, with calibration only by explicit request."""
    require_cohort_approval(
        use_synthetic=use_synthetic,
        approve_cohort_evaluation=approve_cohort_evaluation,
    )
    target_dir = ensure_output_dir(output_dir)
    # Preserve explicit --cohort-grid (False) vs --compact-grid (True) vs config default (None).
    sim_compact: bool | None = (True if quick else None) if compact_grid is None else bool(compact_grid)
    cohort_compact = bool(quick) if compact_grid is None else bool(compact_grid)
    clinical_kw = {"clinical_master_path": clinical_master_path} if clinical_master_path is not None else {}
    all_outputs: dict[str, Path] = {}

    all_outputs.update(run_readiness_audit(output_dir=target_dir, config_path=config_path, **clinical_kw))
    if use_synthetic or include_simulation_calibration:
        sim_kw = {"n_bootstraps": n_bootstraps} if n_bootstraps > 0 else {}
        all_outputs.update(
            run_simulation_and_calibration(
                output_dir=target_dir,
                quick=quick,
                n_scenario_reps=n_scenario_reps,
                n_calibration_reps=n_calibration_reps,
                n_participants=n_participants,
                n_test_participants=n_test_participants,
                scenario_names=scenario_names,
                skip_scenario_suite=skip_scenario_suite,
                run_mde_curve=run_mde_curve,
                resume=resume,
                checkpoint_chunk_size=checkpoint_chunk_size,
                n_jobs=n_jobs,
                use_gpu=use_gpu,
                config_path=config_path,
                seed_set=seed_set,
                calibrate_all_scenarios=calibrate_all_scenarios,
                compact_grid=sim_compact,
                interval_method=interval_method,
                **sim_kw,
            )
        )
    primary_state: dict[str, Any] = {}
    all_outputs.update(
        run_primary_evaluation(
            use_synthetic=use_synthetic,
            approve_cohort_evaluation=approve_cohort_evaluation,
            output_dir=target_dir,
            n_bootstraps=n_bootstraps,
            compact_grid=cohort_compact,
            config_path=config_path,
            use_gpu=use_gpu,
            batch_size=batch_size,
            interval_method=interval_method,
            n_jobs=n_jobs,
            state_out=primary_state,
            **clinical_kw,
        )
    )
    all_outputs.update(
        run_sensitivity_and_secondary(
            use_synthetic=use_synthetic,
            approve_cohort_evaluation=approve_cohort_evaluation,
            output_dir=target_dir,
            compact_grid=cohort_compact,
            config_path=config_path,
            n_bootstraps=sensitivity_bootstraps,
            interval_method=interval_method,
            use_gpu=use_gpu,
            batch_size=batch_size,
            n_jobs=n_jobs,
            primary_state=primary_state,
            **clinical_kw,
        )
    )
    return all_outputs


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all manuscript_ml stages.")
    add_cohort_stage_args(parser, include_compact_grid=False)
    parser.set_defaults(clinical_master=None)
    grid_group = parser.add_mutually_exclusive_group()
    grid_group.add_argument(
        "--compact-grid",
        dest="compact_grid",
        action="store_true",
        default=None,
        help="Force 10 cm^-1 synthetic grid across stages.",
    )
    grid_group.add_argument(
        "--cohort-grid",
        dest="compact_grid",
        action="store_false",
        help="Force 1 cm^-1 cohort spectral grid (1,802 channels) across stages.",
    )
    parser.add_argument(
        "--include-simulation-calibration",
        action="store_true",
        help="Run explicitly labelled simulation calibration separately from cohort evidence.",
    )
    parser.add_argument("--quick", action="store_true", help="Use compact grids and quick repetition counts.")
    parser.add_argument("--bootstraps", type=int, default=0)
    parser.add_argument(
        "--sensitivity-bootstraps",
        type=int,
        default=0,
        help="Optional bootstrap draws per covariate/subset sensitivity setting.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Optional batched bootstrap chunk size.")
    parser.add_argument(
        "--interval-method",
        choices=INTERVAL_METHODS,
        default=None,
        help="Bootstrap interval estimator ('studentized_t' or 'percentile'); defaults to config.",
    )
    parser.add_argument("--scenario-reps", type=int, default=10)
    parser.add_argument("--calibration-reps", type=int, default=20)
    parser.add_argument("--participants", type=int, default=92, help="Participants per calibration replicate.")
    parser.add_argument(
        "--n-test-participants",
        type=int,
        default=None,
        help="Override calibration.n_test_participants from config.",
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument("--use-gpu", dest="use_gpu", action="store_true", default=None, help="Force GPU execution.")
    gpu_group.add_argument("--no-gpu", dest="use_gpu", action="store_false", help="Force CPU execution.")
    parser.add_argument(
        "--seed-set",
        choices=("development", "validation"),
        default="development",
        help="Calibration replicate seeds; only 'validation' gives a confirmatory gate result.",
    )
    parser.add_argument(
        "--calibrate-all-scenarios",
        action="store_true",
        help="Run bootstrap interval calibration across every scenario in simulation.scenarios.",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default=None,
        help="Optional comma-separated subset of scenario names (preserves canonical seed offsets).",
    )
    parser.add_argument(
        "--skip-scenario-suite",
        action="store_true",
        help="Skip simulation_scenarios_summary.csv and run only bootstrap interval calibration.",
    )
    parser.add_argument(
        "--run-mde-curve",
        action="store_true",
        help="Evaluate pre-specified minimum detectable effect curve (simulation_mde_curve.csv).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Save and resume chunked calibration checkpoints under <output_dir>/calibration_checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-chunk-size",
        type=int,
        default=None,
        help="Number of calibration replicates per checkpoint flush when --resume is enabled.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    scenario_list = (
        [tok.strip() for tok in args.scenarios.split(",") if tok.strip()]
        if args.scenarios is not None
        else None
    )
    outputs = run_all_pipeline(
        use_synthetic=args.synthetic,
        approve_cohort_evaluation=args.approve_cohort_evaluation,
        clinical_master_path=args.clinical_master,
        quick=args.quick,
        compact_grid=args.compact_grid,
        output_dir=args.output_dir,
        n_bootstraps=args.bootstraps,
        sensitivity_bootstraps=args.sensitivity_bootstraps,
        config_path=args.config,
        include_simulation_calibration=args.include_simulation_calibration,
        n_scenario_reps=args.scenario_reps,
        n_calibration_reps=args.calibration_reps,
        n_participants=args.participants,
        n_test_participants=args.n_test_participants,
        scenario_names=scenario_list,
        skip_scenario_suite=args.skip_scenario_suite,
        run_mde_curve=args.run_mde_curve,
        resume=args.resume,
        checkpoint_chunk_size=args.checkpoint_chunk_size,
        seed_set=args.seed_set,
        calibrate_all_scenarios=args.calibrate_all_scenarios,
        n_jobs=args.n_jobs,
        use_gpu=args.use_gpu,
        batch_size=args.batch_size,
        interval_method=args.interval_method,
    )
    print_stage_outputs("run_all", outputs)


if __name__ == "__main__":
    main()
