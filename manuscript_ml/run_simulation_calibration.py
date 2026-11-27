"""Step 2: Known-factor synthetic simulations and bootstrap interval calibration.

Runs the synthetic scenarios listed under ``simulation`` in ``config/manuscript_ml.yaml``
(see ``docs/research_design.md``) across the configured participant-count grid, then checks
bootstrap interval coverage against the provisional feasibility gate.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manuscript_ml import (  # noqa: E402
    add_common_stage_args,
    ensure_output_dir,
    load_manuscript_config,
    print_stage_outputs,
    write_run_provenance,
)
from src.analysis.io_utils import write_csv_atomic  # noqa: E402
from src.analysis.paired_lipid.nested_cv import INTERVAL_METHODS, IntervalMethod  # noqa: E402
from src.analysis.paired_lipid.simulation import (  # noqa: E402
    CALIBRATION_SEED_RANGES,
    SeedSet,
    _ldl_factor_recovery_r2,
    check_gate_attainable,
    evaluate_bootstrap_calibration,
    evaluate_mde_signal_curve,
    evaluate_scenario_calibration_suite,
    generate_synthetic_cohort,
    run_synthetic_scenario_suite,
)

__all__ = [
    "IntervalMethod",
    "SeedSet",
    "_ldl_factor_recovery_r2",
    "evaluate_mde_signal_curve",
    "generate_synthetic_cohort",
    "main",
    "run_simulation_and_calibration",
    "run_synthetic_scenario_suite",
]


def run_simulation_and_calibration(
    *,
    output_dir: str | Path | None = None,
    quick: bool = False,
    n_scenario_reps: int = 10,
    n_calibration_reps: int = 20,
    n_bootstraps: int = 100,
    n_participants: int = 92,
    n_jobs: int = -1,
    use_gpu: bool | None = None,
    config_path: str | Path | None = None,
    seed_set: SeedSet = "development",
    calibrate_all_scenarios: bool = False,
    compact_grid: bool | None = None,
    n_test_participants: int | None = None,
    scenario_names: Sequence[str] | None = None,
    skip_scenario_suite: bool = False,
    resume: bool = False,
    checkpoint_chunk_size: int | None = None,
    interval_method: IntervalMethod | None = None,
    run_mde_curve: bool = False,
) -> dict[str, Path]:
    """Run the synthetic scenario suite and bootstrap interval calibration (calibration uses n_participants)."""
    target_dir = ensure_output_dir(output_dir)
    cfg, raw_cfg = load_manuscript_config(config_path)
    seeds = raw_cfg["seeds"]
    sim_cfg = raw_cfg["simulation"]
    cal_cfg = raw_cfg["calibration"]
    n_grid = [int(n) for n in sim_cfg["n_participants_grid"]]
    eff_compact = bool(sim_cfg["compact_grid"]) if compact_grid is None else bool(compact_grid)
    eff_n_test = (
        int(cal_cfg["n_test_participants"]) if n_test_participants is None else int(n_test_participants)
    )
    eff_interval_method: IntervalMethod = (
        cal_cfg.get("interval_method", "studentized_t") if interval_method is None else interval_method
    )
    if eff_interval_method not in INTERVAL_METHODS:
        raise ValueError(f"Unsupported interval_method={eff_interval_method!r}; expected one of {INTERVAL_METHODS}.")
    if quick:
        n_scenario_reps = min(n_scenario_reps, 3)
        n_calibration_reps = min(n_calibration_reps, 4)
        n_bootstraps = min(n_bootstraps, 10)
        n_participants = min(n_participants, 36)
        n_grid = [n_participants]
        if n_test_participants is None:
            eff_n_test = min(eff_n_test, 1000)
        if compact_grid is None:
            eff_compact = True
    # Fail before the scenario suite, not after it.
    if n_bootstraps < 1:
        raise ValueError(f"n_bootstraps must be >= 1 for simulation calibration; got {n_bootstraps}.")
    if n_calibration_reps < 1:
        raise ValueError(f"n_calibration_reps must be >= 1; got {n_calibration_reps}.")
    if eff_n_test < 2:
        raise ValueError(f"n_test_participants must be >= 2; got {eff_n_test}.")
    check_gate_attainable(n_calibration_reps, seed_set, float(cal_cfg["gate_wilson_lower_min"]))

    all_scenarios = list(sim_cfg["scenarios"])
    if scenario_names is not None:
        requested = [str(name).strip() for name in scenario_names if str(name).strip()]
        if not requested:
            raise ValueError("scenario_names filter cannot be empty.")
        by_name = {str(spec["scenario"]): (idx, spec) for idx, spec in enumerate(all_scenarios)}
        unknown = [name for name in requested if name not in by_name]
        if unknown:
            raise ValueError(
                f"Unknown scenario name(s): {unknown}. Available: {sorted(by_name.keys())}."
            )
        selected_pairs = [by_name[name] for name in requested]
    else:
        selected_pairs = list(enumerate(all_scenarios))
    scenario_offsets = [idx for idx, _ in selected_pairs]
    scenarios = [spec for _, spec in selected_pairs]

    cal_base_seed = int(seeds["calibration_base_seed"])
    min_rep_seed = CALIBRATION_SEED_RANGES["development"][0]
    cal_offsets = scenario_offsets if calibrate_all_scenarios else [0]
    if cal_base_seed < 0 or any(cal_base_seed + off >= min_rep_seed for off in cal_offsets):
        raise ValueError(
            f"calibration_base_seed ({cal_base_seed}) with offsets {cal_offsets} must stay in [0, {min_rep_seed})."
        )

    outputs: dict[str, Path] = {}
    scenarios_path = target_dir / "simulation_scenarios_summary.csv"
    if not skip_scenario_suite:
        scenarios_df = run_synthetic_scenario_suite(
            scenarios=scenarios,
            n_participants_grid=n_grid,
            n_repetitions=n_scenario_reps,
            compact_grid=eff_compact,
            base_seed=int(seeds["scenario_base_seed"]),
            n_jobs=n_jobs,
            cfg=cfg,
            scenario_offsets=scenario_offsets,
        )
        write_csv_atomic(scenarios_df, scenarios_path, index=False)
        outputs["simulation_scenarios_summary"] = scenarios_path

    ckpt_dir = (target_dir / "calibration_checkpoints") if resume else None
    if calibrate_all_scenarios:
        cal_df = evaluate_scenario_calibration_suite(
            scenarios=scenarios,
            n_repetitions=n_calibration_reps,
            n_bootstraps=n_bootstraps,
            n_participants=n_participants,
            base_seed=cal_base_seed,
            cfg=cfg,
            n_jobs=n_jobs,
            use_gpu=use_gpu,
            n_test_participants=eff_n_test,
            seed_set=seed_set,
            compact_grid=eff_compact,
            interval_method=eff_interval_method,
            gate_wilson_lower_min=float(cal_cfg["gate_wilson_lower_min"]),
            gate_max_failure_rate=float(cal_cfg["gate_max_failure_rate"]),
            checkpoint_dir=ckpt_dir,
            checkpoint_chunk_size=checkpoint_chunk_size,
            base_seed_offsets=scenario_offsets,
        )
    else:
        single_ckpt = (ckpt_dir / "single_setting.npz") if ckpt_dir is not None else None
        cal_summary = evaluate_bootstrap_calibration(
            n_repetitions=n_calibration_reps,
            n_bootstraps=n_bootstraps,
            n_participants=n_participants,
            signal_strength=float(cal_cfg["signal_strength"]),
            base_seed=cal_base_seed,
            cfg=cfg,
            n_jobs=n_jobs,
            use_gpu=use_gpu,
            n_test_participants=eff_n_test,
            seed_set=seed_set,
            compact_grid=eff_compact,
            interval_method=eff_interval_method,
            gate_wilson_lower_min=float(cal_cfg["gate_wilson_lower_min"]),
            gate_max_failure_rate=float(cal_cfg["gate_max_failure_rate"]),
            checkpoint_path=single_ckpt,
            checkpoint_chunk_size=checkpoint_chunk_size,
        )
        cal_df = pd.DataFrame([cal_summary])

    cal_path = target_dir / "simulation_calibration_report.csv"
    write_csv_atomic(cal_df, cal_path, index=False)
    outputs["simulation_calibration_report"] = cal_path

    mde_grid_used: list[float] | None = None
    if run_mde_curve:
        raw_grid = [float(x) for x in sim_cfg.get("mde_signal_grid", [0.0, 0.25, 0.5, 0.75, 1.0])]
        mde_grid_used = [raw_grid[0], raw_grid[-1]] if (quick and len(raw_grid) > 2) else raw_grid
        mde_base_seed = cal_base_seed + 100
        if mde_base_seed + len(mde_grid_used) >= min_rep_seed:
            raise ValueError(
                f"MDE base seed ({mde_base_seed}) overlaps development calibration seed range ({min_rep_seed})."
            )
        mde_ckpt_dir = (target_dir / "mde_checkpoints") if resume else None
        mde_df = evaluate_mde_signal_curve(
            signal_grid=mde_grid_used,
            n_repetitions=n_calibration_reps,
            n_bootstraps=n_bootstraps,
            n_participants=n_participants,
            base_seed=mde_base_seed,
            cfg=cfg,
            n_jobs=n_jobs,
            use_gpu=use_gpu,
            n_test_participants=eff_n_test,
            seed_set=seed_set,
            compact_grid=eff_compact,
            interval_method=eff_interval_method,
            gate_wilson_lower_min=float(cal_cfg["gate_wilson_lower_min"]),
            gate_max_failure_rate=float(cal_cfg["gate_max_failure_rate"]),
            checkpoint_dir=mde_ckpt_dir,
            checkpoint_chunk_size=checkpoint_chunk_size,
        )
        mde_path = target_dir / "simulation_mde_curve.csv"
        write_csv_atomic(mde_df, mde_path, index=False)
        outputs["simulation_mde_curve"] = mde_path

    prov_path = write_run_provenance(
        target_dir,
        "simulation_calibration",
        config_path=config_path,
        cfg=cfg,
        settings={
            "n_scenario_reps": n_scenario_reps,
            "n_participants_grid": n_grid,
            "compact_grid": eff_compact,
            "n_calibration_reps": n_calibration_reps,
            "n_bootstraps": n_bootstraps,
            "n_calibration_participants": n_participants,
            "n_test_participants": eff_n_test,
            "interval_method": eff_interval_method,
            "run_mde_curve": run_mde_curve,
            "mde_signal_grid": mde_grid_used,
            "scenario_base_seed": int(seeds["scenario_base_seed"]),
            "calibration_base_seed": cal_base_seed,
            "calibration_seed_set": seed_set,
            "calibrate_all_scenarios": calibrate_all_scenarios,
            "scenarios_selected": [str(s["scenario"]) for s in scenarios],
            "skip_scenario_suite": skip_scenario_suite,
            "resume": resume,
            "checkpoint_chunk_size": checkpoint_chunk_size,
            "use_gpu": use_gpu,
        },
    )
    return {
        "run_provenance_simulation_calibration": prov_path,
        **outputs,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run known-factor synthetic simulations and bootstrap calibration.")
    add_common_stage_args(parser)
    parser.add_argument("--quick", action="store_true", help="Run fast smoke-test repetition counts.")
    parser.add_argument("--scenario-reps", type=int, default=10)
    parser.add_argument("--calibration-reps", type=int, default=20)
    parser.add_argument("--bootstraps", type=int, default=100)
    parser.add_argument("--participants", type=int, default=92, help="Participants per calibration replicate.")
    parser.add_argument(
        "--n-test-participants",
        type=int,
        default=None,
        help="Override calibration.n_test_participants from config (e.g. 25000 to cut test-cohort SE).",
    )
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel CPU worker count (-1 for all cores).")
    grid_group = parser.add_mutually_exclusive_group()
    grid_group.add_argument(
        "--compact-grid",
        dest="compact_grid",
        action="store_true",
        default=None,
        help="Force 10 cm^-1 synthetic grid (fast development).",
    )
    grid_group.add_argument(
        "--cohort-grid",
        dest="compact_grid",
        action="store_false",
        help="Force 1 cm^-1 cohort spectral grid (1,802 channels) for final calibration.",
    )
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
        "--interval-method",
        choices=INTERVAL_METHODS,
        default=None,
        help="Bootstrap interval estimator ('studentized_t' or 'percentile'); defaults to config.",
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
    outputs = run_simulation_and_calibration(
        output_dir=args.output_dir,
        quick=args.quick,
        n_scenario_reps=args.scenario_reps,
        n_calibration_reps=args.calibration_reps,
        n_bootstraps=args.bootstraps,
        n_participants=args.participants,
        n_jobs=args.n_jobs,
        use_gpu=args.use_gpu,
        config_path=args.config,
        seed_set=args.seed_set,
        calibrate_all_scenarios=args.calibrate_all_scenarios,
        compact_grid=args.compact_grid,
        n_test_participants=args.n_test_participants,
        scenario_names=scenario_list,
        skip_scenario_suite=args.skip_scenario_suite,
        resume=args.resume,
        checkpoint_chunk_size=args.checkpoint_chunk_size,
        interval_method=args.interval_method,
        run_mde_curve=args.run_mde_curve,
    )
    print_stage_outputs("run_simulation_calibration", outputs)


if __name__ == "__main__":
    main()
