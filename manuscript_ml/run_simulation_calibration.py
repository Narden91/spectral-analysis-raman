"""Step 2: Known-factor synthetic simulations and bootstrap interval calibration.

Runs the synthetic scenarios listed under ``simulation`` in ``config/manuscript_ml.yaml``
(see ``docs/research_design.md``) across the configured participant-count grid, then checks
bootstrap interval coverage against the provisional feasibility gate.
"""

from __future__ import annotations

import argparse
import sys
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
from src.analysis.paired_lipid.simulation import (  # noqa: E402
    SeedSet,
    _ldl_factor_recovery_r2,
    check_gate_attainable,
    evaluate_bootstrap_calibration,
    evaluate_scenario_calibration_suite,
    generate_synthetic_cohort,
    run_synthetic_scenario_suite,
)

__all__ = [
    "SeedSet",
    "_ldl_factor_recovery_r2",
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
) -> dict[str, Path]:
    """Run the synthetic scenario suite and bootstrap interval calibration (calibration uses n_participants)."""
    target_dir = ensure_output_dir(output_dir)
    cfg, raw_cfg = load_manuscript_config(config_path)
    seeds = raw_cfg["seeds"]
    sim_cfg = raw_cfg["simulation"]
    cal_cfg = raw_cfg["calibration"]
    n_grid = [int(n) for n in sim_cfg["n_participants_grid"]]
    compact_grid = bool(sim_cfg["compact_grid"])
    if quick:
        n_scenario_reps = min(n_scenario_reps, 3)
        n_calibration_reps = min(n_calibration_reps, 4)
        n_bootstraps = min(n_bootstraps, 10)
        n_participants = min(n_participants, 36)
        n_grid = [n_participants]
    # Fail before the scenario suite, not after it.
    check_gate_attainable(n_calibration_reps, seed_set, float(cal_cfg["gate_wilson_lower_min"]))

    scenarios = list(sim_cfg["scenarios"])
    scenarios_path = target_dir / "simulation_scenarios_summary.csv"
    scenarios_df = run_synthetic_scenario_suite(
        scenarios=scenarios,
        n_participants_grid=n_grid,
        n_repetitions=n_scenario_reps,
        compact_grid=compact_grid,
        base_seed=int(seeds["scenario_base_seed"]),
        n_jobs=n_jobs,
        cfg=cfg,
    )
    write_csv_atomic(scenarios_df, scenarios_path, index=False)

    if calibrate_all_scenarios:
        cal_df = evaluate_scenario_calibration_suite(
            scenarios=scenarios,
            n_repetitions=n_calibration_reps,
            n_bootstraps=n_bootstraps,
            n_participants=n_participants,
            base_seed=int(seeds["calibration_base_seed"]),
            cfg=cfg,
            n_jobs=n_jobs,
            use_gpu=use_gpu,
            n_test_participants=int(cal_cfg["n_test_participants"]),
            seed_set=seed_set,
            compact_grid=compact_grid,
            gate_wilson_lower_min=float(cal_cfg["gate_wilson_lower_min"]),
            gate_max_failure_rate=float(cal_cfg["gate_max_failure_rate"]),
        )
    else:
        cal_summary = evaluate_bootstrap_calibration(
            n_repetitions=n_calibration_reps,
            n_bootstraps=n_bootstraps,
            n_participants=n_participants,
            signal_strength=float(cal_cfg["signal_strength"]),
            base_seed=int(seeds["calibration_base_seed"]),
            cfg=cfg,
            n_jobs=n_jobs,
            use_gpu=use_gpu,
            n_test_participants=int(cal_cfg["n_test_participants"]),
            seed_set=seed_set,
            compact_grid=compact_grid,
            gate_wilson_lower_min=float(cal_cfg["gate_wilson_lower_min"]),
            gate_max_failure_rate=float(cal_cfg["gate_max_failure_rate"]),
        )
        cal_df = pd.DataFrame([cal_summary])

    cal_path = target_dir / "simulation_calibration_report.csv"
    write_csv_atomic(cal_df, cal_path, index=False)
    prov_path = write_run_provenance(
        target_dir,
        "simulation_calibration",
        config_path=config_path,
        cfg=cfg,
        settings={
            "n_scenario_reps": n_scenario_reps,
            "n_participants_grid": n_grid,
            "compact_grid": compact_grid,
            "n_calibration_reps": n_calibration_reps,
            "n_bootstraps": n_bootstraps,
            "n_calibration_participants": n_participants,
            "scenario_base_seed": int(seeds["scenario_base_seed"]),
            "calibration_base_seed": int(seeds["calibration_base_seed"]),
            "calibration_seed_set": seed_set,
            "calibrate_all_scenarios": calibrate_all_scenarios,
            "use_gpu": use_gpu,
        },
    )
    return {
        "run_provenance_simulation_calibration": prov_path,
        "simulation_scenarios_summary": scenarios_path,
        "simulation_calibration_report": cal_path,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run known-factor synthetic simulations and bootstrap calibration.")
    add_common_stage_args(parser)
    parser.add_argument("--quick", action="store_true", help="Run fast smoke-test repetition counts.")
    parser.add_argument("--scenario-reps", type=int, default=10)
    parser.add_argument("--calibration-reps", type=int, default=20)
    parser.add_argument("--bootstraps", type=int, default=100)
    parser.add_argument("--participants", type=int, default=92, help="Participants per calibration replicate.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel CPU worker count (-1 for all cores).")
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
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
    )
    print_stage_outputs("run_simulation_calibration", outputs)


if __name__ == "__main__":
    main()
