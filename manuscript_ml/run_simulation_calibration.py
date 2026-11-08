"""Step 2: Known-factor synthetic simulations and bootstrap interval calibration.

Runs the synthetic scenarios listed under ``simulation`` in ``config/manuscript_ml.yaml``
(see ``docs/research_design.md``) across the configured participant-count grid, then checks
bootstrap interval coverage against the provisional feasibility gate.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manuscript_ml import ensure_output_dir, load_manuscript_config, write_run_provenance
from src.analysis.io_utils import write_csv_atomic
from src.analysis.paired_lipid.cohort import PairedLipidConfig
from src.analysis.paired_lipid.nested_cv import evaluate_nested_cohort
from src.analysis.paired_lipid.simulation import (
    evaluate_bootstrap_calibration,
    generate_synthetic_cohort,
)

_SCENARIO_META_KEYS = ("scenario", "covariate_preset")


def _ldl_factor_recovery_r2(eval_res: dict[str, Any], z_ldl: np.ndarray) -> float:
    """Mean over outer folds of the held-out R^2 of the true LDL-C factor on the learned paired scores."""
    r2s = []
    for te_idx, scores in eval_res["fold_test_paired_scores"]:
        y = z_ldl[te_idx]
        X = np.column_stack([np.ones(te_idx.size), scores])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2s.append(1.0 - float(np.sum((y - X @ beta) ** 2)) / ss_tot if ss_tot > 0 else float("nan"))
    return float(np.nanmean(r2s))


def _eval_single_scenario_rep(
    seed: int,
    n_participants: int,
    compact_grid: bool,
    spec: dict[str, Any],
    cfg: PairedLipidConfig,
) -> tuple[float, float, float, float, float, float | None, float, float]:
    gen_kwargs = {k: v for k, v in spec.items() if k not in _SCENARIO_META_KEYS}
    ds, truth = generate_synthetic_cohort(
        n_participants=n_participants, seed=seed, compact_grid=compact_grid, **gen_kwargs
    )
    res = evaluate_nested_cohort(ds, cfg=cfg)
    comp = res["comparisons_vs_covariate_ridge"]
    fold_iccs = [
        float(r["paired_spectral_icc"]["icc_a1"])
        for r in res["fold_repeatability"]
        if "paired_spectral_icc" in r and r["paired_spectral_icc"]["estimable"]
    ]
    mean_icc = float(np.mean(fold_iccs)) if fold_iccs else None
    return (
        float(comp["paired_linear_ridge"]["delta_mse"]),
        float(comp["pca_ridge"]["delta_mse"]),
        float(comp["pls_regression"]["delta_mse"]),
        float(comp["fixed_band_ridge"]["delta_mse"]),
        float(res["paired_vs_strongest_comparator"]["delta_mse"]),
        mean_icc,
        _ldl_factor_recovery_r2(res, truth["z_bio"][:, 0]),
        float(np.corrcoef(ds.outcomes["ldl_mmol_l"], ds.outcomes["triglycerides_mmol_l"])[0, 1]),
    )


def run_synthetic_scenario_suite(
    *,
    scenarios: list[dict[str, Any]],
    n_participants_grid: list[int],
    n_repetitions: int = 5,
    compact_grid: bool = True,
    base_seed: int = 1000,
    n_jobs: int = -1,
    cfg: PairedLipidConfig = PairedLipidConfig(),
) -> pd.DataFrame:
    """Evaluate every configured synthetic scenario at every participant count across repetitions."""
    tasks = [
        (
            s_idx,
            n_idx,
            base_seed + 1_000_000 * n_idx + 10_000 * s_idx + rep,
            spec,
            replace(cfg, covariate_preset=str(spec["covariate_preset"])),
        )
        for n_idx in range(len(n_participants_grid))
        for s_idx, spec in enumerate(scenarios)
        for rep in range(n_repetitions)
    ]

    def _run(task: tuple[int, int, int, dict[str, Any], PairedLipidConfig]) -> tuple:
        _, n_idx, seed, spec, task_cfg = task
        return _eval_single_scenario_rep(seed, n_participants_grid[n_idx], compact_grid, spec, task_cfg)

    if n_jobs != 1 and len(tasks) >= 120:
        from joblib import Parallel, delayed

        eff_jobs = min(8, max(1, (os.cpu_count() or 4) // 2)) if n_jobs < 0 else n_jobs
        task_results = Parallel(n_jobs=eff_jobs)(delayed(_run)(t) for t in tasks)
    else:
        task_results = [_run(t) for t in tasks]

    grouped: dict[tuple[int, int], list[tuple]] = {}
    for (s_idx, n_idx, *_), result in zip(tasks, task_results):
        grouped.setdefault((n_idx, s_idx), []).append(result)

    rows: list[dict[str, Any]] = []
    for (n_idx, s_idx), s_res in grouped.items():
        spec = scenarios[s_idx]
        deltas_paired = [r[0] for r in s_res]
        spec_iccs = [r[5] for r in s_res if r[5] is not None]
        rows.append(
            {
                "scenario": spec["scenario"],
                "covariate_preset": spec["covariate_preset"],
                "n_repetitions": n_repetitions,
                "n_participants": n_participants_grid[n_idx],
                "mean_delta_mse_paired": float(np.mean(deltas_paired)),
                "sd_delta_mse_paired": float(np.std(deltas_paired, ddof=1)) if n_repetitions > 1 else 0.0,
                "mean_delta_vs_strongest_comparator": float(np.mean([r[4] for r in s_res])),
                "mean_delta_mse_pca": float(np.mean([r[1] for r in s_res])),
                "mean_delta_mse_pls": float(np.mean([r[2] for r in s_res])),
                "mean_delta_mse_fixed_band": float(np.mean([r[3] for r in s_res])),
                "mean_spectral_icc_a1": float(np.mean(spec_iccs)) if spec_iccs else float("nan"),
                "mean_recovery_r2_ldl_factor": float(np.nanmean([r[6] for r in s_res])),
                "mean_corr_ldl_tg": float(np.mean([r[7] for r in s_res])),
            }
        )
    return pd.DataFrame(rows)


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
    seed_set: str = "development",
) -> dict[str, Path]:
    """Run the synthetic scenario suite and bootstrap interval calibration (calibration uses n_participants)."""
    target_dir = ensure_output_dir(output_dir)
    cfg, raw_cfg = load_manuscript_config(config_path)
    seeds = raw_cfg["seeds"]
    sim_cfg = raw_cfg["simulation"]
    n_grid = [int(n) for n in sim_cfg["n_participants_grid"]]
    compact_grid = bool(sim_cfg["compact_grid"])
    if quick:
        n_scenario_reps = min(n_scenario_reps, 3)
        n_calibration_reps = min(n_calibration_reps, 4)
        n_bootstraps = min(n_bootstraps, 10)
        n_participants = min(n_participants, 36)
        n_grid = [n_participants]

    scenarios_df = run_synthetic_scenario_suite(
        scenarios=list(sim_cfg["scenarios"]),
        n_participants_grid=n_grid,
        n_repetitions=n_scenario_reps,
        compact_grid=compact_grid,
        base_seed=int(seeds["scenario_base_seed"]),
        n_jobs=n_jobs,
        cfg=cfg,
    )
    cal_summary = evaluate_bootstrap_calibration(
        n_repetitions=n_calibration_reps,
        n_bootstraps=n_bootstraps,
        n_participants=n_participants,
        signal_strength=0.85,
        base_seed=int(seeds["calibration_base_seed"]),
        cfg=cfg,
        n_jobs=n_jobs,
        use_gpu=use_gpu,
        seed_set=seed_set,
        compact_grid=compact_grid,
    )
    cal_df = pd.DataFrame([cal_summary])

    scenarios_path = target_dir / "simulation_scenarios_summary.csv"
    cal_path = target_dir / "simulation_calibration_report.csv"
    write_csv_atomic(scenarios_df, scenarios_path)
    write_csv_atomic(cal_df, cal_path)
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
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--quick", action="store_true", help="Run fast smoke-test repetition counts.")
    parser.add_argument("--scenario-reps", type=int, default=10)
    parser.add_argument("--calibration-reps", type=int, default=20)
    parser.add_argument("--bootstraps", type=int, default=100)
    parser.add_argument("--participants", type=int, default=92, help="Participants per calibration replicate.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel CPU worker count (-1 for all cores).")
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument("--use-gpu", dest="use_gpu", action="store_true", default=None, help="Force GPU execution.")
    gpu_group.add_argument("--no-gpu", dest="use_gpu", action="store_false", help="Force CPU execution.")
    parser.add_argument("--config", type=str, default=None, help="Experiment settings YAML.")
    parser.add_argument(
        "--seed-set",
        choices=("development", "validation"),
        default="development",
        help="Calibration replicate seeds; only 'validation' gives a confirmatory gate result.",
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
    )
    for key, path in outputs.items():
        print(f"[run_simulation_calibration] wrote {key}: {path}")


if __name__ == "__main__":
    main()
