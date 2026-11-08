"""Step 2: Known-factor synthetic simulations and bootstrap interval calibration.

Evaluates the five core synthetic scenarios specified in ``docs/research_design.md``
(linear signal, no-association reference, nonlinear covariate proxy, systematic A/B shift
with label drift, and formula-linked LDL-C) plus bootstrap interval coverage against the
provisional feasibility gate.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manuscript_ml import ensure_output_dir
from src.analysis.io_utils import write_csv_atomic
from src.analysis.paired_lipid_representation import (
    PairedLipidConfig,
    evaluate_bootstrap_calibration,
    evaluate_nested_cohort,
    generate_synthetic_cohort,
)


def run_synthetic_scenario_suite(
    *,
    n_repetitions: int = 5,
    n_participants: int = 92,
    compact_grid: bool = True,
    base_seed: int = 1000,
) -> pd.DataFrame:
    """Evaluate all prespecified synthetic data-generating scenarios across repetitions."""
    scenarios: list[dict[str, Any]] = [
        {
            "scenario": "linear_biological_signal",
            "signal_strength": 0.85,
            "ab_shift_scale": 0.15,
            "nonlinear_covariate_proxy": False,
            "formula_linked_ldl": False,
            "label_drift_scale": 0.1,
            "covariate_preset": "primary",
        },
        {
            "scenario": "no_association_reference",
            "signal_strength": 0.0,
            "ab_shift_scale": 0.15,
            "nonlinear_covariate_proxy": False,
            "formula_linked_ldl": False,
            "label_drift_scale": 0.0,
            "covariate_preset": "primary",
        },
        {
            "scenario": "nonlinear_proxy_linear_covariates",
            "signal_strength": 0.0,
            "ab_shift_scale": 0.15,
            "nonlinear_covariate_proxy": True,
            "formula_linked_ldl": False,
            "label_drift_scale": 0.0,
            "covariate_preset": "primary",
        },
        {
            "scenario": "nonlinear_proxy_quadratic_covariates",
            "signal_strength": 0.0,
            "ab_shift_scale": 0.15,
            "nonlinear_covariate_proxy": True,
            "formula_linked_ldl": False,
            "label_drift_scale": 0.0,
            "covariate_preset": "quadratic_age_and_label",
        },
        {
            "scenario": "systematic_ab_shift_and_label_drift",
            "signal_strength": 0.85,
            "ab_shift_scale": 0.45,
            "nonlinear_covariate_proxy": False,
            "formula_linked_ldl": False,
            "label_drift_scale": 0.35,
            "covariate_preset": "primary",
        },
        {
            "scenario": "formula_linked_friedewald_ldl",
            "signal_strength": 0.85,
            "ab_shift_scale": 0.15,
            "nonlinear_covariate_proxy": False,
            "formula_linked_ldl": True,
            "label_drift_scale": 0.1,
            "covariate_preset": "primary",
        },
    ]

    rows: list[dict[str, Any]] = []
    for s_idx, spec in enumerate(scenarios):
        cfg = PairedLipidConfig(covariate_preset=str(spec["covariate_preset"]))
        deltas_paired: list[float] = []
        deltas_vs_comp: list[float] = []
        deltas_pca: list[float] = []
        deltas_pls: list[float] = []
        deltas_band: list[float] = []
        spec_iccs: list[float] = []

        for rep in range(n_repetitions):
            seed = base_seed + 100 * s_idx + rep
            ds, _ = generate_synthetic_cohort(
                n_participants=n_participants,
                seed=seed,
                compact_grid=compact_grid,
                signal_strength=float(spec["signal_strength"]),
                ab_shift_scale=float(spec["ab_shift_scale"]),
                nonlinear_covariate_proxy=bool(spec["nonlinear_covariate_proxy"]),
                formula_linked_ldl=bool(spec["formula_linked_ldl"]),
                label_drift_scale=float(spec["label_drift_scale"]),
            )
            res = evaluate_nested_cohort(ds, cfg=cfg)
            comp = res["comparisons_vs_covariate_ridge"]
            deltas_paired.append(float(comp["paired_linear_ridge"]["delta_mse"]))
            deltas_pca.append(float(comp["pca_ridge"]["delta_mse"]))
            deltas_pls.append(float(comp["pls_regression"]["delta_mse"]))
            deltas_band.append(float(comp["fixed_band_ridge"]["delta_mse"]))
            deltas_vs_comp.append(float(res["paired_vs_strongest_comparator"]["delta_mse"]))

            fold_iccs = [
                float(r["paired_spectral_icc"]["icc_a1"])
                for r in res["fold_repeatability"]
                if "paired_spectral_icc" in r and r["paired_spectral_icc"]["estimable"]
            ]
            if fold_iccs:
                spec_iccs.append(float(np.mean(fold_iccs)))

        rows.append(
            {
                "scenario": spec["scenario"],
                "covariate_preset": spec["covariate_preset"],
                "n_repetitions": n_repetitions,
                "n_participants": n_participants,
                "mean_delta_mse_paired": float(np.mean(deltas_paired)),
                "sd_delta_mse_paired": float(np.std(deltas_paired, ddof=1)) if n_repetitions > 1 else 0.0,
                "mean_delta_vs_strongest_comparator": float(np.mean(deltas_vs_comp)),
                "mean_delta_mse_pca": float(np.mean(deltas_pca)),
                "mean_delta_mse_pls": float(np.mean(deltas_pls)),
                "mean_delta_mse_fixed_band": float(np.mean(deltas_band)),
                "mean_spectral_icc_a1": float(np.mean(spec_iccs)) if spec_iccs else float("nan"),
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
) -> dict[str, Path]:
    """Run synthetic scenario suite and bootstrap interval calibration."""
    target_dir = ensure_output_dir(output_dir)
    if quick:
        n_scenario_reps = min(n_scenario_reps, 3)
        n_calibration_reps = min(n_calibration_reps, 4)
        n_bootstraps = min(n_bootstraps, 10)
        n_participants = min(n_participants, 36)

    scenarios_df = run_synthetic_scenario_suite(
        n_repetitions=n_scenario_reps,
        n_participants=n_participants,
        compact_grid=True,
    )
    cal_summary = evaluate_bootstrap_calibration(
        n_repetitions=n_calibration_reps,
        n_bootstraps=n_bootstraps,
        n_participants=n_participants,
        signal_strength=0.85,
    )
    cal_df = pd.DataFrame([cal_summary])

    scenarios_path = target_dir / "simulation_scenarios_summary.csv"
    cal_path = target_dir / "simulation_calibration_report.csv"
    write_csv_atomic(scenarios_df, scenarios_path)
    write_csv_atomic(cal_df, cal_path)
    return {
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
    parser.add_argument("--participants", type=int, default=92)
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
    )
    for key, path in outputs.items():
        print(f"[run_simulation_calibration] wrote {key}: {path}")


if __name__ == "__main__":
    main()

