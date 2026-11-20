"""Step 3: Primary held-out LDL-C evaluation, comparator comparisons, and repeatability.

Produces:
- ``eligibility_and_split_manifest.json``
- ``nested_held_out_predictions.csv``
- ``comparator_errors_and_differences.csv``
- ``fold_selected_settings.csv``
- ``baseline_repeatability_report.csv``
- ``subspace_principal_angles.csv``
- ``fold_spectral_weights.csv``
- ``bootstrap_draws.npz`` (when ``n_bootstraps > 0``)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manuscript_ml import (  # noqa: E402
    add_cohort_stage_args,
    ensure_output_dir,
    load_manuscript_config,
    print_stage_outputs,
    resolve_experiment_dataset,
    write_json_atomic,
    write_run_provenance,
)
from src.analysis.io_utils import write_csv_atomic  # noqa: E402
from src.analysis.paired_lipid.nested_cv import (  # noqa: E402
    INTERVAL_METHODS,
    IntervalMethod,
    evaluate_nested_cohort,
    run_multiplicity_bootstrap,
)

# Role of each model in the comparison family (docs/research_design.md).
MODEL_ROLES: dict[str, str] = {
    "covariate_ridge": "reference",
    "fixed_band_ridge": "established_comparator",
    "pca_ridge": "established_comparator",
    "pls_regression": "established_comparator",
    "strongest_comparator": "established_comparator_selected_by_inner_mse",
    "paired_linear_ridge": "proposed",
    # Not tuned separately: reuses lambda, r and alpha selected for the PCA-input paired model.
    "band_paired_ridge": "ablation_reuses_pca_paired_hyperparameters",
}

_ICC_REPORT_KEYS: tuple[str, ...] = (
    "paired_full_icc",
    "paired_spectral_icc",
    "fixed_band_full_icc",
    "fixed_band_spectral_icc",
    "pca_full_icc",
    "pca_spectral_icc",
    "pls_full_icc",
    "pls_spectral_icc",
    "strongest_comparator_full_icc",
    "strongest_comparator_spectral_icc",
)

_SCALE_REPORT_PREFIXES: tuple[str, ...] = (
    "paired",
    "fixed_band",
    "pca",
    "pls",
    "strongest_comparator",
)

_DISCRIM_REPORT_KEYS: tuple[str, ...] = (
    "paired_discriminability",
    "pca_discriminability",
    "band_discriminability",
    "pls_discriminability",
)


def _write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.stem}.tmp_{os.getpid()}.npz")
    try:
        np.savez(tmp_path, **arrays)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return path


def _flatten_repeatability_rows(fold_repeatability: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rep in fold_repeatability:
        fold_idx = int(rep["outer_fold"])
        n_paired = int(rep.get("n_paired_test", 0))
        row: dict[str, Any] = {
            "outer_fold": fold_idx,
            "n_paired_test": n_paired,
            "strongest_comparator_name": rep.get("strongest_comparator_name", ""),
        }
        for key in _ICC_REPORT_KEYS:
            info = rep.get(key, {})
            row[f"{key}_estimate"] = info.get("icc_a1", float("nan"))
            row[f"{key}_estimable"] = info.get("estimable", False)
            row[f"{key}_reason"] = info.get("reason", "")
            row[f"{key}_mean_diff"] = info.get("mean_diff", float("nan"))
            row[f"{key}_rmsd"] = info.get("rmsd", float("nan"))
            row[f"{key}_ms_between_participants"] = info.get("ms_between_participants", float("nan"))
            row[f"{key}_ms_between_positions"] = info.get("ms_between_positions", float("nan"))
            row[f"{key}_ms_error"] = info.get("ms_error", float("nan"))
        for prefix in _SCALE_REPORT_PREFIXES:
            row[f"{prefix}_sd_spectral_contrib"] = rep.get(f"{prefix}_sd_spectral_contrib", float("nan"))
            row[f"{prefix}_spectral_variance_share"] = rep.get(f"{prefix}_spectral_variance_share", float("nan"))
        for key in _DISCRIM_REPORT_KEYS:
            info = rep.get(key, {})
            row[f"{key}_score"] = info.get("discriminability", float("nan"))
            row[f"{key}_retrieval_top1"] = info.get("retrieval_top1", float("nan"))
            row[f"{key}_retrieval_chance"] = info.get("retrieval_chance", float("nan"))
            row[f"{key}_global_candidate_size"] = info.get("global_candidate_size", float("nan"))
            row[f"{key}_local_retrieval_top1"] = info.get("local_retrieval_top1", float("nan"))
            row[f"{key}_local_retrieval_chance"] = info.get("local_retrieval_chance", float("nan"))
            row[f"{key}_mean_local_candidate_size"] = info.get("mean_local_candidate_size", float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


def run_primary_evaluation(
    *,
    use_synthetic: bool = False,
    approve_cohort_evaluation: bool = False,
    clinical_master_path: str | Path = PROJECT_ROOT / "data" / "clinical_master.csv",
    output_dir: str | Path | None = None,
    n_bootstraps: int = 0,
    compact_grid: bool = False,
    config_path: str | Path | None = None,
    use_gpu: bool | None = None,
    batch_size: int | None = None,
    interval_method: IntervalMethod | None = None,
) -> dict[str, Path]:
    """Run the primary held-out LDL-C nested cross-validation and repeatability analysis."""
    target_dir = ensure_output_dir(output_dir)
    cfg, raw_cfg = load_manuscript_config(config_path)
    bootstrap_seed = int(raw_cfg["seeds"]["bootstrap_seed"])
    eff_interval_method: IntervalMethod = (
        raw_cfg["calibration"].get("interval_method", "studentized_t")
        if interval_method is None
        else interval_method
    )
    if eff_interval_method not in INTERVAL_METHODS:
        raise ValueError(f"Unsupported interval_method={eff_interval_method!r}; expected one of {INTERVAL_METHODS}.")

    dataset, source_meta = resolve_experiment_dataset(
        use_synthetic=use_synthetic,
        approve_cohort_evaluation=approve_cohort_evaluation,
        clinical_master_path=clinical_master_path,
        compact_grid=compact_grid,
        cfg=cfg,
        readiness=raw_cfg["readiness"],
        config_path=config_path,
    )

    eval_res = evaluate_nested_cohort(dataset, cfg=cfg, include_band_paired=True)
    outer_splits, _ = eval_res["splits"]
    fold_assignment = [-1] * dataset.n_participants
    for f_idx, (_, te_idx) in enumerate(outer_splits):
        for p_i in te_idx:
            fold_assignment[int(p_i)] = int(f_idx)

    manifest_payload: dict[str, Any] = {
        "source_metadata": source_meta,
        "manifest_summary": dataset.manifest_summary,
        "exclusions": dataset.exclusions,
        "outer_seed": eval_res["outer_seed"],
        "participant_outer_fold": {
            str(pid): int(f_id) for pid, f_id in zip(dataset.participant_ids, fold_assignment)
        },
    }

    target_col = (
        "y_true_ldl_mmol_l"
        if cfg.primary_endpoint == "ldl_mmol_l"
        else f"y_true_{cfg.primary_endpoint}"
    )
    pred_df = pd.DataFrame(
        {
            "participant_id": dataset.participant_ids,
            "outer_fold": fold_assignment,
            "n_eligible_samples": dataset.participant_sample_counts,
            target_col: dataset.outcomes[cfg.primary_endpoint],
        }
    )
    for m_name, preds in eval_res["oof_predictions"].items():
        pred_df[f"pred_{m_name}"] = preds
    for m_name, s_vals in eval_res["oof_spectral_contributions"].items():
        pred_df[f"spectral_contrib_{m_name}"] = s_vals

    comp_rows: list[dict[str, Any]] = []
    for m_name, metrics in eval_res["comparisons_vs_covariate_ridge"].items():
        comp_rows.append(
            {
                "comparison": f"{m_name}_vs_covariate_ridge",
                "baseline_model": "covariate_ridge",
                "model": m_name,
                "role": MODEL_ROLES[m_name],
                "mse_baseline": metrics["mse_covariate"],
                **metrics,
            }
        )
    vs_comp = dict(eval_res["paired_vs_strongest_comparator"])
    mse_strongest = vs_comp["mse_covariate"]
    # Keep mse_covariate unambiguous: final row compares against strongest_comparator, not covariate_ridge.
    vs_comp["mse_covariate"] = float("nan")
    comp_rows.append(
        {
            "comparison": "paired_linear_ridge_vs_strongest_comparator",
            "baseline_model": "strongest_comparator",
            "model": "paired_linear_ridge",
            "role": MODEL_ROLES["paired_linear_ridge"],
            "mse_baseline": mse_strongest,
            **vs_comp,
        }
    )

    extra_outputs: dict[str, Path] = {}
    if n_bootstraps > 0:
        boot_res = run_multiplicity_bootstrap(
            dataset,
            cfg=cfg,
            n_bootstraps=n_bootstraps,
            bootstrap_seed=bootstrap_seed,
            interval_method=eff_interval_method,
            use_gpu=use_gpu,
            batch_size=batch_size,
        )
        draws_path = _write_npz_atomic(
            target_dir / "bootstrap_draws.npz",
            {
                "delta_mse": np.asarray(boot_res["bootstrap_draws_delta_mse"], dtype=np.float64),
                "vs_comparator": np.asarray(boot_res["bootstrap_draws_vs_comparator"], dtype=np.float64),
                "strongest_vs_covariate": np.asarray(
                    boot_res["bootstrap_draws_strongest_vs_covariate"], dtype=np.float64
                ),
                "oof_sds_delta_mse": np.asarray(boot_res["bootstrap_oof_sds_delta_mse"], dtype=np.float64),
                "oof_sds_vs_comparator": np.asarray(boot_res["bootstrap_oof_sds_vs_comparator"], dtype=np.float64),
                "oof_sds_strongest_vs_covariate": np.asarray(
                    boot_res["bootstrap_oof_sds_strongest_vs_covariate"], dtype=np.float64
                ),
            },
        )
        extra_outputs["bootstrap_draws"] = draws_path

        # The table estimates come from the sequential reference path (include_band_paired=True);
        # the bootstrap base draw uses the batched engine. Recording both makes a disagreement visible.
        nested_delta = eval_res["comparisons_vs_covariate_ridge"]["paired_linear_ridge"]["delta_mse"]
        nested_vs_comp = eval_res["paired_vs_strongest_comparator"]["delta_mse"]
        nested_strong = eval_res["comparisons_vs_covariate_ridge"]["strongest_comparator"]["delta_mse"]
        abs_diff_delta = abs(boot_res["point_estimate_delta_mse"] - nested_delta)
        abs_diff_comp = abs(boot_res["point_estimate_vs_comparator"] - nested_vs_comp)
        abs_diff_strong = abs(boot_res["point_estimate_strongest_vs_covariate"] - nested_strong)
        manifest_payload["multiplicity_bootstrap"] = {
            "interval_method": boot_res["interval_method"],
            "point_estimate_delta_mse": boot_res["point_estimate_delta_mse"],
            "point_estimate_vs_comparator": boot_res["point_estimate_vs_comparator"],
            "point_estimate_strongest_vs_covariate": boot_res["point_estimate_strongest_vs_covariate"],
            "abs_diff_point_estimate_vs_nested_cv": abs_diff_delta,
            "abs_diff_vs_comparator_vs_nested_cv": abs_diff_comp,
            "abs_diff_strongest_vs_covariate_vs_nested_cv": abs_diff_strong,
            "parity_within_1e6": bool(
                abs_diff_delta <= 1e-6 and abs_diff_comp <= 1e-6 and abs_diff_strong <= 1e-6
            ),
            "n_bootstraps_requested": boot_res["n_bootstraps_requested"],
            "n_bootstraps_succeeded": boot_res["n_bootstraps_succeeded"],
            "n_failed_refits": boot_res["n_failed_refits"],
            "n_fallback_draws": boot_res["n_fallback_draws"],
            "execution": boot_res["execution"],
            "failure_reasons": boot_res["failure_reasons"],
            "interval_procedure_succeeded": boot_res["interval_procedure_succeeded"],
            "ci_95_delta_mse": boot_res["ci_95_delta_mse"],
            "ci_95_vs_comparator": boot_res["ci_95_vs_comparator"],
            "ci_95_strongest_vs_covariate": boot_res["ci_95_strongest_vs_covariate"],
            "ci_95_percentile_delta_mse": boot_res["ci_95_percentile_delta_mse"],
            "ci_95_percentile_vs_comparator": boot_res["ci_95_percentile_vs_comparator"],
            "ci_95_percentile_strongest_vs_covariate": boot_res["ci_95_percentile_strongest_vs_covariate"],
            "mean_distinct_participants": boot_res["mean_distinct_participants"],
            "min_distinct_participants": boot_res["min_distinct_participants"],
            "max_distinct_participants": boot_res["max_distinct_participants"],
            "mean_distinct_paired_participants": boot_res["mean_distinct_paired_participants"],
            "oof_loss_diff_sd_delta_mse": boot_res["oof_loss_diff_sd_delta_mse"],
            "oof_loss_diff_sd_vs_comparator": boot_res["oof_loss_diff_sd_vs_comparator"],
            "oof_loss_diff_sd_strongest_vs_covariate": boot_res["oof_loss_diff_sd_strongest_vs_covariate"],
            "bootstrap_sd_delta_mse": boot_res["bootstrap_sd_delta_mse"],
            "bootstrap_sd_vs_comparator": boot_res["bootstrap_sd_vs_comparator"],
            "bootstrap_sd_strongest_vs_covariate": boot_res["bootstrap_sd_strongest_vs_covariate"],
            "bootstrap_bias_delta_mse": boot_res["bootstrap_bias_delta_mse"],
            "bootstrap_bias_vs_comparator": boot_res["bootstrap_bias_vs_comparator"],
            "bootstrap_bias_strongest_vs_covariate": boot_res["bootstrap_bias_strongest_vs_covariate"],
            "pivot_quantiles_delta_mse": boot_res["pivot_quantiles_delta_mse"],
            "pivot_quantiles_vs_comparator": boot_res["pivot_quantiles_vs_comparator"],
            "pivot_quantiles_strongest_vs_covariate": boot_res["pivot_quantiles_strongest_vs_covariate"],
            "monte_carlo_se_of_mean": boot_res["monte_carlo_se_of_mean"],
            "monte_carlo_se_of_mean_vs_comparator": boot_res["monte_carlo_se_of_mean_vs_comparator"],
            "mc_se_ci_95_low_delta_mse": boot_res["mc_se_ci_95_low_delta_mse"],
            "mc_se_ci_95_high_delta_mse": boot_res["mc_se_ci_95_high_delta_mse"],
            "mc_se_ci_95_low_vs_comparator": boot_res["mc_se_ci_95_low_vs_comparator"],
            "mc_se_ci_95_high_vs_comparator": boot_res["mc_se_ci_95_high_vs_comparator"],
            "mc_se_ci_95_low_strongest_vs_covariate": boot_res["mc_se_ci_95_low_strongest_vs_covariate"],
            "mc_se_ci_95_high_strongest_vs_covariate": boot_res["mc_se_ci_95_high_strongest_vs_covariate"],
            "bootstrap_draws_file": draws_path.name,
        }

    comp_df = pd.DataFrame(comp_rows)
    rep_df = _flatten_repeatability_rows(eval_res["fold_repeatability"])

    manifest_path = write_json_atomic(target_dir / "eligibility_and_split_manifest.json", manifest_payload)
    preds_path = target_dir / "nested_held_out_predictions.csv"
    comp_path = target_dir / "comparator_errors_and_differences.csv"
    settings_path = target_dir / "fold_selected_settings.csv"
    rep_path = target_dir / "baseline_repeatability_report.csv"
    angles_path = target_dir / "subspace_principal_angles.csv"
    weights_path = target_dir / "fold_spectral_weights.csv"

    write_csv_atomic(pred_df, preds_path, index=False)
    write_csv_atomic(comp_df, comp_path, index=False)
    write_csv_atomic(eval_res["fold_settings"], settings_path, index=False)
    write_csv_atomic(rep_df, rep_path, index=False)
    write_csv_atomic(eval_res["subspace_angles"], angles_path, index=False)
    write_csv_atomic(eval_res["fold_spectral_weights"], weights_path, index=False)
    prov_path = write_run_provenance(
        target_dir,
        "primary_evaluation",
        config_path=config_path,
        cfg=cfg,
        settings={
            "n_bootstraps": n_bootstraps,
            "bootstrap_seed": bootstrap_seed,
            "interval_method": eff_interval_method,
            "compact_grid": compact_grid,
            "use_gpu": use_gpu,
            "batch_size": batch_size,
        },
        source_meta=source_meta,
    )

    return {
        "run_provenance_primary_evaluation": prov_path,
        "eligibility_and_split_manifest": manifest_path,
        "nested_held_out_predictions": preds_path,
        "comparator_errors_and_differences": comp_path,
        "fold_selected_settings": settings_path,
        "baseline_repeatability_report": rep_path,
        "subspace_principal_angles": angles_path,
        "fold_spectral_weights": weights_path,
        **extra_outputs,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run primary LDL-C nested evaluation and repeatability report.")
    add_cohort_stage_args(parser)
    parser.add_argument("--bootstraps", type=int, default=0)
    parser.add_argument(
        "--interval-method",
        choices=INTERVAL_METHODS,
        default=None,
        help="Bootstrap interval estimator ('studentized_t' or 'percentile'); defaults to config.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Optional batched bootstrap chunk size.")
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument("--use-gpu", dest="use_gpu", action="store_true", default=None, help="Force GPU execution.")
    gpu_group.add_argument("--no-gpu", dest="use_gpu", action="store_false", help="Force CPU execution.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    outputs = run_primary_evaluation(
        use_synthetic=args.synthetic,
        approve_cohort_evaluation=args.approve_cohort_evaluation,
        clinical_master_path=args.clinical_master,
        output_dir=args.output_dir,
        n_bootstraps=args.bootstraps,
        compact_grid=args.compact_grid,
        config_path=args.config,
        use_gpu=args.use_gpu,
        batch_size=args.batch_size,
        interval_method=args.interval_method,
    )
    print_stage_outputs("run_primary_evaluation", outputs)


if __name__ == "__main__":
    main()
