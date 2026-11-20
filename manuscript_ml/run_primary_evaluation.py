"""Step 3: Primary held-out LDL-C evaluation, comparator comparisons, and repeatability.

Produces:
- ``eligibility_and_split_manifest.json``
- ``nested_held_out_predictions.csv``
- ``comparator_errors_and_differences.csv``
- ``fold_selected_settings.csv``
- ``baseline_repeatability_report.csv``
- ``subspace_principal_angles.csv``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

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


def _flatten_repeatability_rows(fold_repeatability: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rep in fold_repeatability:
        fold_idx = int(rep["outer_fold"])
        n_paired = int(rep.get("n_paired_test", 0))
        row: dict[str, Any] = {
            "outer_fold": fold_idx,
            "n_paired_test": n_paired,
        }
        for key in ("paired_full_icc", "paired_spectral_icc", "fixed_band_full_icc", "fixed_band_spectral_icc"):
            info = rep.get(key, {})
            row[f"{key}_estimate"] = info.get("icc_a1", float("nan"))
            row[f"{key}_estimable"] = info.get("estimable", False)
            row[f"{key}_mean_diff"] = info.get("mean_diff", float("nan"))
            row[f"{key}_rmsd"] = info.get("rmsd", float("nan"))
        for key in ("paired_discriminability", "pca_discriminability", "band_discriminability"):
            info = rep.get(key, {})
            row[f"{key}_score"] = info.get("discriminability", float("nan"))
            row[f"{key}_retrieval_top1"] = info.get("retrieval_top1", float("nan"))
            row[f"{key}_retrieval_chance"] = info.get("retrieval_chance", float("nan"))
            row[f"{key}_local_retrieval_top1"] = info.get("local_retrieval_top1", float("nan"))
            row[f"{key}_local_retrieval_chance"] = info.get("local_retrieval_chance", float("nan"))
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
) -> dict[str, Path]:
    """Run the primary held-out LDL-C nested cross-validation and repeatability analysis."""
    target_dir = ensure_output_dir(output_dir)
    cfg, raw_cfg = load_manuscript_config(config_path)
    bootstrap_seed = int(raw_cfg["seeds"]["bootstrap_seed"])
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

    manifest_payload = {
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
            {"comparison": f"{m_name}_vs_covariate_ridge", "model": m_name, "role": MODEL_ROLES[m_name], **metrics}
        )
    comp_rows.append(
        {
            "comparison": "paired_linear_ridge_vs_strongest_comparator",
            "model": "paired_linear_ridge",
            "role": MODEL_ROLES["paired_linear_ridge"],
            **eval_res["paired_vs_strongest_comparator"],
        }
    )

    if n_bootstraps > 0:
        boot_res = run_multiplicity_bootstrap(
            dataset, cfg=cfg, n_bootstraps=n_bootstraps, bootstrap_seed=bootstrap_seed
        )
        # The table estimates come from the sequential reference path (include_band_paired=True);
        # the bootstrap base draw uses the batched engine. Recording both makes a disagreement visible.
        nested_delta = eval_res["comparisons_vs_covariate_ridge"]["paired_linear_ridge"]["delta_mse"]
        nested_vs_comp = eval_res["paired_vs_strongest_comparator"]["delta_mse"]
        manifest_payload["multiplicity_bootstrap"] = {
            "point_estimate_delta_mse": boot_res["point_estimate_delta_mse"],
            "point_estimate_vs_comparator": boot_res["point_estimate_vs_comparator"],
            "abs_diff_point_estimate_vs_nested_cv": abs(boot_res["point_estimate_delta_mse"] - nested_delta),
            "abs_diff_vs_comparator_vs_nested_cv": abs(boot_res["point_estimate_vs_comparator"] - nested_vs_comp),
            "n_bootstraps_requested": boot_res["n_bootstraps_requested"],
            "n_bootstraps_succeeded": boot_res["n_bootstraps_succeeded"],
            "n_failed_refits": boot_res["n_failed_refits"],
            "n_fallback_draws": boot_res["n_fallback_draws"],
            "execution": boot_res["execution"],
            "failure_reasons": boot_res["failure_reasons"],
            "interval_procedure_succeeded": boot_res["interval_procedure_succeeded"],
            "ci_95_delta_mse": boot_res["ci_95_delta_mse"],
            "ci_95_vs_comparator": boot_res["ci_95_vs_comparator"],
            "monte_carlo_se_of_mean": boot_res["monte_carlo_se_of_mean"],
        }

    comp_df = pd.DataFrame(comp_rows)
    rep_df = _flatten_repeatability_rows(eval_res["fold_repeatability"])

    manifest_path = write_json_atomic(target_dir / "eligibility_and_split_manifest.json", manifest_payload)
    preds_path = target_dir / "nested_held_out_predictions.csv"
    comp_path = target_dir / "comparator_errors_and_differences.csv"
    settings_path = target_dir / "fold_selected_settings.csv"
    rep_path = target_dir / "baseline_repeatability_report.csv"
    angles_path = target_dir / "subspace_principal_angles.csv"

    write_csv_atomic(pred_df, preds_path, index=False)
    write_csv_atomic(comp_df, comp_path, index=False)
    write_csv_atomic(eval_res["fold_settings"], settings_path, index=False)
    write_csv_atomic(rep_df, rep_path, index=False)
    write_csv_atomic(eval_res["subspace_angles"], angles_path, index=False)
    prov_path = write_run_provenance(
        target_dir,
        "primary_evaluation",
        config_path=config_path,
        cfg=cfg,
        settings={"n_bootstraps": n_bootstraps, "bootstrap_seed": bootstrap_seed, "compact_grid": compact_grid},
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
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run primary LDL-C nested evaluation and repeatability report.")
    add_cohort_stage_args(parser)
    parser.add_argument("--bootstraps", type=int, default=0)
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
    )
    print_stage_outputs("run_primary_evaluation", outputs)


if __name__ == "__main__":
    main()
