from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src import config, get_output_root, setup_logging
from src.analysis.ml_latent_structure import generate_latent_structure_ml_report
from src.classification.cv_loop import run_phase6_cv
from src.classification.holdout import score_holdout, split_subject_holdout
from src.classification.run_manifest import write_run_manifest
from src.constants import CANONICAL_FP_CHANNELS, CANONICAL_REGION_BOUNDS
from src.classification.evaluation import (
    aggregate_metrics,
    build_biomarker_report,
    build_ch_lipid_state_audit,
    build_feature_track_ablation,
    compare_classifiers_friedman_nemenyi,
    compute_calibration_metrics,
    compute_sex_stratified_metrics,
    permutation_test,
    pooled_oof_metrics,
    retrain_permutation_summary,
    summarize_fold_metrics,
)
from src.features.compile_features import aggregate_subject_vectors, compile_feature_matrices, infer_split_index
from src.features.feature_metadata import build_feature_metadata, enrich_feature_table
from src.fusion.area_pipeline import process_subject_areas
from src.fusion.area_summary import AreaStore, area_store_from_process_results
from src.io.file_crawler import build_file_index, disambiguate_subject_ids
from src.metadata.clinical import load_clinical_data
from src.secondary.secondary_analyses import run_secondary_analyses
from src.terminal_output import (
    print_artifact_table,
    print_kv_table,
    print_phase,
    print_step,
    progress_iter,
)
from src.visualization.classification_plots import (
    plot_shap_summary,
    plot_spectrum_highlights,
    plot_top_feature_boxplots,
)

logger = logging.getLogger(__name__)


def build_arg_parser(
    description: str = "Run full ML pipeline (Phases 1-8).",
    **parser_kwargs,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description, **parser_kwargs)
    parser.add_argument("--data-dir", type=str, default=None, help="Reorganized data root (default from config)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: config.output.ml_root)")
    parser.add_argument("--max-subjects", type=int, default=None, help="Optional subject cap for quick runs")
    parser.add_argument("--skip-cv", action="store_true", help="Skip Phase 6/7 CV and evaluation")
    parser.add_argument("--skip-secondary", action="store_true", help="Skip Phase 8 secondary analyses")
    parser.add_argument("--n-permutations", type=int, default=None, help="Override permutation test count")
    parser.add_argument("--force-reprocess", action="store_true", help="Ignore cached area store and re-process raw files")
    parser.add_argument("--skip-holdout", action="store_true", help="Skip held-out test set split and evaluation")
    parser.add_argument(
        "--classifiers",
        type=str,
        default=None,
        help="Comma-separated classifier names to run (overrides config classifiers.enabled). "
             "Available: svm_rbf, svm_linear, logreg_elastic_net, decision_tree, knn, pls_da, "
             "random_forest, extra_trees, catboost",
    )
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_arg_parser()
    return parser.parse_args()


def _select_capped_subject_ids(file_index: pd.DataFrame, max_subjects: int) -> list[str]:
    """Select a stable, class-aware subject subset for quick runs."""
    subjects = (
        file_index.drop_duplicates(subset=["subject_id"], keep="first")
        .assign(subject_id=lambda d: d["subject_id"].astype(str))
        .reset_index(drop=True)
    )
    if max_subjects <= 0:
        return []
    if len(subjects) <= max_subjects:
        return subjects["subject_id"].tolist()

    strata_cols = [
        col
        for col in ("is_control", "gender")
        if col in subjects.columns and subjects[col].dropna().nunique() > 1
    ]
    if not strata_cols:
        return subjects["subject_id"].head(max_subjects).tolist()

    queues = [
        group["subject_id"].tolist()
        for _, group in subjects.groupby(strata_cols, sort=False, dropna=False)
    ]
    selected: list[str] = []
    while len(selected) < max_subjects and any(queues):
        for queue in queues:
            if queue and len(selected) < max_subjects:
                selected.append(queue.pop(0))

    return selected


def run_pipeline(
    data_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    max_subjects: int | None = None,
    skip_cv: bool = False,
    skip_secondary: bool = False,
    skip_holdout: bool = False,
    n_permutations: int | None = None,
    force_reprocess: bool = False,
    enabled_classifiers: list[str] | None = None,
    random_state: int = 42,
    cache_dir: Path | None = None,
    deterministic_env: dict[str, str] | None = None,
) -> dict:
    run_started_utc = datetime.now(timezone.utc).isoformat()
    file_index, clinical_df = _phase_index_and_clinical(data_dir, max_subjects)
    area_df, split_index, config_hash = _phase_area_table(
        file_index=file_index,
        output_dir=output_dir,
        cache_dir=cache_dir,
        force_reprocess=force_reprocess,
    )


    holdout_state = _phase_holdout_split(area_df, skip_holdout, random_state)
    cv_area_df = holdout_state["cv_area_df"]
    holdout_area_df = holdout_state["holdout_area_df"]
    holdout_subject_ids = holdout_state["holdout_subject_ids"]
    holdout_random_state = holdout_state["holdout_random_state"]
    holdout_enabled = holdout_state["holdout_enabled"]
    holdout_score_enabled = holdout_state["holdout_score_enabled"]

    feature_blocks = _phase_features(area_df, clinical_df, split_index)
    classifier_feature_matrix = feature_blocks["classifier_feature_matrix"]
    descriptive_feature_matrix = feature_blocks["descriptive_feature_matrix"]

    resolved_n_perm = int(
        config.get("permutation_test", {}).get("n_permutations", 1000)
        if n_permutations is None
        else n_permutations
    )
    cv_state = _phase_cv(
        cv_area_df=cv_area_df,
        clinical_df=clinical_df,
        split_index=split_index,
        skip_cv=skip_cv,
        enabled_classifiers=enabled_classifiers,
        resolved_n_perm=resolved_n_perm,
        random_state=random_state,
    )
    cv_out = cv_state["cv_out"]
    fold_metrics = cv_state["fold_metrics"]
    aggregate = cv_state["aggregate"]
    pooled_oof = cv_state["pooled_oof"]
    permutation = cv_state["permutation"]
    comparison = cv_state["comparison"]
    biomarker = cv_state["biomarker"]
    sex_metrics = cv_state["sex_metrics"]
    calibration = cv_state["calibration"]
    track_ablation = cv_state["track_ablation"]
    ch_audit = cv_state["ch_audit"]


    holdout_metrics = _phase_holdout_scoring(
        cv_area_df=cv_area_df,
        holdout_area_df=holdout_area_df,
        clinical_df=clinical_df,
        split_index=split_index,
        enabled_classifiers=enabled_classifiers,
        random_state=random_state,
        holdout_enabled=holdout_enabled,
        holdout_score_enabled=holdout_score_enabled,
        skip_cv=skip_cv,
    )
    secondary = _phase_secondary(
        descriptive_feature_matrix=descriptive_feature_matrix,
        area_df=area_df,
        clinical_df=clinical_df,
        split_index=split_index,
        random_state=random_state,
        skip_secondary=skip_secondary,
    )

    cv_config_snapshot = _build_cv_config_snapshot(
        split_index=split_index,
        resolved_n_permutations=resolved_n_perm,
        skip_cv=skip_cv,
        skip_secondary=skip_secondary,
        random_state=random_state,
        holdout_random_state=holdout_random_state,
    )

    print_step("Writing deliverables")
    out_dir = _resolve_output_dir(output_dir)
    artifacts = write_deliverables(
        output_dir=out_dir,
        area_df=area_df,
        classifier_feature_matrix=classifier_feature_matrix,
        descriptive_feature_matrix=descriptive_feature_matrix,
        cv_out=cv_out,
        fold_metrics=fold_metrics,
        aggregate=aggregate,
        pooled_oof=pooled_oof,
        permutation=permutation,
        comparison=comparison,
        biomarker=biomarker,
        secondary=secondary,
        cv_config_snapshot=cv_config_snapshot,
        sex_metrics=sex_metrics,
        calibration=calibration,
        track_ablation=track_ablation,
        ch_audit=ch_audit,
        fold_subject_balance=cv_out.get("fold_subject_balance", pd.DataFrame()),
    )

    # --- Latent-structure interpretive report ---
    try:
        suspect_windows = config.get("analysis_robustness", {}).get("region_weighting", {}).get("exclude_windows", [])
        report_path = generate_latent_structure_ml_report(
            ml_output_dir=out_dir,
            suspect_windows=suspect_windows or [],
        )
        artifacts["latent_structure_ml_report_md"] = str(report_path)
    except Exception as exc:
        logger.warning("Latent-structure ML report generation failed: %s", exc)

    # --- Holdout artifacts ---
    if holdout_subject_ids:
        holdout_json = out_dir / "holdout_subjects.json"
        holdout_json.write_text(
            json.dumps(
                {
                    "holdout_subject_ids": holdout_subject_ids,
                    "n_holdout": len(holdout_subject_ids),
                    "n_cv": int(cv_area_df["subject_id"].astype(str).nunique()),
                    "resolved_random_state": holdout_random_state,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        artifacts["holdout_subjects_json"] = str(holdout_json)

    if not holdout_metrics.empty:
        holdout_csv = out_dir / "holdout_metrics.csv"
        holdout_metrics.to_csv(holdout_csv, index=False)
        artifacts["holdout_metrics_csv"] = str(holdout_csv)

        generalization = _build_model_generalization_summary(aggregate, holdout_metrics, biomarker, pooled_oof)
        if not generalization.empty:
            generalization_csv = out_dir / "model_generalization_summary.csv"
            generalization.to_csv(generalization_csv, index=False)
            artifacts["model_generalization_summary_csv"] = str(generalization_csv)

    # --- Run manifest ---
    n_cv_subjects = int(cv_area_df["subject_id"].astype(str).nunique())
    manifest_path = write_run_manifest(
        out_dir,
        holdout_subjects=holdout_subject_ids,
        n_cv_subjects=n_cv_subjects,
        n_holdout_subjects=len(holdout_subject_ids),
        random_state=random_state,
        holdout_random_state=holdout_random_state,
        preprocessing_config_hash=config_hash,
        run_started_utc=run_started_utc,
        run_ended_utc=datetime.now(timezone.utc).isoformat(),
        deterministic_env=deterministic_env,
    )
    artifacts["run_manifest_json"] = str(manifest_path)

    print_phase("ML Pipeline Completed")
    print_kv_table(
        "Run Summary",
        {
            "output_dir": str(out_dir),
            "artifacts_written": int(len(artifacts)),
            "subjects_in_descriptive_matrix": int(descriptive_feature_matrix["subject_id"].astype(str).nunique())
            if "subject_id" in descriptive_feature_matrix.columns
            else int(len(descriptive_feature_matrix)),
        },
    )
    print_artifact_table("Artifacts", artifacts)

    return {
        "output_dir": out_dir,
        "artifacts": artifacts,
        "area_df": area_df,
        "cv_area_df": cv_area_df,
        "holdout_area_df": holdout_area_df,
        "holdout_subject_ids": holdout_subject_ids,
        "holdout_metrics": holdout_metrics,
        "classifier_feature_matrix": classifier_feature_matrix,
        "feature_matrix": descriptive_feature_matrix,
        "descriptive_feature_matrix": descriptive_feature_matrix,
        "split_index": int(split_index),
        "cv": cv_out,
        "fold_metrics": fold_metrics,
        "aggregate": aggregate,
        "permutation": permutation,
        "comparison": comparison,
        "biomarker": biomarker,
        "secondary": secondary,
    }


def _phase_index_and_clinical(
    data_dir: str | Path | None,
    max_subjects: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print_step("Phase 1: indexing and clinical reconciliation")
    file_index = disambiguate_subject_ids(build_file_index(data_dir=data_dir, measurement="all"))
    if file_index.empty:
        raise ValueError("No files found for pipeline")

    if max_subjects is not None:
        logger.info("Applying subject cap: max_subjects=%d", int(max_subjects))
        keep = _select_capped_subject_ids(file_index, int(max_subjects))
        file_index = file_index[file_index["subject_id"].astype(str).isin(keep)].reset_index(drop=True)

    clinical_df = load_clinical_data(data_index=file_index)
    return file_index, clinical_df


def _phase_area_table(
    *,
    file_index: pd.DataFrame,
    output_dir: str | Path | None,
    cache_dir: Path | None,
    force_reprocess: bool,
) -> tuple[pd.DataFrame, int, str]:
    print_step("Phase 2: paired-area fusion and area table construction")
    if cache_dir is None:
        cache_dir = _resolve_output_dir(output_dir) / ".cache"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "area_store.npz"
    config_hash = _compute_preprocessing_config_hash(file_index)

    area_df = _load_cached_area_df(cache_path, config_hash, force_reprocess)
    valid_sids = set(file_index["subject_id"].astype(str).tolist())
    if area_df is not None:
        area_df = area_df[area_df["subject_id"].astype(str).isin(valid_sids)].reset_index(drop=True)
        area_df = _attach_subject_metadata(area_df, file_index)
        logger.info("Phase 2: loaded area table from cache (%d areas)", len(area_df))
    else:
        area_df = _build_area_table(file_index)
        _save_cached_area_df(area_df, cache_path, config_hash)
        logger.info("Phase 2: area table built and cached (%d areas)", len(area_df))

    split_index = _infer_split_index_from_area_df(area_df)
    print_kv_table(
        "Phase 2 - Area Table",
        {
            "paired_areas": int(len(area_df)),
            "subjects_with_areas": int(area_df["subject_id"].astype(str).nunique()),
            "resolved_split_index": int(split_index),
        },
    )
    return area_df, int(split_index), config_hash



def _phase_holdout_split(
    area_df: pd.DataFrame,
    skip_holdout: bool,
    random_state: int,
) -> dict:
    holdout_cfg = config.get("holdout", {})
    holdout_enabled = bool(holdout_cfg.get("enabled", False)) and not skip_holdout
    holdout_score_enabled = holdout_enabled and _should_score_holdout(holdout_cfg)

    if not holdout_enabled:
        return {
            "cv_area_df": area_df,
            "holdout_area_df": pd.DataFrame(),
            "holdout_subject_ids": [],
            "holdout_random_state": None,
            "holdout_enabled": False,
            "holdout_score_enabled": False,
        }

    print_step("Phase 3: held-out test set split")
    holdout_random_state = _resolve_holdout_random_state(holdout_cfg, random_state)
    cv_area_df, holdout_area_df, holdout_subject_ids = split_subject_holdout(
        area_df,
        label_col="is_control",
        use_sex=bool(holdout_cfg.get("stratify_by_sex", True)),
        fraction=float(holdout_cfg.get("fraction", 0.15)),
        random_state=holdout_random_state,
    )
    print_kv_table(
        "Phase 3 - Holdout Split",
        {
            "cv_subjects": int(cv_area_df["subject_id"].astype(str).nunique()),
            "holdout_subjects": len(holdout_subject_ids),
            "holdout_seed": int(holdout_random_state),
            "holdout_scoring": "enabled" if holdout_score_enabled else "disabled",
            "holdout_ids": ", ".join(holdout_subject_ids[:5]) + ("..." if len(holdout_subject_ids) > 5 else ""),
        },
    )
    return {
        "cv_area_df": cv_area_df,
        "holdout_area_df": holdout_area_df,
        "holdout_subject_ids": holdout_subject_ids,
        "holdout_random_state": holdout_random_state,
        "holdout_enabled": holdout_enabled,
        "holdout_score_enabled": holdout_score_enabled,
    }


def _phase_features(
    area_df: pd.DataFrame,
    clinical_df: pd.DataFrame,
    split_index: int,
) -> dict:
    print_step("Phase 4: compiling feature matrices")
    feature_blocks = _build_feature_matrices(area_df, clinical_df, split_index=split_index)
    descriptive_feature_matrix = feature_blocks["descriptive_feature_matrix"]
    print_kv_table(
        "Phase 4 - Feature Matrices",
        {
            "classifier_rows": int(len(feature_blocks["classifier_feature_matrix"])),
            "classifier_feature_columns": int(feature_blocks["n_classifier_features"]),
            "descriptive_rows": int(len(descriptive_feature_matrix)),
            "descriptive_columns": int(descriptive_feature_matrix.shape[1]),
        },
    )
    return feature_blocks


def _empty_cv_comparison() -> dict:
    return {
        "score_matrix": pd.DataFrame(),
        "mean_ranks": pd.DataFrame(),
        "friedman": {},
        "nemenyi": {"pairwise": pd.DataFrame(), "critical_difference": np.nan},
    }


def _phase_cv(
    *,
    cv_area_df: pd.DataFrame,
    clinical_df: pd.DataFrame,
    split_index: int,
    skip_cv: bool,
    enabled_classifiers: list[str] | None,
    resolved_n_perm: int,
    random_state: int,
) -> dict:
    cv_out = {
        "fold_results": pd.DataFrame(),
        "predictions": pd.DataFrame(),
        "selection_trace": pd.DataFrame(),
    }
    state = {
        "cv_out": cv_out,
        "fold_metrics": pd.DataFrame(),
        "aggregate": pd.DataFrame(),
        "pooled_oof": pd.DataFrame(),
        "permutation": pd.DataFrame(),
        "comparison": _empty_cv_comparison(),
        "biomarker": pd.DataFrame(),
        "sex_metrics": pd.DataFrame(),
        "calibration": pd.DataFrame(),

        "track_ablation": pd.DataFrame(),
        "ch_audit": pd.DataFrame(),
    }
    if skip_cv:
        return state

    print_step("Phase 6: repeated subject-level CV")
    cv_out = run_phase6_cv(
        cv_area_df,
        clinical_df=clinical_df,
        label_col="is_control",
        split_index=split_index,
        show_progress=True,
        enabled_classifiers=enabled_classifiers,
        random_state=random_state,
    )
    fold_metrics = summarize_fold_metrics(cv_out["predictions"])
    aggregate = aggregate_metrics(fold_metrics, predictions=cv_out["predictions"])
    pooled_oof = pooled_oof_metrics(cv_out["predictions"])

    print_kv_table(
        "Phase 6 - CV Summary",
        {
            "fold_rows": int(len(cv_out["fold_results"])),
            "prediction_rows": int(len(cv_out["predictions"])),
            "classifiers": int(cv_out["fold_results"]["classifier"].nunique())
            if not cv_out["fold_results"].empty
            else 0,
        },
    )

    permutation = _phase_permutation(
        cv_area_df=cv_area_df,
        clinical_df=clinical_df,
        split_index=split_index,
        enabled_classifiers=enabled_classifiers,
        pooled_oof=pooled_oof,
        predictions=cv_out["predictions"],
        resolved_n_perm=resolved_n_perm,
        random_state=random_state,
    )
    comparison = compare_classifiers_friedman_nemenyi(fold_metrics, metric="balanced_accuracy")
    biomarker = _build_cv_biomarker(cv_out)
    track_g_audits = cv_out.get("track_g_audits", [])

    state.update(
        {
            "cv_out": cv_out,
            "fold_metrics": fold_metrics,
            "aggregate": aggregate,
            "pooled_oof": pooled_oof,
            "permutation": permutation,
            "comparison": comparison,
            "biomarker": biomarker,
            "sex_metrics": compute_sex_stratified_metrics(cv_out["predictions"], cv_area_df),
            "calibration": compute_calibration_metrics(cv_out["predictions"]),

            "track_ablation": build_feature_track_ablation(cv_out["selection_trace"]),
            "ch_audit": build_ch_lipid_state_audit(track_g_audits) if track_g_audits else pd.DataFrame(),
        }
    )
    return state


def _phase_permutation(
    *,
    cv_area_df: pd.DataFrame,
    clinical_df: pd.DataFrame,
    split_index: int,
    enabled_classifiers: list[str] | None,
    pooled_oof: pd.DataFrame,
    predictions: pd.DataFrame,
    resolved_n_perm: int,
    random_state: int,
) -> pd.DataFrame:
    perm_cfg = config.get("permutation_test", {})
    perm_mode = str(perm_cfg.get("mode", "posthoc")).strip().lower()
    perm_metric = str(perm_cfg.get("metric", "balanced_accuracy"))
    if resolved_n_perm > 0 and perm_mode == "retrain":
        primary_classifier = _resolve_primary_permutation_classifier(
            enabled_classifiers=enabled_classifiers,
            pooled_oof=pooled_oof,
            perm_cfg=perm_cfg,
        )
        print_step(
            f"Phase 7: retrain permutation testing "
            f"(classifier={primary_classifier}, n_permutations={resolved_n_perm})"
        )
        return _run_retrain_permutation_test(
            cv_area_df=cv_area_df,
            clinical_df=clinical_df,
            split_index=split_index,
            primary_classifier=primary_classifier,
            observed_metrics=pooled_oof,
            metric=perm_metric,
            n_permutations=resolved_n_perm,
            random_state=random_state,
        )
    if resolved_n_perm > 0:
        print_step(f"Phase 7: post-hoc permutation sanity check (n_permutations={resolved_n_perm})")
        return permutation_test(predictions, metric=perm_metric, n_permutations=resolved_n_perm)
    return pd.DataFrame()


def _build_cv_biomarker(cv_out: dict) -> pd.DataFrame:
    if cv_out["selection_trace"].empty:
        return pd.DataFrame()
    total_folds = int(cv_out["fold_results"]["fold"].nunique())
    if total_folds < 1 and not cv_out.get("fold_subjects", pd.DataFrame()).empty:
        total_folds = int(cv_out["fold_subjects"]["fold"].nunique())
    return build_biomarker_report(cv_out["selection_trace"], total_folds=total_folds)


def _phase_holdout_scoring(
    *,
    cv_area_df: pd.DataFrame,
    holdout_area_df: pd.DataFrame,
    clinical_df: pd.DataFrame,
    split_index: int,
    enabled_classifiers: list[str] | None,
    random_state: int,
    holdout_enabled: bool,
    holdout_score_enabled: bool,
    skip_cv: bool,
) -> pd.DataFrame:
    if holdout_enabled and not holdout_score_enabled and not skip_cv:
        print_step("Phase 7b: holdout evaluation skipped (holdout.score=false)")
    if not (holdout_enabled and holdout_score_enabled and not holdout_area_df.empty and not skip_cv):
        return pd.DataFrame()

    print_step("Phase 7b: holdout evaluation (one-shot)")
    holdout_metrics = score_holdout(
        cv_area_df,
        holdout_area_df,
        clinical_df=clinical_df,
        split_index=split_index,
        enabled_classifiers=enabled_classifiers,
        label_col="is_control",
        random_state=random_state,
    )
    if not holdout_metrics.empty:
        print_kv_table(
            "Phase 7b - Holdout Metrics",
            {
                clf: f"bal_acc={row.get('balanced_accuracy_mean', float('nan')):.3f}  "
                     f"AUC={row.get('auc_mean', float('nan')):.3f}  "
                     f"MCC={row.get('mcc_mean', float('nan')):.3f}"
                for clf, row in holdout_metrics.set_index("classifier").iterrows()
            },
        )
    return holdout_metrics


def _phase_secondary(
    *,
    descriptive_feature_matrix: pd.DataFrame,
    area_df: pd.DataFrame,
    clinical_df: pd.DataFrame,
    split_index: int,
    random_state: int,
    skip_secondary: bool,
) -> dict:
    if skip_secondary:
        return {}
    print_step("Phase 8: secondary analyses")
    return run_secondary_analyses(
        descriptive_feature_matrix,
        area_df=area_df,
        clinical_df=clinical_df,
        split_index=split_index,
        random_state=random_state,
    )


def _build_area_table(file_index: pd.DataFrame, max_workers: int | None = None) -> pd.DataFrame:
    results: list[dict] = []
    subject_groups = [grp for _, grp in file_index.groupby("subject_id", sort=False)]

    if max_workers is None:
        max_workers = int(config.get("analysis", {}).get("io_workers", 8))

    if max_workers > 1 and len(subject_groups) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=min(max_workers, len(subject_groups))) as executor:
            futures = [executor.submit(process_subject_areas, grp) for grp in subject_groups]
            for future in futures:
                results.extend(future.result())
    else:
        for grp in progress_iter(subject_groups, "Phase 2: processing subjects"):
            results.extend(process_subject_areas(grp))

    if not results:
        raise ValueError("No paired 1300/2850 areas survived processing")

    store = area_store_from_process_results(results)
    area_df = store.to_dataframe()
    return _attach_subject_metadata(area_df, file_index)


def _attach_subject_metadata(area_df: pd.DataFrame, file_index: pd.DataFrame) -> pd.DataFrame:
    """Attach subject labels to area rows."""
    subject_meta = _subject_metadata_from_file_index(file_index)
    stale_cols = [col for col in ("is_control", "gender", "subject_id_raw") if col in area_df.columns]
    area_rows = area_df.drop(columns=stale_cols) if stale_cols else area_df.copy()
    area_rows["subject_id"] = area_rows["subject_id"].astype(str)

    area_df = area_rows.merge(subject_meta, on="subject_id", how="left")
    if area_df["is_control"].isna().any():
        missing = area_df.loc[area_df["is_control"].isna(), "subject_id"].astype(str).unique().tolist()
        raise ValueError(
            f"No is_control label found for {len(missing)} area-table subject(s): {missing[:10]}. "
            "Ensure file_index and area_df use the same subject_id values."
        )
    return area_df


def _subject_metadata_from_file_index(file_index: pd.DataFrame) -> pd.DataFrame:
    required = {"subject_id", "is_control"}
    missing = required - set(file_index.columns)
    if missing:
        raise ValueError(f"file_index missing required subject metadata columns: {sorted(missing)}")

    if "gender" not in file_index.columns:
        raise ValueError("file_index must contain 'gender' metadata column")

    meta = file_index.copy()
    meta["subject_id"] = meta["subject_id"].astype(str)
    meta["gender"] = _normalize_sex_metadata(meta["gender"])

    for label_col in ("is_control", "gender"):
        n_unique = meta.groupby("subject_id")[label_col].nunique()
        bad = n_unique[n_unique > 1].index.tolist()
        if bad:
            raise ValueError(
                f"Inconsistent '{label_col}' values for subject(s) {bad[:10]}. "
                "Each subject must have a single consistent label."
            )

    subject_meta_cols = ["subject_id", "is_control", "gender"]
    if "subject_id_raw" in meta.columns:
        subject_meta_cols.append("subject_id_raw")
    subject_meta = meta[subject_meta_cols].drop_duplicates(subset=["subject_id"], keep="first")
    return subject_meta.reset_index(drop=True)


def _normalize_sex_metadata(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.replace(
        {"": "unknown", "nan": "unknown", "none": "unknown", "<na>": "unknown"}
    ).fillna("unknown")


def _infer_split_index_from_area_df(area_df: pd.DataFrame) -> int:
    if area_df.empty:
        raise ValueError("Cannot infer split_index from empty area_df")
    n_features = int(len(np.asarray(area_df.iloc[0]["area_mean"], dtype=float)))
    return infer_split_index(split_index=None, n_features=n_features)


def _build_feature_matrices(
    area_df: pd.DataFrame,
    clinical_df: pd.DataFrame,
    split_index: int,
) -> dict:
    """Build descriptive subject matrices from the same Track A-E feature core.

    classifier_feature_matrix
        Track A-E subject features + authoritative labels (`is_control`, `sex`)
        computed once on the full cohort. This matrix is descriptive and not
        equivalent to fold-wise CV feature spaces.

    descriptive_feature_matrix
        Export-oriented matrix used for deliverables/secondary analyses.
        It starts from classifier_feature_matrix and adds extra subject-level
        clinical columns from curated metadata when available.
    """
    flow_cfg = config.get("classification_flow", {})
    relative_enabled_default = bool(config.get("relative_features", {}).get("enabled", False))
    compiled = compile_feature_matrices(
        train_area_df=area_df,
        test_area_df=area_df,
        clinical_df=clinical_df,
        split_index=split_index,
        include_heterogeneity=bool(flow_cfg.get("include_track_d", False)),
        include_clinical=bool(flow_cfg.get("include_track_e", False)),
        include_relative_features=bool(flow_cfg.get("include_relative_features", relative_enabled_default)),
    )

    classifier_df = compiled["train_features_df"].copy()

    labels = area_df[["subject_id", "is_control", "gender"]].drop_duplicates(subset=["subject_id"], keep="first")
    classifier_df = classifier_df.merge(labels, on="subject_id", how="left")

    feature_df = classifier_df.copy()

    if not clinical_df.empty and "subject_id" in clinical_df.columns:
        keep_cols = [
            c
            for c in clinical_df.columns
            if c not in {"gender_binary", "is_control", "condition"}
        ]
        n_before = len(feature_df)
        feature_df = feature_df.merge(
            clinical_df[keep_cols],
            on="subject_id",
            how="left",
            suffixes=("", "_clinical"),
        )
        if len(feature_df) != n_before:
            raise ValueError(
                f"Clinical merge created duplicate rows ({n_before} → {len(feature_df)}). "
                "clinical_df must have unique subject_id values."
            )

    return {
        "classifier_feature_matrix": classifier_df,
        "descriptive_feature_matrix": feature_df,
        "n_classifier_features": int(len(compiled["feature_names"])),
    }


def _resolve_output_dir(output_dir: str | Path | None) -> Path:
    if output_dir is not None:
        out = Path(output_dir)
    else:
        out = get_output_root("ml")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _build_cv_config_snapshot(
    split_index: int,
    resolved_n_permutations: int,
    skip_cv: bool,
    skip_secondary: bool,
    random_state: int = 42,
    holdout_random_state: int | None = None,
) -> dict:
    fs_cfg = config.get("feature_selection", {})
    cv_cfg = config.get("cv", {})
    pca_cfg = config.get("pca", {})
    peaks_cfg = config.get("peaks", {})
    hetero_cfg = config.get("heterogeneity", {})
    area_cfg = config.get("area_outlier", {})
    sec_cfg = config.get("secondary", {})

    return {
        "resolved": {
            "random_state": random_state,
            "split_index": int(split_index),
            "skip_cv": bool(skip_cv),
            "skip_secondary": bool(skip_secondary),
            "n_permutations": int(resolved_n_permutations),
            "holdout_random_state": holdout_random_state,
        },
        "cv": {
            "n_splits": int(cv_cfg.get("n_splits", 5)),
            "n_repeats": int(cv_cfg.get("n_repeats", 10)),
            "stratify_by_sex": bool(cv_cfg.get("stratify_by_sex", True)),
        },
        "feature_selection": {
            "variance_threshold": float(fs_cfg.get("variance_threshold", 0.01)),
            "mi_permutations": int(fs_cfg.get("mi_permutations", 100)),
            "mi_percentile": float(fs_cfg.get("mi_percentile", 95)),
            "mi_n_neighbors": int(fs_cfg.get("mi_n_neighbors", 3)),
            "target_feature_counts": list(fs_cfg.get("target_feature_counts", [5, 10, 15])),
            "inner_cv_folds": int(fs_cfg.get("inner_cv_folds", 3)),
        },
        "feature_extraction": {
            "pca_variance_threshold": float(pca_cfg.get("variance_threshold", 0.95)),
            "peak_prominence_percentile": float(peaks_cfg.get("prominence_percentile", 75)),
            "peak_min_distance": int(peaks_cfg.get("min_distance", 15)),
            "peak_window": int(peaks_cfg.get("peak_window", 5)),
            "peak_n_top_for_ratios": int(peaks_cfg.get("n_top_for_ratios", 6)),
            "peak_canonical_bin_width": int(peaks_cfg.get("canonical_bin_width", 5)),
            "heterogeneity_n_targeted_peaks": int(hetero_cfg.get("n_targeted_peaks", 5)),
        },
        "outlier_screening": {
            "threshold_factor": float(area_cfg.get("threshold_factor", 2.0)),
            "distance_metric": str(area_cfg.get("distance_metric", "correlation")),
            "min_areas_per_slide": int(area_cfg.get("min_areas_per_slide", 2)),
        },
        "secondary": {
            "moca_method": str(sec_cfg.get("moca_method", "ridge")),
            "moca_cv": str(sec_cfg.get("moca_cv", "loo")),
            "aspects_method": str(sec_cfg.get("aspects_method", "spearman")),
        },
        "classifiers": config.get("classifiers", {}),
        "classification_search": config.get("classification_search", {}),
        "decision_threshold": config.get("decision_threshold", {}),
        "permutation_test": config.get("permutation_test", {}),
        "holdout": {
            **config.get("holdout", {}),
            "resolved_random_state": holdout_random_state,
        },
        "classification_flow": config.get("classification_flow", {}),
        "ch_lipid_state": config.get("ch_lipid_state", {}),
        "calibration": config.get("calibration", {}),
        "interpretability": config.get("interpretability", {}),
    }


def _resolve_holdout_random_state(holdout_cfg: dict, run_random_state: int) -> int:
    raw = holdout_cfg.get("random_state", None)
    if raw is None:
        return int(run_random_state)
    if isinstance(raw, str) and raw.strip().lower() in {"", "none", "null"}:
        return int(run_random_state)
    return int(raw)


def _should_score_holdout(holdout_cfg: dict) -> bool:
    """Return whether the already-split holdout lockbox should be scored."""
    return bool(holdout_cfg.get("score", True))


def _resolve_primary_permutation_classifier(
    *,
    enabled_classifiers: list[str] | None,
    pooled_oof: pd.DataFrame,
    perm_cfg: dict,
) -> str:
    requested = str(perm_cfg.get("primary_classifier", "")).strip()
    if requested:
        return requested
    if not pooled_oof.empty and {"classifier", "balanced_accuracy"}.issubset(pooled_oof.columns):
        ranked = pooled_oof.sort_values("balanced_accuracy", ascending=False)
        return str(ranked["classifier"].iloc[0])
    if enabled_classifiers:
        return str(enabled_classifiers[0])
    cfg_enabled = config.get("classifiers", {}).get("enabled", [])
    if cfg_enabled:
        return str(cfg_enabled[0])
    return "pls_da"


def _run_retrain_permutation_test(
    *,
    cv_area_df: pd.DataFrame,
    clinical_df: pd.DataFrame | None,
    split_index: int,
    primary_classifier: str,
    observed_metrics: pd.DataFrame,
    metric: str,
    n_permutations: int,
    random_state: int,
) -> pd.DataFrame:
    observed = _lookup_metric_value(observed_metrics, primary_classifier, metric)
    if not np.isfinite(observed):
        logger.warning(
            "Retrain permutation skipped: observed %s unavailable for %s.",
            metric,
            primary_classifier,
        )
        return pd.DataFrame()

    rng = np.random.default_rng(random_state)
    null_scores: list[float] = []
    subject_ids = cv_area_df["subject_id"].astype(str).drop_duplicates().to_numpy(dtype=object)
    label_map = (
        cv_area_df[["subject_id", "is_control"]]
        .drop_duplicates(subset=["subject_id"], keep="first")
        .assign(subject_id=lambda d: d["subject_id"].astype(str))
        .set_index("subject_id")["is_control"]
        .astype(int)
        .reindex(subject_ids)
    )
    labels = label_map.to_numpy(dtype=int)

    for perm_idx in range(int(n_permutations)):
        permuted = cv_area_df.copy()
        shuffled = rng.permutation(labels)
        shuffled_map = dict(zip(subject_ids.tolist(), shuffled.tolist()))
        permuted["is_control"] = permuted["subject_id"].astype(str).map(shuffled_map).astype(int)

        try:
            perm_cv = run_phase6_cv(
                permuted,
                clinical_df=clinical_df,
                label_col="is_control",
                split_index=split_index,
                show_progress=False,
                enabled_classifiers=[primary_classifier],
                random_state=random_state + perm_idx + 1,
            )
            perm_pooled = pooled_oof_metrics(
                perm_cv["predictions"],
                n_boot=0,
                random_state=random_state + perm_idx + 1,
            )
            null_scores.append(_lookup_metric_value(perm_pooled, primary_classifier, metric))
        except Exception as exc:
            logger.warning("Retrain permutation %d failed for %s: %s", perm_idx + 1, primary_classifier, exc)
            null_scores.append(float("nan"))

    return retrain_permutation_summary(
        classifier=primary_classifier,
        metric=metric,
        observed=observed,
        null_scores=null_scores,
    )


def _lookup_metric_value(metrics: pd.DataFrame, classifier: str, metric: str) -> float:
    if metrics.empty or "classifier" not in metrics.columns:
        return float("nan")
    metric_col = str(metric)
    if metric_col not in metrics.columns and f"{metric_col}_mean" in metrics.columns:
        metric_col = f"{metric_col}_mean"
    if metric_col not in metrics.columns:
        return float("nan")
    row = metrics[metrics["classifier"].astype(str) == str(classifier)]
    if row.empty:
        return float("nan")
    return float(row[metric_col].iloc[0])


def _build_model_generalization_summary(
    aggregate: pd.DataFrame,
    holdout_metrics: pd.DataFrame,
    biomarker: pd.DataFrame,
    pooled_oof: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if aggregate.empty or holdout_metrics.empty:
        return pd.DataFrame()

    cols = [
        "classifier",
        "sensitivity_mean",
        "specificity_mean",
        "auc_mean",
        "balanced_accuracy_mean",
        "mcc_mean",
    ]
    cv_cols = [col for col in cols if col in aggregate.columns]
    ho_cols = [col for col in cols if col in holdout_metrics.columns]
    summary = aggregate[cv_cols].merge(
        holdout_metrics[ho_cols],
        on="classifier",
        how="inner",
        suffixes=("_cv", "_holdout"),
    )
    if summary.empty:
        return summary

    if {"auc_mean_cv", "auc_mean_holdout"}.issubset(summary.columns):
        summary["cv_holdout_auc_gap"] = summary["auc_mean_cv"] - summary["auc_mean_holdout"]
        summary["abs_cv_holdout_auc_gap"] = summary["cv_holdout_auc_gap"].abs()

    if pooled_oof is not None and not pooled_oof.empty and "classifier" in pooled_oof.columns:
        pooled_cols = [
            "classifier",
            "balanced_accuracy",
            "auc",
            "sensitivity",
            "specificity",
            "mcc",
            "brier",
        ]
        pooled_cols = [col for col in pooled_cols if col in pooled_oof.columns]
        pooled = pooled_oof[pooled_cols].rename(
            columns={col: f"{col}_pooled_oof" for col in pooled_cols if col != "classifier"}
        )
        summary = summary.merge(pooled, on="classifier", how="left")
        if {"balanced_accuracy_pooled_oof", "balanced_accuracy_mean_holdout"}.issubset(summary.columns):
            summary["pooled_oof_holdout_bacc_gap"] = (
                summary["balanced_accuracy_pooled_oof"] - summary["balanced_accuracy_mean_holdout"]
            )
            summary["abs_pooled_oof_holdout_bacc_gap"] = summary["pooled_oof_holdout_bacc_gap"].abs()
        if {"auc_pooled_oof", "auc_mean_holdout"}.issubset(summary.columns):
            summary["pooled_oof_holdout_auc_gap"] = summary["auc_pooled_oof"] - summary["auc_mean_holdout"]
            summary["abs_pooled_oof_holdout_auc_gap"] = summary["pooled_oof_holdout_auc_gap"].abs()

    stability = np.nan
    if not biomarker.empty and "selection_frequency" in biomarker.columns:
        top = biomarker["selection_frequency"].to_numpy(dtype=float)
        stability = float(np.nanmean(np.sort(top)[-min(10, top.size):])) if top.size else np.nan
    summary["feature_stability_top10_mean"] = stability

    preferred = {
        "pls_da": 0,
        "logreg_elastic_net": 1,
        "svm_linear": 2,
        "random_forest": 3,
        "extra_trees": 4,
    }
    summary["_order"] = summary["classifier"].map(lambda name: preferred.get(str(name), 99))
    summary = summary.sort_values(["_order", "classifier"]).drop(columns=["_order"]).reset_index(drop=True)
    return summary


def write_deliverables(
    output_dir: Path,
    area_df: pd.DataFrame,
    classifier_feature_matrix: pd.DataFrame,
    descriptive_feature_matrix: pd.DataFrame,
    cv_out: dict,
    fold_metrics: pd.DataFrame,
    aggregate: pd.DataFrame,
    pooled_oof: pd.DataFrame | None,
    permutation: pd.DataFrame,
    comparison: dict,
    biomarker: pd.DataFrame,
    secondary: dict,
    cv_config_snapshot: dict,
    sex_metrics: pd.DataFrame | None = None,
    calibration: pd.DataFrame | None = None,
    track_ablation: pd.DataFrame | None = None,
    ch_audit: pd.DataFrame | None = None,
    fold_subject_balance: pd.DataFrame | None = None,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_dir = output_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}
    split_index = int(cv_config_snapshot.get("resolved", {}).get("split_index", CANONICAL_FP_CHANNELS))
    feature_metadata = _build_output_feature_metadata(
        classifier_feature_matrix=classifier_feature_matrix,
        biomarker=biomarker,
        selection_trace=cv_out.get("selection_trace", pd.DataFrame()),
        shap_df=cv_out.get("shap", pd.DataFrame()),
        shap_local=cv_out.get("shap_local", pd.DataFrame()),
        split_index=split_index,
    )
    feature_dictionary_csv = output_dir / "feature_dictionary.csv"
    feature_metadata.to_csv(feature_dictionary_csv, index=False)
    artifacts["feature_dictionary_csv"] = str(feature_dictionary_csv)

    # --- Archive tier: intermediate / diagnostic files (not needed for publication) ---
    classifier_feature_csv = archive_dir / "classifier_feature_matrix_descriptive.csv"
    classifier_feature_matrix.to_csv(classifier_feature_csv, index=False)
    artifacts["classifier_feature_matrix_csv"] = str(classifier_feature_csv)

    feature_csv = archive_dir / "feature_matrix_descriptive.csv"
    descriptive_feature_matrix.to_csv(feature_csv, index=False)
    artifacts["feature_matrix_descriptive_csv"] = str(feature_csv)

    feature_notice = archive_dir / "feature_matrix_notice.txt"
    feature_notice.write_text(
        "classifier_feature_matrix_descriptive.csv and feature_matrix_descriptive.csv are generated "
        "from full-cohort feature fitting and are for descriptive/reporting use only.\n"
        "Do not treat either matrix as fold-wise CV feature space.\n"
        "Secondary modeling in Phase 8 is re-fit per fold with train-only features.\n",
        encoding="utf-8",
    )
    artifacts["feature_matrix_notice_txt"] = str(feature_notice)

    subject_vectors = aggregate_subject_vectors(area_df)
    spectra_npz = archive_dir / "subject_spectra.npz"
    subject_means = np.vstack(subject_vectors["mean_vector"].to_numpy())
    np.savez_compressed(
        spectra_npz,
        subject_ids=subject_vectors["subject_id"].astype(str).to_numpy(),
        subject_means=subject_means,
        channel_index=np.arange(subject_means.shape[1], dtype=int),
    )
    artifacts["subject_spectra_npz"] = str(spectra_npz)

    preproc_log = archive_dir / "preprocessing_log.csv"
    n_channels = area_df["area_mean"].map(lambda v: len(np.asarray(v))).astype(int)
    preproc_cols = ["subject_id", "slide_number", "area_id", "n_valid_pixels"]
    preproc_df = area_df[preproc_cols].copy()
    preproc_df["n_channels"] = n_channels
    preproc_df.to_csv(preproc_log, index=False)
    artifacts["preprocessing_log_csv"] = str(preproc_log)

    cv_results = archive_dir / "cv_results.csv"
    cv_out.get("fold_results", pd.DataFrame()).to_csv(cv_results, index=False)
    artifacts["cv_results_csv"] = str(cv_results)

    selection_trace_csv = archive_dir / "selection_trace.csv"
    selection_trace = enrich_feature_table(
        cv_out.get("selection_trace", pd.DataFrame()),
        metadata=feature_metadata,
        split_index=split_index,
    )
    selection_trace.to_csv(selection_trace_csv, index=False)
    artifacts["selection_trace_csv"] = str(selection_trace_csv)

    fold_metrics_csv = output_dir / "fold_metrics.csv"
    fold_metrics.to_csv(fold_metrics_csv, index=False)
    artifacts["fold_metrics_csv"] = str(fold_metrics_csv)

    agg_metrics_csv = output_dir / "aggregated_metrics.csv"
    aggregate.to_csv(agg_metrics_csv, index=False)
    artifacts["aggregated_metrics_csv"] = str(agg_metrics_csv)

    if pooled_oof is not None and not pooled_oof.empty:
        pooled_oof_csv = output_dir / "pooled_oof_metrics.csv"
        pooled_oof.to_csv(pooled_oof_csv, index=False)
        artifacts["pooled_oof_metrics_csv"] = str(pooled_oof_csv)

    pred_csv = output_dir / "predictions.csv"
    cv_out.get("predictions", pd.DataFrame()).to_csv(pred_csv, index=False)
    artifacts["predictions_csv"] = str(pred_csv)

    cm_dir = output_dir / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)
    preds = cv_out.get("predictions", pd.DataFrame())
    if not preds.empty:
        for clf, grp in preds.groupby("classifier", sort=False):
            cm = confusion_matrix(grp["y_true"].to_numpy(dtype=int), grp["y_pred"].to_numpy(dtype=int), labels=[0, 1])
            cm_df = pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"])
            cm_path = cm_dir / f"{clf}_confusion.csv"
            cm_df.to_csv(cm_path)
    artifacts["confusion_matrices_dir"] = str(cm_dir)

    perm_npz = output_dir / "permutation_test.npz"
    if permutation.empty:
        np.savez_compressed(perm_npz)
    else:
        np.savez_compressed(
            perm_npz,
            classifier=permutation["classifier"].astype(str).to_numpy(),
            observed=permutation["observed"].to_numpy(dtype=float),
            p_value=permutation["p_value"].to_numpy(dtype=float),
            null_mean=permutation["null_mean"].to_numpy(dtype=float),
            null_std=permutation["null_std"].to_numpy(dtype=float),
            n_permutations=permutation["n_permutations"].to_numpy(dtype=int)
            if "n_permutations" in permutation.columns
            else np.full(len(permutation), np.nan),
            mode=permutation["mode"].astype(str).to_numpy()
            if "mode" in permutation.columns
            else np.array(["posthoc_sanity"] * len(permutation)),
        )
    artifacts["permutation_test_npz"] = str(perm_npz)

    comparison_csv = output_dir / "classifier_comparison.csv"
    pairwise = comparison.get("nemenyi", {}).get("pairwise", pd.DataFrame())
    if isinstance(pairwise, pd.DataFrame) and not pairwise.empty:
        comp_df = pairwise.copy()
    else:
        comp_df = pd.DataFrame()
    friedman = comparison.get("friedman", {})
    if friedman:
        for key, value in friedman.items():
            comp_df[key] = value
    comp_df.to_csv(comparison_csv, index=False)
    artifacts["classifier_comparison_csv"] = str(comparison_csv)

    importance_csv = output_dir / "feature_importance.csv"
    biomarker_enriched = enrich_feature_table(
        biomarker,
        metadata=feature_metadata,
        split_index=split_index,
    )
    biomarker_enriched.to_csv(importance_csv, index=False)
    artifacts["feature_importance_csv"] = str(importance_csv)

    stringency_csv = archive_dir / "stringency_distribution.csv"
    fold = cv_out.get("fold_results", pd.DataFrame())
    if not fold.empty and "n_features_after_selection" in fold.columns:
        stringency = (
            fold.groupby(["classifier", "n_features_after_selection"], dropna=False)
            .size()
            .reset_index(name="count")
        )
    else:
        stringency = pd.DataFrame(columns=["classifier", "n_features_after_selection", "count"])
    stringency.to_csv(stringency_csv, index=False)
    artifacts["stringency_distribution_csv"] = str(stringency_csv)

    replicate_audit_csv = archive_dir / "replicate_weighting_audit.csv"
    replicate_audit = (
        area_df.groupby("subject_id", sort=False)
        .agg(
            n_areas=("area_id", "nunique"),
            n_slides=("slide_number", "nunique"),
            n_valid_pixels_total=("n_valid_pixels", "sum"),
            n_valid_pixels_mean=("n_valid_pixels", "mean"),
        )
        .reset_index()
    )
    total_pixels = float(replicate_audit["n_valid_pixels_total"].sum())
    if total_pixels > 0:
        replicate_audit["pixel_weight_share"] = replicate_audit["n_valid_pixels_total"] / total_pixels
    else:
        replicate_audit["pixel_weight_share"] = 0.0
    replicate_audit.to_csv(replicate_audit_csv, index=False)
    artifacts["replicate_weighting_audit_csv"] = str(replicate_audit_csv)

    spectrum_png = output_dir / "top_features_spectrum.png"
    _write_spectrum_plot(
        area_df=area_df,
        subject_vectors=subject_vectors,
        biomarker=biomarker_enriched,
        output_path=spectrum_png,
        split_index=split_index,
    )
    artifacts["top_features_spectrum_png"] = str(spectrum_png)

    boxplot_png = output_dir / "top_features_boxplots.png"
    _write_boxplot(descriptive_feature_matrix, biomarker_enriched, boxplot_png)
    artifacts["top_features_boxplots_png"] = str(boxplot_png)

    shap_png = output_dir / "shap_summary.png"
    shap_df = enrich_feature_table(
        cv_out.get("shap", pd.DataFrame()),
        metadata=feature_metadata,
        split_index=split_index,
    )
    _write_shap_plot(shap_df, shap_png)
    artifacts["shap_summary_png"] = str(shap_png)

    shap_csv = output_dir / "shap_values.csv"
    if not shap_df.empty:
        shap_df.to_csv(shap_csv, index=False)
        artifacts["shap_values_csv"] = str(shap_csv)

    shap_local = enrich_feature_table(
        cv_out.get("shap_local", pd.DataFrame()),
        metadata=feature_metadata,
        split_index=split_index,
    )
    if not shap_local.empty:
        shap_local_csv = archive_dir / "shap_local_values.csv"
        shap_local.to_csv(shap_local_csv, index=False)
        artifacts["shap_local_values_csv"] = str(shap_local_csv)
        force_artifacts = _write_shap_force_plots(shap_local, output_dir)
        artifacts.update(force_artifacts)

    secondary_dir = output_dir / "secondary"
    secondary_dir.mkdir(parents=True, exist_ok=True)
    _write_secondary_outputs(secondary, secondary_dir)
    artifacts["secondary_dir"] = str(secondary_dir)

    cv_snapshot_json = output_dir / "cv_config_snapshot.json"
    cv_snapshot_json.write_text(json.dumps(cv_config_snapshot, indent=2, sort_keys=True), encoding="utf-8")
    artifacts["cv_config_snapshot_json"] = str(cv_snapshot_json)

    if sex_metrics is not None and not sex_metrics.empty:
        sex_csv = output_dir / "sex_stratified_metrics.csv"
        sex_metrics.to_csv(sex_csv, index=False)
        artifacts["sex_stratified_metrics_csv"] = str(sex_csv)

    if calibration is not None and not calibration.empty:
        cal_csv = output_dir / "calibration_metrics.csv"
        calibration.to_csv(cal_csv, index=False)
        artifacts["calibration_metrics_csv"] = str(cal_csv)

    if track_ablation is not None and not track_ablation.empty:
        abl_csv = output_dir / "feature_track_ablation.csv"
        track_ablation.to_csv(abl_csv, index=False)
        artifacts["feature_track_ablation_csv"] = str(abl_csv)

    if ch_audit is not None and not ch_audit.empty:
        audit_csv = archive_dir / "ch_lipid_state_audit.csv"
        ch_audit.to_csv(audit_csv, index=False)
        artifacts["ch_lipid_state_audit_csv"] = str(audit_csv)

    if fold_subject_balance is not None and not fold_subject_balance.empty:
        balance_csv = archive_dir / "fold_subject_balance.csv"
        fold_subject_balance.to_csv(balance_csv, index=False)
        artifacts["fold_subject_balance_csv"] = str(balance_csv)

    return artifacts


def _build_output_feature_metadata(
    classifier_feature_matrix: pd.DataFrame,
    biomarker: pd.DataFrame,
    selection_trace: pd.DataFrame,
    shap_df: pd.DataFrame,
    shap_local: pd.DataFrame,
    split_index: int,
) -> pd.DataFrame:
    excluded = {
        "subject_id",
        "subject_id_raw",
        "is_control",
        "gender",
        "condition",
    }
    feature_ids: list[str] = [
        str(col)
        for col in classifier_feature_matrix.columns
        if str(col) not in excluded
    ]
    for table in (biomarker, selection_trace, shap_df, shap_local):
        if table is not None and not table.empty and "feature" in table.columns:
            feature_ids.extend(table["feature"].dropna().astype(str).tolist())
    seen: set[str] = set()
    ordered = []
    for feature_id in feature_ids:
        if feature_id not in seen:
            seen.add(feature_id)
            ordered.append(feature_id)
    return build_feature_metadata(ordered, split_index=split_index)


def _write_shap_force_plots(shap_local: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    required = {"fold", "classifier", "subject_id", "feature", "shap_value", "base_value", "model_output_value"}
    if shap_local.empty or not required.issubset(shap_local.columns):
        return {}

    force_cfg = config.get("interpretability", {}).get("shap", {}).get("force_plots", {})
    if not bool(force_cfg.get("enabled", True)):
        return {}

    requested = str(force_cfg.get("classifier", "extra_trees"))
    max_features = int(force_cfg.get("max_features", 15))
    classifiers = shap_local["classifier"].astype(str).unique().tolist()
    classifier = requested if requested in classifiers else classifiers[0]
    selected = shap_local[shap_local["classifier"].astype(str) == classifier].copy()
    if selected.empty:
        return {}

    root = output_dir / "shap_force_plots" / _safe_filename(classifier)
    root.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict] = []
    for (fold, subject_id), grp in selected.groupby(["fold", "subject_id"], sort=False):
        group = grp.copy()
        group["abs_shap"] = group["shap_value"].abs()
        top = group.sort_values("abs_shap", ascending=False).head(max(1, max_features))
        labels = (
            top["short_label"].fillna(top["feature"]).astype(str).tolist()
            if "short_label" in top.columns
            else top["feature"].astype(str).tolist()
        )
        values = top["shap_value"].to_numpy(dtype=float)
        feature_values = (
            top["feature_value"].to_numpy(dtype=float)
            if "feature_value" in top.columns
            else np.zeros(len(top), dtype=float)
        )
        base_value = float(top["base_value"].iloc[0])
        model_output_value = float(top["model_output_value"].iloc[0])
        shap_sum_value = float(top["shap_sum_value"].iloc[0]) if "shap_sum_value" in top.columns else float("nan")

        fold_dir = root / f"fold_{int(fold):03d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        html_path = fold_dir / f"subject_{_safe_filename(subject_id)}.html"
        _save_force_plot_html(
            html_path=html_path,
            base_value=base_value,
            shap_values=values,
            feature_values=feature_values,
            feature_names=labels,
            model_output_value=model_output_value,
        )
        index_rows.append(
            {
                "classifier": classifier,
                "fold": int(fold),
                "subject_id": str(subject_id),
                "path": str(html_path.relative_to(output_dir)),
                "n_features_displayed": int(len(top)),
                "base_value": base_value,
                "model_output_value": model_output_value,
                "shap_sum_value": shap_sum_value,
                "explained_output": "P(disease/stroke)",
            }
        )

    index = pd.DataFrame(index_rows)
    index_csv = output_dir / "shap_force_index.csv"
    index.to_csv(index_csv, index=False)
    return {
        "shap_force_plots_dir": str(root),
        "shap_force_index_csv": str(index_csv),
    }


def _save_force_plot_html(
    html_path: Path,
    base_value: float,
    shap_values: np.ndarray,
    feature_values: np.ndarray,
    feature_names: list[str],
    model_output_value: float,
) -> None:
    try:
        import shap

        force = shap.force_plot(
            base_value,
            shap_values,
            features=feature_values,
            feature_names=feature_names,
            matplotlib=False,
            show=False,
        )
        shap.save_html(str(html_path), force)
    except Exception as exc:
        rows = "\n".join(
            f"<tr><td>{name}</td><td>{value:.6g}</td><td>{shap_value:+.6g}</td></tr>"
            for name, value, shap_value in zip(feature_names, feature_values, shap_values, strict=False)
        )
        html_path.write_text(
            "\n".join(
                [
                    "<!doctype html><html><head><meta charset='utf-8'><title>SHAP force plot fallback</title></head><body>",
                    "<h1>SHAP force plot fallback</h1>",
                    f"<p>Interactive SHAP rendering failed: {exc}</p>",
                    f"<p>Explained output: P(disease/stroke); base={base_value:.6g}; model output={model_output_value:.6g}</p>",
                    "<table><thead><tr><th>Feature</th><th>Scaled value</th><th>SHAP value</th></tr></thead><tbody>",
                    rows,
                    "</tbody></table></body></html>",
                ]
            ),
            encoding="utf-8",
        )


def _safe_filename(value: object) -> str:
    token = str(value).strip()
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", token)
    return token.strip("._") or "unknown"


def _build_wavenumber_axis(n_channels: int, split_index: int) -> np.ndarray:
    """Return a cm⁻¹ wavenumber axis for the concatenated fingerprint+CH-stretch spectrum.

    Tries the live canonical region axes first; falls back to a linear approximation
    using the known boundary constants (fingerprint 734–1788, CH-stretch 2451–3197).
    """
    try:
        from src.fusion.area_pipeline import _CANONICAL_REGION_AXES  # type: ignore[attr-defined]

        fp = _CANONICAL_REGION_AXES.get("fingerprint")
        ch = _CANONICAL_REGION_AXES.get("ch_stretch")
        if fp is not None and ch is not None:
            axis = np.concatenate([np.asarray(fp, dtype=float), np.asarray(ch, dtype=float)])
            if len(axis) == n_channels:
                return axis
    except Exception:
        pass
    fp_len = min(split_index, n_channels)
    ch_len = n_channels - fp_len
    fp_start, fp_end = CANONICAL_REGION_BOUNDS["fingerprint"]
    ch_start, ch_end = CANONICAL_REGION_BOUNDS["ch_stretch"]
    fp_wn = np.linspace(fp_start, fp_end, fp_len) if fp_len > 0 else np.empty(0, dtype=float)
    ch_wn = np.linspace(ch_start, ch_end, ch_len) if ch_len > 0 else np.empty(0, dtype=float)
    return np.concatenate([fp_wn, ch_wn])


def _build_spectrum_highlight_regions(
    biomarker: pd.DataFrame,
    top_n: int = 15,
    point_half_width: float = 20.0,
) -> list[dict]:
    """Extract wavenumber highlight spans from an enriched biomarker table.

    Each span dict has keys: ``wn_lo``, ``wn_hi``, ``label``, ``weight`` (0–1).
    Spans are sorted by ``selection_frequency`` (descending) and deduplicated so
    that two regions whose centres are within 30 cm⁻¹ of each other are merged.
    Features with no spectral location (PCA, stat, CH-lipid state) are skipped.
    """
    if biomarker.empty:
        return []

    freq_col = "selection_frequency" if "selection_frequency" in biomarker.columns else None
    df = biomarker.copy()
    if freq_col:
        df = df.sort_values(freq_col, ascending=False)

    max_freq = float(df[freq_col].max()) if freq_col and len(df) > 0 else 1.0
    if max_freq <= 0.0:
        max_freq = 1.0

    regions: list[dict] = []
    seen_centers: list[float] = []

    for _, row in df.iterrows():
        wn_raw = row.get("wavenumber_cm1") if "wavenumber_cm1" in df.columns else float("nan")
        window_str = str(row.get("window_cm1") or "") if "window_cm1" in df.columns else ""
        label = str(row.get("short_label") or row.get("feature") or "").strip()
        weight = float(row.get(freq_col, 1.0)) / max_freq if freq_col else 1.0

        lo, hi = float("nan"), float("nan")

        # Band / ratio features: parse "lo-hi" from window_cm1.
        # For ratio features window_cm1 = "lo-hi / lo-hi" — take the numerator part.
        if window_str:
            primary = window_str.split("/")[0].strip()
            m = re.match(r"^([\d.]+)-([\d.]+)$", primary)
            if m:
                try:
                    lo, hi = float(m.group(1)), float(m.group(2))
                except ValueError:
                    pass

        # Peak features: use wavenumber_cm1 ± point_half_width
        if not (np.isfinite(lo) and np.isfinite(hi)):
            try:
                wn = float(wn_raw)
            except (TypeError, ValueError):
                wn = float("nan")
            if np.isfinite(wn):
                lo, hi = wn - point_half_width, wn + point_half_width

        # Features with no spectral position (PCA, stat, CH-lipid): skip
        if not (np.isfinite(lo) and np.isfinite(hi)):
            continue

        center = (lo + hi) / 2.0
        if any(abs(center - c) < 30.0 for c in seen_centers):
            continue  # deduplicate nearby spans

        seen_centers.append(center)
        regions.append({"wn_lo": lo, "wn_hi": hi, "label": label, "weight": weight})

        if len(regions) >= top_n:
            break

    return regions


def _write_spectrum_plot(
    area_df: pd.DataFrame,
    subject_vectors: pd.DataFrame,
    biomarker: pd.DataFrame,
    output_path: Path,
    split_index: int = CANONICAL_FP_CHANNELS,
) -> None:
    merged = subject_vectors.merge(
        area_df[["subject_id", "is_control"]].drop_duplicates(subset=["subject_id"], keep="first"),
        on="subject_id",
        how="left",
    )

    if merged.empty:
        return

    stacked = np.vstack(merged["mean_vector"].to_numpy())
    labels = merged["is_control"].to_numpy()
    control_value = _infer_control_value(labels)

    control_mask = labels == control_value
    disease_mask = ~control_mask

    if not np.any(control_mask) or not np.any(disease_mask):
        return

    control_mean = np.nanmean(stacked[control_mask], axis=0)
    disease_mean = np.nanmean(stacked[disease_mask], axis=0)

    wavenumbers = _build_wavenumber_axis(stacked.shape[1], split_index)
    highlight_regions = _build_spectrum_highlight_regions(biomarker)
    fig, _ = plot_spectrum_highlights(
        wavenumbers=wavenumbers,
        control_mean=control_mean,
        disease_mean=disease_mean,
        highlighted_indices=[],
        output_path=output_path,
        highlight_regions=highlight_regions,
        split_index=split_index,
    )
    fig.clf()


def _write_boxplot(feature_matrix: pd.DataFrame, biomarker: pd.DataFrame, output_path: Path) -> None:
    if "is_control" not in feature_matrix.columns:
        return

    if biomarker.empty or "feature" not in biomarker.columns:
        cols = [c for c in feature_matrix.columns if c.startswith("track_")][:10]
    else:
        cols = [c for c in biomarker["feature"].astype(str).tolist() if c in feature_matrix.columns][:10]

    if not cols:
        return

    fig, _ = plot_top_feature_boxplots(
        feature_df=feature_matrix,
        label_col="is_control",
        feature_cols=cols,
        max_features=min(10, len(cols)),
        output_path=output_path,
    )
    fig.clf()


def _write_shap_plot(shap_df: pd.DataFrame, output_path: Path) -> None:
    if shap_df.empty or "mean_abs_shap" not in shap_df.columns:
        return

    fig, _ = plot_shap_summary(
        mean_abs_shap=shap_df,
        max_features=20,
        output_path=output_path,
    )
    fig.clf()


def _write_secondary_outputs(secondary: dict, secondary_dir: Path) -> None:
    moca = secondary.get("moca", {}) if isinstance(secondary, dict) else {}
    moca_metrics = pd.DataFrame([moca.get("metrics", {})]) if moca.get("status") == "ok" else pd.DataFrame()
    if not moca_metrics.empty:
        moca_metrics["method"] = moca.get("method")
        moca_metrics["n_subjects"] = moca.get("n_subjects")
        moca_metrics["feature_source"] = moca.get("feature_source")
    moca_metrics.to_csv(secondary_dir / "moca_prediction.csv", index=False)

    aspects = secondary.get("aspects", pd.DataFrame()) if isinstance(secondary, dict) else pd.DataFrame()
    if isinstance(aspects, pd.DataFrame):
        aspects.to_csv(secondary_dir / "aspects_correlation.csv", index=False)

    subtype = secondary.get("subtype", {}) if isinstance(secondary, dict) else {}
    if isinstance(subtype, dict):
        subtype_df = subtype.get("results", pd.DataFrame())
        if isinstance(subtype_df, pd.DataFrame):
            if subtype.get("status") == "ok":
                subtype_df = subtype_df.copy()
                subtype_df["n_classes"] = subtype.get("n_classes")
                subtype_df["n_subjects"] = subtype.get("n_subjects")
                subtype_df["mean_balanced_accuracy"] = subtype.get("mean_balanced_accuracy")
                subtype_df["std_balanced_accuracy"] = subtype.get("std_balanced_accuracy")
                subtype_df["feature_source"] = subtype.get("feature_source")
            else:
                subtype_df = pd.DataFrame([{"status": subtype.get("status"), "reason": subtype.get("reason")}])
            subtype_df.to_csv(secondary_dir / "subtype_exploration.csv", index=False)


def _infer_control_value(labels: np.ndarray) -> int | bool:
    vals = pd.Series(labels).dropna().unique().tolist()
    if set(vals).issubset({0, 1}):
        return 1
    if set(vals).issubset({False, True}):
        return True
    return vals[0] if vals else 1


# ---------------------------------------------------------------------------
# Area-store caching helpers
# ---------------------------------------------------------------------------


def _compute_index_fingerprint(file_index: pd.DataFrame | None) -> dict:
    if file_index is None or file_index.empty:
        return {"row_hash": "", "file_signatures": []}

    relevant_cols = [
        "subject_id",
        "slide_number",
        "area_id",
        "spectral_range_cm-1",
        "measurement_number",
        "filepath",
    ]
    available_cols = [col for col in relevant_cols if col in file_index.columns]
    relevant = file_index[available_cols].copy()
    relevant = relevant.sort_values(available_cols, kind="stable").reset_index(drop=True)
    row_hash = hashlib.sha256(
        pd.util.hash_pandas_object(relevant.astype(str), index=False).to_numpy().tobytes()
    ).hexdigest()

    file_signatures = []
    if "filepath" in file_index.columns:
        for filepath in file_index["filepath"].dropna().astype(str).unique().tolist():
            path = Path(filepath)
            try:
                stat = path.stat()
                file_signatures.append((str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size)))
            except OSError:
                file_signatures.append((str(path), None, None))
    file_signatures = sorted(file_signatures, key=lambda item: item[0])
    return {"row_hash": row_hash, "file_signatures": file_signatures}


def _compute_preprocessing_config_hash(file_index: pd.DataFrame | None = None) -> str:
    """Hash preprocessing config and indexed file identity for cache invalidation."""
    relevant = {
        "schema_version": "area_store_v5",

        "preprocessing": config.get("preprocessing", {}),
        "ml_spectral_scaler": config.get("ml_spectral_scaler", {}),
        "index_fingerprint": _compute_index_fingerprint(file_index),
    }
    payload = json.dumps(relevant, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _load_cached_area_df(
    cache_path: Path,
    config_hash: str,
    force_reprocess: bool,
) -> pd.DataFrame | None:
    """Load area_df from cache if valid and not forced to reprocess."""
    if force_reprocess or not cache_path.exists():
        return None

    hash_path = cache_path.with_suffix(".hash")
    if not hash_path.exists():
        return None

    try:
        # Legacy caches that stored mode metadata must be rebuilt.
        with np.load(cache_path, allow_pickle=True) as data:
            if "instrument_modes" in data:
                logger.info("Legacy area store with mode metadata detected; deleting cache.")
                cache_path.unlink(missing_ok=True)
                hash_path.unlink(missing_ok=True)
                return None

        stored_hash = hash_path.read_text(encoding="utf-8").strip()
        if stored_hash != config_hash:
            logger.info("Cache invalidated: preprocessing config changed.")
            return None

        store = AreaStore.load_npz(cache_path)
        area_df = store.to_dataframe()
        if area_df.empty:
            return None
        logger.info("Loaded cached area store: %d areas", len(area_df))
        return area_df
    except Exception as exc:
        logger.warning("Failed to load cached area store: %s", exc)
        return None


def _save_cached_area_df(
    area_df: pd.DataFrame,
    cache_path: Path,
    config_hash: str,
) -> None:
    """Persist area_df to compressed NPZ with a sidecar config hash."""
    try:
        store = AreaStore()
        ch_col = "ch_pixels" if "ch_pixels" in area_df.columns else None
        for _, row in area_df.iterrows():
            ch_arr = np.asarray(row[ch_col]) if ch_col and row.get(ch_col) is not None else None
            store.add(
                subject_id=str(row["subject_id"]),
                slide_number=str(row.get("slide_number", "")),
                area_id=str(row.get("area_id", "")),
                area_mean=np.asarray(row["area_mean"], dtype=float),
                area_std=np.asarray(row.get("area_std", np.full_like(row["area_mean"], np.nan)), dtype=float),
                n_valid_pixels=int(row.get("n_valid_pixels", 0)),
                area_mean_bc=(
                    None
                    if row.get("area_mean_bc") is None
                    else np.asarray(row.get("area_mean_bc"), dtype=float)
                ),
                ch_pixels=ch_arr if (ch_arr is not None and ch_arr.size > 0) else None,
            )
        store.save_npz(cache_path)
        cache_path.with_suffix(".hash").write_text(config_hash, encoding="utf-8")
        logger.info("Cached area store written to %s", cache_path)
    except Exception as exc:
        logger.warning("Failed to cache area store: %s", exc)


def main() -> None:
    setup_logging()
    args = parse_args()

    logger.info("=== Running Full Pipeline ===")
    enabled_classifiers = (
        [n.strip() for n in args.classifiers.split(",") if n.strip()]
        if args.classifiers
        else None
    )
    out = run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_subjects=args.max_subjects,
        skip_cv=args.skip_cv,
        skip_secondary=args.skip_secondary,
        skip_holdout=args.skip_holdout,
        n_permutations=args.n_permutations,
        force_reprocess=args.force_reprocess,
        enabled_classifiers=enabled_classifiers,
    )
    logger.info("Pipeline outputs written to: %s", out["output_dir"])


if __name__ == "__main__":
    main()
