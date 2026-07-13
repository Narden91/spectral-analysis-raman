from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

EDA_ANALYSIS_MODES = ("disease_vs_control", "disease_only")
_LEGACY_SUBJECT_GROUP_MAP = {
    "disease_only": "disease_only",
}


def _normalize_mode(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not normalized:
        return None
    if normalized not in EDA_ANALYSIS_MODES:
        raise ValueError(
            f"Invalid EDA analysis mode '{value}'. Valid options: {list(EDA_ANALYSIS_MODES)}"
        )
    return normalized


def _legacy_mode_from_subject_group(cfg: dict[str, Any]) -> str | None:
    subject_group = cfg.get("data", {}).get("subject_group")
    if subject_group is None:
        return None
    normalized = str(subject_group).strip().lower()
    if not normalized:
        return None
    mapped = _LEGACY_SUBJECT_GROUP_MAP.get(normalized)
    if mapped is None:
        raise ValueError(
            "Unsupported legacy data.subject_group value "
            f"'{subject_group}'. Use analysis.mode instead."
        )
    return mapped


def resolve_eda_analysis_mode(
    cfg: dict[str, Any],
    override: str | None = None,
) -> str:
    resolved_override = _normalize_mode(override)
    configured_mode = _normalize_mode(cfg.get("analysis", {}).get("mode"))
    legacy_mode = _legacy_mode_from_subject_group(cfg)

    if resolved_override is not None:
        if legacy_mode is not None and legacy_mode != resolved_override:
            raise ValueError(
                "Conflicting EDA mode configuration: "
                f"--analysis-mode={resolved_override} but data.subject_group implies {legacy_mode}"
            )
        return resolved_override

    if configured_mode is None:
        if legacy_mode is not None:
            logger.warning(
                "data.subject_group is deprecated for EDA mode selection; "
                "set analysis.mode explicitly."
            )
            return legacy_mode
        return "disease_vs_control"

    if legacy_mode is not None and legacy_mode != configured_mode:
        raise ValueError(
            "Conflicting EDA mode configuration: "
            f"analysis.mode={configured_mode} but data.subject_group implies {legacy_mode}"
        )
    return configured_mode


def apply_eda_analysis_mode(cfg: dict[str, Any], mode: str) -> None:
    cfg.setdefault("analysis", {})["mode"] = mode


def eda_output_root(mode: str) -> str:
    if mode == "disease_only":
        return "Results_intra_disease"
    if mode == "disease_vs_control":
        return "Results_control_vs_disease"
    raise ValueError(f"Unsupported EDA analysis mode '{mode}'")


def validate_eda_mode_config(
    cfg: dict[str, Any],
    mode: str,
    df_index: pd.DataFrame | None = None,
) -> None:
    intra_disease_cfg = cfg.get("analysis", {}).get("advanced", {}).get("intra_disease", {})
    intra_enabled = bool(intra_disease_cfg.get("enabled", False))
    group_configs = group_configs_for_eda_mode(mode)
    has_control_group = any(bool(group.get("is_control")) for group in group_configs.values())
    has_disease_group = any(not bool(group.get("is_control")) for group in group_configs.values())

    if mode == "disease_only" and not intra_enabled:
        raise ValueError(
            "EDA mode 'disease_only' requires analysis.advanced.intra_disease.enabled=true."
        )
    if mode == "disease_only" and has_control_group:
        raise ValueError(
            "EDA mode 'disease_only' must not define control cohorts in its group routing."
        )
    if mode == "disease_vs_control" and intra_enabled:
        raise ValueError(
            "EDA mode 'disease_vs_control' requires "
            "analysis.advanced.intra_disease.enabled=false."
        )
    if mode == "disease_vs_control" and (not has_control_group or not has_disease_group):
        raise ValueError(
            "EDA mode 'disease_vs_control' requires both control and disease cohorts in group routing."
        )

    if df_index is None:
        return

    missing_columns = {"subject_id", "is_control"} - set(df_index.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(
            "EDA mode validation requires file index columns: "
            f"{missing_list}."
        )

    control_mask = df_index["is_control"].fillna(False).astype(bool)
    control_subjects = int(df_index.loc[control_mask, "subject_id"].astype(str).nunique())
    disease_subjects = int(df_index.loc[~control_mask, "subject_id"].astype(str).nunique())

    if mode == "disease_only" and disease_subjects == 0:
        raise ValueError(
            "EDA mode 'disease_only' requires at least one disease subject in the selected data source."
        )
    if mode == "disease_vs_control" and (control_subjects == 0 or disease_subjects == 0):
        raise ValueError(
            "EDA mode 'disease_vs_control' requires both control and disease subjects "
            "in the selected data source."
        )


_SUPPORTED_THRESHOLD_METRICS = ("balanced_accuracy",)
_SUPPORTED_REPRESENTATIONS = ("whole", "band_panel")
_SUPPORTED_ML_MODELS = ("pls_da", "l1_logistic", "random_forest")


def validate_control_vs_disease_ml_config(cfg: dict[str, Any]) -> None:
    """Validate analysis.control_vs_disease_ml. No declarative schema in this repo."""
    ml_cfg = cfg.get("analysis", {}).get("control_vs_disease_ml", {})
    if not bool(ml_cfg.get("enabled", True)):
        return

    representations = ml_cfg.get("representations", [])
    if not representations:
        raise ValueError("analysis.control_vs_disease_ml.representations must be non-empty")
    unknown_representations = set(representations) - set(_SUPPORTED_REPRESENTATIONS)
    if unknown_representations:
        raise ValueError(
            f"analysis.control_vs_disease_ml.representations has unknown value(s): {sorted(unknown_representations)}. "
            f"Valid options: {list(_SUPPORTED_REPRESENTATIONS)}"
        )

    primary_model = ml_cfg.get("primary_model")
    if primary_model not in _SUPPORTED_ML_MODELS:
        raise ValueError(
            f"analysis.control_vs_disease_ml.primary_model must be one of {list(_SUPPORTED_ML_MODELS)}, got {primary_model!r}"
        )
    primary_representation = ml_cfg.get("primary_representation")
    if primary_representation not in representations:
        raise ValueError(
            "analysis.control_vs_disease_ml.primary_representation "
            f"({primary_representation!r}) must be one of the configured representations: {representations}"
        )

    threshold_metric = ml_cfg.get("threshold_metric")
    if threshold_metric not in _SUPPORTED_THRESHOLD_METRICS:
        raise ValueError(
            f"analysis.control_vs_disease_ml.threshold_metric must be one of {list(_SUPPORTED_THRESHOLD_METRICS)}, "
            f"got {threshold_metric!r}"
        )

    positive_int_fields = (
        "n_outer_splits",
        "n_outer_repeats",
        "n_inner_splits",
        "n_permutations",
        "n_bootstrap",
    )
    for field in positive_int_fields:
        value = ml_cfg.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"analysis.control_vs_disease_ml.{field} must be a positive integer, got {value!r}")

    grid_fields = (
        "pls_components_grid",
        "logistic_c_grid",
        "rf_n_estimators_grid",
        "rf_max_depth_grid",
    )
    for field in grid_fields:
        value = ml_cfg.get(field)
        if not value:
            raise ValueError(f"analysis.control_vs_disease_ml.{field} must be a non-empty list")


def filter_index_for_eda_mode(df_index: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "disease_only":
        return df_index.loc[~df_index["is_control"]].copy()
    if mode == "disease_vs_control":
        return df_index
    raise ValueError(f"Unsupported EDA analysis mode '{mode}'")


def group_configs_for_eda_mode(mode: str) -> dict[str, dict[str, Any]]:
    if mode == "disease_only":
        return {
            "female_disease": {"gender": "female", "is_control": False},
            "male_disease": {"gender": "male", "is_control": False},
        }
    if mode == "disease_vs_control":
        return {
            "female_disease": {"gender": "female", "is_control": False},
            "female_control": {"gender": "female", "is_control": True},
            "male_disease": {"gender": "male", "is_control": False},
            "male_control": {"gender": "male", "is_control": True},
        }
    raise ValueError(f"Unsupported EDA analysis mode '{mode}'")
