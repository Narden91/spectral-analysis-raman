"""Phase L: descriptive evidence flags for fixed post hoc panel features."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import get_confirmatory_panel_dir, get_spectral_interpretability_dir
from src import resolve_cfg as _cfg
from src.analysis.io_utils import load_csv_or_empty, write_csv_atomic

logger = logging.getLogger(__name__)


def _panel_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("analysis", {}).get("preregistered_panel", {})


def _empty_evidence_flags() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "feature_name",
            "feature_family",
            "is_anchor_feature",
            "max_abs_cliffs_delta",
            "effect_size_flag",
            "bootstrap_ci_excludes_zero",
            "permutation_flag",
            "endpoint_sign_summary",
        ]
    )


def _ci_excludes_zero(row: pd.Series) -> bool:
    lower = pd.to_numeric(pd.Series([row.get("bootstrap_ci_lower")]), errors="coerce").iloc[0]
    upper = pd.to_numeric(pd.Series([row.get("bootstrap_ci_upper")]), errors="coerce").iloc[0]
    return bool(np.isfinite(lower) and np.isfinite(upper) and (lower > 0 or upper < 0))


def compute_evidence_flags(
    contrasts: pd.DataFrame,
    consistency: pd.DataFrame,
    *,
    effect_size_threshold: float,
    perm_p_threshold: float,
    anchor_feature: str,
) -> pd.DataFrame:
    """Summarize dependent exploratory checks without creating a composite score."""
    if contrasts.empty or "feature_name" not in contrasts.columns:
        return _empty_evidence_flags()

    usable = contrasts.copy()
    if "status" in usable.columns:
        usable = usable.loc[usable["status"].astype(str) == "ok"].copy()
    if usable.empty:
        return _empty_evidence_flags()

    consistency_summaries: dict[str, str] = {}
    if not consistency.empty and "feature_name" in consistency.columns:
        consistency_summaries = {
            str(row["feature_name"]): str(row.get("endpoint_signs", "not_available"))
            for _, row in consistency.iterrows()
        }

    rows: list[dict[str, object]] = []
    for feature, group in usable.groupby("feature_name"):
        deltas = pd.to_numeric(group.get("cliffs_delta"), errors="coerce").abs()
        max_abs_delta = float(deltas.max()) if deltas.notna().any() else np.nan
        effect_flag = bool(np.isfinite(max_abs_delta) and max_abs_delta > effect_size_threshold)
        ci_flag = bool(group.apply(_ci_excludes_zero, axis=1).any())
        perm_values = pd.to_numeric(group.get("perm_p_value"), errors="coerce")
        perm_flag = bool((perm_values < perm_p_threshold).any())
        endpoint_sign_summary = consistency_summaries.get(str(feature), "not_available")
        has_feature_family = "feature_family" in group and group["feature_family"].notna().any()
        feature_family = str(group["feature_family"].dropna().iloc[0]) if has_feature_family else "unknown"
        rows.append(
            {
                "feature_name": feature,
                "feature_family": feature_family,
                "is_anchor_feature": str(feature) == str(anchor_feature),
                "max_abs_cliffs_delta": max_abs_delta,
                "effect_size_flag": effect_flag,
                "bootstrap_ci_excludes_zero": ci_flag,
                "permutation_flag": perm_flag,
                "endpoint_sign_summary": endpoint_sign_summary,
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return _empty_evidence_flags()
    return result.sort_values(["feature_name"], kind="stable").reset_index(drop=True)


def run_robustness_ranking(
    *,
    contrasts_path: str | Path | None = None,
    consistency_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write Phase L descriptive evidence flags."""
    cfg = _cfg(cfg)
    phase_cfg = cfg.get("analysis", {}).get("robustness_ranking", {})
    if not bool(phase_cfg.get("enabled", True)):
        return {}

    panel = _panel_cfg(cfg)
    robustness = panel.get("robustness", {})
    target_dir = Path(output_dir) if output_dir is not None else get_confirmatory_panel_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    contrasts = load_csv_or_empty(contrasts_path or target_dir / "panel_contrast_tests.csv")
    consistency = load_csv_or_empty(
        consistency_path or get_spectral_interpretability_dir("consistency") / "endpoint_consistency.csv"
    )
    evidence_flags = compute_evidence_flags(
        contrasts,
        consistency,
        effect_size_threshold=float(robustness.get("effect_size_threshold", 0.147)),
        perm_p_threshold=float(robustness.get("perm_p_threshold", 0.10)),
        anchor_feature=str(panel.get("primary_hypothesis", {}).get("feature", "oxidative_index")),
    )

    output_path = target_dir / "panel_evidence_flags.csv"
    write_csv_atomic(evidence_flags, output_path, index=False)
    logger.info("Panel evidence flags written to %s", output_path)
    return {"panel_evidence_flags": output_path}
