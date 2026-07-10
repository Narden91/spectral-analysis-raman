"""Phase L: descriptive stability ranking for fixed post hoc panel features."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import config, get_confirmatory_panel_dir, get_spectral_interpretability_dir
from src.analysis.io_utils import load_csv_or_empty, write_csv_atomic

logger = logging.getLogger(__name__)


def _cfg(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    return config if cfg is None else cfg


def _panel_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("analysis", {}).get("preregistered_panel", {})


def _empty_ranking() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "rank",
            "feature_name",
            "feature_family",
            "is_anchor_feature",
            "max_abs_cliffs_delta",
            "effect_size_flag",
            "bootstrap_ci_excludes_zero",
            "permutation_flag",
            "endpoint_consistency_flag",
            "robust_score",
            "evidence_tier",
        ]
    )


def _ci_excludes_zero(row: pd.Series) -> bool:
    lower = pd.to_numeric(pd.Series([row.get("bootstrap_ci_lower")]), errors="coerce").iloc[0]
    upper = pd.to_numeric(pd.Series([row.get("bootstrap_ci_upper")]), errors="coerce").iloc[0]
    return bool(np.isfinite(lower) and np.isfinite(upper) and (lower > 0 or upper < 0))


def _tier(score: int) -> str:
    if score >= 3:
        return "strong_trend"
    if score == 2:
        return "moderate_trend"
    return "weak_or_unstable"


def compute_robustness_scores(
    contrasts: pd.DataFrame,
    consistency: pd.DataFrame,
    *,
    effect_size_threshold: float,
    perm_p_threshold: float,
    anchor_feature: str,
) -> pd.DataFrame:
    """Aggregate contrast and endpoint-consistency evidence into 0-4 scores."""
    if contrasts.empty or "feature_name" not in contrasts.columns:
        return _empty_ranking()

    usable = contrasts.copy()
    if "status" in usable.columns:
        usable = usable.loc[usable["status"].astype(str) == "ok"].copy()
    if usable.empty:
        return _empty_ranking()

    consistency_flags = {}
    if not consistency.empty and {"feature_name", "consistent"}.issubset(consistency.columns):
        consistency_flags = {
            str(row["feature_name"]): bool(row["consistent"])
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
        consistency_flag = bool(consistency_flags.get(str(feature), False))
        robust_score = int(effect_flag) + int(ci_flag) + int(perm_flag) + int(consistency_flag)
        feature_family = str(group["feature_family"].dropna().iloc[0]) if "feature_family" in group and group["feature_family"].notna().any() else "unknown"
        rows.append(
            {
                "feature_name": feature,
                "feature_family": feature_family,
                "is_anchor_feature": str(feature) == str(anchor_feature),
                "max_abs_cliffs_delta": max_abs_delta,
                "effect_size_flag": effect_flag,
                "bootstrap_ci_excludes_zero": ci_flag,
                "permutation_flag": perm_flag,
                "endpoint_consistency_flag": consistency_flag,
                "robust_score": robust_score,
                "evidence_tier": _tier(robust_score),
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return _empty_ranking()
    result = result.sort_values(
        ["robust_score", "max_abs_cliffs_delta", "feature_name"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    return result


def run_robustness_ranking(
    *,
    contrasts_path: str | Path | None = None,
    consistency_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write Phase L feature robustness ranking."""
    cfg = _cfg(cfg)
    phase_cfg = cfg.get("analysis", {}).get("robustness_ranking", {})
    if not bool(phase_cfg.get("enabled", True)):
        return {}

    panel = _panel_cfg(cfg)
    robustness = panel.get("robustness", {})
    target_dir = Path(output_dir) if output_dir is not None else get_confirmatory_panel_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    contrasts = load_csv_or_empty(contrasts_path or target_dir / "panel_contrast_tests.csv")
    consistency = load_csv_or_empty(consistency_path or get_spectral_interpretability_dir("consistency") / "endpoint_consistency.csv")
    ranking = compute_robustness_scores(
        contrasts,
        consistency,
        effect_size_threshold=float(robustness.get("effect_size_threshold", 0.147)),
        perm_p_threshold=float(robustness.get("perm_p_threshold", 0.10)),
        anchor_feature=str(panel.get("primary_hypothesis", {}).get("feature", "oxidative_index")),
    )

    output_path = target_dir / "feature_robustness_ranking.csv"
    write_csv_atomic(ranking, output_path, index=False)
    logger.info("Robustness ranking written to %s", output_path)
    return {"feature_robustness_ranking": output_path}
