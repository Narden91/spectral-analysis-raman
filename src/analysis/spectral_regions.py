"""Candidate-region helpers and unified cluster-corrected outputs."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src import config, get_cross_region_phase_dir

logger = logging.getLogger(__name__)


def build_candidate_regions(
    tests: pd.DataFrame,
    *,
    group_columns: tuple[str, ...],
    effect_column: str,
    min_width_cm: float = 8.0,
    merge_gap_cm: float = 6.0,
    fallback_quantile: float = 0.9,
) -> pd.DataFrame:
    base_columns = list(group_columns) + [
        "spectral_block",
        "candidate_region_id",
        "selection_basis",
        "start_cm-1",
        "end_cm-1",
        "width_cm-1",
        "n_features",
        "min_fdr_q",
        "min_cluster_p_value",
        "n_fdr_significant",
        "n_cluster_significant",
        "max_abs_effect",
        "mean_abs_effect",
        "integrated_abs_effect",
    ]
    if tests.empty:
        return pd.DataFrame(columns=base_columns)

    working = tests.copy()
    if "status" in working.columns:
        working = working.loc[working["status"].eq("ok")].copy()
    if working.empty:
        return pd.DataFrame(columns=base_columns)

    rows: list[dict[str, object]] = []
    group_keys = list(group_columns) + ["spectral_block"]

    def append_region(window: pd.DataFrame, *, selection_basis: str, region_index: int) -> None:
        rows.append(
            {
                "spectral_block": str(window["spectral_block"].iloc[0]),
                "candidate_region_id": f"{window['spectral_block'].iloc[0]}_R{region_index}",
                "selection_basis": selection_basis,
                "start_cm-1": float(window["wavenumber"].iloc[0]),
                "end_cm-1": float(window["wavenumber"].iloc[-1]),
                "width_cm-1": float(window["wavenumber"].iloc[-1] - window["wavenumber"].iloc[0]),
                "n_features": int(len(window)),
                "min_fdr_q": float(window["fdr_q"].min()) if "fdr_q" in window.columns else np.nan,
                "min_cluster_p_value": (
                    float(window["cluster_p_value"].min())
                    if "cluster_p_value" in window.columns
                    else np.nan
                ),
                "n_fdr_significant": int(
                    window["significant_fdr"].astype("boolean").fillna(False).sum()
                ) if "significant_fdr" in window.columns else 0,
                "n_cluster_significant": int(
                    window["cluster_p_value"].lt(0.05).fillna(False).sum()
                ) if "cluster_p_value" in window.columns else 0,
                "max_abs_effect": float(window[effect_column].abs().max()),
                "mean_abs_effect": float(window[effect_column].abs().mean()),
                "integrated_abs_effect": float(
                    np.trapz(
                        window[effect_column].abs().to_numpy(dtype=float),
                        window["wavenumber"].to_numpy(dtype=float),
                    )
                ),
            }
        )

    for group_values, block_df in working.groupby(group_keys, sort=False):
        ordered_full = block_df.sort_values("wavenumber").reset_index(drop=True)
        sig_mask = pd.Series(False, index=ordered_full.index, dtype=bool)
        if "significant_fdr" in ordered_full.columns:
            sig_mask = sig_mask | ordered_full["significant_fdr"].astype("boolean").fillna(False).to_numpy(dtype=bool)
        if "cluster_p_value" in ordered_full.columns:
            sig_mask = sig_mask | ordered_full["cluster_p_value"].lt(0.05).fillna(False).to_numpy(dtype=bool)

        if sig_mask.any():
            ordered = ordered_full.loc[sig_mask].copy().reset_index(drop=True)
            selection_basis = "significant_or_cluster"
        else:
            abs_effect = ordered_full[effect_column].abs().to_numpy(dtype=float)
            finite = np.isfinite(abs_effect)
            if not finite.any():
                continue
            threshold = float(np.nanquantile(abs_effect[finite], fallback_quantile))
            fallback_mask = abs_effect >= threshold
            ordered = ordered_full.loc[fallback_mask].copy().reset_index(drop=True)
            selection_basis = "top_abs_effect_fallback"
            if ordered.empty:
                peak_index = int(np.nanargmax(abs_effect))
                peak_wavenumber = float(ordered_full.loc[peak_index, "wavenumber"])
                half_width = max(min_width_cm / 2.0, merge_gap_cm)
                ordered = ordered_full.loc[
                    ordered_full["wavenumber"].between(peak_wavenumber - half_width, peak_wavenumber + half_width)
                ].copy().reset_index(drop=True)
                selection_basis = "top_peak_window_fallback"

        start = 0
        region_index = 0
        for index in range(1, len(ordered) + 1):
            is_break = index == len(ordered) or (
                ordered.loc[index, "wavenumber"] - ordered.loc[index - 1, "wavenumber"]
            ) > merge_gap_cm
            if not is_break:
                continue
            window = ordered.iloc[start:index].copy()
            width = float(window["wavenumber"].iloc[-1] - window["wavenumber"].iloc[0])
            if width >= min_width_cm or len(window) > 1:
                region_index += 1
                append_region(window, selection_basis=selection_basis, region_index=region_index)
            start = index

        if region_index == 0:
            abs_effect = ordered_full[effect_column].abs().to_numpy(dtype=float)
            if np.isfinite(abs_effect).any():
                peak_index = int(np.nanargmax(abs_effect))
                peak_wavenumber = float(ordered_full.loc[peak_index, "wavenumber"])
                half_width = max(min_width_cm / 2.0, merge_gap_cm)
                window = ordered_full.loc[
                    ordered_full["wavenumber"].between(peak_wavenumber - half_width, peak_wavenumber + half_width)
                ].copy().reset_index(drop=True)
                region_index = 1
                append_region(window, selection_basis="top_peak_window_fallback", region_index=region_index)

        if region_index > 0:
            if not isinstance(group_values, tuple):
                group_values = (group_values,)
            for row in rows[-region_index:]:
                for key, value in zip(group_keys, group_values):
                    row[key] = value

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=base_columns)
    return out.loc[:, base_columns].sort_values(
        list(group_columns) + ["selection_basis", "integrated_abs_effect"],
        ascending=[True] * len(group_columns) + [True, False],
    ).reset_index(drop=True)


def _cfg(cfg: dict | None = None) -> dict:
    return config if cfg is None else cfg


def _candidate_cfg(cfg: dict) -> dict:
    return cfg.get("analysis", {}).get("candidate_regions", {})


def candidate_region_feature_names(feature_columns: list[str], region: pd.Series) -> list[str]:
    block = str(region["spectral_block"])
    lower = float(region["start_cm-1"])
    upper = float(region["end_cm-1"])
    out: list[str] = []
    for column in feature_columns:
        if not column.startswith(f"{block}__"):
            continue
        _, raw_wavenumber = column.split("__", 1)
        wavenumber = float(raw_wavenumber)
        if lower <= wavenumber <= upper:
            out.append(column)
    return out


def _normalize_candidates(source_analysis: str, frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    if normalized.empty:
        return pd.DataFrame(columns=[
            "source_analysis",
            "analysis_name",
            "target",
            "contrast_name",
            "group_variable",
            "group_a",
            "group_b",
            "spectral_block",
            "candidate_region_id",
            "selection_basis",
            "start_cm-1",
            "end_cm-1",
            "width_cm-1",
            "n_features",
            "min_fdr_q",
            "min_cluster_p_value",
            "n_fdr_significant",
            "n_cluster_significant",
            "max_abs_effect",
            "mean_abs_effect",
            "integrated_abs_effect",
            "effect_metric",
        ])

    rename_map = {
        "max_abs_rho": "max_abs_effect",
        "mean_abs_rho": "mean_abs_effect",
        "integrated_abs_rho": "integrated_abs_effect",
    }
    normalized = normalized.rename(columns=rename_map)
    normalized["source_analysis"] = source_analysis
    for column in [
        "analysis_name",
        "target",
        "contrast_name",
        "group_variable",
        "group_a",
        "group_b",
        "effect_metric",
    ]:
        if column not in normalized.columns:
            normalized[column] = pd.NA
    for column in [
        "n_fdr_significant",
        "n_cluster_significant",
        "max_abs_effect",
        "mean_abs_effect",
        "integrated_abs_effect",
    ]:
        if column not in normalized.columns:
            normalized[column] = np.nan

    if source_analysis == "clinical_correlation_matrix":
        normalized["analysis_name"] = "clinical_correlation_matrix"
        normalized["contrast_name"] = normalized["target"]
        normalized["effect_metric"] = "spearman_rho"
    elif source_analysis == "disease_stratification":
        normalized["analysis_name"] = normalized["contrast_name"]
        normalized["effect_metric"] = normalized["effect_metric"].fillna("rank_biserial_or_epsilon_squared")
    elif source_analysis == "longitudinal_moca":
        fallback_metric = pd.Series(
            np.where(
                normalized["analysis_name"].eq("delta_moca_association"),
                "spearman_rho",
                "rank_biserial",
            ),
            index=normalized.index,
        )
        normalized["effect_metric"] = normalized["effect_metric"].fillna(fallback_metric)

    ordered_columns = [
        "source_analysis",
        "analysis_name",
        "target",
        "contrast_name",
        "group_variable",
        "group_a",
        "group_b",
        "spectral_block",
        "candidate_region_id",
        "selection_basis",
        "start_cm-1",
        "end_cm-1",
        "width_cm-1",
        "n_features",
        "min_fdr_q",
        "min_cluster_p_value",
        "n_fdr_significant",
        "n_cluster_significant",
        "max_abs_effect",
        "mean_abs_effect",
        "integrated_abs_effect",
        "effect_metric",
    ]
    return normalized.loc[:, ordered_columns]


def run_candidate_region_detector(
    *,
    clinical_outputs: dict[str, Path] | None = None,
    longitudinal_outputs: dict[str, Path] | None = None,
    stratification_outputs: dict[str, Path] | None = None,
    output_dir: str | Path | None = None,
    cfg: dict | None = None,
) -> dict[str, Path]:
    cfg = _cfg(cfg)
    candidate_cfg = _candidate_cfg(cfg)
    if not bool(candidate_cfg.get("enabled", True)):
        return {}

    target_dir = (
        Path(output_dir)
        if output_dir is not None
        else get_cross_region_phase_dir("candidate_regions")
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    sources = [
        ("clinical_correlation_matrix", clinical_outputs, "clinical_correlation_candidate_regions"),
        ("longitudinal_moca", longitudinal_outputs, "delta_moca_candidate_regions"),
        ("disease_stratification", stratification_outputs, "stratification_candidate_regions"),
    ]
    for source_name, outputs, key in sources:
        if not outputs or key not in outputs:
            continue
        path = Path(outputs[key])
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        normalized = _normalize_candidates(source_name, frame)
        if not normalized.empty:
            frames.append(normalized)

    if frames:
        combined = pd.concat(frames, ignore_index=True, sort=False)
        combined = combined.sort_values(
            ["source_analysis", "analysis_name", "contrast_name", "target", "min_cluster_p_value", "integrated_abs_effect"],
            ascending=[True, True, True, True, True, False],
            na_position="last",
        ).reset_index(drop=True)
    else:
        combined = _normalize_candidates("clinical_correlation_matrix", pd.DataFrame())
    path = target_dir / "candidate_regions_cluster_corrected.csv"
    combined.to_csv(path, index=False)
    logger.info("Unified candidate-region table written to %s", path)
    return {"candidate_regions_cluster_corrected": path}
