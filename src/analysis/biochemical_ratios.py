"""Phase 10: targeted biochemical band and ratio associations."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src import config, get_cross_region_phase_dir
from src.analysis.band_features import band_integral, infer_block_from_upper, ratio_from_integrals
from src.analysis.clinical_feature_associations import (
    feature_columns,
    load_disease_feature_frame,
)
from src.analysis.statistics import fdr_correction

logger = logging.getLogger(__name__)

_DEFAULT_TARGETS = ("moca_baseline", "moca_3m", "delta_moca")


def _cfg(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    return config if cfg is None else cfg


def _ratio_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("analysis", {}).get("biochemical_ratios", {})


def _band_windows(cfg: dict[str, Any]) -> dict[str, tuple[float, float]]:
    bands_cfg = cfg.get("whole_spectra", {}).get("bands", {})
    windows: dict[str, tuple[float, float]] = {}
    for band_name, limits in bands_cfg.items():
        if isinstance(limits, (list, tuple)) and len(limits) == 2:
            windows[str(band_name)] = (float(limits[0]), float(limits[1]))
    return windows


def _build_band_table(
    frame: pd.DataFrame,
    columns: list[str],
    cfg: dict[str, Any],
) -> pd.DataFrame:
    band_features: dict[str, pd.Series] = {}
    for band_name, (lower, upper) in _band_windows(cfg).items():
        block = infer_block_from_upper(upper)
        series = band_integral(frame, columns, block=block, lower=lower, upper=upper)
        if series.notna().any():
            band_features[band_name] = series
    return pd.DataFrame(band_features, index=frame.index)


def _build_ratio_table(
    bands: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    ratios: dict[str, pd.Series] = {}
    for pair in cfg.get("whole_spectra", {}).get("ratio_pairs", []):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        numerator_name, denominator_name = str(pair[0]), str(pair[1])
        if numerator_name not in bands or denominator_name not in bands:
            continue
        ratios[f"{numerator_name}__over__{denominator_name}"] = ratio_from_integrals(
            bands[numerator_name],
            bands[denominator_name],
        )
    return pd.DataFrame(ratios, index=bands.index)


def _spearman_associations(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    *,
    targets: tuple[str, ...],
    feature_source: str,
    min_subjects: int,
    fdr_alpha: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target in targets:
        if target not in frame:
            logger.info("Biochemical ratio target '%s' not found; skipping", target)
            continue
        y = pd.to_numeric(frame[target], errors="coerce")
        for feature_name in features.columns:
            x = pd.to_numeric(features[feature_name], errors="coerce")
            valid = y.notna() & x.notna() & np.isfinite(x)
            n_valid = int(valid.sum())
            if n_valid < min_subjects:
                rows.append(
                    {
                        "target": target,
                        "feature_source": feature_source,
                        "feature_name": feature_name,
                        "rho": np.nan,
                        "p_value": np.nan,
                        "n": n_valid,
                        "status": "insufficient sample size",
                    }
                )
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", stats.ConstantInputWarning)
                rho, p_value = stats.spearmanr(
                    x.loc[valid].to_numpy(dtype=float),
                    y.loc[valid].to_numpy(dtype=float),
                )
            rows.append(
                {
                    "target": target,
                    "feature_source": feature_source,
                    "feature_name": feature_name,
                    "rho": float(rho) if np.isfinite(rho) else np.nan,
                    "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
                    "n": n_valid,
                    "status": "ok",
                }
            )

    associations = pd.DataFrame(
        rows,
        columns=[
            "target",
            "feature_source",
            "feature_name",
            "rho",
            "p_value",
            "fdr_q",
            "significant_fdr",
            "n",
            "status",
        ],
    )
    if associations.empty:
        return associations

    q_values, significant = fdr_correction(
        associations["p_value"].to_numpy(dtype=float),
        alpha=fdr_alpha,
    )
    associations["fdr_q"] = q_values
    associations["significant_fdr"] = significant
    return associations


def run_biochemical_ratios(
    *,
    feature_matrix_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Run targeted biochemical band-integral and ratio associations."""
    cfg = _cfg(cfg)
    ratio_cfg = _ratio_cfg(cfg)
    if not bool(ratio_cfg.get("enabled", True)):
        return {}

    disease_only, _feature_meta = load_disease_feature_frame(feature_matrix_path=feature_matrix_path)
    all_feature_columns = feature_columns(disease_only)
    targets = tuple(str(target) for target in ratio_cfg.get("targets", _DEFAULT_TARGETS))
    min_subjects = int(ratio_cfg.get("min_subjects", 5))
    fdr_alpha = float(ratio_cfg.get("fdr_alpha", 0.05))

    target_dir = (
        Path(output_dir)
        if output_dir is not None
        else get_cross_region_phase_dir("biochemical_ratios")
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    bands = _build_band_table(disease_only, all_feature_columns, cfg)
    ratios = _build_ratio_table(bands, cfg)
    band_associations = _spearman_associations(
        disease_only,
        bands,
        targets=targets,
        feature_source="band_integral",
        min_subjects=min_subjects,
        fdr_alpha=fdr_alpha,
    )
    ratio_associations = _spearman_associations(
        disease_only,
        ratios,
        targets=targets,
        feature_source="band_ratio",
        min_subjects=min_subjects,
        fdr_alpha=fdr_alpha,
    )
    associations = pd.concat(
        [band_associations, ratio_associations],
        ignore_index=True,
        sort=False,
    )

    bands_path = target_dir / "biochemical_band_integrals.csv"
    ratios_path = target_dir / "biochemical_ratios.csv"
    associations_path = target_dir / "biochemical_ratio_associations.csv"
    bands.to_csv(bands_path, index=False)
    ratios.to_csv(ratios_path, index=False)
    associations.to_csv(associations_path, index=False)

    logger.info("Biochemical ratio outputs written to %s", target_dir)
    return {
        "biochemical_band_integrals": bands_path,
        "biochemical_ratios": ratios_path,
        "biochemical_ratio_associations": associations_path,
    }
