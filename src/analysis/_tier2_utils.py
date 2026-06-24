"""Shared utilities for Tier-2 spectral interpretability phases (M–P).

Single implementation of the per-feature × endpoint Spearman association
loop used by spectral_shape, nmf_decomposition, and spectral_descriptors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from src.analysis.statistics import fdr_correction, partial_spearman


def compute_spearman_associations(
    feature_frame: pd.DataFrame,
    clinical: pd.DataFrame,
    feature_cols: list[str],
    endpoints: list[str],
    covariates: list[str],
    min_subjects: int,
    fdr_alpha: float,
    *,
    feature_col_name: str = "feature",
) -> pd.DataFrame:
    """Per (feature × endpoint): raw Spearman, partial Spearman | covariates, BH-FDR.

    Parameters
    ----------
    feature_frame:
        Wide subject-level frame with ``subject_id`` + ``feature_cols``.
    clinical:
        Subject-level frame with ``subject_id`` + endpoints + covariates.
    feature_col_name:
        Name for the feature-label column in the output (e.g. ``"parameter"``,
        ``"component"``, ``"metric"``).

    Returns
    -------
    DataFrame with columns ``{feature_col_name}``, ``endpoint``, ``n``,
    ``status``, ``raw_spearman_rho``, ``raw_spearman_p``,
    ``partial_spearman_rho``, ``partial_spearman_p``,
    ``raw_spearman_fdr_q``, ``raw_spearman_significant_fdr``.
    """
    feature_cols = [c for c in feature_cols if c in feature_frame.columns]
    if not feature_cols or not endpoints:
        return pd.DataFrame()

    avail_clinical = [c for c in ["subject_id"] + endpoints + covariates if c in clinical.columns]
    merged = clinical[avail_clinical].merge(
        feature_frame[["subject_id"] + feature_cols].assign(subject_id=lambda df: df["subject_id"].astype(str)),
        on="subject_id",
        how="inner",
    )

    rows: list[dict[str, object]] = []
    for feature in feature_cols:
        for endpoint in endpoints:
            if endpoint not in merged.columns:
                continue
            avail_covs = [c for c in covariates if c in merged.columns]
            complete = merged[[feature, endpoint, *avail_covs]].dropna()
            row: dict[str, object] = {
                feature_col_name: feature,
                "endpoint": endpoint,
                "n": int(len(complete)),
                "status": "ok",
            }
            if len(complete) < min_subjects:
                row["status"] = "insufficient_n"
                rows.append(row)
                continue
            raw = scipy_stats.spearmanr(complete[feature], complete[endpoint])
            rho, p_val, _ = partial_spearman(complete[feature], complete[endpoint], complete[avail_covs])
            row.update({
                "raw_spearman_rho": float(raw.statistic),
                "raw_spearman_p": float(raw.pvalue),
                "partial_spearman_rho": rho,
                "partial_spearman_p": p_val,
            })
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    ok_mask = result["status"] == "ok"
    if ok_mask.any():
        p_vals = result["raw_spearman_p"].where(ok_mask).to_numpy(dtype=float)
        q_vals, sig = fdr_correction(p_vals, alpha=fdr_alpha)
        result["raw_spearman_fdr_q"] = q_vals
        result["raw_spearman_significant_fdr"] = sig
    return result
