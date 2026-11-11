"""Shared helpers for subject-level clinical feature association analyses."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src import get_cross_region_phase_dir
from src.analysis.statistics import fdr_correction
from src.statistics.cluster_permutation import (
    prepare_ranked_features,
    prepared_spearman,
    rank_biserial_effect,
)

logger = logging.getLogger(__name__)

FEATURE_PREFIXES = ("1300__", "2850__")
# Baseline-corrected, un-normalized companion columns (positive-valued basis for
# physical band-integral intensities). See src/analysis/subject_features.py.
BC_FEATURE_PREFIXES = ("1300_bc__", "2850_bc__")


def _feature_matrix_path(feature_matrix_path: str | Path | None = None) -> Path:
    return (
        Path(feature_matrix_path)
        if feature_matrix_path is not None
        else get_cross_region_phase_dir("subject_features") / "subject_feature_matrix_whole.parquet"
    )


def _bc_feature_matrix_path(feature_matrix_path: str | Path | None = None) -> Path:
    return (
        Path(feature_matrix_path)
        if feature_matrix_path is not None
        else get_cross_region_phase_dir("subject_features") / "subject_feature_matrix_whole_bc.parquet"
    )


@lru_cache(maxsize=4)
def _read_subject_feature_matrix_cached(path: str, mtime_ns: int, size: int) -> pd.DataFrame:
    logger.debug("Reading subject feature matrix from %s", path)
    return pd.read_parquet(path)


def load_subject_feature_matrix(
    *,
    feature_matrix_path: str | Path | None = None,
) -> pd.DataFrame:
    path = _feature_matrix_path(feature_matrix_path)
    stat = path.stat()
    frame = _read_subject_feature_matrix_cached(str(path.resolve()), stat.st_mtime_ns, stat.st_size).copy()
    if "subject_id" not in frame.columns:
        raise ValueError("Subject feature matrix is missing subject_id")
    frame = frame.copy()
    frame["subject_id"] = frame["subject_id"].astype(str)
    if frame["subject_id"].duplicated().any():
        raise ValueError("Subject feature matrix requires one row per subject")
    return frame


def filter_disease_subjects(frame: pd.DataFrame) -> pd.DataFrame:
    """Select subjects with an explicit disease (is_control=False) label.

    Subjects with missing/unknown clinical status are excluded rather than
    coerced to disease, since defaulting an unlabeled subject into the disease
    arm would silently bias disease-only analyses.
    """
    disease_only = frame.copy()
    if "is_control" in disease_only.columns:
        is_control = disease_only["is_control"]
        unknown = is_control.isna()
        keep = ~unknown & ~is_control.astype("boolean").fillna(True)
    elif "condition" in disease_only.columns:
        condition = disease_only["condition"].astype("string").str.lower()
        unknown = condition.isna() | condition.eq("unknown") | condition.eq("")
        keep = ~unknown & ~condition.eq("control")
    else:
        raise ValueError("Subject feature matrix must include is_control or condition for disease-only filtering")

    dropped_unknown = disease_only.loc[unknown]
    if not dropped_unknown.empty and "subject_id" in disease_only.columns:
        logger.warning(
            "filter_disease_subjects: excluding %d subject(s) with missing/unknown "
            "clinical status: %s",
            len(dropped_unknown),
            dropped_unknown["subject_id"].astype(str).tolist(),
        )
    return disease_only.loc[keep].copy()


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith(FEATURE_PREFIXES)]


def load_subject_bc_feature_matrix(
    *,
    feature_matrix_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load the baseline-corrected, un-normalized companion feature matrix.

    Positive-valued basis for physical band-integral intensities (see
    src/analysis/biochemical_ratios.py). Raises if the file is missing rather than
    silently falling back to the SNV/area matrix (repo rule: no silent fallbacks).
    """
    return load_subject_feature_matrix(feature_matrix_path=_bc_feature_matrix_path(feature_matrix_path))


def bc_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith(BC_FEATURE_PREFIXES)]


def feature_metadata(columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in columns:
        block, wavenumber = column.split("__", 1)
        rows.append(
            {
                "feature_name": column,
                "spectral_block": block,
                "wavenumber": float(wavenumber),
            }
        )
    return pd.DataFrame(rows).sort_values(["spectral_block", "wavenumber"]).reset_index(drop=True)


def load_disease_feature_frame(
    *,
    feature_matrix_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = load_subject_feature_matrix(feature_matrix_path=feature_matrix_path)
    disease_only = filter_disease_subjects(frame)
    features = feature_columns(disease_only)
    if not features:
        raise ValueError("Subject feature matrix requires whole-spectrum feature columns")
    return disease_only, feature_metadata(features)


def endpoint_missingness_report(
    frame: pd.DataFrame,
    endpoints: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    n_subjects = int(len(frame))
    for endpoint in endpoints:
        if endpoint not in frame.columns:
            rows.append(
                {
                    "endpoint": endpoint,
                    "n_subjects_with_spectra": n_subjects,
                    "n_non_missing": 0,
                    "n_missing": n_subjects,
                    "missing_rate": 1.0 if n_subjects else np.nan,
                    "status": "missing column",
                }
            )
            continue
        values = pd.to_numeric(frame[endpoint], errors="coerce")
        n_non_missing = int(values.notna().sum())
        n_missing = int(n_subjects - n_non_missing)
        rows.append(
            {
                "endpoint": endpoint,
                "n_subjects_with_spectra": n_subjects,
                "n_non_missing": n_non_missing,
                "n_missing": n_missing,
                "missing_rate": float(n_missing / n_subjects) if n_subjects else np.nan,
                "status": "ok",
            }
        )
    return pd.DataFrame(rows)


def empty_spearman_association_table() -> pd.DataFrame:
    """Header-only frame matching the columns of a cluster-tested Spearman screen."""
    return pd.DataFrame(
        columns=[
            "target",
            "feature_name",
            "spectral_block",
            "wavenumber",
            "rho",
            "p_value",
            "fdr_q",
            "significant_fdr",
            "n",
            "status",
            "cluster_id",
            "cluster_mass",
            "cluster_p_value",
        ]
    )


def compute_spearman_feature_tests(
    frame: pd.DataFrame,
    feature_meta: pd.DataFrame,
    *,
    target: str,
    min_subjects: int = 5,
    fdr_alpha: float = 0.05,
) -> pd.DataFrame:
    """Whole-spectrum channel-wise screen against a clinical endpoint.

    Uses marginal (unadjusted) Spearman correlation, not covariate-residualized
    `partial_spearman`. The cluster-permutation reference distribution in
    `apply_spearman_cluster_permutation` is also marginal, so the screen is
    internally consistent, but it remains exposed to age/sex confounding.
    Covariate adjustment is applied only in the confirmatory fixed panel
    (see `src.analysis.robust_contrasts`). Treat this screen as exploratory.
    """
    if target not in frame.columns:
        logger.warning("Target '%s' not found in disease-only feature matrix", target)
        return pd.DataFrame()

    target_values = pd.to_numeric(frame[target], errors="coerce")
    valid_rows = target_values.notna().to_numpy(dtype=bool)
    n_valid = int(valid_rows.sum())
    if n_valid < min_subjects:
        logger.info("Target '%s' has only %d valid subjects; skipping", target, n_valid)
        return pd.DataFrame()

    feature_names = feature_meta["feature_name"].tolist()
    values = frame.loc[valid_rows, feature_names].to_numpy(dtype=float)
    prepared_columns, centered_features, feature_denoms = prepare_ranked_features(values)
    ranked_target = stats.rankdata(target_values.loc[valid_rows].to_numpy(dtype=float), method="average")
    centered_target = ranked_target - float(np.mean(ranked_target))
    rho, p_values = prepared_spearman(
        prepared_columns,
        centered_features,
        feature_denoms,
        centered_target,
    )
    q_values, significant = fdr_correction(p_values, alpha=fdr_alpha)

    # Fisher Z-transform 95% CI for each Spearman rho.
    # z = arctanh(rho), SE(z) = 1/sqrt(n-3), CI on z then back-transformed.
    rho_ci_low = np.full_like(rho, np.nan)
    rho_ci_high = np.full_like(rho, np.nan)
    if n_valid >= 5:
        finite_rho = np.isfinite(rho)
        rho_clipped = np.clip(rho[finite_rho], -1.0 + 1e-10, 1.0 - 1e-10)
        z = np.arctanh(rho_clipped)
        se_z = 1.0 / np.sqrt(max(n_valid - 3, 1))
        z_crit = 1.96
        rho_ci_low[finite_rho] = np.tanh(z - z_crit * se_z)
        rho_ci_high[finite_rho] = np.tanh(z + z_crit * se_z)

    tests = feature_meta.copy()
    tests.insert(0, "target", target)
    tests["rho"] = rho
    tests["rho_ci_low"] = rho_ci_low
    tests["rho_ci_high"] = rho_ci_high
    tests["p_value"] = p_values
    tests["fdr_q"] = q_values
    tests["significant_fdr"] = significant
    tests["n"] = n_valid
    tests["status"] = "ok"
    return tests


def compute_binary_group_feature_tests(
    frame: pd.DataFrame,
    feature_meta: pd.DataFrame,
    *,
    group_column: str,
    ordered_groups: tuple[str, str],
    contrast_name: str,
    min_subjects_per_group: int = 5,
    fdr_alpha: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if group_column not in frame.columns:
        raise ValueError(f"Feature matrix is missing group column '{group_column}'")

    subset = frame.loc[frame[group_column].notna()].copy()
    subset[group_column] = subset[group_column].astype(str)
    counts = (
        subset[group_column]
        .value_counts()
        .reindex(ordered_groups, fill_value=0)
        .rename_axis("group_name")
        .reset_index(name="n_subjects")
    )
    counts["group_variable"] = group_column
    counts["contrast_name"] = contrast_name
    insufficient = counts["n_subjects"].lt(min_subjects_per_group).any()
    counts["status"] = np.where(insufficient, "insufficient sample size", "ok")
    counts["min_subjects_per_group"] = min_subjects_per_group

    if insufficient:
        return pd.DataFrame(
            [
                {
                    "contrast_name": contrast_name,
                    "group_variable": group_column,
                    "group_a": ordered_groups[0],
                    "group_b": ordered_groups[1],
                    "test": "mannwhitney",
                    "status": "insufficient sample size",
                }
            ]
        ), counts

    left_group, right_group = ordered_groups
    left = subset.loc[subset[group_column] == left_group, feature_meta["feature_name"]].to_numpy(dtype=float)
    right = subset.loc[subset[group_column] == right_group, feature_meta["feature_name"]].to_numpy(dtype=float)
    test_result = stats.mannwhitneyu(
        left,
        right,
        axis=0,
        alternative="two-sided",
    )
    u_statistics = np.asarray(test_result.statistic, dtype=float)
    p_values = np.asarray(test_result.pvalue, dtype=float)
    effect_sizes = rank_biserial_effect(left, right, u_statistics)

    rows: list[dict[str, object]] = []
    for feature_index, feature_row in feature_meta.iterrows():
        rows.append(
            {
                "contrast_name": contrast_name,
                "group_variable": group_column,
                "group_a": left_group,
                "group_b": right_group,
                "test": "mannwhitney",
                "feature_name": feature_row["feature_name"],
                "spectral_block": feature_row["spectral_block"],
                "wavenumber": feature_row["wavenumber"],
                "n_group_a": int(left.shape[0]),
                "n_group_b": int(right.shape[0]),
                "u_statistic": float(u_statistics[feature_index]),
                "p_value": float(p_values[feature_index]),
                "effect_size": float(effect_sizes[feature_index]),
                "effect_size_name": "rank_biserial",
                "status": "ok",
            }
        )

    tests = pd.DataFrame(rows)
    q_values, significant = fdr_correction(tests["p_value"].to_numpy(dtype=float), alpha=fdr_alpha)
    tests["fdr_q"] = q_values
    tests["significant_fdr"] = significant
    return tests, counts
