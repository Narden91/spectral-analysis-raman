"""Shared helpers for subject-level clinical feature association analyses."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src import config, get_cross_region_summary_dir
from src.analysis.statistics import fdr_correction

logger = logging.getLogger(__name__)

FEATURE_PREFIXES = ("1300__", "2850__")


def _cfg(cfg: dict | None = None) -> dict:
    return config if cfg is None else cfg


def load_subject_feature_matrix(
    *,
    feature_matrix_path: str | Path | None = None,
) -> pd.DataFrame:
    path = (
        Path(feature_matrix_path)
        if feature_matrix_path is not None
        else get_cross_region_summary_dir() / "subject_feature_matrix_whole.parquet"
    )
    frame = pd.read_parquet(path)
    if "subject_id" not in frame.columns:
        raise ValueError("Subject feature matrix is missing subject_id")
    frame = frame.copy()
    frame["subject_id"] = frame["subject_id"].astype(str)
    if frame["subject_id"].duplicated().any():
        raise ValueError("Subject feature matrix requires one row per subject")
    return frame


def filter_disease_subjects(frame: pd.DataFrame) -> pd.DataFrame:
    disease_only = frame.copy()
    if "is_control" in disease_only.columns:
        mask = ~disease_only["is_control"].fillna(False).astype(bool)
        return disease_only.loc[mask].copy()
    if "condition" in disease_only.columns:
        mask = disease_only["condition"].astype(str).str.lower() != "control"
        return disease_only.loc[mask].copy()
    raise ValueError("Subject feature matrix must include is_control or condition for disease-only filtering")


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith(FEATURE_PREFIXES)]


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


def _rank_biserial_effect(
    left: np.ndarray,
    right: np.ndarray,
    u_statistic: float | np.ndarray,
) -> float | np.ndarray:
    denominator = float(len(left) * len(right))
    if denominator == 0:
        if np.ndim(u_statistic) == 0:
            return float("nan")
        return np.full(np.asarray(u_statistic).shape, np.nan, dtype=float)
    effect = (2.0 * np.asarray(u_statistic, dtype=float) / denominator) - 1.0
    if np.ndim(effect) == 0:
        return float(effect)
    return effect


def _correlation_p_values(rho: np.ndarray, n_subjects: int) -> np.ndarray:
    out = np.full_like(rho, np.nan, dtype=float)
    if n_subjects < 3:
        return out

    finite = np.isfinite(rho)
    if not finite.any():
        return out

    clipped = np.clip(rho[finite], -1.0 + 1e-12, 1.0 - 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = clipped * np.sqrt((n_subjects - 2) / np.clip(1.0 - clipped**2, 1e-12, None))
    p_values = 2.0 * stats.t.sf(np.abs(t_stat), df=n_subjects - 2)
    p_values[np.abs(rho[finite]) >= 1.0] = 0.0
    out[finite] = p_values
    return out


def _prepare_ranked_features(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid_columns = np.all(np.isfinite(values), axis=0)
    centered = np.empty((values.shape[0], int(valid_columns.sum())), dtype=float)
    denoms = np.empty(int(valid_columns.sum()), dtype=float)
    for output_index, column_index in enumerate(np.flatnonzero(valid_columns)):
        ranked = stats.rankdata(values[:, column_index], method="average")
        centered_column = ranked - float(np.mean(ranked))
        centered[:, output_index] = centered_column
        denoms[output_index] = float(np.linalg.norm(centered_column))
    return valid_columns, centered, denoms


def _prepared_spearman(
    prepared_columns: np.ndarray,
    centered_features: np.ndarray,
    feature_denoms: np.ndarray,
    centered_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rho = np.full(prepared_columns.shape[0], np.nan, dtype=float)
    if centered_features.size == 0:
        return rho, np.full_like(rho, np.nan)

    target_denom = float(np.linalg.norm(centered_target))
    if target_denom == 0:
        return rho, np.full_like(rho, np.nan)

    usable = feature_denoms > 0
    if not usable.any():
        return rho, np.full_like(rho, np.nan)

    numerators = centered_features[:, usable].T @ centered_target
    rho_values = numerators / (feature_denoms[usable] * target_denom)
    full_indices = np.flatnonzero(prepared_columns)[usable]
    rho[full_indices] = rho_values
    p_values = _correlation_p_values(rho, len(centered_target))
    return rho, p_values


def compute_spearman_feature_tests(
    frame: pd.DataFrame,
    feature_meta: pd.DataFrame,
    *,
    target: str,
    min_subjects: int = 5,
    fdr_alpha: float = 0.05,
) -> pd.DataFrame:
    if target not in frame.columns:
        logger.warning("Target '%s' not found in disease-only feature matrix", target)
        return pd.DataFrame()

    target_values = pd.to_numeric(frame[target], errors="coerce")
    valid_rows = target_values.notna().to_numpy(dtype=bool)
    n_valid = int(valid_rows.sum())
    if n_valid < min_subjects:
        logger.warning("Target '%s' has only %d valid subjects; skipping", target, n_valid)
        return pd.DataFrame()

    feature_names = feature_meta["feature_name"].tolist()
    values = frame.loc[valid_rows, feature_names].to_numpy(dtype=float)
    prepared_columns, centered_features, feature_denoms = _prepare_ranked_features(values)
    ranked_target = stats.rankdata(target_values.loc[valid_rows].to_numpy(dtype=float), method="average")
    centered_target = ranked_target - float(np.mean(ranked_target))
    rho, p_values = _prepared_spearman(
        prepared_columns,
        centered_features,
        feature_denoms,
        centered_target,
    )
    q_values, significant = fdr_correction(p_values, alpha=fdr_alpha)

    tests = feature_meta.copy()
    tests.insert(0, "target", target)
    tests["rho"] = rho
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
    effect_sizes = _rank_biserial_effect(left, right, u_statistics)

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


def _cluster_masses(effects: np.ndarray, p_values: np.ndarray, alpha: float) -> list[tuple[int, int, float]]:
    active = (p_values < alpha) & np.isfinite(effects)
    clusters: list[tuple[int, int, float]] = []
    start: int | None = None
    sign = 0.0
    for index, is_active in enumerate(active):
        current_sign = np.sign(effects[index]) if is_active else 0.0
        if is_active and start is None:
            start = index
            sign = current_sign
            continue
        if is_active and start is not None and current_sign == sign:
            continue
        if start is not None:
            stop = index - 1
            mass = float(np.nansum(np.abs(effects[start : stop + 1])))
            clusters.append((start, stop, mass))
            start = index if is_active else None
            sign = current_sign if is_active else 0.0
    if start is not None:
        stop = len(active) - 1
        mass = float(np.nansum(np.abs(effects[start : stop + 1])))
        clusters.append((start, stop, mass))
    return clusters


def _spearman_cluster_maxima(
    values: np.ndarray,
    target_values: np.ndarray,
    *,
    n_permutations: int,
    cluster_alpha: float,
    rng: np.random.Generator,
) -> np.ndarray:
    maxima = np.zeros(n_permutations, dtype=float)
    prepared_columns, centered_features, feature_denoms = _prepare_ranked_features(values)
    ranked_target = stats.rankdata(target_values, method="average")
    centered_target = ranked_target - float(np.mean(ranked_target))

    for perm_index in range(n_permutations):
        permuted_target = centered_target[rng.permutation(len(centered_target))]
        rho, p_values = _prepared_spearman(
            prepared_columns,
            centered_features,
            feature_denoms,
            permuted_target,
        )
        clusters = _cluster_masses(rho, p_values, cluster_alpha)
        maxima[perm_index] = max((mass for _, _, mass in clusters), default=0.0)
    return maxima


def apply_spearman_cluster_permutation(
    tests: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    target: str,
    n_permutations: int,
    cluster_alpha: float,
    random_state: int,
) -> pd.DataFrame:
    if tests.empty or tests["status"].ne("ok").all():
        return tests

    target_values = pd.to_numeric(frame[target], errors="coerce")
    valid_rows = target_values.notna().to_numpy(dtype=bool)
    if int(valid_rows.sum()) < 5:
        return tests

    out = tests.copy()
    out["cluster_id"] = pd.NA
    out["cluster_mass"] = np.nan
    out["cluster_p_value"] = np.nan
    rng = np.random.default_rng(random_state)

    for block, block_df in out.groupby("spectral_block", sort=False):
        ordered = block_df.sort_values("wavenumber").reset_index()
        rho = ordered["rho"].to_numpy(dtype=float)
        p_values = ordered["p_value"].to_numpy(dtype=float)
        clusters = _cluster_masses(rho, p_values, cluster_alpha)
        if not clusters:
            continue
        values = frame.loc[valid_rows, ordered["feature_name"]].to_numpy(dtype=float)
        maxima = _spearman_cluster_maxima(
            values,
            target_values.loc[valid_rows].to_numpy(dtype=float),
            n_permutations=n_permutations,
            cluster_alpha=cluster_alpha,
            rng=rng,
        )
        for cluster_index, (start, stop, mass) in enumerate(clusters, start=1):
            corrected = float((1 + np.sum(maxima >= mass)) / (n_permutations + 1))
            for row_index in ordered.loc[start : stop, "index"]:
                out.loc[row_index, "cluster_id"] = f"{block}_C{cluster_index}"
                out.loc[row_index, "cluster_mass"] = mass
                out.loc[row_index, "cluster_p_value"] = corrected
    return out


def _binary_cluster_maxima(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    n_permutations: int,
    cluster_alpha: float,
    rng: np.random.Generator,
) -> np.ndarray:
    maxima = np.zeros(n_permutations, dtype=float)
    n_left = int((labels == 0).sum())
    for perm_index in range(n_permutations):
        order = rng.permutation(len(labels))
        left = values[order[:n_left]]
        right = values[order[n_left:]]
        test_result = stats.mannwhitneyu(
            left,
            right,
            axis=0,
            alternative="two-sided",
        )
        p_values = np.asarray(test_result.pvalue, dtype=float)
        effects = np.asarray(
            _rank_biserial_effect(left, right, np.asarray(test_result.statistic, dtype=float)),
            dtype=float,
        )
        clusters = _cluster_masses(effects, p_values, cluster_alpha)
        maxima[perm_index] = max((mass for _, _, mass in clusters), default=0.0)
    return maxima


def apply_binary_cluster_permutation(
    tests: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    group_column: str,
    ordered_groups: tuple[str, str],
    n_permutations: int,
    cluster_alpha: float,
    random_state: int,
) -> pd.DataFrame:
    if tests.empty or tests["status"].ne("ok").all():
        return tests

    out = tests.copy()
    out["cluster_id"] = pd.NA
    out["cluster_mass"] = np.nan
    out["cluster_p_value"] = np.nan
    rng = np.random.default_rng(random_state)

    subset = frame.loc[frame[group_column].isin(ordered_groups)].copy()
    subset[group_column] = subset[group_column].astype(str)
    labels = np.where(subset[group_column] == ordered_groups[0], 0, 1)

    for block, block_df in out.groupby("spectral_block", sort=False):
        ordered = block_df.sort_values("wavenumber").reset_index()
        effects = ordered["effect_size"].to_numpy(dtype=float)
        p_values = ordered["p_value"].to_numpy(dtype=float)
        clusters = _cluster_masses(effects, p_values, cluster_alpha)
        if not clusters:
            continue
        values = subset[ordered["feature_name"]].to_numpy(dtype=float)
        maxima = _binary_cluster_maxima(
            values,
            labels,
            n_permutations=n_permutations,
            cluster_alpha=cluster_alpha,
            rng=rng,
        )
        for cluster_index, (start, stop, mass) in enumerate(clusters, start=1):
            corrected = float((1 + np.sum(maxima >= mass)) / (n_permutations + 1))
            for row_index in ordered.loc[start : stop, "index"]:
                out.loc[row_index, "cluster_id"] = f"{block}_C{cluster_index}"
                out.loc[row_index, "cluster_mass"] = mass
                out.loc[row_index, "cluster_p_value"] = corrected
    return out


def build_candidate_regions(
    tests: pd.DataFrame,
    *,
    group_columns: tuple[str, ...],
    effect_column: str,
    min_width_cm: float = 8.0,
    merge_gap_cm: float = 6.0,
) -> pd.DataFrame:
    base_columns = list(group_columns) + [
        "spectral_block",
        "candidate_region_id",
        "start_cm-1",
        "end_cm-1",
        "width_cm-1",
        "n_features",
        "min_fdr_q",
        "min_cluster_p_value",
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

    candidate_mask = working["significant_fdr"].fillna(False)
    if "cluster_p_value" in working.columns:
        candidate_mask = candidate_mask | working["cluster_p_value"].lt(0.05).fillna(False)
    candidates = working.loc[candidate_mask].copy()
    if candidates.empty:
        return pd.DataFrame(columns=base_columns)

    rows: list[dict[str, object]] = []
    group_keys = list(group_columns) + ["spectral_block"]
    for group_values, block_df in candidates.groupby(group_keys, sort=False):
        ordered = block_df.sort_values("wavenumber").reset_index(drop=True)
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
                row = {
                    "spectral_block": str(window["spectral_block"].iloc[0]),
                    "candidate_region_id": f"{window['spectral_block'].iloc[0]}_R{region_index}",
                    "start_cm-1": float(window["wavenumber"].iloc[0]),
                    "end_cm-1": float(window["wavenumber"].iloc[-1]),
                    "width_cm-1": width,
                    "n_features": int(len(window)),
                    "min_fdr_q": float(window["fdr_q"].min()) if "fdr_q" in window.columns else np.nan,
                    "min_cluster_p_value": (
                        float(window["cluster_p_value"].min())
                        if "cluster_p_value" in window.columns
                        else np.nan
                    ),
                    "max_abs_effect": float(window[effect_column].abs().max()),
                    "mean_abs_effect": float(window[effect_column].abs().mean()),
                    "integrated_abs_effect": float(
                        np.trapz(
                            window[effect_column].abs().to_numpy(dtype=float),
                            window["wavenumber"].to_numpy(dtype=float),
                        )
                    ),
                }
                if not isinstance(group_values, tuple):
                    group_values = (group_values,)
                for key, value in zip(group_keys, group_values):
                    row[key] = value
                rows.append(row)
            start = index

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=base_columns)
    return out.loc[:, base_columns].sort_values(
        list(group_columns) + ["integrated_abs_effect"],
        ascending=[True] * len(group_columns) + [False],
    ).reset_index(drop=True)
