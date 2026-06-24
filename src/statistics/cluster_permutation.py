"""Shared subject-level cluster permutation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats


def _normalize_n_jobs(n_jobs: int) -> int:
    return 1 if n_jobs == 0 else n_jobs


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


def find_signed_clusters(
    effects: np.ndarray,
    p_values: np.ndarray,
    *,
    alpha: float,
) -> list[tuple[int, int, float]]:
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


def _binary_cluster_maxima(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    n_permutations: int,
    cluster_alpha: float,
    rng: np.random.Generator,
    n_jobs: int = 1,
) -> np.ndarray:
    n_jobs = _normalize_n_jobs(n_jobs)
    if n_jobs != 1:
        seeds = rng.integers(0, np.iinfo(np.int32).max, size=n_permutations, dtype=np.int64)
        return np.array(
            Parallel(n_jobs=n_jobs)(
                delayed(_binary_cluster_maximum)(values, labels, cluster_alpha, int(seed))
                for seed in seeds
            ),
            dtype=float,
        )

    maxima = np.zeros(n_permutations, dtype=float)
    n_left = int((labels == 0).sum())
    for perm_index in range(n_permutations):
        order = rng.permutation(len(labels))
        maxima[perm_index] = _binary_cluster_mass_for_order(values, order, int((labels == 0).sum()), cluster_alpha)
    return maxima


def _binary_cluster_maximum(values: np.ndarray, labels: np.ndarray, cluster_alpha: float, seed: int) -> float:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(labels))
    return _binary_cluster_mass_for_order(values, order, int((labels == 0).sum()), cluster_alpha)


def _binary_cluster_mass_for_order(
    values: np.ndarray,
    order: np.ndarray,
    n_left: int,
    cluster_alpha: float,
) -> float:
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
    clusters = find_signed_clusters(effects, p_values, alpha=cluster_alpha)
    return max((mass for _, _, mass in clusters), default=0.0)


def _spearman_cluster_maxima(
    values: np.ndarray,
    target_values: np.ndarray,
    *,
    n_permutations: int,
    cluster_alpha: float,
    rng: np.random.Generator,
    n_jobs: int = 1,
) -> np.ndarray:
    maxima = np.zeros(n_permutations, dtype=float)
    prepared_columns, centered_features, feature_denoms = _prepare_ranked_features(values)
    ranked_target = stats.rankdata(target_values, method="average")
    centered_target = ranked_target - float(np.mean(ranked_target))

    n_jobs = _normalize_n_jobs(n_jobs)
    if n_jobs != 1:
        seeds = rng.integers(0, np.iinfo(np.int32).max, size=n_permutations, dtype=np.int64)
        return np.array(
            Parallel(n_jobs=n_jobs)(
                delayed(_spearman_cluster_maximum)(
                    prepared_columns,
                    centered_features,
                    feature_denoms,
                    centered_target,
                    cluster_alpha,
                    int(seed),
                )
                for seed in seeds
            ),
            dtype=float,
        )

    for perm_index in range(n_permutations):
        permuted_target = centered_target[rng.permutation(len(centered_target))]
        maxima[perm_index] = _spearman_cluster_mass_for_target(
            prepared_columns,
            centered_features,
            feature_denoms,
            permuted_target,
            cluster_alpha,
        )
    return maxima


def _spearman_cluster_maximum(
    prepared_columns: np.ndarray,
    centered_features: np.ndarray,
    feature_denoms: np.ndarray,
    centered_target: np.ndarray,
    cluster_alpha: float,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    permuted_target = centered_target[rng.permutation(len(centered_target))]
    return _spearman_cluster_mass_for_target(
        prepared_columns,
        centered_features,
        feature_denoms,
        permuted_target,
        cluster_alpha,
    )


def _spearman_cluster_mass_for_target(
    prepared_columns: np.ndarray,
    centered_features: np.ndarray,
    feature_denoms: np.ndarray,
    permuted_target: np.ndarray,
    cluster_alpha: float,
) -> float:
    rho, p_values = _prepared_spearman(
        prepared_columns,
        centered_features,
        feature_denoms,
        permuted_target,
    )
    clusters = find_signed_clusters(rho, p_values, alpha=cluster_alpha)
    return max((mass for _, _, mass in clusters), default=0.0)


def apply_spearman_cluster_permutation(
    tests: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    target: str,
    n_permutations: int,
    cluster_alpha: float,
    random_state: int,
    n_jobs: int = 1,
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
        clusters = find_signed_clusters(rho, p_values, alpha=cluster_alpha)
        if not clusters:
            continue
        values = frame.loc[valid_rows, ordered["feature_name"]].to_numpy(dtype=float)
        maxima = _spearman_cluster_maxima(
            values,
            target_values.loc[valid_rows].to_numpy(dtype=float),
            n_permutations=n_permutations,
            cluster_alpha=cluster_alpha,
            rng=rng,
            n_jobs=n_jobs,
        )
        for cluster_index, (start, stop, mass) in enumerate(clusters, start=1):
            corrected = float((1 + np.sum(maxima >= mass)) / (n_permutations + 1))
            for row_index in ordered.loc[start : stop, "index"]:
                out.loc[row_index, "cluster_id"] = f"{block}_C{cluster_index}"
                out.loc[row_index, "cluster_mass"] = mass
                out.loc[row_index, "cluster_p_value"] = corrected
    return out


def apply_binary_cluster_permutation(
    tests: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    group_column: str,
    ordered_groups: tuple[str, str],
    n_permutations: int,
    cluster_alpha: float,
    random_state: int,
    n_jobs: int = 1,
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
        clusters = find_signed_clusters(effects, p_values, alpha=cluster_alpha)
        if not clusters:
            continue
        values = subset[ordered["feature_name"]].to_numpy(dtype=float)
        maxima = _binary_cluster_maxima(
            values,
            labels,
            n_permutations=n_permutations,
            cluster_alpha=cluster_alpha,
            rng=rng,
            n_jobs=n_jobs,
        )
        for cluster_index, (start, stop, mass) in enumerate(clusters, start=1):
            corrected = float((1 + np.sum(maxima >= mass)) / (n_permutations + 1))
            for row_index in ordered.loc[start : stop, "index"]:
                out.loc[row_index, "cluster_id"] = f"{block}_C{cluster_index}"
                out.loc[row_index, "cluster_mass"] = mass
                out.loc[row_index, "cluster_p_value"] = corrected
    return out
