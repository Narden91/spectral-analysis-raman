"""Shared subject-level cluster permutation helpers."""

from __future__ import annotations

import multiprocessing
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import special, stats

from src.analysis.paired_lipid._backend import _single_thread_blas

_PARALLEL_MIN_PERMS = 512
_PERM_CHUNK_SIZE = 128


def normalize_n_jobs(n_jobs: int) -> int:
    return 1 if n_jobs == 0 else n_jobs


def _resolve_parallel_workers(n_tasks: int, n_jobs: int | None) -> int:
    if n_tasks <= 1 or n_jobs == 0 or n_jobs == 1:
        return 1
    if os.environ.get("_PAIRED_LIPID_IN_BOOTSTRAP_WORKER") == "1":
        return 1
    if multiprocessing.parent_process() is not None:
        return 1
    try:
        import joblib.parallel

        active_backend, _ = joblib.parallel.get_active_backend()
        if getattr(active_backend, "nesting_level", 0) > 0:
            return 1
    except Exception:
        pass
    cpu_cnt = os.cpu_count() or 1
    if n_jobs is None or int(n_jobs) == -1:
        raw = cpu_cnt
    elif int(n_jobs) < -1:
        raw = max(1, cpu_cnt + 1 + int(n_jobs))
    else:
        raw = int(n_jobs)
    if raw <= 1:
        return 1
    return min(raw, n_tasks)


def rank_biserial_effect(
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


def correlation_p_values(rho: np.ndarray, n_subjects: int) -> np.ndarray:
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


def prepare_ranked_features(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid_columns = np.all(np.isfinite(values), axis=0)
    n_valid = int(valid_columns.sum())
    if n_valid == 0:
        return (
            valid_columns,
            np.empty((values.shape[0], 0), dtype=float),
            np.empty(0, dtype=float),
        )
    ranked = stats.rankdata(values[:, valid_columns], method="average", axis=0)
    centered = ranked - np.mean(ranked, axis=0, keepdims=True)
    denoms = np.linalg.norm(centered, axis=0)
    return valid_columns, centered, denoms


def prepared_spearman(
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
    p_values = correlation_p_values(rho, len(centered_target))
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


def _batch_max_cluster_mass(
    effects: np.ndarray,
    p_values: np.ndarray,
    alpha: float,
) -> np.ndarray:
    active = (p_values < alpha) & np.isfinite(effects)
    out = np.zeros(effects.shape[0], dtype=float)
    active_rows = np.flatnonzero(active.any(axis=1))
    for row_idx in active_rows:
        idx = np.flatnonzero(active[row_idx])
        e_sub = effects[row_idx, idx]
        if idx.size == 1:
            out[row_idx] = float(abs(e_sub[0]))
            continue
        splits = np.flatnonzero((np.diff(idx) != 1) | (np.diff(np.sign(e_sub)) != 0)) + 1
        if splits.size == 0:
            out[row_idx] = float(np.sum(np.abs(e_sub)))
        else:
            starts = np.empty(splits.size + 1, dtype=np.intp)
            starts[0] = 0
            starts[1:] = splits
            out[row_idx] = float(np.max(np.add.reduceat(np.abs(e_sub), starts)))
    return out


def _binary_cluster_chunk(
    seed_chunk: np.ndarray,
    ranks_valid: np.ndarray,
    s_valid: np.ndarray,
    n_obs: int,
    n_feat: int,
    n_left: int,
    n_right: int,
    valid_cols: np.ndarray,
    has_nan_cols: bool,
    cluster_alpha: float,
) -> np.ndarray:
    with _single_thread_blas():
        p_chunk = len(seed_chunk)
        m_mat = np.zeros((p_chunk, n_obs), dtype=np.float64)
        left_indices = np.empty((p_chunk, n_left), dtype=np.intp)
        for row_i, seed in enumerate(seed_chunk):
            left_indices[row_i] = np.random.default_rng(int(seed)).permutation(n_obs)[:n_left]
        m_mat[np.arange(p_chunk)[:, np.newaxis], left_indices] = 1.0
        r1 = m_mat @ ranks_valid
        u1 = r1 - (n_left * (n_left + 1) * 0.5)
        n_prod = float(n_left * n_right)
        u2 = n_prod - u1
        u_stat = np.maximum(u1, u2)
        mu = n_prod * 0.5
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (u_stat - mu - 0.5) / s_valid[np.newaxis, :]
        p_valid = np.clip(2.0 * special.ndtr(-z), 0.0, 1.0)
        eff_valid = (2.0 * u1 / n_prod) - 1.0
        if not has_nan_cols:
            return _batch_max_cluster_mass(eff_valid, p_valid, cluster_alpha)
        effects_full = np.full((p_chunk, n_feat), np.nan, dtype=float)
        pvals_full = np.full((p_chunk, n_feat), np.nan, dtype=float)
        effects_full[:, valid_cols] = eff_valid
        pvals_full[:, valid_cols] = p_valid
        return _batch_max_cluster_mass(effects_full, pvals_full, cluster_alpha)


def _binary_cluster_maxima(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    n_permutations: int,
    cluster_alpha: float,
    rng: np.random.Generator,
    n_jobs: int = 1,
) -> np.ndarray:
    n_jobs = normalize_n_jobs(n_jobs)
    seeds = rng.integers(0, np.iinfo(np.int32).max, size=n_permutations, dtype=np.int64)
    if n_permutations == 0:
        return np.empty(0, dtype=float)

    n_obs, n_feat = values.shape
    n_left = int((labels == 0).sum())
    n_right = n_obs - n_left
    if n_left == 0 or n_right == 0:
        return np.array(
            [_binary_cluster_maximum(values, labels, cluster_alpha, int(seed)) for seed in seeds],
            dtype=float,
        )
    if n_feat == 0:
        return np.zeros(n_permutations, dtype=float)

    nan_cols = np.any(np.isnan(values), axis=0)
    valid_cols = ~nan_cols
    has_nan_cols = bool(nan_cols.any())
    if not valid_cols.any():
        return np.zeros(n_permutations, dtype=float)

    vals_valid = values[:, valid_cols]
    sorted_vals = np.sort(vals_valid, axis=0)
    has_tie_col = np.any(sorted_vals[1:] == sorted_vals[:-1], axis=0)

    # Preserve SciPy mannwhitneyu method="auto" exact mode for small-N groups when
    # any NaN column triggers column-wise dispatch or any column lacks ties.
    use_exact_fallback = (n_left <= 8 or n_right <= 8) and (
        has_nan_cols or not bool(np.all(has_tie_col))
    )
    if use_exact_fallback:
        workers = _resolve_parallel_workers(n_permutations, n_jobs) if n_permutations >= _PARALLEL_MIN_PERMS else 1
        if workers > 1:
            return np.array(
                Parallel(n_jobs=workers)(
                    delayed(_binary_cluster_maximum)(values, labels, cluster_alpha, int(seed))
                    for seed in seeds
                ),
                dtype=float,
            )
        return np.array(
            [_binary_cluster_maximum(values, labels, cluster_alpha, int(seed)) for seed in seeds],
            dtype=float,
        )

    ranks_valid = stats.rankdata(vals_valid, method="average", axis=0)
    tie_terms_valid = np.zeros(vals_valid.shape[1], dtype=float)
    for col_j in np.flatnonzero(has_tie_col):
        _, counts = np.unique(sorted_vals[:, col_j], return_counts=True)
        tie_terms_valid[col_j] = float(np.sum(counts**3 - counts))
    denom_n = float(n_obs * (n_obs - 1))
    var_factor = np.maximum((n_obs + 1.0) - tie_terms_valid / denom_n, 0.0)
    s_valid = np.sqrt((n_left * n_right / 12.0) * var_factor)

    seed_chunks = [seeds[i : i + _PERM_CHUNK_SIZE] for i in range(0, n_permutations, _PERM_CHUNK_SIZE)]
    workers = _resolve_parallel_workers(len(seed_chunks), n_jobs) if n_permutations >= _PARALLEL_MIN_PERMS else 1
    if workers > 1:
        parts = Parallel(n_jobs=workers)(
            delayed(_binary_cluster_chunk)(
                chunk,
                ranks_valid,
                s_valid,
                n_obs,
                n_feat,
                n_left,
                n_right,
                valid_cols,
                has_nan_cols,
                cluster_alpha,
            )
            for chunk in seed_chunks
        )
        return np.concatenate(parts)

    parts = [
        _binary_cluster_chunk(
            chunk,
            ranks_valid,
            s_valid,
            n_obs,
            n_feat,
            n_left,
            n_right,
            valid_cols,
            has_nan_cols,
            cluster_alpha,
        )
        for chunk in seed_chunks
    ]
    return np.concatenate(parts)


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
        rank_biserial_effect(left, right, np.asarray(test_result.statistic, dtype=float)),
        dtype=float,
    )
    clusters = find_signed_clusters(effects, p_values, alpha=cluster_alpha)
    return max((mass for _, _, mass in clusters), default=0.0)


def _spearman_cluster_chunk(
    seed_chunk: np.ndarray,
    centered_usable: np.ndarray,
    centered_target: np.ndarray,
    denom_prod: np.ndarray,
    n_obs: int,
    n_feat: int,
    full_indices: np.ndarray,
    all_usable: bool,
    cluster_alpha: float,
    rho_crit: float | None,
) -> np.ndarray:
    with _single_thread_blas():
        p_chunk = len(seed_chunk)
        t_mat = np.empty((p_chunk, n_obs), dtype=np.float64)
        for row_i, seed in enumerate(seed_chunk):
            perm = np.random.default_rng(int(seed)).permutation(n_obs)
            t_mat[row_i] = centered_target[perm]
        numerators = t_mat @ centered_usable
        rho_u = numerators / denom_prod[np.newaxis, :]
        df = n_obs - 2
        if rho_crit is not None:
            p_u = np.ones_like(rho_u)
            cand = np.abs(rho_u) >= rho_crit
            if cand.any():
                r_c = rho_u[cand]
                clipped = np.clip(r_c, -1.0 + 1e-12, 1.0 - 1e-12)
                with np.errstate(divide="ignore", invalid="ignore"):
                    t_stat = clipped * np.sqrt(df / np.clip(1.0 - clipped**2, 1e-12, None))
                p_c = 2.0 * special.stdtr(df, -np.abs(t_stat))
                p_c[np.abs(r_c) >= 1.0] = 0.0
                p_u[cand] = p_c
        else:
            clipped = np.clip(rho_u, -1.0 + 1e-12, 1.0 - 1e-12)
            with np.errstate(divide="ignore", invalid="ignore"):
                t_stat = clipped * np.sqrt(df / np.clip(1.0 - clipped**2, 1e-12, None))
            p_u = 2.0 * special.stdtr(df, -np.abs(t_stat))
            p_u[np.abs(rho_u) >= 1.0] = 0.0

        if all_usable:
            return _batch_max_cluster_mass(rho_u, p_u, cluster_alpha)
        rho_full = np.full((p_chunk, n_feat), np.nan, dtype=float)
        pvals_full = np.full((p_chunk, n_feat), np.nan, dtype=float)
        rho_full[:, full_indices] = rho_u
        pvals_full[:, full_indices] = p_u
        return _batch_max_cluster_mass(rho_full, pvals_full, cluster_alpha)


def _spearman_cluster_maxima(
    values: np.ndarray,
    target_values: np.ndarray,
    *,
    n_permutations: int,
    cluster_alpha: float,
    rng: np.random.Generator,
    n_jobs: int = 1,
) -> np.ndarray:
    prepared_columns, centered_features, feature_denoms = prepare_ranked_features(values)
    ranked_target = stats.rankdata(target_values, method="average")
    centered_target = ranked_target - float(np.mean(ranked_target))

    n_jobs = normalize_n_jobs(n_jobs)
    seeds = rng.integers(0, np.iinfo(np.int32).max, size=n_permutations, dtype=np.int64)
    if n_permutations == 0:
        return np.empty(0, dtype=float)

    n_obs = len(centered_target)
    n_feat = len(prepared_columns)
    target_denom = float(np.linalg.norm(centered_target))
    usable = feature_denoms > 0
    if (
        centered_features.size == 0
        or target_denom == 0
        or not usable.any()
        or n_obs < 3
        or cluster_alpha <= 0.0
    ):
        return np.zeros(n_permutations, dtype=float)

    centered_usable = np.ascontiguousarray(centered_features[:, usable])
    denom_prod = target_denom * feature_denoms[usable]
    all_usable = bool(np.all(prepared_columns)) and bool(np.all(usable))
    full_indices = np.flatnonzero(prepared_columns)[usable]
    rho_crit: float | None = None
    if 0.0 < cluster_alpha < 1.0:
        t_crit = float(stats.t.isf(cluster_alpha * 0.5, df=n_obs - 2))
        if np.isfinite(t_crit) and t_crit > 0.0:
            rho_crit = (1.0 / float(np.hypot(np.sqrt(n_obs - 2) / t_crit, 1.0))) * (1.0 - 1e-6)
        else:
            rho_crit = None

    seed_chunks = [seeds[i : i + _PERM_CHUNK_SIZE] for i in range(0, n_permutations, _PERM_CHUNK_SIZE)]
    workers = _resolve_parallel_workers(len(seed_chunks), n_jobs) if n_permutations >= _PARALLEL_MIN_PERMS else 1
    if workers > 1:
        parts = Parallel(n_jobs=workers)(
            delayed(_spearman_cluster_chunk)(
                chunk,
                centered_usable,
                centered_target,
                denom_prod,
                n_obs,
                n_feat,
                full_indices,
                all_usable,
                cluster_alpha,
                rho_crit,
            )
            for chunk in seed_chunks
        )
        return np.concatenate(parts)

    parts = [
        _spearman_cluster_chunk(
            chunk,
            centered_usable,
            centered_target,
            denom_prod,
            n_obs,
            n_feat,
            full_indices,
            all_usable,
            cluster_alpha,
            rho_crit,
        )
        for chunk in seed_chunks
    ]
    return np.concatenate(parts)


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
    rho, p_values = prepared_spearman(
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

    ordered_str = (str(ordered_groups[0]), str(ordered_groups[1]))
    subset = frame.loc[frame[group_column].notna()].copy()
    subset[group_column] = subset[group_column].astype(str)
    subset = subset.loc[subset[group_column].isin(ordered_str)]
    labels = np.where(subset[group_column] == ordered_str[0], 0, 1)
    n_left = int((labels == 0).sum())
    n_right = len(labels) - n_left
    if n_left == 0 or n_right == 0:
        raise ValueError(
            f"Binary cluster permutation requires non-empty groups {ordered_str!r} "
            f"in '{group_column}' (got n_left={n_left}, n_right={n_right})"
        )

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
