"""Batched closed-form evaluation of multiplicity bootstrap draws on CPU (NumPy) or GPU (CuPy)."""

from __future__ import annotations

import contextlib
import multiprocessing
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from joblib import Parallel, delayed

from src.analysis.paired_lipid._backend import (
    _get_gpu_xp as _get_gpu_xp,
)
from src.analysis.paired_lipid._backend import (
    _probe_cupy as _probe_cupy,
)
from src.analysis.paired_lipid._backend import (
    _single_thread_blas as _single_thread_blas,
)
from src.analysis.paired_lipid._backend import (
    _to_np as _to_np,
)
from src.analysis.paired_lipid._backend import (
    resolve_array_backend as _resolve_array_backend_impl,
)
from src.analysis.paired_lipid._backend import (
    suggest_batch_size as suggest_batch_size,
)
from src.analysis.paired_lipid._batched_models import (
    _prep_mfa_pca_batch as _prep_mfa_pca_batch,
)
from src.analysis.paired_lipid._batched_models import (
    _prep_paired_moments_batch as _prep_paired_moments_batch,
)
from src.analysis.paired_lipid._batched_models import (
    batched_division_prep as batched_division_prep,
)
from src.analysis.paired_lipid._batched_models import (
    batched_paired_schur_sse as batched_paired_schur_sse,
)
from src.analysis.paired_lipid._batched_models import (
    batched_pca_schur_sse as batched_pca_schur_sse,
)
from src.analysis.paired_lipid._batched_models import (
    batched_pls_sse_and_preds as batched_pls_sse_and_preds,
)
from src.analysis.paired_lipid._batched_models import (
    batched_ridge_preds_single_alpha as batched_ridge_preds_single_alpha,
)
from src.analysis.paired_lipid._batched_models import (
    batched_ridge_sse_and_preds as batched_ridge_sse_and_preds,
)
from src.analysis.paired_lipid._eigensolvers import (
    _chefsi_top_k_gpu as _chefsi_top_k_gpu,
)
from src.analysis.paired_lipid._eigensolvers import (
    _exact_top_k_eigh_active as _exact_top_k_eigh_active,
)
from src.analysis.paired_lipid._eigensolvers import (
    _fast_top1_eigvalsh as _fast_top1_eigvalsh,
)
from src.analysis.paired_lipid._eigensolvers import (
    _rayleigh_ritz_32 as _rayleigh_ritz_32,
)
from src.analysis.paired_lipid._eigensolvers import (
    _top_k_eigh as _top_k_eigh,
)
from src.analysis.paired_lipid._eigensolvers import (
    _whitened_generalized_eigh as _whitened_generalized_eigh,
)
from src.analysis.paired_lipid._eigensolvers import (
    batched_solve_directions_grid as batched_solve_directions_grid,
)
from src.analysis.paired_lipid._gram_cache import (
    _get_dataset_gram_cache as _get_dataset_gram_cache,
)
from src.analysis.paired_lipid._gram_cache import (
    _get_single_sample_gram_cache as _get_single_sample_gram_cache,
)
from src.analysis.paired_lipid._gram_cache import (
    _get_spectral_gram_cache as _get_spectral_gram_cache,
)
from src.analysis.paired_lipid._gram_cache import (
    _precompute_division_constants as _precompute_division_constants,
)
from src.analysis.paired_lipid.cohort import CohortDataset, PairedLipidConfig


@dataclass(slots=True)
class SpectralDivisionTrajectory:
    """Precomputed covariate-invariant spectral MFA-PCA and paired-moment trajectory for one division."""

    part_pca_tr: np.ndarray
    part_pca_val: np.ndarray
    W_max: np.ndarray
    T_max: np.ndarray
    scale1_sq: np.ndarray
    scale2_sq: np.ndarray
    rank_ok: np.ndarray
    V_grid: dict[int, np.ndarray] | None = None
    numerical_rank: np.ndarray | None = None

    def slice(self, start: int, end: int) -> SpectralDivisionTrajectory:
        """Return a draw-axis slice ``[start:end]`` sharing underlying NumPy buffers."""
        return SpectralDivisionTrajectory(
            part_pca_tr=self.part_pca_tr[start:end],
            part_pca_val=self.part_pca_val[start:end],
            W_max=self.W_max[start:end],
            T_max=self.T_max[start:end],
            scale1_sq=self.scale1_sq[start:end],
            scale2_sq=self.scale2_sq[start:end],
            rank_ok=self.rank_ok[start:end],
            V_grid=(
                {k: v[start:end] for k, v in self.V_grid.items()}
                if self.V_grid is not None
                else None
            ),
            numerical_rank=(
                self.numerical_rank[start:end]
                if self.numerical_rank is not None
                else None
            ),
        )


def _concat_spectral_division_trajectories(
    chunks: list[SpectralDivisionTrajectory],
) -> SpectralDivisionTrajectory:
    """Concatenate chunked ``SpectralDivisionTrajectory`` objects along draw axis 0."""
    first = chunks[0]
    v_grid: dict[int, np.ndarray] | None = None
    if first.V_grid is not None:
        v_grid = {
            k: np.concatenate([c.V_grid[k] for c in chunks if c.V_grid is not None], axis=0)
            for k in first.V_grid
        }
    num_rank = (
        np.concatenate([c.numerical_rank for c in chunks if c.numerical_rank is not None], axis=0)
        if first.numerical_rank is not None
        else None
    )
    return SpectralDivisionTrajectory(
        part_pca_tr=np.concatenate([c.part_pca_tr for c in chunks], axis=0),
        part_pca_val=np.concatenate([c.part_pca_val for c in chunks], axis=0),
        W_max=np.concatenate([c.W_max for c in chunks], axis=0),
        T_max=np.concatenate([c.T_max for c in chunks], axis=0),
        scale1_sq=np.concatenate([c.scale1_sq for c in chunks], axis=0),
        scale2_sq=np.concatenate([c.scale2_sq for c in chunks], axis=0),
        rank_ok=np.concatenate([c.rank_ok for c in chunks], axis=0),
        V_grid=v_grid,
        numerical_rank=num_rank,
    )


def _resolve_cpu_bootstrap_workers(
    n_sub: int,
    n_jobs: int | None,
    *,
    min_parallel_draws: int = 256,
) -> int:
    """Resolve safe CPU worker count for draw-chunk parallelism with nested-pool protection."""
    if n_jobs == 1 or n_sub < min_parallel_draws:
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
    return max(1, min(raw, n_sub // 16))


def _compute_spec_traj_batch(
    xp: Any,
    dconst: dict[str, Any],
    M_tr: Any,
    cfg: PairedLipidConfig,
    *,
    compute_v_grid: bool,
    compute_full_rank: bool,
) -> SpectralDivisionTrajectory:
    """Compute host-resident ``SpectralDivisionTrajectory`` for one draw batch of one division."""
    tr_p = dconst["tr_p"]
    val_p = dconst["val_p"]
    w_tr_sum = xp.sum(M_tr, axis=1, keepdims=True)
    mfa = _prep_mfa_pca_batch(xp, {}, dconst, M_tr, w_tr_sum, cfg, compute_full_rank=compute_full_rank)
    W_max, T_max = _prep_paired_moments_batch(
        xp,
        dconst,
        M_tr,
        mfa["w_s"],
        mfa["alpha"],
        mfa["alpha_sum"],
        mfa["scale1_sq"],
        mfa["scale2_sq"],
        mfa["part_pca_max"],
    )
    part_pca_tr = _to_np(xp, mfa["part_pca_max"][:, tr_p, :])
    part_pca_val = _to_np(xp, mfa["part_pca_max"][:, val_p, :])
    W_np = _to_np(xp, W_max)
    T_np = _to_np(xp, T_max)
    s1_np = _to_np(xp, mfa["scale1_sq"])
    s2_np = _to_np(xp, mfa["scale2_sq"])
    rank_np = _to_np(xp, mfa["rank_ok"]) > 0.5
    num_rank_np = (
        _to_np(xp, mfa["numerical_rank"]).astype(int)
        if compute_full_rank and "numerical_rank" in mfa
        else None
    )
    v_grid_np: dict[int, np.ndarray] | None = None
    if compute_v_grid:
        k_list = [int(k) for k in cfg.pca_dimensions]
        v_grid_xp = batched_solve_directions_grid(
            xp,
            W_max,
            T_max,
            k_list,
            list(cfg.lambda_grid),
            eigenvalue_tolerance=cfg.eigenvalue_tolerance,
        )
        v_grid_np = {k: _to_np(xp, v) for k, v in v_grid_xp.items()}
        del v_grid_xp
    del mfa, W_max, T_max
    return SpectralDivisionTrajectory(
        part_pca_tr=part_pca_tr,
        part_pca_val=part_pca_val,
        W_max=W_np,
        T_max=T_np,
        scale1_sq=s1_np,
        scale2_sq=s2_np,
        rank_ok=rank_np,
        V_grid=v_grid_np,
        numerical_rank=num_rank_np,
    )


def _build_spec_trajs_chunk_core(
    xp: Any,
    fold_spec_dconsts: list[tuple[list[dict[str, Any]], dict[str, Any]]],
    M_chunk: np.ndarray,
    cfg: PairedLipidConfig,
    batch_size: int,
) -> list[tuple[list[SpectralDivisionTrajectory], SpectralDivisionTrajectory]]:
    """Build host ``SpectralDivisionTrajectory`` pairs across all folds for a contiguous draw chunk."""
    n_chunk = int(M_chunk.shape[0])
    per_fold_in_batches: list[list[list[SpectralDivisionTrajectory]]] = [
        [[] for _ in in_dcs] for in_dcs, _ in fold_spec_dconsts
    ]
    per_fold_out_batches: list[list[SpectralDivisionTrajectory]] = [[] for _ in fold_spec_dconsts]

    for b_start in range(0, n_chunk, batch_size):
        b_end = min(b_start + batch_size, n_chunk)
        M_batch = xp.asarray(M_chunk[b_start:b_end], dtype=xp.float64)
        for f_idx, (in_dcs, out_dc) in enumerate(fold_spec_dconsts):
            for i_idx, in_dc in enumerate(in_dcs):
                per_fold_in_batches[f_idx][i_idx].append(
                    _compute_spec_traj_batch(
                        xp,
                        in_dc,
                        M_batch[:, in_dc["tr_p"]],
                        cfg,
                        compute_v_grid=True,
                        compute_full_rank=False,
                    )
                )
            per_fold_out_batches[f_idx].append(
                _compute_spec_traj_batch(
                    xp,
                    out_dc,
                    M_batch[:, out_dc["tr_p"]],
                    cfg,
                    compute_v_grid=False,
                    compute_full_rank=False,
                )
            )
        del M_batch

    out: list[tuple[list[SpectralDivisionTrajectory], SpectralDivisionTrajectory]] = []
    for f_idx in range(len(fold_spec_dconsts)):
        in_merged = [
            _concat_spectral_division_trajectories(batches) for batches in per_fold_in_batches[f_idx]
        ]
        out_merged = _concat_spectral_division_trajectories(per_fold_out_batches[f_idx])
        out.append((in_merged, out_merged))
    return out


def _build_cpu_spec_traj_chunk_worker(
    fold_spec_dconsts: list[tuple[list[dict[str, Any]], dict[str, Any]]],
    M_chunk: np.ndarray,
    cfg: PairedLipidConfig,
    batch_size: int,
) -> list[tuple[list[SpectralDivisionTrajectory], SpectralDivisionTrajectory]]:
    """Top-level picklable worker building CPU spectral trajectories for one draw chunk."""
    prev_env = os.environ.get("_PAIRED_LIPID_IN_BOOTSTRAP_WORKER")
    os.environ["_PAIRED_LIPID_IN_BOOTSTRAP_WORKER"] = "1"
    try:
        with _single_thread_blas():
            return _build_spec_trajs_chunk_core(np, fold_spec_dconsts, M_chunk, cfg, batch_size)
    finally:
        if prev_env is None:
            os.environ.pop("_PAIRED_LIPID_IN_BOOTSTRAP_WORKER", None)
        else:
            os.environ["_PAIRED_LIPID_IN_BOOTSTRAP_WORKER"] = prev_env


def _empty_spectral_division_trajectory(
    n_tr: int,
    n_val: int,
    cfg: PairedLipidConfig,
    *,
    compute_v_grid: bool,
) -> SpectralDivisionTrajectory:
    """Construct an empty host ``SpectralDivisionTrajectory`` with exact zero-batch shapes and dtypes."""
    max_k = max(int(k) for k in cfg.pca_dimensions)
    n_lam = len(cfg.lambda_grid)
    v_grid = (
        {
            int(k): np.empty((0, n_lam, int(k), int(k)), dtype=np.float64)
            for k in cfg.pca_dimensions
        }
        if compute_v_grid
        else None
    )
    return SpectralDivisionTrajectory(
        part_pca_tr=np.empty((0, n_tr, max_k), dtype=np.float64),
        part_pca_val=np.empty((0, n_val, max_k), dtype=np.float64),
        W_max=np.empty((0, max_k, max_k), dtype=np.float64),
        T_max=np.empty((0, max_k, max_k), dtype=np.float64),
        scale1_sq=np.empty((0,), dtype=np.float64),
        scale2_sq=np.empty((0,), dtype=np.float64),
        rank_ok=np.empty((0,), dtype=bool),
        V_grid=v_grid,
        numerical_rank=None,
    )


def batched_build_spectral_division_trajectories(
    dataset: CohortDataset,
    outer_splits: list[tuple[np.ndarray, np.ndarray]],
    inner_splits_per_fold: list[list[tuple[np.ndarray, np.ndarray]]],
    M_run: np.ndarray,
    cfg: PairedLipidConfig,
    *,
    use_gpu: bool | None = None,
    batch_size: int | None = None,
    n_jobs: int | None = None,
    single_sample_downstream: bool = False,
) -> list[tuple[list[SpectralDivisionTrajectory], SpectralDivisionTrajectory]]:
    """Precompute covariate-invariant spectral division trajectories for ``M_run``."""
    n_sub = int(M_run.shape[0])
    if n_sub == 0:
        return [
            (
                [
                    _empty_spectral_division_trajectory(
                        int(tr_idx[in_tr_rel].size),
                        int(tr_idx[in_val_rel].size),
                        cfg,
                        compute_v_grid=True,
                    )
                    for in_tr_rel, in_val_rel in in_splits
                ],
                _empty_spectral_division_trajectory(
                    int(tr_idx.size),
                    int(te_idx.size),
                    cfg,
                    compute_v_grid=False,
                ),
            )
            for (tr_idx, te_idx), in_splits in zip(outer_splits, inner_splits_per_fold)
        ]

    xp, _ = resolve_array_backend(use_gpu, n_sub, gpu_probe=_get_gpu_xp)
    spec_gcache = _get_spectral_gram_cache(dataset)
    fold_spec_dconsts: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []
    with _single_thread_blas():
        for (tr_idx, te_idx), in_splits in zip(outer_splits, inner_splits_per_fold):
            in_dcs = [
                _precompute_division_constants(
                    xp,
                    spec_gcache,
                    dataset,
                    tr_idx[in_tr_rel],
                    tr_idx[in_val_rel],
                    cfg,
                    single_sample_downstream=single_sample_downstream,
                    spectral_only=True,
                )
                for in_tr_rel, in_val_rel in in_splits
            ]
            out_dc = _precompute_division_constants(
                xp,
                spec_gcache,
                dataset,
                tr_idx,
                te_idx,
                cfg,
                single_sample_downstream=single_sample_downstream,
                spectral_only=True,
            )
            fold_spec_dconsts.append((in_dcs, out_dc))

    if batch_size is None:
        n_samples_outer = max((int(out_dc["K1_div"].shape[0]) for _, out_dc in fold_spec_dconsts), default=M_run.shape[1])
        batch_size = suggest_batch_size(xp, n_sub, n_samples_outer)

    if xp is np:
        n_workers = _resolve_cpu_bootstrap_workers(n_sub, n_jobs, min_parallel_draws=256)
        if n_workers > 1:
            n_chunks = min(n_sub, n_workers * 2)
            idx_chunks = np.array_split(np.arange(n_sub), n_chunks)
            chunk_results = Parallel(n_jobs=n_workers)(
                delayed(_build_cpu_spec_traj_chunk_worker)(
                    fold_spec_dconsts,
                    np.ascontiguousarray(M_run[int(c[0]) : int(c[-1]) + 1], dtype=np.float64),
                    cfg,
                    min(batch_size, int(c.size)),
                )
                for c in idx_chunks
            )
            merged: list[tuple[list[SpectralDivisionTrajectory], SpectralDivisionTrajectory]] = []
            for f_idx, (in_dcs, _) in enumerate(fold_spec_dconsts):
                in_merged = [
                    _concat_spectral_division_trajectories([res[f_idx][0][i_idx] for res in chunk_results])
                    for i_idx in range(len(in_dcs))
                ]
                out_merged = _concat_spectral_division_trajectories(
                    [res[f_idx][1] for res in chunk_results]
                )
                merged.append((in_merged, out_merged))
            return merged
        with _single_thread_blas():
            return _build_spec_trajs_chunk_core(np, fold_spec_dconsts, M_run, cfg, batch_size)

    return _build_spec_trajs_chunk_core(xp, fold_spec_dconsts, M_run, cfg, batch_size)


def resolve_array_backend(
    use_gpu: bool | None,
    n_draws: int,
    *,
    min_auto_draws: int = 256,
    gpu_probe: Callable[[], Any | None] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Select CuPy or NumPy backend while honoring monkeypatches to ``batched._get_gpu_xp``."""
    probe = _get_gpu_xp if gpu_probe is None else gpu_probe
    return _resolve_array_backend_impl(use_gpu, n_draws, min_auto_draws=min_auto_draws, gpu_probe=probe)


# Tie rules (research_design.md): fewer retained directions, fewer PCA dimensions, then stronger
# regularization. MSEs arrive rounded to 12 decimals, so exact ties are well defined.
def _argmin_last(values: np.ndarray) -> np.ndarray:
    """Argmin over the last axis; an exact tie goes to the last index (the larger ridge alpha)."""
    return values.shape[-1] - 1 - np.argmin(values[..., ::-1], axis=-1)


def _select_pca_setting(pca_r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Best (k index, alpha index) per draw from MSEs of shape (B, n_k, n_alphas)."""
    n_alphas = pca_r.shape[-1]
    flat = np.argmin(pca_r[:, :, ::-1].reshape(pca_r.shape[0], -1), axis=1)
    return flat // n_alphas, (n_alphas - 1) - (flat % n_alphas)


def _select_paired_setting(pair_r: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Best (r, k, lambda, alpha) indices per draw from MSEs of shape (B, n_r, n_k, n_lam, n_alphas)."""
    n_b, _, n_k, n_lam, n_alphas = pair_r.shape
    rem = np.argmin(pair_r[:, :, :, ::-1, ::-1].reshape(n_b, -1), axis=1)
    a_idx = (n_alphas - 1) - rem % n_alphas
    rem = rem // n_alphas
    lam_idx = (n_lam - 1) - rem % n_lam
    rem = rem // n_lam
    return rem // n_k, rem % n_k, lam_idx, a_idx


def _check_fast_batch_eligibility(
    dataset: CohortDataset,
    outer_splits: list[tuple[np.ndarray, np.ndarray]],
    inner_splits_per_fold: list[list[tuple[np.ndarray, np.ndarray]]],
    M_all: np.ndarray,
    cfg: PairedLipidConfig,
    gcache: dict[str, Any],
) -> np.ndarray:
    """Return boolean mask of bootstrap draws eligible for vectorized full-grid Schur evaluation."""
    del gcache
    n_b = M_all.shape[0]
    if (
        list(cfg.retained_directions) != [1, 2, 3]
        or min(cfg.pca_dimensions) < max(cfg.retained_directions)
    ):
        return np.zeros(n_b, dtype=bool)

    max_pls_c = max(cfg.pls_component_grid)
    eligible = np.ones(n_b, dtype=bool)
    p_pair = dataset.paired_participant_indices
    # Every configured k must stay below the design cap (paired participants - 1); otherwise
    # the sequential path caps k, which the fixed-shape batched grid cannot represent.
    min_inner_pairs = max(cfg.pca_dimensions) + 1

    for (tr_idx, te_idx), in_splits in zip(outer_splits, inner_splits_per_fold):
        if tr_idx.size - 2 < max_pls_c:
            return np.zeros(n_b, dtype=bool)
        eligible &= np.sum(M_all[:, te_idx] > 0, axis=1) >= 1
        eligible &= np.sum(M_all[:, tr_idx] > 0, axis=1) >= max(3, max_pls_c + 2)
        pair_otr = p_pair[np.isin(p_pair, tr_idx)]
        n_pair_otr = np.sum(M_all[:, pair_otr] > 0, axis=1)
        eligible &= n_pair_otr >= (min_inner_pairs + 1)
        eligible &= np.sum(M_all[:, pair_otr], axis=1) > 1.0

        min_pair_itr = np.full(n_b, 999999, dtype=int)
        for in_tr_rel, in_val_rel in in_splits:
            in_tr = tr_idx[in_tr_rel]
            in_val = tr_idx[in_val_rel]
            if in_tr.size - 2 < max_pls_c:
                return np.zeros(n_b, dtype=bool)
            eligible &= np.sum(M_all[:, in_tr] > 0, axis=1) >= max(3, max_pls_c + 2)
            eligible &= np.sum(M_all[:, in_val] > 0, axis=1) >= 1
            pair_itr = p_pair[np.isin(p_pair, in_tr)]
            n_pair_itr = np.sum(M_all[:, pair_itr] > 0, axis=1)
            eligible &= n_pair_itr >= min_inner_pairs
            eligible &= np.sum(M_all[:, pair_itr], axis=1) > 1.0
            min_pair_itr = np.minimum(min_pair_itr, n_pair_itr)
        eligible &= n_pair_otr > min_pair_itr

    return eligible


def batched_select_inner_settings(
    xp: Any,
    gcache: dict[str, Any],
    in_dconsts: list[dict[str, Any]],
    M_batch: Any,
    y_all_xp: Any,
    alphas_xp: Any,
    part_band_xp: Any,
    cfg: PairedLipidConfig,
    *,
    compute_inner_diagnostics: bool = False,
    inner_spec_trajs: list[SpectralDivisionTrajectory] | None = None,
) -> dict[str, Any]:
    """Stream inner folds one-at-a-time to select hyperparameters for B draws simultaneously.

    Streaming each inner fold without retaining all ``inner_divs`` simultaneously cuts inner-CV
    peak device/host memory by ``n_inner_splits`` (~3x) while keeping identical operations.
    """
    B = int(M_batch.shape[0])
    n_alphas = int(alphas_xp.shape[0])
    k_list = [int(k) for k in cfg.pca_dimensions]
    lam_list = list(cfg.lambda_grid)
    r_list = list(cfg.retained_directions)
    max_pls_c = max(cfg.pls_component_grid)
    pls_idx = np.asarray(sorted({int(c) - 1 for c in cfg.pls_component_grid}), dtype=int)
    n_k, n_lam, n_r = len(k_list), len(lam_list), len(r_list)

    total_val_w = xp.zeros(B, dtype=xp.float64)
    cov_sse = xp.zeros((B, n_alphas), dtype=xp.float64)
    band_sse = xp.zeros((B, n_alphas), dtype=xp.float64)
    pca_sse = xp.zeros((B, n_k, n_alphas), dtype=xp.float64)
    pls_sse = xp.zeros((B, max_pls_c), dtype=xp.float64)
    pair_sse = xp.zeros((B, n_r, n_k * n_lam, n_alphas), dtype=xp.float64)
    rank_ok_xp = xp.ones(B, dtype=bool)
    inner_paired_counts: list[np.ndarray] = []
    inner_ranks: list[np.ndarray] = []
    inner_dropped_cov_sets: list[set[str]] = [set() for _ in range(B)] if compute_inner_diagnostics else []

    for div_idx, dconst in enumerate(in_dconsts):
        spec_slice = inner_spec_trajs[div_idx] if inner_spec_trajs is not None else None
        in_tr = dconst["tr_p"]
        in_val = dconst["val_p"]
        div = batched_division_prep(
            xp,
            gcache,
            dconst,
            M_batch[:, in_tr],
            M_batch[:, in_val],
            y_all_xp,
            alphas_xp,
            cfg,
            compute_pca_suff=True,
            compute_full_rank=compute_inner_diagnostics,
            spec_traj_slice=spec_slice,
        )
        rank_ok_xp &= div["rank_ok"]
        if compute_inner_diagnostics:
            p_pair_tr = dconst["p_pair_tr"]
            n_pair_div = _to_np(xp, xp.sum(M_batch[:, p_pair_tr] > 0.0, axis=1)).astype(int)
            inner_paired_counts.append(n_pair_div)
            inner_ranks.append(_to_np(xp, div["numerical_rank"]).astype(int))
            base_dropped = tuple(dconst.get("dropped_cov_cols_base", ()))
            kept_cols = tuple(dconst.get("kept_cov_cols", ()))
            valid_mask_np = _to_np(xp, div["valid_cov_mask"]) > 0.5
            for b_i in range(B):
                inner_dropped_cov_sets[b_i].update(base_dropped)
                for col_name, is_ok in zip(kept_cols, valid_mask_np[b_i]):
                    if not is_ok:
                        inner_dropped_cov_sets[b_i].add(col_name)
        total_val_w += xp.sum(div["w_val"], axis=1)
        cov_sse += div["sse_0"]

        sse_b, _ = batched_ridge_sse_and_preds(
            xp,
            div["X_cov_tr"],
            div["X_cov_val"],
            part_band_xp[in_tr],
            part_band_xp[in_val],
            div["y_tr"],
            div["y_val"],
            div["w_tr"],
            div["w_val"],
            alphas_xp,
        )
        band_sse += sse_b

        pca_sse += batched_pca_schur_sse(xp, div, k_list, alphas_xp)

        sse_pls, _ = batched_pls_sse_and_preds(xp, div, max_pls_c, return_preds=False)
        pls_sse += sse_pls

        if spec_slice is not None and spec_slice.V_grid is not None:
            V_grid = {k: xp.asarray(v, dtype=xp.float64) for k, v in spec_slice.V_grid.items()}
        else:
            V_grid = batched_solve_directions_grid(
                xp,
                div["W_max"],
                div["T_max"],
                k_list,
                lam_list,
                eigenvalue_tolerance=cfg.eigenvalue_tolerance,
            )
        pair_sse += batched_paired_schur_sse(xp, div, V_grid, k_list, alphas_xp)
        del div, V_grid

    cov_mses = _to_np(xp, cov_sse / total_val_w[:, None])
    band_mses = _to_np(xp, band_sse / total_val_w[:, None])
    pca_mses = _to_np(xp, pca_sse / total_val_w[:, None, None])
    pls_mses = _to_np(xp, pls_sse / total_val_w[:, None])
    pair_mses = _to_np(xp, pair_sse / total_val_w[:, None, None, None])
    rank_ok = _to_np(xp, rank_ok_xp) > 0.5

    b_arange = np.arange(B)
    best_cov_a = _argmin_last(np.round(cov_mses, 12))
    best_cov_mse = cov_mses[b_arange, best_cov_a]

    best_band_a = _argmin_last(np.round(band_mses, 12))
    best_band_mse_raw = band_mses[b_arange, best_band_a]
    best_band_mse_r = np.round(best_band_mse_raw, 12)

    pca_r = np.round(pca_mses, 12)
    best_pca_k_idx, best_pca_a = _select_pca_setting(pca_r)
    best_pca_mse_raw = pca_mses[b_arange, best_pca_k_idx, best_pca_a]
    best_pca_mse_r = pca_r[b_arange, best_pca_k_idx, best_pca_a]

    pls_r_cand = np.round(pls_mses[:, pls_idx], 12)
    best_pls_c_idx = pls_idx[np.argmin(pls_r_cand, axis=1)]
    best_pls_mse_raw = pls_mses[b_arange, best_pls_c_idx]
    best_pls_mse_r = np.round(best_pls_mse_raw, 12)

    comp_stack = np.stack([best_band_mse_r, best_pca_mse_r, best_pls_mse_r], axis=1)
    best_comp_idx = np.argmin(comp_stack, axis=1)

    pair_mses_5d = pair_mses.reshape(B, n_r, n_k, n_lam, n_alphas)
    pair_r = np.round(pair_mses_5d, 12)
    pair_finite = np.isfinite(pair_r)
    pair_ok = np.any(pair_finite, axis=(1, 2, 3, 4))
    pair_r_safe = np.where(pair_finite, pair_r, np.inf)
    best_r_idx, best_k_idx, best_pair_l_idx, best_pair_a = _select_paired_setting(pair_r_safe)
    best_pair_mse = pair_mses_5d[b_arange, best_r_idx, best_k_idx, best_pair_l_idx, best_pair_a]

    out: dict[str, Any] = {
        "best_cov_a": best_cov_a,
        "best_cov_mse": best_cov_mse,
        "best_band_a": best_band_a,
        "best_band_mse": best_band_mse_raw,
        "best_pca_k_idx": best_pca_k_idx,
        "best_pca_a": best_pca_a,
        "best_pca_mse": best_pca_mse_raw,
        "best_pls_c_idx": best_pls_c_idx,
        "best_pls_mse": best_pls_mse_raw,
        "best_comp_idx": best_comp_idx,
        "best_r_idx": best_r_idx,
        "best_k_idx": best_k_idx,
        "best_pair_l_idx": best_pair_l_idx,
        "best_pair_a": best_pair_a,
        "best_pair_mse": best_pair_mse,
        "rank_ok": rank_ok & pair_ok,
    }
    if compute_inner_diagnostics and inner_paired_counts:
        pair_stack = np.vstack(inner_paired_counts)
        rank_stack = np.vstack(inner_ranks)
        out["inner_min_paired_distinct"] = np.min(pair_stack, axis=0)
        out["inner_max_paired_distinct"] = np.max(pair_stack, axis=0)
        out["inner_min_numerical_rank"] = np.min(rank_stack, axis=0)
        out["inner_dropped_covariates"] = [sorted(s) for s in inner_dropped_cov_sets]
    return out


def _predict_grouped_ridge_outer(
    xp: Any,
    outer_div: dict[str, Any],
    b_idx: np.ndarray,
    Z_otr: Any,
    Z_ote: Any,
    alpha_indices: np.ndarray,
    alphas_xp: Any,
    out_preds: Any,
) -> None:
    """Fit outer ridge for each draw's single selected alpha and store predictions on device."""
    if b_idx.size == 0:
        return
    b_xp = xp.asarray(b_idx)
    sel_alphas = alphas_xp[xp.asarray(alpha_indices)]
    out_preds[b_xp] = batched_ridge_preds_single_alpha(
        xp,
        outer_div["X_cov_tr"][b_xp],
        outer_div["X_cov_val"][b_xp],
        Z_otr,
        Z_ote,
        outer_div["y_tr"],
        outer_div["w_tr"][b_xp],
        sel_alphas,
    )


def _run_draws_chunk_core(
    xp: Any,
    fold_dconsts: list[tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]],
    y_all_xp: Any,
    alphas_xp: Any,
    lam_xp: Any,
    part_band_xp: Any,
    M_chunk: np.ndarray,
    cfg: PairedLipidConfig,
    gcache: dict[str, Any],
    batch_size: int,
    return_oof_sd: bool,
    record_fallback: bool,
    spectral_trajectories: list[tuple[list[SpectralDivisionTrajectory], SpectralDivisionTrajectory]] | None = None,
) -> tuple[tuple[np.ndarray, np.ndarray] | dict[str, np.ndarray], set[int]]:
    """Evaluate a contiguous draw chunk on the active backend and return OOF summaries + bad indices."""
    n_sub, n_p = M_chunk.shape
    k_list = [int(k) for k in cfg.pca_dimensions]
    r_list = [int(r) for r in cfg.retained_directions]

    oof_cov = xp.zeros((n_sub, n_p), dtype=xp.float64)
    oof_paired = xp.zeros((n_sub, n_p), dtype=xp.float64)
    oof_strongest = xp.zeros((n_sub, n_p), dtype=xp.float64)
    bad_local_draws: set[int] = set()

    for b_start in range(0, n_sub, batch_size):
        b_end = min(b_start + batch_size, n_sub)
        B = b_end - b_start
        M_batch = xp.asarray(M_chunk[b_start:b_end], dtype=xp.float64)
        batch_ok = np.ones(B, dtype=bool)
        b_arange_xp = xp.arange(B)

        for fold_idx, (tr_idx, te_idx, in_dconsts, out_dconst) in enumerate(fold_dconsts):
            in_spec_slices: list[SpectralDivisionTrajectory] | None = None
            out_spec_slice: SpectralDivisionTrajectory | None = None
            if spectral_trajectories is not None:
                in_trajs, out_traj = spectral_trajectories[fold_idx]
                in_spec_slices = [t.slice(b_start, b_end) for t in in_trajs]
                out_spec_slice = out_traj.slice(b_start, b_end)

            sel = batched_select_inner_settings(
                xp,
                gcache,
                in_dconsts,
                M_batch,
                y_all_xp,
                alphas_xp,
                part_band_xp,
                cfg,
                inner_spec_trajs=in_spec_slices,
            )
            batch_ok &= sel["rank_ok"]

            best_cov_a = sel["best_cov_a"]
            best_band_a = sel["best_band_a"]
            best_pca_k_idx = sel["best_pca_k_idx"]
            best_pca_a = sel["best_pca_a"]
            best_pls_c_idx = sel["best_pls_c_idx"]
            best_comp_idx = sel["best_comp_idx"]
            best_r_idx = sel["best_r_idx"]
            best_k_idx = sel["best_k_idx"]
            best_pair_l_idx = sel["best_pair_l_idx"]
            best_pair_a = sel["best_pair_a"]

            tr_xp = out_dconst["tr_p"]
            te_xp = out_dconst["val_p"]
            outer_div = batched_division_prep(
                xp,
                gcache,
                out_dconst,
                M_batch[:, tr_xp],
                M_batch[:, te_xp],
                y_all_xp,
                alphas_xp,
                cfg,
                compute_pca_suff=False,
                compute_cov_preds=True,
                spec_traj_slice=out_spec_slice,
            )
            batch_ok &= _to_np(xp, outer_div["rank_ok"]) > 0.5
            if "part_pca_tr" in outer_div and "part_pca_val" in outer_div:
                pca_otr = outer_div["part_pca_tr"]
                pca_ote = outer_div["part_pca_val"]
            else:
                pca_otr = outer_div["part_pca_max"][:, tr_xp, :]
                pca_ote = outer_div["part_pca_max"][:, te_xp, :]

            # Solve outer paired generalized eigensystem only for valid draws' selected (k, lam).
            V_outer_sel: dict[int, Any] = {}
            b_k_by_k: dict[int, np.ndarray] = {}
            for k_idx, k in enumerate(k_list):
                b_k = np.flatnonzero(batch_ok & (best_k_idx == k_idx))
                if b_k.size == 0:
                    continue
                b_k_xp = xp.asarray(b_k)
                lam_sel = lam_xp[xp.asarray(best_pair_l_idx[b_k])]
                V_k = _whitened_generalized_eigh(
                    xp,
                    outer_div["W_max"][b_k_xp, :k, :k],
                    outer_div["T_max"][b_k_xp, :k, :k],
                    lam_sel,
                    grid_mode=False,
                    eigenvalue_tolerance=cfg.eigenvalue_tolerance,
                )
                v_finite = _to_np(xp, xp.all(xp.isfinite(V_k), axis=(-2, -1))) > 0.5
                if not np.all(v_finite):
                    batch_ok[b_k[~v_finite]] = False
                V_outer_sel[k_idx] = V_k
                b_k_by_k[k_idx] = b_k

            pred_paired_ote = xp.zeros((B, te_idx.size), dtype=xp.float64)
            for r_idx, r in enumerate(r_list):
                mask_b = np.flatnonzero(batch_ok & (best_r_idx == r_idx))
                if mask_b.size == 0:
                    continue
                Z_otr_sel = xp.zeros((mask_b.size, tr_idx.size, r), dtype=xp.float64)
                Z_ote_sel = xp.zeros((mask_b.size, te_idx.size, r), dtype=xp.float64)
                for k_idx, k in enumerate(k_list):
                    sub_m = np.flatnonzero(best_k_idx[mask_b] == k_idx)
                    if sub_m.size == 0:
                        continue
                    b_ids = mask_b[sub_m]
                    b_ids_xp = xp.asarray(b_ids)
                    b_k_all = b_k_by_k[k_idx]
                    pos_in_bk = xp.asarray(np.searchsorted(b_k_all, b_ids))
                    V_sel = V_outer_sel[k_idx][pos_in_bk, :, :r]
                    sub_m_xp = xp.asarray(sub_m)
                    Z_otr_sel[sub_m_xp] = pca_otr[b_ids_xp, :, :k] @ V_sel
                    Z_ote_sel[sub_m_xp] = pca_ote[b_ids_xp, :, :k] @ V_sel

                _predict_grouped_ridge_outer(
                    xp, outer_div, mask_b, Z_otr_sel, Z_ote_sel, best_pair_a[mask_b], alphas_xp, pred_paired_ote
                )

            pred_comp_sel = xp.zeros((B, te_idx.size), dtype=xp.float64)
            b_band = np.flatnonzero(batch_ok & (best_comp_idx == 0))
            _predict_grouped_ridge_outer(
                xp,
                outer_div,
                b_band,
                part_band_xp[tr_xp],
                part_band_xp[te_xp],
                best_band_a[b_band],
                alphas_xp,
                pred_comp_sel,
            )

            for k_idx, k in enumerate(k_list):
                b_pca = np.flatnonzero(batch_ok & (best_comp_idx == 1) & (best_pca_k_idx == k_idx))
                if b_pca.size > 0:
                    b_pca_xp = xp.asarray(b_pca)
                    _predict_grouped_ridge_outer(
                        xp,
                        outer_div,
                        b_pca,
                        pca_otr[b_pca_xp, :, :k],
                        pca_ote[b_pca_xp, :, :k],
                        best_pca_a[b_pca],
                        alphas_xp,
                        pred_comp_sel,
                    )

            b_pls = np.flatnonzero(batch_ok & (best_comp_idx == 2))
            if b_pls.size > 0:
                b_pls_xp = xp.asarray(b_pls)
                b_pls_caps = best_pls_c_idx[b_pls]
                max_comp_outer = int(np.max(b_pls_caps)) + 1
                pls_div = {
                    "w_tr": outer_div["w_tr"][b_pls_xp],
                    "w_val": outer_div["w_val"][b_pls_xp],
                    "y_tr": outer_div["y_tr"],
                    "y_val": outer_div["y_val"],
                    "X_cov_tr": outer_div["X_cov_tr"][b_pls_xp],
                    "X_cov_val": outer_div["X_cov_val"][b_pls_xp],
                    "scale1_sq": outer_div["scale1_sq"][b_pls_xp],
                    "scale2_sq": outer_div["scale2_sq"][b_pls_xp],
                    "K_p1_tr": outer_div["K_p1_tr"],
                    "K_p1_vt": outer_div["K_p1_vt"],
                    "K_p2_tr": outer_div["K_p2_tr"],
                    "K_p2_vt": outer_div["K_p2_vt"],
                }
                _, pr_pls = batched_pls_sse_and_preds(
                    xp,
                    pls_div,
                    max_comp_outer,
                    return_sse=False,
                    target_comp_idx=b_pls_caps,
                )
                pred_comp_sel[b_pls_xp] = pr_pls

            oof_cov[b_start:b_end, te_xp] = outer_div["preds_cov"][b_arange_xp, :, xp.asarray(best_cov_a)]
            oof_paired[b_start:b_end, te_xp] = pred_paired_ote
            oof_strongest[b_start:b_end, te_xp] = pred_comp_sel
            del outer_div

        if not np.all(batch_ok):
            bad_local = np.flatnonzero(~batch_ok)
            bad_global_xp = xp.asarray(b_start + bad_local)
            oof_cov[bad_global_xp] = xp.nan
            oof_paired[bad_global_xp] = xp.nan
            oof_strongest[bad_global_xp] = xp.nan
            if record_fallback:
                bad_local_draws.update(int(b_start + idx) for idx in bad_local)

    M_sub_xp = xp.asarray(M_chunk, dtype=xp.float64)
    w_norm_all = M_sub_xp / xp.sum(M_sub_xp, axis=1, keepdims=True)
    diff_cov = y_all_xp[None, :] - oof_cov
    diff_pair = y_all_xp[None, :] - oof_paired
    diff_comp = y_all_xp[None, :] - oof_strongest
    sq_cov = diff_cov * diff_cov
    sq_pair = diff_pair * diff_pair
    sq_comp = diff_comp * diff_comp

    d_pair_mat = sq_cov - sq_pair
    d_comp_mat = sq_comp - sq_pair
    d_strong_mat = sq_cov - sq_comp

    delta_pair_xp = xp.sum(w_norm_all * d_pair_mat, axis=1)
    delta_comp_xp = xp.sum(w_norm_all * d_comp_mat, axis=1)
    delta_strong_xp = xp.sum(w_norm_all * d_strong_mat, axis=1)
    delta_pair = _to_np(xp, delta_pair_xp)
    delta_comp = _to_np(xp, delta_comp_xp)

    if return_oof_sd:
        delta_strong = _to_np(xp, delta_strong_xp)
        var_pair = xp.sum(w_norm_all * ((d_pair_mat - delta_pair_xp[:, None]) ** 2), axis=1)
        var_comp = xp.sum(w_norm_all * ((d_comp_mat - delta_comp_xp[:, None]) ** 2), axis=1)
        var_strong = xp.sum(w_norm_all * ((d_strong_mat - delta_strong_xp[:, None]) ** 2), axis=1)
        sd_pair = _to_np(xp, xp.sqrt(xp.maximum(var_pair, 0.0)))
        sd_comp = _to_np(xp, xp.sqrt(xp.maximum(var_comp, 0.0)))
        sd_strong = _to_np(xp, xp.sqrt(xp.maximum(var_strong, 0.0)))
        if record_fallback:
            bad_final = np.flatnonzero(
                ~(
                    np.isfinite(delta_pair)
                    & np.isfinite(delta_comp)
                    & np.isfinite(delta_strong)
                    & np.isfinite(sd_pair)
                    & np.isfinite(sd_comp)
                    & np.isfinite(sd_strong)
                )
            )
            bad_local_draws.update(int(idx) for idx in bad_final)
        return {
            "delta_pair": delta_pair,
            "delta_comp": delta_comp,
            "delta_strong": delta_strong,
            "sd_pair": sd_pair,
            "sd_comp": sd_comp,
            "sd_strong": sd_strong,
        }, bad_local_draws

    if record_fallback:
        bad_final = np.flatnonzero(~(np.isfinite(delta_pair) & np.isfinite(delta_comp)))
        bad_local_draws.update(int(idx) for idx in bad_final)
    return (delta_pair, delta_comp), bad_local_draws


def _run_cpu_draw_chunk_worker(
    fold_dconsts: list[tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]],
    y_all: np.ndarray,
    alphas: np.ndarray,
    lam_grid: np.ndarray,
    part_band_src: np.ndarray,
    M_chunk: np.ndarray,
    cfg: PairedLipidConfig,
    batch_size: int,
    return_oof_sd: bool,
    record_fallback: bool,
    spectral_trajectories_chunk: list[tuple[list[SpectralDivisionTrajectory], SpectralDivisionTrajectory]] | None = None,
) -> tuple[tuple[np.ndarray, np.ndarray] | dict[str, np.ndarray], set[int]]:
    """Top-level picklable worker evaluating a CPU bootstrap draw chunk under single-thread BLAS."""
    prev_env = os.environ.get("_PAIRED_LIPID_IN_BOOTSTRAP_WORKER")
    os.environ["_PAIRED_LIPID_IN_BOOTSTRAP_WORKER"] = "1"
    try:
        with _single_thread_blas():
            return _run_draws_chunk_core(
                np,
                fold_dconsts,
                np.asarray(y_all, dtype=np.float64),
                np.asarray(alphas, dtype=np.float64),
                np.asarray(lam_grid, dtype=np.float64),
                np.asarray(part_band_src, dtype=np.float64),
                M_chunk,
                cfg,
                {},
                batch_size,
                return_oof_sd,
                record_fallback,
                spectral_trajectories=spectral_trajectories_chunk,
            )
    finally:
        if prev_env is None:
            os.environ.pop("_PAIRED_LIPID_IN_BOOTSTRAP_WORKER", None)
        else:
            os.environ["_PAIRED_LIPID_IN_BOOTSTRAP_WORKER"] = prev_env


def _run_batched_multiplicity_draws(
    dataset: CohortDataset,
    y_all: np.ndarray,
    outer_splits: list[tuple[np.ndarray, np.ndarray]],
    inner_splits_per_fold: list[list[tuple[np.ndarray, np.ndarray]]],
    M_sub: np.ndarray,
    cfg: PairedLipidConfig,
    gcache: dict[str, Any],
    *,
    use_gpu: bool | None = None,
    batch_size: int | None = None,
    execution_metadata: dict[str, Any] | None = None,
    fallback_draws_out: set[int] | None = None,
    return_oof_sd: bool = False,
    single_sample_downstream: bool = False,
    n_jobs: int | None = None,
    spectral_trajectories: list[tuple[list[SpectralDivisionTrajectory], SpectralDivisionTrajectory]] | None = None,
) -> tuple[np.ndarray, np.ndarray] | dict[str, np.ndarray]:
    """Evaluate multiplicity bootstrap draws in batches on GPU (CuPy) or CPU (NumPy)."""
    n_sub, n_p = M_sub.shape
    xp, backend_meta = resolve_array_backend(use_gpu, n_sub, gpu_probe=_get_gpu_xp)
    if execution_metadata is not None:
        execution_metadata.update(
            device=backend_meta["device"],
            gpu_requested=backend_meta["gpu_requested"],
            gpu_fallback=backend_meta["gpu_fallback"],
        )
        if "gpu_fallback_reason" in backend_meta:
            execution_metadata["gpu_fallback_reason"] = backend_meta["gpu_fallback_reason"]

    y_all_xp = xp.asarray(y_all, dtype=xp.float64)
    alphas_xp = xp.asarray(list(cfg.ridge_alpha_grid), dtype=xp.float64)
    lam_xp = xp.asarray(list(cfg.lambda_grid), dtype=xp.float64)
    part_band_src = (
        _get_single_sample_gram_cache(dataset, gcache)["part_band_mat_single"]
        if single_sample_downstream
        else gcache["part_band_mat"]
    )
    part_band_xp = xp.asarray(part_band_src, dtype=xp.float64)

    skip_mfa_dev = spectral_trajectories is not None
    fold_dconsts = []
    with _single_thread_blas():
        for (tr_idx, te_idx), in_splits in zip(outer_splits, inner_splits_per_fold):
            in_dconsts = [
                _precompute_division_constants(
                    xp,
                    gcache,
                    dataset,
                    tr_idx[in_tr_rel],
                    tr_idx[in_val_rel],
                    cfg,
                    single_sample_downstream=single_sample_downstream,
                    skip_mfa_device_transfer=skip_mfa_dev,
                )
                for in_tr_rel, in_val_rel in in_splits
            ]
            out_dconst = _precompute_division_constants(
                xp,
                gcache,
                dataset,
                tr_idx,
                te_idx,
                cfg,
                single_sample_downstream=single_sample_downstream,
                skip_mfa_device_transfer=skip_mfa_dev,
            )
            fold_dconsts.append((tr_idx, te_idx, in_dconsts, out_dconst))

    if batch_size is None:
        n_samples_outer = max((int(out_dc["s_div"].size) for _, _, _, out_dc in fold_dconsts), default=n_p)
        batch_size = suggest_batch_size(xp, n_sub, n_samples_outer)

    record_fallback = fallback_draws_out is not None
    if xp is np:
        n_workers = _resolve_cpu_bootstrap_workers(n_sub, n_jobs, min_parallel_draws=256)
        if n_workers > 1:
            if execution_metadata is not None:
                execution_metadata["cpu_n_workers"] = n_workers
            n_chunks = min(n_sub, n_workers * 2)
            idx_chunks = np.array_split(np.arange(n_sub), n_chunks)
            alphas_np = np.asarray(list(cfg.ridge_alpha_grid), dtype=np.float64)
            lam_np = np.asarray(list(cfg.lambda_grid), dtype=np.float64)
            y_all_np = np.asarray(y_all, dtype=np.float64)
            part_band_np = np.asarray(part_band_src, dtype=np.float64)

            def _slice_traj(c_start: int, c_end: int) -> list[tuple[list[SpectralDivisionTrajectory], SpectralDivisionTrajectory]] | None:
                if spectral_trajectories is None:
                    return None
                return [
                    ([t.slice(c_start, c_end) for t in in_t], out_t.slice(c_start, c_end))
                    for in_t, out_t in spectral_trajectories
                ]

            chunk_outputs = Parallel(n_jobs=n_workers)(
                delayed(_run_cpu_draw_chunk_worker)(
                    fold_dconsts,
                    y_all_np,
                    alphas_np,
                    lam_np,
                    part_band_np,
                    np.ascontiguousarray(M_sub[int(c[0]) : int(c[-1]) + 1], dtype=np.float64),
                    cfg,
                    min(batch_size, int(c.size)),
                    return_oof_sd,
                    record_fallback,
                    _slice_traj(int(c[0]), int(c[-1]) + 1),
                )
                for c in idx_chunks
            )
            if fallback_draws_out is not None:
                for c, (_, local_bad) in zip(idx_chunks, chunk_outputs):
                    c_start = int(c[0])
                    fallback_draws_out.update(c_start + int(idx) for idx in local_bad)
            if return_oof_sd:
                res_dicts = [res for res, _ in chunk_outputs]
                return {
                    key: np.concatenate([d[key] for d in res_dicts], axis=0)
                    for key in ("delta_pair", "delta_comp", "delta_strong", "sd_pair", "sd_comp", "sd_strong")
                }
            res_pairs = [res for res, _ in chunk_outputs]
            return (
                np.concatenate([p[0] for p in res_pairs], axis=0),
                np.concatenate([p[1] for p in res_pairs], axis=0),
            )

    blas_ctx = _single_thread_blas() if xp is np else contextlib.nullcontext()
    with blas_ctx:
        res_out, bad_draws = _run_draws_chunk_core(
            xp,
            fold_dconsts,
            y_all_xp,
            alphas_xp,
            lam_xp,
            part_band_xp,
            M_sub,
            cfg,
            gcache,
            batch_size,
            return_oof_sd,
            record_fallback,
            spectral_trajectories=spectral_trajectories,
        )
    if fallback_draws_out is not None and bad_draws:
        fallback_draws_out.update(bad_draws)
    return res_out
