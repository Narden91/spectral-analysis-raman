"""Fold-local preprocessors and weighted linear models: covariate/spectral scaling, paired moments, ridge, PLS."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import linalg

from src.analysis.paired_lipid.cohort import (
    COVARIATE_PRESETS,
    PRE_EVENT_RISK_FACTORS,
    CohortDataset,
    PairedLipidConfig,
)


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    """Return the exact symmetric part ``0.5 * (M + M.T)`` of a square matrix."""
    return 0.5 * (matrix + matrix.T)


def _normalize_weights(weights: np.ndarray, *, context: str) -> np.ndarray:
    """Validate 1D finite non-negative weights with strictly positive sum and return unit-sum weights."""
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or w.size == 0:
        raise ValueError(f"{context}: weights must be a non-empty 1D array.")
    if not np.all(np.isfinite(w)):
        raise ValueError(f"{context}: weights must be finite.")
    if np.any(w < 0.0):
        raise ValueError(f"{context}: weights must be non-negative.")
    w_sum = float(np.sum(w))
    if not np.isfinite(w_sum) or w_sum <= 0.0:
        raise ValueError(f"{context}: weights must have a strictly positive sum.")
    return w / w_sum


def _weighted_mean_and_scale(
    X: np.ndarray,
    w: np.ndarray,
    *,
    min_var: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute weighted column means, scales and centered matrix using BLAS matrix-vector products."""
    means = w @ X
    centered = X - means
    variances = w @ (centered * centered)
    scales = np.sqrt(np.maximum(variances, min_var))
    return means, scales, centered


def _enforce_deterministic_signs(vectors: np.ndarray, *, axis: int) -> np.ndarray:
    """Flip each vector along ``axis`` so its largest-magnitude entry is positive."""
    if vectors.size == 0:
        return vectors
    max_abs_idx = np.argmax(np.abs(vectors), axis=axis)
    if axis == 1:
        signs = np.sign(vectors[np.arange(vectors.shape[0]), max_abs_idx])
        signs[signs == 0] = 1.0
        vectors *= signs[:, None]
    elif axis == 0:
        signs = np.sign(vectors[max_abs_idx, np.arange(vectors.shape[1])])
        signs[signs == 0] = 1.0
        vectors *= signs[None, :]
    else:
        raise ValueError(f"Unsupported sign axis: {axis}")
    return vectors


@dataclass(slots=True)
class CovariatePreprocessor:
    """Training-only covariate imputation, missingness indicator, and standardization."""

    feature_names: list[str]
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    dropped_columns: list[str]
    include_missing_indicators_for: list[str]
    preset: str
    _raw_columns: tuple[str, ...] = field(default=(), repr=False, compare=False)
    _base_indices: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=int), repr=False, compare=False
    )
    _miss_indices: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=int), repr=False, compare=False
    )

    @classmethod
    def fit(
        cls,
        cov_frame: pd.DataFrame,
        train_indices: np.ndarray,
        participant_weights: np.ndarray,
        *,
        preset: str = "primary",
        raw_cache: dict[str, Any] | None = None,
    ) -> CovariatePreprocessor:
        raw_mat, raw_cols = cls.raw_design_matrix(cov_frame, preset=preset, raw_cache=raw_cache)
        w_norm_all = _normalize_weights(participant_weights, context="CovariatePreprocessor.fit")
        active = w_norm_all > 0.0
        sub_mat = raw_mat[np.asarray(train_indices, dtype=int)[active]]
        w = w_norm_all[active]

        kept_cols: list[str] = []
        base_indices_list: list[int] = []
        dropped_cols: list[str] = []
        miss_indicator_cols: list[str] = []
        miss_indices_list: list[int] = []
        medians_list: list[float] = []
        base_design: list[np.ndarray] = []
        miss_design: list[np.ndarray] = []

        for c_idx, col in enumerate(raw_cols):
            vals = sub_mat[:, c_idx]
            finite = np.isfinite(vals)
            if not np.any(finite):
                dropped_cols.append(col)
                continue
            f_vals = vals[finite]
            f_w = w[finite]
            order = np.argsort(f_vals)
            cdf = np.cumsum(f_w[order]) / np.sum(f_w)
            med = float(f_vals[order[np.searchsorted(cdf, 0.5)]])
            imputed = np.where(finite, vals, med)
            if np.ptp(imputed) <= 1e-12:
                dropped_cols.append(col)
                continue
            kept_cols.append(col)
            base_indices_list.append(c_idx)
            medians_list.append(med)
            base_design.append(imputed)
            if not np.all(finite):
                miss_indicator_cols.append(col)
                miss_indices_list.append(c_idx)
                miss_design.append((~finite).astype(float))

        if not kept_cols:
            return cls(
                feature_names=[],
                medians=np.empty(0, dtype=float),
                means=np.empty(0, dtype=float),
                scales=np.empty(0, dtype=float),
                dropped_columns=dropped_cols,
                include_missing_indicators_for=[],
                preset=preset,
                _raw_columns=raw_cols,
            )

        names = [*kept_cols, *(f"{col}__missing" for col in miss_indicator_cols)]
        X = np.column_stack([*base_design, *miss_design])
        means, scales, _ = _weighted_mean_and_scale(X, w)
        return cls(
            feature_names=names,
            medians=np.asarray(medians_list, dtype=float),
            means=means,
            scales=scales,
            dropped_columns=dropped_cols,
            include_missing_indicators_for=miss_indicator_cols,
            preset=preset,
            _raw_columns=raw_cols,
            _base_indices=np.asarray(base_indices_list, dtype=int),
            _miss_indices=np.asarray(miss_indices_list, dtype=int),
        )

    def transform(
        self,
        cov_frame: pd.DataFrame,
        indices: np.ndarray,
        *,
        raw_cache: dict[str, Any] | None = None,
    ) -> np.ndarray:
        idx_arr = np.asarray(indices, dtype=int)
        if not self.feature_names:
            return np.zeros((idx_arr.size, 0), dtype=float)

        raw_mat, raw_cols = self.raw_design_matrix(cov_frame, preset=self.preset, raw_cache=raw_cache)
        sub_mat = raw_mat[idx_arr]

        if (
            raw_cols == self._raw_columns
            and self._base_indices.size == self.medians.size
            and self._miss_indices.size == len(self.include_missing_indicators_for)
        ):
            base_vals = sub_mat[:, self._base_indices]
            base_imputed = np.where(np.isfinite(base_vals), base_vals, self.medians[None, :])
            if self._miss_indices.size > 0:
                miss_vals = (~np.isfinite(sub_mat[:, self._miss_indices])).astype(float)
                X = np.hstack([base_imputed, miss_vals])
            else:
                X = base_imputed
            return (X - self.means) / self.scales

        # Fallback when transforming a frame whose raw column set differs from training.
        col_to_idx = {c: i for i, c in enumerate(raw_cols)}
        base_cols = [c for c in self.feature_names if not c.endswith("__missing")]
        cols_out: list[np.ndarray] = []
        for idx, col in enumerate(base_cols):
            vals = (
                sub_mat[:, col_to_idx[col]]
                if col in col_to_idx
                else np.full(idx_arr.size, np.nan, dtype=float)
            )
            cols_out.append(np.where(np.isfinite(vals), vals, self.medians[idx]))
        for col in self.include_missing_indicators_for:
            if f"{col}__missing" in self.feature_names:
                vals = (
                    sub_mat[:, col_to_idx[col]]
                    if col in col_to_idx
                    else np.full(idx_arr.size, np.nan, dtype=float)
                )
                cols_out.append((~np.isfinite(vals)).astype(float))
        X = np.column_stack(cols_out)
        return (X - self.means) / self.scales

    @classmethod
    def raw_design_matrix(
        cls,
        cov_frame: pd.DataFrame,
        *,
        preset: str,
        raw_cache: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, tuple[str, ...]]:
        """Return a cached read-only float64 raw design matrix and immutable column names."""
        cache_key = f"raw_cov_matrix__{preset}"
        if raw_cache is not None and cache_key in raw_cache:
            cached_mat, cached_cols = raw_cache[cache_key]
            return cached_mat, tuple(cached_cols)
        raw_df = cls._build_raw_design(cov_frame, preset=preset)
        mat = np.ascontiguousarray(raw_df.to_numpy(dtype=np.float64), dtype=np.float64)
        mat.setflags(write=False)
        cols = tuple(str(c) for c in raw_df.columns)
        res = (mat, cols)
        if raw_cache is not None:
            raw_cache[cache_key] = res
        return res

    @classmethod
    def _get_raw_matrix(
        cls,
        cov_frame: pd.DataFrame,
        *,
        preset: str,
        raw_cache: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        mat, cols = cls.raw_design_matrix(cov_frame, preset=preset, raw_cache=raw_cache)
        return mat, list(cols)

    @staticmethod
    def _build_raw_design(df: pd.DataFrame, *, preset: str) -> pd.DataFrame:
        if preset not in COVARIATE_PRESETS:
            raise ValueError(f"Unknown covariate preset: {preset!r}")
        out = pd.DataFrame(index=df.index)
        out["age_years"] = pd.to_numeric(df.get("age_years"), errors="coerce")
        if preset == "quadratic_age_and_label":
            out["age_years_sq"] = out["age_years"] ** 2

        sex_s = df.get("sex", pd.Series(np.nan, index=df.index)).astype(str).str.strip().str.lower()
        sex_num = np.where(sex_s == "female", 1.0, np.where(sex_s == "male", 0.0, np.nan))
        out["sex_female"] = sex_num
        out["statin"] = pd.to_numeric(df.get("statin"), errors="coerce")

        # Acquisition modes are pooled as equivalent, so the label trend is the only technical covariate.
        if preset != "no_label_trend":
            lbl = pd.to_numeric(df.get("sample_label_trend"), errors="coerce")
            out["sample_label_trend"] = lbl
            if preset in ("quadratic_label_trend", "quadratic_age_and_label"):
                out["sample_label_trend_sq"] = lbl**2

        if preset == "pre_event_risk_factors":
            for rf in PRE_EVENT_RISK_FACTORS:
                if rf not in df.columns:
                    continue
                if rf == "smoking_status":
                    sm = df[rf].astype(str).str.strip().str.lower()
                    out["smoking_current"] = np.where(
                        sm == "current", 1.0, np.where(sm.isin(["never", "former"]), 0.0, np.nan)
                    )
                    out["smoking_former"] = np.where(
                        sm == "former", 1.0, np.where(sm.isin(["never", "current"]), 0.0, np.nan)
                    )
                else:
                    out[rf] = pd.to_numeric(df[rf], errors="coerce")
        return out


@dataclass(slots=True)
class WeightedSpectralPreprocessor:
    """Participant-weighted MFA block scaling and truncated SVD for training divisions."""

    mean_1300: np.ndarray
    mean_2850: np.ndarray
    scale_1300: float
    scale_2850: float
    pca_components: np.ndarray
    pca_singular_values: np.ndarray
    numerical_rank: int
    n_distinct_paired_train: int

    @classmethod
    def fit(
        cls,
        dataset: CohortDataset,
        train_participant_indices: np.ndarray,
        participant_multiplicities: np.ndarray,
        *,
        cfg: PairedLipidConfig = PairedLipidConfig(),
    ) -> WeightedSpectralPreprocessor:
        train_p = np.asarray(train_participant_indices, dtype=int)
        mult = np.asarray(participant_multiplicities, dtype=float)
        if mult.ndim != 1 or mult.size == 0 or not np.all(np.isfinite(mult)) or np.any(mult < 0.0):
            raise ValueError("WeightedSpectralPreprocessor requires finite non-negative participant multiplicities.")
        if np.any(mult <= 0.0):
            keep = mult > 0.0
            train_p = train_p[keep]
            mult = mult[keep]
        if train_p.size == 0:
            raise ValueError("WeightedSpectralPreprocessor requires at least one positively weighted participant.")

        n_eff = float(np.sum(mult))
        p_to_mult = np.zeros(dataset.n_participants, dtype=float)
        p_to_mult[train_p] = mult

        sample_p = dataset.sample_participant_idx
        s_indices = np.flatnonzero(p_to_mult[sample_p] > 0.0)
        if s_indices.size < 2:
            raise ValueError("Block requires at least two weighted observations.")
        s_p = sample_p[s_indices]

        m_i = dataset.participant_sample_counts[s_p].astype(float)
        w_samples = _normalize_weights(
            p_to_mult[s_p] / (n_eff * m_i),
            context="WeightedSpectralPreprocessor.fit",
        )

        X1 = dataset.sample_1300[s_indices]
        X2 = dataset.sample_2850[s_indices]

        mean1 = w_samples @ X1
        mean2 = w_samples @ X2

        # Effective total sample count for sample covariance denominator.
        m_total = float(np.sum(p_to_mult[s_p]))
        denom_factor = m_total / max(m_total - 1.0, 1.0)
        sqrt_w = np.sqrt(w_samples * denom_factor)[:, None]

        # Build each block's weighted matrix and sample Gram matrix once; reuse both for
        # leading-eigenvalue MFA scaling, combined Gram eigendecomposition and loading recovery.
        SW1 = sqrt_w * (X1 - mean1)
        SW2 = sqrt_w * (X2 - mean2)
        n_s = SW1.shape[0]
        G1 = _symmetrize(SW1 @ SW1.T)
        G2 = _symmetrize(SW2 @ SW2.T)
        if not (np.all(np.isfinite(G1)) and np.all(np.isfinite(G2))):
            raise ValueError("Non-finite spectral block Gram matrix in WeightedSpectralPreprocessor.fit.")

        eig1_1300 = float(
            linalg.eigvalsh(G1, subset_by_index=[n_s - 1, n_s - 1], driver="evr", check_finite=False)[-1]
        )
        if not np.isfinite(eig1_1300) or eig1_1300 <= cfg.eigenvalue_tolerance:
            raise ValueError(f"Degenerate leading block eigenvalue: {eig1_1300}")

        eig1_2850 = float(
            linalg.eigvalsh(G2, subset_by_index=[n_s - 1, n_s - 1], driver="evr", check_finite=False)[-1]
        )
        if not np.isfinite(eig1_2850) or eig1_2850 <= cfg.eigenvalue_tolerance:
            raise ValueError(f"Degenerate leading block eigenvalue: {eig1_2850}")

        scale1 = float(1.0 / np.sqrt(eig1_1300))
        scale2 = float(1.0 / np.sqrt(eig1_2850))

        # Method-of-snapshots Gram eigendecomposition: (n_samples x n_samples) instead of (n_samples x p_total) SVD.
        gram = _symmetrize(G1 * (scale1 * scale1) + G2 * (scale2 * scale2))
        eigvals, eigvecs = np.linalg.eigh(gram)
        order = np.argsort(eigvals)[::-1]
        eigvals = np.maximum(eigvals[order], 0.0)
        eigvecs = eigvecs[:, order]
        s_vals = np.sqrt(eigvals)

        max_dim = max(SW1.shape[0], SW1.shape[1] + SW2.shape[1])
        eps = float(np.finfo(float).eps)
        s0 = max(float(s_vals[0]) if s_vals.size else 1.0, 1.0)
        # Forming SW @ SW.T squares singular values, raising zero modes to O(sqrt(max_dim * eps) * s0).
        # Fall back to exact primal SVD whenever any requested top-k mode approaches that floor.
        gram_s_tol = float(np.sqrt(max_dim * eps)) * s0
        max_cfg_k = max(cfg.pca_dimensions)
        if s_vals.size < max_cfg_k or float(s_vals[max_cfg_k - 1]) <= 10.0 * gram_s_tol:
            _, s_vals, vt = linalg.svd(
                np.hstack([SW1 * scale1, SW2 * scale2]),
                full_matrices=False,
                check_finite=False,
            )
            tol = max_dim * eps * max(float(s_vals[0]) if s_vals.size else 1.0, 1.0)
            rank = int(np.sum(s_vals > max(tol, cfg.rank_tolerance)))
            max_k = min(max_cfg_k, rank, s_vals.size)
            s_kept = s_vals[:max_k].copy()
            comps = vt[:max_k].copy()
        else:
            rank = int(np.sum(s_vals > max(gram_s_tol, cfg.rank_tolerance)))
            max_k = min(max_cfg_k, rank, s_vals.size)
            s_kept = s_vals[:max_k].copy()
            u_kept = eigvecs[:, :max_k]
            inv_s = 1.0 / s_kept[:, None]
            comps_1300 = (scale1 * inv_s) * (u_kept.T @ SW1)
            comps_2850 = (scale2 * inv_s) * (u_kept.T @ SW2)
            comps = np.hstack([comps_1300, comps_2850])

        # Deterministic sign rule: largest-magnitude loading element in each component is positive.
        _enforce_deterministic_signs(comps, axis=1)

        paired_in_train = np.intersect1d(train_p, dataset.paired_participant_indices, assume_unique=False)
        return cls(
            mean_1300=mean1,
            mean_2850=mean2,
            scale_1300=scale1,
            scale_2850=scale2,
            pca_components=comps,
            pca_singular_values=s_kept,
            numerical_rank=rank,
            n_distinct_paired_train=int(paired_in_train.size),
        )

    @staticmethod
    def _leading_eigenvalue(weighted_matrix: np.ndarray, tol: float) -> float:
        if weighted_matrix.shape[0] < 2:
            raise ValueError("Block requires at least two weighted observations.")
        # Gram matrix is n_samples x n_samples (typically <= 200), much smaller than p_b x p_b.
        gram = _symmetrize(weighted_matrix @ weighted_matrix.T)
        if not np.all(np.isfinite(gram)):
            raise ValueError("Non-finite block Gram matrix.")
        n_s = gram.shape[0]
        top_eig = float(
            linalg.eigvalsh(gram, subset_by_index=[n_s - 1, n_s - 1], driver="evr", check_finite=False)[-1]
        )
        if not np.isfinite(top_eig) or top_eig <= tol:
            raise ValueError(f"Degenerate leading block eigenvalue: {top_eig}")
        return top_eig

    def transform_scaled_blocks(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Center and block-scale spectra without PCA truncation."""
        return np.hstack(
            [
                (X1 - self.mean_1300) * self.scale_1300,
                (X2 - self.mean_2850) * self.scale_2850,
            ]
        )

    def transform_pca(self, X1: np.ndarray, X2: np.ndarray, k: int | None = None) -> np.ndarray:
        """Project spectra into top-k weighted MFA-PCA coordinates without allocating centered copies."""
        comps = self.pca_components if k is None else self.pca_components[:k]
        p1 = self.mean_1300.size
        w1 = self.scale_1300 * comps[:, :p1].T
        w2 = self.scale_2850 * comps[:, p1:].T
        return (X1 @ w1 - self.mean_1300 @ w1) + (X2 @ w2 - self.mean_2850 @ w2)

    def project_linear_subspace(self, X1: np.ndarray, X2: np.ndarray, k: int, V_r: np.ndarray) -> np.ndarray:
        """Fuse block-PCA and paired subspace weights into (p_b, r) before multiplying spectra."""
        comps = self.pca_components[:k]
        p1 = self.mean_1300.size
        w1_r = self.scale_1300 * (comps[:, :p1].T @ V_r)
        w2_r = self.scale_2850 * (comps[:, p1:].T @ V_r)
        return (X1 @ w1_r - self.mean_1300 @ w1_r) + (X2 @ w2_r - self.mean_2850 @ w2_r)

    def candidate_pca_dims(self, cfg: PairedLipidConfig) -> list[int]:
        """Return deduplicated valid PCA dimensions capped by rank and n_p - 1."""
        cap = min(self.numerical_rank, self.n_distinct_paired_train - 1, self.pca_components.shape[0])
        if cap < 1:
            return []
        dims: list[int] = []
        for d in cfg.pca_dimensions:
            capped = min(int(d), cap)
            if capped >= 1 and capped not in dims:
                dims.append(capped)
        return dims


@dataclass(slots=True)
class PairedMoments:
    """Precomputed pair disagreement W and between-participant T in max-PCA or band space."""

    W_max: np.ndarray
    T_max: np.ndarray
    mean_ab_shift: np.ndarray
    n_paired_multiplicity: float
    n_paired_distinct: int

    @classmethod
    def from_sample_coordinates(
        cls,
        sample_coords: np.ndarray,
        dataset: CohortDataset,
        train_participant_indices: np.ndarray,
        participant_multiplicities: np.ndarray,
    ) -> PairedMoments:
        p_to_mult = np.zeros(dataset.n_participants, dtype=float)
        p_to_mult[train_participant_indices] = participant_multiplicities

        paired_p = dataset.paired_participant_indices
        mult_p = p_to_mult[paired_p]
        active = mult_p > 0.0
        n_distinct = int(np.sum(active))
        if n_distinct < 2:
            raise ValueError("At least two distinct paired training participants are required.")

        mult_active = mult_p[active]
        idx_a = dataset.pair_sample_a_indices[active]
        idx_b = dataset.pair_sample_b_indices[active]

        x_a = sample_coords[idx_a]
        x_b = sample_coords[idx_b]
        diffs = x_a - x_b
        means = 0.5 * (x_a + x_b)

        n_p = float(np.sum(mult_active))
        if not np.isfinite(n_p) or n_p <= 1.0:
            raise ValueError("Total paired multiplicity must exceed 1.")

        w_diff = np.sqrt(mult_active / (2.0 * n_p))[:, None] * diffs
        W = _symmetrize(w_diff.T @ w_diff)

        u_bar = (mult_active @ means) / n_p
        u_centered = means - u_bar
        w_u = np.sqrt(mult_active / (n_p - 1.0))[:, None] * u_centered
        T = _symmetrize(w_u.T @ w_u)

        mean_shift = (mult_active @ diffs) / n_p
        return cls(
            W_max=W,
            T_max=T,
            mean_ab_shift=mean_shift,
            n_paired_multiplicity=n_p,
            n_paired_distinct=n_distinct,
        )

    def solve_directions(
        self,
        k: int,
        lam: float,
        *,
        eigenvalue_tolerance: float = 1e-10,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Solve T v = rho (W + lam * tau * I) v for the top-k subspace in O(k^3)."""
        W_k = self.W_max[:k, :k]
        T_k = self.T_max[:k, :k]
        if not (np.all(np.isfinite(W_k)) and np.all(np.isfinite(T_k))):
            raise ValueError("Non-finite covariance matrices in paired eigensolver.")

        tr_t = float(np.trace(T_k))
        tau = tr_t / float(k)
        if not np.isfinite(tau) or tau <= eigenvalue_tolerance:
            raise ValueError(f"Non-positive or degenerate tau={tau} in paired eigensolver.")

        denom = _symmetrize(W_k + float(lam) * tau * np.eye(k, dtype=float))

        eigvals, eigvecs = linalg.eigh(T_k, denom, check_finite=True)
        order = np.argsort(eigvals)[::-1]
        eigvals = np.asarray(eigvals[order], dtype=float)
        eigvecs = np.asarray(eigvecs[:, order], dtype=float)

        # Normalize in the regularized denominator metric and enforce deterministic signs.
        metric_norms = np.sqrt(np.maximum(np.sum(eigvecs * (denom @ eigvecs), axis=0), 1e-15))
        eigvecs = eigvecs / metric_norms[None, :]
        _enforce_deterministic_signs(eigvecs, axis=0)
        return eigvecs, eigvals, tau


@dataclass(slots=True)
class MultiAlphaRidgeResult:
    """Vectorized multi-penalty ridge fit for a single representation."""

    alphas: np.ndarray
    intercepts: np.ndarray
    coef_cov: np.ndarray
    coef_spec: np.ndarray
    z_mean: np.ndarray
    z_scale: np.ndarray


def fit_ridge_grid(
    X_cov_train: np.ndarray,
    Z_spec_train: np.ndarray | None,
    y_train: np.ndarray,
    participant_weights: np.ndarray,
    alphas: Sequence[float],
) -> MultiAlphaRidgeResult:
    """Fit weighted ridge regression across all candidate alphas via eigendecomposition of the weighted covariance matrix.

    Solves ``min_{b0, b} (1 / N_eff) sum_i w_i (y_i - b0 - x_i^T b)^2 + alpha ||b||^2``
    with an unpenalized intercept.
    """
    w = _normalize_weights(participant_weights, context="fit_ridge_grid")
    p_cov = X_cov_train.shape[1]

    if Z_spec_train is not None and Z_spec_train.shape[1] > 0:
        z_mean, z_scale, z_cent = _weighted_mean_and_scale(Z_spec_train, w)
        Z_std = z_cent / z_scale
        X = np.hstack([X_cov_train, Z_std])
    else:
        z_mean = np.empty(0, dtype=float)
        z_scale = np.empty(0, dtype=float)
        X = X_cov_train

    y_mean = float(w @ y_train)
    y_cent = y_train - y_mean
    alpha_arr = np.asarray(alphas, dtype=float)

    if X.shape[1] == 0:
        return MultiAlphaRidgeResult(
            alphas=alpha_arr,
            intercepts=np.full(alpha_arr.size, y_mean, dtype=float),
            coef_cov=np.zeros((alpha_arr.size, 0), dtype=float),
            coef_spec=np.zeros((alpha_arr.size, 0), dtype=float),
            z_mean=z_mean,
            z_scale=z_scale,
        )

    x_mean = w @ X
    X_cent = X - x_mean
    Xw_t_w = X_cent.T * w[None, :]
    gram_x = _symmetrize(Xw_t_w @ X_cent)
    xt_wy = Xw_t_w @ y_cent

    eigvals, eigvecs = np.linalg.eigh(gram_x)
    eigvals = np.maximum(eigvals, 0.0)
    v_proj = eigvecs.T @ xt_wy
    # Shrinkage in eigenbasis for all alphas simultaneously: shape (n_alphas, p_total).
    denom = eigvals[None, :] + alpha_arr[:, None]
    weights_eig = np.where(denom > 1e-15, v_proj[None, :] / denom, 0.0)
    betas = weights_eig @ eigvecs.T
    intercepts = y_mean - betas @ x_mean

    return MultiAlphaRidgeResult(
        alphas=alpha_arr,
        intercepts=intercepts,
        coef_cov=betas[:, :p_cov],
        coef_spec=betas[:, p_cov:],
        z_mean=z_mean,
        z_scale=z_scale,
    )


def predict_ridge_grid(
    fit: MultiAlphaRidgeResult,
    X_cov: np.ndarray,
    Z_spec: np.ndarray | None,
    *,
    alpha_idx: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return full predictions and spectral contributions s = beta_Z^T ((z - mean)/scale)."""
    if Z_spec is not None and Z_spec.shape[1] > 0:
        Z_std = (Z_spec - fit.z_mean) / fit.z_scale
    else:
        Z_std = np.zeros((X_cov.shape[0], 0), dtype=float)

    if alpha_idx is not None:
        b0 = float(fit.intercepts[alpha_idx])
        b_cov = fit.coef_cov[alpha_idx]
        b_spec = fit.coef_spec[alpha_idx]
        spec_contrib = Z_std @ b_spec if Z_std.shape[1] > 0 else np.zeros(X_cov.shape[0], dtype=float)
        cov_contrib = X_cov @ b_cov if X_cov.shape[1] > 0 else np.zeros(X_cov.shape[0], dtype=float)
        return b0 + cov_contrib + spec_contrib, spec_contrib

    spec_mat = Z_std @ fit.coef_spec.T if Z_std.shape[1] > 0 else np.zeros((X_cov.shape[0], fit.alphas.size))
    cov_mat = X_cov @ fit.coef_cov.T if X_cov.shape[1] > 0 else np.zeros((X_cov.shape[0], fit.alphas.size))
    return fit.intercepts[None, :] + cov_mat + spec_mat, spec_mat


@dataclass(slots=True)
class WeightedPLSResult:
    """Multi-component 1D weighted PLS regression model."""

    max_components: int
    x_mean: np.ndarray
    y_mean: float
    p_cov: int
    coefs_by_comp: np.ndarray  # shape (max_components, p_total)

    @classmethod
    def fit(
        cls,
        X_cov_train: np.ndarray,
        X_spec_train: np.ndarray,
        y_train: np.ndarray,
        participant_weights: np.ndarray,
        max_components: int,
    ) -> WeightedPLSResult:
        w = _normalize_weights(participant_weights, context="WeightedPLSResult.fit")
        X = np.hstack([X_cov_train, X_spec_train])
        p_cov = X_cov_train.shape[1]

        x_mean = w @ X
        y_mean = float(w @ y_train)
        Xk = X - x_mean
        yk = y_train - y_mean

        n_active = int(np.sum(w > 0.0))
        n_comp = max(1, min(int(max_components), n_active - 2, X.shape[1]))
        W_mat = np.empty((X.shape[1], n_comp), dtype=float)
        P_mat = np.empty((X.shape[1], n_comp), dtype=float)
        q_vec = np.empty(n_comp, dtype=float)
        ptw = np.empty((n_comp, n_comp), dtype=float)
        coefs = np.zeros((n_comp, X.shape[1]), dtype=float)

        for comp_idx in range(n_comp):
            wk = Xk.T @ (w * yk)
            w_norm = float(linalg.norm(wk))
            if not np.isfinite(w_norm) or w_norm <= 1e-12:
                if comp_idx > 0:
                    coefs[comp_idx:] = coefs[comp_idx - 1]
                break
            wk = wk / w_norm
            tk = Xk @ wk
            w_tk = w * tk
            t_sq = float(w_tk @ tk)
            if not np.isfinite(t_sq) or t_sq <= 1e-12:
                if comp_idx > 0:
                    coefs[comp_idx:] = coefs[comp_idx - 1]
                break
            pk = (Xk.T @ w_tk) / t_sq
            qk = float((w_tk @ yk) / t_sq)

            Xk -= np.outer(tk, pk)
            yk -= tk * qk

            c_end = comp_idx + 1
            W_mat[:, comp_idx] = wk
            P_mat[:, comp_idx] = pk
            q_vec[comp_idx] = qk

            # R = W (P^T W)^{-1}, beta = R q; update only the new row and column of P^T W.
            ptw[comp_idx, :c_end] = pk @ W_mat[:, :c_end]
            if comp_idx > 0:
                ptw[:comp_idx, comp_idx] = P_mat[:, :comp_idx].T @ wk
            r_q = linalg.solve(ptw[:c_end, :c_end], q_vec[:c_end])
            coefs[comp_idx] = W_mat[:, :c_end] @ r_q

        return cls(
            max_components=n_comp,
            x_mean=x_mean,
            y_mean=y_mean,
            p_cov=p_cov,
            coefs_by_comp=coefs,
        )

    def predict(
        self,
        X_cov: np.ndarray,
        X_spec: np.ndarray,
        *,
        n_components: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        x_mean_cov = self.x_mean[: self.p_cov]
        x_mean_spec = self.x_mean[self.p_cov :]

        if n_components is not None:
            idx = min(max(int(n_components), 1), self.max_components) - 1
            beta = self.coefs_by_comp[idx]
            b_cov = beta[: self.p_cov]
            b_spec = beta[self.p_cov :]
            cov_part = (X_cov @ b_cov - float(x_mean_cov @ b_cov)) if self.p_cov > 0 else 0.0
            spec_part = X_spec @ b_spec - float(x_mean_spec @ b_spec)
            return self.y_mean + cov_part + spec_part, spec_part

        B_cov = self.coefs_by_comp[:, : self.p_cov].T
        B_spec = self.coefs_by_comp[:, self.p_cov :].T
        cov_part = (X_cov @ B_cov - x_mean_cov @ B_cov) if self.p_cov > 0 else 0.0
        spec_part = X_spec @ B_spec - x_mean_spec @ B_spec
        return self.y_mean + cov_part + spec_part, spec_part
