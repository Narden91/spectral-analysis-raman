"""Fold-local preprocessors and weighted linear models: covariate/spectral scaling, paired moments, ridge, PLS."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Sequence
import numpy as np
import pandas as pd
from scipy import linalg

from src.analysis.paired_lipid.cohort import (
    CohortDataset,
    PRE_EVENT_RISK_FACTORS,
    PairedLipidConfig,
)


@dataclass
class CovariatePreprocessor:
    """Training-only covariate imputation, missingness indicator, and standardization."""

    feature_names: list[str]
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    dropped_columns: list[str]
    include_missing_indicators_for: list[str]
    preset: str

    @classmethod
    def fit(
        cls,
        cov_frame: pd.DataFrame,
        train_indices: np.ndarray,
        participant_weights: np.ndarray,
        *,
        preset: str = "primary",
        raw_cache: dict[str, Any] | None = None,
    ) -> "CovariatePreprocessor":
        raw_mat, raw_cols = cls._get_raw_matrix(cov_frame, preset=preset, raw_cache=raw_cache)
        sub_mat = raw_mat[train_indices]
        w = np.asarray(participant_weights, dtype=float)
        w = w / np.sum(w)

        kept_cols: list[str] = []
        dropped_cols: list[str] = []
        miss_indicator_cols: list[str] = []
        medians_list: list[float] = []

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
            medians_list.append(med)
            if not np.all(finite):
                miss_indicator_cols.append(col)

        if not kept_cols:
            return cls(
                feature_names=[],
                medians=np.empty(0, dtype=float),
                means=np.empty(0, dtype=float),
                scales=np.empty(0, dtype=float),
                dropped_columns=dropped_cols,
                include_missing_indicators_for=[],
                preset=preset,
            )

        col_to_idx = {c: i for i, c in enumerate(raw_cols)}
        medians = np.asarray(medians_list, dtype=float)
        design_list: list[np.ndarray] = []
        names: list[str] = []
        for idx, col in enumerate(kept_cols):
            vals = sub_mat[:, col_to_idx[col]]
            finite = np.isfinite(vals)
            design_list.append(np.where(finite, vals, medians[idx]))
            names.append(col)
        for col in miss_indicator_cols:
            vals = sub_mat[:, col_to_idx[col]]
            ind = (~np.isfinite(vals)).astype(float)
            if np.ptp(ind) > 1e-12:
                design_list.append(ind)
                names.append(f"{col}__missing")

        X = np.column_stack(design_list)
        means = np.sum(w[:, None] * X, axis=0)
        centered = X - means
        variances = np.sum(w[:, None] * (centered**2), axis=0)
        scales = np.sqrt(np.maximum(variances, 1e-12))
        return cls(
            feature_names=names,
            medians=medians,
            means=means,
            scales=scales,
            dropped_columns=dropped_cols,
            include_missing_indicators_for=miss_indicator_cols,
            preset=preset,
        )

    def transform(
        self,
        cov_frame: pd.DataFrame,
        indices: np.ndarray,
        *,
        raw_cache: dict[str, Any] | None = None,
    ) -> np.ndarray:
        if not self.feature_names:
            return np.zeros((len(indices), 0), dtype=float)
        raw_mat, raw_cols = self._get_raw_matrix(cov_frame, preset=self.preset, raw_cache=raw_cache)
        sub_mat = raw_mat[indices]
        col_to_idx = {c: i for i, c in enumerate(raw_cols)}
        base_cols = [c for c in self.feature_names if not c.endswith("__missing")]
        cols_out: list[np.ndarray] = []
        for idx, col in enumerate(base_cols):
            vals = (
                sub_mat[:, col_to_idx[col]]
                if col in col_to_idx
                else np.full(len(indices), np.nan, dtype=float)
            )
            finite = np.isfinite(vals)
            cols_out.append(np.where(finite, vals, self.medians[idx]))
        for col in self.include_missing_indicators_for:
            if f"{col}__missing" in self.feature_names:
                vals = (
                    sub_mat[:, col_to_idx[col]]
                    if col in col_to_idx
                    else np.full(len(indices), np.nan, dtype=float)
                )
                cols_out.append((~np.isfinite(vals)).astype(float))
        X = np.column_stack(cols_out)
        return (X - self.means) / self.scales

    @classmethod
    def _get_raw_matrix(
        cls,
        cov_frame: pd.DataFrame,
        *,
        preset: str,
        raw_cache: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        cache_key = f"raw_cov_matrix__{preset}"
        if raw_cache is not None and cache_key in raw_cache:
            return raw_cache[cache_key]
        raw_df = cls._build_raw_design(cov_frame, preset=preset)
        res = (raw_df.to_numpy(dtype=float), list(raw_df.columns))
        if raw_cache is not None:
            raw_cache[cache_key] = res
        return res

    @staticmethod
    def _build_raw_design(df: pd.DataFrame, *, preset: str) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        out["age_years"] = pd.to_numeric(df.get("age_years"), errors="coerce")
        if preset == "quadratic_age_and_label":
            out["age_years_sq"] = out["age_years"] ** 2

        sex_s = df.get("sex", pd.Series(np.nan, index=df.index)).astype(str).str.lower()
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
                    sm = df[rf].astype(str).str.lower()
                    out["smoking_current"] = np.where(
                        sm == "current", 1.0, np.where(sm.isin(["never", "former"]), 0.0, np.nan)
                    )
                    out["smoking_former"] = np.where(
                        sm == "former", 1.0, np.where(sm.isin(["never", "current"]), 0.0, np.nan)
                    )
                else:
                    out[rf] = pd.to_numeric(df[rf], errors="coerce")
        return out


@dataclass
class WeightedSpectralPreprocessor:
    """Participant-weighted MFA block scaling and truncated SVD for training divisions."""

    mean_1300: np.ndarray
    mean_2850: np.ndarray
    scale_1300: float
    scale_2850: float
    eig1_1300: float
    eig1_2850: float
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
    ) -> "WeightedSpectralPreprocessor":
        train_p = np.asarray(train_participant_indices, dtype=int)
        mult = np.asarray(participant_multiplicities, dtype=float)
        if np.any(mult <= 0):
            keep = mult > 0
            train_p = train_p[keep]
            mult = mult[keep]

        n_eff = float(np.sum(mult))
        p_to_mult = np.zeros(dataset.n_participants, dtype=float)
        p_to_mult[train_p] = mult

        sample_p = dataset.sample_participant_idx
        in_train = p_to_mult[sample_p] > 0
        s_indices = np.flatnonzero(in_train)
        s_p = sample_p[s_indices]

        m_i = dataset.participant_sample_counts[s_p].astype(float)
        w_samples = p_to_mult[s_p] / (n_eff * m_i)
        w_sum = float(np.sum(w_samples))
        if not np.isfinite(w_sum) or w_sum <= 0:
            raise ValueError("Invalid sample weights in spectral preprocessing.")
        w_samples = w_samples / w_sum

        X1 = dataset.sample_1300[s_indices]
        X2 = dataset.sample_2850[s_indices]

        mean1 = np.sum(w_samples[:, None] * X1, axis=0)
        mean2 = np.sum(w_samples[:, None] * X2, axis=0)
        C1 = X1 - mean1
        C2 = X2 - mean2

        # Effective total sample count for sample covariance denominator
        m_total = float(np.sum(p_to_mult[s_p]))
        denom_factor = m_total / max(m_total - 1.0, 1.0)
        sqrt_w = np.sqrt(w_samples * denom_factor)[:, None]

        eig1_1300 = cls._leading_eigenvalue(sqrt_w * C1, cfg.eigenvalue_tolerance)
        eig1_2850 = cls._leading_eigenvalue(sqrt_w * C2, cfg.eigenvalue_tolerance)
        scale1 = float(1.0 / np.sqrt(eig1_1300))
        scale2 = float(1.0 / np.sqrt(eig1_2850))

        # Method-of-snapshots Gram eigendecomposition: (n_samples x n_samples) instead of (n_samples x p_total) SVD
        W1 = (sqrt_w * scale1) * C1
        W2 = (sqrt_w * scale2) * C2
        gram = (W1 @ W1.T) + (W2 @ W2.T)
        gram = 0.5 * (gram + gram.T)
        eigvals, eigvecs = np.linalg.eigh(gram)
        order = np.argsort(eigvals)[::-1]
        eigvals = np.maximum(eigvals[order], 0.0)
        eigvecs = eigvecs[:, order]
        s_vals = np.sqrt(eigvals)

        max_dim = max(W1.shape[0], W1.shape[1] + W2.shape[1])
        tol = max_dim * np.finfo(float).eps * max(float(s_vals[0]) if s_vals.size else 1.0, 1.0)
        rank = int(np.sum(s_vals > max(tol, cfg.rank_tolerance)))

        max_k = min(max(cfg.pca_dimensions), rank, s_vals.size)
        s_kept = s_vals[:max_k].copy()
        u_kept = eigvecs[:, :max_k]
        inv_s = 1.0 / s_kept[:, None]
        comps_1300 = inv_s * (u_kept.T @ W1)
        comps_2850 = inv_s * (u_kept.T @ W2)
        comps = np.hstack([comps_1300, comps_2850])

        # Deterministic sign rule: largest-magnitude loading element in each component is positive
        max_abs_idx = np.argmax(np.abs(comps), axis=1)
        signs = np.sign(comps[np.arange(max_k), max_abs_idx])
        signs[signs == 0] = 1.0
        comps *= signs[:, None]

        paired_in_train = np.intersect1d(train_p, dataset.paired_participant_indices, assume_unique=False)
        return cls(
            mean_1300=mean1,
            mean_2850=mean2,
            scale_1300=scale1,
            scale_2850=scale2,
            eig1_1300=eig1_1300,
            eig1_2850=eig1_2850,
            pca_components=comps,
            pca_singular_values=s_kept,
            numerical_rank=rank,
            n_distinct_paired_train=int(paired_in_train.size),
        )

    @staticmethod
    def _leading_eigenvalue(weighted_matrix: np.ndarray, tol: float) -> float:
        if weighted_matrix.shape[0] < 2:
            raise ValueError("Block requires at least two weighted observations.")
        # Gram matrix is n_samples x n_samples (typically <= 200), much smaller than p_b x p_b
        gram = weighted_matrix @ weighted_matrix.T
        gram = 0.5 * (gram + gram.T)
        eigvals = np.linalg.eigvalsh(gram)
        top_eig = float(eigvals[-1])
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
        """Project spectra into top-k weighted MFA-PCA coordinates without intermediate hstack."""
        comps = self.pca_components if k is None else self.pca_components[:k]
        p1 = self.mean_1300.size
        proj1 = (X1 - self.mean_1300) @ (self.scale_1300 * comps[:, :p1].T)
        proj2 = (X2 - self.mean_2850) @ (self.scale_2850 * comps[:, p1:].T)
        return proj1 + proj2

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


@dataclass
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
    ) -> "PairedMoments":
        p_to_mult = np.zeros(dataset.n_participants, dtype=float)
        p_to_mult[train_participant_indices] = participant_multiplicities

        paired_p = dataset.paired_participant_indices
        mult_p = p_to_mult[paired_p]
        active = mult_p > 0
        if int(np.sum(active)) < 2:
            raise ValueError("At least two distinct paired training participants are required.")

        mult_active = mult_p[active]
        idx_a = dataset.pair_sample_a_indices[active]
        idx_b = dataset.pair_sample_b_indices[active]

        x_a = sample_coords[idx_a]
        x_b = sample_coords[idx_b]
        diffs = x_a - x_b
        means = 0.5 * (x_a + x_b)

        n_p = float(np.sum(mult_active))
        if n_p <= 1.0:
            raise ValueError("Total paired multiplicity must exceed 1.")

        w_diff = np.sqrt(mult_active / (2.0 * n_p))[:, None] * diffs
        W = w_diff.T @ w_diff
        W = 0.5 * (W + W.T)

        u_bar = np.sum(mult_active[:, None] * means, axis=0) / n_p
        u_centered = means - u_bar
        w_u = np.sqrt(mult_active / (n_p - 1.0))[:, None] * u_centered
        T = w_u.T @ w_u
        T = 0.5 * (T + T.T)

        mean_shift = np.sum(mult_active[:, None] * diffs, axis=0) / n_p
        return cls(
            W_max=W,
            T_max=T,
            mean_ab_shift=mean_shift,
            n_paired_multiplicity=n_p,
            n_paired_distinct=int(np.sum(active)),
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

        denom = W_k + float(lam) * tau * np.eye(k, dtype=float)
        denom = 0.5 * (denom + denom.T)

        eigvals, eigvecs = linalg.eigh(T_k, denom, check_finite=True)
        order = np.argsort(eigvals)[::-1]
        eigvals = np.asarray(eigvals[order], dtype=float)
        eigvecs = np.asarray(eigvecs[:, order], dtype=float)

        # Normalize in the regularized denominator metric and enforce deterministic signs
        metric_norms = np.sqrt(np.maximum(np.sum(eigvecs * (denom @ eigvecs), axis=0), 1e-15))
        eigvecs = eigvecs / metric_norms[None, :]
        max_abs_idx = np.argmax(np.abs(eigvecs), axis=0)
        signs = np.sign(eigvecs[max_abs_idx, np.arange(k)])
        signs[signs == 0] = 1.0
        eigvecs *= signs[None, :]
        return eigvecs, eigvals, tau


@dataclass
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
    """Fit weighted ridge regression across all candidate alphas simultaneously via thin SVD.

    Solves ``min_{b0, b} (1 / N_eff) sum_i w_i (y_i - b0 - x_i^T b)^2 + alpha ||b||^2``
    with an unpenalized intercept.
    """
    w = np.asarray(participant_weights, dtype=float)
    w = w / np.sum(w)
    p_cov = X_cov_train.shape[1]

    if Z_spec_train is not None and Z_spec_train.shape[1] > 0:
        z_mean = np.sum(w[:, None] * Z_spec_train, axis=0)
        z_cent = Z_spec_train - z_mean
        z_var = np.sum(w[:, None] * (z_cent**2), axis=0)
        z_scale = np.sqrt(np.maximum(z_var, 1e-12))
        Z_std = z_cent / z_scale
        X = np.hstack([X_cov_train, Z_std])
    else:
        z_mean = np.empty(0, dtype=float)
        z_scale = np.empty(0, dtype=float)
        X = X_cov_train

    y_mean = float(np.sum(w * y_train))
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

    x_mean = np.sum(w[:, None] * X, axis=0)
    X_cent = X - x_mean
    Xw_t_w = X_cent.T * w[None, :]
    gram_x = Xw_t_w @ X_cent
    gram_x = 0.5 * (gram_x + gram_x.T)
    xt_wy = Xw_t_w @ y_cent

    eigvals, eigvecs = np.linalg.eigh(gram_x)
    eigvals = np.maximum(eigvals, 0.0)
    v_proj = eigvecs.T @ xt_wy
    # Shrinkage in eigenbasis for all alphas simultaneously: shape (n_alphas, p_total)
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


@dataclass
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
    ) -> "WeightedPLSResult":
        w = np.asarray(participant_weights, dtype=float)
        w = w / np.sum(w)
        X = np.hstack([X_cov_train, X_spec_train])
        p_cov = X_cov_train.shape[1]

        x_mean = np.sum(w[:, None] * X, axis=0)
        y_mean = float(np.sum(w * y_train))
        Xk = X - x_mean
        yk = y_train - y_mean

        n_comp = max(1, min(int(max_components), X.shape[0] - 2, X.shape[1]))
        W_list: list[np.ndarray] = []
        P_list: list[np.ndarray] = []
        q_list: list[float] = []
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
            t_sq = float(np.sum(w * (tk**2)))
            if not np.isfinite(t_sq) or t_sq <= 1e-12:
                if comp_idx > 0:
                    coefs[comp_idx:] = coefs[comp_idx - 1]
                break
            pk = (Xk.T @ (w * tk)) / t_sq
            qk = float(np.sum(w * tk * yk) / t_sq)

            Xk = Xk - np.outer(tk, pk)
            yk = yk - tk * qk

            W_list.append(wk)
            P_list.append(pk)
            q_list.append(qk)

            W_mat = np.column_stack(W_list)
            P_mat = np.column_stack(P_list)
            q_vec = np.asarray(q_list, dtype=float)
            # R = W (P^T W)^{-1}, beta = R q
            ptw = P_mat.T @ W_mat
            r_q = linalg.solve(ptw, q_vec)
            coefs[comp_idx] = W_mat @ r_q

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
        X_cent = np.hstack([X_cov, X_spec]) - self.x_mean
        X_cov_cent = X_cent[:, : self.p_cov]
        X_spec_cent = X_cent[:, self.p_cov :]

        if n_components is not None:
            idx = min(max(int(n_components), 1), self.max_components) - 1
            beta = self.coefs_by_comp[idx]
            cov_part = X_cov_cent @ beta[: self.p_cov] if self.p_cov > 0 else 0.0
            spec_part = X_spec_cent @ beta[self.p_cov :]
            return self.y_mean + cov_part + spec_part, spec_part

        cov_part = X_cov_cent @ self.coefs_by_comp[:, : self.p_cov].T if self.p_cov > 0 else 0.0
        spec_part = X_spec_cent @ self.coefs_by_comp[:, self.p_cov :].T
        return self.y_mean + cov_part + spec_part, spec_part
