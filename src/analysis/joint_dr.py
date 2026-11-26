"""Joint dimensionality-reduction helpers for whole-spectrum analyses."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import eigh, eigvalsh
from sklearn.decomposition import PCA

from src.analysis.statistics import run_pca
from src.analysis.whole_spectra import _validate_split_index


def _validate_spectra(spectra: np.ndarray) -> np.ndarray:
    spectra = np.asarray(spectra, dtype=float)
    if spectra.ndim != 2:
        raise ValueError("spectra must be a 2D array")
    return spectra


def _first_eigenvalue(block: np.ndarray) -> float:
    if block.shape[0] < 2:
        raise ValueError("MFA block requires at least two observations")
    n_s, n_p = block.shape
    if n_p > max(n_s, 16) and bool(np.isfinite(block).all()):
        means = np.mean(block, axis=0, keepdims=True)
        centered = block - means
        gram = centered @ centered.T
        gram = 0.5 * (gram + gram.T)
        tr_g = float(np.trace(gram))
        variance_scale = tr_g / (n_s - 1)
        tolerance = np.finfo(float).eps * max(variance_scale, 1.0) * max(centered.shape)
        if not np.isfinite(variance_scale) or variance_scale <= tolerance:
            raise ValueError("MFA block is numerically degenerate")
        mu_sq = float(np.dot(means.ravel(), means.ravel()))
        if np.isfinite(mu_sq) and mu_sq <= 1e2 * tr_g:
            eigenvalue = float(
                eigvalsh(
                    gram,
                    subset_by_index=[n_s - 1, n_s - 1],
                    driver="evr",
                    check_finite=False,
                )[-1]
            ) / (n_s - 1)
            if not np.isfinite(eigenvalue) or eigenvalue <= tolerance:
                raise ValueError("MFA first eigenvalue is numerically degenerate")
            return eigenvalue
    centered = block - np.nanmean(block, axis=0, keepdims=True)
    variance_scale = float(np.nansum(np.nanvar(centered, axis=0, ddof=1)))
    tolerance = np.finfo(float).eps * max(variance_scale, 1.0) * max(centered.shape)
    if not np.isfinite(variance_scale) or variance_scale <= tolerance:
        raise ValueError("MFA block is numerically degenerate")
    first_component = PCA(n_components=1, svd_solver="full").fit(centered)
    eigenvalue = float(first_component.explained_variance_[0])
    if not np.isfinite(eigenvalue) or eigenvalue <= tolerance:
        raise ValueError("MFA first eigenvalue is numerically degenerate")
    return eigenvalue


def joint_pca_mfa(
    spectra: np.ndarray,
    split_index: int,
    n_components: int,
) -> dict[str, object]:
    """
    Multiple Factor Analysis-style joint PCA.

    Each block is centered, then scaled by 1/sqrt(first eigenvalue of that block).
    Concat-PCA is then run on the scaled blocks.
    """
    spectra = _validate_spectra(spectra)
    split_index = _validate_split_index(split_index, spectra.shape[1])

    fingerprint = spectra[:, :split_index]
    ch_stretch = spectra[:, split_index:]
    n_s, n_p = spectra.shape
    p_fp = split_index
    p_ch = n_p - split_index

    result: dict[str, object] | None = None
    fingerprint_scaled: np.ndarray
    ch_scaled: np.ndarray
    fingerprint_eigenvalue: float
    ch_eigenvalue: float

    used_fast_mfa = False
    if n_s >= 2 and p_fp > max(n_s, 16) and p_ch > max(n_s, 16) and bool(np.isfinite(spectra).all()):
        fp_means = np.mean(fingerprint, axis=0, keepdims=True)
        ch_means = np.mean(ch_stretch, axis=0, keepdims=True)
        fp_c = fingerprint - fp_means
        ch_c = ch_stretch - ch_means

        g1 = fp_c @ fp_c.T
        g1 = 0.5 * (g1 + g1.T)
        tr_g1 = float(np.trace(g1))
        var_scale_1 = tr_g1 / (n_s - 1)
        tol_1 = np.finfo(float).eps * max(var_scale_1, 1.0) * max(n_s, p_fp)
        if not np.isfinite(var_scale_1) or var_scale_1 <= tol_1:
            raise ValueError("MFA block is numerically degenerate")

        g2 = ch_c @ ch_c.T
        g2 = 0.5 * (g2 + g2.T)
        tr_g2 = float(np.trace(g2))
        var_scale_2 = tr_g2 / (n_s - 1)
        tol_2 = np.finfo(float).eps * max(var_scale_2, 1.0) * max(n_s, p_ch)
        if not np.isfinite(var_scale_2) or var_scale_2 <= tol_2:
            raise ValueError("MFA block is numerically degenerate")

        mu1_sq = float(np.dot(fp_means.ravel(), fp_means.ravel()))
        mu2_sq = float(np.dot(ch_means.ravel(), ch_means.ravel()))
        if (
            np.isfinite(mu1_sq)
            and np.isfinite(mu2_sq)
            and mu1_sq <= 1e2 * tr_g1
            and mu2_sq <= 1e2 * tr_g2
        ):
            ev1 = float(
                eigvalsh(
                    g1,
                    subset_by_index=[n_s - 1, n_s - 1],
                    driver="evr",
                    check_finite=False,
                )[-1]
            ) / (n_s - 1)
            if not np.isfinite(ev1) or ev1 <= tol_1:
                raise ValueError("MFA first eigenvalue is numerically degenerate")
            ev2 = float(
                eigvalsh(
                    g2,
                    subset_by_index=[n_s - 1, n_s - 1],
                    driver="evr",
                    check_finite=False,
                )[-1]
            ) / (n_s - 1)
            if not np.isfinite(ev2) or ev2 <= tol_2:
                raise ValueError("MFA first eigenvalue is numerically degenerate")

            fingerprint_eigenvalue = ev1
            ch_eigenvalue = ev2
            fingerprint_scaled = fp_c / np.sqrt(ev1)
            ch_scaled = ch_c / np.sqrt(ev2)
            used_fast_mfa = True

            k = min(int(n_components), n_s, n_p) if n_components is not None else 0
            if k >= 1:
                g_comb = (1.0 / ev1) * g1 + (1.0 / ev2) * g2
                g_comb = 0.5 * (g_comb + g_comb.T)
                tr_comb = float(np.trace(g_comb))
                if np.isfinite(tr_comb) and tr_comb > 0.0:
                    evals_asc, evecs_asc = eigh(
                        g_comb,
                        subset_by_index=[n_s - k, n_s - 1],
                        driver="evr",
                        check_finite=False,
                    )
                    eigvals = evals_asc[::-1]
                    u_kept = evecs_asc[:, ::-1]
                    eps = float(np.finfo(float).eps)
                    gram_cond = max(1e4 * max(n_s, n_p) * eps, 1e-24)
                    if (
                        np.all(np.isfinite(eigvals))
                        and np.all(eigvals > 0.0)
                        and eigvals[-1] > gram_cond * eigvals[0]
                    ):
                        s_kept = np.sqrt(eigvals)
                        loadings = np.hstack(
                            [
                                (u_kept.T @ fingerprint_scaled) / s_kept[:, np.newaxis],
                                (u_kept.T @ ch_scaled) / s_kept[:, np.newaxis],
                            ]
                        )
                        scores = u_kept * s_kept[np.newaxis, :]
                        signs = np.sign(
                            loadings[np.arange(k), np.argmax(np.abs(loadings), axis=1)]
                        )
                        signs[signs == 0] = 1.0
                        loadings *= signs[:, np.newaxis]
                        scores *= signs[np.newaxis, :]
                        explained_variance = eigvals / (n_s - 1)
                        explained_variance_ratio = eigvals / tr_comb
                        cumvar = np.cumsum(explained_variance_ratio)
                        pca = PCA(n_components=k, svd_solver="full")
                        pca.n_samples_ = n_s
                        pca.n_features_in_ = n_p
                        pca.n_components_ = k
                        pca.mean_ = np.zeros(n_p, dtype=float)
                        pca.singular_values_ = s_kept
                        pca.explained_variance_ = explained_variance
                        pca.explained_variance_ratio_ = explained_variance_ratio
                        pca.components_ = loadings
                        if k < min(n_s, n_p):
                            rem_var = max(
                                0.0,
                                (tr_comb / (n_s - 1)) - float(np.sum(explained_variance)),
                            )
                            pca.noise_variance_ = float(rem_var / (min(n_s, n_p) - k))
                        else:
                            pca.noise_variance_ = 0.0
                        result = {
                            "scores": scores,
                            "loadings": loadings,
                            "explained_variance_ratio": explained_variance_ratio,
                            "cumulative_variance": cumvar,
                            "pca_model": pca,
                            "scaler": None,
                            "pc1_variance_ratio": float(explained_variance_ratio[0]),
                        }

    if not used_fast_mfa:
        fingerprint_centered = fingerprint - np.nanmean(fingerprint, axis=0, keepdims=True)
        ch_centered = ch_stretch - np.nanmean(ch_stretch, axis=0, keepdims=True)
        fingerprint_eigenvalue = _first_eigenvalue(fingerprint)
        ch_eigenvalue = _first_eigenvalue(ch_stretch)
        fingerprint_scaled = fingerprint_centered / np.sqrt(fingerprint_eigenvalue)
        ch_scaled = ch_centered / np.sqrt(ch_eigenvalue)

    if result is None:
        combined = np.hstack([fingerprint_scaled, ch_scaled])
        result = run_pca(combined, n_components=n_components, scale=False)
    loadings = np.asarray(result["loadings"], dtype=float)
    block_contributions = pd.DataFrame(
        {
            "component": np.arange(1, loadings.shape[0] + 1),
            "fingerprint_contribution": np.sum(loadings[:, :split_index] ** 2, axis=1),
            "ch_stretch_contribution": np.sum(loadings[:, split_index:] ** 2, axis=1),
        }
    )
    result["block_contributions"] = block_contributions
    result["scaling_factors"] = {
        "fingerprint": float(1.0 / np.sqrt(fingerprint_eigenvalue)),
        "ch_stretch": float(1.0 / np.sqrt(ch_eigenvalue)),
    }
    result["block_eigenvalues"] = {
        "fingerprint": fingerprint_eigenvalue,
        "ch_stretch": ch_eigenvalue,
    }
    result["method"] = "mfa_first_eigenvalue_scaling_v1"
    result["split_index"] = split_index
    result["scaled_blocks"] = {
        "fingerprint": fingerprint_scaled,
        "ch_stretch": ch_scaled,
    }
    return result
