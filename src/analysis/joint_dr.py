"""Joint dimensionality-reduction helpers for whole-spectrum analyses."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.analysis.statistics import run_pca
from src.analysis.whole_spectra import _validate_split_index


def _validate_spectra(spectra: np.ndarray) -> np.ndarray:
    spectra = np.asarray(spectra, dtype=float)
    if spectra.ndim != 2:
        raise ValueError("spectra must be a 2D array")
    return spectra


def _first_eigenvalue(block: np.ndarray) -> float:
    centered = block - np.nanmean(block, axis=0, keepdims=True)
    if centered.shape[0] < 2:
        raise ValueError("MFA block requires at least two observations")
    variance_scale = float(np.nansum(np.nanvar(centered, axis=0, ddof=1)))
    tolerance = np.finfo(float).eps * max(variance_scale, 1.0) * max(centered.shape)
    if not np.isfinite(variance_scale) or variance_scale <= tolerance:
        raise ValueError("MFA block is numerically degenerate")
    first_component = PCA(n_components=1).fit(centered)
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
    fingerprint_centered = fingerprint - np.nanmean(fingerprint, axis=0, keepdims=True)
    ch_centered = ch_stretch - np.nanmean(ch_stretch, axis=0, keepdims=True)

    fingerprint_eigenvalue = _first_eigenvalue(fingerprint)
    ch_eigenvalue = _first_eigenvalue(ch_stretch)
    fingerprint_scaled = fingerprint_centered / np.sqrt(fingerprint_eigenvalue)
    ch_scaled = ch_centered / np.sqrt(ch_eigenvalue)
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
