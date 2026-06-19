"""Joint dimensionality-reduction helpers for whole-spectrum analyses."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA

from src.analysis.statistics import run_pca


def _validate_spectra(spectra: np.ndarray) -> np.ndarray:
    spectra = np.asarray(spectra, dtype=float)
    if spectra.ndim != 2:
        raise ValueError("spectra must be a 2D array")
    return spectra


def _validate_split_index(split_index: int, n_features: int) -> int:
    split_index = int(split_index)
    if split_index <= 0 or split_index >= n_features:
        raise ValueError("split_index must fall strictly inside the feature axis")
    return split_index


def _first_eigenvalue(block: np.ndarray) -> float:
    centered = block - np.nanmean(block, axis=0, keepdims=True)
    if centered.shape[0] < 2 or np.nanvar(centered) == 0:
        return 1.0
    first_component = PCA(n_components=1).fit(centered)
    eigenvalue = float(first_component.explained_variance_[0])
    if not np.isfinite(eigenvalue) or eigenvalue <= 0:
        return 1.0
    return max(eigenvalue, 1.0)


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
    result["scaled_blocks"] = {
        "fingerprint": fingerprint_scaled,
        "ch_stretch": ch_scaled,
    }
    return result


def cca_between_regions(
    fingerprint_scores: np.ndarray,
    ch_scores: np.ndarray,
    n_components: int = 3,
) -> dict[str, object]:
    """Canonical-correlation analysis between per-region PCA score matrices."""
    fingerprint_scores = _validate_spectra(fingerprint_scores)
    ch_scores = _validate_spectra(ch_scores)
    if fingerprint_scores.shape[0] != ch_scores.shape[0]:
        raise ValueError("fingerprint_scores and ch_scores must have the same number of rows")

    max_components = min(
        int(n_components),
        fingerprint_scores.shape[1],
        ch_scores.shape[1],
        fingerprint_scores.shape[0] - 1,
    )
    if max_components < 1:
        raise ValueError("CCA requires at least two samples and one component")

    cca = CCA(n_components=max_components, max_iter=1000)
    fingerprint_c, ch_c = cca.fit_transform(fingerprint_scores, ch_scores)

    correlations = []
    for component_index in range(max_components):
        left = fingerprint_c[:, component_index]
        right = ch_c[:, component_index]
        if np.std(left) == 0 or np.std(right) == 0:
            correlations.append(np.nan)
        else:
            correlations.append(float(np.corrcoef(left, right)[0, 1]))

    return {
        "fingerprint_scores": fingerprint_c,
        "ch_scores": ch_c,
        "correlations": np.asarray(correlations, dtype=float),
        "x_weights": cca.x_weights_,
        "y_weights": cca.y_weights_,
        "model": cca,
    }


def regional_pca_scores(
    spectra: np.ndarray,
    split_index: int,
    n_components: int,
) -> tuple[dict[str, object], dict[str, object]]:
    """Run PCA separately on fingerprint and CH-stretch blocks."""
    spectra = _validate_spectra(spectra)
    split_index = _validate_split_index(split_index, spectra.shape[1])
    fingerprint = spectra[:, :split_index]
    ch_stretch = spectra[:, split_index:]
    return run_pca(fingerprint, n_components=n_components, scale=False), run_pca(
        ch_stretch,
        n_components=n_components,
        scale=False,
    )
