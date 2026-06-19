from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src import config
from src.constants import CANONICAL_FP_CHANNELS


def _default_split_index() -> int:
    """Read split_index from config, falling back to the canonical fingerprint width."""
    return int(config.get("pca", {}).get("split_index", CANONICAL_FP_CHANNELS))


def _split_regions(vectors: np.ndarray, split_index: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    if split_index is None:
        split_index = _default_split_index()
    if vectors.ndim != 2:
        raise ValueError("vectors must be 2D")
    if split_index <= 0 or split_index >= vectors.shape[1]:
        raise ValueError("split_index must be between 1 and n_features-1")
    return vectors[:, :split_index], vectors[:, split_index:]


def _fit_pca_block(
    train_block: np.ndarray,
    variance_threshold: float,
    random_state: int,
) -> dict:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_block)

    if train_scaled.shape[0] < 2:
        return {
            "kind": "degenerate",
            "scaler": scaler,
            "scores": np.zeros((train_scaled.shape[0], 1), dtype=float),
            "n_components": 1,
            "explained_variance_ratio": np.array([0.0], dtype=float),
        }

    # Single-pass: fit full PCA once, then truncate to the required n_components
    # without refitting, since transform only uses components_.
    pca_full = PCA(random_state=random_state)
    pca_full.fit(train_scaled)
    cumulative = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumulative, variance_threshold) + 1)
    n_components = max(1, min(n_components, train_scaled.shape[1]))

    # Truncate the fitted decomposition in-place (avoids a second solver pass).
    pca_full.components_ = pca_full.components_[:n_components]
    pca_full.explained_variance_ = pca_full.explained_variance_[:n_components]
    pca_full.explained_variance_ratio_ = pca_full.explained_variance_ratio_[:n_components]
    scores = pca_full.transform(train_scaled)

    return {
        "kind": "pca",
        "scaler": scaler,
        "pca": pca_full,
        "scores": scores,
        "n_components": n_components,
        "explained_variance_ratio": pca_full.explained_variance_ratio_,
    }


def fit_region_pca(
    train_vectors: np.ndarray,
    variance_threshold: float | None = None,
    split_index: int | None = None,
    random_state: int = 42,
) -> dict:
    """Fit per-region PCA on training vectors and return training scores and models."""
    if train_vectors.size == 0:
        raise ValueError("train_vectors is empty")

    if variance_threshold is None:
        variance_threshold = float(config.get("pca", {}).get("variance_threshold", 0.95))
    if split_index is None:
        split_index = _default_split_index()

    region_1300, region_2850 = _split_regions(train_vectors, split_index)
    model_1300 = _fit_pca_block(region_1300, variance_threshold, random_state)
    model_2850 = _fit_pca_block(region_2850, variance_threshold, random_state)

    train_scores = np.hstack([model_1300["scores"], model_2850["scores"]])
    return {
        "split_index": split_index,
        "variance_threshold": variance_threshold,
        "region_1300": model_1300,
        "region_2850": model_2850,
        "train_scores": train_scores,
    }


def transform_region_pca(model: dict, vectors: np.ndarray) -> np.ndarray:
    """Transform vectors with an already-fitted per-region PCA model."""
    split_index = int(model["split_index"])
    block_1300, block_2850 = _split_regions(vectors, split_index)

    m1300 = model["region_1300"]
    m2850 = model["region_2850"]

    s1300 = _transform_pca_block(m1300, block_1300)
    s2850 = _transform_pca_block(m2850, block_2850)
    return np.hstack([s1300, s2850])


def _transform_pca_block(model: dict, block: np.ndarray) -> np.ndarray:
    scaled = model["scaler"].transform(block)
    if model.get("kind") == "degenerate":
        return np.zeros((scaled.shape[0], int(model["n_components"])), dtype=float)
    return model["pca"].transform(scaled)
