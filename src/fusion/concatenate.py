from __future__ import annotations

import numpy as np


def concatenate_regions(
    spectra_1300: np.ndarray,
    spectra_2850: np.ndarray,
    valid_mask: np.ndarray | None = None,
    weight_by_channels: bool = True,
) -> np.ndarray:
    """Concatenate fingerprint and CH-stretch spectra with optional channel weighting."""
    if spectra_1300.ndim != 2 or spectra_2850.ndim != 2:
        raise ValueError("Input spectra must be 2D arrays")
    if spectra_1300.shape[0] != spectra_2850.shape[0]:
        raise ValueError("Region spectra must have the same number of pixels")

    left = np.asarray(spectra_1300, dtype=float)
    right = np.asarray(spectra_2850, dtype=float)

    if valid_mask is not None:
        if valid_mask.shape[0] != left.shape[0]:
            raise ValueError("valid_mask length must match number of pixels")
        keep = np.asarray(valid_mask, dtype=bool)
        left = left[keep]
        right = right[keep]

    if left.size == 0:
        return np.empty((0, spectra_1300.shape[1] + spectra_2850.shape[1]), dtype=float)

    if weight_by_channels:
        left = left / np.sqrt(left.shape[1])
        right = right / np.sqrt(right.shape[1])

    return np.hstack([left, right])
