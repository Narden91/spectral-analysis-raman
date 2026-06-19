"""Fold-local spectral-channel scalers for ML raw-channel paths.

Engineered-feature paths in the ML pipeline extract features from physically
preprocessed subject-mean spectra and then apply a fold-local feature scaler
(StandardScaler) after feature selection.  The scalers here are used when the
configured ML path needs an additional per-wavenumber spectral scaling step
applied to subject vectors for raw-channel learned transforms such as PCA.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import StandardScaler


@dataclass
class RobustSpectralScaler:
    """Per-wavenumber robust scaler fit on subject-level aggregated spectra.

    Parameters
    ----------
    epsilon : float
        Floor for the scale denominator.  Channels whose MAD is below this
        value use the floor instead.
    """

    epsilon: float = 1.0e-9
    center_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    n_zero_scale_channels_: int = 0

    def fit(self, spectra: np.ndarray) -> "RobustSpectralScaler":
        """Fit median/MAD statistics on subject-level vectors.

        ``spectra`` should already be aggregated to one row per subject so that
        subjects with more pixels/maps do not overweight the statistics.
        """
        arr = np.asarray(spectra, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D spectra array; got ndim={arr.ndim}")

        self.center_ = np.median(arr, axis=0)
        mad = np.median(np.abs(arr - self.center_), axis=0)
        self.n_zero_scale_channels_ = int(np.sum(mad <= self.epsilon))
        self.scale_ = np.where(mad > self.epsilon, mad, self.epsilon)
        return self

    def transform(self, spectra: np.ndarray) -> np.ndarray:
        """Transform spectra using the fitted robust statistics."""
        if self.center_ is None or self.scale_ is None:
            raise ValueError("Scaler has not been fitted.")
        arr = np.asarray(spectra, dtype=float)
        return (arr - self.center_) / self.scale_

    def fit_transform(self, spectra: np.ndarray) -> np.ndarray:
        return self.fit(spectra).transform(spectra)


def fit_spectral_scaler(
    spectra: np.ndarray,
    scaler_type: str = "robust",
    epsilon: float = 1.0e-9,
) -> RobustSpectralScaler | StandardScaler:
    """Factory for a fold-local spectral scaler.

    Parameters
    ----------
    spectra : np.ndarray
        Subject-level spectral vectors (n_subjects, n_channels).
    scaler_type : str
        "robust" for median/MAD or "standard" for sklearn StandardScaler.
    epsilon : float
        Robust scale floor (ignored for standard scaler).

    Returns
    -------
    Fitted scaler.
    """
    scaler_type = str(scaler_type).strip().lower()
    if scaler_type == "robust":
        return RobustSpectralScaler(epsilon=epsilon).fit(spectra)
    if scaler_type == "standard":
        return StandardScaler().fit(spectra)
    raise ValueError(f"Unknown spectral scaler type: {scaler_type}")


def transform_spectral_scaler(
    scaler: RobustSpectralScaler | StandardScaler,
    spectra: np.ndarray,
) -> np.ndarray:
    """Transform spectra with a fitted scaler."""
    return scaler.transform(np.asarray(spectra, dtype=float))
