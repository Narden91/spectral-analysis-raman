"""Phase M: local peak-shape analysis — centroid shift, FWHM, asymmetry.

All features are subject-level (one row per subject), read from the
subject Biological-PCA feature matrix (block columns ``1300__*`` /
``2850__*``). Outputs land in ``06_spectral_interpretability/peak_shape/``;
a summary copy goes to ``05_confirmatory_panel/panel_peak_shape_features.csv``.

Shifts below the instrument calibration floor (``calibration_floor_cm``) are
flagged as ``not_interpretable`` rather than silently discarded.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning, curve_fit

from src import config, get_confirmatory_panel_dir, get_spectral_interpretability_dir
from src.analysis._tier2_utils import compute_spearman_associations
from src.analysis.band_features import block_columns
from src.analysis.clinical_feature_associations import (
    filter_disease_subjects,
    load_subject_feature_matrix,
)
from src.analysis.io_utils import write_csv_atomic
from src.analysis.preregistered_panel import panel_cfg

logger = logging.getLogger(__name__)

_SHAPE_PARAMS = ("centroid", "fwhm", "asymmetry")


# ---------------------------------------------------------------------------
# Gaussian peak model
# ---------------------------------------------------------------------------


def _gaussian(x: np.ndarray, amplitude: float, centroid: float, sigma: float) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((x - centroid) / sigma) ** 2)


def _r_squared(y_obs: np.ndarray, y_fit: np.ndarray) -> float:
    ss_res = float(np.sum((y_obs - y_fit) ** 2))
    ss_tot = float(np.sum((y_obs - np.mean(y_obs)) ** 2))
    if ss_tot == 0.0:
        return np.nan
    return 1.0 - ss_res / ss_tot


def _snr_from_fit(intensities: np.ndarray, y_fit: np.ndarray, amplitude: float) -> float:
    residuals = intensities - y_fit
    noise = float(np.std(residuals, ddof=1)) if residuals.size > 1 else 0.0
    if noise <= 0.0 or not np.isfinite(noise):
        return np.nan
    return float(abs(amplitude) / noise)


def _empirical_asymmetry(wavenumbers: np.ndarray, intensities: np.ndarray, centroid: float) -> float:
    """Right half-width / left half-width at half-maximum.

    Values > 1 = right-skew; < 1 = left-skew; 1 = symmetric.
    Returns NaN when the peak cannot be bisected reliably.
    """
    half_max = float(np.max(intensities)) / 2.0
    above = wavenumbers[intensities >= half_max]
    if above.size < 2:
        return np.nan
    left = float(centroid) - float(above[0])
    if left <= 0.0:
        return np.nan
    return (float(above[-1]) - float(centroid)) / left


# ---------------------------------------------------------------------------
# Single-peak fitter
# ---------------------------------------------------------------------------


def fit_local_peak(
    wavenumbers: np.ndarray,
    intensities: np.ndarray,
    *,
    lo: float,
    hi: float,
    min_snr: float = 5.0,
    calibration_floor_cm: float = 1.5,
) -> dict[str, float | str]:
    """Fit a Gaussian to the window [lo, hi] and return shape descriptors.

    Returns a dict with keys:
    ``centroid``, ``fwhm``, ``asymmetry``, ``amplitude``, ``r2``, ``snr``,
    ``snr_flag``, ``calibration_flag``, ``nominal_centre``, ``fit_status``.
    Numeric values are NaN when the fit fails or the SNR gate rejects the window.
    """
    nominal_centre = (lo + hi) / 2.0
    result: dict[str, float | str] = {
        "nominal_centre": nominal_centre,
        "centroid": np.nan,
        "fwhm": np.nan,
        "asymmetry": np.nan,
        "amplitude": np.nan,
        "r2": np.nan,
        "snr": np.nan,
        "snr_flag": "ok",
        "calibration_flag": "ok",
        "fit_status": "ok",
    }

    mask = (wavenumbers >= lo) & (wavenumbers <= hi)
    w_win = wavenumbers[mask]
    i_win = intensities[mask]

    if w_win.size < 4:
        result["fit_status"] = "too_few_channels"
        return result

    amp0 = float(np.max(i_win) - np.min(i_win)) or 1.0
    c0 = float(w_win[np.argmax(i_win)])
    sigma0 = (hi - lo) / 4.0 or 1.0

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, _ = curve_fit(
                _gaussian, w_win, i_win,
                p0=[amp0, c0, sigma0],
                bounds=([0.0, lo, 0.01], [np.inf, hi, hi - lo]),
                maxfev=5000,
            )
    except (RuntimeError, ValueError):
        result["fit_status"] = "fit_failed"
        return result

    amplitude, centroid, sigma = float(popt[0]), float(popt[1]), float(popt[2])
    y_fit = _gaussian(w_win, amplitude, centroid, sigma)

    result.update({
        "amplitude": amplitude,
        "centroid": centroid,
        "fwhm": 2.3548 * sigma,
        "asymmetry": _empirical_asymmetry(w_win, i_win, centroid),
        "r2": _r_squared(i_win, y_fit),
        "snr": _snr_from_fit(i_win, y_fit, amplitude),
    })

    snr = float(result["snr"])
    if np.isfinite(snr) and snr < min_snr:
        result["snr_flag"] = "below_snr_gate"

    if np.isfinite(float(result["centroid"])) and abs(centroid - nominal_centre) < calibration_floor_cm:
        result["calibration_flag"] = "below_calibration_floor_not_interpretable"

    return result


# ---------------------------------------------------------------------------
# Subject-level feature extraction
# ---------------------------------------------------------------------------


def _window_wavenumbers_and_intensities(
    frame: pd.DataFrame,
    columns: list[str],
    block: str,
    lo: float,
    hi: float,
) -> tuple[np.ndarray, np.ndarray]:
    col_wn = [(col, wn) for col, wn in block_columns(columns, block) if lo <= wn <= hi]
    if not col_wn:
        return np.array([]), np.empty((len(frame), 0))
    wavenumbers = np.array([wn for _, wn in col_wn], dtype=float)
    intensities = frame[[col for col, _ in col_wn]].to_numpy(dtype=float)
    return wavenumbers, intensities


def extract_peak_shape_features(
    frame: pd.DataFrame,
    peak_windows: dict[str, tuple[float, float]],
    *,
    min_snr: float,
    calibration_floor_cm: float,
) -> pd.DataFrame:
    """Per-subject peak-shape features for each registered window.

    Returns one row per (subject × window), or empty when no fits succeed.
    """
    spectral_cols = [c for c in frame.columns if c.startswith(("1300__", "2850__"))]
    rows: list[dict[str, object]] = []

    for window_name, (lo, hi) in peak_windows.items():
        block = "1300" if hi < 2000 else "2850"
        wavenumbers, intensities_mat = _window_wavenumbers_and_intensities(
            frame, spectral_cols, block, lo, hi
        )
        if wavenumbers.size == 0:
            logger.warning("Phase M: no channels for window %s [%.0f–%.0f]", window_name, lo, hi)
            continue

        for idx, subject_id in enumerate(frame["subject_id"]):
            intensities = intensities_mat[idx]
            if not np.any(np.isfinite(intensities)):
                continue
            fit = fit_local_peak(wavenumbers, intensities, lo=lo, hi=hi,
                                  min_snr=min_snr, calibration_floor_cm=calibration_floor_cm)
            rows.append({"subject_id": str(subject_id), "window_name": window_name, "block": block, **fit})

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Population summary with bootstrap CIs
# ---------------------------------------------------------------------------


def _bootstrap_median_ci(values: np.ndarray, *, n_bootstrap: int, random_state: int) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size < 2 or n_bootstrap <= 0:
        return np.nan, np.nan
    rng = np.random.default_rng(random_state)
    samples = np.array(
        [float(np.median(rng.choice(finite, size=finite.size, replace=True))) for _ in range(n_bootstrap)]
    )
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def compute_shape_summary(
    features: pd.DataFrame,
    *,
    n_bootstrap: int = 2000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Per-window population-level summary: median centroid/FWHM/asymmetry + 95% bootstrap CIs."""
    if features.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for window_name, group in features.groupby("window_name"):
        usable = group.loc[(group["snr_flag"] == "ok") & (group["fit_status"] == "ok")]
        for param in _SHAPE_PARAMS:
            finite = usable[param].to_numpy(dtype=float) if param in usable.columns else np.array([])
            finite = finite[np.isfinite(finite)]
            ci_lo, ci_hi = _bootstrap_median_ci(finite, n_bootstrap=n_bootstrap, random_state=random_state)
            rows.append({
                "window_name": window_name,
                "parameter": param,
                "n_subjects": int(finite.size),
                "median": float(np.median(finite)) if finite.size > 0 else np.nan,
                "bootstrap_ci_lower": ci_lo,
                "bootstrap_ci_upper": ci_hi,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Endpoint associations
# ---------------------------------------------------------------------------


def compute_shape_endpoint_associations(
    features: pd.DataFrame,
    clinical: pd.DataFrame,
    *,
    endpoints: list[str],
    covariates: list[str],
    min_subjects: int,
    fdr_alpha: float,
) -> pd.DataFrame:
    """Per (window × parameter × endpoint) Spearman + partial Spearman + BH-FDR."""
    if features.empty:
        return pd.DataFrame()

    usable = features.loc[(features["snr_flag"] == "ok") & (features["fit_status"] == "ok")]
    per_window: list[pd.DataFrame] = []
    for window_name, group in usable.groupby("window_name"):
        pivot = group[["subject_id", *_SHAPE_PARAMS]].copy()
        assoc = compute_spearman_associations(
            pivot, clinical, list(_SHAPE_PARAMS), endpoints, covariates,
            min_subjects, fdr_alpha, feature_col_name="parameter",
        )
        if not assoc.empty:
            assoc.insert(0, "window_name", window_name)
            per_window.append(assoc)

    return pd.concat(per_window, ignore_index=True) if per_window else pd.DataFrame()


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def _plot_shape_summary(summary: pd.DataFrame, output_dir: Path) -> Path | None:
    centroid_rows = summary.loc[summary["parameter"] == "centroid"].copy()
    if centroid_rows.empty:
        return None

    fig, ax = plt.subplots(figsize=(6, max(2.5, 0.4 * len(centroid_rows))))
    y = np.arange(len(centroid_rows))
    xerr = np.vstack([
        (centroid_rows["median"] - centroid_rows["bootstrap_ci_lower"]).to_numpy(dtype=float),
        (centroid_rows["bootstrap_ci_upper"] - centroid_rows["median"]).to_numpy(dtype=float),
    ])
    ax.errorbar(centroid_rows["median"], y, xerr=xerr,
                fmt="o", color="#D55E00", ecolor="#999999", capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(centroid_rows["window_name"])
    ax.set_xlabel("Median centroid (cm⁻¹) ± 95% bootstrap CI")
    ax.set_title("Peak centroid shifts (population summary)")
    fig.tight_layout()

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / "peak_centroid_summary.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Orchestrating runner
# ---------------------------------------------------------------------------


def run_spectral_shape(
    *,
    feature_matrix_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Phase M orchestrator: fit peaks, bootstrap CIs, associate with endpoints."""
    if cfg is None:
        cfg = config
    panel = panel_cfg(cfg)
    if not bool(cfg.get("analysis", {}).get("spectral_shape", {}).get("enabled", True)):
        return {}

    min_snr = float(panel.get("min_snr_for_peak_fit", 5.0))
    calibration_floor = float(panel.get("calibration_floor_cm", 1.5))
    n_bootstrap = int(panel.get("n_bootstrap", 2000))
    random_state = int(panel.get("random_state", 42))
    endpoints = [str(e) for e in panel.get("endpoints", {}).get("continuous", [])]
    covariates = [str(c) for c in panel.get("covariates", [])]
    fdr_alpha = float(panel.get("fdr_alpha", 0.05))
    min_subjects = int(panel.get("min_subjects_per_group", 5))

    window_cfg = panel.get("peak_shape_windows", {})
    peak_windows: dict[str, tuple[float, float]] = {
        name: (float(bounds[0]), float(bounds[1]))
        for name, bounds in window_cfg.items()
        if isinstance(bounds, (list, tuple)) and len(bounds) == 2
    }
    if not peak_windows:
        peak_windows = {
            "lipid_carbonyl_1721": (1710.0, 1735.0),
            "protein_1555": (1540.0, 1570.0),
            "carotenoid_1525": (1510.0, 1535.0),
        }
        logger.warning("Phase M: peak_shape_windows not in config; using built-in defaults")

    target_dir = Path(output_dir) if output_dir is not None else get_spectral_interpretability_dir("peak_shape")
    target_dir.mkdir(parents=True, exist_ok=True)

    clinical = filter_disease_subjects(load_subject_feature_matrix(feature_matrix_path=feature_matrix_path))
    features = extract_peak_shape_features(
        clinical, peak_windows, min_snr=min_snr, calibration_floor_cm=calibration_floor
    )
    if not features.empty:
        n_below_snr = int((features["snr_flag"] != "ok").sum())
        if n_below_snr:
            logger.info("Phase M: %d subject×window fits below SNR gate", n_below_snr)

    features_path = target_dir / "peak_shape_features.csv"
    write_csv_atomic(features, features_path, index=False)

    summary = compute_shape_summary(features, n_bootstrap=n_bootstrap, random_state=random_state)
    panel_summary_path = get_confirmatory_panel_dir() / "panel_peak_shape_features.csv"
    write_csv_atomic(summary, panel_summary_path, index=False)

    outputs: dict[str, Path] = {
        "peak_shape_features": features_path,
        "panel_peak_shape_features": panel_summary_path,
    }

    associations = (
        compute_shape_endpoint_associations(
            features, clinical, endpoints=endpoints, covariates=covariates,
            min_subjects=min_subjects, fdr_alpha=fdr_alpha,
        )
        if not features.empty and endpoints
        else pd.DataFrame()
    )
    if not associations.empty:
        assoc_path = target_dir / "peak_shape_associations.csv"
        write_csv_atomic(associations, assoc_path, index=False)
        outputs["peak_shape_associations"] = assoc_path

    if not summary.empty:
        fig_path = _plot_shape_summary(summary, target_dir)
        if fig_path:
            outputs["peak_shape_figure"] = fig_path

    logger.info("Phase M outputs written to %s", target_dir)
    return outputs
