from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.constants import CANONICAL_REGION_BOUNDS


def _wn_to_channel_idx(wn_value: float, n_channels: int, split_index: int, round_up: bool = False) -> int:
    wavenumbers = _get_canonical_wavenumbers()
    if wavenumbers is not None and wavenumbers.size == n_channels:
        diff = np.abs(wavenumbers - wn_value)
        idx = int(np.argmin(diff))
        if round_up and idx < n_channels - 1:
            idx += 1
        return idx
    fp_start_wn, fp_end_wn = CANONICAL_REGION_BOUNDS["fingerprint"]
    ch_start_wn, ch_end_wn = CANONICAL_REGION_BOUNDS["ch_stretch"]
    if wn_value >= ch_start_wn:
        frac = (wn_value - ch_start_wn) / (ch_end_wn - ch_start_wn)
        base = split_index + max(0, min(n_channels - split_index - 1, int(frac * (n_channels - split_index))))
    else:
        frac = (wn_value - fp_start_wn) / (fp_end_wn - fp_start_wn)
        base = max(0, min(split_index - 1, int(frac * split_index)))
    if round_up and base < n_channels - 1:
        base += 1
    return base


def _get_canonical_wavenumbers() -> np.ndarray | None:
    try:
        from src.fusion.area_pipeline import _CANONICAL_REGION_AXES
        fp = _CANONICAL_REGION_AXES.get("fingerprint")
        ch = _CANONICAL_REGION_AXES.get("ch_stretch")
        if fp is not None and ch is not None:
            return np.concatenate([fp, ch])
    except Exception:
        pass
    return None


def _band_integrals(
    vec: np.ndarray,
    bands_cfg: dict,
    n_channels: int,
    split_index: int,
) -> dict[str, float]:
    row: dict[str, float] = {}
    for band_name, band_range in bands_cfg.items():
        lo_wn, hi_wn = float(band_range[0]), float(band_range[1])
        lo_ch = _wn_to_channel_idx(lo_wn, n_channels, split_index)
        hi_ch = _wn_to_channel_idx(hi_wn, n_channels, split_index, round_up=True)
        lo_ch = max(0, lo_ch)
        hi_ch = min(n_channels, hi_ch + 1)
        if lo_ch >= n_channels or hi_ch <= lo_ch:
            row[band_name] = np.nan
            continue
        region = np.asarray(vec[lo_ch:hi_ch], dtype=float)
        row[band_name] = float(_integrate_band(region))
    return row


def extract_band_features(vectors: np.ndarray, split_index: int) -> pd.DataFrame:
    """Extract Amide III, Amide I, DNA PO2, CH2/CH3 band integrals.

    Parameters
    ----------
    vectors : np.ndarray  (n_subjects, n_channels)
    split_index : int  index where CH-stretch region begins

    Returns
    -------
    pd.DataFrame with columns like band_amide_iii, band_amide_i, band_dna_po2, band_ch2_ch3.
    """
    bands_cfg = config.get("whole_spectra", {}).get("bands", {})
    if not bands_cfg:
        return pd.DataFrame()

    n_channels = vectors.shape[1]
    rows: list[dict] = []
    for vec in np.asarray(vectors, dtype=float):
        integrals = _band_integrals(vec, bands_cfg, n_channels, split_index)
        row = {f"band_{band_name}": value for band_name, value in integrals.items()}
        rows.append(row)
    return pd.DataFrame(rows)


def extract_band_fraction_features(
    vectors: np.ndarray,
    split_index: int,
    eps: float = 1e-9,
) -> pd.DataFrame:
    bands_cfg = config.get("whole_spectra", {}).get("bands", {})
    if not bands_cfg:
        return pd.DataFrame()

    eps_val = float(eps)
    n_channels = vectors.shape[1]
    rows: list[dict] = []
    for vec in np.asarray(vectors, dtype=float):
        integrals = _band_integrals(vec, bands_cfg, n_channels, split_index)
        finite_values = np.asarray([value for value in integrals.values() if np.isfinite(value)], dtype=float)
        total = float(np.sum(finite_values)) if finite_values.size > 0 else 0.0
        row: dict[str, float] = {}
        for band_name, value in integrals.items():
            if not np.isfinite(value):
                row[f"frac_{band_name}"] = np.nan
                continue
            row[f"frac_{band_name}"] = float(value / (total + eps_val))
        rows.append(row)
    return pd.DataFrame(rows)


def extract_ratio_features(vectors: np.ndarray, split_index: int) -> pd.DataFrame:
    """Extract configured cross-region ratio features.

    Each ratio is a/b using the mean intensity over the band window.
    """
    ratio_pairs = config.get("whole_spectra", {}).get("ratio_pairs", [])
    if not ratio_pairs:
        return pd.DataFrame()

    n_channels = vectors.shape[1]
    bands_cfg = config.get("whole_spectra", {}).get("bands", {})
    rows: list[dict] = []
    for vec in np.asarray(vectors, dtype=float):
        row: dict[str, float] = {}
        for num_key, den_key in ratio_pairs:
            num_range = bands_cfg.get(num_key)
            den_range = bands_cfg.get(den_key)
            if num_range is None or den_range is None:
                row[f"ratio_{num_key}__{den_key}"] = np.nan
                continue
            num_val = _band_mean(vec, num_range, n_channels, split_index)
            den_val = _band_mean(vec, den_range, n_channels, split_index)
            row[f"ratio_{num_key}__{den_key}"] = float(num_val / den_val) if den_val != 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _band_mean(vec: np.ndarray, band_range: list, n_channels: int, split_index: int) -> float:
    lo_wn, hi_wn = float(band_range[0]), float(band_range[1])
    lo_ch = _wn_to_channel_idx(lo_wn, n_channels, split_index)
    hi_ch = _wn_to_channel_idx(hi_wn, n_channels, split_index, round_up=True)
    lo_ch = max(0, lo_ch)
    hi_ch = min(n_channels, hi_ch + 1)
    if lo_ch >= n_channels or hi_ch <= lo_ch:
        return np.nan
    return float(np.mean(vec[lo_ch:hi_ch]))


def extract_sex_covariate_block(area_df: pd.DataFrame) -> pd.DataFrame:
    """Build gender_binary covariate from area_df metadata.

    Returns
    -------
    pd.DataFrame with columns: subject_id, sex_binary
    """
    if "subject_id" not in area_df.columns or "gender" not in area_df.columns:
        raise ValueError("area_df must contain 'subject_id' and 'gender' columns")

    mapping = (
        area_df[["subject_id", "gender"]]
        .drop_duplicates(subset=["subject_id"], keep="first")
        .copy()
    )
    mapping["subject_id"] = mapping["subject_id"].astype(str)
    gender = mapping["gender"].astype(str).str.strip().str.lower()
    mapping["sex_binary"] = gender.map({"female": 0.0, "f": 0.0, "male": 1.0, "m": 1.0})
    return mapping[["subject_id", "sex_binary"]]


def _integrate_band(region: np.ndarray) -> float:
    region = np.maximum(region, 0)
    try:
        import scipy.integrate
        return float(scipy.integrate.trapezoid(region))
    except (ImportError, AttributeError):
        pass
    try:
        return float(np.trapezoid(region))
    except AttributeError:
        return float(np.trapz(region))
