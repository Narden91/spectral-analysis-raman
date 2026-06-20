import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.metadata.parser import parse_filename

logger = logging.getLogger(__name__)


def read_txt_spectrum(
    filepath: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Reads a Renishaw exported .txt file containing Raman mapping data.

    The file is formatted with whitespace-separated columns:
    #X      #Y      #Wave      #Intensity

    Returns:
        wavenumbers (np.ndarray): 1D array of length W
        x_coords (np.ndarray): 1D array of length N (unique X positions)
        y_coords (np.ndarray): 1D array of length M (unique Y positions)
        spectra (np.ndarray): 3D array of shape (N, M, W) or 2D (N*M, W)
        metadata (dict): Extracted metadata from filename
    """
    try:
        df = pd.read_csv(filepath, sep=r"\s+", engine="c", dtype=np.float64, encoding="utf-8")
    except UnicodeDecodeError:
        logger.debug("UTF-8 decode failed for %s; retrying with latin-1", Path(filepath).name)
        df = pd.read_csv(filepath, sep=r"\s+", engine="c", dtype=np.float64, encoding="latin-1")
    df.columns = [c.strip().lstrip("#") for c in df.columns]

    try:
        x_col = next(c for c in df.columns if c.upper() == "X")
    except StopIteration:
        raise KeyError(
            f"No 'X' column found in '{Path(filepath).name}'. "
            f"Available columns: {list(df.columns)}"
        )
    try:
        y_col = next(c for c in df.columns if c.upper() == "Y")
    except StopIteration:
        raise KeyError(
            f"No 'Y' column found in '{Path(filepath).name}'. "
            f"Available columns: {list(df.columns)}"
        )
    try:
        wave_col = next(
            c for c in df.columns if "WAVE" in c.upper() or "SHIFT" in c.upper()
        )
    except StopIteration:
        raise KeyError(
            f"No wavenumber column (WAVE/SHIFT) found in '{Path(filepath).name}'. "
            f"Available columns: {list(df.columns)}"
        )
    try:
        int_col = next(c for c in df.columns if "INT" in c.upper())
    except StopIteration:
        raise KeyError(
            f"No intensity column (INT) found in '{Path(filepath).name}'. "
            f"Available columns: {list(df.columns)}"
        )

    wavenumbers = np.sort(df[wave_col].unique())
    num_wavenumbers = len(wavenumbers)
    num_points = len(df) // num_wavenumbers

    if len(df) % num_wavenumbers != 0:
        raise ValueError(
            f"Data length {len(df)} is not a multiple of {num_wavenumbers} wavenumbers."
        )

    # Sort by (point index, wavenumber) so the reshape always maps columns
    # to sorted wavenumbers regardless of file-internal ordering.
    df = df.sort_values(by=[x_col, y_col, wave_col]).reset_index(drop=True)

    # Use zero-copy strided views to extract the full X and Y matrices
    x_coords_full = df[x_col].values[::num_wavenumbers]
    y_coords_full = df[y_col].values[::num_wavenumbers]

    # Ignore NaNs when finding unique coordinates
    x_unique = np.unique(x_coords_full[~np.isnan(x_coords_full)])
    y_unique = np.unique(y_coords_full[~np.isnan(y_coords_full)])

    spectra_flat = df[int_col].values.reshape(num_points, num_wavenumbers)

    if len(x_unique) * len(y_unique) == num_points:
        spectra_grid = spectra_flat.reshape(
            len(x_unique), len(y_unique), num_wavenumbers
        )
        return wavenumbers, x_unique, y_unique, spectra_grid, parse_filename(filepath)

    logger.warning(
        "Non-rectangular spatial grid in '%s' (%d x_unique × %d y_unique ≠ %d points); "
        "returning flat 2D spectra array.",
        Path(filepath).name,
        len(x_unique),
        len(y_unique),
        num_points,
    )
    return (
        wavenumbers,
        x_coords_full,
        y_coords_full,
        spectra_flat,
        parse_filename(filepath),
    )
