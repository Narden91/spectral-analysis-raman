"""
Statistical analysis utilities for Raman spectroscopy data.

Provides univariate and multivariate statistical tests, PCA,
and group comparison tools for plasma-derived extracellular vesicle (EV) datasets.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src import config
from src.constants import CANONICAL_REGION_BOUNDS

logger = logging.getLogger(__name__)


REGION_BOUNDS = CANONICAL_REGION_BOUNDS




def cohens_d(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.size < 2 or right.size < 2:
        return np.nan

    pooled_var = (
        np.var(left, ddof=1) * (left.size - 1)
        + np.var(right, ddof=1) * (right.size - 1)
    ) / max(left.size + right.size - 2, 1)
    if pooled_var <= 0 or not np.isfinite(pooled_var):
        return 0.0
    return float((np.mean(left) - np.mean(right)) / np.sqrt(pooled_var))


def _finite_1d(values: np.ndarray | pd.Series) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    return array[np.isfinite(array)]


def cliffs_delta(left: np.ndarray | pd.Series, right: np.ndarray | pd.Series) -> float:
    """Return Cliff's delta, oriented as left group minus right group."""
    left_values = _finite_1d(left)
    right_values = _finite_1d(right)
    if left_values.size == 0 or right_values.size == 0:
        return np.nan

    sorted_right = np.sort(right_values)
    less = np.searchsorted(sorted_right, left_values, side="left")
    greater = right_values.size - np.searchsorted(sorted_right, left_values, side="right")
    n_pairs = float(left_values.size * right_values.size)
    return float((np.sum(less) - np.sum(greater)) / n_pairs)


def _rank_covariates(covars: np.ndarray | pd.DataFrame | pd.Series, valid: np.ndarray) -> np.ndarray:
    if isinstance(covars, pd.Series):
        covar_frame = covars.to_frame()
    elif isinstance(covars, pd.DataFrame):
        covar_frame = covars.copy()
    else:
        covar_array = np.asarray(covars)
        if covar_array.ndim == 1:
            covar_array = covar_array[:, np.newaxis]
        covar_frame = pd.DataFrame(covar_array)

    ranked_columns: list[np.ndarray] = []
    for column in covar_frame.columns:
        values = covar_frame.loc[valid, column]
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.notna().all():
            encoded = numeric.to_numpy(dtype=float)
        else:
            encoded = pd.factorize(values.astype("string"), sort=True)[0].astype(float)
        ranked = stats.rankdata(encoded, method="average")
        if np.nanstd(ranked) > 0:
            ranked_columns.append(ranked)

    if not ranked_columns:
        return np.empty((int(valid.sum()), 0), dtype=float)
    return np.column_stack(ranked_columns)


def partial_spearman(
    x: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
    covars: np.ndarray | pd.DataFrame | pd.Series,
) -> tuple[float, float, int]:
    """Partial Spearman correlation of x and y after residualizing rank covariates."""
    x_series = pd.Series(x)
    y_series = pd.Series(y)
    x_numeric = pd.to_numeric(x_series, errors="coerce")
    y_numeric = pd.to_numeric(y_series, errors="coerce")

    covar_frame = covars.to_frame() if isinstance(covars, pd.Series) else pd.DataFrame(covars)
    valid = x_numeric.notna() & y_numeric.notna()
    for column in covar_frame.columns:
        valid &= covar_frame[column].notna()
    valid_array = valid.to_numpy(dtype=bool)
    n = int(valid_array.sum())
    if n < 3:
        return np.nan, np.nan, n

    x_rank = stats.rankdata(x_numeric.loc[valid_array].to_numpy(dtype=float), method="average")
    y_rank = stats.rankdata(y_numeric.loc[valid_array].to_numpy(dtype=float), method="average")
    covar_ranks = _rank_covariates(covar_frame, valid_array)

    if covar_ranks.shape[1] == 0:
        rho, p_value = stats.spearmanr(x_rank, y_rank)
        return float(rho), float(p_value), n

    design = np.column_stack([np.ones(n, dtype=float), covar_ranks])
    rank = int(np.linalg.matrix_rank(design) - 1)
    outcomes = np.column_stack([x_rank, y_rank])
    fitted = design @ np.linalg.lstsq(design, outcomes, rcond=None)[0]
    x_resid = x_rank - fitted[:, 0]
    y_resid = y_rank - fitted[:, 1]

    if np.std(x_resid) == 0 or np.std(y_resid) == 0:
        return np.nan, np.nan, n

    rho = float(np.corrcoef(x_resid, y_resid)[0, 1])
    df = n - rank - 2
    if df <= 0 or not np.isfinite(rho):
        return rho, np.nan, n
    clipped = float(np.clip(rho, -1.0 + 1e-12, 1.0 - 1e-12))
    t_stat = clipped * np.sqrt(df / max(1.0 - clipped**2, 1e-12))
    p_value = float(2.0 * stats.t.sf(abs(t_stat), df=df))
    if abs(rho) >= 1.0:
        p_value = 0.0
    return rho, p_value, n


def variance_homogeneity_test(
    left: np.ndarray | pd.Series,
    right: np.ndarray | pd.Series,
    *,
    center: str = "mean",
) -> tuple[float, float]:
    """Levene-family test for equal variances between two groups."""
    if center not in {"mean", "median"}:
        raise ValueError("center must be 'mean' or 'median'")
    left_values = _finite_1d(left)
    right_values = _finite_1d(right)
    if left_values.size < 2 or right_values.size < 2:
        return np.nan, np.nan
    statistic, p_value = stats.levene(left_values, right_values, center=center)
    return float(statistic), float(p_value)


def _expected_whole_spectra_boundary() -> tuple[float, float]:
    bounds_cfg = config.get("data", {}).get("spectral_range_bounds", {})
    _, default_fingerprint_end = CANONICAL_REGION_BOUNDS["fingerprint"]
    default_ch_stretch_start, _ = CANONICAL_REGION_BOUNDS["ch_stretch"]
    fingerprint_end = float(bounds_cfg.get("fingerprint_end", default_fingerprint_end))
    ch_stretch_start = float(bounds_cfg.get("ch_stretch_start", default_ch_stretch_start))
    return fingerprint_end, ch_stretch_start


# ---------------------------------------------------------------------------
# Normality testing
# ---------------------------------------------------------------------------


def test_normality(data: np.ndarray, alpha: float = 0.05) -> dict:
    """
    Test normality per feature (wavenumber) using Shapiro-Wilk.

    When n_features exceeds the configured subsample threshold, only a
    random subset of wavenumbers is tested to save computation time.

    Parameters
    ----------
    data : np.ndarray
        2D array (n_samples, n_features).
    alpha : float
        Significance level.

    Returns
    -------
    dict
        Keys: 'is_normal' (bool array), 'p_values' (float array),
              'fraction_normal' (float), 'n_tested' (int).
    """
    n_features = data.shape[1]
    max_test = int(
        config.get("analysis", {}).get("normality_subsample_wavenumbers", n_features)
    )

    if max_test < n_features:
        rng = np.random.default_rng(
            int(config.get("analysis", {}).get("random_state", 42))
        )
        test_idx = rng.choice(n_features, size=max_test, replace=False)
        test_idx.sort()
    else:
        test_idx = np.arange(n_features)

    p_values = np.full(n_features, np.nan)

    for j in test_idx:
        col = data[:, j]
        col_clean = col[~np.isnan(col)]
        if len(col_clean) < 8:
            p_values[j] = np.nan
        else:
            _, p_values[j] = stats.shapiro(col_clean)

    is_normal = p_values > alpha
    fraction = np.nanmean(is_normal[test_idx])

    logger.debug(
        "Normality test (%d/%d features): %.1f%% are normal (alpha=%.3f)",
        len(test_idx),
        n_features,
        fraction * 100,
        alpha,
    )
    return {
        "is_normal": is_normal,
        "p_values": p_values,
        "fraction_normal": fraction,
        "n_tested": len(test_idx),
    }


# ---------------------------------------------------------------------------
# Multiple comparison correction (standalone)
# ---------------------------------------------------------------------------


def fdr_correction(
    p_values: np.ndarray, alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """
    Benjamini-Hochberg FDR correction.

    Returns
    -------
    corrected : np.ndarray
        Adjusted p-values.
    significant : np.ndarray
        Boolean mask of significant results.
    """
    valid = ~np.isnan(p_values)
    corrected = np.full_like(p_values, np.nan)

    if valid.sum() == 0:
        return corrected, np.zeros_like(p_values, dtype=bool)

    pv = p_values[valid]
    n = len(pv)
    sort_idx = np.argsort(pv)
    ranks = np.empty(n, dtype=int)
    ranks[sort_idx] = np.arange(1, n + 1)

    adj = np.minimum(pv * n / ranks, 1.0)
    # Enforce monotonicity: cumulative min from largest to smallest rank
    adj_sorted = adj[sort_idx]
    adj_sorted[:] = np.minimum.accumulate(adj_sorted[::-1])[::-1]
    adj[sort_idx] = adj_sorted

    corrected[valid] = adj
    significant = corrected < alpha
    return corrected, significant


def infer_concatenated_range_split_index(wavenumbers: np.ndarray) -> int:
    """Infer the boundary between concatenated spectral ranges.

    Whole-spectrum mode concatenates the fingerprint and CH-stretch regions,
    which creates one much larger gap than the within-range channel spacing.
    """
    wavenumbers = np.asarray(wavenumbers, dtype=float)
    if wavenumbers.ndim != 1 or wavenumbers.size < 3:
        raise ValueError("wavenumbers must be a 1D array with at least 3 values")
    if not np.all(np.diff(wavenumbers) > 0):
        raise ValueError("wavenumbers must be strictly increasing")

    gaps = np.diff(wavenumbers)
    max_gap_index = int(np.argmax(gaps))
    max_gap = float(gaps[max_gap_index])
    typical_gap = float(np.nanmedian(np.abs(gaps)))
    if not np.isfinite(typical_gap) or typical_gap <= 0:
        raise ValueError("Could not determine typical wavenumber spacing")
    if max_gap <= typical_gap * 5.0:
        raise ValueError("No concatenated spectral-range boundary detected")

    _, expected_ch_stretch_start = _expected_whole_spectra_boundary()
    split_index = max_gap_index + 1
    if abs(float(wavenumbers[split_index]) - expected_ch_stretch_start) >= 200.0:
        raise ValueError(
            "Detected concatenated spectral-range boundary is implausibly far from the "
            "expected CH-stretch start"
        )

    return split_index


def summarize_group_centroid_distances(
    groups: dict[str, np.ndarray],
    metric: str = "euclidean",
) -> pd.DataFrame:
    """Compute pairwise distances between group centroid spectra."""
    if not groups:
        return pd.DataFrame()
    if metric != "euclidean":
        raise ValueError(f"Unsupported metric '{metric}'")

    group_names = list(groups.keys())
    centroids = {
        group_name: np.nanmean(np.asarray(spectra, dtype=float), axis=0)
        for group_name, spectra in groups.items()
    }

    distances = np.zeros((len(group_names), len(group_names)), dtype=float)
    for i, left_name in enumerate(group_names):
        for j, right_name in enumerate(group_names):
            left = centroids[left_name]
            right = centroids[right_name]
            distances[i, j] = float(np.linalg.norm(left - right))

    return pd.DataFrame(distances, index=group_names, columns=group_names)










# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------


def run_pca(
    spectra: np.ndarray,
    n_components: Optional[int] = None,
    random_state: Optional[int] = None,
    scale: bool = True,
    *,
    log_summary: bool = True,
    warn_on_high_pc1: bool = True,
) -> dict:
    """
    Run PCA on spectra matrix.

    Parameters
    ----------
    spectra : np.ndarray
        2D array (n_samples, n_wavenumbers).
    n_components : int
        Number of components to keep.
    random_state : int
        Random seed for reproducibility.
    scale : bool
        If True (default), apply StandardScaler (zero-mean, unit-variance per
        wavenumber). If False, center only (subtract feature means).  Use False
        when spectra have already been SNV-normalised and you want to preserve
        the relative scale of wavenumber channels.

    Returns
    -------
    dict
        'scores': (n_samples, n_components),
        'loadings': (n_components, n_wavenumbers),
        'explained_variance_ratio': (n_components,),
        'cumulative_variance': (n_components,),
        'pca_model': fitted PCA object,
        'scaler': fitted StandardScaler or None when scale=False.
    """
    analysis_cfg = config.get("analysis", {})
    n_components = (
        n_components
        if n_components is not None
        else analysis_cfg.get("pca_n_components", 10)
    )
    random_state = (
        random_state
        if random_state is not None
        else analysis_cfg.get("random_state", 42)
    )

    # Drop columns that are all-NaN (e.g. channels with no valid data after
    # nanmedian aggregation).  sklearn PCA does not handle NaN; leaving them in
    # would corrupt scores silently or raise.
    spectra = np.asarray(spectra, dtype=float)
    all_nan_cols = np.all(~np.isfinite(spectra), axis=0)
    if all_nan_cols.any():
        n_dropped = int(all_nan_cols.sum())
        logger.warning(
            "run_pca: dropping %d all-NaN/inf feature column(s) before PCA "
            "(out of %d total)",
            n_dropped,
            spectra.shape[1],
        )
        spectra = spectra[:, ~all_nan_cols]
    if spectra.size == 0:
        raise ValueError("No valid features remain after dropping all-NaN columns.")

    # Clamp n_components to valid range
    max_components = min(spectra.shape[0], spectra.shape[1])
    n_components = min(n_components, max_components)

    if scale:
        scaler = StandardScaler()
        spectra_processed = scaler.fit_transform(spectra)
    else:
        scaler = None
        means = np.nanmean(spectra, axis=0, keepdims=True)
        spectra_processed = spectra - means

    # svd_solver="full" is deterministic regardless of matrix size; "auto" may
    # select the randomized solver for large matrices, introducing variance even
    # with a fixed random_state across sklearn versions.
    pca = PCA(n_components=n_components, random_state=random_state, svd_solver="full")
    scores = pca.fit_transform(spectra_processed)

    # Deterministic sign convention: flip each PC so the loading with the largest
    # absolute value is positive.  Scikit-learn does not guarantee sign across
    # versions; this ensures stable direction for interpretation and figures.
    signs = np.sign(pca.components_[np.arange(n_components), np.argmax(np.abs(pca.components_), axis=1)])
    signs[signs == 0] = 1.0
    pca.components_ *= signs[:, np.newaxis]
    scores *= signs[np.newaxis, :]

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    
    pc1_variance_ratio = (
        float(pca.explained_variance_ratio_[0])
        if len(pca.explained_variance_ratio_) > 0
        else np.nan
    )

    if warn_on_high_pc1 and np.isfinite(pc1_variance_ratio) and pc1_variance_ratio > 0.8:
        logger.warning(
            "PC1 explains > 80%% of the variance (%.1f%%). This is unusually high "
            "for normalized Raman spectra and may indicate baseline artifacts "
            "or a dominant background signal.",
            pc1_variance_ratio * 100
        )

    if log_summary:
        logger.info(
            "PCA: %d components explain %.1f%% of variance (scale=%s)",
            n_components,
            cumvar[-1] * 100,
            scale,
        )

    return {
        "scores": scores,
        "loadings": pca.components_,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_variance": cumvar,
        "pca_model": pca,
        "scaler": scaler,
        "pc1_variance_ratio": pc1_variance_ratio,
    }


# ---------------------------------------------------------------------------
# Known Raman bands — single source of truth is config whole_spectra.bands.
# ---------------------------------------------------------------------------


def _load_raman_bands() -> dict[str, float]:
    """Build {band_name: centre_wavenumber} from config whole_spectra.bands."""
    bands_cfg: dict = config.get("whole_spectra", {}).get("bands", {})
    return {name: float(lo + hi) / 2.0 for name, (lo, hi) in bands_cfg.items()}


def get_raman_bands() -> dict[str, float]:
    """Return the canonical Raman band centres derived from settings.yaml."""
    return _load_raman_bands()


# Backward-compatible module-level alias — populated once at import time.
# Callers that used RAMAN_BANDS directly still work; the source of truth is config.
RAMAN_BANDS: dict[str, float] = _load_raman_bands()


def extract_band_intensities(
    spectra: np.ndarray,
    wavenumbers: np.ndarray,
    bands: Optional[dict] = None,
    tolerance: float = 10.0,
) -> pd.DataFrame:
    """
    Extract mean intensity at known Raman bands for each spectrum.

    Parameters
    ----------
    spectra : np.ndarray
        2D array (n_spectra, n_wavenumbers).
    wavenumbers : np.ndarray
        1D array.
    bands : dict
        {band_name: target_wavenumber}. Defaults to RAMAN_BANDS.
    tolerance : float
        +/- wavenumber tolerance for band integration.

    Returns
    -------
    pd.DataFrame
        Columns correspond to band names; rows to spectra.
    """
    if bands is None:
        # Filter bands to those within the wavenumber range
        wmin, wmax = wavenumbers.min(), wavenumbers.max()
        bands = {
            k: v
            for k, v in RAMAN_BANDS.items()
            if wmin - tolerance <= v <= wmax + tolerance
        }

    result = {}
    for name, center in bands.items():
        mask = (wavenumbers >= center - tolerance) & (wavenumbers <= center + tolerance)
        if mask.any():
            result[name] = np.nanmean(spectra[:, mask], axis=1)
        else:
            result[name] = np.full(spectra.shape[0], np.nan)

    return pd.DataFrame(result)


def compare_band_intensities_between_groups(
    band_df: pd.DataFrame,
    group_labels: np.ndarray,
    group_a: str,
    group_b: str,
    method: Optional[str] = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Per-band statistical test between two groups, with BH-FDR across bands.

    Parameters
    ----------
    band_df : pd.DataFrame
        Output of :func:`extract_band_intensities` (one column per band).
    group_labels : np.ndarray
        1D array aligned with ``band_df`` rows.
    group_a, group_b : str
        Labels identifying the two groups to compare within ``group_labels``.
    method : str or None
        'parametric' (Welch t-test), 'non-parametric' (Mann-Whitney U), or
        'auto' (Welch t if both groups have n >= 20, else Mann-Whitney U).
    alpha : float
        Significance level for the BH-FDR correction.

    Returns
    -------
    pd.DataFrame
        Columns: band, statistic, p_value, fdr_q, significant, mean_a, mean_b,
        effect_size (Cohen's d).
    """
    analysis_cfg = config.get("analysis", {})
    method = method if method is not None else analysis_cfg.get("statistical_test", "auto")

    mask_a = np.asarray(group_labels) == group_a
    mask_b = np.asarray(group_labels) == group_b

    rows = []
    for band in band_df.columns:
        a = band_df.loc[mask_a, band].to_numpy(dtype=float)
        b = band_df.loc[mask_b, band].to_numpy(dtype=float)
        a = a[~np.isnan(a)]
        b = b[~np.isnan(b)]

        if len(a) < 2 or len(b) < 2:
            rows.append(
                {
                    "band": band, "statistic": np.nan, "p_value": np.nan,
                    "mean_a": np.nan, "mean_b": np.nan, "effect_size": np.nan,
                }
            )
            continue

        # Test selection heuristic: n>=20 → Welch t-test; n<20 → Mann-Whitney U.
        # The Shapiro-Wilk normality test (computed separately by check_normality)
        # is NOT wired into this selection.  Rationale: in small Raman cohorts
        # n is typically <20 so Mann-Whitney is chosen anyway, and FDR correction
        # is applied across bands.  Note that adjacent wavenumber bands are
        # correlated; the FDR correction does not account for this dependency.
        use_parametric = (
            method == "parametric"
            or (method == "auto" and len(a) >= 20 and len(b) >= 20)
        )
        if use_parametric:
            stat, p = stats.ttest_ind(a, b, equal_var=False)
        else:
            stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")

        pooled_std = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        effect_size = float((a.mean() - b.mean()) / pooled_std) if pooled_std > 0 else np.nan

        rows.append(
            {
                "band": band,
                "statistic": float(stat),
                "p_value": float(p),
                "mean_a": float(a.mean()),
                "mean_b": float(b.mean()),
                "effect_size": effect_size,
            }
        )

    result_df = pd.DataFrame(rows)
    fdr_q, significant = fdr_correction(result_df["p_value"].to_numpy(dtype=float), alpha=alpha)
    result_df["fdr_q"] = fdr_q
    result_df["significant"] = significant
    return result_df


def rank_wavenumbers_per_pc(
    loadings: np.ndarray,
    wavenumbers: np.ndarray,
    top_k: int = 20,
) -> pd.DataFrame:
    """Return the top-K wavenumbers ranked by absolute loading for each PC.

    Parameters
    ----------
    loadings : np.ndarray
        Shape (n_components, n_wavenumbers) — the ``loadings`` field from
        :func:`run_pca`.
    wavenumbers : np.ndarray
        1D wavenumber axis, length n_wavenumbers.
    top_k : int
        Number of wavenumbers to return per component.

    Returns
    -------
    pd.DataFrame
        Columns: ``pc``, ``rank``, ``wavenumber``, ``loading``.
        Shape: (n_components × top_k, 4).
    """
    n_components = loadings.shape[0]
    top_k = min(top_k, loadings.shape[1])
    rows = []
    for i in range(n_components):
        abs_loadings = np.abs(loadings[i])
        idx = np.argsort(abs_loadings)[::-1][:top_k]
        for rank, j in enumerate(idx, start=1):
            rows.append(
                {
                    "pc": f"PC{i + 1}",
                    "rank": rank,
                    "wavenumber": float(wavenumbers[j]),
                    "loading": float(loadings[i, j]),
                }
            )
    return pd.DataFrame(rows)


def _spectral_region_label(wavenumber: float) -> str:
    """Return a coarse region label for a wavenumber."""
    from src.constants import CANONICAL_REGION_BOUNDS

    for region, (lo, hi) in CANONICAL_REGION_BOUNDS.items():
        if lo <= wavenumber <= hi:
            return region
    return "unknown"


def build_pca_loading_interpretation(
    loadings: np.ndarray,
    wavenumbers: np.ndarray,
    explained_variance_ratio: np.ndarray,
    suspect_windows: list[list[float]] | None = None,
    top_k: int = 20,
) -> pd.DataFrame:
    """Build a compact, interpretable PCA loading table.

    Parameters
    ----------
    loadings : np.ndarray
        Shape (n_components, n_wavenumbers).
    wavenumbers : np.ndarray
        1D wavenumber axis.
    explained_variance_ratio : np.ndarray
        Per-component explained variance ratio.
    suspect_windows : list of [lo, hi] or None
        Optional windows flagged as suspicious (e.g. 2655-2685 cm^-1).
    top_k : int
        Number of wavenumbers to keep per PC.

    Returns
    -------
    pd.DataFrame
        Columns include ``pc``, ``rank``, ``wavenumber``, ``loading``,
        ``abs_loading``, ``explained_variance_ratio``, ``spectral_region``,
        ``is_in_suspect_window``, ``nearest_known_band``.
    """
    bands = get_raman_bands()
    ranked = rank_wavenumbers_per_pc(loadings, wavenumbers, top_k=top_k)
    if ranked.empty:
        return ranked

    pc_to_var = {
        f"PC{i + 1}": float(explained_variance_ratio[i])
        for i in range(min(len(explained_variance_ratio), loadings.shape[0]))
    }

    suspect_windows = suspect_windows or []

    def _nearest_band(w: float) -> str:
        if not bands:
            return ""
        return min(bands, key=lambda name: abs(bands[name] - w))

    def _in_suspect(w: float) -> bool:
        return any(float(lo) <= w <= float(hi) for lo, hi in suspect_windows)

    ranked["explained_variance_ratio"] = ranked["pc"].map(pc_to_var)
    ranked["spectral_region"] = ranked["wavenumber"].apply(_spectral_region_label)
    ranked["is_in_suspect_window"] = ranked["wavenumber"].apply(_in_suspect)
    ranked["abs_loading"] = ranked["loading"].abs()
    ranked["nearest_known_band"] = ranked["wavenumber"].apply(_nearest_band)
    return ranked[
        [
            "pc",
            "rank",
            "wavenumber",
            "loading",
            "abs_loading",
            "explained_variance_ratio",
            "spectral_region",
            "is_in_suspect_window",
            "nearest_known_band",
        ]
    ]

