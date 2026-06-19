"""Region-influence robustness analysis for Raman EDA.

Compares baseline and region-weighted/excluded PCA outputs to determine
whether suspicious spectral regions drive the latent-structure story.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis._reporting import dataframe_to_markdown
from src.analysis.statistics import run_pca
from src.preprocessing.pipeline import apply_region_weights

logger = logging.getLogger(__name__)


def _suspect_window_str(windows: list[list[float]]) -> str:
    if not windows:
        return "none"
    return ", ".join(f"{lo:.1f}-{hi:.1f}" for lo, hi in windows)


def compare_pca_robustness(
    baseline_spectra: np.ndarray,
    weighted_spectra: np.ndarray,
    wavenumbers: np.ndarray,
    n_components: int = 3,
    random_state: int = 42,
) -> dict:
    """Compare baseline vs region-weighted PCA.

    Returns
    -------
    dict
        ``pca_baseline``, ``pca_weighted``, and summary DataFrames.
    """
    pca_base = run_pca(baseline_spectra, n_components=n_components, scale=False, random_state=random_state)
    pca_weight = run_pca(weighted_spectra, n_components=n_components, scale=False, random_state=random_state)

    n_comp = min(n_components, pca_base["loadings"].shape[0], pca_weight["loadings"].shape[0])

    score_corr = []
    loading_corr = []
    for i in range(n_comp):
        s1 = pca_base["scores"][:, i]
        s2 = pca_weight["scores"][:, i]
        score_corr.append(float(np.corrcoef(s1, s2)[0, 1]) if len(s1) > 1 else np.nan)

        l1 = pca_base["loadings"][i]
        l2 = pca_weight["loadings"][i]
        loading_corr.append(float(np.corrcoef(l1, l2)[0, 1]) if len(l1) > 1 else np.nan)

    windows = _infer_suspect_windows(baseline_spectra, weighted_spectra, wavenumbers)
    mean_abs_loading_inside = []
    mean_abs_loading_outside = []
    for i in range(n_comp):
        abs_l = np.abs(pca_base["loadings"][i])
        inside = np.zeros_like(abs_l, dtype=bool)
        for lo, hi in windows:
            inside |= (wavenumbers >= lo) & (wavenumbers <= hi)
        mean_abs_loading_inside.append(float(abs_l[inside].mean()) if inside.any() else np.nan)
        mean_abs_loading_outside.append(float(abs_l[~inside].mean()) if (~inside).any() else np.nan)

    summary_df = pd.DataFrame(
        {
            "pc": [f"PC{i + 1}" for i in range(n_comp)],
            "score_correlation": score_corr,
            "loading_correlation": loading_corr,
            "explained_variance_ratio_baseline": pca_base["explained_variance_ratio"][:n_comp],
            "explained_variance_ratio_weighted": pca_weight["explained_variance_ratio"][:n_comp],
            "mean_abs_loading_inside_suspect": mean_abs_loading_inside,
            "mean_abs_loading_outside_suspect": mean_abs_loading_outside,
        }
    )

    return {
        "pca_baseline": pca_base,
        "pca_weighted": pca_weight,
        "summary": summary_df,
    }


def _infer_suspect_windows(
    baseline_spectra: np.ndarray,
    weighted_spectra: np.ndarray,
    wavenumbers: np.ndarray,
) -> list[list[float]]:
    """Infer which windows were modified by comparing baseline and weighted spectra."""
    diff = np.abs(baseline_spectra - weighted_spectra).mean(axis=0)
    threshold = diff.max() * 0.5 if diff.max() > 0 else 1.0
    affected = diff > threshold
    if not affected.any():
        return []

    # Collapse contiguous affected regions
    regions = []
    in_region = False
    start = 0
    for i, flag in enumerate(affected):
        if flag and not in_region:
            in_region = True
            start = i
        elif not flag and in_region:
            in_region = False
            regions.append([float(wavenumbers[start]), float(wavenumbers[i - 1])])
    if in_region:
        regions.append([float(wavenumbers[start]), float(wavenumbers[-1])])
    return regions


def compute_loading_shift(
    pca_baseline: dict,
    pca_weighted: dict,
    wavenumbers: np.ndarray,
    top_k: int = 20,
) -> pd.DataFrame:
    """Compare top driver wavenumbers before and after region weighting.

    Returns
    -------
    pd.DataFrame
        Columns: ``pc``, ``rank``, ``wavenumber``, ``loading_baseline``,
        ``loading_weighted``, ``loading_shift``.
    """
    n_comp = min(
        pca_baseline["loadings"].shape[0],
        pca_weighted["loadings"].shape[0],
    )
    rows = []
    for i in range(n_comp):
        abs_load = np.abs(pca_baseline["loadings"][i])
        top_idx = np.argsort(abs_load)[::-1][:top_k]
        for rank, j in enumerate(top_idx, start=1):
            rows.append(
                {
                    "pc": f"PC{i + 1}",
                    "rank": rank,
                    "wavenumber": float(wavenumbers[j]),
                    "loading_baseline": float(pca_baseline["loadings"][i, j]),
                    "loading_weighted": float(pca_weighted["loadings"][i, j]),
                    "loading_shift": float(
                        pca_weighted["loadings"][i, j] - pca_baseline["loadings"][i, j]
                    ),
                }
            )
    return pd.DataFrame(rows)





def run_region_robustness(
    baseline_spectra: np.ndarray,
    wavenumbers: np.ndarray,
    labels: np.ndarray,
    region_weighting: dict,
    random_state: int = 42,
    n_components: int = 3,
) -> dict:
    """Run the full region-influence robustness comparison.

    Parameters
    ----------
    baseline_spectra : np.ndarray
        Subject-mean or pixel spectra matrix (baseline).
    wavenumbers : np.ndarray
        1D wavenumber axis.
    labels : np.ndarray
        Group/condition labels aligned with ``baseline_spectra``.
    region_weighting : dict
        Region-weighting config for ``apply_region_weights``.
    random_state : int
        RNG seed.
    n_components : int
        Number of PCA components to compare.

    Returns
    -------
    dict
        ``pca_comparison``, ``loading_shift``.
    """
    weighted_spectra = apply_region_weights(baseline_spectra, wavenumbers, region_weighting)

    pca_comp = compare_pca_robustness(
        baseline_spectra,
        weighted_spectra,
        wavenumbers,
        n_components=n_components,
        random_state=random_state,
    )
    loading_shift = compute_loading_shift(
        pca_comp["pca_baseline"],
        pca_comp["pca_weighted"],
        wavenumbers,
    )

    result = {
        "pca_comparison": pca_comp,
        "loading_shift": loading_shift,
    }



    return result


def write_region_influence_report(
    pca_summary: pd.DataFrame,
    loading_shift: pd.DataFrame,
    suspect_windows: list[list[float]],
    out_path: str | Path,
) -> None:
    """Write a concise region-influence markdown report."""
    out_path = Path(out_path)

    def _loading_changed() -> bool:
        if pca_summary.empty or "loading_correlation" not in pca_summary.columns:
            return False
        return bool((pca_summary["loading_correlation"] < 0.9).any())

    if _loading_changed():
        recommendation = "The suspect region materially influences results: report as sensitivity limitation or exclude/downweight."
    else:
        recommendation = "Main structural conclusions are robust to the tested window."

    lines = [
        "# Region-influence robustness report",
        "",
        f"Tested windows: {_suspect_window_str(suspect_windows)}",
        "",
        "## PCA robustness",
        "",
        dataframe_to_markdown(pca_summary),
        "",
        "## Loading shifts (top drivers)",
        "",
        dataframe_to_markdown(loading_shift.head(20)),
        "",
    ]



    lines.extend(
        [
            "## Recommendation",
            "",
            recommendation,
            "",
        ]
    )

    out_path.write_text("\n".join(lines), encoding="utf-8")
