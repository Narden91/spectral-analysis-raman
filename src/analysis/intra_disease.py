"""Intra-disease clinical correlation analysis for Raman EDA.

Restricted to disease subjects only. Computes per-wavenumber Spearman
correlations between subject-mean spectra and continuous clinical targets
(MoCA baseline/3-month, delta_moca, RBANS, ASPECTS, age), with BH-FDR
correction across wavenumbers per target.

This is a subject-level association analysis — it does not assign or
predict disease/control labels (see CLAUDE.md Method Guardrails).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from src.analysis.statistics import fdr_correction
from src.data_preparation.loaders import load_area_summaries

logger = logging.getLogger(__name__)

_DEFAULT_TARGETS = (
    "moca_baseline",
    "moca_3m",
    "delta_moca",
    "rbans_total",
    "aspects_score",
    "age",
)


def smooth_correlation_profile(values: np.ndarray, window: int = 9) -> np.ndarray:
    """Light rolling smoother for visualization only."""
    series = pd.Series(np.asarray(values, dtype=float))
    return series.rolling(window=window, center=True, min_periods=1).mean().to_numpy(dtype=float)


def build_disease_subject_mean_matrix(
    df_index: pd.DataFrame,
    clinical_df: pd.DataFrame,
    spectral_range: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build a disease-only subject-mean spectra matrix aligned with clinical data.

    Parameters
    ----------
    df_index : pd.DataFrame
        File index with ``subject_id`` and ``is_control``.
    clinical_df : pd.DataFrame
        Curated clinical table from ``load_clinical_data``.
    spectral_range : str
        ``'1300'`` or ``'2850'``.

    Returns
    -------
    wavenumbers : np.ndarray
        1D wavenumber axis.
    mean_spectra : np.ndarray
        2D array (n_disease_subjects, n_wavenumbers).
    clinical_aligned : pd.DataFrame
        Clinical rows aligned row-wise with ``mean_spectra``, indexed by
        ``subject_id``.
    """
    payload = load_area_summaries(None, spectral_range)
    disease_subjects = (
        df_index.loc[~df_index["is_control"].astype(bool), "subject_id"].astype(str).unique().tolist()
    )
    rows: list[np.ndarray] = []
    kept_subjects: list[str] = []
    wavenumbers = np.asarray([float(name) for name in payload["feature_names"].astype(str)], dtype=float)
    for sid in disease_subjects:
        mask = payload["subject_ids"].astype(str) == str(sid)
        if not mask.any():
            continue
        rows.append(np.nanmedian(np.asarray(payload["area_summary"][mask], dtype=float), axis=0))
        kept_subjects.append(str(sid))
    if not rows:
        return np.array([]), np.empty((0, 0)), pd.DataFrame()
    mean_spectra = np.vstack(rows)
    clinical = clinical_df.copy()
    clinical["subject_id"] = clinical["subject_id"].astype(str)
    clinical_aligned = (
        pd.DataFrame({"subject_id": kept_subjects})
        .merge(clinical, on="subject_id", how="left")
        .set_index("subject_id")
    )

    return wavenumbers, mean_spectra, clinical_aligned


def compute_intra_disease_correlations(
    mean_spectra: np.ndarray,
    wavenumbers: np.ndarray,
    clinical_aligned: pd.DataFrame,
    targets: tuple[str, ...] = _DEFAULT_TARGETS,
    fdr_alpha: float = 0.05,
) -> pd.DataFrame:
    """Per-wavenumber Spearman correlation against each clinical target.

    Parameters
    ----------
    mean_spectra : np.ndarray
        2D array (n_disease_subjects, n_wavenumbers).
    wavenumbers : np.ndarray
        1D wavenumber axis matching ``mean_spectra`` columns.
    clinical_aligned : pd.DataFrame
        Row-aligned clinical table (same row order as ``mean_spectra``).
    targets : tuple[str, ...]
        Clinical column names to correlate against.
    fdr_alpha : float
        Significance threshold on BH-FDR q-value.

    Returns
    -------
    pd.DataFrame
        Columns: ``target``, ``wavenumber``, ``rho``, ``p_value``, ``fdr_q``,
        ``significant``, ``n``.
    """
    n_wave = mean_spectra.shape[1]
    all_rows: list[dict] = []

    for target in targets:
        if target not in clinical_aligned.columns:
            logger.warning("Target '%s' not found in clinical table; skipping", target)
            continue

        values = pd.to_numeric(clinical_aligned[target], errors="coerce").to_numpy()
        valid = ~np.isnan(values)
        n_valid = int(valid.sum())

        if n_valid < 5:
            logger.warning(
                "Target '%s': only %d valid subjects, skipping correlation", target, n_valid
            )
            continue

        rho_arr = np.empty(n_wave)
        p_arr = np.empty(n_wave)
        for i in range(n_wave):
            rho_arr[i], p_arr[i] = scipy_stats.spearmanr(
                mean_spectra[valid, i], values[valid]
            )

        fdr_q, significant = fdr_correction(p_arr, alpha=fdr_alpha)

        for i, w in enumerate(wavenumbers):
            all_rows.append(
                {
                    "target": target,
                    "wavenumber": float(w),
                    "rho": float(rho_arr[i]),
                    "p_value": float(p_arr[i]),
                    "fdr_q": float(fdr_q[i]),
                    "significant": bool(significant[i]),
                    "n": n_valid,
                }
            )

    return pd.DataFrame(all_rows)


def summarize_significant_targets(
    correlations: pd.DataFrame,
) -> pd.DataFrame:
    """Count significant wavenumbers per target and report the strongest hit.

    Returns
    -------
    pd.DataFrame
        Columns: ``target``, ``n_significant``, ``n_tested``,
        ``max_abs_rho``, ``wavenumber_at_max``.
    """
    if correlations.empty:
        return pd.DataFrame(
            columns=["target", "n_significant", "n_tested", "max_abs_rho", "wavenumber_at_max"]
        )

    rows = []
    for target, grp in correlations.groupby("target"):
        idx_max = grp["rho"].abs().idxmax()
        rows.append(
            {
                "target": target,
                "n_significant": int(grp["significant"].sum()),
                "n_tested": len(grp),
                "max_abs_rho": float(grp.loc[idx_max, "rho"]),
                "wavenumber_at_max": float(grp.loc[idx_max, "wavenumber"]),
            }
        )
    return pd.DataFrame(rows)


def summarize_correlation_regions(
    correlations: pd.DataFrame,
    *,
    spectral_range: str,
    biological_regions: pd.DataFrame | None = None,
    abs_quantile: float = 0.9,
    min_width_cm: float = 8.0,
    merge_gap_cm: float = 6.0,
) -> pd.DataFrame:
    """Merge adjacent high-absolute-rho intervals into region-level summaries."""
    if correlations.empty:
        return pd.DataFrame(
            columns=[
                "target",
                "spectral_range",
                "region_id",
                "start_cm-1",
                "end_cm-1",
                "width_cm-1",
                "integrated_abs_rho",
                "max_abs_rho",
                "mean_abs_rho",
                "min_fdr_q",
                "n_significant_wavenumbers",
                "has_fdr_significant",
                "overlaps_biological_pca_region",
                "overlapping_pc",
                "overlap_fraction",
                "overlap_region_id",
            ]
        )

    rows: list[dict[str, object]] = []
    biological_regions = biological_regions.copy() if biological_regions is not None else pd.DataFrame()
    if not biological_regions.empty and "spectral_block" in biological_regions.columns:
        biological_regions = biological_regions[biological_regions["spectral_block"].astype(str) == str(spectral_range)].copy()

    for target, grp in correlations.groupby("target", sort=False):
        ordered = grp.sort_values("wavenumber").reset_index(drop=True)
        abs_rho = ordered["rho"].abs().to_numpy(dtype=float)
        if not np.isfinite(abs_rho).any():
            continue
        threshold = float(np.nanquantile(abs_rho, abs_quantile))
        active = abs_rho >= threshold
        wavenumbers = ordered["wavenumber"].to_numpy(dtype=float)
        starts: list[int] = []
        stops: list[int] = []
        start: int | None = None
        for idx, is_active in enumerate(active):
            if is_active and start is None:
                start = idx
            elif not is_active and start is not None:
                starts.append(start)
                stops.append(idx - 1)
                start = None
        if start is not None:
            starts.append(start)
            stops.append(len(active) - 1)

        merged: list[list[int]] = []
        for left, right in zip(starts, stops):
            if not merged:
                merged.append([left, right])
                continue
            gap = float(wavenumbers[left] - wavenumbers[merged[-1][1]])
            if gap <= merge_gap_cm:
                merged[-1][1] = right
            else:
                merged.append([left, right])

        region_counter = 0
        for left, right in merged:
            start_cm = float(wavenumbers[left])
            end_cm = float(wavenumbers[right])
            width = end_cm - start_cm
            if width < min_width_cm:
                continue
            region_counter += 1
            window = ordered.iloc[left : right + 1].copy()
            overlap_flag = False
            overlapping_pc = None
            overlap_fraction = 0.0
            overlap_region_id = None
            if not biological_regions.empty:
                overlaps = biological_regions[
                    (biological_regions["start_cm-1"].astype(float) <= end_cm)
                    & (biological_regions["end_cm-1"].astype(float) >= start_cm)
                ].copy()
                if not overlaps.empty:
                    overlaps["overlap_width"] = overlaps.apply(
                        lambda row: max(
                            0.0,
                            min(end_cm, float(row["end_cm-1"])) - max(start_cm, float(row["start_cm-1"])),
                        ),
                        axis=1,
                    )
                    best = overlaps.sort_values("overlap_width", ascending=False).iloc[0]
                    overlap_width = float(best["overlap_width"])
                    overlap_flag = overlap_width > 0
                    overlap_fraction = overlap_width / max(width, 1e-9)
                    overlapping_pc = best.get("PC")
                    overlap_region_id = best.get("region_id")

            rows.append(
                {
                    "target": target,
                    "spectral_range": spectral_range,
                    "region_id": f"{target}_R{region_counter}",
                    "start_cm-1": start_cm,
                    "end_cm-1": end_cm,
                    "width_cm-1": width,
                    "integrated_abs_rho": float(np.trapezoid(window["rho"].abs().to_numpy(dtype=float), window["wavenumber"].to_numpy(dtype=float))),
                    "max_abs_rho": float(window["rho"].abs().max()),
                    "mean_abs_rho": float(window["rho"].abs().mean()),
                    "min_fdr_q": float(window["fdr_q"].min()),
                    "n_significant_wavenumbers": int(window["significant"].sum()),
                    "has_fdr_significant": bool(window["significant"].any()),
                    "overlaps_biological_pca_region": overlap_flag,
                    "overlapping_pc": overlapping_pc,
                    "overlap_fraction": overlap_fraction,
                    "overlap_region_id": overlap_region_id,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["target", "integrated_abs_rho"], ascending=[True, False]).reset_index(drop=True)


def run_intra_disease_pca(
    mean_spectra: np.ndarray,
    n_components: int = 10,
    random_state: int = 42,
    *,
    warn_on_high_pc1: bool = True,
) -> dict:
    """Run PCA specifically on disease-only subjects.

    Parameters
    ----------
    mean_spectra : np.ndarray
        2D array (n_disease_subjects, n_wavenumbers) of subject means.
    n_components : int
        Number of principal components to compute.
    random_state : int
        Random seed.

    Returns
    -------
    dict
        PCA results containing 'scores', 'loadings', 'explained_variance_ratio', etc.
    """
    from src.analysis.statistics import run_pca

    if mean_spectra.size == 0:
        return {}

    return run_pca(
        mean_spectra,
        n_components=min(n_components, mean_spectra.shape[0] - 1),
        random_state=random_state,
        scale=False,
        warn_on_high_pc1=warn_on_high_pc1,
    )
