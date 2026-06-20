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
from src.data_preparation.store import get_active_data_source
from src.preprocessing.cache import load_subject_mean
from src.preprocessing.pipeline import resolve_region_key

logger = logging.getLogger(__name__)

_DEFAULT_TARGETS = (
    "moca_baseline_total",
    "moca_3_months_total",
    "delta_moca",
    "rbans_total_3_months",
    "aspects_scale_value",
    "age_years",
)


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
    active_source = get_active_data_source()
    if active_source is not None and active_source.source_name != "raw_text":
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

    region_key = resolve_region_key(spectral_range)
    disease_subjects = (
        df_index.loc[~df_index["is_control"].astype(bool), "subject_id"].unique().tolist()
    )

    rows: list[np.ndarray] = []
    kept_subjects: list[str] = []
    wavenumbers: np.ndarray | None = None

    for sid in disease_subjects:
        wave, mean_spec = load_subject_mean(str(sid), region_key)
        if mean_spec.size == 0:
            continue
        if wavenumbers is None:
            wavenumbers = wave
        rows.append(mean_spec)
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


def run_intra_disease_pca(
    mean_spectra: np.ndarray,
    n_components: int = 10,
    random_state: int = 42,
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
    )
