"""Cluster interpretation and wavenumber driver analysis for Raman EDA.

Implements:
- Per-cluster descriptive profiles.
- Cluster mean spectra (subject-aggregated to avoid pseudoreplication).
- One-vs-rest Welch t + BH-FDR + 200-permutation null at subject level.
- UMAP-axis Spearman correlations with wavenumbers + BH-FDR.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from src.analysis.statistics import fdr_correction

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def profile_clusters(
    embedding_df: pd.DataFrame,
) -> pd.DataFrame:
    """Descriptive statistics per HDBSCAN cluster.

    Parameters
    ----------
    embedding_df : pd.DataFrame
        Subject embedding from ``run_subject_umap``.  Must have
        ``hdbscan_cluster``, ``condition``, ``gender`` columns.

    Returns
    -------
    pd.DataFrame
        One row per cluster; columns: ``cluster``, ``n_subjects``,
        ``n_disease``, ``n_control``, ``pct_female``, ``mean_UMAP1``,
        ``mean_UMAP2``.
    """
    rows = []
    for cluster_id, grp in embedding_df.groupby("hdbscan_cluster"):
        n = len(grp)
        n_disease = int((grp["condition"] == "disease").sum())
        n_control = int((grp["condition"] == "control").sum())
        n_female = int((grp["gender"] == "female").sum()) if "gender" in grp.columns else np.nan
        rows.append(
            {
                "cluster": int(cluster_id),
                "n_subjects": n,
                "n_disease": n_disease,
                "n_control": n_control,
                "pct_female": round(n_female / n * 100, 1) if n > 0 else np.nan,
                "mean_UMAP1": float(grp["UMAP1"].mean()),
                "mean_UMAP2": float(grp["UMAP2"].mean()),
            }
        )
    return pd.DataFrame(rows)


def cluster_mean_spectra(
    mean_spectra_df: pd.DataFrame,
    embedding_df: pd.DataFrame,
    wavenumbers: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Compute per-cluster mean spectra from subject-mean spectra.

    Aggregation is at subject level, so no pseudoreplication.

    Parameters
    ----------
    mean_spectra_df : pd.DataFrame
        Output of ``run_subject_umap``'s ``'mean_spectra'`` field.
        First column is ``subject_id``; rest are spectral channels.
    embedding_df : pd.DataFrame
        Must have ``subject_id``, ``hdbscan_cluster``.
    wavenumbers : np.ndarray, optional
        1D wavenumber axis for the spectral columns.  If None, integer indices
        are used as column names.

    Returns
    -------
    pd.DataFrame
        Rows = clusters, columns = [``cluster``] + wavenumber labels.
    """
    spec_cols = [c for c in mean_spectra_df.columns if c != "subject_id"]
    spectra = mean_spectra_df.set_index("subject_id")[spec_cols]
    cluster_map = embedding_df.set_index("subject_id")["hdbscan_cluster"]
    cluster_series = spectra.index.map(cluster_map)

    rows = []
    for cid in np.unique(cluster_series):
        if cid == -1:
            continue
        grp = spectra[cluster_series == cid]
        mean_vec = grp[spec_cols].mean(axis=0).values
        row = {"cluster": int(cid)}
        if wavenumbers is not None:
            for i, w in enumerate(wavenumbers):
                row[f"{w:.2f}"] = float(mean_vec[i])
        else:
            for i, col in enumerate(spec_cols):
                row[col] = float(mean_vec[i])
        rows.append(row)

    return pd.DataFrame(rows)


def cluster_driver_wavenumbers(
    mean_spectra_df: pd.DataFrame,
    embedding_df: pd.DataFrame,
    wavenumbers: np.ndarray,
    n_perm: int = 200,
    fdr_alpha: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """One-vs-rest Welch t per cluster per wavenumber, BH-FDR + permutation null.

    Parameters
    ----------
    mean_spectra_df : pd.DataFrame
        Subject-mean spectra from ``run_subject_umap``.
    embedding_df : pd.DataFrame
        Must have ``subject_id``, ``hdbscan_cluster``.
    wavenumbers : np.ndarray
        1D wavenumber axis.
    n_perm : int
        Permutations for cluster-label shuffle null.
    fdr_alpha : float
        Significance threshold on BH-FDR q-value.

    Returns
    -------
    pd.DataFrame
        Columns: ``cluster``, ``wavenumber``, ``t_stat``, ``p_value``,
        ``fdr_q``, ``significant``, ``perm_p``.
    """
    rng = np.random.default_rng(random_state)

    spec_cols = [c for c in mean_spectra_df.columns if c != "subject_id"]
    spectra = mean_spectra_df.set_index("subject_id")[spec_cols].values
    subject_ids = mean_spectra_df["subject_id"].tolist()

    cluster_map = embedding_df.set_index("subject_id")["hdbscan_cluster"]
    clusters = np.array([cluster_map.get(sid, -1) for sid in subject_ids])

    unique_clusters = [c for c in np.unique(clusters) if c != -1]

    all_rows: list[dict] = []

    for cid in unique_clusters:
        in_mask = clusters == cid
        out_mask = (clusters != cid) & (clusters != -1)

        if in_mask.sum() < 2 or out_mask.sum() < 2:
            logger.warning("Cluster %d: insufficient subjects for driver test, skipping", cid)
            continue

        in_spectra = spectra[in_mask]
        out_spectra = spectra[out_mask]

        # One-vs-rest Welch t per wavenumber
        t_stats, p_vals = scipy_stats.ttest_ind(
            in_spectra, out_spectra, axis=0, equal_var=False
        )
        fdr_q = fdr_correction(np.asarray(p_vals, dtype=float))[0]

        # Permutation null: max-|t| distribution (subject-level label shuffle)
        null_max_t = np.empty(n_perm)
        in_idx = np.where(in_mask | out_mask)[0]
        labels_pool = clusters[in_idx]
        spectra_pool = spectra[in_idx]

        for perm_i in range(n_perm):
            shuffled = rng.permutation(labels_pool)
            in_m = shuffled == cid
            out_m = (shuffled != cid) & (shuffled != -1)
            if in_m.sum() < 2 or out_m.sum() < 2:
                null_max_t[perm_i] = 0.0
                continue
            t_null, _ = scipy_stats.ttest_ind(
                spectra_pool[in_m], spectra_pool[out_m], axis=0, equal_var=False
            )
            null_max_t[perm_i] = float(np.max(np.abs(t_null)))

        obs_t_abs = np.abs(t_stats)
        perm_p = np.array([(null_max_t >= t_abs).mean() for t_abs in obs_t_abs])

        for i, w in enumerate(wavenumbers):
            all_rows.append(
                {
                    "cluster": int(cid),
                    "wavenumber": float(w),
                    "t_stat": float(t_stats[i]),
                    "p_value": float(p_vals[i]),
                    "fdr_q": float(fdr_q[i]),
                    "significant": bool(fdr_q[i] < fdr_alpha),
                    "perm_p": float(perm_p[i]),
                }
            )

    return pd.DataFrame(all_rows)


def umap_axis_correlations(
    mean_spectra_df: pd.DataFrame,
    embedding_df: pd.DataFrame,
    wavenumbers: np.ndarray,
    top_k: int = 20,
) -> pd.DataFrame:
    """Spearman correlation of each wavenumber against UMAP1 and UMAP2.

    NOTE: UMAP axes are nonlinear — correlations are exploratory only.
    Computed at subject level (N ≈ 197).

    Parameters
    ----------
    mean_spectra_df : pd.DataFrame
        Subject-mean spectra.
    embedding_df : pd.DataFrame
        Must have ``subject_id``, ``UMAP1``, ``UMAP2``.
    wavenumbers : np.ndarray
        1D wavenumber axis.
    top_k : int
        Stored for downstream reference in the output (informational only).

    Returns
    -------
    pd.DataFrame
        Columns: ``wavenumber``, ``rho_UMAP1``, ``p_UMAP1``, ``fdr_UMAP1``,
        ``rho_UMAP2``, ``p_UMAP2``, ``fdr_UMAP2``.
    """
    spec_cols = [c for c in mean_spectra_df.columns if c != "subject_id"]
    spectra = mean_spectra_df.set_index("subject_id")[spec_cols].values

    emb_sub = embedding_df.set_index("subject_id")[["UMAP1", "UMAP2"]]
    subject_ids = mean_spectra_df["subject_id"].tolist()
    umap1 = np.array([emb_sub.loc[sid, "UMAP1"] if sid in emb_sub.index else np.nan for sid in subject_ids])
    umap2 = np.array([emb_sub.loc[sid, "UMAP2"] if sid in emb_sub.index else np.nan for sid in subject_ids])

    valid = ~(np.isnan(umap1) | np.isnan(umap2))
    if valid.sum() < 5:
        logger.warning("Too few valid subjects for UMAP-axis correlations (%d)", valid.sum())
        return pd.DataFrame(
            columns=["wavenumber", "rho_UMAP1", "p_UMAP1", "fdr_UMAP1", "rho_UMAP2", "p_UMAP2", "fdr_UMAP2"]
        )

    n_wave = spectra.shape[1]
    rho1_arr = np.empty(n_wave)
    p1_arr = np.empty(n_wave)
    rho2_arr = np.empty(n_wave)
    p2_arr = np.empty(n_wave)

    for i in range(n_wave):
        col = spectra[valid, i]
        rho1_arr[i], p1_arr[i] = scipy_stats.spearmanr(col, umap1[valid])
        rho2_arr[i], p2_arr[i] = scipy_stats.spearmanr(col, umap2[valid])

    fdr1 = fdr_correction(np.asarray(p1_arr, dtype=float))[0]
    fdr2 = fdr_correction(np.asarray(p2_arr, dtype=float))[0]

    logger.info(
        "UMAP-axis correlations: %d/%d wavenumbers FDR<0.05 (UMAP1), %d/%d (UMAP2)."
        "  NOTE: UMAP axes are nonlinear — correlations are exploratory.",
        int((fdr1 < 0.05).sum()), n_wave,
        int((fdr2 < 0.05).sum()), n_wave,
    )

    return pd.DataFrame(
        {
            "wavenumber": wavenumbers,
            "rho_UMAP1": rho1_arr,
            "p_UMAP1": p1_arr,
            "fdr_UMAP1": fdr1,
            "rho_UMAP2": rho2_arr,
            "p_UMAP2": p2_arr,
            "fdr_UMAP2": fdr2,
        }
    )


