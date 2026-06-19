"""Subject-level UMAP + continuous clinical overlays for Raman EDA.

Aggregates per-subject mean SNV spectra, fits a UMAP embedding at subject
level (N ≈ tens), clusters with HDBSCAN, and attaches clinical annotations
(MoCA, age, gender, condition) including continuous viridis overlays and
longitudinal trajectory arrows.

Subject-level semantics respect the CLAUDE.md N/3 guard for n_neighbors.
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_mutual_info_score

from src import config
from src.preprocessing.cache import load_subject_pixels
from src.preprocessing.pipeline import apply_region_weights, resolve_region_key

logger = logging.getLogger(__name__)

_MOCA_COLS = ("moca_baseline_total", "moca_3_months_total")
_CLINICAL_OVERLAY_COLS = (
    "moca_baseline_total",
    "moca_3_months_total",
    "delta_moca",
    "age_years",
)


# ---------------------------------------------------------------------------
# AMI helper (subject level)
# ---------------------------------------------------------------------------


def _ami_pvalue(
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """AMI + empirical p-value. Noise labels (-1) excluded."""
    valid = labels_pred != -1
    if valid.sum() < 5:
        return np.nan, np.nan
    obs = adjusted_mutual_info_score(labels_true[valid], labels_pred[valid])
    nulls = np.array(
        [
            adjusted_mutual_info_score(rng.permutation(labels_true)[valid], labels_pred[valid])
            for _ in range(n_perm)
        ]
    )
    return float(obs), float((nulls >= obs).mean())


# ---------------------------------------------------------------------------
# Core subject-level UMAP
# ---------------------------------------------------------------------------


def run_subject_umap(
    df_index: pd.DataFrame,
    spectral_range: str,
    clinical_df: Optional[pd.DataFrame] = None,
    random_state: Optional[int] = None,
    wavenumbers: Optional[np.ndarray] = None,
    region_weighting: Optional[dict] = None,
) -> dict:
    """Compute UMAP embedding at subject level from mean SNV spectra.

    Parameters
    ----------
    df_index : pd.DataFrame
        File index with ``subject_id``, ``is_control``, ``gender`` / ``sex``.
    spectral_range : str
        ``'1300'``, ``'2850'``, or ``'both'``.
    clinical_df : pd.DataFrame, optional
        Curated clinical table from ``load_clinical_data``.  Used to attach
        MoCA, age, and other continuous overlay variables.
    random_state : int, optional
        Overrides ``analysis.random_state`` from config.
    wavenumbers : np.ndarray or None
        Optional wavenumber axis; if None, the axis from the first subject is used.
    region_weighting : dict or None
        Optional region-weighting config forwarded to ``apply_region_weights``.

    Returns
    -------
    dict
        ``'embedding'``  – subject-level DataFrame with UMAP1/2 + metadata.
        ``'purity'``     – AMI + p-value per target.
        ``'mean_spectra'`` – subject-mean SNV spectra (n_subjects × n_wavenumbers).
        ``'wavenumbers'`` – 1D wavenumber axis matching the mean_spectra columns.
        ``'umap_model'`` – fitted UMAP reducer.
        ``'hdbscan_model'`` – fitted HDBSCAN.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Tensorflow not installed; ParametricUMAP will be unavailable",
            category=ImportWarning,
        )
        import umap  # local import

    rs = int(
        random_state
        if random_state is not None
        else config.get("analysis", {}).get("random_state", 42)
    )
    rng = np.random.default_rng(rs)

    subj_cfg = (
        config.get("analysis", {})
        .get("advanced", {})
        .get("subject_umap", {})
    )
    n_perm = int(subj_cfg.get("n_permutations", 200))
    umap_metric = subj_cfg.get("umap_metric", "cosine")

    subjects = df_index["subject_id"].unique().tolist()
    if not subjects:
        raise RuntimeError("No subjects in df_index for subject UMAP")

    # -- Build per-subject mean SNV spectra --
    mean_rows: list[np.ndarray] = []
    meta_rows: list[dict] = []
    wavenumbers_out: np.ndarray | None = None

    for sid in subjects:
        rows = df_index[df_index["subject_id"] == sid]
        if rows.empty:
            continue
        gender = (
            str(rows["gender"].iloc[0]).lower()
            if "gender" in rows.columns and pd.notna(rows["gender"].iloc[0])
            else "unknown"
        )
        condition = "control" if bool(rows["is_control"].iloc[0]) else "disease"

        if spectral_range == "both":
            wave_fp, pix_fp = load_subject_pixels(str(sid), "fingerprint")
            wave_ch, pix_ch = load_subject_pixels(str(sid), "ch_stretch")
            if pix_fp.size == 0 or pix_ch.size == 0 or pix_fp.shape[0] < 1 or pix_ch.shape[0] < 1:
                logger.warning("Subject %s: missing cached spectra for one of the regions in both, skipping", sid)
                continue
            mean_fp = np.nanmean(pix_fp, axis=0)
            mean_ch = np.nanmean(pix_ch, axis=0)
            _wave = np.concatenate([wave_fp, wave_ch])
            mean_spec = np.concatenate([mean_fp, mean_ch])
        else:
            region_key = resolve_region_key(spectral_range)
            _wave, pix = load_subject_pixels(str(sid), region_key)
            if pix.size == 0 or pix.shape[0] < 1:
                logger.warning("Subject %s: no cached spectra for range=%s, skipping", sid, spectral_range)
                continue
            mean_spec = np.nanmean(pix, axis=0)

        if wavenumbers_out is None:
            wavenumbers_out = _wave
        elif _wave is not None and _wave.size > 0:
            from src.analysis.spectral_loader import wave_axes_compatible
            if not wave_axes_compatible(wavenumbers_out, _wave):
                logger.warning(
                    "Subject %s: wavenumber axis mismatch (%.2f vs %.2f start); skipping",
                    sid, float(_wave[0]), float(wavenumbers_out[0]),
                )
                continue

        mean_rows.append(mean_spec)
        meta_rows.append(
            {
                "subject_id": str(sid),
                "gender": gender,
                "condition": condition,
            }
        )

    if len(mean_rows) < 4:
        raise RuntimeError(
            f"Too few subjects ({len(mean_rows)}) for subject UMAP — need ≥ 4"
        )

    mean_spectra = np.vstack(mean_rows)
    meta_df = pd.DataFrame(meta_rows)
    N = mean_spectra.shape[0]

    wavenumbers_out = (
        np.asarray(wavenumbers, dtype=float)
        if wavenumbers is not None and np.asarray(wavenumbers).size > 0
        else (wavenumbers_out if wavenumbers_out is not None else np.array([]))
    )

    umap_input = mean_spectra
    if region_weighting is not None and bool(region_weighting.get("enabled", False)):
        if wavenumbers_out is None or wavenumbers_out.size == 0:
            logger.warning("Region weighting requested but wavenumber axis unavailable; skipping")
        else:
            umap_input = apply_region_weights(mean_spectra, wavenumbers_out, region_weighting)

    # -- Merge clinical overlays --
    if clinical_df is not None and not clinical_df.empty:
        clin_sub = _extract_clinical_overlays(clinical_df, meta_df["subject_id"].tolist())
        meta_df = meta_df.merge(clin_sub, on="subject_id", how="left")
    else:
        for col in list(_CLINICAL_OVERLAY_COLS):
            meta_df[col] = np.nan

    # delta_moca derived
    if "moca_baseline_total" in meta_df.columns and "moca_3_months_total" in meta_df.columns:
        meta_df["delta_moca"] = meta_df["moca_3_months_total"] - meta_df["moca_baseline_total"]
    else:
        meta_df["delta_moca"] = np.nan

    # -- UMAP (N/3 guard per CLAUDE.md) --
    n_neighbors = min(15, max(2, N // 3))
    logger.info(
        "Subject UMAP: N=%d subjects, n_neighbors=%d (N/3 guard), metric=%s",
        N, n_neighbors, umap_metric,
    )
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric=umap_metric,
        random_state=rs,
        n_components=2,
        n_jobs=1,
    )
    embedding = reducer.fit_transform(umap_input)

    meta_df["UMAP1"] = embedding[:, 0]
    meta_df["UMAP2"] = embedding[:, 1]

    # -- HDBSCAN --
    min_cluster_size = max(5, max(1, N // 20))
    min_samples = 1
    if N < min_cluster_size:
        logger.warning(
            "Subject UMAP: N=%d < min_cluster_size=%d; skipping HDBSCAN, all labels set to -1 (noise).",
            N, min_cluster_size,
        )
        clusters = np.full(N, -1, dtype=int)
        clusterer = None
    else:
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min(min_samples, N),
            copy=True,
        )
        clusters = clusterer.fit_predict(embedding)
    meta_df["hdbscan_cluster"] = clusters

    # -- AMI purity with permutation null --
    purity_rows = []
    for target in ["condition", "gender", "hdbscan_cluster"]:
        if target == "hdbscan_cluster":
            # Self-consistency check vs condition
            ami, pval = _ami_pvalue(
                meta_df["condition"].to_numpy(), clusters, n_perm, rng
            )
            purity_rows.append(
                {
                    "target": "hdbscan_vs_condition",
                    "ami": ami,
                    "p_value": pval,
                    "n_subjects": N,
                    "n_noise": int((clusters == -1).sum()),
                }
            )
        else:
            ami, pval = _ami_pvalue(
                meta_df[target].to_numpy(), clusters, n_perm, rng
            )
            purity_rows.append(
                {
                    "target": target,
                    "ami": ami,
                    "p_value": pval,
                    "n_subjects": N,
                    "n_noise": int((clusters == -1).sum()),
                }
            )

    purity_df = pd.DataFrame(purity_rows)
    logger.info(
        "Subject UMAP complete: N=%d, clusters=%d (+%d noise). "
        "condition AMI=%.3f",
        N,
        len(set(clusters)) - (1 if -1 in clusters else 0),
        int((clusters == -1).sum()),
        purity_df.loc[purity_df["target"] == "condition", "ami"].values[0]
        if len(purity_df) > 0 else np.nan,
    )

    # mean_spectra as DataFrame for export
    wave_labels = [f"w{i}" for i in range(mean_spectra.shape[1])]
    mean_spectra_df = pd.DataFrame(mean_spectra, columns=wave_labels)
    mean_spectra_df.insert(0, "subject_id", meta_df["subject_id"].values)

    return {
        "embedding": meta_df,
        "purity": purity_df,
        "mean_spectra": mean_spectra_df,
        "wavenumbers": wavenumbers_out if wavenumbers_out is not None else np.array([]),
        "umap_model": reducer,
        "hdbscan_model": clusterer,
    }


# ---------------------------------------------------------------------------
# Longitudinal trajectory helper
# ---------------------------------------------------------------------------


def get_longitudinal_pairs(
    embedding_df: pd.DataFrame,
    min_pairs: int = 20,
) -> Optional[pd.DataFrame]:
    """Return subjects with both baseline and 3-month MoCA, or None if < min_pairs.

    Parameters
    ----------
    embedding_df : pd.DataFrame
        Embedding DataFrame from :func:`run_subject_umap`.
    min_pairs : int
        Minimum number of paired subjects required. Returns None if below threshold.

    Returns
    -------
    pd.DataFrame or None
        Same schema as embedding_df, filtered to subjects with both MoCA timepoints.
    """
    required = ["moca_baseline_total", "moca_3_months_total"]
    if not all(c in embedding_df.columns for c in required):
        return None

    paired = embedding_df.dropna(subset=required)
    n_paired = len(paired)

    if n_paired < min_pairs:
        logger.warning(
            "Longitudinal trajectories skipped: only %d subjects with paired MoCA "
            "(threshold=%d). Caption: 'single-timepoint spectra — descriptive only'.",
            n_paired,
            min_pairs,
        )
        return None

    return paired.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_clinical_overlays(
    clinical_df: pd.DataFrame,
    subject_ids: list[str],
) -> pd.DataFrame:
    """Pull overlay columns from the curated clinical table."""
    src_cols = {
        "moca_baseline_total": ["moca_baseline_total", "moca_baseline", "MOCA baseline Total [TEST]"],
        "moca_3_months_total": ["moca_3_months_total", "moca_3m", "MOCA +3 months Total [TEST]"],
        "age_years": ["age_years", "age"],
    }

    out = pd.DataFrame({"subject_id": [str(s) for s in subject_ids]})
    clin = clinical_df.copy()
    if "subject_id" in clin.columns:
        clin["subject_id"] = clin["subject_id"].astype(str)

    for out_col, candidates in src_cols.items():
        src_col = next((c for c in candidates if c in clin.columns), None)
        if src_col is None:
            out[out_col] = np.nan
            continue
        lookup = clin[["subject_id", src_col]].drop_duplicates("subject_id")
        out = out.merge(lookup.rename(columns={src_col: out_col}), on="subject_id", how="left")

    return out
