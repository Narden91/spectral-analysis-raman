"""Phase R (optional): index-space agglomerative clustering on named indices.

Clusters subjects using only the four pre-registered named indices
(membrane_integrity, oxidative, protein_remodeling, tyrosine_environment).
Evaluates k=2–4 via Gap statistic; reports bootstrap Adjusted Rand Index (ARI)
stability for the selected k.  All clustering is exploratory; results are
never labeled as validated subtypes.

Outputs go to ``06_spectral_interpretability/index_clustering/``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from src import config, get_spectral_interpretability_dir
from src.analysis.io_utils import load_csv_or_empty, write_csv_atomic
from src.analysis.preregistered_panel import panel_cfg

logger = logging.getLogger(__name__)


def _cfg(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    return config if cfg is None else cfg


def _gap_statistic(
    X: np.ndarray,
    k: int,
    *,
    n_refs: int = 20,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Compute gap(k) and its standard error for Ward linkage."""

    def _inertia(data: np.ndarray, k_: int) -> float:
        labels = AgglomerativeClustering(n_clusters=k_, linkage="ward").fit_predict(data)
        total = 0.0
        for cl in np.unique(labels):
            pts = data[labels == cl]
            if len(pts) > 1:
                dists = np.sum((pts - pts.mean(axis=0)) ** 2)
                total += dists
        return float(np.log(total + 1e-12))

    obs = _inertia(X, k)
    refs = np.empty(n_refs)
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    for i in range(n_refs):
        ref = rng.uniform(mins, maxs, size=X.shape)
        refs[i] = _inertia(ref, k)
    gap = float(np.mean(refs)) - obs
    se = float(np.std(refs, ddof=1) * np.sqrt(1.0 + 1.0 / n_refs))
    return gap, se


def _bootstrap_ari(
    X: np.ndarray,
    k: int,
    *,
    n_bootstrap: int = 100,
    rng: np.random.Generator,
) -> float:
    """Mean bootstrap ARI: stability of Ward k-clusters over sub-samples."""
    n = X.shape[0]
    base_labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
    scores: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        labels_boot = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X[idx])
        scores.append(float(adjusted_rand_score(base_labels[idx], labels_boot)))
    return float(np.mean(scores)) if scores else np.nan


def run_index_clustering(
    *,
    panel_matrix_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Phase R orchestrator: Ward clustering on named indices only."""
    cfg = _cfg(cfg)
    panel = panel_cfg(cfg)
    phase_cfg = cfg.get("analysis", {}).get("index_clustering", {})
    if not bool(phase_cfg.get("enabled", False)):
        # Default off (optional phase)
        return {}

    target_dir = (
        Path(output_dir) if output_dir is not None
        else get_spectral_interpretability_dir("index_clustering")
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = (
        Path(panel_matrix_path)
        if panel_matrix_path is not None
        else _default_panel_path()
    )
    frame = load_csv_or_empty(matrix_path)
    if frame.empty:
        logger.warning("Phase R: panel matrix not found at %s; skipping", matrix_path)
        return {}

    index_names = list(panel.get("indices", {}).keys())
    index_cols = [c for c in index_names if c in frame.columns]
    if len(index_cols) < 2:
        logger.warning("Phase R: fewer than 2 index columns available; skipping")
        return {}

    subgroup_cfg = phase_cfg
    random_state = int(subgroup_cfg.get("random_state", panel.get("random_state", 42)))
    max_k = int(subgroup_cfg.get("max_k", 4))
    n_refs = int(subgroup_cfg.get("n_gap_refs", 20))
    n_bootstrap = int(subgroup_cfg.get("n_bootstrap", 100))
    min_subjects_per_group = int(panel.get("min_subjects_per_group", 5))

    rng = np.random.default_rng(random_state)

    X_raw = frame[index_cols].dropna(subset=index_cols).to_numpy(dtype=float)
    valid_idx = frame[index_cols].dropna(subset=index_cols).index
    subject_ids = frame.loc[valid_idx, "subject_id"].tolist() if "subject_id" in frame.columns else [str(i) for i in valid_idx]

    n_subjects = X_raw.shape[0]
    if n_subjects < 2 * min_subjects_per_group:
        logger.warning("Phase R: only %d subjects after dropna — skipping", n_subjects)
        return {}

    X = StandardScaler().fit_transform(X_raw)

    k_range = range(2, min(max_k + 1, n_subjects))
    gap_rows: list[dict[str, object]] = []
    for k in k_range:
        gap, se = _gap_statistic(X, k, n_refs=n_refs, rng=rng)
        gap_rows.append({"k": k, "gap": gap, "se": se})

    gap_df = pd.DataFrame(gap_rows)

    # Select k by gap rule: smallest k such that gap(k) >= gap(k+1) - se(k+1)
    best_k = int(k_range.start)
    for i, row in gap_df.iterrows():
        k_val = int(row["k"])
        if k_val == max(k_range):
            best_k = k_val
            break
        next_row = gap_df.loc[i + 1] if i + 1 in gap_df.index else None
        if next_row is None:
            best_k = k_val
            break
        if float(row["gap"]) >= float(next_row["gap"]) - float(next_row["se"]):
            best_k = k_val
            break

    logger.info("Phase R: selected k=%d via gap statistic", best_k)

    labels = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit_predict(X)
    ari = _bootstrap_ari(X, best_k, n_bootstrap=n_bootstrap, rng=rng)

    assignments = pd.DataFrame({"subject_id": subject_ids, "index_cluster": labels.tolist()})
    assignments["index_cluster"] = assignments["index_cluster"].astype(str)

    # Cluster summary: mean indices per cluster
    summary_data = pd.DataFrame(X_raw, columns=index_cols)
    summary_data["index_cluster"] = labels.tolist()
    cluster_summary = summary_data.groupby("index_cluster")[index_cols].mean().reset_index()
    cluster_summary.insert(0, "n_subjects", summary_data.groupby("index_cluster").size().values)
    cluster_summary.insert(1, "bootstrap_ari", ari)

    assignments_path = target_dir / "index_cluster_assignments.csv"
    write_csv_atomic(assignments, assignments_path, index=False)

    summary_path = target_dir / "index_cluster_summary.csv"
    write_csv_atomic(cluster_summary, summary_path, index=False)

    gap_path = target_dir / "index_gap_statistic.csv"
    write_csv_atomic(gap_df, gap_path, index=False)

    logger.info(
        "Phase R: k=%d clusters, bootstrap ARI=%.3f. Outputs: %s",
        best_k, ari, target_dir,
    )
    return {
        "index_cluster_assignments": assignments_path,
        "index_cluster_summary": summary_path,
        "index_gap_statistic": gap_path,
    }


def _default_panel_path() -> Path:
    from src import get_confirmatory_panel_dir
    return get_confirmatory_panel_dir() / "panel_feature_matrix.csv"
