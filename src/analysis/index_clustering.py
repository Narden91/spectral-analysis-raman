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

from src import config, get_confirmatory_panel_dir, get_spectral_interpretability_dir
from src.analysis.io_utils import load_csv_or_empty, write_csv_atomic
from src.analysis.preregistered_panel import panel_cfg

logger = logging.getLogger(__name__)


def _ward_inertia(data: np.ndarray, k: int) -> float:
    labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(data)
    return float(np.log(sum(
        np.sum((data[labels == cl] - data[labels == cl].mean(axis=0)) ** 2)
        for cl in np.unique(labels)
        if (labels == cl).sum() > 1
    ) + 1e-12))


def _gap_statistic(
    X: np.ndarray,
    k: int,
    *,
    n_refs: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Return (gap, standard_error) for Ward linkage at k clusters."""
    obs = _ward_inertia(X, k)
    refs = np.array([
        _ward_inertia(rng.uniform(X.min(axis=0), X.max(axis=0), size=X.shape), k)
        for _ in range(n_refs)
    ])
    gap = float(np.mean(refs)) - obs
    se = float(np.std(refs, ddof=1) * np.sqrt(1.0 + 1.0 / n_refs))
    return gap, se


def _bootstrap_ari(
    X: np.ndarray,
    k: int,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> float:
    """Mean bootstrap ARI: stability of Ward k-clusters over sub-samples."""
    n = X.shape[0]
    base_labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
    scores: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X[idx])
        scores.append(float(adjusted_rand_score(base_labels[idx], boot_labels)))
    return float(np.mean(scores)) if scores else np.nan


def _select_best_k(gap_df: pd.DataFrame) -> int:
    """Return smallest k with gap(k) >= gap(k+1) - se(k+1), else last k."""
    for i in range(len(gap_df) - 1):
        row, next_row = gap_df.iloc[i], gap_df.iloc[i + 1]
        if float(row["gap"]) >= float(next_row["gap"]) - float(next_row["se"]):
            return int(row["k"])
    return int(gap_df.iloc[-1]["k"])


def run_index_clustering(
    *,
    panel_matrix_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Phase R orchestrator: Ward clustering on named indices only."""
    cfg = config if cfg is None else cfg
    panel = panel_cfg(cfg)
    phase_cfg = cfg.get("analysis", {}).get("index_clustering", {})
    if not bool(phase_cfg.get("enabled", False)):
        return {}

    target_dir = (
        Path(output_dir) if output_dir is not None
        else get_spectral_interpretability_dir("index_clustering")
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    resolved_matrix = (
        Path(panel_matrix_path) if panel_matrix_path is not None
        else get_confirmatory_panel_dir() / "panel_feature_matrix.csv"
    )
    frame = load_csv_or_empty(resolved_matrix)
    if frame.empty:
        logger.warning("Phase R: panel matrix not found at %s; skipping", resolved_matrix)
        return {}

    index_cols = [c for c in panel.get("indices", {}).keys() if c in frame.columns]
    if len(index_cols) < 2:
        logger.warning("Phase R: fewer than 2 index columns available; skipping")
        return {}

    random_state = int(phase_cfg.get("random_state", panel.get("random_state", 42)))
    max_k = int(phase_cfg.get("max_k", 4))
    n_refs = int(phase_cfg.get("n_gap_refs", 20))
    n_bootstrap = int(phase_cfg.get("n_bootstrap", 100))
    min_subjects_per_group = int(panel.get("min_subjects_per_group", 5))

    valid_frame = frame[index_cols].dropna()
    X_raw = valid_frame.to_numpy(dtype=float)
    subject_ids = (
        frame.loc[valid_frame.index, "subject_id"].tolist()
        if "subject_id" in frame.columns
        else [str(i) for i in valid_frame.index]
    )

    if X_raw.shape[0] < 2 * min_subjects_per_group:
        logger.warning("Phase R: only %d subjects after dropna — skipping", X_raw.shape[0])
        return {}

    rng = np.random.default_rng(random_state)
    X = StandardScaler().fit_transform(X_raw)
    k_range = range(2, min(max_k + 1, X_raw.shape[0]))

    gap_df = pd.DataFrame([
        {"k": k, **dict(zip(["gap", "se"], _gap_statistic(X, k, n_refs=n_refs, rng=rng)))}
        for k in k_range
    ])
    best_k = _select_best_k(gap_df)
    logger.info("Phase R: selected k=%d via gap statistic", best_k)

    labels = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit_predict(X)
    ari = _bootstrap_ari(X, best_k, n_bootstrap=n_bootstrap, rng=rng)

    assignments = pd.DataFrame({"subject_id": subject_ids, "index_cluster": labels.astype(str).tolist()})

    summary_data = pd.DataFrame(X_raw, columns=index_cols)
    summary_data["index_cluster"] = labels
    cluster_summary = summary_data.groupby("index_cluster")[index_cols].mean().reset_index()
    cluster_summary.insert(0, "n_subjects", summary_data.groupby("index_cluster").size().values)
    cluster_summary.insert(1, "bootstrap_ari", ari)

    assignments_path = target_dir / "index_cluster_assignments.csv"
    write_csv_atomic(assignments, assignments_path, index=False)
    summary_path = target_dir / "index_cluster_summary.csv"
    write_csv_atomic(cluster_summary, summary_path, index=False)
    gap_path = target_dir / "index_gap_statistic.csv"
    write_csv_atomic(gap_df, gap_path, index=False)

    logger.info("Phase R: k=%d clusters, bootstrap ARI=%.3f. Outputs: %s", best_k, ari, target_dir)
    return {
        "index_cluster_assignments": assignments_path,
        "index_cluster_summary": summary_path,
        "index_gap_statistic": gap_path,
    }
