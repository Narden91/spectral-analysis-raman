from __future__ import annotations

import hashlib
import logging
import warnings

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy

from src import config

logger = logging.getLogger(__name__)


def extract_ch_lipid_state_features(
    area_df: pd.DataFrame,
    train_subject_ids: list[str],
    test_subject_ids: list[str],
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Track G: CH-stretch lipid-state features via UMAP + MiniBatchKMeans.

    Fit train-only: UMAP on CH pixels → MiniBatchKMeans(n_clusters).
    Transform test CH pixels through learned models.
    Aggregate per subject: cluster proportions, entropy, max state share,
    distance-based noise proxy.

    Parameters
    ----------
    area_df
        Area-level DataFrame with ``subject_id`` and ``ch_pixels`` columns.
    train_subject_ids
        Subject IDs in the training fold.
    test_subject_ids
        Subject IDs in the held-out fold.
    random_state
        Seed for reproducibility.

    Returns
    -------
    train_features, test_features, audit_info
    """
    ch_cfg = config.get("ch_lipid_state", {})
    n_clusters = int(ch_cfg.get("n_clusters", 8))
    max_pixels = int(ch_cfg.get("max_pixels_per_subject", 50))
    umap_metric = str(ch_cfg.get("umap_metric", "cosine"))
    noise_q = float(ch_cfg.get("noise_quantile", 0.95))

    if "ch_pixels" not in area_df.columns:
        return pd.DataFrame(), pd.DataFrame(), {"status": "skipped", "reason": "no_ch_pixels_column"}

    subject_pixels = _index_subject_pixels(area_df)
    train_pixels, train_subj_labels = _collect_pixels(subject_pixels, train_subject_ids, max_pixels)
    test_pixels, test_subj_labels = _collect_pixels(subject_pixels, test_subject_ids, max_pixels)

    if train_pixels.shape[0] < n_clusters * 2 or train_pixels.shape[1] < 2:
        return (
            _empty_fallback(train_subject_ids, n_clusters),
            _empty_fallback(test_subject_ids, n_clusters),
            {"status": "skipped", "reason": f"insufficient_train_pixels_{train_pixels.shape[0]}"},
        )

    try:
        from sklearn.cluster import MiniBatchKMeans
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Tensorflow not installed; ParametricUMAP will be unavailable",
                category=ImportWarning,
            )
            from umap import UMAP
    except ImportError as exc:
        logger.warning("UMAP not available for Track G: %s", exc)
        return (
            _empty_fallback(train_subject_ids, n_clusters),
            _empty_fallback(test_subject_ids, n_clusters),
            {"status": "skipped", "reason": f"import_error:{exc}"},
        )

    umap = UMAP(
        n_components=2,
        metric=umap_metric,
        random_state=random_state,
        n_jobs=1,
    )
    train_emb = umap.fit_transform(train_pixels)

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=3,
        batch_size=min(256, max(16, train_emb.shape[0] // 4)),
    )
    train_labels = kmeans.fit_predict(train_emb)
    train_features = _aggregate_per_subject(train_labels, train_subj_labels, n_clusters, train_emb, noise_q)
    train_features = _ensure_subject_feature_rows(train_features, train_subject_ids, n_clusters)

    if test_pixels.shape[0] == 0:
        test_features = _empty_fallback(test_subject_ids, n_clusters)
    else:
        test_emb = umap.transform(test_pixels)
        test_labels = kmeans.predict(test_emb)
        test_features = _aggregate_per_subject(test_labels, test_subj_labels, n_clusters, test_emb, noise_q)
        test_features = _ensure_subject_feature_rows(test_features, test_subject_ids, n_clusters)

    audit = {
        "status": "ok",
        "n_train_pixels": int(train_pixels.shape[0]),
        "n_test_pixels": int(test_pixels.shape[0]),
        "n_clusters": n_clusters,
        "n_train_subjects": len(train_subject_ids),
        "n_test_subjects": len(test_subject_ids),
        "umap_metric": umap_metric,
    }

    return train_features, test_features, audit


def _collect_pixels(
    subject_pixels: dict[str, list[object]],
    subject_ids: list[str],
    max_pixels: int,
) -> tuple[np.ndarray, np.ndarray]:
    pixels_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    for sid in subject_ids:
        subject_id = str(sid)
        for arr in subject_pixels.get(subject_id, []):
            if arr is None or (isinstance(arr, np.ndarray) and arr.size == 0):
                continue
            arr = np.asarray(arr, dtype=float)
            if arr.ndim != 2 or arr.shape[1] < 2:
                continue
            n_keep = min(arr.shape[0], max_pixels)
            if n_keep < arr.shape[0]:
                seed = _subject_seed(subject_id)
                rng = np.random.default_rng(seed)
                idx = rng.choice(arr.shape[0], size=n_keep, replace=False)
                arr = arr[idx]
            pixels_list.append(arr)
            labels_list.append(np.full(arr.shape[0], subject_id, dtype=object))
    if not pixels_list:
        return np.empty((0, 0), dtype=float), np.empty((0,), dtype=object)
    return np.vstack(pixels_list), np.concatenate(labels_list)


def _index_subject_pixels(area_df: pd.DataFrame) -> dict[str, list[object]]:
    subject_pixels: dict[str, list[object]] = {}
    for subject_id, pixels in zip(area_df["subject_id"].astype(str), area_df["ch_pixels"], strict=False):
        subject_pixels.setdefault(subject_id, []).append(pixels)
    return subject_pixels


def _subject_seed(subject_id: str) -> int:
    return int(hashlib.md5(subject_id.encode()).hexdigest()[:8], 16) % (2**31)


def _aggregate_per_subject(
    cluster_labels: np.ndarray,
    subject_labels: np.ndarray,
    n_clusters: int,
    embedding: np.ndarray,
    noise_quantile: float = 0.95,
) -> pd.DataFrame:
    rows: list[dict] = []
    noise_pct = noise_quantile * 100.0
    cluster_centers = _cluster_centers(embedding, cluster_labels, n_clusters)
    for sid in np.unique(subject_labels):
        mask = subject_labels == sid
        labels_subj = cluster_labels[mask]
        n_pix = len(labels_subj)
        counts = np.bincount(labels_subj, minlength=n_clusters).astype(float)
        props = counts / max(n_pix, 1)
        ent = float(scipy_entropy(props + 1e-12))
        max_share = float(np.max(props))
        max_cluster = int(np.argmax(counts))
        distances = (
            np.linalg.norm(embedding[mask] - cluster_centers[max_cluster], axis=1)
            if n_pix > 0 and embedding.shape[1] >= 2
            else np.array([], dtype=float)
        )
        noise_dist = _safe_percentile(distances, noise_pct)

        row = {
            "subject_id": str(sid),
            "ch_entropy": ent,
            "ch_max_share": max_share,
            "ch_max_cluster": max_cluster,
            "ch_noise_dist": noise_dist,
        }
        for c in range(n_clusters):
            row[f"ch_prop_{c}"] = float(props[c])
        rows.append(row)
    return pd.DataFrame(rows)


def _cluster_centers(
    embedding: np.ndarray,
    cluster_labels: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    centers = np.zeros((n_clusters, embedding.shape[1]), dtype=float)
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        if np.any(mask):
            centers[cluster_id] = np.mean(embedding[mask], axis=0)
    return centers


def _safe_percentile(values: np.ndarray, percentile: float) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    result = float(np.percentile(finite, percentile))
    return result if np.isfinite(result) else 0.0


def _feature_columns(n_clusters: int) -> list[str]:
    return [
        "ch_entropy",
        "ch_max_share",
        "ch_max_cluster",
        "ch_noise_dist",
        *[f"ch_prop_{c}" for c in range(n_clusters)],
    ]


def _default_feature_row(subject_id: str, n_clusters: int) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {
        "subject_id": str(subject_id),
        "ch_entropy": 0.0,
        "ch_max_share": 0.0,
        "ch_max_cluster": -1,
        "ch_noise_dist": 0.0,
    }
    for c in range(n_clusters):
        row[f"ch_prop_{c}"] = 0.0
    return row


def _empty_fallback(subject_ids: list[str], n_clusters: int) -> pd.DataFrame:
    rows = [_default_feature_row(str(s), n_clusters) for s in subject_ids]
    return pd.DataFrame(rows, columns=["subject_id", *_feature_columns(n_clusters)])


def _ensure_subject_feature_rows(
    features: pd.DataFrame,
    subject_ids: list[str],
    n_clusters: int,
) -> pd.DataFrame:
    defaults = _empty_fallback(subject_ids, n_clusters).set_index("subject_id")
    if features.empty or "subject_id" not in features.columns:
        return defaults.reset_index()

    actual = features.copy()
    actual["subject_id"] = actual["subject_id"].astype(str)
    actual = actual.drop_duplicates(subset=["subject_id"], keep="first").set_index("subject_id")
    for col in _feature_columns(n_clusters):
        if col not in actual.columns:
            actual[col] = np.nan

    defaults.update(actual[_feature_columns(n_clusters)])
    return defaults.reset_index()
