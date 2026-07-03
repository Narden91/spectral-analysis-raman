"""Phase 12: exploratory disease-only subgroup stability analysis."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from src import config, get_cross_region_phase_dir
from src.analysis.clinical_feature_associations import (
    feature_columns,
    filter_disease_subjects,
    load_disease_feature_frame,
    load_subject_feature_matrix,
)
from src.analysis.discriminant_spectra import run_subgroup_mean_spectra

logger = logging.getLogger(__name__)


def _cfg(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    return config if cfg is None else cfg


def _subgroup_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("analysis", {}).get("disease_subgroups", {})


def _load_integrated_candidates(path: str | Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        logger.warning("Integrated candidate table not found for disease subgroups: %s", path)
        return None
    df = pd.read_csv(path)
    return None if df.empty else df


def _region_feature_names(feature_columns_: list[str], region: pd.Series) -> list[str]:
    block = str(region["spectral_block"])
    lower = float(region["start_cm-1"])
    upper = float(region["end_cm-1"])
    out: list[str] = []
    for column in feature_columns_:
        if not column.startswith(f"{block}__"):
            continue
        try:
            wavenumber = float(column.split("__", 1)[1])
        except ValueError:
            continue
        if lower <= wavenumber <= upper:
            out.append(column)
    return out


def _band_mean(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    lower: float,
    upper: float,
) -> pd.Series:
    block = "1300" if upper < 2000 else "2850"
    selected = []
    for column in columns:
        if not column.startswith(f"{block}__"):
            continue
        try:
            wavenumber = float(column.split("__", 1)[1])
        except ValueError:
            continue
        if lower <= wavenumber <= upper:
            selected.append(column)
    if not selected:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return frame[selected].mean(axis=1)


def _fallback_band_features(
    frame: pd.DataFrame,
    columns: list[str],
    cfg: dict[str, Any],
) -> pd.DataFrame:
    bands_cfg = cfg.get("whole_spectra", {}).get("bands", {})
    features: dict[str, pd.Series] = {}
    for band_name, limits in bands_cfg.items():
        if not isinstance(limits, (list, tuple)) or len(limits) != 2:
            continue
        series = _band_mean(
            frame,
            columns,
            lower=float(limits[0]),
            upper=float(limits[1]),
        )
        if series.notna().any():
            features[f"band__{band_name}"] = series
    return pd.DataFrame(features, index=frame.index)


def _candidate_region_features(
    frame: pd.DataFrame,
    columns: list[str],
    integrated: pd.DataFrame | None,
) -> pd.DataFrame:
    if integrated is None or integrated.empty:
        return pd.DataFrame(index=frame.index)
    if "evidence_level" in integrated:
        regions = integrated.loc[integrated["evidence_level"].isin({"High", "Medium"})].copy()
        if regions.empty:
            regions = integrated.copy()
    else:
        regions = integrated.copy()

    features: dict[str, pd.Series] = {}
    for _, region in regions.iterrows():
        required = {"candidate_region_id", "spectral_block", "start_cm-1", "end_cm-1"}
        if not required.issubset(region.index):
            continue
        selected = _region_feature_names(columns, region)
        if selected:
            features[f"region__{region['candidate_region_id']}"] = frame[selected].mean(axis=1)
    return pd.DataFrame(features, index=frame.index)


def _prepare_feature_matrix(features: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    numeric = features.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(axis=1, how="all")
    if numeric.empty:
        return numeric, np.empty((len(numeric), 0), dtype=float)
    numeric = numeric.fillna(numeric.median(axis=0))
    variances = numeric.var(axis=0)
    numeric = numeric.loc[:, variances.gt(0.0)]
    if numeric.empty:
        return numeric, np.empty((len(numeric), 0), dtype=float)
    X = StandardScaler().fit_transform(numeric.to_numpy(dtype=float))
    return numeric, X


def _cluster_labels(X: np.ndarray, n_clusters: int) -> np.ndarray:
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    return model.fit_predict(X)


def _within_cluster_dispersion(X: np.ndarray, labels: np.ndarray) -> float:
    dispersion = 0.0
    for label in np.unique(labels):
        cluster = X[labels == label]
        if len(cluster) == 0:
            continue
        centroid = cluster.mean(axis=0)
        dispersion += float(np.sum((cluster - centroid) ** 2))
    return max(dispersion, np.finfo(float).tiny)


def _gap_statistic(
    X: np.ndarray,
    *,
    k_values: list[int],
    n_bootstrap: int,
    random_state: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    rows: list[dict[str, object]] = []
    for k in k_values:
        labels = _cluster_labels(X, k)
        observed = _within_cluster_dispersion(X, labels)
        reference_logs: list[float] = []
        for _ in range(n_bootstrap):
            reference = rng.uniform(mins, maxs, size=X.shape)
            reference_labels = _cluster_labels(reference, k)
            reference_logs.append(float(np.log(_within_cluster_dispersion(reference, reference_labels))))
        reference_arr = np.array(reference_logs, dtype=float)
        rows.append(
            {
                "n_clusters": k,
                "observed_log_dispersion": float(np.log(observed)),
                "reference_log_dispersion_mean": float(np.mean(reference_arr)),
                "reference_log_dispersion_sd": float(np.std(reference_arr, ddof=1)) if len(reference_arr) > 1 else 0.0,
                "gap": float(np.mean(reference_arr) - np.log(observed)),
                "n_bootstrap": n_bootstrap,
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_ari(
    X: np.ndarray,
    baseline_labels: np.ndarray,
    *,
    n_clusters: int,
    n_bootstrap: int,
    random_state: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    scores: list[float] = []
    n_samples = X.shape[0]
    for _ in range(n_bootstrap):
        sampled = rng.integers(0, n_samples, size=n_samples)
        unique_idx = np.unique(sampled)
        if len(unique_idx) < max(n_clusters, 3):
            continue
        labels = _cluster_labels(X[unique_idx], n_clusters)
        scores.append(float(adjusted_rand_score(baseline_labels[unique_idx], labels)))

    score_arr = np.array(scores, dtype=float)
    return pd.DataFrame(
        [
            {
                "selected_n_clusters": n_clusters,
                "mean_bootstrap_ari": float(np.mean(score_arr)) if len(score_arr) else np.nan,
                "sd_bootstrap_ari": float(np.std(score_arr, ddof=1)) if len(score_arr) > 1 else np.nan,
                "n_bootstrap_requested": n_bootstrap,
                "n_bootstrap_used": int(len(score_arr)),
            }
        ]
    )


def _empty_outputs(target_dir: Path) -> dict[str, Path]:
    assignments_path = target_dir / "disease_subgroup_assignments.csv"
    gap_path = target_dir / "disease_subgroup_gap_statistic.csv"
    stability_path = target_dir / "disease_subgroup_stability.csv"
    feature_path = target_dir / "disease_subgroup_feature_matrix.csv"
    mean_spectra_path = target_dir / "subgroup_mean_spectra.csv"
    pd.DataFrame(columns=["subject_id", "subgroup_label", "n_clusters"]).to_csv(assignments_path, index=False)
    pd.DataFrame(columns=["n_clusters", "observed_log_dispersion", "reference_log_dispersion_mean", "reference_log_dispersion_sd", "gap", "n_bootstrap"]).to_csv(gap_path, index=False)
    pd.DataFrame(columns=["selected_n_clusters", "mean_bootstrap_ari", "sd_bootstrap_ari", "n_bootstrap_requested", "n_bootstrap_used"]).to_csv(stability_path, index=False)
    pd.DataFrame(columns=["subject_id"]).to_csv(feature_path, index=False)
    pd.DataFrame(columns=["block", "subgroup_label", "n_subjects", "wavenumber", "mean_intensity", "std_intensity", "nearest_band"]).to_csv(mean_spectra_path, index=False)
    return {
        "disease_subgroup_assignments": assignments_path,
        "disease_subgroup_gap_statistic": gap_path,
        "disease_subgroup_stability": stability_path,
        "disease_subgroup_feature_matrix": feature_path,
        "subgroup_mean_spectra": mean_spectra_path,
    }


def run_disease_subgroups(
    *,
    feature_matrix_path: str | Path | None = None,
    integrated_candidates_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Run exploratory Ward clustering with gap statistic and bootstrap ARI."""
    cfg = _cfg(cfg)
    subgroup_cfg = _subgroup_cfg(cfg)
    if not bool(subgroup_cfg.get("enabled", True)):
        return {}

    target_dir = (
        Path(output_dir)
        if output_dir is not None
        else get_cross_region_phase_dir("disease_subgroups")
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    disease_only, _feature_meta = load_disease_feature_frame(feature_matrix_path=feature_matrix_path)
    columns = feature_columns(disease_only)
    if integrated_candidates_path is None:
        integrated_candidates_path = get_cross_region_phase_dir("integrated_candidates") / "integrated_candidate_regions.csv"
    integrated = _load_integrated_candidates(integrated_candidates_path)

    pool = _candidate_region_features(disease_only, columns, integrated)
    if pool.empty:
        pool = _fallback_band_features(disease_only, columns, cfg)
    feature_frame, X = _prepare_feature_matrix(pool)

    min_subjects = int(subgroup_cfg.get("min_subjects", 10))
    if len(disease_only) < min_subjects or X.shape[1] == 0:
        logger.info("Disease subgroup analysis has insufficient subjects or features; writing empty outputs")
        return _empty_outputs(target_dir)

    min_clusters = int(subgroup_cfg.get("min_clusters", 2))
    max_clusters = int(subgroup_cfg.get("max_clusters", 4))
    n_bootstrap = int(subgroup_cfg.get("n_bootstrap", 100))
    random_state = int(subgroup_cfg.get("random_state", 42))
    k_values = [
        k
        for k in range(min_clusters, max_clusters + 1)
        if 1 < k < len(disease_only)
    ]
    if not k_values:
        logger.info("Disease subgroup analysis has no valid cluster counts; writing empty outputs")
        return _empty_outputs(target_dir)

    gap = _gap_statistic(
        X,
        k_values=k_values,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )
    selected_k = int(gap.sort_values("gap", ascending=False)["n_clusters"].iloc[0])
    labels = _cluster_labels(X, selected_k)
    stability = _bootstrap_ari(
        X,
        labels,
        n_clusters=selected_k,
        n_bootstrap=n_bootstrap,
        random_state=random_state + 1,
    )

    assignments = pd.DataFrame(
        {
            "subject_id": disease_only["subject_id"].astype(str).to_numpy(),
            "subgroup_label": [f"subgroup_{label + 1}" for label in labels],
            "n_clusters": selected_k,
        }
    )
    output_features = pd.concat(
        [disease_only[["subject_id"]].reset_index(drop=True), feature_frame.reset_index(drop=True)],
        axis=1,
    )

    assignments_path = target_dir / "disease_subgroup_assignments.csv"
    gap_path = target_dir / "disease_subgroup_gap_statistic.csv"
    stability_path = target_dir / "disease_subgroup_stability.csv"
    feature_path = target_dir / "disease_subgroup_feature_matrix.csv"
    assignments.to_csv(assignments_path, index=False)
    gap.to_csv(gap_path, index=False)
    stability.to_csv(stability_path, index=False)
    output_features.to_csv(feature_path, index=False)

    # Emit per-subgroup mean spectra (reviewer request: "show the average spectrum per ML group")
    subgroup_cfg_emit = bool(subgroup_cfg.get("emit_cluster_mean_spectra", True))
    if subgroup_cfg_emit:
        full_frame = filter_disease_subjects(load_subject_feature_matrix(feature_matrix_path=feature_matrix_path))
        annotated = full_frame.merge(
            assignments[["subject_id", "subgroup_label"]],
            on="subject_id",
            how="inner",
        )
        if not annotated.empty:
            mean_spectra_outputs = run_subgroup_mean_spectra(annotated, "subgroup_label", target_dir)
        else:
            mean_spectra_outputs = {}
    else:
        mean_spectra_outputs = {}

    logger.info("Disease subgroup outputs written to %s", target_dir)
    return {
        "disease_subgroup_assignments": assignments_path,
        "disease_subgroup_gap_statistic": gap_path,
        "disease_subgroup_stability": stability_path,
        "disease_subgroup_feature_matrix": feature_path,
        **mean_spectra_outputs,
    }
