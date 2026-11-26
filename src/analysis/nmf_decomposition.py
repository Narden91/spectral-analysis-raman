"""Phase N: NMF decomposition of fingerprint-region spectra.

Decomposes the fingerprint block (``1300__*`` columns) into ``n_components``
non-negative latent profiles using multi-seed NMF. Component stability is
assessed by matching component spectra across seeded random initializations. Subject
weights are correlated with clinical endpoints (Spearman + partial Spearman,
BH-FDR). All outputs go to ``06_spectral_interpretability/02_nmf_decomposition/``.

Components are labeled by the band windows they overlap (using config
``whole_spectra.bands``). Labels use "carotenoid-like / lipoprotein-associated"
for the 1525 cm⁻¹ region per the conservative claim-boundary rule.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning

from src import config, get_spectral_interpretability_dir
from src.analysis._tier2_utils import compute_spearman_associations
from src.analysis.band_features import block_columns
from src.analysis.clinical_feature_associations import (
    filter_disease_subjects,
    load_subject_feature_matrix,
)
from src.analysis.io_utils import write_csv_atomic
from src.analysis.paired_lipid._backend import _single_thread_blas
from src.analysis.preregistered_panel import panel_cfg
from src.analysis.raman_bands import RAMAN_BANDS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conservative band-overlap annotation
# ---------------------------------------------------------------------------

_BAND_LABELS: dict[str, str] = {
    "lipid_carbonyl_1721": "carbonyl-overlapping region (~1721)",
    "carotenoid_1525": "carotenoid-overlapping region (~1525)",
    "carotenoid_1156": "carotenoid-overlapping C-C region (~1156)",
    "tryptophan_1555": "tryptophan-overlapping region (~1555)",
    "c_o_c_sugar_925": "sugar C-O-C-overlapping region (~925)",
    "beta_sheet_980": "beta-sheet C-C-overlapping region (~980)",
    "tyrosine_851": "tyrosine/glycan-overlapping region (~851)",
    "o_p_o_821": "O-P-O-overlapping region (~821)",
}



def _annotate_component(component_loadings: np.ndarray, wavenumbers: np.ndarray, top_k: int = 3) -> str:
    """Return a conservative region-overlap annotation for one NMF component."""
    raman_bands = RAMAN_BANDS
    if not raman_bands or component_loadings.size == 0:
        return "unassigned"

    top_wavenumbers = wavenumbers[np.argsort(component_loadings)[::-1][:top_k]]
    labels: list[str] = []
    for band_name, band in raman_bands.items():
        band_centre = band.peak
        if any(band.bounds[0] <= wn <= band.bounds[1] for wn in top_wavenumbers):
            label = _BAND_LABELS.get(band_name, f"region overlapping {band_name} (~{band_centre:.0f})")
            if label not in labels:
                labels.append(label)

    return "; ".join(labels) if labels else "unassigned"


# ---------------------------------------------------------------------------
# Multi-seed NMF fit with stability metric
# ---------------------------------------------------------------------------


def _reconstruction_error(X: np.ndarray, W: np.ndarray, H: np.ndarray) -> float:
    return float(np.mean((X - W @ H) ** 2))


@dataclass(frozen=True)
class NMFMultiInitResult:
    W: np.ndarray
    H: np.ndarray
    best_error: float
    reconstruction_error_std: float
    component_stability: np.ndarray
    n_successful_initializations: int


def _matched_component_stability(reference: np.ndarray, components: list[np.ndarray]) -> np.ndarray:
    """Return median cosine similarity after permutation-invariant matching."""
    if not components:
        return np.full(reference.shape[0], np.nan, dtype=float)

    def _normalize(rows: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        return np.divide(rows, norms, out=np.zeros_like(rows), where=norms > 0)

    reference_norm = _normalize(reference)
    matched_scores: list[np.ndarray] = []
    for candidate in components:
        similarity = reference_norm @ _normalize(candidate).T
        row_idx, col_idx = linear_sum_assignment(-similarity)
        ordered = np.empty(reference.shape[0], dtype=float)
        ordered[row_idx] = similarity[row_idx, col_idx]
        matched_scores.append(ordered)
    return np.median(np.vstack(matched_scores), axis=0)


def _resolve_parallel_workers(n_tasks: int, n_jobs: int | None) -> int:
    if n_tasks <= 1 or n_jobs == 0 or n_jobs == 1:
        return 1
    if os.environ.get("_PAIRED_LIPID_IN_BOOTSTRAP_WORKER") == "1":
        return 1
    if multiprocessing.parent_process() is not None:
        return 1
    try:
        import joblib.parallel

        active_backend, _ = joblib.parallel.get_active_backend()
        if getattr(active_backend, "nesting_level", 0) > 0:
            return 1
    except Exception:
        pass
    cpu_cnt = os.cpu_count() or 1
    if n_jobs is None or int(n_jobs) == -1:
        raw = cpu_cnt
    elif int(n_jobs) < -1:
        raw = max(1, cpu_cnt + 1 + int(n_jobs))
    else:
        raw = int(n_jobs)
    if raw <= 1:
        return 1
    return min(raw, n_tasks)


def _fit_single_nmf_seed(
    X: np.ndarray,
    n_components: int,
    alpha_w: float,
    alpha_h: float,
    seed: int,
) -> tuple[float | None, np.ndarray | None, np.ndarray | None, str | None]:
    try:
        with _single_thread_blas():
            model = NMF(
                n_components=n_components,
                init="random",
                l1_ratio=1.0,
                alpha_W=alpha_w,
                alpha_H=alpha_h,
                max_iter=1000,
                random_state=int(seed),
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                W = model.fit_transform(X)
            err = _reconstruction_error(X, W, model.components_)
        return err, W, model.components_, None
    except Exception as exc:
        return None, None, None, str(exc)


def _fit_nmf_multi_init(
    X: np.ndarray,
    *,
    n_components: int,
    alpha_w: float,
    alpha_h: float,
    n_init: int,
    random_state: int,
    n_jobs: int | None = None,
) -> NMFMultiInitResult:
    """Fit seeded NMF starts and quantify matched component-spectrum stability."""
    rng = np.random.default_rng(random_state)
    seeds = rng.integers(0, np.iinfo(np.int32).max, size=n_init, dtype=np.int64).tolist()

    auto_mode = n_jobs is None or int(n_jobs) < 0
    if auto_mode and (n_init < 4 or X.size < 4000):
        workers = 1
    else:
        workers = _resolve_parallel_workers(len(seeds), n_jobs)

    if workers > 1:
        raw_results = Parallel(n_jobs=workers)(
            delayed(_fit_single_nmf_seed)(X, n_components, alpha_w, alpha_h, int(seed))
            for seed in seeds
        )
    else:
        raw_results = [
            _fit_single_nmf_seed(X, n_components, alpha_w, alpha_h, int(seed))
            for seed in seeds
        ]

    errors: list[float] = []
    solutions: list[tuple[np.ndarray, np.ndarray]] = []
    for seed, (err, W, H, exc_msg) in zip(seeds, raw_results):
        if exc_msg is not None or err is None or W is None or H is None:
            logger.warning("NMF init failed (seed=%d): %s", seed, exc_msg)
        else:
            errors.append(err)
            solutions.append((W, H))

    if not solutions:
        raise RuntimeError("All NMF initializations failed")

    best_idx = int(np.argmin(errors))
    error_std = float(np.std(errors, ddof=1)) if len(errors) > 1 else 0.0
    best_w, best_h = solutions[best_idx]
    other_components = [h for index, (_, h) in enumerate(solutions) if index != best_idx]
    return NMFMultiInitResult(
        W=best_w,
        H=best_h,
        best_error=float(errors[best_idx]),
        reconstruction_error_std=error_std,
        component_stability=_matched_component_stability(best_h, other_components),
        n_successful_initializations=len(solutions),
    )


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def _plot_nmf_components(
    H: np.ndarray, wavenumbers: np.ndarray, annotations: list[str], output_dir: Path
) -> Path | None:
    n_comp = H.shape[0]
    fig, axes = plt.subplots(n_comp, 1, figsize=(7, 2.5 * n_comp), sharex=True)
    if n_comp == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(wavenumbers, H[i], color="#0072B2", linewidth=1.5)
        ax.fill_between(wavenumbers, H[i], alpha=0.25, color="#0072B2")
        ax.set_ylabel(f"NMF C{i + 1}", fontsize=9)
        ax.set_title(annotations[i] if i < len(annotations) else f"Component {i + 1}", fontsize=8, loc="right")
        ax.set_yticks([])

    axes[-1].set_xlabel("Raman shift (cm⁻¹)")
    fig.suptitle("NMF component spectra (exploratory)", fontsize=10, y=1.01)
    fig.tight_layout()

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / "nmf_component_spectra.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Orchestrating runner
# ---------------------------------------------------------------------------


def run_nmf_decomposition(
    *,
    feature_matrix_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Phase N orchestrator: NMF latent components + weight–endpoint associations."""
    if cfg is None:
        cfg = config
    panel = panel_cfg(cfg)
    if not bool(cfg.get("analysis", {}).get("nmf_decomposition", {}).get("enabled", True)):
        return {}

    nmf_cfg = panel.get("nmf", {})
    n_components = int(nmf_cfg.get("n_components", 3))
    alpha = float(nmf_cfg.get("sparsity_alpha", 0.1))
    n_init = int(nmf_cfg.get("n_init", 20))
    n_jobs = nmf_cfg.get("n_jobs", None)
    block = "1300" if str(nmf_cfg.get("apply_to", "fingerprint")) == "fingerprint" else "2850"
    random_state = int(panel.get("random_state", 42))
    endpoints = [str(e) for e in panel.get("endpoints", {}).get("continuous", [])]
    covariates = [str(c) for c in panel.get("covariates", [])]
    fdr_alpha = float(panel.get("fdr_alpha", 0.05))
    min_subjects = int(panel.get("min_subjects_per_group", 5))

    target_dir = Path(output_dir) if output_dir is not None else get_spectral_interpretability_dir("nmf")
    target_dir.mkdir(parents=True, exist_ok=True)

    clinical = filter_disease_subjects(load_subject_feature_matrix(feature_matrix_path=feature_matrix_path))
    col_wn = block_columns(list(clinical.columns), block)
    if not col_wn:
        logger.warning("Phase N: no %s block columns found", block)
        return {}

    spec_cols = [c for c, _ in col_wn]
    wavenumbers = np.array([wn for _, wn in col_wn], dtype=float)
    X = clinical[spec_cols].to_numpy(dtype=float)

    valid_rows = np.all(np.isfinite(X), axis=1)
    X_clean = X[valid_rows]
    subject_ids = clinical.loc[valid_rows, "subject_id"].astype(str).tolist()

    n_components_actual = min(n_components, X_clean.shape[0] - 1, X_clean.shape[1])
    if X_clean.shape[0] < n_components_actual + 1:
        logger.warning("Phase N: too few subjects (%d) for NMF; skipping", X_clean.shape[0])
        return {}

    # NMF requires non-negative input; shift per-row to ≥ 0
    X_shifted = np.clip(X_clean - X_clean.min(axis=1, keepdims=True), 0.0, None)

    fit_result = _fit_nmf_multi_init(
        X_shifted, n_components=n_components_actual,
        alpha_w=alpha, alpha_h=0.0, n_init=n_init, random_state=random_state,
        n_jobs=n_jobs,
    )
    W, H = fit_result.W, fit_result.H
    component_stability = fit_result.component_stability
    stability_metric = float(np.nanmin(component_stability)) if np.isfinite(component_stability).any() else np.nan
    if not np.isfinite(stability_metric):
        logger.warning("Phase N: component stability unavailable; fewer than two successful initializations")
    logger.info(
        "Phase N: NMF fitted (%d components, best_error=%.4f, error_std=%.4f, min_component_stability=%.4f)",
        n_components_actual, fit_result.best_error, fit_result.reconstruction_error_std, stability_metric,
    )

    annotations = [_annotate_component(H[i], wavenumbers) for i in range(n_components_actual)]

    # Build long-form components DataFrame using vectorised tiling
    n_wn = len(col_wn)
    block_cols_list = [c for c, _ in col_wn]
    wn_arr = [wn for _, wn in col_wn]
    components_df = pd.DataFrame({
        "component": np.repeat([f"component_{i + 1}" for i in range(n_components_actual)], n_wn),
        "block_column": np.tile(block_cols_list, n_components_actual),
        "wavenumber": np.tile(wn_arr, n_components_actual),
        "loading": H.ravel(),
        "annotation": np.repeat(annotations, n_wn),
        "n_components_fitted": n_components_actual,
        "best_reconstruction_error": fit_result.best_error,
        "reconstruction_error_std": fit_result.reconstruction_error_std,
        "component_stability_cosine": np.repeat(component_stability, n_wn),
        "stability_metric": stability_metric,
        "n_initializations_requested": n_init,
        "n_initializations_succeeded": fit_result.n_successful_initializations,
        "exploratory": True,
    })
    components_path = target_dir / "nmf_components.csv"
    write_csv_atomic(components_df, components_path, index=False)

    weights_df = pd.DataFrame(
        {"subject_id": subject_ids, **{f"component_{i + 1}": W[:, i] for i in range(n_components_actual)}}
    )
    weights_path = target_dir / "nmf_weights_per_subject.csv"
    write_csv_atomic(weights_df, weights_path, index=False)

    outputs: dict[str, Path] = {
        "nmf_components": components_path,
        "nmf_weights_per_subject": weights_path,
    }

    if endpoints:
        component_cols = [f"component_{i + 1}" for i in range(n_components_actual)]
        associations = compute_spearman_associations(
            weights_df, clinical, component_cols, endpoints, covariates,
            min_subjects, fdr_alpha, feature_col_name="component",
        )
        assoc_path = target_dir / "nmf_weight_associations.csv"
        write_csv_atomic(associations, assoc_path, index=False)
        outputs["nmf_weight_associations"] = assoc_path

    fig_path = _plot_nmf_components(H, wavenumbers, annotations, target_dir)
    if fig_path:
        outputs["nmf_component_spectra"] = fig_path

    logger.info("Phase N outputs written to %s", target_dir)
    return outputs
