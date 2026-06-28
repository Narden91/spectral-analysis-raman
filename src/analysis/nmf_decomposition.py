"""Phase N: NMF decomposition of fingerprint-region spectra.

Decomposes the fingerprint block (``1300__*`` columns) into ``n_components``
non-negative latent profiles using multi-seed NMF. Component stability is
assessed via reconstruction-error variance across initializations. Subject
weights are correlated with clinical endpoints (Spearman + partial Spearman,
BH-FDR). All outputs go to ``06_spectral_interpretability/nmf_decomposition/``.

Components are labeled by the band windows they overlap (using config
``whole_spectra.bands``). Labels use "carotenoid-like / lipoprotein-associated"
for the 1525 cm⁻¹ region per the conservative claim-boundary rule.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
from src.analysis.preregistered_panel import panel_cfg
from src.analysis.statistics import get_raman_bands

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conservative band-overlap annotation
# ---------------------------------------------------------------------------

_BAND_LABELS: dict[str, str] = {
    "lipid_carbonyl": "lipid-carbonyl-like region (~1720)",
    "carotenoid_1525": "carotenoid-like / lipoprotein-associated region (~1525)",
    "carotenoid_cc_1156": "carotenoid-like C-C region (~1156)",
    "protein_1555": "protein-overlapping region (~1555)",
    "structural_protein_925": "backbone C-C/C-N region (~925)",
    "structural_protein_980": "backbone C-N region (~980)",
    "tyrosine_851": "tyrosine Fermi doublet region (~851)",
    "tyrosine_821": "tyrosine Fermi doublet region (~821)",
}


def _annotate_component(component_loadings: np.ndarray, wavenumbers: np.ndarray, top_k: int = 3) -> str:
    """Return a conservative region-overlap annotation for one NMF component."""
    raman_bands = get_raman_bands()
    if not raman_bands or component_loadings.size == 0:
        return "unassigned"

    top_wavenumbers = wavenumbers[np.argsort(component_loadings)[::-1][:top_k]]
    labels: list[str] = []
    for band_name, band_centre in raman_bands.items():
        if any(abs(wn - band_centre) <= 15.0 for wn in top_wavenumbers):
            label = _BAND_LABELS.get(band_name, f"region overlapping {band_name} (~{band_centre:.0f})")
            if label not in labels:
                labels.append(label)

    return "; ".join(labels) if labels else "unassigned"


# ---------------------------------------------------------------------------
# Multi-seed NMF fit with stability metric
# ---------------------------------------------------------------------------


def _reconstruction_error(X: np.ndarray, W: np.ndarray, H: np.ndarray) -> float:
    return float(np.mean((X - W @ H) ** 2))


def _fit_nmf_multi_init(
    X: np.ndarray,
    *,
    n_components: int,
    alpha_w: float,
    alpha_h: float,
    n_init: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Fit NMF ``n_init`` times; return (W, H, best_error, error_std_across_inits)."""
    rng = np.random.default_rng(random_state)
    seeds = rng.integers(0, np.iinfo(np.int32).max, size=n_init, dtype=np.int64).tolist()

    errors: list[float] = []
    solutions: list[tuple[np.ndarray, np.ndarray]] = []

    for seed in seeds:
        try:
            model = NMF(
                n_components=n_components,
                init="nndsvda",
                l1_ratio=1.0,
                alpha_W=alpha_w,
                alpha_H=alpha_h,
                max_iter=1000,
                random_state=int(seed),
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                W = model.fit_transform(X)
            errors.append(_reconstruction_error(X, W, model.components_))
            solutions.append((W, model.components_))
        except Exception as exc:
            logger.debug("NMF init failed (seed=%d): %s", seed, exc)

    if not solutions:
        raise RuntimeError("All NMF initializations failed")

    best_idx = int(np.argmin(errors))
    error_std = float(np.std(errors, ddof=1)) if len(errors) > 1 else 0.0
    return solutions[best_idx][0], solutions[best_idx][1], float(errors[best_idx]), error_std


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

    W, H, best_error, error_std = _fit_nmf_multi_init(
        X_shifted, n_components=n_components_actual,
        alpha_w=alpha, alpha_h=0.0, n_init=n_init, random_state=random_state,
    )
    stability_metric = float(error_std / (best_error + 1e-12))
    logger.info(
        "Phase N: NMF fitted (%d components, best_error=%.4f, error_std=%.4f, stability=%.4f)",
        n_components_actual, best_error, error_std, stability_metric,
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
        "best_reconstruction_error": best_error,
        "reconstruction_error_std": error_std,
        "stability_metric": stability_metric,
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
