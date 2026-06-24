"""Phase O: spectral entropy and dispersion descriptors.

Computes per-subject Shannon entropy and a secondary Gini-coefficient
dispersion metric over the SNV-normalized fingerprint region
(``1300__*`` columns).  Both metrics use a fixed, documented per-subject
shift-to-non-negative normalization; the representation is explicitly
noted in the output.

Entropy formula: H = −Σ p_i · ln(p_i),  p_i = I_i / Σ I_i   (after shifting
per-subject minimum to zero so all intensities ≥ 0).

Outputs go to ``06_spectral_interpretability/spectral_entropy/``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config, get_spectral_interpretability_dir
from src.analysis._tier2_utils import compute_spearman_associations
from src.analysis.band_features import block_columns
from src.analysis.clinical_feature_associations import (
    filter_disease_subjects,
    load_subject_feature_matrix,
)
from src.analysis.io_utils import write_csv_atomic
from src.analysis.preregistered_panel import panel_cfg

logger = logging.getLogger(__name__)

_NORMALIZATION_NOTE = (
    "Shannon entropy computed on per-subject shift-to-non-negative SNV intensities "
    "(subtract per-subject minimum, then normalize to sum = 1). "
    "Entropy is a dispersion descriptor, not thermodynamic entropy."
)

_METRICS = ["spectral_entropy", "gini_dispersion"]


# ---------------------------------------------------------------------------
# Entropy and Gini
# ---------------------------------------------------------------------------


def spectral_entropy(intensities: np.ndarray) -> float:
    """Shannon entropy of one subject's intensity vector (natural log).

    Shifts to non-negative before normalization. Returns NaN for constant
    or all-NaN spectra.
    """
    finite = intensities[np.isfinite(intensities)]
    if finite.size < 2:
        return np.nan
    shifted = finite - float(np.min(finite))
    total = float(np.sum(shifted))
    if total <= 0.0:
        return np.nan
    p = shifted / total
    nonzero = p[p > 0]
    return float(-np.sum(nonzero * np.log(nonzero)))


def gini_coefficient(intensities: np.ndarray) -> float:
    """Gini coefficient as secondary dispersion metric (values in [0, 1]).

    Works on non-negative shifted intensities. Returns NaN for single-value
    or all-NaN vectors.
    """
    finite = intensities[np.isfinite(intensities)]
    if finite.size < 2:
        return np.nan
    shifted = finite - float(np.min(finite))
    total = float(np.sum(shifted))
    if total <= 0.0:
        return 0.0
    n = float(finite.size)
    ranks = np.arange(1, int(n) + 1, dtype=float)
    return float((2.0 * np.sum(ranks * np.sort(shifted))) / (n * total) - (n + 1.0) / n)


# ---------------------------------------------------------------------------
# Subject-level computation
# ---------------------------------------------------------------------------


def compute_spectral_entropy_features(frame: pd.DataFrame, *, block: str = "1300") -> pd.DataFrame:
    """Per-subject entropy + Gini for the given spectral block.

    Parameters
    ----------
    frame:
        Subject feature matrix with ``block`` columns (e.g. ``1300__*``).
    block:
        Spectral block prefix; ``"1300"`` = fingerprint (default).
    """
    col_wn = block_columns(list(frame.columns), block)
    if not col_wn:
        logger.warning("Phase O: no %s block columns found", block)
        return pd.DataFrame()

    X = frame[[c for c, _ in col_wn]].to_numpy(dtype=float)
    rows = [
        {
            "subject_id": str(sid),
            "spectral_block": block,
            "n_channels": int(np.sum(np.isfinite(X[i]))),
            "spectral_entropy": spectral_entropy(X[i]),
            "gini_dispersion": gini_coefficient(X[i]),
            "normalization_note": _NORMALIZATION_NOTE,
        }
        for i, sid in enumerate(frame["subject_id"])
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def _plot_entropy_distribution(entropy_df: pd.DataFrame, output_dir: Path) -> Path | None:
    values = entropy_df["spectral_entropy"].dropna().to_numpy(dtype=float) if "spectral_entropy" in entropy_df.columns else np.array([])
    if values.size < 2:
        return None

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(values, bins=max(5, values.size // 3), color="#56B4E9", edgecolor="white", alpha=0.85)
    ax.axvline(float(np.median(values)), color="#D55E00", linewidth=1.5, linestyle="--", label="Median")
    ax.set_xlabel("Spectral entropy (fingerprint, natural log)")
    ax.set_ylabel("Subjects")
    ax.set_title("Distribution of per-subject spectral entropy")
    ax.legend(fontsize=8)
    fig.tight_layout()

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / "spectral_entropy_distribution.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Orchestrating runner
# ---------------------------------------------------------------------------


def run_spectral_descriptors(
    *,
    feature_matrix_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Phase O orchestrator: spectral entropy + Gini dispersion + associations."""
    if cfg is None:
        cfg = config
    panel = panel_cfg(cfg)
    if not bool(cfg.get("analysis", {}).get("spectral_descriptors", {}).get("enabled", True)):
        return {}

    endpoints = [str(e) for e in panel.get("endpoints", {}).get("continuous", [])]
    covariates = [str(c) for c in panel.get("covariates", [])]
    fdr_alpha = float(panel.get("fdr_alpha", 0.05))
    min_subjects = int(panel.get("min_subjects_per_group", 5))

    target_dir = Path(output_dir) if output_dir is not None else get_spectral_interpretability_dir("entropy")
    target_dir.mkdir(parents=True, exist_ok=True)

    clinical = filter_disease_subjects(load_subject_feature_matrix(feature_matrix_path=feature_matrix_path))
    entropy_df = compute_spectral_entropy_features(clinical, block="1300")

    entropy_path = target_dir / "spectral_entropy.csv"
    write_csv_atomic(entropy_df, entropy_path, index=False)
    outputs: dict[str, Path] = {"spectral_entropy": entropy_path}

    assoc_path = target_dir / "spectral_entropy_associations.csv"
    associations = (
        compute_spearman_associations(
            entropy_df[["subject_id"] + _METRICS], clinical, _METRICS,
            endpoints, covariates, min_subjects, fdr_alpha, feature_col_name="metric",
        )
        if not entropy_df.empty and endpoints
        else pd.DataFrame()
    )
    write_csv_atomic(associations, assoc_path, index=False)
    outputs["spectral_entropy_associations"] = assoc_path

    if not entropy_df.empty:
        fig_path = _plot_entropy_distribution(entropy_df, target_dir)
        if fig_path:
            outputs["spectral_entropy_figure"] = fig_path

    logger.info("Phase O outputs written to %s", target_dir)
    return outputs
