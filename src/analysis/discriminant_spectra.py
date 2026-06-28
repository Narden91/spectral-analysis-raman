"""Phase P: difference-spectrum signatures and signed-effect maps.

Computes per-block difference spectra (group_A_mean − group_B_mean) for the
pre-registered contrasts (severity_extreme, trajectory_extreme) and a
signed-effect spectral map (sign(Δ) × |Cohen's d| per wavenumber).

Fingerprint (1300 block) and CH-stretch (2850 block) are rendered in
separate figure panels to avoid mixing block normalizations.

Outputs go to ``06_spectral_interpretability/discriminant_signatures/``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config, get_spectral_interpretability_dir
from src.analysis.band_features import block_columns
from src.analysis.clinical_feature_associations import (
    filter_disease_subjects,
    load_subject_feature_matrix,
)
from src.analysis.io_utils import write_csv_atomic
from src.analysis.preregistered_panel import panel_cfg
from src.analysis.robust_contrasts import _contrast_specs
from src.analysis.statistics import cohens_d, get_raman_bands

logger = logging.getLogger(__name__)

_BLOCKS = ["1300", "2850"]
_BLOCK_LABELS = {"1300": "Fingerprint (1300 block)", "2850": "CH-stretch (2850 block)"}


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def _group_spectra(
    frame: pd.DataFrame,
    *,
    group_column: str,
    left_label: str,
    right_label: str,
    block: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (wavenumbers, X_left, X_right) for the given block and contrast."""
    col_wn = block_columns(list(frame.columns), block)
    if not col_wn:
        return np.array([]), np.empty((0, 0)), np.empty((0, 0))
    spec_cols = [c for c, _ in col_wn]
    wavenumbers = np.array([wn for _, wn in col_wn], dtype=float)

    groups = frame[group_column].astype("string")
    # Extreme-contrast columns use pd.NA for excluded middle subjects; fillna(False)
    # before bool conversion to avoid Arrow/pandas NA ambiguity errors.
    mask_left = groups.eq(left_label).fillna(False).to_numpy(dtype=bool)
    mask_right = groups.eq(right_label).fillna(False).to_numpy(dtype=bool)

    X_left = frame.loc[mask_left, spec_cols].to_numpy(dtype=float)
    X_right = frame.loc[mask_right, spec_cols].to_numpy(dtype=float)
    return wavenumbers, X_left, X_right


def compute_difference_spectrum(
    X_left: np.ndarray,
    X_right: np.ndarray,
    wavenumbers: np.ndarray,
) -> pd.DataFrame:
    """Return per-wavenumber statistics for a group contrast.

    Columns: ``wavenumber``, ``mean_group_a``, ``mean_group_b``,
    ``difference``, ``std_group_a``, ``std_group_b``,
    ``cohens_d``, ``signed_effect``.
    """
    mean_left = np.nanmean(X_left, axis=0) if X_left.shape[0] > 0 else np.full(wavenumbers.size, np.nan)
    mean_right = np.nanmean(X_right, axis=0) if X_right.shape[0] > 0 else np.full(wavenumbers.size, np.nan)
    std_left = np.nanstd(X_left, axis=0, ddof=1) if X_left.shape[0] > 1 else np.full(wavenumbers.size, np.nan)
    std_right = np.nanstd(X_right, axis=0, ddof=1) if X_right.shape[0] > 1 else np.full(wavenumbers.size, np.nan)
    diff = mean_left - mean_right

    cd_vals = np.array(
        [
            cohens_d(
                X_left[:, j][np.isfinite(X_left[:, j])],
                X_right[:, j][np.isfinite(X_right[:, j])],
            )
            for j in range(wavenumbers.size)
        ],
        dtype=float,
    )
    signed_effect = np.sign(diff) * np.abs(cd_vals)

    return pd.DataFrame(
        {
            "wavenumber": wavenumbers,
            "mean_group_a": mean_left,
            "mean_group_b": mean_right,
            "difference": diff,
            "std_group_a": std_left,
            "std_group_b": std_right,
            "cohens_d": cd_vals,
            "signed_effect": signed_effect,
        }
    )


# ---------------------------------------------------------------------------
# Nearest known band annotation
# ---------------------------------------------------------------------------


def _annotate_wavenumbers(wavenumbers: np.ndarray, tolerance: float = 15.0) -> list[str]:
    """Return nearest-band label for each wavenumber (empty string if none within tolerance)."""
    raman_bands = get_raman_bands()
    labels: list[str] = []
    for wn in wavenumbers:
        near = [name for name, centre in raman_bands.items() if abs(wn - centre) <= tolerance]
        labels.append(";".join(near) if near else "")
    return labels


# ---------------------------------------------------------------------------
# Figures — fingerprint and CH panels separated
# ---------------------------------------------------------------------------


def _plot_difference_spectrum(
    rows_by_block: dict[str, pd.DataFrame],
    *,
    contrast_name: str,
    left_label: str,
    right_label: str,
    output_dir: Path,
) -> list[Path]:
    saved: list[Path] = []
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    for block, data in rows_by_block.items():
        if data.empty or "wavenumber" not in data.columns:
            continue
        raman_bands = get_raman_bands()
        fig, axes = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True)

        wn = data["wavenumber"].to_numpy(dtype=float)
        diff = data["difference"].to_numpy(dtype=float)
        se = data["signed_effect"].to_numpy(dtype=float)

        # Top panel: mean spectra + shaded difference
        ax0 = axes[0]
        mean_a = data["mean_group_a"].to_numpy(dtype=float)
        mean_b = data["mean_group_b"].to_numpy(dtype=float)
        ax0.plot(wn, mean_a, color="#0072B2", linewidth=1.2, label=left_label)
        ax0.plot(wn, mean_b, color="#E69F00", linewidth=1.2, label=right_label)
        ax0.fill_between(wn, mean_a, mean_b, where=(mean_a > mean_b),
                         alpha=0.25, color="#0072B2", label="A > B")
        ax0.fill_between(wn, mean_a, mean_b, where=(mean_a <= mean_b),
                         alpha=0.25, color="#E69F00", label="B ≥ A")
        ax0.set_ylabel("Norm. intensity (a.u.)")
        ax0.legend(fontsize=7, loc="upper right")
        ax0.set_title(f"{_BLOCK_LABELS.get(block, block)} — {contrast_name}")

        # Bottom panel: signed-effect map
        ax1 = axes[1]
        pos = se >= 0
        ax1.bar(wn[pos], se[pos], width=np.diff(wn).mean() * 0.9 if wn.size > 1 else 1.0,
                color="#0072B2", alpha=0.7, label="A > B (Cohen's d)")
        ax1.bar(wn[~pos], se[~pos], width=np.diff(wn).mean() * 0.9 if wn.size > 1 else 1.0,
                color="#E69F00", alpha=0.7, label="B > A")
        ax1.axhline(0, color="#333333", linewidth=0.8)

        # Annotate known bands within the block
        block_centre_lo = 700.0 if block == "1300" else 2400.0
        block_centre_hi = 1900.0 if block == "1300" else 3200.0
        for band_name, centre in raman_bands.items():
            if block_centre_lo <= centre <= block_centre_hi:
                ax1.axvline(centre, color="#999999", linewidth=0.6, linestyle=":")
                ax1.text(centre, float(np.nanmin(se)) * 0.9, band_name,
                         rotation=90, fontsize=5, va="bottom", ha="center", color="#666666")

        ax1.set_xlabel("Raman shift (cm⁻¹)")
        ax1.set_ylabel("sign(Δ)×|Cohen's d|")
        ax1.legend(fontsize=7)
        fig.tight_layout()

        fname = f"difference_spectrum_{contrast_name}_{block}.png"
        path = figures_dir / fname
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
    return saved


# ---------------------------------------------------------------------------
# Orchestrating runner
# ---------------------------------------------------------------------------


def run_discriminant_spectra(
    *,
    feature_matrix_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Phase P orchestrator: difference spectra + signed-effect maps + figures."""
    if cfg is None:
        cfg = config
    panel = panel_cfg(cfg)
    phase_cfg = cfg.get("analysis", {}).get("discriminant_spectra", {})
    if not bool(phase_cfg.get("enabled", True)):
        return {}

    target_dir = Path(output_dir) if output_dir is not None else get_spectral_interpretability_dir("discriminant")
    target_dir.mkdir(parents=True, exist_ok=True)

    clinical = filter_disease_subjects(load_subject_feature_matrix(feature_matrix_path=feature_matrix_path))
    contrasts = _contrast_specs(panel)
    if not contrasts:
        logger.warning("Phase P: no contrast specifications found in config; skipping")
        return {}

    all_diff_rows: list[pd.DataFrame] = []
    outputs: dict[str, Path] = {}

    for contrast_name, group_column, left_label, right_label in contrasts:
        if group_column not in clinical.columns:
            logger.warning("Phase P: group column '%s' not found; skipping contrast %s", group_column, contrast_name)
            continue

        rows_by_block: dict[str, pd.DataFrame] = {}

        for block in _BLOCKS:
            wavenumbers, X_left, X_right = _group_spectra(
                clinical,
                group_column=group_column,
                left_label=left_label,
                right_label=right_label,
                block=block,
            )
            if wavenumbers.size == 0:
                continue
            n_left = int(np.sum(np.all(np.isfinite(X_left), axis=1))) if X_left.ndim == 2 else 0
            n_right = int(np.sum(np.all(np.isfinite(X_right), axis=1))) if X_right.ndim == 2 else 0
            if n_left == 0 or n_right == 0:
                logger.info(
                    "Phase P: contrast %s block %s — n_left=%d n_right=%d; skipping block",
                    contrast_name, block, n_left, n_right,
                )
                continue

            block_df = compute_difference_spectrum(X_left, X_right, wavenumbers)
            band_labels = _annotate_wavenumbers(wavenumbers)
            block_df.insert(0, "contrast_name", contrast_name)
            block_df.insert(1, "block", block)
            block_df["group_a"] = left_label
            block_df["group_b"] = right_label
            block_df["n_group_a"] = n_left
            block_df["n_group_b"] = n_right
            block_df["nearest_band"] = band_labels
            rows_by_block[block] = block_df
            all_diff_rows.append(block_df)

        # Figures for this contrast (fingerprint and CH separate)
        if rows_by_block:
            fig_paths = _plot_difference_spectrum(
                rows_by_block,
                contrast_name=contrast_name,
                left_label=left_label,
                right_label=right_label,
                output_dir=target_dir,
            )
            for i, fp in enumerate(fig_paths):
                outputs[f"difference_spectrum_fig_{contrast_name}_{i}"] = fp

    diff_df = pd.concat(all_diff_rows, ignore_index=True) if all_diff_rows else pd.DataFrame()
    diff_path = target_dir / "difference_spectrum.csv"
    write_csv_atomic(diff_df, diff_path, index=False)
    outputs["difference_spectrum"] = diff_path

    if not diff_df.empty:
        effect_cols = ["contrast_name", "block", "wavenumber", "group_a", "group_b",
                       "n_group_a", "n_group_b", "cohens_d", "signed_effect", "nearest_band"]
        effect_path = target_dir / "signed_effect_map.csv"
        write_csv_atomic(diff_df[effect_cols].copy(), effect_path, index=False)
        outputs["signed_effect_map"] = effect_path

    logger.info("Phase P outputs written to %s", target_dir)
    return outputs
