"""Disease-only clinical correlation matrix on subject-level Raman features."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config, get_cross_region_phase_dir
from src.analysis.clinical_feature_associations import (
    apply_spearman_cluster_permutation,
    build_candidate_regions,
    compute_spearman_feature_tests,
    endpoint_missingness_report,
    load_disease_feature_frame,
)

logger = logging.getLogger(__name__)

TARGETS = (
    "moca_baseline",
    "moca_3m",
    "delta_moca",
    "rbans_total",
    "age",
    "aspects_score",
)
TARGET_LABELS = {
    "moca_baseline": "MoCA baseline",
    "moca_3m": "MoCA 3 months",
    "delta_moca": "Delta MoCA",
    "rbans_total": "RBANS",
    "age": "Age",
    "aspects_score": "ASPECTS score",
}


def _cfg(cfg: dict | None = None) -> dict:
    return config if cfg is None else cfg


def _matrix_cfg(cfg: dict) -> dict:
    return cfg.get("analysis", {}).get("clinical_correlation_matrix", {})


def _empty_correlation_table() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "target",
            "feature_name",
            "spectral_block",
            "wavenumber",
            "rho",
            "p_value",
            "fdr_q",
            "significant_fdr",
            "n",
            "status",
            "cluster_id",
            "cluster_mass",
            "cluster_p_value",
        ]
    )


def _plot_placeholder(path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.text(0.5, 0.5, "No clinical correlations available", ha="center", va="center")
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_correlation_heatmap(correlations: pd.DataFrame, path: Path) -> None:
    if correlations.empty:
        _plot_placeholder(path, "Clinical correlation matrix")
        return

    present_targets = [target for target in TARGETS if target in set(correlations["target"])]
    if not present_targets:
        _plot_placeholder(path, "Clinical correlation matrix")
        return

    max_abs = float(np.nanmax(np.abs(correlations["rho"].to_numpy(dtype=float))))
    vmax = max(max_abs, 0.05)
    fig, axes = plt.subplots(2, 1, figsize=(16, 6), constrained_layout=True)

    for axis, block in zip(axes, ("1300", "2850")):
        block_df = correlations.loc[correlations["spectral_block"] == block].copy()
        if block_df.empty:
            axis.text(0.5, 0.5, f"No {block} correlations", ha="center", va="center")
            axis.set_axis_off()
            continue

        ordered = (
            block_df.pivot(index="target", columns="wavenumber", values="rho")
            .reindex(present_targets)
            .sort_index(axis=1)
        )
        image = axis.imshow(
            ordered.to_numpy(dtype=float),
            aspect="auto",
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
        )
        axis.set_title(f"{block} feature correlations")
        axis.set_yticks(np.arange(len(ordered.index)))
        axis.set_yticklabels([TARGET_LABELS.get(target, target) for target in ordered.index], fontsize=9)

        if ordered.shape[1] > 1:
            tick_count = min(8, ordered.shape[1])
            tick_positions = np.linspace(0, ordered.shape[1] - 1, tick_count, dtype=int)
            axis.set_xticks(tick_positions)
            axis.set_xticklabels(
                [f"{ordered.columns[position]:.0f}" for position in tick_positions],
                fontsize=8,
            )
        axis.set_xlabel("Raman shift (cm-1)")

    colorbar = fig.colorbar(image, ax=axes, fraction=0.02, pad=0.02)
    colorbar.set_label("Spearman rho")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_clinical_correlation_matrix(
    *,
    feature_matrix_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    cfg: dict | None = None,
) -> dict[str, Path]:
    cfg = _cfg(cfg)
    matrix_cfg = _matrix_cfg(cfg)
    if not bool(matrix_cfg.get("enabled", True)):
        return {}

    disease_only, feature_meta = load_disease_feature_frame(feature_matrix_path=feature_matrix_path)
    targets = tuple(matrix_cfg.get("targets", TARGETS))
    fdr_alpha = float(matrix_cfg.get("fdr_alpha", 0.05))
    min_subjects = int(matrix_cfg.get("min_subjects", 5))
    cluster_alpha = float(matrix_cfg.get("cluster_alpha", 0.05))
    n_permutations = int(matrix_cfg.get("n_permutations", 1000))
    n_jobs = int(matrix_cfg.get("n_jobs", 1))
    random_state = int(matrix_cfg.get("random_state", 42))

    target_dir = (
        Path(output_dir)
        if output_dir is not None
        else get_cross_region_phase_dir("clinical_correlations")
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    missingness = endpoint_missingness_report(disease_only, targets)
    correlation_frames: list[pd.DataFrame] = []
    for target in targets:
        tests = compute_spearman_feature_tests(
            disease_only,
            feature_meta,
            target=target,
            min_subjects=min_subjects,
            fdr_alpha=fdr_alpha,
        )
        if tests.empty:
            continue
        tests = apply_spearman_cluster_permutation(
            tests,
            disease_only,
            target=target,
            n_permutations=n_permutations,
            cluster_alpha=cluster_alpha,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        correlation_frames.append(tests)

    correlations = (
        pd.concat(correlation_frames, ignore_index=True, sort=False)
        if correlation_frames
        else _empty_correlation_table()
    )
    if not correlations.empty:
        correlations = correlations.merge(
            missingness.rename(
                columns={
                    "n_subjects_with_spectra": "n_total_disease_subjects",
                    "n_non_missing": "n_non_missing_target",
                    "n_missing": "n_missing_target",
                    "missing_rate": "missing_rate_target",
                }
            ),
            left_on="target",
            right_on="endpoint",
            how="left",
        ).drop(columns=["endpoint", "status_y"], errors="ignore").rename(
            columns={"status_x": "status"}
        )
    candidate_regions = build_candidate_regions(
        correlations,
        group_columns=("target",),
        effect_column="rho",
        min_width_cm=float(matrix_cfg.get("candidate_region_min_width_cm", 8.0)),
        merge_gap_cm=float(matrix_cfg.get("candidate_region_merge_gap_cm", 6.0)),
        fallback_quantile=float(matrix_cfg.get("candidate_region_fallback_quantile", 0.9)),
    ).rename(
        columns={
            "max_abs_effect": "max_abs_rho",
            "mean_abs_effect": "mean_abs_rho",
            "integrated_abs_effect": "integrated_abs_rho",
        }
    )

    matrix_path = target_dir / "clinical_correlation_matrix.csv"
    heatmap_path = target_dir / "clinical_correlation_heatmap.png"
    candidate_path = target_dir / "clinical_correlation_candidate_regions.csv"
    obsolete_paths = [
        target_dir / "endpoint_missingness_report.csv",
    ]
    for obsolete_path in obsolete_paths:
        if obsolete_path.exists():
            obsolete_path.unlink()

    correlations.to_csv(matrix_path, index=False)
    candidate_regions.to_csv(candidate_path, index=False)
    _plot_correlation_heatmap(correlations, heatmap_path)

    logger.info("Clinical correlation matrix outputs written to %s", target_dir)
    return {
        "clinical_correlation_matrix": matrix_path,
        "clinical_correlation_heatmap": heatmap_path,
        "clinical_correlation_candidate_regions": candidate_path,
    }
