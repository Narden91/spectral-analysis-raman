"""Disease-only longitudinal Delta MoCA association analysis."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src import get_cross_region_phase_dir
from src import resolve_cfg as _cfg
from src.analysis.clinical_feature_associations import (
    compute_binary_group_feature_tests,
    compute_spearman_feature_tests,
    empty_spearman_association_table,
    load_disease_feature_frame,
)
from src.analysis.spectral_regions import build_candidate_regions
from src.statistics.cluster_permutation import apply_binary_cluster_permutation, apply_spearman_cluster_permutation

logger = logging.getLogger(__name__)

DELTA_TARGET = "delta_moca"
DECLINE_GROUPS = ("decline", "stable_or_improved")


def _longitudinal_cfg(cfg: dict) -> dict:
    return cfg.get("analysis", {}).get("longitudinal_moca", {})


def _empty_group_table() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "contrast_name",
            "group_variable",
            "group_a",
            "group_b",
            "test",
            "feature_name",
            "spectral_block",
            "wavenumber",
            "n_group_a",
            "n_group_b",
            "u_statistic",
            "p_value",
            "effect_size",
            "effect_size_name",
            "status",
            "fdr_q",
            "significant_fdr",
            "cluster_id",
            "cluster_mass",
            "cluster_p_value",
        ]
    )


def _rank_candidate_regions(candidate_regions: pd.DataFrame) -> pd.DataFrame:
    if candidate_regions.empty:
        return candidate_regions

    ranked = candidate_regions.copy()
    ranked["_cluster_sort"] = ranked["min_cluster_p_value"].fillna(1.0)
    ranked["_fdr_sort"] = ranked["min_fdr_q"].fillna(1.0)
    ranked = ranked.sort_values(
        ["_cluster_sort", "_fdr_sort", "integrated_abs_effect"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    return ranked.drop(columns=["_cluster_sort", "_fdr_sort"])


def run_longitudinal_moca(
    *,
    feature_matrix_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    cfg: dict | None = None,
) -> dict[str, Path]:
    cfg = _cfg(cfg)
    longitudinal_cfg = _longitudinal_cfg(cfg)
    if not bool(longitudinal_cfg.get("enabled", True)):
        return {}

    disease_only, feature_meta = load_disease_feature_frame(feature_matrix_path=feature_matrix_path)
    target_dir = (
        Path(output_dir)
        if output_dir is not None
        else get_cross_region_phase_dir("longitudinal_moca")
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Longitudinal Delta MoCA: %d disease subjects, %d whole-spectrum features",
        len(disease_only),
        len(feature_meta),
    )

    fdr_alpha = float(longitudinal_cfg.get("fdr_alpha", 0.05))
    min_subjects = int(longitudinal_cfg.get("min_subjects", 5))
    cluster_alpha = float(longitudinal_cfg.get("cluster_alpha", 0.05))
    n_permutations = int(longitudinal_cfg.get("n_permutations", 1000))
    n_jobs = int(longitudinal_cfg.get("n_jobs", 1))
    random_state = int(longitudinal_cfg.get("random_state", 42))

    delta_associations = compute_spearman_feature_tests(
        disease_only,
        feature_meta,
        target=DELTA_TARGET,
        min_subjects=min_subjects,
        fdr_alpha=fdr_alpha,
    )
    if not delta_associations.empty:
        delta_associations = apply_spearman_cluster_permutation(
            delta_associations,
            disease_only,
            target=DELTA_TARGET,
            n_permutations=n_permutations,
            cluster_alpha=cluster_alpha,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    else:
        delta_associations = empty_spearman_association_table()
    logger.info("Longitudinal Delta MoCA: Delta-MoCA association model complete")

    group_comparison, _group_counts = compute_binary_group_feature_tests(
        disease_only,
        feature_meta,
        group_column="delta_moca_group",
        ordered_groups=DECLINE_GROUPS,
        contrast_name="decline_vs_stable_or_improved",
        min_subjects_per_group=min_subjects,
        fdr_alpha=fdr_alpha,
    )
    if not group_comparison.empty and group_comparison["status"].eq("ok").any():
        group_comparison = apply_binary_cluster_permutation(
            group_comparison,
            disease_only,
            group_column="delta_moca_group",
            ordered_groups=DECLINE_GROUPS,
            n_permutations=n_permutations,
            cluster_alpha=cluster_alpha,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    elif group_comparison.empty:
        group_comparison = _empty_group_table()
    logger.info("Longitudinal Delta MoCA: decline-group comparison complete")

    association_candidates = build_candidate_regions(
        delta_associations.assign(
            analysis_name="delta_moca_association",
            contrast_name="delta_moca",
        ),
        group_columns=("analysis_name", "contrast_name"),
        effect_column="rho",
        min_width_cm=float(longitudinal_cfg.get("candidate_region_min_width_cm", 8.0)),
        merge_gap_cm=float(longitudinal_cfg.get("candidate_region_merge_gap_cm", 6.0)),
        fallback_quantile=float(longitudinal_cfg.get("candidate_region_fallback_quantile", 0.9)),
    )
    group_candidates = build_candidate_regions(
        group_comparison.assign(analysis_name="decline_group_comparison"),
        group_columns=("analysis_name", "contrast_name"),
        effect_column="effect_size",
        min_width_cm=float(longitudinal_cfg.get("candidate_region_min_width_cm", 8.0)),
        merge_gap_cm=float(longitudinal_cfg.get("candidate_region_merge_gap_cm", 6.0)),
        fallback_quantile=float(longitudinal_cfg.get("candidate_region_fallback_quantile", 0.9)),
    )
    candidate_regions = pd.concat(
        [association_candidates, group_candidates],
        ignore_index=True,
        sort=False,
    )
    if candidate_regions.empty:
        candidate_regions = pd.DataFrame(
            columns=[
                "analysis_name",
                "contrast_name",
                "spectral_block",
                "candidate_region_id",
                "start_cm-1",
                "end_cm-1",
                "width_cm-1",
                "n_features",
                "min_fdr_q",
                "min_cluster_p_value",
                "max_abs_effect",
                "mean_abs_effect",
                "integrated_abs_effect",
                "effect_metric",
            ]
        )
    else:
        candidate_regions["effect_metric"] = np.where(
            candidate_regions["analysis_name"].eq("delta_moca_association"),
            "spearman_rho",
            "rank_biserial",
        )
        candidate_regions = _rank_candidate_regions(candidate_regions)

    combined_results = pd.concat(
        [
            delta_associations.assign(
                analysis_name="delta_moca_association",
                effect_metric="spearman_rho",
            ),
            group_comparison.assign(
                analysis_name="decline_group_comparison",
                effect_metric="rank_biserial",
            ),
        ],
        ignore_index=True,
        sort=False,
    )
    association_path = target_dir / "delta_moca_associations.csv"
    candidate_path = target_dir / "delta_moca_candidate_regions.csv"

    combined_results.to_csv(association_path, index=False)
    candidate_regions.to_csv(candidate_path, index=False)

    logger.info("Longitudinal Delta MoCA outputs written to %s", target_dir)
    return {
        "delta_moca_associations": association_path,
        "delta_moca_candidate_regions": candidate_path,
    }
