"""Disease-internal subject-level stratification analysis."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src import get_cross_region_phase_dir
from src import resolve_cfg as _cfg
from src.analysis.clinical_feature_associations import (
    compute_binary_group_feature_tests,
    feature_columns,
    feature_metadata,
    filter_disease_subjects,
    load_subject_feature_matrix,
)
from src.analysis.spectral_regions import build_candidate_regions
from src.analysis.statistics import fdr_correction
from src.statistics.cluster_permutation import apply_binary_cluster_permutation

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StratificationSpec:
    variable: str
    label: str
    ordered_groups: tuple[str, ...]
    kind: str


BINARY_SPECS = (
    StratificationSpec(
        variable="moca_severity_group",
        label="low_vs_high_moca",
        ordered_groups=("low_moca", "high_moca"),
        kind="binary",
    ),
    StratificationSpec(
        variable="delta_moca_group",
        label="decline_vs_stable_or_improved",
        ordered_groups=("decline", "stable_or_improved"),
        kind="binary",
    ),
)
LESION_SPEC = StratificationSpec(
    variable="lesion_location_group",
    label="lesion_location_groups",
    ordered_groups=("left", "right", "posterior", "other"),
    kind="multigroup",
)


def _stratification_cfg(cfg: dict) -> dict:
    return cfg.get("analysis", {}).get("disease_stratification", {})


def _rank_biserial_effect(
    left: np.ndarray,
    right: np.ndarray,
    u_statistic: float | np.ndarray,
) -> float | np.ndarray:
    denominator = float(len(left) * len(right))
    if denominator == 0:
        if np.ndim(u_statistic) == 0:
            return float("nan")
        return np.full(np.asarray(u_statistic).shape, np.nan, dtype=float)
    effect = (2.0 * np.asarray(u_statistic, dtype=float) / denominator) - 1.0
    if np.ndim(effect) == 0:
        return float(effect)
    return effect


def _apply_cluster_permutation(
    tests: pd.DataFrame,
    frame: pd.DataFrame,
    spec: StratificationSpec,
    *,
    n_permutations: int,
    cluster_alpha: float,
    random_state: int,
    n_jobs: int = 1,
) -> pd.DataFrame:
    return apply_binary_cluster_permutation(
        tests,
        frame,
        group_column=spec.variable,
        ordered_groups=(spec.ordered_groups[0], spec.ordered_groups[1]),
        n_permutations=n_permutations,
        cluster_alpha=cluster_alpha,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def _kruskal_effect_size(h_statistic: float, n_subjects: int, n_groups: int) -> float:
    denominator = n_subjects - n_groups
    if denominator <= 0:
        return float("nan")
    return float(max((h_statistic - n_groups + 1.0) / denominator, 0.0))


def _multigroup_feature_tests(
    frame: pd.DataFrame,
    feature_meta: pd.DataFrame,
    spec: StratificationSpec,
    min_subjects_per_group: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = frame.loc[frame[spec.variable].notna()].copy()
    subset[spec.variable] = subset[spec.variable].astype(str)
    present_groups = [
        group for group in spec.ordered_groups if group in subset[spec.variable].unique()
    ]
    counts = (
        subset[spec.variable]
        .value_counts()
        .reindex(present_groups, fill_value=0)
        .rename_axis("group_name")
        .reset_index(name="n_subjects")
    )
    counts["group_variable"] = spec.variable
    counts["contrast_name"] = spec.label
    insufficient = counts["n_subjects"].lt(min_subjects_per_group).any() or len(present_groups) < 2
    counts["status"] = np.where(insufficient, "insufficient sample size", "ok")
    counts["min_subjects_per_group"] = min_subjects_per_group

    if insufficient:
        return pd.DataFrame(
            [
                {
                    "contrast_name": spec.label,
                    "group_variable": spec.variable,
                    "test": "kruskal",
                    "status": "insufficient sample size",
                }
            ]
        ), counts

    group_arrays = {
        group: subset.loc[subset[spec.variable] == group, feature_meta["feature_name"]].to_numpy(dtype=float)
        for group in present_groups
    }
    rows: list[dict[str, object]] = []
    for index, feature_row in feature_meta.iterrows():
        series = [group_arrays[group][:, index] for group in present_groups]
        h_statistic, p_value = stats.kruskal(*series)
        rows.append(
            {
                "contrast_name": spec.label,
                "group_variable": spec.variable,
                "test": "kruskal",
                "feature_name": feature_row["feature_name"],
                "spectral_block": feature_row["spectral_block"],
                "wavenumber": feature_row["wavenumber"],
                "h_statistic": float(h_statistic),
                "p_value": float(p_value),
                "effect_size": _kruskal_effect_size(float(h_statistic), len(subset), len(present_groups)),
                "effect_size_name": "epsilon_squared",
                "status": "ok",
            }
        )

    tests = pd.DataFrame(rows)
    q_values, significant = fdr_correction(tests["p_value"].to_numpy(dtype=float))
    tests["fdr_q"] = q_values
    tests["significant_fdr"] = significant

    pairwise_rows: list[dict[str, object]] = []
    for left_index, group_a in enumerate(present_groups):
        for group_b in present_groups[left_index + 1 :]:
            left = group_arrays[group_a]
            right = group_arrays[group_b]
            test_result = stats.mannwhitneyu(
                left,
                right,
                axis=0,
                alternative="two-sided",
            )
            u_statistics = np.asarray(test_result.statistic, dtype=float)
            p_values = np.asarray(test_result.pvalue, dtype=float)
            effects = np.asarray(_rank_biserial_effect(left, right, u_statistics), dtype=float)
            for feature_index, feature_row in feature_meta.iterrows():
                pairwise_rows.append(
                    {
                        "contrast_name": f"{spec.label}__{group_a}_vs_{group_b}",
                        "group_variable": spec.variable,
                        "group_a": group_a,
                        "group_b": group_b,
                        "test": "mannwhitney_posthoc",
                        "feature_name": feature_row["feature_name"],
                        "spectral_block": feature_row["spectral_block"],
                        "wavenumber": feature_row["wavenumber"],
                        "n_group_a": left.shape[0],
                        "n_group_b": right.shape[0],
                        "u_statistic": float(u_statistics[feature_index]),
                        "p_value": float(p_values[feature_index]),
                        "effect_size": float(effects[feature_index]),
                        "effect_size_name": "rank_biserial",
                        "status": "ok",
                    }
                )
    if pairwise_rows:
        pairwise = pd.DataFrame(pairwise_rows)
        q_values, significant = fdr_correction(pairwise["p_value"].to_numpy(dtype=float))
        pairwise["fdr_q"] = q_values
        pairwise["significant_fdr"] = significant
        tests = pd.concat([tests, pairwise], ignore_index=True, sort=False)
    return tests, counts


def _plot_effects(tests: pd.DataFrame, output_dir: Path) -> None:
    ok = tests.loc[tests["status"].eq("ok")].copy()
    if ok.empty or "effect_size" not in ok.columns:
        return

    for contrast_name, contrast_df in ok.groupby("contrast_name", sort=False):
        if contrast_df["test"].iloc[0] not in {"mannwhitney", "mannwhitney_posthoc", "kruskal"}:
            continue
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=False)
        for axis, block in zip(axes, ("1300", "2850")):
            block_df = contrast_df.loc[contrast_df["spectral_block"] == block].sort_values("wavenumber")
            if block_df.empty:
                axis.set_visible(False)
                continue
            axis.plot(block_df["wavenumber"], block_df["effect_size"], color="#264653", linewidth=1.0)
            sig = block_df.loc[block_df["significant_fdr"].fillna(False)]
            if not sig.empty:
                axis.scatter(sig["wavenumber"], sig["effect_size"], color="#E76F51", s=14, label="FDR")
            if "cluster_p_value" in block_df.columns:
                cluster_sig = block_df.loc[block_df["cluster_p_value"].lt(0.05).fillna(False)]
                if not cluster_sig.empty:
                    axis.scatter(cluster_sig["wavenumber"], cluster_sig["effect_size"], color="#2A9D8F", s=18, label="Cluster")
            axis.axhline(0.0, color="black", linestyle="--", linewidth=0.6)
            axis.set_title(f"{contrast_name} | {block}")
            axis.set_ylabel("Effect size")
            handles, labels = axis.get_legend_handles_labels()
            if handles:
                axis.legend(handles, labels, loc="upper right", fontsize=8)
        axes[-1].set_xlabel("Raman shift (cm-1)")
        fig.tight_layout()
        fig.savefig(output_dir / f"{contrast_name}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def run_disease_stratification(
    *,
    feature_matrix_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    cfg: dict | None = None,
) -> dict[str, Path]:
    cfg = _cfg(cfg)
    strat_cfg = _stratification_cfg(cfg)
    if not bool(strat_cfg.get("enabled", True)):
        return {}

    frame = load_subject_feature_matrix(feature_matrix_path=feature_matrix_path)
    disease_only = filter_disease_subjects(frame)

    spectral_feature_columns = feature_columns(disease_only)
    if not spectral_feature_columns:
        raise ValueError("Disease stratification requires whole-spectrum subject feature columns")
    feature_meta = feature_metadata(spectral_feature_columns)
    logger.info(
        "Disease stratification: %d disease subjects, %d whole-spectrum features",
        len(disease_only),
        len(feature_meta),
    )

    min_subjects = int(strat_cfg.get("min_subjects_per_group", 5))
    cluster_alpha = float(strat_cfg.get("cluster_alpha", 0.05))
    n_permutations = int(strat_cfg.get("n_permutations", 1000))
    n_jobs = int(strat_cfg.get("n_jobs", 1))
    random_state = int(strat_cfg.get("random_state", 42))

    test_frames: list[pd.DataFrame] = []
    count_frames: list[pd.DataFrame] = []

    for spec in BINARY_SPECS:
        tests, counts = compute_binary_group_feature_tests(
            disease_only,
            feature_meta,
            group_column=spec.variable,
            ordered_groups=(spec.ordered_groups[0], spec.ordered_groups[1]),
            contrast_name=spec.label,
            min_subjects_per_group=min_subjects,
            fdr_alpha=0.05,
        )
        tests = _apply_cluster_permutation(
            tests,
            disease_only,
            spec,
            n_permutations=n_permutations,
            cluster_alpha=cluster_alpha,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        test_frames.append(tests)
        count_frames.append(counts)
        logger.info("Disease stratification: %s complete", spec.label)

    lesion_tests, lesion_counts = _multigroup_feature_tests(
        disease_only,
        feature_meta,
        LESION_SPEC,
        min_subjects,
    )
    test_frames.append(lesion_tests)
    count_frames.append(lesion_counts)
    logger.info("Disease stratification: %s complete", LESION_SPEC.label)

    tests = pd.concat(test_frames, ignore_index=True, sort=False)
    counts = pd.concat(count_frames, ignore_index=True, sort=False)
    candidate_regions = build_candidate_regions(
        tests,
        group_columns=("contrast_name", "group_variable"),
        effect_column="effect_size",
        min_width_cm=float(strat_cfg.get("candidate_region_min_width_cm", 8.0)),
        merge_gap_cm=float(strat_cfg.get("candidate_region_merge_gap_cm", 6.0)),
        fallback_quantile=float(strat_cfg.get("candidate_region_fallback_quantile", 0.9)),
    )

    target_dir = (
        Path(output_dir)
        if output_dir is not None
        else get_cross_region_phase_dir("disease_stratification")
    )
    plots_dir = target_dir / "stratification_effect_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for stale_plot in plots_dir.glob("*.png"):
        stale_plot.unlink()

    tests_path = target_dir / "stratification_wavenumber_tests.csv"
    candidates_path = target_dir / "stratification_candidate_regions.csv"
    obsolete_paths = [
        target_dir / "stratification_group_counts.csv",
    ]
    for obsolete_path in obsolete_paths:
        if obsolete_path.exists():
            obsolete_path.unlink()
    tests.to_csv(tests_path, index=False)
    candidate_regions.to_csv(candidates_path, index=False)
    _plot_effects(tests, plots_dir)

    logger.info("Disease stratification outputs written to %s", target_dir)
    return {
        "stratification_wavenumber_tests": tests_path,
        "stratification_candidate_regions": candidates_path,
        "stratification_effect_plots": plots_dir,
    }
