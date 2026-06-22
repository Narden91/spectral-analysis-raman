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

from src import config, get_cross_region_summary_dir
from src.analysis.statistics import fdr_correction

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
FEATURE_PREFIXES = ("1300__", "2850__")


def _cfg(cfg: dict | None = None) -> dict:
    return config if cfg is None else cfg


def _stratification_cfg(cfg: dict) -> dict:
    return cfg.get("analysis", {}).get("disease_stratification", {})


def _feature_metadata(columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in columns:
        block, wavenumber = column.split("__", 1)
        rows.append(
            {
                "feature_name": column,
                "spectral_block": block,
                "wavenumber": float(wavenumber),
            }
        )
    return pd.DataFrame(rows).sort_values(["spectral_block", "wavenumber"]).reset_index(drop=True)


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column.startswith(FEATURE_PREFIXES)
    ]


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


def _binary_feature_tests(
    frame: pd.DataFrame,
    feature_meta: pd.DataFrame,
    spec: StratificationSpec,
    min_subjects_per_group: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = frame.loc[frame[spec.variable].notna()].copy()
    subset[spec.variable] = subset[spec.variable].astype(str)

    counts = (
        subset[spec.variable]
        .value_counts()
        .reindex(spec.ordered_groups, fill_value=0)
        .rename_axis("group_name")
        .reset_index(name="n_subjects")
    )
    counts["group_variable"] = spec.variable
    counts["contrast_name"] = spec.label
    insufficient = counts["n_subjects"].lt(min_subjects_per_group).any()
    counts["status"] = np.where(insufficient, "insufficient sample size", "ok")
    counts["min_subjects_per_group"] = min_subjects_per_group

    if insufficient:
        return pd.DataFrame(
            [
                {
                    "contrast_name": spec.label,
                    "group_variable": spec.variable,
                    "group_a": spec.ordered_groups[0],
                    "group_b": spec.ordered_groups[1],
                    "test": "mannwhitney",
                    "status": "insufficient sample size",
                }
            ]
        ), counts

    group_a, group_b = spec.ordered_groups
    left = subset.loc[subset[spec.variable] == group_a, feature_meta["feature_name"]].to_numpy(dtype=float)
    right = subset.loc[subset[spec.variable] == group_b, feature_meta["feature_name"]].to_numpy(dtype=float)
    test_result = stats.mannwhitneyu(
        left,
        right,
        axis=0,
        alternative="two-sided",
    )
    u_statistics = np.asarray(test_result.statistic, dtype=float)
    p_values = np.asarray(test_result.pvalue, dtype=float)
    effects = np.asarray(_rank_biserial_effect(left, right, u_statistics), dtype=float)

    rows: list[dict[str, object]] = []
    for index, feature_row in feature_meta.iterrows():
        rows.append(
            {
                "contrast_name": spec.label,
                "group_variable": spec.variable,
                "group_a": group_a,
                "group_b": group_b,
                "test": "mannwhitney",
                "feature_name": feature_row["feature_name"],
                "spectral_block": feature_row["spectral_block"],
                "wavenumber": feature_row["wavenumber"],
                "n_group_a": left.shape[0],
                "n_group_b": right.shape[0],
                "u_statistic": float(u_statistics[index]),
                "p_value": float(p_values[index]),
                "effect_size": float(effects[index]),
                "effect_size_name": "rank_biserial",
                "status": "ok",
            }
        )

    tests = pd.DataFrame(rows)
    q_values, significant = fdr_correction(tests["p_value"].to_numpy(dtype=float))
    tests["fdr_q"] = q_values
    tests["significant_fdr"] = significant
    return tests, counts


def _cluster_masses(effects: np.ndarray, p_values: np.ndarray, alpha: float) -> list[tuple[int, int, float]]:
    active = (p_values < alpha) & np.isfinite(effects)
    clusters: list[tuple[int, int, float]] = []
    start: int | None = None
    sign = 0.0
    for index, is_active in enumerate(active):
        current_sign = np.sign(effects[index]) if is_active else 0.0
        if is_active and start is None:
            start = index
            sign = current_sign
            continue
        if is_active and start is not None and current_sign == sign:
            continue
        if start is not None:
            stop = index - 1 if not is_active else index - 1
            mass = float(np.nansum(np.abs(effects[start : stop + 1])))
            clusters.append((start, stop, mass))
            start = index if is_active else None
            sign = current_sign if is_active else 0.0
    if start is not None:
        stop = len(active) - 1
        mass = float(np.nansum(np.abs(effects[start : stop + 1])))
        clusters.append((start, stop, mass))
    return clusters


def _permutation_cluster_maxima(
    values: np.ndarray,
    labels: np.ndarray,
    n_permutations: int,
    alpha: float,
    rng: np.random.Generator,
) -> np.ndarray:
    n_left = int((labels == 0).sum())
    maxima = np.zeros(n_permutations, dtype=float)
    pooled = np.asarray(values, dtype=float)

    for perm_index in range(n_permutations):
        order = rng.permutation(len(labels))
        left_idx = order[:n_left]
        right_idx = order[n_left:]
        left = pooled[left_idx]
        right = pooled[right_idx]
        test_result = stats.mannwhitneyu(
            left,
            right,
            axis=0,
            alternative="two-sided",
        )
        p_values = np.asarray(test_result.pvalue, dtype=float)
        effects = np.asarray(
            _rank_biserial_effect(left, right, np.asarray(test_result.statistic, dtype=float)),
            dtype=float,
        )
        clusters = _cluster_masses(effects, p_values, alpha)
        maxima[perm_index] = max((mass for _, _, mass in clusters), default=0.0)
    return maxima


def _apply_cluster_permutation(
    tests: pd.DataFrame,
    frame: pd.DataFrame,
    spec: StratificationSpec,
    *,
    n_permutations: int,
    cluster_alpha: float,
    random_state: int,
) -> pd.DataFrame:
    if tests.empty or tests["status"].ne("ok").all():
        return tests

    out = tests.copy()
    out["cluster_id"] = pd.NA
    out["cluster_mass"] = np.nan
    out["cluster_p_value"] = np.nan
    rng = np.random.default_rng(random_state)

    subset = frame.loc[frame[spec.variable].isin(spec.ordered_groups)].copy()
    subset[spec.variable] = subset[spec.variable].astype(str)
    labels = np.where(subset[spec.variable] == spec.ordered_groups[0], 0, 1)

    for block, block_df in out.groupby("spectral_block", sort=False):
        ordered = block_df.sort_values("wavenumber").reset_index()
        effects = ordered["effect_size"].to_numpy(dtype=float)
        p_values = ordered["p_value"].to_numpy(dtype=float)
        clusters = _cluster_masses(effects, p_values, cluster_alpha)
        if not clusters:
            continue

        feature_names = ordered["feature_name"].tolist()
        values = subset[feature_names].to_numpy(dtype=float)
        maxima = _permutation_cluster_maxima(
            values,
            labels,
            n_permutations=n_permutations,
            alpha=cluster_alpha,
            rng=rng,
        )
        for cluster_index, (start, stop, mass) in enumerate(clusters, start=1):
            corrected = float((1 + np.sum(maxima >= mass)) / (n_permutations + 1))
            for row_index in ordered.loc[start : stop, "index"]:
                out.loc[row_index, "cluster_id"] = f"{block}_C{cluster_index}"
                out.loc[row_index, "cluster_mass"] = mass
                out.loc[row_index, "cluster_p_value"] = corrected
    return out


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


def _candidate_regions(
    tests: pd.DataFrame,
    *,
    min_width_cm: float = 8.0,
    merge_gap_cm: float = 6.0,
    fallback_quantile: float = 0.9,
) -> pd.DataFrame:
    columns = [
        "contrast_name",
        "group_variable",
        "spectral_block",
        "candidate_region_id",
        "selection_basis",
        "start_cm-1",
        "end_cm-1",
        "width_cm-1",
        "n_features",
        "min_fdr_q",
        "min_cluster_p_value",
        "max_abs_effect",
        "mean_abs_effect",
        "integrated_abs_effect",
    ]
    rows: list[dict[str, object]] = []

    def _append_region(
        *,
        contrast_name: str,
        group_variable: str,
        block: str,
        selection_basis: str,
        region_index: int,
        window: pd.DataFrame,
    ) -> None:
        rows.append(
            {
                "contrast_name": contrast_name,
                "group_variable": group_variable,
                "spectral_block": block,
                "candidate_region_id": f"{block}_R{region_index}",
                "selection_basis": selection_basis,
                "start_cm-1": float(window["wavenumber"].iloc[0]),
                "end_cm-1": float(window["wavenumber"].iloc[-1]),
                "width_cm-1": float(window["wavenumber"].iloc[-1] - window["wavenumber"].iloc[0]),
                "n_features": int(len(window)),
                "min_fdr_q": float(window["fdr_q"].min()) if "fdr_q" in window.columns else np.nan,
                "min_cluster_p_value": float(window["cluster_p_value"].min()) if "cluster_p_value" in window.columns else np.nan,
                "max_abs_effect": float(window["effect_size"].abs().max()) if "effect_size" in window.columns else np.nan,
                "mean_abs_effect": float(window["effect_size"].abs().mean()) if "effect_size" in window.columns else np.nan,
                "integrated_abs_effect": float(
                    np.trapz(
                        window["effect_size"].abs().to_numpy(dtype=float),
                        window["wavenumber"].to_numpy(dtype=float),
                    )
                ) if "effect_size" in window.columns else np.nan,
            }
        )

    ok = tests.loc[tests["status"].eq("ok")].copy()
    if ok.empty:
        return pd.DataFrame(columns=columns)

    for (contrast_name, block), block_df in ok.groupby(["contrast_name", "spectral_block"], sort=False):
        ordered_full = block_df.sort_values("wavenumber").reset_index(drop=True)
        candidate_mask = ordered_full["significant_fdr"].astype("boolean").fillna(False).to_numpy(dtype=bool)
        if "cluster_p_value" in ordered_full.columns:
            candidate_mask = candidate_mask | ordered_full["cluster_p_value"].lt(0.05).fillna(False).to_numpy(dtype=bool)

        if candidate_mask.any():
            ordered = ordered_full.loc[candidate_mask].copy().reset_index(drop=True)
            selection_basis = "significant_or_cluster"
        else:
            abs_effect = ordered_full["effect_size"].abs().to_numpy(dtype=float)
            finite = np.isfinite(abs_effect)
            if not finite.any():
                continue
            threshold = float(np.nanquantile(abs_effect[finite], fallback_quantile))
            fallback_mask = abs_effect >= threshold
            ordered = ordered_full.loc[fallback_mask].copy().reset_index(drop=True)
            selection_basis = "top_abs_effect_fallback"
            if ordered.empty:
                peak_index = int(np.nanargmax(abs_effect))
                peak_wavenumber = float(ordered_full.loc[peak_index, "wavenumber"])
                half_width = max(min_width_cm / 2.0, merge_gap_cm)
                ordered = ordered_full.loc[
                    ordered_full["wavenumber"].between(peak_wavenumber - half_width, peak_wavenumber + half_width)
                ].copy().reset_index(drop=True)
                selection_basis = "top_peak_window_fallback"

        start = 0
        region_index = 0
        for index in range(1, len(ordered) + 1):
            is_break = index == len(ordered) or (ordered.loc[index, "wavenumber"] - ordered.loc[index - 1, "wavenumber"]) > merge_gap_cm
            if not is_break:
                continue
            window = ordered.iloc[start:index].copy()
            width = float(window["wavenumber"].iloc[-1] - window["wavenumber"].iloc[0])
            if width >= min_width_cm or len(window) > 1:
                region_index += 1
                _append_region(
                    contrast_name=contrast_name,
                    group_variable=str(window["group_variable"].iloc[0]),
                    block=block,
                    selection_basis=selection_basis,
                    region_index=region_index,
                    window=window,
                )
            start = index

        if region_index == 0:
            abs_effect = ordered_full["effect_size"].abs().to_numpy(dtype=float)
            if np.isfinite(abs_effect).any():
                peak_index = int(np.nanargmax(abs_effect))
                peak_wavenumber = float(ordered_full.loc[peak_index, "wavenumber"])
                half_width = max(min_width_cm / 2.0, merge_gap_cm)
                window = ordered_full.loc[
                    ordered_full["wavenumber"].between(peak_wavenumber - half_width, peak_wavenumber + half_width)
                ].copy().reset_index(drop=True)
                _append_region(
                    contrast_name=contrast_name,
                    group_variable=str(window["group_variable"].iloc[0]),
                    block=block,
                    selection_basis="top_peak_window_fallback",
                    region_index=1,
                    window=window,
                )
    return pd.DataFrame(rows, columns=columns)


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

    matrix_path = (
        Path(feature_matrix_path)
        if feature_matrix_path is not None
        else get_cross_region_summary_dir() / "subject_feature_matrix_whole.parquet"
    )
    frame = pd.read_parquet(matrix_path)
    frame["subject_id"] = frame["subject_id"].astype(str)
    if frame["subject_id"].duplicated().any():
        raise ValueError("Disease stratification requires one row per subject")

    disease_only = frame.copy()
    if "is_control" in disease_only.columns:
        disease_only = disease_only.loc[~disease_only["is_control"].fillna(False).astype(bool)].copy()
    elif "condition" in disease_only.columns:
        disease_only = disease_only.loc[disease_only["condition"].astype(str).str.lower() != "control"].copy()

    feature_columns = _feature_columns(disease_only)
    if not feature_columns:
        raise ValueError("Disease stratification requires whole-spectrum subject feature columns")
    feature_meta = _feature_metadata(feature_columns)
    logger.info(
        "Disease stratification: %d disease subjects, %d whole-spectrum features",
        len(disease_only),
        len(feature_meta),
    )

    min_subjects = int(strat_cfg.get("min_subjects_per_group", 5))
    cluster_alpha = float(strat_cfg.get("cluster_alpha", 0.05))
    n_permutations = int(strat_cfg.get("n_permutations", 1000))
    random_state = int(strat_cfg.get("random_state", 42))

    test_frames: list[pd.DataFrame] = []
    count_frames: list[pd.DataFrame] = []

    for spec in BINARY_SPECS:
        tests, counts = _binary_feature_tests(disease_only, feature_meta, spec, min_subjects)
        tests = _apply_cluster_permutation(
            tests,
            disease_only,
            spec,
            n_permutations=n_permutations,
            cluster_alpha=cluster_alpha,
            random_state=random_state,
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
    candidate_regions = _candidate_regions(
        tests,
        min_width_cm=float(strat_cfg.get("candidate_region_min_width_cm", 8.0)),
        merge_gap_cm=float(strat_cfg.get("candidate_region_merge_gap_cm", 6.0)),
        fallback_quantile=float(strat_cfg.get("candidate_region_fallback_quantile", 0.9)),
    )

    target_dir = (
        Path(output_dir)
        if output_dir is not None
        else get_cross_region_summary_dir() / "disease_stratification"
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
