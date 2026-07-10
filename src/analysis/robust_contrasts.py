"""Exploratory robust statistics for the provenance-aware fixed panel."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

from src import config, get_confirmatory_panel_dir
from src.analysis.clinical_feature_associations import filter_disease_subjects, load_subject_feature_matrix
from src.analysis.io_utils import write_csv_atomic
from src.analysis.preregistered_panel import panel_cfg
from src.analysis.statistics import (
    cliffs_delta,
    cohens_d,
    fdr_correction,
    partial_spearman,
    variance_homogeneity_test,
)

logger = logging.getLogger(__name__)


def _cfg(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    return config if cfg is None else cfg


def _resampling_settings(panel: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(panel.get("n_bootstrap", 2000)),
        int(panel.get("n_permutations", 1000)),
        int(panel.get("random_state", 42)),
        int(panel.get("n_jobs", 1)),
    )


def _load_panel_and_clinical(
    *,
    feature_matrix_path: str | Path | None,
    panel_matrix_path: str | Path,
) -> pd.DataFrame:
    clinical = filter_disease_subjects(load_subject_feature_matrix(feature_matrix_path=feature_matrix_path))
    panel = pd.read_csv(panel_matrix_path)
    if "subject_id" not in panel.columns:
        raise ValueError("Panel feature matrix is missing subject_id")

    clinical["subject_id"] = clinical["subject_id"].astype(str)
    panel["subject_id"] = panel["subject_id"].astype(str)
    return clinical.merge(panel, on="subject_id", how="inner", validate="one_to_one")


def _panel_features(frame: pd.DataFrame, panel: dict[str, Any]) -> list[str]:
    registered = list(panel.get("indices", {}).keys()) + [
        name
        for name in panel.get("integrals", {})
        if name != "lipid_carbonyl_1720_fallback"
    ]
    return [feature for feature in registered if feature in frame.columns]


def _feature_family(feature: str, panel: dict[str, Any]) -> str:
    if feature in panel.get("indices", {}):
        return "named_index"
    if feature in panel.get("integrals", {}):
        return "integral"
    return "shape_or_external"


def _contrast_specs(panel: dict[str, Any]) -> list[tuple[str, str, str, str]]:
    specs: list[tuple[str, str, str, str]] = []
    for contrast in panel.get("endpoints", {}).get("contrasts", []):
        ordered_groups = [str(item) for item in contrast.get("ordered_groups", [])]
        if len(ordered_groups) != 2:
            continue
        specs.append(
            (
                str(contrast.get("name", "contrast")),
                str(contrast.get("group_variable", "")),
                ordered_groups[0],
                ordered_groups[1],
            )
        )
    return specs


def _feature_group_values(
    frame: pd.DataFrame,
    *,
    group_column: str,
    left_label: str,
    right_label: str,
    feature: str,
) -> tuple[np.ndarray, np.ndarray]:
    groups = frame[group_column].astype("string")
    left = frame.loc[groups == left_label, feature].to_numpy(dtype=float)
    right = frame.loc[groups == right_label, feature].to_numpy(dtype=float)
    return left[np.isfinite(left)], right[np.isfinite(right)]


def _base_contrast_row(
    *,
    contrast_name: str,
    group_column: str,
    left_label: str,
    right_label: str,
    feature: str,
    feature_family: str,
    left: np.ndarray,
    right: np.ndarray,
    min_n: int,
) -> dict[str, object]:
    return {
        "contrast_name": contrast_name,
        "group_variable": group_column,
        "group_a": left_label,
        "group_b": right_label,
        "feature_name": feature,
        "feature_family": feature_family,
        "n_group_a": int(left.size),
        "n_group_b": int(right.size),
        "status": "ok" if left.size >= min_n and right.size >= min_n else "insufficient_n",
        "u_statistic": np.nan,
        "p_value": np.nan,
        "cliffs_delta": np.nan,
        "cohens_d": np.nan,
        "bootstrap_ci_lower": np.nan,
        "bootstrap_ci_upper": np.nan,
        "perm_p_value": np.nan,
        "mean_group_a": np.nan,
        "mean_group_b": np.nan,
    }


def _p_values(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        return np.full(len(frame), np.nan, dtype=float)
    return frame[column].to_numpy(dtype=float)


def _add_fdr_columns(
    frame: pd.DataFrame,
    *,
    p_column: str,
    q_column: str,
    significant_column: str,
    alpha: float,
) -> pd.DataFrame:
    q_values, significant = fdr_correction(_p_values(frame, p_column), alpha=alpha)
    frame[q_column] = q_values
    frame[significant_column] = significant
    return frame


def _bootstrap_ci(
    left: np.ndarray,
    right: np.ndarray,
    statistic_fn,
    *,
    n_bootstrap: int,
    random_state: int,
    n_jobs: int = 1,
) -> tuple[float, float]:
    left = left[np.isfinite(left)]
    right = right[np.isfinite(right)]
    if left.size == 0 or right.size == 0 or n_bootstrap <= 0:
        return np.nan, np.nan

    rng = np.random.default_rng(random_state)
    n_jobs = 1 if n_jobs == 0 else n_jobs
    if n_jobs != 1:
        seeds = rng.integers(0, np.iinfo(np.int32).max, size=n_bootstrap, dtype=np.int64)
        estimates = np.array(
            Parallel(n_jobs=n_jobs)(
                delayed(_bootstrap_two_group_statistic)(left, right, statistic_fn, int(seed))
                for seed in seeds
            ),
            dtype=float,
        )
    else:
        estimates = np.empty(n_bootstrap, dtype=float)
        for i in range(n_bootstrap):
            boot_left = rng.choice(left, size=left.size, replace=True)
            boot_right = rng.choice(right, size=right.size, replace=True)
            estimates[i] = statistic_fn(boot_left, boot_right)
    finite = estimates[np.isfinite(estimates)]
    if finite.size == 0:
        return np.nan, np.nan
    lower, upper = np.percentile(finite, [2.5, 97.5])
    return float(lower), float(upper)


def _bootstrap_partial_spearman_ci(
    frame: pd.DataFrame,
    *,
    feature: str,
    endpoint: str,
    covariates: list[str],
    n_bootstrap: int,
    random_state: int,
    n_jobs: int = 1,
) -> tuple[float, float]:
    columns = [feature, endpoint, *covariates]
    complete = frame[columns].dropna()
    if len(complete) < 3 or n_bootstrap <= 0:
        return np.nan, np.nan

    rng = np.random.default_rng(random_state)
    n_jobs = 1 if n_jobs == 0 else n_jobs
    if n_jobs != 1:
        seeds = rng.integers(0, np.iinfo(np.int32).max, size=n_bootstrap, dtype=np.int64)
        estimates = np.array(
            Parallel(n_jobs=n_jobs)(
                delayed(_bootstrap_partial_spearman_statistic)(
                    complete,
                    feature,
                    endpoint,
                    covariates,
                    int(seed),
                )
                for seed in seeds
            ),
            dtype=float,
        )
    else:
        estimates = np.empty(n_bootstrap, dtype=float)
        for i in range(n_bootstrap):
            sample = complete.iloc[rng.integers(0, len(complete), size=len(complete))]
            estimates[i] = partial_spearman(
                sample[feature],
                sample[endpoint],
                sample[covariates],
            )[0]
    finite = estimates[np.isfinite(estimates)]
    if finite.size == 0:
        return np.nan, np.nan
    lower, upper = np.percentile(finite, [2.5, 97.5])
    return float(lower), float(upper)


def _permutation_p_value(
    left: np.ndarray,
    right: np.ndarray,
    *,
    n_permutations: int,
    random_state: int,
    n_jobs: int = 1,
) -> float:
    left = left[np.isfinite(left)]
    right = right[np.isfinite(right)]
    if left.size == 0 or right.size == 0 or n_permutations <= 0:
        return np.nan

    observed = abs(cliffs_delta(left, right))
    pooled = np.concatenate([left, right])
    n_left = left.size
    rng = np.random.default_rng(random_state)
    n_jobs = 1 if n_jobs == 0 else n_jobs
    if n_jobs != 1:
        seeds = rng.integers(0, np.iinfo(np.int32).max, size=n_permutations, dtype=np.int64)
        exceedances = sum(
            Parallel(n_jobs=n_jobs)(
                delayed(_permuted_cliffs_delta_exceeds)(pooled, n_left, observed, int(seed))
                for seed in seeds
            )
        )
    else:
        exceedances = 0
        for _ in range(n_permutations):
            permuted = rng.permutation(pooled)
            permuted_delta = abs(cliffs_delta(permuted[:n_left], permuted[n_left:]))
            if permuted_delta >= observed - 1e-12:
                exceedances += 1
    return float((exceedances + 1) / (n_permutations + 1))


def _permutation_partial_spearman_p_value(
    frame: pd.DataFrame,
    *,
    feature: str,
    endpoint: str,
    covariates: list[str],
    n_permutations: int,
    random_state: int,
    n_jobs: int = 1,
) -> float:
    columns = [feature, endpoint, *covariates]
    complete = frame[columns].dropna()
    if len(complete) < 3 or n_permutations <= 0:
        return np.nan

    observed = abs(partial_spearman(complete[feature], complete[endpoint], complete[covariates])[0])
    if not np.isfinite(observed):
        return np.nan

    rng = np.random.default_rng(random_state)
    endpoint_values = complete[endpoint].to_numpy(dtype=float)
    n_jobs = 1 if n_jobs == 0 else n_jobs
    if n_jobs != 1:
        seeds = rng.integers(0, np.iinfo(np.int32).max, size=n_permutations, dtype=np.int64)
        exceedances = sum(
            Parallel(n_jobs=n_jobs)(
                delayed(_permuted_partial_spearman_exceeds)(
                    complete,
                    feature,
                    endpoint,
                    covariates,
                    endpoint_values,
                    observed,
                    int(seed),
                )
                for seed in seeds
            )
        )
    else:
        exceedances = 0
        for _ in range(n_permutations):
            shuffled = complete.copy()
            shuffled[endpoint] = rng.permutation(endpoint_values)
            permuted = abs(partial_spearman(shuffled[feature], shuffled[endpoint], shuffled[covariates])[0])
            if np.isfinite(permuted) and permuted >= observed - 1e-12:
                exceedances += 1
    return float((exceedances + 1) / (n_permutations + 1))


def _bootstrap_two_group_statistic(
    left: np.ndarray,
    right: np.ndarray,
    statistic_fn,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    boot_left = rng.choice(left, size=left.size, replace=True)
    boot_right = rng.choice(right, size=right.size, replace=True)
    return float(statistic_fn(boot_left, boot_right))


def _bootstrap_partial_spearman_statistic(
    complete: pd.DataFrame,
    feature: str,
    endpoint: str,
    covariates: list[str],
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    sample = complete.iloc[rng.integers(0, len(complete), size=len(complete))]
    return float(partial_spearman(sample[feature], sample[endpoint], sample[covariates])[0])


def _permuted_cliffs_delta_exceeds(
    pooled: np.ndarray,
    n_left: int,
    observed: float,
    seed: int,
) -> bool:
    rng = np.random.default_rng(seed)
    permuted = rng.permutation(pooled)
    permuted_delta = abs(cliffs_delta(permuted[:n_left], permuted[n_left:]))
    return bool(permuted_delta >= observed - 1e-12)


def _permuted_partial_spearman_exceeds(
    complete: pd.DataFrame,
    feature: str,
    endpoint: str,
    covariates: list[str],
    endpoint_values: np.ndarray,
    observed: float,
    seed: int,
) -> bool:
    rng = np.random.default_rng(seed)
    shuffled = complete.copy()
    shuffled[endpoint] = rng.permutation(endpoint_values)
    permuted = abs(partial_spearman(shuffled[feature], shuffled[endpoint], shuffled[covariates])[0])
    return bool(np.isfinite(permuted) and permuted >= observed - 1e-12)


def run_anchor_test(
    frame: pd.DataFrame,
    *,
    cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    cfg = _cfg(cfg)
    panel = panel_cfg(cfg)
    primary = panel.get("primary_hypothesis", {})
    feature = str(primary.get("feature", "oxidative_index"))
    endpoint = str(primary.get("endpoint", "delta_moca"))
    covariates = [str(item) for item in primary.get("covariates", panel.get("covariates", []))]
    n_bootstrap, n_permutations, random_state, n_jobs = _resampling_settings(panel)

    row: dict[str, object] = {
        "analysis_order": 1,
        "feature_name": feature,
        "endpoint": endpoint,
        "test": "partial_spearman",
        "covariates": ",".join(covariates),
        "direction": primary.get("direction", "negative"),
        "status": "ok",
    }
    required = [feature, endpoint, *covariates]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        row["status"] = f"missing columns: {','.join(missing)}"
        return pd.DataFrame([row])

    rho, p_value, n = partial_spearman(frame[feature], frame[endpoint], frame[covariates])
    ci_lower, ci_upper = _bootstrap_partial_spearman_ci(
        frame,
        feature=feature,
        endpoint=endpoint,
        covariates=covariates,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    perm_p = _permutation_partial_spearman_p_value(
        frame,
        feature=feature,
        endpoint=endpoint,
        covariates=covariates,
        n_permutations=n_permutations,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    row.update(
        {
            "rho": rho,
            "p_value": p_value,
            "bootstrap_ci_lower": ci_lower,
            "bootstrap_ci_upper": ci_upper,
            "perm_p_value": perm_p,
            "n": n,
        }
    )

    trajectory = next((spec for spec in _contrast_specs(panel) if spec[0] == "trajectory_extreme"), None)
    if trajectory is not None and feature in frame.columns and trajectory[1] in frame.columns:
        _, group_column, left_label, right_label = trajectory
        left, right = _feature_group_values(
            frame,
            group_column=group_column,
            left_label=left_label,
            right_label=right_label,
            feature=feature,
        )
        row["trajectory_cliffs_delta"] = cliffs_delta(left, right)
        row["trajectory_n_group_a"] = int(left.size)
        row["trajectory_n_group_b"] = int(right.size)

    return pd.DataFrame([row])


def run_panel_contrasts(
    frame: pd.DataFrame,
    *,
    cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    cfg = _cfg(cfg)
    panel = panel_cfg(cfg)
    features = _panel_features(frame, panel)
    n_bootstrap, n_permutations, random_state, n_jobs = _resampling_settings(panel)
    min_n = int(panel.get("min_subjects_per_group", 5))

    rows: list[dict[str, object]] = []
    for contrast_name, group_column, left_label, right_label in _contrast_specs(panel):
        if group_column not in frame.columns:
            continue
        for feature_index, feature in enumerate(features):
            left, right = _feature_group_values(
                frame,
                group_column=group_column,
                left_label=left_label,
                right_label=right_label,
                feature=feature,
            )
            row = _base_contrast_row(
                contrast_name=contrast_name,
                group_column=group_column,
                left_label=left_label,
                right_label=right_label,
                feature=feature,
                feature_family=_feature_family(feature, panel),
                left=left,
                right=right,
                min_n=min_n,
            )
            if row["status"] == "ok":
                test = stats.mannwhitneyu(left, right, alternative="two-sided")
                ci_lower, ci_upper = _bootstrap_ci(
                    left,
                    right,
                    cliffs_delta,
                    n_bootstrap=n_bootstrap,
                    random_state=random_state + feature_index,
                    n_jobs=n_jobs,
                )
                row.update(
                    {
                        "u_statistic": float(test.statistic),
                        "p_value": float(test.pvalue),
                        "cliffs_delta": cliffs_delta(left, right),
                        "cohens_d": cohens_d(left, right),
                        "bootstrap_ci_lower": ci_lower,
                        "bootstrap_ci_upper": ci_upper,
                        "perm_p_value": _permutation_p_value(
                            left,
                            right,
                            n_permutations=n_permutations,
                            random_state=random_state + feature_index,
                            n_jobs=n_jobs,
                        ),
                        "mean_group_a": float(np.mean(left)),
                        "mean_group_b": float(np.mean(right)),
                    }
                )
            rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = _add_fdr_columns(
        result,
        p_column="p_value",
        q_column="fdr_q",
        significant_column="significant_fdr",
        alpha=float(panel.get("fdr_alpha", 0.05)),
    )
    result["raw_p"] = pd.to_numeric(
        result.get("p_value", pd.Series(np.nan, index=result.index)), errors="coerce"
    )
    result["within_scan_p"] = result["raw_p"]
    result["family_adjusted_p"] = result["fdr_q"]
    result["family_id"] = "fixed_panel_group_contrasts"
    result["inference_role"] = "exploratory"
    return result


def run_continuous_associations(
    frame: pd.DataFrame,
    *,
    cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    cfg = _cfg(cfg)
    panel = panel_cfg(cfg)
    features = _panel_features(frame, panel)
    endpoints = [str(item) for item in panel.get("endpoints", {}).get("continuous", [])]
    covariates = [str(item) for item in panel.get("covariates", [])]
    min_subjects = int(panel.get("min_subjects_continuous", 20))

    rows: list[dict[str, object]] = []
    for endpoint in endpoints:
        for feature in features:
            row = {
                "endpoint": endpoint,
                "feature_name": feature,
                "feature_family": _feature_family(feature, panel),
                "covariates": ",".join(covariates),
                "status": "ok",
            }
            required = [feature, endpoint, *covariates]
            missing = [column for column in required if column not in frame.columns]
            if missing:
                row["status"] = "missing_required_data"
                row["missing_columns"] = ",".join(missing)
                rows.append(row)
                continue

            complete = frame[required].dropna()
            row["n"] = int(len(complete))
            if len(complete) < min_subjects:
                row["status"] = "insufficient_n"
                rows.append(row)
                continue

            raw = stats.spearmanr(complete[feature], complete[endpoint])
            rho, p_value, n = partial_spearman(complete[feature], complete[endpoint], complete[covariates])
            row.update(
                {
                    "raw_spearman_rho": float(raw.statistic),
                    "raw_spearman_p_value": float(raw.pvalue),
                    "partial_spearman_rho": rho,
                    "partial_spearman_p_value": p_value,
                    "n": n,
                }
            )
            rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    fdr_alpha = float(panel.get("fdr_alpha", 0.05))
    result = _add_fdr_columns(
        result,
        p_column="raw_spearman_p_value",
        q_column="raw_spearman_fdr_q",
        significant_column="raw_spearman_significant_fdr",
        alpha=fdr_alpha,
    )
    result = _add_fdr_columns(
        result,
        p_column="partial_spearman_p_value",
        q_column="partial_spearman_fdr_q",
        significant_column="partial_spearman_significant_fdr",
        alpha=fdr_alpha,
    )
    result["raw_p"] = pd.to_numeric(
        result.get("partial_spearman_p_value", pd.Series(np.nan, index=result.index)), errors="coerce"
    )
    result["within_scan_p"] = result["raw_p"]
    result["family_adjusted_p"] = pd.to_numeric(
        result.get("partial_spearman_fdr_q", pd.Series(np.nan, index=result.index)), errors="coerce"
    )
    result["family_id"] = "fixed_panel_partial_spearman"
    result["inference_role"] = "exploratory"
    return result


def run_variance_homogeneity(
    frame: pd.DataFrame,
    *,
    cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    cfg = _cfg(cfg)
    panel = panel_cfg(cfg)
    features = _panel_features(frame, panel)
    min_n = int(panel.get("min_subjects_per_group", 5))

    rows: list[dict[str, object]] = []
    for contrast_name, group_column, left_label, right_label in _contrast_specs(panel):
        if group_column not in frame.columns:
            continue
        for feature in features:
            left, right = _feature_group_values(
                frame,
                group_column=group_column,
                left_label=left_label,
                right_label=right_label,
                feature=feature,
            )
            row = _base_contrast_row(
                contrast_name=contrast_name,
                group_column=group_column,
                left_label=left_label,
                right_label=right_label,
                feature=feature,
                feature_family=_feature_family(feature, panel),
                left=left,
                right=right,
                min_n=min_n,
            )
            row["variance_group_a"] = float(np.var(left, ddof=1)) if left.size > 1 else np.nan
            row["variance_group_b"] = float(np.var(right, ddof=1)) if right.size > 1 else np.nan
            if row["status"] == "ok":
                levene_stat, levene_p = variance_homogeneity_test(left, right, center="mean")
                bf_stat, bf_p = variance_homogeneity_test(left, right, center="median")
                row.update(
                    {
                        "levene_statistic": levene_stat,
                        "levene_p_value": levene_p,
                        "brown_forsythe_statistic": bf_stat,
                        "brown_forsythe_p_value": bf_p,
                    }
                )
            rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    fdr_alpha = float(panel.get("fdr_alpha", 0.05))
    result = _add_fdr_columns(
        result,
        p_column="levene_p_value",
        q_column="levene_fdr_q",
        significant_column="levene_significant_fdr",
        alpha=fdr_alpha,
    )
    return _add_fdr_columns(
        result,
        p_column="brown_forsythe_p_value",
        q_column="brown_forsythe_fdr_q",
        significant_column="brown_forsythe_significant_fdr",
        alpha=fdr_alpha,
    )


def _plot_effect_sizes(contrasts: pd.DataFrame, output_dir: Path) -> Path | None:
    ok = contrasts.loc[contrasts.get("status", "") == "ok"].copy()
    if ok.empty or "cliffs_delta" not in ok.columns:
        return None
    ok["label"] = ok["contrast_name"].astype(str) + " | " + ok["feature_name"].astype(str)
    ok = ok.sort_values("cliffs_delta")

    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.32 * len(ok))))
    y = np.arange(len(ok))
    xerr = np.vstack(
        [
            ok["cliffs_delta"].to_numpy(dtype=float) - ok["bootstrap_ci_lower"].to_numpy(dtype=float),
            ok["bootstrap_ci_upper"].to_numpy(dtype=float) - ok["cliffs_delta"].to_numpy(dtype=float),
        ]
    )
    ax.errorbar(ok["cliffs_delta"], y, xerr=xerr, fmt="o", color="#0072B2", ecolor="#999999", capsize=3)
    ax.axvline(0.0, color="#333333", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(ok["label"])
    ax.set_xlabel("Cliff's delta (group A - group B)")
    ax.set_title("Fixed post hoc panel effect sizes")
    fig.tight_layout()

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / "panel_effect_sizes.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _plot_raincloud_surrogate(frame: pd.DataFrame, contrasts: pd.DataFrame, output_dir: Path) -> Path | None:
    ok = contrasts.loc[contrasts.get("status", "") == "ok"].copy()
    if ok.empty:
        return None
    contrast = ok.iloc[0]
    feature = str(contrast["feature_name"])
    group_column = str(contrast["group_variable"])
    groups = [str(contrast["group_a"]), str(contrast["group_b"])]
    values = [
        frame.loc[frame[group_column].astype("string") == group, feature].dropna().to_numpy(dtype=float)
        for group in groups
    ]
    if any(value.size == 0 for value in values):
        return None

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.violinplot(values, showmeans=False, showmedians=True)
    for idx, value in enumerate(values, start=1):
        jitter = np.linspace(-0.06, 0.06, value.size) if value.size > 1 else np.array([0.0])
        ax.scatter(np.full(value.size, idx) + jitter, value, alpha=0.75, color="#0072B2")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(groups)
    ax.set_ylabel(feature)
    ax.set_title(f"{feature} by {group_column}")
    fig.tight_layout()

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / "panel_rainclouds.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def run_confirmatory_panel_statistics(
    *,
    feature_matrix_path: str | Path | None = None,
    panel_matrix_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Path]:
    cfg = _cfg(cfg)
    panel = panel_cfg(cfg)
    if not bool(panel.get("enabled", True)):
        return {}

    target_dir = Path(output_dir) if output_dir is not None else get_confirmatory_panel_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    panel_path = Path(panel_matrix_path) if panel_matrix_path is not None else target_dir / "panel_feature_matrix.csv"
    frame = _load_panel_and_clinical(
        feature_matrix_path=feature_matrix_path,
        panel_matrix_path=panel_path,
    )

    outputs: dict[str, Path] = {}
    anchor = run_anchor_test(frame, cfg=cfg)
    anchor_path = target_dir / "primary_anchor_result.csv"
    write_csv_atomic(anchor, anchor_path, index=False)
    outputs["primary_anchor_result"] = anchor_path

    contrasts = run_panel_contrasts(frame, cfg=cfg)
    contrasts_path = target_dir / "panel_contrast_tests.csv"
    write_csv_atomic(contrasts, contrasts_path, index=False)
    outputs["panel_contrast_tests"] = contrasts_path

    continuous = run_continuous_associations(frame, cfg=cfg)
    continuous_path = target_dir / "panel_continuous_associations.csv"
    write_csv_atomic(continuous, continuous_path, index=False)
    outputs["panel_continuous_associations"] = continuous_path

    variance = run_variance_homogeneity(frame, cfg=cfg)
    variance_path = target_dir / "panel_variance_homogeneity.csv"
    write_csv_atomic(variance, variance_path, index=False)
    outputs["panel_variance_homogeneity"] = variance_path

    effect_plot = _plot_effect_sizes(contrasts, target_dir)
    if effect_plot is not None:
        outputs["panel_effect_sizes"] = effect_plot
    raincloud_plot = _plot_raincloud_surrogate(frame, contrasts, target_dir)
    if raincloud_plot is not None:
        outputs["panel_rainclouds"] = raincloud_plot

    logger.info("Exploratory fixed-panel statistics written to %s", target_dir)
    return outputs
