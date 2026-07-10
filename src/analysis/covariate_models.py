"""Covariate-adjusted models for candidate regions and predefined ratios."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src import config, get_cross_region_phase_dir
from src.analysis.band_features import band_mean, infer_block_from_upper, ratio_from_integrals
from src.analysis.clinical_feature_associations import (
    feature_columns,
    load_disease_feature_frame,
)
from src.analysis.sklearn_utils import covariate_logistic_model, fit_with_convergence_capture
from src.analysis.spectral_regions import candidate_region_feature_names

logger = logging.getLogger(__name__)

DEFAULT_BINARY_OUTCOMES = {
    "moca_severity_group": ("low_moca", "high_moca"),
    "delta_moca_group": ("decline", "stable_or_improved"),
}


def _cfg(cfg: dict | None = None) -> dict:
    return config if cfg is None else cfg


def _model_cfg(cfg: dict) -> dict:
    return cfg.get("analysis", {}).get("covariate_models", {})


def _empty_model_table() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "source_analysis",
            "analysis_name",
            "target",
            "contrast_name",
            "group_variable",
            "outcome_type",
            "predictor_type",
            "predictor_name",
            "spectral_block",
            "candidate_region_id",
            "start_cm-1",
            "end_cm-1",
            "formula",
            "n_subjects",
            "n_events",
            "coefficient",
            "coef_p_value",
            "coef_fdr_q",
            "coef_significant_fdr",
            "intercept",
            "r_squared",
            "accuracy",
            "available_covariates",
            "missing_covariates",
            "selection_scope",
            "inference_role",
            "status",
        ]
    )


def _empty_availability_table() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "source_analysis",
            "analysis_name",
            "target",
            "contrast_name",
            "group_variable",
            "outcome_type",
            "requested_covariates",
            "available_covariates",
            "missing_covariates",
            "n_candidate_regions",
            "n_ratio_models",
            "status",
        ]
    )


def _candidate_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    return get_cross_region_phase_dir("candidate_regions") / "candidate_regions_cluster_corrected.csv"


def _encode_covariate(frame: pd.DataFrame, column: str) -> tuple[pd.Series | None, str | None]:
    if column not in frame.columns:
        return None, "missing column"

    series = frame[column]
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == len(frame):
        return numeric.astype(float), None

    normalized = series.astype("string").str.strip().str.lower()
    if column in {"sex", "gender"}:
        mapped = normalized.map({"male": 0.0, "female": 1.0, "m": 0.0, "f": 1.0})
        if mapped.notna().sum() >= 2:
            return mapped.astype(float), None

    return None, "unsupported encoding"


def _collect_covariates(
    frame: pd.DataFrame,
    requested_covariates: list[str],
    *,
    outcome_name: str | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    encoded: dict[str, pd.Series] = {}
    missing: list[str] = []
    for covariate in requested_covariates:
        if outcome_name is not None and covariate == outcome_name:
            continue
        values, error = _encode_covariate(frame, covariate)
        if values is None:
            missing.append(covariate if error == "missing column" else f"{covariate} ({error})")
            continue
        encoded[covariate] = values
    return pd.DataFrame(encoded, index=frame.index), list(encoded.keys()), missing


def _fit_ols(
    outcome: pd.Series,
    predictor: pd.Series,
    covariates: pd.DataFrame,
) -> tuple[float, float, float, float]:
    design = pd.concat([predictor.rename("predictor"), covariates], axis=1)
    design = design.dropna()
    y = outcome.loc[design.index]
    design_matrix = np.column_stack([np.ones(len(design)), design.to_numpy(dtype=float)])
    if np.linalg.matrix_rank(design_matrix) < design_matrix.shape[1]:
        raise ValueError("rank_deficient")
    beta, *_ = np.linalg.lstsq(design_matrix, y.to_numpy(dtype=float), rcond=None)
    fitted = design_matrix @ beta
    residual = y.to_numpy(dtype=float) - fitted
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y.to_numpy(dtype=float) - float(y.mean())) ** 2))
    r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    rank = int(np.linalg.matrix_rank(design_matrix))
    dof = len(design_matrix) - rank
    p_value = np.nan
    if dof > 0:
        sigma2 = ss_res / dof
        cov_beta = sigma2 * np.linalg.pinv(design_matrix.T @ design_matrix)
        se = float(np.sqrt(max(cov_beta[1, 1], 0.0)))
        if se > 0.0 and np.isfinite(se):
            t_stat = float(beta[1] / se)
            p_value = float(2.0 * stats.t.sf(abs(t_stat), dof))
    return float(beta[1]), float(beta[0]), r_squared, p_value


def _fit_logistic_wald(
    outcome: pd.Series,
    predictor: pd.Series,
    covariates: pd.DataFrame,
    *,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> tuple[float, float, float]:
    """Fit unpenalized logistic regression and return predictor Wald p-value."""
    design = pd.concat([predictor.rename("predictor"), covariates], axis=1)
    design = design.dropna()
    y = outcome.loc[design.index].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(design)), design.to_numpy(dtype=float)])
    if np.linalg.matrix_rank(x) < x.shape[1]:
        raise ValueError("rank_deficient")
    beta = np.zeros(x.shape[1], dtype=float)
    eps = 1e-9

    for _ in range(max_iter):
        linear = np.clip(x @ beta, -35.0, 35.0)
        prob = 1.0 / (1.0 + np.exp(-linear))
        weight = np.clip(prob * (1.0 - prob), eps, None)
        hessian = x.T @ (x * weight[:, None])
        gradient = x.T @ (y - prob)
        step = np.linalg.pinv(hessian) @ gradient
        beta_next = beta + step
        if np.max(np.abs(step)) < tol:
            beta = beta_next
            break
        beta = beta_next

    linear = np.clip(x @ beta, -35.0, 35.0)
    prob = 1.0 / (1.0 + np.exp(-linear))
    weight = np.clip(prob * (1.0 - prob), eps, None)
    hessian = x.T @ (x * weight[:, None])
    cov_beta = np.linalg.pinv(hessian)
    se = float(np.sqrt(max(cov_beta[1, 1], 0.0)))
    p_value = np.nan
    if se > 0.0 and np.isfinite(se):
        z_stat = float(beta[1] / se)
        p_value = float(2.0 * stats.norm.sf(abs(z_stat)))
    return float(beta[1]), float(beta[0]), p_value


def _fit_logistic(
    outcome: pd.Series,
    predictor: pd.Series,
    covariates: pd.DataFrame,
    *,
    random_state: int,
) -> tuple[float, float, float, float]:
    design = pd.concat([predictor.rename("predictor"), covariates], axis=1)
    design = design.dropna()
    y = outcome.loc[design.index].to_numpy(dtype=int)
    model = covariate_logistic_model(random_state=random_state)
    fit_with_convergence_capture(
        model,
        design.to_numpy(dtype=float),
        y,
        context="covariate logistic",
    )
    accuracy = float(np.mean(model.predict(design.to_numpy(dtype=float)) == y))
    coefficient, intercept, p_value = _fit_logistic_wald(outcome, predictor, covariates)
    return coefficient, intercept, accuracy, p_value


def _fdr_bh(values: pd.Series) -> pd.Series:
    p_values = pd.to_numeric(values, errors="coerce")
    out = pd.Series(np.nan, index=values.index, dtype=float)
    valid = p_values.dropna()
    if valid.empty:
        return out
    ordered = valid.sort_values()
    n = len(ordered)
    adjusted = ordered.to_numpy(dtype=float) * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out.loc[ordered.index] = np.clip(adjusted, 0.0, 1.0)
    return out


def _region_predictor(
    frame: pd.DataFrame,
    all_feature_columns: list[str],
    region: pd.Series,
) -> pd.Series | None:
    columns = candidate_region_feature_names(all_feature_columns, region)
    if not columns:
        return None
    return frame[columns].mean(axis=1)


def _band_feature(frame: pd.DataFrame, feature_names: list[str], block: str, lower: float, upper: float) -> pd.Series:
    return band_mean(frame, feature_names, block=block, lower=lower, upper=upper)


def _ratio_predictors(frame: pd.DataFrame, feature_names: list[str], cfg: dict) -> pd.DataFrame:
    bands_cfg = cfg.get("whole_spectra", {}).get("bands", {})
    ratio_pairs = cfg.get("whole_spectra", {}).get("ratio_pairs", [])
    band_series: dict[str, pd.Series] = {}
    for band_name, limits in bands_cfg.items():
        lower, upper = float(limits[0]), float(limits[1])
        block = infer_block_from_upper(upper)
        band_series[str(band_name)] = _band_feature(frame, feature_names, block, lower, upper)

    ratios: dict[str, pd.Series] = {}
    for pair in ratio_pairs:
        left, right = str(pair[0]), str(pair[1])
        if left not in band_series or right not in band_series:
            continue
        ratios[f"{left}__over__{right}"] = ratio_from_integrals(band_series[left], band_series[right])
    return pd.DataFrame(ratios, index=frame.index)


def _binary_outcome(frame: pd.DataFrame, group_variable: str, positive: str, negative: str) -> pd.Series:
    values = frame[group_variable].astype("string")
    valid = values.isin([positive, negative])
    encoded = pd.Series(np.nan, index=frame.index, dtype=float)
    encoded.loc[valid] = np.where(values.loc[valid] == positive, 1.0, 0.0)
    return encoded


def _iter_outcomes(candidates: pd.DataFrame) -> list[dict[str, object]]:
    outcomes: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    for _, row in candidates.iterrows():
        if pd.notna(row.get("target")):
            key = ("continuous", row["source_analysis"], row["analysis_name"], row["target"])
            if key in seen:
                continue
            seen.add(key)
            outcomes.append(
                {
                    "outcome_type": "continuous",
                    "source_analysis": row["source_analysis"],
                    "analysis_name": row["analysis_name"],
                    "target": row["target"],
                    "contrast_name": row.get("contrast_name"),
                    "group_variable": row.get("group_variable"),
                }
            )
            continue

        group_variable = row.get("group_variable")
        if pd.isna(group_variable) or str(group_variable) not in DEFAULT_BINARY_OUTCOMES:
            continue
        key = ("binary", row["source_analysis"], row["analysis_name"], str(group_variable))
        if key in seen:
            continue
        seen.add(key)
        outcomes.append(
            {
                "outcome_type": "binary",
                "source_analysis": row["source_analysis"],
                "analysis_name": row["analysis_name"],
                "target": pd.NA,
                "contrast_name": row.get("contrast_name"),
                "group_variable": str(group_variable),
            }
        )
    return outcomes


def run_covariate_region_models(
    *,
    feature_matrix_path: str | Path | None = None,
    candidate_regions_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    cfg: dict | None = None,
) -> dict[str, Path]:
    cfg = _cfg(cfg)
    model_cfg = _model_cfg(cfg)
    if not bool(model_cfg.get("enabled", True)):
        return {}

    disease_only, _feature_meta = load_disease_feature_frame(feature_matrix_path=feature_matrix_path)
    all_feature_columns = feature_columns(disease_only)
    candidate_path = _candidate_path(candidate_regions_path)
    candidates = pd.read_csv(candidate_path) if candidate_path.exists() else pd.DataFrame()

    target_dir = (
        Path(output_dir)
        if output_dir is not None
        else get_cross_region_phase_dir("covariate_models")
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    if candidates.empty:
        models = _empty_model_table()
        availability = _empty_availability_table()
        models_path = target_dir / "covariate_adjusted_region_models.csv"
        availability_path = target_dir / "covariate_availability_report.csv"
        models.to_csv(models_path, index=False)
        availability.to_csv(availability_path, index=False)
        return {
            "covariate_adjusted_region_models": models_path,
            "covariate_availability_report": availability_path,
        }

    requested_covariates = [str(item) for item in model_cfg.get("covariates", ["age", "sex"])]
    min_subjects = int(model_cfg.get("min_subjects", 8))
    random_state = int(model_cfg.get("random_state", 42))
    ratio_frame = _ratio_predictors(disease_only, all_feature_columns, cfg)

    model_rows: list[dict[str, object]] = []
    availability_rows: list[dict[str, object]] = []

    for outcome_spec in _iter_outcomes(candidates):
        source_analysis = str(outcome_spec["source_analysis"])
        analysis_name = str(outcome_spec["analysis_name"])
        outcome_type = str(outcome_spec["outcome_type"])
        if outcome_type == "continuous":
            target_name = str(outcome_spec["target"])
            outcome = pd.to_numeric(disease_only[target_name], errors="coerce")
            candidate_subset = candidates.loc[
                candidates["source_analysis"].eq(source_analysis)
                & candidates["analysis_name"].eq(analysis_name)
                & candidates["target"].eq(target_name)
            ].copy()
            outcome_label = target_name
        else:
            group_variable = str(outcome_spec["group_variable"])
            positive, negative = DEFAULT_BINARY_OUTCOMES[group_variable]
            outcome = _binary_outcome(disease_only, group_variable, positive=positive, negative=negative)
            candidate_subset = candidates.loc[
                candidates["source_analysis"].eq(source_analysis)
                & candidates["analysis_name"].eq(analysis_name)
                & candidates["group_variable"].astype("string").eq(group_variable)
            ].copy()
            outcome_label = group_variable

        covariate_frame, available_covariates, missing_covariates = _collect_covariates(
            disease_only,
            requested_covariates,
            outcome_name=outcome_label,
        )
        availability_rows.append(
            {
                "source_analysis": source_analysis,
                "analysis_name": analysis_name,
                "target": outcome_spec.get("target", pd.NA),
                "contrast_name": outcome_spec.get("contrast_name", pd.NA),
                "group_variable": outcome_spec.get("group_variable", pd.NA),
                "outcome_type": outcome_type,
                "requested_covariates": ", ".join(requested_covariates),
                "available_covariates": ", ".join(available_covariates),
                "missing_covariates": ", ".join(missing_covariates),
                "n_candidate_regions": int(len(candidate_subset)),
                "n_ratio_models": int(ratio_frame.shape[1]),
                "status": "ok" if candidate_subset.shape[0] > 0 else "no candidate regions",
            }
        )

        predictor_specs: list[tuple[str, str, pd.Series, pd.Series | None]] = []
        for _, region in candidate_subset.iterrows():
            predictor = _region_predictor(disease_only, all_feature_columns, region)
            predictor_specs.append(("region", str(region["candidate_region_id"]), predictor, region))
        for ratio_name in ratio_frame.columns:
            predictor_specs.append(("ratio", ratio_name, ratio_frame[ratio_name], None))

        for predictor_type, predictor_name, predictor, region in predictor_specs:
            if predictor is None:
                continue
            design = pd.concat([outcome.rename("outcome"), predictor.rename("predictor"), covariate_frame], axis=1)
            design = design.dropna()
            n_parameters = 1 + len(available_covariates)
            ols_min_subjects = max(30, 10 * n_parameters)
            n_events_observed = int(design["outcome"].sum()) if outcome_type == "binary" else 0
            n_nonevents_observed = int(len(design) - n_events_observed) if outcome_type == "binary" else 0
            insufficient = len(design) < max(min_subjects, ols_min_subjects if outcome_type == "continuous" else 0)
            if outcome_type == "binary":
                insufficient = insufficient or min(n_events_observed, n_nonevents_observed) < 10 * n_parameters
            if insufficient:
                status = "insufficient_n"
                n_subjects = int(len(design))
                n_events = int(design["outcome"].sum()) if outcome_type == "binary" else np.nan
                coefficient = np.nan
                coef_p_value = np.nan
                intercept = np.nan
                r_squared = np.nan
                accuracy = np.nan
            else:
                try:
                    if outcome_type == "continuous":
                        coefficient, intercept, r_squared, coef_p_value = _fit_ols(
                            design["outcome"],
                            design["predictor"],
                            design.drop(columns=["outcome", "predictor"]),
                        )
                        accuracy = np.nan
                        n_events = np.nan
                    else:
                        if design["outcome"].nunique() < 2:
                            raise ValueError("insufficient_n")
                        coefficient, intercept, accuracy, coef_p_value = _fit_logistic(
                            design["outcome"],
                            design["predictor"],
                            design.drop(columns=["outcome", "predictor"]),
                            random_state=random_state,
                        )
                        r_squared = np.nan
                        n_events = int(design["outcome"].sum())
                    status = "ok"
                except ValueError as exc:
                    status = str(exc) if str(exc) in {"rank_deficient", "separation", "insufficient_n"} else "missing_required_data"
                    coefficient = np.nan
                    coef_p_value = np.nan
                    intercept = np.nan
                    r_squared = np.nan
                    accuracy = np.nan
                    n_events = int(design["outcome"].sum()) if outcome_type == "binary" else np.nan
                except Exception:
                    status = "missing_required_data"
                    coefficient = np.nan
                    coef_p_value = np.nan
                    intercept = np.nan
                    r_squared = np.nan
                    accuracy = np.nan
                    n_events = int(design["outcome"].sum()) if outcome_type == "binary" else np.nan
                n_subjects = int(len(design))

            formula_terms = [predictor_name, *available_covariates]
            if outcome_type == "continuous":
                formula = f"{outcome_label} ~ {' + '.join(formula_terms)}"
            else:
                positive, _negative = DEFAULT_BINARY_OUTCOMES[str(outcome_spec['group_variable'])]
                formula = f"{outcome_label}[{positive}] ~ {' + '.join(formula_terms)}"

            model_rows.append(
                {
                    "source_analysis": source_analysis,
                    "analysis_name": analysis_name,
                    "target": outcome_spec.get("target", pd.NA),
                    "contrast_name": outcome_spec.get("contrast_name", pd.NA),
                    "group_variable": outcome_spec.get("group_variable", pd.NA),
                    "outcome_type": outcome_type,
                    "predictor_type": predictor_type,
                    "predictor_name": predictor_name,
                    "spectral_block": region["spectral_block"] if region is not None else pd.NA,
                    "candidate_region_id": region["candidate_region_id"] if region is not None else pd.NA,
                    "start_cm-1": region["start_cm-1"] if region is not None else np.nan,
                    "end_cm-1": region["end_cm-1"] if region is not None else np.nan,
                    "formula": formula,
                    "n_subjects": n_subjects,
                    "n_events": n_events,
                    "coefficient": coefficient,
                    "coef_p_value": coef_p_value,
                    "coef_fdr_q": np.nan,
                    "coef_significant_fdr": False,
                    "intercept": intercept,
                    "r_squared": r_squared,
                    "accuracy": accuracy,
                    "available_covariates": ", ".join(available_covariates),
                    "missing_covariates": ", ".join(missing_covariates),
                    "selection_scope": "same_cohort_selected",
                    "inference_role": "post_selection_descriptive",
                    "status": status,
                }
            )

    models = pd.DataFrame(model_rows) if model_rows else _empty_model_table()
    if not models.empty and "coef_p_value" in models.columns:
        group_cols = ["source_analysis", "analysis_name", "outcome_type", "target", "group_variable"]
        models["coef_fdr_q"] = (
            models.groupby(group_cols, dropna=False)["coef_p_value"]
            .transform(_fdr_bh)
            .astype(float)
        )
        models["coef_significant_fdr"] = models["coef_fdr_q"].lt(0.05).fillna(False)
    availability = pd.DataFrame(availability_rows) if availability_rows else _empty_availability_table()
    models_path = target_dir / "covariate_adjusted_region_models.csv"
    availability_path = target_dir / "covariate_availability_report.csv"
    models.to_csv(models_path, index=False)
    availability.to_csv(availability_path, index=False)
    logger.info("Covariate-adjusted region models written to %s", target_dir)
    return {
        "covariate_adjusted_region_models": models_path,
        "covariate_availability_report": availability_path,
    }
