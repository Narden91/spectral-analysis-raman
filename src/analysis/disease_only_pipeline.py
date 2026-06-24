"""Cross-region disease-only phase orchestration."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import pandas as pd

from src import config, get_confirmatory_panel_dir, get_run_metadata_dir
from src.analysis.biochemical_ratios import run_biochemical_ratios
from src.analysis.clinical_correlation_matrix import run_clinical_correlation_matrix
from src.analysis.covariate_models import run_covariate_region_models
from src.analysis.disease_subgroups import run_disease_subgroups
from src.analysis.disease_stratification import run_disease_stratification
from src.analysis.cluster_panel_summary import run_panel_cluster_summary
from src.analysis.endpoint_consistency import run_endpoint_consistency
from src.analysis.integrated_candidates import run_integrated_candidates
from src.analysis.io_utils import write_csv_atomic
from src.analysis.longitudinal_moca import run_longitudinal_moca
from src.analysis.minimal_panel import run_minimal_panel
from src.analysis.pls_regression import run_pls_regression
from src.analysis.preregistered_panel import run_preregistered_panel
from src.analysis.robust_contrasts import run_confirmatory_panel_statistics
from src.analysis.robustness_ranking import run_robustness_ranking
from src.analysis.spectral_regions import run_candidate_region_detector
from src.analysis.splsda import run_splsda_classifier
from src.reporting.candidate_region_report import run_candidate_region_report, run_panel_report_section

logger = logging.getLogger(__name__)


def _phase_enabled(cfg: dict, key: str, default: bool = True) -> bool:
    return bool(cfg.get("analysis", {}).get(key, {}).get("enabled", default))


@contextmanager
def _phase_timer(timings: list[dict[str, object]], phase_key: str, phase_label: str):
    start = time.perf_counter()
    status = "ok"
    try:
        yield
    except Exception:
        status = "failed"
        raise
    finally:
        elapsed = time.perf_counter() - start
        timings.append(
            {
                "phase_key": phase_key,
                "phase_label": phase_label,
                "status": status,
                "elapsed_seconds": round(elapsed, 6),
            }
        )
        logger.info("%s finished with status=%s in %.2fs", phase_label, status, elapsed)


def _write_phase_timings(timings: list[dict[str, object]]) -> Path | None:
    if not timings:
        return None
    path = get_run_metadata_dir() / "disease_only_cross_region_phase_timing.csv"
    write_csv_atomic(pd.DataFrame(timings), path, index=False)
    return path


def _cfg_for_preregistered_minimal_panel(cfg: dict) -> dict:
    panel_cfg = deepcopy(cfg)
    analysis = panel_cfg.setdefault("analysis", {})
    minimal = dict(analysis.get("minimal_panel", {}))
    minimal.update(analysis.get("panel_minimal_model", {}))
    minimal["enabled"] = True
    minimal["feature_source"] = "preregistered_panel"
    analysis["minimal_panel"] = minimal
    return panel_cfg


def run_disease_only_cross_region_pipeline(
    *,
    feature_matrix_path: str | Path | None,
    cfg: dict | None = None,
) -> dict[str, dict[str, Path]]:
    """Run disease-only cross-region phases that depend on Biological PCA exports."""
    cfg = config if cfg is None else cfg
    outputs: dict[str, dict[str, Path]] = {}
    phase_timings: list[dict[str, object]] = []

    clinical_outputs: dict[str, Path] = {}
    if _phase_enabled(cfg, "clinical_correlation_matrix"):
        logger.info("Starting disease-only clinical correlation matrix")
        with _phase_timer(phase_timings, "clinical_correlation_matrix", "Clinical correlation matrix"):
            clinical_outputs = run_clinical_correlation_matrix(
                feature_matrix_path=feature_matrix_path,
                cfg=cfg,
            )
            outputs["clinical_correlation_matrix"] = clinical_outputs

    longitudinal_outputs: dict[str, Path] = {}
    if _phase_enabled(cfg, "longitudinal_moca"):
        logger.info("Starting longitudinal Delta MoCA analysis")
        with _phase_timer(phase_timings, "longitudinal_moca", "Longitudinal Delta MoCA"):
            longitudinal_outputs = run_longitudinal_moca(
                feature_matrix_path=feature_matrix_path,
                cfg=cfg,
            )
            outputs["longitudinal_moca"] = longitudinal_outputs

    stratification_outputs: dict[str, Path] = {}
    if _phase_enabled(cfg, "disease_stratification"):
        logger.info("Starting disease stratification analysis")
        with _phase_timer(phase_timings, "disease_stratification", "Disease stratification"):
            stratification_outputs = run_disease_stratification(
                feature_matrix_path=feature_matrix_path,
                cfg=cfg,
            )
            outputs["disease_stratification"] = stratification_outputs

    candidate_outputs: dict[str, Path] = {}
    if _phase_enabled(cfg, "candidate_regions"):
        logger.info("Starting unified candidate-region detector")
        with _phase_timer(phase_timings, "candidate_regions", "Candidate-region detector"):
            candidate_outputs = run_candidate_region_detector(
                clinical_outputs=clinical_outputs,
                longitudinal_outputs=longitudinal_outputs,
                stratification_outputs=stratification_outputs,
                cfg=cfg,
            )
            outputs["candidate_regions"] = candidate_outputs

    if _phase_enabled(cfg, "covariate_models"):
        logger.info("Starting covariate-adjusted region models")
        with _phase_timer(phase_timings, "covariate_models", "Covariate-adjusted region models"):
            outputs["covariate_models"] = run_covariate_region_models(
                feature_matrix_path=feature_matrix_path,
                candidate_regions_path=candidate_outputs.get("candidate_regions_cluster_corrected"),
                cfg=cfg,
            )

    pls_outputs: dict[str, Path] = {}
    if _phase_enabled(cfg, "pls_regression"):
        logger.info("Starting PLS1 regression (Phase 8)")
        with _phase_timer(phase_timings, "pls_regression", "PLS1 regression"):
            pls_outputs = run_pls_regression(
                feature_matrix_path=feature_matrix_path,
                cfg=cfg,
            )
            outputs["pls_regression"] = pls_outputs

    splsda_outputs: dict[str, Path] = {}
    if _phase_enabled(cfg, "splsda"):
        logger.info("Starting PLS-L1DA exploratory classifier (Phase 9)")
        with _phase_timer(phase_timings, "splsda", "PLS-L1DA exploratory classifier"):
            splsda_outputs = run_splsda_classifier(
                feature_matrix_path=feature_matrix_path,
                pls_outputs=pls_outputs,
                cfg=cfg,
            )
            outputs["splsda"] = splsda_outputs

    if _phase_enabled(cfg, "biochemical_ratios"):
        logger.info("Starting biochemical ratio analysis (Phase 10)")
        with _phase_timer(phase_timings, "biochemical_ratios", "Biochemical ratio analysis"):
            outputs["biochemical_ratios"] = run_biochemical_ratios(
                feature_matrix_path=feature_matrix_path,
                cfg=cfg,
            )

    preregistered_outputs: dict[str, Path] = {}
    if _phase_enabled(cfg, "preregistered_panel"):
        logger.info("Starting pre-registered confirmatory panel")
        with _phase_timer(phase_timings, "preregistered_panel", "Pre-registered confirmatory panel"):
            preregistered_outputs = run_preregistered_panel(
                feature_matrix_path=feature_matrix_path,
                cfg=cfg,
            )
            outputs["preregistered_panel"] = preregistered_outputs

        logger.info("Starting robust confirmatory panel statistics (Phases E-H)")
        with _phase_timer(phase_timings, "confirmatory_panel", "Robust confirmatory panel statistics"):
            outputs["confirmatory_panel"] = run_confirmatory_panel_statistics(
                feature_matrix_path=feature_matrix_path,
                panel_matrix_path=preregistered_outputs.get("panel_feature_matrix"),
                cfg=cfg,
            )

        if _phase_enabled(cfg, "endpoint_consistency"):
            logger.info("Starting endpoint consistency analysis (Phase S)")
            with _phase_timer(phase_timings, "endpoint_consistency", "Endpoint consistency analysis"):
                outputs["endpoint_consistency"] = run_endpoint_consistency(
                    associations_path=outputs["confirmatory_panel"].get("panel_continuous_associations"),
                    cfg=cfg,
                )

        if _phase_enabled(cfg, "robustness_ranking"):
            logger.info("Starting graded robustness ranking (Phase L)")
            with _phase_timer(phase_timings, "robustness_ranking", "Graded robustness ranking"):
                outputs["robustness_ranking"] = run_robustness_ranking(
                    contrasts_path=outputs["confirmatory_panel"].get("panel_contrast_tests"),
                    consistency_path=outputs.get("endpoint_consistency", {}).get("endpoint_consistency"),
                    cfg=cfg,
                )

    integrated_outputs: dict[str, Path] = {}
    if _phase_enabled(cfg, "integrated_candidates"):
        logger.info("Starting integrated candidate evidence table (Phase 11)")
        with _phase_timer(phase_timings, "integrated_candidates", "Integrated candidate evidence table"):
            integrated_outputs = run_integrated_candidates(
                feature_matrix_path=feature_matrix_path,
                pls_vip_path=pls_outputs.get("pls_vip_bands"),
                pls_performance_path=pls_outputs.get("pls_regression_performance"),
                splsda_bands_path=splsda_outputs.get("pls_l1da_selected_bands", ""),
                cfg=cfg,
            )
            outputs["integrated_candidates"] = integrated_outputs

    if _phase_enabled(cfg, "disease_subgroups"):
        logger.info("Starting disease subgroup analysis (Phase 12)")
        with _phase_timer(phase_timings, "disease_subgroups", "Disease subgroup analysis"):
            outputs["disease_subgroups"] = run_disease_subgroups(
                feature_matrix_path=feature_matrix_path,
                integrated_candidates_path=integrated_outputs.get("integrated_candidate_regions"),
                cfg=cfg,
            )

    if preregistered_outputs and _phase_enabled(cfg, "panel_cluster_summary"):
        logger.info("Starting panel cluster summary (Phase I)")
        with _phase_timer(phase_timings, "panel_cluster_summary", "Panel cluster summary"):
            outputs["panel_cluster_summary"] = run_panel_cluster_summary(
                feature_matrix_path=feature_matrix_path,
                panel_matrix_path=preregistered_outputs.get("panel_feature_matrix"),
                subgroup_assignments_path=outputs.get("disease_subgroups", {}).get("disease_subgroup_assignments"),
                cfg=cfg,
            )

    panel_outputs: dict[str, Path] = {}
    if _phase_enabled(cfg, "minimal_panel"):
        logger.info("Starting minimal candidate Raman panel (Phase 13)")
        with _phase_timer(phase_timings, "minimal_panel", "Minimal candidate Raman panel"):
            panel_outputs = run_minimal_panel(
                feature_matrix_path=feature_matrix_path,
                integrated_candidates_path=integrated_outputs.get("integrated_candidate_regions"),
                cfg=cfg,
            )
            outputs["minimal_panel"] = panel_outputs

    preregistered_minimal_outputs: dict[str, Path] = {}
    if preregistered_outputs and _phase_enabled(cfg, "panel_minimal_model"):
        logger.info("Starting panel-restricted minimal model (Phase J)")
        with _phase_timer(phase_timings, "panel_minimal_model", "Panel-restricted minimal model"):
            preregistered_minimal_outputs = run_minimal_panel(
                feature_matrix_path=feature_matrix_path,
                panel_matrix_path=preregistered_outputs.get("panel_feature_matrix"),
                output_dir=get_confirmatory_panel_dir(),
                cfg=_cfg_for_preregistered_minimal_panel(cfg),
            )
            outputs["panel_minimal_model"] = preregistered_minimal_outputs

    panel_report_outputs: dict[str, Path] = {}
    if preregistered_outputs and _phase_enabled(cfg, "panel_report"):
        logger.info("Starting four-domain panel report (Phase K)")
        with _phase_timer(phase_timings, "panel_report", "Four-domain panel report"):
            panel_report_outputs = run_panel_report_section(
                robustness_ranking_path=outputs.get("robustness_ranking", {}).get("feature_robustness_ranking"),
                anchor_result_path=outputs.get("confirmatory_panel", {}).get("primary_anchor_result"),
                cfg=cfg,
            )
            outputs["panel_report"] = panel_report_outputs

    if _phase_enabled(cfg, "publication_report"):
        logger.info("Starting publication report generator (Phase 14)")
        with _phase_timer(phase_timings, "publication_report", "Publication report generator"):
            outputs["publication_report"] = run_candidate_region_report(
                integrated_candidates_path=integrated_outputs.get("integrated_candidate_regions"),
                minimal_panel_path=panel_outputs.get("minimal_candidate_panel"),
                panel_report_section_path=panel_report_outputs.get("panel_report_section"),
                cfg=cfg,
            )

    _write_phase_timings(phase_timings)
    return outputs
