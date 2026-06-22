"""Cross-region disease-only phase orchestration."""

from __future__ import annotations

import logging
from pathlib import Path

from src import config
from src.analysis.biochemical_ratios import run_biochemical_ratios
from src.analysis.clinical_correlation_matrix import run_clinical_correlation_matrix
from src.analysis.covariate_models import run_covariate_region_models
from src.analysis.disease_subgroups import run_disease_subgroups
from src.analysis.disease_stratification import run_disease_stratification
from src.analysis.integrated_candidates import run_integrated_candidates
from src.analysis.longitudinal_moca import run_longitudinal_moca
from src.analysis.minimal_panel import run_minimal_panel
from src.analysis.pls_regression import run_pls_regression
from src.analysis.spectral_regions import run_candidate_region_detector
from src.analysis.splsda import run_splsda_classifier
from src.reporting.candidate_region_report import run_candidate_region_report

logger = logging.getLogger(__name__)


def _phase_enabled(cfg: dict, key: str, default: bool = True) -> bool:
    return bool(cfg.get("analysis", {}).get(key, {}).get("enabled", default))


def run_disease_only_cross_region_pipeline(
    *,
    feature_matrix_path: str | Path | None,
    cfg: dict | None = None,
) -> dict[str, dict[str, Path]]:
    """Run disease-only cross-region phases that depend on Biological PCA exports."""
    cfg = config if cfg is None else cfg
    outputs: dict[str, dict[str, Path]] = {}

    clinical_outputs: dict[str, Path] = {}
    if _phase_enabled(cfg, "clinical_correlation_matrix"):
        logger.info("Starting disease-only clinical correlation matrix")
        clinical_outputs = run_clinical_correlation_matrix(
            feature_matrix_path=feature_matrix_path,
            cfg=cfg,
        )
        outputs["clinical_correlation_matrix"] = clinical_outputs

    longitudinal_outputs: dict[str, Path] = {}
    if _phase_enabled(cfg, "longitudinal_moca"):
        logger.info("Starting longitudinal Delta MoCA analysis")
        longitudinal_outputs = run_longitudinal_moca(
            feature_matrix_path=feature_matrix_path,
            cfg=cfg,
        )
        outputs["longitudinal_moca"] = longitudinal_outputs

    stratification_outputs: dict[str, Path] = {}
    if _phase_enabled(cfg, "disease_stratification"):
        logger.info("Starting disease stratification analysis")
        stratification_outputs = run_disease_stratification(
            feature_matrix_path=feature_matrix_path,
            cfg=cfg,
        )
        outputs["disease_stratification"] = stratification_outputs

    candidate_outputs: dict[str, Path] = {}
    if _phase_enabled(cfg, "candidate_regions"):
        logger.info("Starting unified candidate-region detector")
        candidate_outputs = run_candidate_region_detector(
            clinical_outputs=clinical_outputs,
            longitudinal_outputs=longitudinal_outputs,
            stratification_outputs=stratification_outputs,
            cfg=cfg,
        )
        outputs["candidate_regions"] = candidate_outputs

    if _phase_enabled(cfg, "covariate_models"):
        logger.info("Starting covariate-adjusted region models")
        outputs["covariate_models"] = run_covariate_region_models(
            feature_matrix_path=feature_matrix_path,
            candidate_regions_path=candidate_outputs.get("candidate_regions_cluster_corrected"),
            cfg=cfg,
        )

    pls_outputs: dict[str, Path] = {}
    if _phase_enabled(cfg, "pls_regression"):
        logger.info("Starting PLS1 regression (Phase 8)")
        pls_outputs = run_pls_regression(
            feature_matrix_path=feature_matrix_path,
            cfg=cfg,
        )
        outputs["pls_regression"] = pls_outputs

    splsda_outputs: dict[str, Path] = {}
    if _phase_enabled(cfg, "splsda"):
        logger.info("Starting PLS-L1DA exploratory classifier (Phase 9)")
        splsda_outputs = run_splsda_classifier(
            feature_matrix_path=feature_matrix_path,
            pls_outputs=pls_outputs,
            cfg=cfg,
        )
        outputs["splsda"] = splsda_outputs

    if _phase_enabled(cfg, "biochemical_ratios"):
        logger.info("Starting biochemical ratio analysis (Phase 10)")
        outputs["biochemical_ratios"] = run_biochemical_ratios(
            feature_matrix_path=feature_matrix_path,
            cfg=cfg,
        )

    integrated_outputs: dict[str, Path] = {}
    if _phase_enabled(cfg, "integrated_candidates"):
        logger.info("Starting integrated candidate evidence table (Phase 11)")
        integrated_outputs = run_integrated_candidates(
            feature_matrix_path=feature_matrix_path,
            pls_vip_path=pls_outputs.get("pls_vip_bands"),
            splsda_bands_path=splsda_outputs.get("pls_l1da_selected_bands"),
            cfg=cfg,
        )
        outputs["integrated_candidates"] = integrated_outputs

    if _phase_enabled(cfg, "disease_subgroups"):
        logger.info("Starting disease subgroup analysis (Phase 12)")
        outputs["disease_subgroups"] = run_disease_subgroups(
            feature_matrix_path=feature_matrix_path,
            integrated_candidates_path=integrated_outputs.get("integrated_candidate_regions"),
            cfg=cfg,
        )

    panel_outputs: dict[str, Path] = {}
    if _phase_enabled(cfg, "minimal_panel"):
        logger.info("Starting minimal candidate Raman panel (Phase 13)")
        panel_outputs = run_minimal_panel(
            feature_matrix_path=feature_matrix_path,
            integrated_candidates_path=integrated_outputs.get("integrated_candidate_regions"),
            cfg=cfg,
        )
        outputs["minimal_panel"] = panel_outputs

    if _phase_enabled(cfg, "publication_report"):
        logger.info("Starting publication report generator (Phase 14)")
        outputs["publication_report"] = run_candidate_region_report(
            integrated_candidates_path=integrated_outputs.get("integrated_candidate_regions"),
            minimal_panel_path=panel_outputs.get("minimal_candidate_panel"),
            cfg=cfg,
        )

    return outputs
