"""Cross-region disease-only phase orchestration."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from contextlib import AbstractContextManager, contextmanager
from copy import deepcopy
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from src import config, get_confirmatory_panel_dir, get_run_metadata_dir
from src.analysis.biochemical_ratios import run_biochemical_ratios
from src.analysis.clinical_correlation_matrix import run_clinical_correlation_matrix
from src.analysis.cluster_panel_summary import run_panel_cluster_summary
from src.analysis.covariate_models import run_covariate_region_models
from src.analysis.discriminant_spectra import run_discriminant_spectra
from src.analysis.disease_stratification import run_disease_stratification
from src.analysis.disease_subgroups import run_disease_subgroups
from src.analysis.endpoint_consistency import run_endpoint_consistency
from src.analysis.feature_coupling import run_feature_coupling
from src.analysis.integrated_candidates import run_integrated_candidates
from src.analysis.io_utils import write_csv_atomic
from src.analysis.journal_figures import run_journal_figures
from src.analysis.longitudinal_moca import run_longitudinal_moca
from src.analysis.minimal_panel import run_minimal_panel
from src.analysis.nmf_decomposition import run_nmf_decomposition
from src.analysis.phase_status import STATUS_DISABLED_OR_SKIPPED, STATUS_FAILED_EXECUTION, STATUS_OK, PhaseStatus
from src.analysis.pls_regression import run_pls_regression
from src.analysis.preregistered_panel import run_preregistered_panel
from src.analysis.robust_contrasts import run_confirmatory_panel_statistics
from src.analysis.robustness_ranking import run_robustness_ranking
from src.analysis.spectral_descriptors import run_spectral_descriptors
from src.analysis.spectral_regions import run_candidate_region_detector
from src.analysis.spectral_shape import run_spectral_shape
from src.analysis.splsda import run_splsda_classifier
from src.reporting.candidate_region_report import run_candidate_region_report, run_panel_report_section
from src.statistics.multiple_testing import holm_adjust

logger = logging.getLogger(__name__)


def _phase_enabled(cfg: dict, key: str, default: bool = True) -> bool:
    return bool(cfg.get("analysis", {}).get(key, {}).get("enabled", default))


@contextmanager
def _phase_timer(
    timings: list[dict[str, object]],
    phase_key: str,
    phase_label: str,
    progress: Progress | None = None,
    task_id: int | None = None,
    outputs: dict[str, dict[str, object]] | None = None,
):
    if progress is not None and task_id is not None:
        progress.update(task_id, description=f"[bold blue]{phase_label}")
    start = time.perf_counter()
    status = "ok"
    try:
        yield
    except BaseException as exc:
        status = "failed"
        if outputs is not None:
            outputs[phase_key] = {
                "_status": PhaseStatus(
                    phase_key=phase_key, status=STATUS_FAILED_EXECUTION, reason=str(exc),
                ).as_row()
            }
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
        if progress is not None and task_id is not None:
            progress.advance(task_id)


def _write_phase_timings(timings: list[dict[str, object]]) -> Path | None:
    if not timings:
        return None
    path = get_run_metadata_dir() / "disease_only_cross_region_phase_timing.csv"
    write_csv_atomic(pd.DataFrame(timings), path, index=False)
    return path


def _write_phase_status_manifest(outputs: dict[str, dict[str, object]]) -> Path | None:
    """Aggregate each phase's outcome into a single skip-manifest CSV.

    Phases that skip their CSV writes because the whole-phase input was empty
    (see ``src.analysis.phase_status``) report a ``"_status"`` row directly.
    Phases that ran normally are synthesized as ``ok``; phases whose output
    dict is empty (for example, disabled via config) are synthesized as
    ``disabled_or_skipped``. Phases with no key in ``outputs`` get no row.
    """
    if not outputs:
        return None

    rows: list[dict[str, object]] = []
    for phase_key, phase_outputs in outputs.items():
        if phase_outputs and "_status" in phase_outputs:
            row = dict(phase_outputs["_status"])
            row.setdefault("phase_key", phase_key)
            rows.append(row)
        elif not phase_outputs:
            rows.append(PhaseStatus(phase_key=phase_key, status=STATUS_DISABLED_OR_SKIPPED).as_row())
        else:
            rows.append(
                PhaseStatus(phase_key=phase_key, status=STATUS_OK, n_artifacts=len(phase_outputs)).as_row()
            )

    path = get_run_metadata_dir() / "phase_status.csv"
    write_csv_atomic(pd.DataFrame(rows), path, index=False)
    return path


def _collect_cross_phase_multiplicity(
    outputs: dict[str, dict[str, Path]],
) -> pd.DataFrame | None:
    """Collect cluster p-values across exploratory scans and apply Holm correction.

    The primary statistical control is within-phase cluster permutation correction.
    This adds family-wise control across phases, spectral blocks, and endpoints.

    Phase 9 (sPLS-DA) outputs are excluded because the gate makes them non-exchangeable
    with the unconditional Phase 1-5 tests.
    """
    source_keys: list[tuple[str, str, str]] = [
        ("clinical_correlation_matrix", "clinical_correlation_matrix", "Phase1_ClinicalCorr"),
        ("longitudinal_moca", "delta_moca_associations", "Phase2_LongitudinalMoCA"),
        ("disease_stratification", "stratification_wavenumber_tests", "Phase3_Stratification"),
    ]
    records: list[dict] = []
    for phase_output_key, file_key, phase_label in source_keys:
        phase_outputs = outputs.get(phase_output_key, {})
        csv_path = phase_outputs.get(file_key)
        if csv_path is None:
            continue
        df = pd.read_csv(csv_path)
        if "cluster_p_value" not in df.columns:
            continue
        cluster_rows = df[df["cluster_p_value"].notna()].copy()
        if cluster_rows.empty:
            continue
        cluster_rows = cluster_rows.drop_duplicates(
            subset=["cluster_id"] if "cluster_id" in cluster_rows.columns else None,
            keep="first",
        )
        for _, row in cluster_rows.iterrows():
            records.append({
                "phase": phase_label,
                "spectral_block": row.get("spectral_block", pd.NA),
                "cluster_id": row.get("cluster_id", pd.NA),
                "target": row.get("target", row.get("endpoint", pd.NA)),
                "cluster_p_value": float(row["cluster_p_value"]),
            })

    if not records:
        return None

    summary = pd.DataFrame(records)
    finite_mask = summary["cluster_p_value"].notna() & np.isfinite(summary["cluster_p_value"])
    q = np.full(len(summary), np.nan)
    if finite_mask.any():
        q[finite_mask] = holm_adjust(summary.loc[finite_mask, "cluster_p_value"].to_numpy(dtype=float))
    summary["family_id"] = "exploratory_whole_spectrum_clusters"
    summary["raw_p"] = summary["cluster_p_value"]
    summary["within_scan_p"] = summary["cluster_p_value"]
    summary["family_adjusted_p"] = q
    summary["inference_role"] = "exploratory"
    summary["cross_phase_significant_holm05"] = summary["family_adjusted_p"] < 0.05
    return summary.sort_values("cluster_p_value").reset_index(drop=True)


def _cfg_for_preregistered_minimal_panel(cfg: dict) -> dict:
    panel_cfg = deepcopy(cfg)
    analysis = panel_cfg.setdefault("analysis", {})
    minimal = dict(analysis.get("minimal_panel", {}))
    minimal.update(analysis.get("panel_minimal_model", {}))
    minimal["enabled"] = True
    minimal["feature_source"] = "preregistered_panel"
    analysis["minimal_panel"] = minimal
    return panel_cfg


def _run_clinical_and_candidate_phases(
    *,
    feature_matrix_path: str | Path | None,
    cfg: dict,
    outputs: dict[str, dict[str, object]],
    phase: Callable[[str, str], AbstractContextManager[None]],
) -> tuple[dict[str, Path], dict[str, Path]]:
    clinical_outputs: dict[str, Path] = {}
    if _phase_enabled(cfg, "clinical_correlation_matrix"):
        with phase("clinical_correlation_matrix", "Clinical correlation matrix"):
            clinical_outputs = run_clinical_correlation_matrix(
                feature_matrix_path=feature_matrix_path,
                cfg=cfg,
            )
            outputs["clinical_correlation_matrix"] = clinical_outputs

    longitudinal_outputs: dict[str, Path] = {}
    if _phase_enabled(cfg, "longitudinal_moca"):
        with phase("longitudinal_moca", "Longitudinal Delta MoCA"):
            longitudinal_outputs = run_longitudinal_moca(
                feature_matrix_path=feature_matrix_path,
                cfg=cfg,
            )
            outputs["longitudinal_moca"] = longitudinal_outputs

    stratification_outputs: dict[str, Path] = {}
    if _phase_enabled(cfg, "disease_stratification"):
        with phase("disease_stratification", "Disease stratification"):
            stratification_outputs = run_disease_stratification(
                feature_matrix_path=feature_matrix_path,
                cfg=cfg,
            )
            outputs["disease_stratification"] = stratification_outputs

    candidate_outputs: dict[str, Path] = {}
    if _phase_enabled(cfg, "candidate_regions"):
        with phase("candidate_regions", "Candidate-region detector"):
            candidate_outputs = run_candidate_region_detector(
                clinical_outputs=clinical_outputs,
                longitudinal_outputs=longitudinal_outputs,
                stratification_outputs=stratification_outputs,
                cfg=cfg,
            )
            outputs["candidate_regions"] = candidate_outputs

    if _phase_enabled(cfg, "covariate_models"):
        with phase("covariate_models", "Covariate-adjusted region models"):
            outputs["covariate_models"] = run_covariate_region_models(
                feature_matrix_path=feature_matrix_path,
                candidate_regions_path=candidate_outputs.get("candidate_regions_cluster_corrected"),
                cfg=cfg,
            )

    pls_outputs: dict[str, Path] = {}
    if _phase_enabled(cfg, "pls_regression"):
        with phase("pls_regression", "PLS1 regression"):
            pls_outputs = run_pls_regression(
                feature_matrix_path=feature_matrix_path,
                cfg=cfg,
            )
            outputs["pls_regression"] = pls_outputs

    splsda_outputs: dict[str, Path] = {}
    if _phase_enabled(cfg, "splsda"):
        with phase("splsda", "PLS-L1DA exploratory classifier"):
            splsda_outputs = run_splsda_classifier(
                feature_matrix_path=feature_matrix_path,
                pls_outputs=pls_outputs,
                cfg=cfg,
            )
            outputs["splsda"] = splsda_outputs

    if _phase_enabled(cfg, "biochemical_ratios"):
        with phase("biochemical_ratios", "Biochemical ratio analysis"):
            outputs["biochemical_ratios"] = run_biochemical_ratios(
                feature_matrix_path=feature_matrix_path,
                cfg=cfg,
            )
    return pls_outputs, splsda_outputs


def _run_fixed_panel_phases(
    *,
    feature_matrix_path: str | Path | None,
    cfg: dict,
    outputs: dict[str, dict[str, object]],
    phase: Callable[[str, str], AbstractContextManager[None]],
) -> dict[str, Path]:
    preregistered_outputs: dict[str, Path] = {}
    if _phase_enabled(cfg, "preregistered_panel"):
        with phase("preregistered_panel", "Fixed post hoc Raman panel"):
            preregistered_outputs = run_preregistered_panel(
                feature_matrix_path=feature_matrix_path,
                cfg=cfg,
            )
            outputs["preregistered_panel"] = preregistered_outputs

        with phase("confirmatory_panel", "Exploratory fixed-panel statistics"):
            outputs["confirmatory_panel"] = run_confirmatory_panel_statistics(
                feature_matrix_path=feature_matrix_path,
                panel_matrix_path=preregistered_outputs.get("panel_feature_matrix"),
                cfg=cfg,
            )

        if _phase_enabled(cfg, "endpoint_consistency"):
            with phase("endpoint_consistency", "Endpoint consistency analysis"):
                outputs["endpoint_consistency"] = run_endpoint_consistency(
                    associations_path=outputs["confirmatory_panel"].get("panel_continuous_associations"),
                    cfg=cfg,
                )

        if _phase_enabled(cfg, "robustness_ranking"):
            with phase("robustness_ranking", "Descriptive evidence flags"):
                outputs["robustness_ranking"] = run_robustness_ranking(
                    contrasts_path=outputs["confirmatory_panel"].get("panel_contrast_tests"),
                    consistency_path=outputs.get("endpoint_consistency", {}).get("endpoint_consistency"),
                    cfg=cfg,
                )
    return preregistered_outputs


def _run_candidate_followup_phases(
    *,
    feature_matrix_path: str | Path | None,
    cfg: dict,
    outputs: dict[str, dict[str, object]],
    phase: Callable[[str, str], AbstractContextManager[None]],
    pls_outputs: dict[str, Path],
    splsda_outputs: dict[str, Path],
    preregistered_outputs: dict[str, Path],
) -> None:
    integrated_outputs: dict[str, Path] = {}
    if _phase_enabled(cfg, "integrated_candidates"):
        with phase("integrated_candidates", "Integrated candidate evidence table"):
            integrated_outputs = run_integrated_candidates(
                feature_matrix_path=feature_matrix_path,
                pls_feature_effects_path=pls_outputs.get("pls_feature_effects"),
                pls_performance_path=pls_outputs.get("pls_regression_performance"),
                splsda_bands_path=splsda_outputs.get("pls_l1da_selected_bands", ""),
                cfg=cfg,
            )
            outputs["integrated_candidates"] = integrated_outputs

    if _phase_enabled(cfg, "disease_subgroups"):
        with phase("disease_subgroups", "Disease subgroup analysis"):
            outputs["disease_subgroups"] = run_disease_subgroups(
                feature_matrix_path=feature_matrix_path,
                integrated_candidates_path=integrated_outputs.get("integrated_candidate_regions"),
                cfg=cfg,
            )

    if preregistered_outputs and _phase_enabled(cfg, "panel_cluster_summary"):
        with phase("panel_cluster_summary", "Panel cluster summary"):
            outputs["panel_cluster_summary"] = run_panel_cluster_summary(
                feature_matrix_path=feature_matrix_path,
                panel_matrix_path=preregistered_outputs.get("panel_feature_matrix"),
                subgroup_assignments_path=outputs.get("disease_subgroups", {}).get("disease_subgroup_assignments"),
                cfg=cfg,
            )

    panel_outputs: dict[str, Path] = {}
    if _phase_enabled(cfg, "minimal_panel"):
        with phase("minimal_panel", "Minimal candidate Raman panel"):
            panel_outputs = run_minimal_panel(
                feature_matrix_path=feature_matrix_path,
                integrated_candidates_path=integrated_outputs.get("integrated_candidate_regions"),
                cfg=cfg,
            )
            outputs["minimal_panel"] = panel_outputs

    preregistered_minimal_outputs: dict[str, Path] = {}
    if preregistered_outputs and _phase_enabled(cfg, "panel_minimal_model"):
        with phase("panel_minimal_model", "Fixed-panel minimal model"):
            preregistered_minimal_outputs = run_minimal_panel(
                feature_matrix_path=feature_matrix_path,
                panel_matrix_path=preregistered_outputs.get("panel_feature_matrix"),
                output_dir=get_confirmatory_panel_dir(),
                cfg=_cfg_for_preregistered_minimal_panel(cfg),
            )
            outputs["panel_minimal_model"] = preregistered_minimal_outputs

    panel_report_outputs: dict[str, Path] = {}
    if preregistered_outputs and _phase_enabled(cfg, "panel_report"):
        with phase("panel_report", "Four-domain panel report"):
            panel_report_outputs = run_panel_report_section(
                evidence_flags_path=outputs.get("robustness_ranking", {}).get("panel_evidence_flags"),
                anchor_result_path=outputs.get("confirmatory_panel", {}).get("primary_anchor_result"),
                cfg=cfg,
            )
            outputs["panel_report"] = panel_report_outputs

    if _phase_enabled(cfg, "publication_report"):
        with phase("publication_report", "Publication report generator"):
            outputs["publication_report"] = run_candidate_region_report(
                integrated_candidates_path=integrated_outputs.get("integrated_candidate_regions"),
                minimal_panel_path=panel_outputs.get("minimal_candidate_panel"),
                minimal_panel_status=panel_outputs.get("_status"),
                panel_report_section_path=panel_report_outputs.get("panel_report_section"),
                cfg=cfg,
            )


def _run_spectral_structure_phases(
    *,
    feature_matrix_path: str | Path | None,
    cfg: dict,
    outputs: dict[str, dict[str, object]],
    phase: Callable[[str, str], AbstractContextManager[None]],
    preregistered_outputs: dict[str, Path],
) -> None:
    if _phase_enabled(cfg, "spectral_shape"):
        with phase("spectral_shape", "Spectral-shape analysis"):
            outputs["spectral_shape"] = run_spectral_shape(
                feature_matrix_path=feature_matrix_path,
                cfg=cfg,
            )

    if _phase_enabled(cfg, "nmf_decomposition"):
        with phase("nmf_decomposition", "NMF decomposition"):
            outputs["nmf_decomposition"] = run_nmf_decomposition(
                feature_matrix_path=feature_matrix_path,
                cfg=cfg,
            )

    if _phase_enabled(cfg, "spectral_descriptors"):
        with phase("spectral_descriptors", "Spectral entropy / dispersion"):
            outputs["spectral_descriptors"] = run_spectral_descriptors(
                feature_matrix_path=feature_matrix_path,
                cfg=cfg,
            )

    if _phase_enabled(cfg, "discriminant_spectra"):
        with phase("discriminant_spectra", "Difference-spectrum signatures"):
            outputs["discriminant_spectra"] = run_discriminant_spectra(
                feature_matrix_path=feature_matrix_path,
                cfg=cfg,
            )

    if preregistered_outputs and _phase_enabled(cfg, "feature_coupling"):
        with phase("feature_coupling", "Cross-feature coupling"):
            outputs["feature_coupling"] = run_feature_coupling(
                panel_matrix_path=preregistered_outputs.get("panel_feature_matrix"),
                cfg=cfg,
            )

    if _phase_enabled(cfg, "journal_figures"):
        with phase("journal_figures", "Journal figure set"):
            outputs["journal_figures"] = run_journal_figures(
                panel_matrix_path=preregistered_outputs.get("panel_feature_matrix")
                if preregistered_outputs else None,
                cfg=cfg,
            )


def _write_cross_phase_multiplicity(
    outputs: dict[str, dict[str, object]],
    cfg: dict,
) -> None:
    try:
        multiplicity_df = _collect_cross_phase_multiplicity(outputs)
        if multiplicity_df is not None and not multiplicity_df.empty:
            multiplicity_path = get_run_metadata_dir() / "cross_phase_multiplicity_summary.csv"
            write_csv_atomic(multiplicity_df, multiplicity_path, index=False)
            outputs["cross_phase_multiplicity"] = {"cross_phase_multiplicity_summary": multiplicity_path}
            logger.info(
                "Cross-phase multiplicity summary written: %d cluster entries, %d survive Holm p<0.05",
                len(multiplicity_df),
                int(multiplicity_df["cross_phase_significant_holm05"].sum()),
            )
    except Exception as exc:
        logger.warning("Cross-phase multiplicity summary failed: %s", exc)
        outputs["cross_phase_multiplicity"] = {
            "_status": PhaseStatus(
                phase_key="cross_phase_multiplicity",
                status=STATUS_FAILED_EXECUTION,
                reason=str(exc),
            ).as_row()
        }
        if bool(cfg.get("analysis", {}).get("publication_run", False)):
            raise RuntimeError("Cross-phase multiplicity summary failed") from exc


def run_disease_only_cross_region_pipeline(
    *,
    feature_matrix_path: str | Path | None,
    cfg: dict | None = None,
) -> dict[str, dict[str, Path]]:
    """Run disease-only cross-region phases that depend on Biological PCA exports."""
    cfg = config if cfg is None else cfg
    outputs: dict[str, dict[str, Path]] = {}
    phase_timings: list[dict[str, object]] = []

    _progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        transient=False,
    )
    # 25 = number of `with phase(...)` blocks; config-disabled phases never advance the bar.
    _task = _progress.add_task("Disease-only cross-region phases", total=25)

    phase = partial(_phase_timer, phase_timings, progress=_progress, task_id=_task, outputs=outputs)
    try:
        with _progress:
            pls_outputs, splsda_outputs = _run_clinical_and_candidate_phases(
                feature_matrix_path=feature_matrix_path, cfg=cfg, outputs=outputs,
                phase=phase,
            )
            preregistered_outputs = _run_fixed_panel_phases(
                feature_matrix_path=feature_matrix_path, cfg=cfg, outputs=outputs,
                phase=phase,
            )
            _run_candidate_followup_phases(
                feature_matrix_path=feature_matrix_path, cfg=cfg, outputs=outputs,
                phase=phase,
                pls_outputs=pls_outputs, splsda_outputs=splsda_outputs,
                preregistered_outputs=preregistered_outputs,
            )
            _run_spectral_structure_phases(
                feature_matrix_path=feature_matrix_path, cfg=cfg, outputs=outputs,
                phase=phase,
                preregistered_outputs=preregistered_outputs,
            )

        _write_cross_phase_multiplicity(outputs, cfg)
    finally:
        _write_phase_status_manifest(outputs)
        _write_phase_timings(phase_timings)
    return outputs
