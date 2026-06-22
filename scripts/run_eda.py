#!/usr/bin/env python3
"""Executable runner for the Raman EDA pipeline.

This is the only supported command-line entry point for EDA.
``src.analysis.run_eda.run_eda()`` remains available as an importable API.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "_raman_cli_util", ROOT / "src" / "cli_util.py"
)
_cli_util = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_cli_util)
_cli_util.apply_config_override("config/eda.yaml", ROOT)

from src import (
    config,
    get_biological_pca_canonical_dir,
    get_run_metadata_dir,
    setup_logging,
    _output_dir,
    _range_label,
)
from src.data_preparation import data_source_summary, resolve_data_source, write_analysis_provenance
from src.clinical.endpoints import build_and_write_clinical_endpoints
from src.terminal_output import print_kv_table

from src.analysis._single_range import _VALID_RANGES
from src.analysis.biological_pca import run_biological_pca
from src.analysis.clinical_correlation_matrix import run_clinical_correlation_matrix
from src.analysis.covariate_models import run_covariate_region_models
from src.analysis.disease_stratification import run_disease_stratification
from src.analysis.eda_mode import (
    EDA_ANALYSIS_MODES,
    apply_eda_analysis_mode,
    eda_output_root,
    resolve_eda_analysis_mode,
    validate_eda_mode_config,
)
from src.analysis.longitudinal_moca import run_longitudinal_moca
from src.analysis.pls_regression import run_pls_regression
from src.analysis.run_eda import run_eda
from src.analysis.integrated_candidates import run_integrated_candidates
from src.analysis.minimal_panel import run_minimal_panel
from src.analysis.spectral_regions import run_candidate_region_detector
from src.analysis.splsda import run_splsda_classifier
from src.reporting.candidate_region_report import run_candidate_region_report

logger = logging.getLogger(__name__)


def _attach_run_log_file(log_path: Path) -> None:
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == log_path:
            return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    )
    root_logger.addHandler(handler)


def _set_analysis_output_root(mode: str) -> None:
    """Route all EDA-owned outputs to the selected analysis family."""
    config.setdefault("output", {})["eda_root"] = eda_output_root(mode)


def _apply_analysis_mode_runtime_defaults(mode: str) -> None:
    intra_disease_cfg = (
        config.setdefault("analysis", {})
        .setdefault("advanced", {})
        .setdefault("intra_disease", {})
    )
    intra_disease_cfg["enabled"] = mode == "disease_only"


def _requested_ranges(cli_range: str | None) -> list[str]:
    """Normalize requested ranges so `both` expands to the full fused analysis."""
    if cli_range is not None:
        ranges = [cli_range]
    else:
        ranges = [str(r) for r in config.get("run", {}).get("ranges", ["1300"])]
    if "both" in ranges:
        return ["1300", "2850", "both"]
    return ranges


def _validate_ranges(ranges: list[str]) -> None:
    for spectral_range in ranges:
        if spectral_range not in _VALID_RANGES:
            raise ValueError(
                f"Invalid spectral range '{spectral_range}'. Valid options: {sorted(_VALID_RANGES)}"
            )


def _should_run_biological_pca(ranges: list[str], skip_biological_pca: bool) -> bool:
    biological_cfg = config.get("biological_pca", {})
    has_required_blocks = {"1300", "2850"}.issubset(set(ranges))
    return (
        bool(biological_cfg.get("enabled", True))
        and not skip_biological_pca
        and has_required_blocks
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full Raman EDA pipeline")
    parser.add_argument(
        "--range",
        type=str,
        default=None,
        help=(
            "Spectral range to analyze: '1300', '2850', or 'both'. "
            "Omit to use run.ranges from config/eda.yaml."
        ),
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip preprocessing (use raw spectra)",
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Limit subjects per group (for quick testing)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML (overrides config/eda.yaml). "
        "Relative paths are resolved from the project root. "
        "Supports _extends: <base.yaml> for inheritance.",
    )
    parser.add_argument(
        "--analysis-mode",
        choices=list(EDA_ANALYSIS_MODES),
        default=None,
        help=(
            "Explicit EDA analysis family. Overrides analysis.mode from config. "
            "Valid options: disease_vs_control, disease_only."
        ),
    )
    parser.add_argument(
        "--skip-biological-pca",
        action="store_true",
        help="Skip the subject-level Biological PCA layer.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Override data.source with one of: raw_text, parsed, preprocessed.",
    )
    return parser.parse_args()


def _write_warning_summaries(output_root: Path) -> list[str]:
    warnings: list[str] = []
    for report_path in sorted(output_root.rglob("pc1_dominance_report.md")):
        text = report_path.read_text(encoding="utf-8")
        content_lines = [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]
        summary_line = next(
            (line for line in content_lines if "PC1 variance:" in line or line.startswith("PC1 explains")),
            content_lines[0] if content_lines else "PC1 dominance report written.",
        )
        relative_report = report_path.relative_to(output_root)
        warnings.append(f"{relative_report.parent.name}: {summary_line} See {relative_report}")

    metadata_dir = get_run_metadata_dir()
    (metadata_dir / "warning_summary.md").write_text(
        "\n".join(
            ["# Warning Summary", ""]
            + ([f"- {warning}" for warning in warnings] if warnings else ["No warnings collected."])
        ),
        encoding="utf-8",
    )
    (metadata_dir / "warning_summary.csv").write_text(
        "warning\n" + "\n".join(json.dumps(warning) for warning in warnings),
        encoding="utf-8",
    )
    return warnings


def _log_final_summary(ranges: list[str], biological_ran: bool, warnings: list[str]) -> None:
    output_root = _output_dir("eda_subdir")
    logger.info("EDA complete - output root: %s", output_root)
    if biological_ran:
        logger.info("Biological PCA:")
        for model_name in ("PCA_1330", "PCA_2850", "PCA_whole"):
            logger.info(
                "  %s -> %s/",
                model_name,
                get_biological_pca_canonical_dir(model_name).relative_to(output_root),
            )
    if warnings:
        logger.warning("Warnings:")
        for warning in warnings:
            logger.warning("  %s", warning)


def _main() -> None:
    args = _parse_args()
    setup_logging()
    analysis_mode = resolve_eda_analysis_mode(config, override=args.analysis_mode)
    apply_eda_analysis_mode(config, analysis_mode)
    _apply_analysis_mode_runtime_defaults(analysis_mode)
    validate_eda_mode_config(config, analysis_mode)
    _set_analysis_output_root(analysis_mode)
    metadata_dir = get_run_metadata_dir()
    _attach_run_log_file(metadata_dir / "run.log")
    data_source = resolve_data_source(config, source_override=args.source)
    print_kv_table("Active Data Source", data_source_summary(data_source))
    ranges = _requested_ranges(args.range)
    write_analysis_provenance(
        _output_dir("eda_subdir"),
        data_source,
        config,
        command=" ".join(sys.argv),
    )
    write_analysis_provenance(
        metadata_dir,
        data_source,
        config,
        command=" ".join(sys.argv),
    )

    try:
        _validate_ranges(ranges)
        if (
            analysis_mode == "disease_only"
            and bool(config.get("analysis", {}).get("clinical_endpoints", {}).get("enabled", True))
        ):
            endpoint_result = build_and_write_clinical_endpoints(cfg=config)
            config.setdefault("analysis", {})["clinical_endpoints_runtime"] = {
                "subject_table_path": str(endpoint_result.output_dir / "clinical_subject_table.csv")
            }
            logger.info("Clinical endpoints written to %s", endpoint_result.output_dir)

        results: dict[str, dict] = {}
        for spectral_range in ranges:
            results[spectral_range] = run_eda(
                spectral_range=spectral_range,
                skip_preprocess=args.skip_preprocess,
                max_subjects=args.max_subjects,
            )

        biological_ran = False
        biological_outputs: dict[str, Path] = {}
        if _should_run_biological_pca(ranges, args.skip_biological_pca):
            biological_outputs = run_biological_pca(max_subjects=args.max_subjects)
            biological_ran = True
            if analysis_mode == "disease_only":
                feature_matrix_path = biological_outputs.get("subject_feature_matrix_whole")
                clinical_matrix_cfg = config.get("analysis", {}).get("clinical_correlation_matrix", {})
                longitudinal_cfg = config.get("analysis", {}).get("longitudinal_moca", {})
                stratification_cfg = config.get("analysis", {}).get("disease_stratification", {})
                candidate_cfg = config.get("analysis", {}).get("candidate_regions", {})
                covariate_cfg = config.get("analysis", {}).get("covariate_models", {})
                clinical_outputs: dict[str, Path] = {}
                longitudinal_outputs: dict[str, Path] = {}
                stratification_outputs: dict[str, Path] = {}
                if bool(clinical_matrix_cfg.get("enabled", True)):
                    logger.info("Starting disease-only clinical correlation matrix")
                    clinical_outputs = run_clinical_correlation_matrix(
                        feature_matrix_path=feature_matrix_path,
                        cfg=config,
                    )
                if bool(longitudinal_cfg.get("enabled", True)):
                    logger.info("Starting longitudinal Delta MoCA analysis")
                    longitudinal_outputs = run_longitudinal_moca(
                        feature_matrix_path=feature_matrix_path,
                        cfg=config,
                    )
                if bool(stratification_cfg.get("enabled", True)):
                    logger.info("Starting disease stratification analysis")
                    stratification_outputs = run_disease_stratification(
                        feature_matrix_path=feature_matrix_path,
                        cfg=config,
                    )
                candidate_outputs: dict[str, Path] = {}
                if bool(candidate_cfg.get("enabled", True)):
                    logger.info("Starting unified candidate-region detector")
                    candidate_outputs = run_candidate_region_detector(
                        clinical_outputs=clinical_outputs,
                        longitudinal_outputs=longitudinal_outputs,
                        stratification_outputs=stratification_outputs,
                        cfg=config,
                    )
                if bool(covariate_cfg.get("enabled", True)):
                    logger.info("Starting covariate-adjusted region models")
                    run_covariate_region_models(
                        feature_matrix_path=feature_matrix_path,
                        candidate_regions_path=candidate_outputs.get("candidate_regions_cluster_corrected"),
                        cfg=config,
                    )
                
                pls_cfg = config.get("analysis", {}).get("pls_regression", {})
                pls_outputs: dict[str, Path] = {}
                if bool(pls_cfg.get("enabled", True)):
                    logger.info("Starting PLS1 regression (Phase 8)")
                    pls_outputs = run_pls_regression(
                        feature_matrix_path=feature_matrix_path,
                        cfg=config,
                    )

                splsda_cfg = config.get("analysis", {}).get("splsda", {})
                splsda_outputs: dict[str, Path] = {}
                if bool(splsda_cfg.get("enabled", True)):
                    logger.info("Starting PLS-L1DA exploratory classifier (Phase 9)")
                    splsda_outputs = run_splsda_classifier(
                        feature_matrix_path=feature_matrix_path,
                        pls_outputs=pls_outputs,
                        cfg=config,
                    )

                integrated_cfg = config.get("analysis", {}).get("integrated_candidates", {})
                integrated_outputs: dict[str, Path] = {}
                if bool(integrated_cfg.get("enabled", True)):
                    logger.info("Starting integrated candidate evidence table (Phase 11)")
                    integrated_outputs = run_integrated_candidates(
                        feature_matrix_path=feature_matrix_path,
                        pls_vip_path=pls_outputs.get("pls_vip_bands"),
                        splsda_bands_path=splsda_outputs.get("pls_l1da_selected_bands"),
                        cfg=config,
                    )

                panel_cfg = config.get("analysis", {}).get("minimal_panel", {})
                panel_outputs: dict[str, Path] = {}
                if bool(panel_cfg.get("enabled", True)):
                    logger.info("Starting minimal candidate Raman panel (Phase 13)")
                    panel_outputs = run_minimal_panel(
                        feature_matrix_path=feature_matrix_path,
                        integrated_candidates_path=integrated_outputs.get("integrated_candidate_regions"),
                        cfg=config,
                    )

                report_cfg = config.get("analysis", {}).get("publication_report", {})
                if bool(report_cfg.get("enabled", True)):
                    logger.info("Starting publication report generator (Phase 14)")
                    run_candidate_region_report(
                        integrated_candidates_path=integrated_outputs.get("integrated_candidate_regions"),
                        minimal_panel_path=panel_outputs.get("minimal_candidate_panel"),
                        cfg=config,
                    )
        warnings = _write_warning_summaries(_output_dir("eda_subdir")) if biological_ran else []
        _log_final_summary(ranges, biological_ran, warnings)
    except (ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    _main()
