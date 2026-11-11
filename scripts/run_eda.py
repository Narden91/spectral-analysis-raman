#!/usr/bin/env python3
"""Executable runner for the Raman EDA pipeline.

This is the only supported command-line entry point for EDA.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import subprocess
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
    _output_dir,
    config,
    get_biological_pca_canonical_dir,
    get_run_metadata_dir,
    setup_logging,
)
from src.analysis._single_range import _VALID_RANGES
from src.analysis._single_range import _run_single_range_eda as run_eda
from src.analysis.biological_pca import run_biological_pca
from src.analysis.control_vs_disease_ml import run_control_vs_disease_ml_pipeline
from src.analysis.disease_only_pipeline import run_disease_only_cross_region_pipeline
from src.analysis.eda_mode import (
    EDA_ANALYSIS_MODES,
    apply_eda_analysis_mode,
    eda_output_root,
    resolve_eda_analysis_mode,
    validate_control_vs_disease_ml_config,
    validate_eda_mode_config,
)
from src.analysis.io_utils import write_text_atomic
from src.clinical.endpoints import build_and_write_clinical_endpoints
from src.data_preparation import (
    data_source_summary,
    resolve_data_source,
    write_analysis_provenance,
)
from src.terminal_output import print_kv_table, print_status

logger = logging.getLogger(__name__)


_CONTROL_VS_DISEASE_LAYOUT = """# Control vs. Disease Results

Folders follow execution order. `00_run_metadata/run.log` records progress and
ends with `EDA complete` after a successful run.

## Top-level workflow

1. `00_run_metadata/` — resolved config, data manifest, command, log, timings
2. `01_fingerprint_1300/` — fingerprint-region EDA
3. `02_ch_stretch_2850/` — CH-stretch-region EDA
4. `03_whole_spectrum_1300_plus_2850/` — fused whole-spectrum EDA
5. `04_cross_region_summary/` — subject feature matrices and PCA summaries
6. `05_ml_classification/` — exploratory subject-level classification

Single-range folders use: `01_group_comparisons`, `02_area_level_pca_qc`,
`03_known_band_context`, `04_variance_context`, `05_region_robustness`, and
`06_biological_pca`.

ML folders use: `01_cohort`, `02_model_evaluation`, `03_confounder_checks`,
`04_permutation_tests`, `05_univariate_band_screen`, and `06_report`.

Only generated artifacts belong here. Raw data, caches, and disease-only
clinical analyses belong outside this result root.
"""


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


def _write_control_vs_disease_layout(output_root: Path) -> Path:
    return write_text_atomic(output_root / "README.md", _CONTROL_VS_DISEASE_LAYOUT)


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
        "--verbose",
        action="store_true",
        help="Show detailed INFO logs in the terminal. Detailed logs are always written to run.log.",
    )
    parser.add_argument(
        "--publication-run",
        action="store_true",
        help=(
            "Refuse quick/subset runs and require a clean Git worktree before generating a publication run. "
            "This does not make private results public; use export_publication_bundle afterwards."
        ),
    )
    parser.add_argument(
        "--resume-ml",
        action="store_true",
        help=(
            "Resume compatible control-vs-disease ML permutation/bootstrap checkpoints. "
            "Requires disease_vs_control mode with Biological PCA enabled."
        ),
    )
    return parser.parse_args()


def _validate_publication_run(args: argparse.Namespace, ranges: list[str]) -> None:
    """Reject non-reproducible shortcuts before any publication-run artifact is written."""
    if not args.publication_run:
        return
    violations: list[str] = []
    if args.max_subjects is not None:
        violations.append("--max-subjects is not allowed")
    if ranges != ["1300", "2850", "both"]:
        violations.append("the full fused range set (1300, 2850, both) is required")
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        violations.append("a clean tracked Git worktree is required")
    if violations:
        raise ValueError("Publication-run preflight failed: " + "; ".join(violations))


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
        logger.warning(
            "%d warning summaries written to %s",
            len(warnings),
            get_run_metadata_dir() / "warning_summary.md",
        )
        for warning in warnings:
            logger.info("Warning summary: %s", warning)


def _main() -> None:
    args = _parse_args()
    setup_logging(console_level=logging.INFO if args.verbose else logging.WARNING)
    analysis_mode = resolve_eda_analysis_mode(config, override=args.analysis_mode)
    apply_eda_analysis_mode(config, analysis_mode)
    _apply_analysis_mode_runtime_defaults(analysis_mode)
    validate_eda_mode_config(config, analysis_mode)
    if analysis_mode == "disease_vs_control":
        validate_control_vs_disease_ml_config(config)
    ranges = _requested_ranges(args.range)
    if args.resume_ml and (
        analysis_mode != "disease_vs_control" or not _should_run_biological_pca(ranges, args.skip_biological_pca)
    ):
        raise ValueError("--resume-ml requires disease_vs_control mode and a run that includes Biological PCA/ML")
    _validate_publication_run(args, ranges)
    config.setdefault("analysis", {})["publication_run"] = bool(args.publication_run)
    _set_analysis_output_root(analysis_mode)
    metadata_dir = get_run_metadata_dir()
    _attach_run_log_file(metadata_dir / "run.log")
    data_source = resolve_data_source(config)
    print_kv_table("Active Data Source", data_source_summary(data_source))
    print_status(
        f"Running EDA mode={analysis_mode} ranges={', '.join(ranges)} "
        f"output={_output_dir('eda_subdir')}"
    )
    write_analysis_provenance(
        metadata_dir,
        data_source,
        config,
        command=" ".join(sys.argv),
    )
    if analysis_mode == "disease_vs_control":
        _write_control_vs_disease_layout(_output_dir("eda_subdir"))

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

        for spectral_range in ranges:
            print_status(f"Starting {spectral_range} EDA")
            run_eda(
                spectral_range=spectral_range,
                max_subjects=args.max_subjects,
            )
            print_status(f"Finished {spectral_range} EDA")

        biological_ran = False
        biological_outputs: dict[str, Path] = {}
        if _should_run_biological_pca(ranges, args.skip_biological_pca):
            print_status("Starting Biological PCA")
            biological_outputs = run_biological_pca(max_subjects=args.max_subjects, data_source=data_source)
            biological_ran = True
            print_status("Finished Biological PCA")
            if analysis_mode == "disease_only":
                feature_matrix_path = biological_outputs.get("subject_feature_matrix_whole")
                print_status("Starting disease-only cross-region phases")
                run_disease_only_cross_region_pipeline(
                    feature_matrix_path=feature_matrix_path,
                    cfg=config,
                )
                print_status("Finished disease-only cross-region phases")
            elif analysis_mode == "disease_vs_control":
                feature_matrix_path = biological_outputs.get("subject_feature_matrix_whole")
                print_status("Starting control-vs-disease ML classification")
                run_control_vs_disease_ml_pipeline(
                    feature_matrix_path=feature_matrix_path,
                    cfg=config,
                    resume=args.resume_ml,
                    data_manifest_hash=data_source.manifest_hash,
                )
                print_status("Finished control-vs-disease ML classification")
        warnings = _write_warning_summaries(_output_dir("eda_subdir")) if biological_ran else []
        _log_final_summary(ranges, biological_ran, warnings)
        print_status(f"EDA complete. Outputs: {_output_dir('eda_subdir')}")
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("EDA interrupted; completed ML resampling checkpoints are available for --resume-ml")
        print_status("EDA interrupted. Re-run with --resume-ml to continue compatible ML resampling.")
        sys.exit(130)


if __name__ == "__main__":
    _main()
