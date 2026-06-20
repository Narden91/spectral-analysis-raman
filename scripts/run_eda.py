#!/usr/bin/env python3
"""Executable runner for the Raman EDA pipeline.

This is the only supported command-line entry point for EDA.
``src.analysis.run_eda.run_eda()`` remains available as an importable API.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import importlib.util
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

from src import config, setup_logging, _output_dir
from src.data_preparation import data_source_summary, resolve_data_source, write_analysis_provenance
from src.terminal_output import print_kv_table

from src.analysis._single_range import _VALID_RANGES
from src.analysis.biological_pca import run_biological_pca
from src.analysis.run_eda import run_eda

logger = logging.getLogger(__name__)


def _set_analysis_output_root() -> None:
    """Route all EDA-owned outputs to the selected analysis family."""
    subject_group = config.get("data", {}).get("subject_group")
    if subject_group == "disease_only":
        config.setdefault("output", {})["eda_root"] = "Results_intra_disease"
    else:
        config.setdefault("output", {})["eda_root"] = "Results_control_vs_disease"


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


def _main() -> None:
    setup_logging()
    args = _parse_args()

    _set_analysis_output_root()
    data_source = resolve_data_source(config, source_override=args.source)
    print_kv_table("Active Data Source", data_source_summary(data_source))
    ranges = _requested_ranges(args.range)
    write_analysis_provenance(
        _output_dir("eda_subdir"),
        data_source,
        config,
        command=" ".join(sys.argv),
    )

    try:
        _validate_ranges(ranges)

        results: dict[str, dict] = {}
        for spectral_range in ranges:
            results[spectral_range] = run_eda(
                spectral_range=spectral_range,
                skip_preprocess=args.skip_preprocess,
                max_subjects=args.max_subjects,
            )

        if _should_run_biological_pca(ranges, args.skip_biological_pca):
            run_biological_pca(max_subjects=args.max_subjects)
    except (ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    _main()
