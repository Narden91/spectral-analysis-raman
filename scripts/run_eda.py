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

from src import config, get_eda_phase_dir, setup_logging, _output_dir
from src.analysis._single_range import _PHASE_DIRS, _VALID_RANGES
from src.analysis.analysis_inventory import run_inventory
from src.analysis.cross_range_report import write_cross_range_report
from src.analysis.run_eda import run_eda

logger = logging.getLogger(__name__)


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
    return parser.parse_args()


def _main() -> None:
    setup_logging()
    args = _parse_args()

    subject_group = config.get("data", {}).get("subject_group")
    if subject_group == "disease_only":
        config.setdefault("output", {})["eda_root"] = "Results_intra_disease"
    else:
        config.setdefault("output", {})["eda_root"] = "Results_control_vs_disease"

    if args.range is not None:
        ranges = [args.range]
    else:
        ranges = [str(r) for r in config.get("run", {}).get("ranges", ["1300"])]

    if "both" in ranges:
        ranges = ["1300", "2850", "both"]

    try:
        for r in ranges:
            if r not in _VALID_RANGES:
                raise ValueError(
                    f"Invalid spectral range '{r}'. Valid options: {sorted(_VALID_RANGES)}"
                )

        for spectral_range in ranges:
            run_eda(
                spectral_range=spectral_range,
                skip_preprocess=args.skip_preprocess,
                max_subjects=args.max_subjects,
            )

        single_ranges = [r for r in ranges if r != "both"]
        if len(single_ranges) >= 1:
            eda_base = _output_dir("eda_subdir")

            def _make_phase_dir_fn(r: str):
                def _fn(key: str) -> Path:
                    return get_eda_phase_dir(r, _PHASE_DIRS[key])

                return _fn

            phase_dir_fns = {r: _make_phase_dir_fn(r) for r in single_ranges}
            report_dir = eda_base / "cross_range_report"
            write_cross_range_report(
                ranges=single_ranges,
                phase_dir_fns=phase_dir_fns,
                output_base=eda_base,
                out_dir=report_dir,
            )

            run_inventory(
                eda_base=eda_base,
                output_dir=eda_base / "analysis_inventory",
            )
    except (ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    _main()
