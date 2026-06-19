#!/usr/bin/env python3
"""Run strict subject-level Biological PCA outputs."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import importlib.util
import logging
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "_raman_cli_util", ROOT / "src" / "cli_util.py"
)
_cli_util = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_cli_util)
_cli_util.apply_config_override("config/eda.yaml", ROOT)

from src import config, setup_logging
from src.analysis.biological_pca import run_biological_pca

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run subject-level Biological PCA")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to biological_pca.output_root from config.",
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Use the first N sorted subject_id values for deterministic smoke runs.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML. Relative paths are resolved from the project root.",
    )
    return parser.parse_args()


def _main() -> None:
    setup_logging()
    args = _parse_args()
    subject_group = config.get("data", {}).get("subject_group")
    if args.output_dir is None:
        if subject_group == "disease_only":
            config.setdefault("output", {})["eda_root"] = "Results_intra_disease"
        else:
            config.setdefault("output", {})["eda_root"] = "Results_control_vs_disease"
    try:
        output_dir = run_biological_pca(output_dir=args.output_dir, max_subjects=args.max_subjects)
        logger.info("Completed Biological PCA: %s", output_dir)
    except Exception as exc:
        logger.error("Biological PCA failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    _main()
