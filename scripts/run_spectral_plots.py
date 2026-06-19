# ruff: noqa: E402
#!/usr/bin/env python3
"""Independent script wrapper for spectral comparison and QC plots."""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "_raman_cli_util", ROOT / "src" / "cli_util.py"
)
_cli_util = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_cli_util)
_cli_util.apply_config_override("config/plots.yaml", ROOT)

from src import setup_logging
from src.analysis.spectral_plot_runner import _parse_args, run_spectral_plots

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    args = _parse_args()
    outputs = run_spectral_plots(outdir=args.outdir, max_subjects=args.max_subjects)
    logger.info("QC plots: %s", outputs["qc_dir"])
    logger.info("Spectral plots: %s", outputs["spectral_plot_dir"])


if __name__ == "__main__":
    main()
