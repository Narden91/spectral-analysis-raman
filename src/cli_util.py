from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def ensure_project_root_on_path(root: Path | None = None) -> Path:
    project_root = root or Path(__file__).resolve().parents[1]
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return project_root


def apply_config_override(default: str, root: Path | None = None) -> None:
    project_root = ensure_project_root_on_path(root)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default=None)
    args, _ = parser.parse_known_args()

    if args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = project_root / config_path
        os.environ["RAMAN_CONFIG"] = str(config_path)
        return

    if "RAMAN_CONFIG" not in os.environ:
        os.environ["RAMAN_CONFIG"] = str(project_root / default)
