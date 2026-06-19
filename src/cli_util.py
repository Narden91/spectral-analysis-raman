from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable, TypeVar


T = TypeVar("T")


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


def run_multi_seed(
    n_runs: int,
    base_seed: int,
    base_dir: Path,
    fn: Callable[[int, int, Path], T],
) -> list[T]:
    results: list[T] = []
    for run_idx in range(int(n_runs)):
        seed = int(base_seed) + run_idx
        run_dir = base_dir / f"run_{run_idx + 1:03d}" if int(n_runs) > 1 else base_dir
        results.append(fn(run_idx, seed, run_dir))
    return results
