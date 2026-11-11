#!/usr/bin/env python3
"""Export an allowlisted aggregate-only public bundle from a private run."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "_raman_cli_util", ROOT / "src" / "cli_util.py"
)
_cli_util = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_cli_util)
_cli_util.ensure_project_root_on_path(ROOT)

from src.reporting.publication_bundle import export_publication_bundle  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Export an aggregate-only publication bundle")
    parser.add_argument("--source", required=True, help="Private Results directory")
    parser.add_argument("--output", required=True, help="New destination directory for the public bundle")
    args = parser.parse_args()
    exported = export_publication_bundle(args.source, args.output)
    print(f"Exported {len(exported)} aggregate publication artifacts to {args.output}")


if __name__ == "__main__":
    main()
