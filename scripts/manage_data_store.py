#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import logging
import shutil
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

from src import config, setup_logging
from src.data_preparation import (
    build_parsed_store,
    build_preprocessed_store,
    data_source_summary,
    resolve_data_source,
)
from src.data_preparation.store import _store_dir, _validated_source
from src.data_preparation.validation import validate_store
from src.terminal_output import print_kv_table

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage parsed/preprocessed Raman data stores.")
    parser.add_argument("--config", type=str, default=None, help="Optional config override.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without mutating stores.")

    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("build", "validate", "info", "clean"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--source", choices=["parsed", "preprocessed"], required=True)
        sub.add_argument("--force", action="store_true", help="Rebuild even if store exists.")
    return parser.parse_args()


def _build(source: str, force: bool, dry_run: bool) -> None:
    if dry_run:
        logger.info("Dry-run: build %s force=%s", source, force)
        return
    if source == "parsed":
        build_parsed_store(config, force=force)
    else:
        build_preprocessed_store(config, force=force)
    resolved = _validated_source(source, config, was_built=True)
    print_kv_table("Store Summary", data_source_summary(resolved))


def _validate(source: str) -> None:
    result = validate_store(_store_dir(source, config), source, config)
    if not result.is_valid:
        for error in result.errors:
            logger.error("%s", error)
        raise SystemExit(1)
    print_kv_table(
        "Store Validation",
        {
            "Source": source,
            "Manifest hash": result.manifest_hash or "n/a",
            "Valid": "yes",
            "Stale": "yes" if result.is_stale else "no",
        },
    )


def _info(source: str) -> None:
    resolved = resolve_data_source(config, source_override=source)
    print_kv_table("Store Summary", data_source_summary(resolved))


def _clean(source: str, dry_run: bool) -> None:
    target = _store_dir(source, config)
    if dry_run:
        logger.info("Dry-run: clean %s", target)
        return
    if target.exists():
        shutil.rmtree(target)
        logger.info("Removed %s", target)
    else:
        logger.info("Store does not exist: %s", target)


def main() -> None:
    setup_logging()
    args = _parse_args()
    if args.command == "build":
        _build(args.source, args.force, args.dry_run)
        return
    if args.command == "validate":
        _validate(args.source)
        return
    if args.command == "info":
        _info(args.source)
        return
    if args.command == "clean":
        _clean(args.source, args.dry_run)
        return
    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
