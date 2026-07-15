from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_cli_module():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("manage_data_store_cli", root / "scripts" / "manage_data_store.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_strict_validation_flag_is_accepted_after_subcommand() -> None:
    cli = _load_cli_module()

    args = cli._parse_args(["validate", "--source", "parsed", "--strict"])

    assert args.command == "validate"
    assert args.source == "parsed"
    assert args.strict is True


def test_strict_validation_flag_remains_accepted_before_subcommand() -> None:
    cli = _load_cli_module()

    args = cli._parse_args(["--strict", "validate", "--source", "preprocessed"])

    assert args.command == "validate"
    assert args.source == "preprocessed"
    assert args.strict is True


def test_prepare_command_requires_no_source_selection() -> None:
    cli = _load_cli_module()

    args = cli._parse_args(["prepare", "--force"])

    assert args.command == "prepare"
    assert args.force is True


def test_prepare_dry_run_flag_is_accepted_after_subcommand() -> None:
    cli = _load_cli_module()

    args = cli._parse_args(["prepare", "--dry-run"])

    assert args.command == "prepare"
    assert args.dry_run is True
