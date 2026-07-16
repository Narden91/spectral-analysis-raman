from __future__ import annotations

import argparse
import importlib.util
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_eda.py"
_SPEC = importlib.util.spec_from_file_location("run_eda_publication_preflight", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
run_eda_cli = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_eda_cli)


def _args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "publication_run": True,
        "max_subjects": None,
        "skip_preprocess": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_publication_preflight_accepts_full_clean_run() -> None:
    completed = subprocess.CompletedProcess(args=["git"], returncode=0, stdout="", stderr="")
    with patch.object(run_eda_cli.subprocess, "run", return_value=completed):
        run_eda_cli._validate_publication_run(_args(), ["1300", "2850", "both"])


def test_publication_preflight_rejects_shortcuts_and_dirty_worktree() -> None:
    completed = subprocess.CompletedProcess(args=["git"], returncode=0, stdout=" M src/file.py\n", stderr="")
    with patch.object(run_eda_cli.subprocess, "run", return_value=completed), pytest.raises(ValueError) as error:
        run_eda_cli._validate_publication_run(
            _args(max_subjects=8, skip_preprocess=True), ["1300"]
        )
    assert "--max-subjects" in str(error.value)
    assert "clean tracked Git worktree" in str(error.value)
