"""Regression tests for cross-region phase-status aggregation."""

from __future__ import annotations

import pandas as pd

import src.analysis.disease_only_pipeline as pipeline
from src.analysis.phase_status import STATUS_FAILED_EXECUTION, PhaseStatus


def test_phase_status_accepts_failed_execution() -> None:
    status = PhaseStatus(phase_key="cross_phase_multiplicity", status=STATUS_FAILED_EXECUTION, reason="synthetic failure")

    assert status.as_row()["status"] == STATUS_FAILED_EXECUTION


def test_phase_status_manifest_preserves_explicit_status_with_artifacts(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "get_run_metadata_dir", lambda: tmp_path)
    outputs = {
        "splsda": {
            "pls_l1da_cv_performance": tmp_path / "performance.csv",
            "pls_l1da_permutation_results": tmp_path / "permutation.csv",
            "_status": {
                "phase_key": "splsda",
                "status": "not_tested_gate",
                "reason": "no PLS model passed permutation gate",
                "n_subjects": None,
                "n_artifacts": 2,
            },
        },
    }

    manifest_path = pipeline._write_phase_status_manifest(outputs)
    manifest = pd.read_csv(manifest_path)
    assert manifest.loc[0, "status"] == "not_tested_gate"
    assert manifest.loc[0, "n_artifacts"] == 2
