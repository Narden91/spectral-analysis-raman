"""Tests for the Phase 14 publication report generator."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.reporting.candidate_region_report import run_candidate_region_report


def _integrated_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_region_id": ["1300_R1", "1300_R2", "1300_R3", "2850_R1"],
            "spectral_block": ["1300", "1300", "1300", "2850"],
            "start_cm-1": [1238.0, 1240.0, 1500.0, 2838.0],
            "end_cm-1": [1252.0, 1252.0, 1510.0, 2852.0],
            "max_abs_effect": [0.8, 0.5, 0.3, 0.4],
            "integrated_abs_effect": [7.0, 4.0, 2.0, 3.0],
            "evidence_level": ["High", "Medium", "Low", "Medium"],
            "fdr_significant": [True, True, False, True],
            "cluster_permutation_significant": [True, False, False, False],
            "covariate_persisted": [True, True, False, False],
            "pls_vip_coverage": [True, False, False, False],
            "splsda_selected": [False, False, False, False],
            "n_supporting_sources": [2, 2, 1, 1],
            "biochemical_class": ["protein", "protein", "unassigned", "lipid"],
            "interpretation_note": ["note"] * 4,
        }
    )


def test_run_candidate_region_report_produces_outputs(tmp_path: Path) -> None:
    integrated_path = tmp_path / "integrated_candidate_regions.csv"
    _integrated_candidates().to_csv(integrated_path, index=False)

    outputs = run_candidate_region_report(
        integrated_candidates_path=integrated_path,
        output_dir=tmp_path,
        cfg={
            "analysis": {
                "mode": "disease_only",
                "publication_report": {"enabled": True},
            }
        },
    )

    assert "candidate_region_report" in outputs
    assert "candidate_region_summary" in outputs
    assert "candidate_region_figures" in outputs

    report_text = outputs["candidate_region_report"].read_text(encoding="utf-8")
    assert "Robust" in report_text
    assert "Exploratory" in report_text
    assert "Unstable" in report_text
    assert "Confounded" in report_text

    summary = pd.read_csv(outputs["candidate_region_summary"])
    assert set(summary["finding_class"]) == {"robust", "exploratory", "unstable", "confounded"}
    assert summary["n_regions"].sum() == 4

    assert any(outputs["candidate_region_figures"].glob("*.png"))


def test_run_candidate_region_report_disabled_returns_empty(tmp_path: Path) -> None:
    outputs = run_candidate_region_report(
        output_dir=tmp_path,
        cfg={"analysis": {"publication_report": {"enabled": False}}},
    )
    assert outputs == {}


def test_run_candidate_region_report_fails_without_integrated_table(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_candidate_region_report(
            output_dir=tmp_path,
            cfg={"analysis": {"publication_report": {"enabled": True}}},
        )
