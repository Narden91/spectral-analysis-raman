"""Tests for the Phase 14 publication report generator."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import src.reporting.candidate_region_report as candidate_region_report
from src.reporting.candidate_region_report import (
    build_panel_report_section,
    run_candidate_region_report,
    run_panel_report_section,
)


@pytest.fixture(autouse=True)
def _isolate_cross_region_dir(monkeypatch, tmp_path: Path) -> None:
    # Without this, tests that omit paths resolve into the real Results_* tree.
    def _phase_dir(key: str) -> Path:
        path = tmp_path / "cross_region" / key
        path.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr(candidate_region_report, "get_cross_region_phase_dir", _phase_dir)


def _integrated_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_region_id": ["1300_R1", "1300_R2", "1300_R3", "2850_R1"],
            "spectral_block": ["1300", "1300", "1300", "2850"],
            "start_cm-1": [1238.0, 1240.0, 1500.0, 2838.0],
            "end_cm-1": [1252.0, 1252.0, 1510.0, 2852.0],
            "max_abs_effect": [0.8, 0.5, 0.3, 0.4],
            "integrated_abs_effect": [7.0, 4.0, 2.0, 3.0],
            "evidence_status": [
                "cluster_supported_exploratory",
                "cluster_supported_exploratory",
                "effect_only_fallback",
                "fdr_descriptive",
            ],
            "fdr_significant": [True, True, False, True],
            "cluster_permutation_significant": [True, True, False, False],
            "covariate_persisted": [True, True, False, False],
            "pls_vip_coverage": [True, False, False, False],
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
    assert "Robust" not in report_text
    assert "Exploratory" in report_text
    assert "Unsupported" in report_text
    assert "Confounded" not in report_text

    summary = pd.read_csv(outputs["candidate_region_summary"])
    assert set(summary["finding_class"]) == {"exploratory", "unsupported"}
    assert summary["n_regions"].sum() == 4

    assert any(outputs["candidate_region_figures"].glob("*.png"))


def test_build_panel_report_section_contains_four_domains_without_blocked_terms() -> None:
    ranking = pd.DataFrame(
        {
            "feature_name": ["carotenoid_to_carbonyl_ratio", "ch_bulk_to_carbonyl_ratio", "tryptophan_to_sugar_beta_sheet_ratio"],
            "effect_size_flag": [True, True, True],
            "bootstrap_ci_excludes_zero": [True, False, False],
            "permutation_flag": [True, False, False],
            "max_abs_cliffs_delta": [0.8, 0.4, 0.2],
        }
    )

    text = build_panel_report_section(ranking=ranking)

    for section in [
        "Carbonyl and CH-Stretch Features",
        "Fingerprint Features and Ratios",
            "Tentatively Assigned Raman Ratios",
        "Overall Descriptive Feature Summary",
    ]:
        assert section in text
    assert "do not identify a molecular source" in text
    blocked = ["biomarker", "diagnostic", "predictive", "causal"]
    assert not any(term in text.lower() for term in blocked)


def test_run_panel_report_section_writes_section_and_readme(tmp_path: Path) -> None:
    evidence_flags_path = tmp_path / "panel_evidence_flags.csv"
    pd.DataFrame(
        {
            "feature_name": ["carotenoid_to_carbonyl_ratio"],
            "effect_size_flag": [True],
            "bootstrap_ci_excludes_zero": [True],
            "permutation_flag": [True],
            "max_abs_cliffs_delta": [0.9],
        }
    ).to_csv(evidence_flags_path, index=False)

    outputs = run_panel_report_section(
        evidence_flags_path=evidence_flags_path,
        output_dir=tmp_path,
        cfg={"analysis": {"panel_report": {"enabled": True}}},
    )

    assert outputs["panel_report_section"].exists()
    assert outputs["panel_report_readme"].exists()


def test_run_candidate_region_report_disabled_returns_empty(tmp_path: Path) -> None:
    outputs = run_candidate_region_report(
        output_dir=tmp_path,
        cfg={"analysis": {"publication_report": {"enabled": False}}},
    )
    assert outputs == {}


def test_run_candidate_region_report_handles_missing_integrated_table(tmp_path: Path) -> None:
    """Phase 11 skips its write (no _status stub file) when it has zero candidates;
    Phase 14 must treat that absence the same as an empty table, not raise."""
    outputs = run_candidate_region_report(
        output_dir=tmp_path,
        cfg={"analysis": {"publication_report": {"enabled": True}}},
    )
    assert outputs["candidate_region_report"].exists()
    assert outputs["candidate_region_summary"].exists()
    summary = pd.read_csv(outputs["candidate_region_summary"])
    assert summary["n_regions"].sum() == 0


def test_report_discloses_marginal_screen_and_gated_minimal_panel(tmp_path: Path) -> None:
    integrated_path = tmp_path / "integrated_candidate_regions.csv"
    _integrated_candidates().to_csv(integrated_path, index=False)

    outputs = run_candidate_region_report(
        integrated_candidates_path=integrated_path,
        minimal_panel_status={"status": "not_tested_gate"},
        output_dir=tmp_path,
        cfg={"analysis": {"publication_report": {"enabled": True}}},
    )

    report_text = outputs["candidate_region_report"].read_text(encoding="utf-8")
    assert "marginal and unadjusted for age and sex" in report_text
    assert "was not tested because no cluster-supported candidate region passed" in report_text
    assert "The candidate-derived minimal panel was selected" not in report_text


def test_report_claims_minimal_panel_only_for_nonempty_artifact(tmp_path: Path) -> None:
    integrated_path = tmp_path / "integrated_candidate_regions.csv"
    panel_path = tmp_path / "minimal_candidate_panel.csv"
    _integrated_candidates().to_csv(integrated_path, index=False)
    pd.DataFrame({"feature_name": ["region__1300_R1"], "status": ["ok"]}).to_csv(panel_path, index=False)

    outputs = run_candidate_region_report(
        integrated_candidates_path=integrated_path,
        minimal_panel_path=panel_path,
        minimal_panel_status={"status": "ok"},
        output_dir=tmp_path,
        cfg={"analysis": {"publication_report": {"enabled": True}}},
    )
    report_text = outputs["candidate_region_report"].read_text(encoding="utf-8")
    assert "The candidate-derived minimal panel was selected" in report_text


def test_candidate_label_cannot_fabricate_independent_validation():
    row = pd.Series({"evidence_status": "independently_validated", "cluster_permutation_significant": True})
    with pytest.raises(ValueError, match="cannot verify a claim of independent validation"):
        candidate_region_report._classify_candidate(row)


def test_cluster_status_label_alone_is_not_selection_evidence():
    assert candidate_region_report._classify_candidate(pd.Series({"evidence_status": "cluster_supported_exploratory"})) == "unsupported"


def test_absent_panel_checks_are_missing_not_zero():
    counts = candidate_region_report._panel_flag_counts(pd.DataFrame({"feature_name": ["recorded_feature"]}))
    assert counts["n_features"].isna().all()
