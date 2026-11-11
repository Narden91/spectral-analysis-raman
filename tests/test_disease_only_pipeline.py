"""Check phase dependencies without reading campaign data."""

from copy import deepcopy

import pandas as pd
import pytest

import src.analysis.disease_only_pipeline as pipeline

PHASE_FUNCTIONS = {
    "clinical_correlation_matrix": "run_clinical_correlation_matrix",
    "longitudinal_moca": "run_longitudinal_moca",
    "disease_stratification": "run_disease_stratification",
    "candidate_regions": "run_candidate_region_detector",
    "covariate_models": "run_covariate_region_models",
    "pls_regression": "run_pls_regression",
    "splsda": "run_splsda_classifier",
    "biochemical_ratios": "run_biochemical_ratios",
    "preregistered_panel": "run_preregistered_panel",
    "confirmatory_panel": "run_confirmatory_panel_statistics",
    "endpoint_consistency": "run_endpoint_consistency",
    "robustness_ranking": "run_robustness_ranking",
    "integrated_candidates": "run_integrated_candidates",
    "disease_subgroups": "run_disease_subgroups",
    "panel_cluster_summary": "run_panel_cluster_summary",
    "minimal_panel": "run_minimal_panel",
    "panel_report": "run_panel_report_section",
    "publication_report": "run_candidate_region_report",
    "spectral_shape": "run_spectral_shape",
    "nmf_decomposition": "run_nmf_decomposition",
    "spectral_descriptors": "run_spectral_descriptors",
    "discriminant_spectra": "run_discriminant_spectra",
    "feature_coupling": "run_feature_coupling",
    "journal_figures": "run_journal_figures",
}


@pytest.fixture
def phase_calls(monkeypatch, tmp_path):
    calls = []

    def stub(phase):
        def run(**kwargs):
            calls.append((phase, kwargs))
            return {
                key: tmp_path / f"{phase}_{key}.csv"
                for key in (
                    "candidate_regions_cluster_corrected", "pls_feature_effects",
                    "pls_regression_performance", "pls_l1da_selected_bands", "panel_feature_matrix",
                    "panel_continuous_associations", "panel_contrast_tests", "primary_anchor_result",
                    "panel_evidence_flags", "integrated_candidate_regions", "disease_subgroup_assignments",
                    "minimal_candidate_panel", "panel_report_section",
                )
            }
        return run

    for phase, function in PHASE_FUNCTIONS.items():
        monkeypatch.setattr(pipeline, function, stub(phase))
    monkeypatch.setattr(pipeline, "get_run_metadata_dir", lambda: tmp_path)
    monkeypatch.setattr(pipeline, "get_confirmatory_panel_dir", lambda: tmp_path / "panel")
    monkeypatch.setattr(pipeline, "_collect_cross_phase_multiplicity", lambda outputs: pd.DataFrame())
    return calls


def test_phase_dependencies_and_config_are_preserved(phase_calls, tmp_path):
    cfg = {"analysis": {"minimal_panel": {"min_subjects": 17}}}
    original_cfg = deepcopy(cfg)
    feature_matrix = tmp_path / "features.parquet"
    outputs = pipeline.run_disease_only_cross_region_pipeline(feature_matrix_path=feature_matrix, cfg=cfg)
    by_phase = {phase: kwargs for phase, kwargs in phase_calls}

    assert by_phase["covariate_models"]["candidate_regions_path"] == outputs["candidate_regions"]["candidate_regions_cluster_corrected"]
    assert by_phase["splsda"]["pls_outputs"] == outputs["pls_regression"]
    assert by_phase["endpoint_consistency"]["associations_path"] == outputs["confirmatory_panel"]["panel_continuous_associations"]
    assert by_phase["disease_subgroups"]["integrated_candidates_path"] == outputs["integrated_candidates"]["integrated_candidate_regions"]
    assert by_phase["panel_cluster_summary"]["subgroup_assignments_path"] == outputs["disease_subgroups"]["disease_subgroup_assignments"]
    assert by_phase["feature_coupling"]["panel_matrix_path"] == outputs["preregistered_panel"]["panel_feature_matrix"]
    assert by_phase["publication_report"]["panel_report_section_path"] == outputs["panel_report"]["panel_report_section"]

    names = [phase for phase, _ in phase_calls]
    for source, consumer in [
        ("candidate_regions", "covariate_models"),
        ("pls_regression", "splsda"),
        ("preregistered_panel", "confirmatory_panel"),
        ("confirmatory_panel", "endpoint_consistency"),
        ("integrated_candidates", "disease_subgroups"),
        ("disease_subgroups", "panel_cluster_summary"),
        ("panel_report", "publication_report"),
    ]:
        assert names.index(source) < names.index(consumer)

    minimal_calls = [kwargs for phase, kwargs in phase_calls if phase == "minimal_panel"]
    assert len(minimal_calls) == 2
    assert minimal_calls[0]["cfg"] is cfg
    assert minimal_calls[1]["cfg"]["analysis"]["minimal_panel"]["feature_source"] == "preregistered_panel"
    assert minimal_calls[1]["cfg"]["analysis"]["minimal_panel"]["min_subjects"] == 17
    assert cfg == original_cfg

    timings = pd.read_csv(tmp_path / "disease_only_cross_region_phase_timing.csv")
    assert timings["phase_key"].tolist() == list(outputs)
    assert set(timings["status"]) == {"ok"}
    assert (timings["elapsed_seconds"] >= 0).all()


def test_disabled_panel_skips_dependent_phases(phase_calls, tmp_path):
    cfg = {"analysis": {"preregistered_panel": {"enabled": False}}}
    outputs = pipeline.run_disease_only_cross_region_pipeline(feature_matrix_path=None, cfg=cfg)
    blocked = {
        "preregistered_panel", "confirmatory_panel", "endpoint_consistency", "robustness_ranking",
        "panel_cluster_summary", "panel_minimal_model", "panel_report", "feature_coupling",
    }
    assert blocked.isdisjoint(outputs)
    by_phase = {phase: kwargs for phase, kwargs in phase_calls}
    assert by_phase["journal_figures"]["panel_matrix_path"] is None
    assert by_phase["publication_report"]["panel_report_section_path"] is None
    assert len([phase for phase, _ in phase_calls if phase == "minimal_panel"]) == 1


def test_all_disabled_phases_produce_no_phase_artifacts(phase_calls, tmp_path):
    cfg = {"analysis": {key: {"enabled": False} for key in PHASE_FUNCTIONS}}
    assert pipeline.run_disease_only_cross_region_pipeline(feature_matrix_path=None, cfg=cfg) == {}
    assert phase_calls == []
    assert list(tmp_path.iterdir()) == []
