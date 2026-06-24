from __future__ import annotations

from pathlib import Path

import pytest

from src.analysis import disease_only_pipeline as pipeline


def _cfg_with_enabled_phases(**enabled: bool) -> dict:
    return {
        "analysis": {
            key: {"enabled": value}
            for key, value in enabled.items()
        }
    }


def test_disease_only_pipeline_runs_enabled_phases_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    feature_matrix_path = Path("features.parquet")

    def record(name: str, outputs: dict[str, Path] | None = None):
        def _runner(**kwargs):
            calls.append(name)
            assert kwargs["feature_matrix_path"] == feature_matrix_path
            return outputs or {}

        return _runner

    monkeypatch.setattr(
        pipeline,
        "run_clinical_correlation_matrix",
        record("clinical", {"clinical": Path("clinical.csv")}),
    )
    monkeypatch.setattr(
        pipeline,
        "run_longitudinal_moca",
        record("longitudinal", {"longitudinal": Path("longitudinal.csv")}),
    )
    monkeypatch.setattr(
        pipeline,
        "run_disease_stratification",
        record("stratification", {"stratification": Path("stratification.csv")}),
    )

    def candidate_runner(**kwargs):
        calls.append("candidate")
        assert kwargs["clinical_outputs"] == {"clinical": Path("clinical.csv")}
        assert kwargs["longitudinal_outputs"] == {"longitudinal": Path("longitudinal.csv")}
        assert kwargs["stratification_outputs"] == {"stratification": Path("stratification.csv")}
        return {"candidate_regions_cluster_corrected": Path("candidates.csv")}

    def covariate_runner(**kwargs):
        calls.append("covariate")
        assert kwargs["feature_matrix_path"] == feature_matrix_path
        assert kwargs["candidate_regions_path"] == Path("candidates.csv")
        return {}

    def pls_runner(**kwargs):
        calls.append("pls")
        assert kwargs["feature_matrix_path"] == feature_matrix_path
        return {"pls_vip_bands": Path("vip.csv")}

    def splsda_runner(**kwargs):
        calls.append("splsda")
        assert kwargs["feature_matrix_path"] == feature_matrix_path
        assert kwargs["pls_outputs"] == {"pls_vip_bands": Path("vip.csv")}
        return {"pls_l1da_selected_bands": Path("selected.csv")}

    def biochemical_runner(**kwargs):
        calls.append("biochemical")
        assert kwargs["feature_matrix_path"] == feature_matrix_path
        return {"biochemical_ratio_associations": Path("ratios.csv")}

    def integrated_runner(**kwargs):
        calls.append("integrated")
        assert kwargs["feature_matrix_path"] == feature_matrix_path
        assert kwargs["pls_vip_path"] == Path("vip.csv")
        assert kwargs["splsda_bands_path"] == Path("selected.csv")
        return {"integrated_candidate_regions": Path("integrated.csv")}

    def preregistered_runner(**kwargs):
        calls.append("preregistered")
        assert kwargs["feature_matrix_path"] == feature_matrix_path
        return {"panel_feature_matrix": Path("panel_features.csv")}

    def confirmatory_runner(**kwargs):
        calls.append("confirmatory")
        assert kwargs["feature_matrix_path"] == feature_matrix_path
        assert kwargs["panel_matrix_path"] == Path("panel_features.csv")
        return {
            "primary_anchor_result": Path("anchor.csv"),
            "panel_contrast_tests": Path("contrasts.csv"),
            "panel_continuous_associations": Path("continuous.csv"),
        }

    def endpoint_consistency_runner(**kwargs):
        calls.append("endpoint_consistency")
        assert kwargs["associations_path"] == Path("continuous.csv")
        return {"endpoint_consistency": Path("endpoint_consistency.csv")}

    def robustness_runner(**kwargs):
        calls.append("robustness")
        assert kwargs["contrasts_path"] == Path("contrasts.csv")
        assert kwargs["consistency_path"] == Path("endpoint_consistency.csv")
        return {"feature_robustness_ranking": Path("ranking.csv")}

    def subgroup_runner(**kwargs):
        calls.append("subgroups")
        assert kwargs["feature_matrix_path"] == feature_matrix_path
        assert kwargs["integrated_candidates_path"] == Path("integrated.csv")
        return {"disease_subgroup_assignments": Path("subgroups.csv")}

    def panel_cluster_runner(**kwargs):
        calls.append("panel_cluster")
        assert kwargs["feature_matrix_path"] == feature_matrix_path
        assert kwargs["panel_matrix_path"] == Path("panel_features.csv")
        assert kwargs["subgroup_assignments_path"] == Path("subgroups.csv")
        return {"panel_cluster_summary": Path("cluster_summary.csv")}

    def panel_runner(**kwargs):
        if kwargs.get("panel_matrix_path") == Path("panel_features.csv"):
            calls.append("panel_minimal")
            assert kwargs["output_dir"] is not None
            return {"panel_minimal_model_features": Path("panel_minimal.csv")}
        calls.append("panel")
        assert kwargs["feature_matrix_path"] == feature_matrix_path
        assert kwargs["integrated_candidates_path"] == Path("integrated.csv")
        return {"minimal_candidate_panel": Path("panel.csv")}

    def panel_report_runner(**kwargs):
        calls.append("panel_report")
        assert kwargs["robustness_ranking_path"] == Path("ranking.csv")
        assert kwargs["anchor_result_path"] == Path("anchor.csv")
        return {"panel_report_section": Path("panel_report_section.md")}

    def report_runner(**kwargs):
        calls.append("report")
        assert kwargs["integrated_candidates_path"] == Path("integrated.csv")
        assert kwargs["minimal_panel_path"] == Path("panel.csv")
        assert kwargs["panel_report_section_path"] == Path("panel_report_section.md")
        return {}

    monkeypatch.setattr(pipeline, "run_candidate_region_detector", candidate_runner)
    monkeypatch.setattr(pipeline, "run_covariate_region_models", covariate_runner)
    monkeypatch.setattr(pipeline, "run_pls_regression", pls_runner)
    monkeypatch.setattr(pipeline, "run_splsda_classifier", splsda_runner)
    monkeypatch.setattr(pipeline, "run_biochemical_ratios", biochemical_runner)
    monkeypatch.setattr(pipeline, "run_preregistered_panel", preregistered_runner)
    monkeypatch.setattr(pipeline, "run_confirmatory_panel_statistics", confirmatory_runner)
    monkeypatch.setattr(pipeline, "run_endpoint_consistency", endpoint_consistency_runner)
    monkeypatch.setattr(pipeline, "run_robustness_ranking", robustness_runner)
    monkeypatch.setattr(pipeline, "run_integrated_candidates", integrated_runner)
    monkeypatch.setattr(pipeline, "run_disease_subgroups", subgroup_runner)
    monkeypatch.setattr(pipeline, "run_panel_cluster_summary", panel_cluster_runner)
    monkeypatch.setattr(pipeline, "run_minimal_panel", panel_runner)
    monkeypatch.setattr(pipeline, "run_panel_report_section", panel_report_runner)
    monkeypatch.setattr(pipeline, "run_candidate_region_report", report_runner)
    monkeypatch.setattr(pipeline, "run_spectral_shape", record("spectral_shape"))
    monkeypatch.setattr(pipeline, "run_nmf_decomposition", record("nmf_decomposition"))
    monkeypatch.setattr(pipeline, "run_spectral_descriptors", record("spectral_descriptors"))
    monkeypatch.setattr(pipeline, "run_discriminant_spectra", record("discriminant_spectra"))

    outputs = pipeline.run_disease_only_cross_region_pipeline(
        feature_matrix_path=feature_matrix_path,
        cfg=_cfg_with_enabled_phases(
            clinical_correlation_matrix=True,
            longitudinal_moca=True,
            disease_stratification=True,
            candidate_regions=True,
            covariate_models=True,
            pls_regression=True,
            splsda=True,
            biochemical_ratios=True,
            preregistered_panel=True,
            endpoint_consistency=True,
            robustness_ranking=True,
            integrated_candidates=True,
            disease_subgroups=True,
            panel_cluster_summary=True,
            minimal_panel=True,
            panel_minimal_model=True,
            panel_report=True,
            publication_report=True,
            spectral_shape=True,
            nmf_decomposition=True,
            spectral_descriptors=True,
            discriminant_spectra=True,
        ),
    )

    assert calls == [
        "clinical",
        "longitudinal",
        "stratification",
        "candidate",
        "covariate",
        "pls",
        "splsda",
        "biochemical",
        "preregistered",
        "confirmatory",
        "endpoint_consistency",
        "robustness",
        "integrated",
        "subgroups",
        "panel_cluster",
        "panel",
        "panel_minimal",
        "panel_report",
        "report",
        "spectral_shape",
        "nmf_decomposition",
        "spectral_descriptors",
        "discriminant_spectra",
    ]
    assert outputs["clinical_correlation_matrix"] == {"clinical": Path("clinical.csv")}
    assert outputs["candidate_regions"] == {
        "candidate_regions_cluster_corrected": Path("candidates.csv")
    }


def test_disease_only_pipeline_respects_disabled_phases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    monkeypatch.setattr(
        pipeline,
        "run_clinical_correlation_matrix",
        lambda **kwargs: calls.append("clinical") or {},
    )
    monkeypatch.setattr(
        pipeline,
        "run_longitudinal_moca",
        lambda **kwargs: calls.append("longitudinal") or {},
    )

    outputs = pipeline.run_disease_only_cross_region_pipeline(
        feature_matrix_path=Path("features.parquet"),
        cfg=_cfg_with_enabled_phases(
            clinical_correlation_matrix=False,
            longitudinal_moca=True,
            disease_stratification=False,
            candidate_regions=False,
            covariate_models=False,
            pls_regression=False,
            splsda=False,
            biochemical_ratios=False,
            preregistered_panel=False,
            endpoint_consistency=False,
            robustness_ranking=False,
            integrated_candidates=False,
            disease_subgroups=False,
            panel_cluster_summary=False,
            minimal_panel=False,
            panel_minimal_model=False,
            panel_report=False,
            publication_report=False,
            spectral_shape=False,
            nmf_decomposition=False,
            spectral_descriptors=False,
            discriminant_spectra=False,
        ),
    )

    assert calls == ["longitudinal"]
    assert outputs == {"longitudinal_moca": {}}
