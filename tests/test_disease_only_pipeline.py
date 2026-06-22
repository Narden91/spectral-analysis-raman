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

    def integrated_runner(**kwargs):
        calls.append("integrated")
        assert kwargs["feature_matrix_path"] == feature_matrix_path
        assert kwargs["pls_vip_path"] == Path("vip.csv")
        assert kwargs["splsda_bands_path"] == Path("selected.csv")
        return {"integrated_candidate_regions": Path("integrated.csv")}

    def panel_runner(**kwargs):
        calls.append("panel")
        assert kwargs["feature_matrix_path"] == feature_matrix_path
        assert kwargs["integrated_candidates_path"] == Path("integrated.csv")
        return {"minimal_candidate_panel": Path("panel.csv")}

    def report_runner(**kwargs):
        calls.append("report")
        assert kwargs["integrated_candidates_path"] == Path("integrated.csv")
        assert kwargs["minimal_panel_path"] == Path("panel.csv")
        return {}

    monkeypatch.setattr(pipeline, "run_candidate_region_detector", candidate_runner)
    monkeypatch.setattr(pipeline, "run_covariate_region_models", covariate_runner)
    monkeypatch.setattr(pipeline, "run_pls_regression", pls_runner)
    monkeypatch.setattr(pipeline, "run_splsda_classifier", splsda_runner)
    monkeypatch.setattr(pipeline, "run_integrated_candidates", integrated_runner)
    monkeypatch.setattr(pipeline, "run_minimal_panel", panel_runner)
    monkeypatch.setattr(pipeline, "run_candidate_region_report", report_runner)

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
            integrated_candidates=True,
            minimal_panel=True,
            publication_report=True,
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
        "integrated",
        "panel",
        "report",
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
            integrated_candidates=False,
            minimal_panel=False,
            publication_report=False,
        ),
    )

    assert calls == ["longitudinal"]
    assert outputs == {"longitudinal_moca": {}}
