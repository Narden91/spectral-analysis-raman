"""Tests for Phase U journal figure assembly."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import src.analysis.journal_figures as journal_figures
from src.analysis.journal_figures import (
    _fig2_effect_forest,
    _fig4_nmf_entropy,
    _fig6_moca_association_landscape,
    _plot_nmf_components,
    run_journal_figures,
)


def test_plot_nmf_components_long_format():
    wn = np.linspace(1200.0, 1300.0, 5)
    rows = []
    for comp in ("component_1", "component_2"):
        for w, loading in zip(wn, np.arange(5, dtype=float)):
            rows.append({"component": comp, "wavenumber": w, "loading": loading})
    components = pd.DataFrame(rows)

    fig, ax = plt.subplots()
    assert _plot_nmf_components(ax, components) is True
    plt.close(fig)


def test_fig4_nmf_entropy_writes_from_long_format_components(tmp_path: Path):
    wn = np.linspace(1200.0, 1300.0, 4)
    components = pd.DataFrame(
        {
            "component": np.repeat(["component_1", "component_2"], len(wn)),
            "wavenumber": np.tile(wn, 2),
            "loading": np.linspace(0.1, 0.8, len(wn) * 2),
        }
    )
    entropy = pd.DataFrame(
        {
            "subject_id": ["S01", "S02", "S03"],
            "spectral_entropy": [1.0, 1.2, 0.9],
            "delta_moca": [-2.0, 1.0, 0.5],
        }
    )
    components_path = tmp_path / "nmf_components.csv"
    entropy_path = tmp_path / "spectral_entropy.csv"
    output_path = tmp_path / "fig4.png"
    components.to_csv(components_path, index=False)
    entropy.to_csv(entropy_path, index=False)

    assert _fig4_nmf_entropy(components_path, entropy_path, output_path) is True
    assert output_path.exists()


def test_fig2_falls_back_to_valid_trajectory_contrast(tmp_path: Path):
    contrasts_path = tmp_path / "panel_contrast_tests.csv"
    ranking_path = tmp_path / "panel_evidence_flags.csv"
    output_path = tmp_path / "fig2.png"
    pd.DataFrame(
        {
            "contrast_name": ["severity_extreme", "trajectory_extreme"],
            "feature_name": ["oxidative_index", "oxidative_index"],
            "cliffs_delta": [np.nan, 0.35],
            "bootstrap_ci_lower": [np.nan, 0.1],
            "bootstrap_ci_upper": [np.nan, 0.55],
            "status": ["insufficient_n", "ok"],
        }
    ).to_csv(contrasts_path, index=False)
    pd.DataFrame({"feature_name": ["oxidative_index"], "permutation_flag": [True]}).to_csv(
        ranking_path, index=False
    )

    assert _fig2_effect_forest(contrasts_path, ranking_path, "oxidative_index", output_path) is True
    assert output_path.exists()


def test_fig2_skips_when_no_valid_contrast(tmp_path: Path):
    contrasts_path = tmp_path / "panel_contrast_tests.csv"
    ranking_path = tmp_path / "panel_evidence_flags.csv"
    output_path = tmp_path / "fig2.png"
    output_path.write_bytes(b"stale")
    pd.DataFrame(
        {
            "contrast_name": ["severity_extreme"],
            "feature_name": ["oxidative_index"],
            "cliffs_delta": [np.nan],
            "bootstrap_ci_lower": [np.nan],
            "bootstrap_ci_upper": [np.nan],
            "status": ["insufficient_n"],
        }
    ).to_csv(contrasts_path, index=False)
    pd.DataFrame().to_csv(ranking_path, index=False)

    assert _fig2_effect_forest(contrasts_path, ranking_path, "oxidative_index", output_path) is False


def test_fig6_moca_association_landscape_writes_both_panels(tmp_path: Path):
    associations_path = tmp_path / "delta_moca_associations.csv"
    performance_path = tmp_path / "pls_regression_performance.csv"
    output_path = tmp_path / "fig6.png"

    wn_1300 = np.linspace(800.0, 1700.0, 20)
    wn_2850 = np.linspace(2500.0, 3100.0, 10)
    rows = []
    for wn in wn_1300:
        rows.append({"analysis_name": "delta_moca_association", "spectral_block": "1300",
                     "wavenumber": wn, "rho": 0.1, "fdr_q": 0.95, "cluster_p_value": 0.4})
    for wn in wn_2850:
        rows.append({"analysis_name": "delta_moca_association", "spectral_block": "2850",
                     "wavenumber": wn, "rho": -0.05, "fdr_q": 0.97, "cluster_p_value": 0.6})
    pd.DataFrame(rows).to_csv(associations_path, index=False)

    pd.DataFrame(
        {
            "target": ["moca_baseline", "moca_3m", "delta_moca"],
            "r2": [-0.33, -0.07, -0.40],
            "permutation_p_value_holm": [1.0, 0.64, 1.0],
            "model_supported": [False, False, False],
        }
    ).to_csv(performance_path, index=False)

    assert _fig6_moca_association_landscape(associations_path, performance_path, output_path) is True
    assert output_path.exists()


def test_fig6_moca_association_landscape_skips_when_no_data(tmp_path: Path):
    associations_path = tmp_path / "delta_moca_associations.csv"
    performance_path = tmp_path / "pls_regression_performance.csv"
    output_path = tmp_path / "fig6.png"
    pd.DataFrame().to_csv(associations_path, index=False)
    pd.DataFrame().to_csv(performance_path, index=False)

    assert _fig6_moca_association_landscape(associations_path, performance_path, output_path) is False


def test_journal_figure_phase_reports_partial_output(monkeypatch, tmp_path: Path):
    requested_dirs: list[str] = []

    def _phase_dir(key: str) -> Path:
        requested_dirs.append(key)
        path = tmp_path / key
        path.mkdir(parents=True, exist_ok=True)
        return path

    fig_path = _phase_dir("figures") / "fig1.png"
    monkeypatch.setattr(journal_figures, "get_confirmatory_panel_dir", lambda: _phase_dir("panel"))
    monkeypatch.setattr(journal_figures, "get_spectral_interpretability_dir", _phase_dir)
    monkeypatch.setattr(journal_figures, "get_cross_region_phase_dir", _phase_dir)
    monkeypatch.setattr(journal_figures, "get_classification_phase_dir", _phase_dir)
    monkeypatch.setattr(
        journal_figures,
        "_try_figure",
        lambda key, path, fn, *args: ("fig1", fig_path) if key == "fig1" else None,
    )

    outputs = run_journal_figures(cfg={"analysis": {"journal_figures": {"enabled": True}}})
    assert outputs["_status"]["status"] == "missing_required_data"
    assert outputs["_status"]["n_artifacts"] == 1
    assert {"figures", "discriminant", "nmf", "entropy", "coupling"}.issubset(requested_dirs)
