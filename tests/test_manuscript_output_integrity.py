from __future__ import annotations

import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import scripts.generate_manuscript_cohort_table as cohort_script
import scripts.generate_manuscript_figures as figure_script
import src.analysis.journal_figures as journal_figures


def _primary_tables():
    comparison = pd.DataFrame({
        "model": ["pls_da", "random_forest"],
        "representation": ["whole", "whole"],
        "is_primary_model": [False, True],
        "permutation_observed_balanced_accuracy": [0.9, 0.4],
        "permutation_p_value_sex_stratified": [0.005, 0.5],
        "n_permutations_sex_stratified": [3, 3],
    })
    reference = pd.DataFrame({
        "model": ["random_forest"] * 3,
        "representation": ["whole"] * 3,
        "null_type": ["sex_stratified"] * 3,
        "null_balanced_accuracy": [0.2, 0.3, 0.8],
    })
    return comparison, reference


def test_recorded_reference_summary_preserves_values_without_sampling(monkeypatch):
    def fail_sampling(*args, **kwargs):
        raise AssertionError("An empirical reference summary must not be replaced by sampled values")

    monkeypatch.setattr(np.random, "default_rng", fail_sampling)
    performance = pd.DataFrame({"r2": [0.2]}, index=["delta_moca"])
    permutation = pd.DataFrame({"null_distribution_mean": [-0.4], "null_distribution_std": [0.3]}, index=["delta_moca"])
    fig, ax = plt.subplots()
    figure_script._plot_moca_reference_summary(ax, performance, permutation)
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [-0.4])
    np.testing.assert_allclose(ax.collections[-1].get_offsets()[:, 1], [0.2])
    assert {text.get_text() for text in ax.get_legend().get_texts()} == {"Reference mean +/- SD", "Observed"}
    plt.close(fig)


def test_primary_reference_matches_configured_model_and_locked_score():
    comparison, reference = _primary_tables()
    primary, draws = figure_script._recorded_primary_permutation(comparison, reference)
    assert primary["model"] == "random_forest"
    assert primary["permutation_observed_balanced_accuracy"] == 0.4
    np.testing.assert_allclose(draws, [0.2, 0.3, 0.8])


@pytest.mark.parametrize("corruption", ["p_value", "draw_count", "primary_count"])
def test_inconsistent_saved_reference_is_rejected(corruption):
    comparison, reference = _primary_tables()
    if corruption == "p_value":
        comparison.loc[1, "permutation_p_value_sex_stratified"] = 0.005
    elif corruption == "draw_count":
        comparison.loc[1, "n_permutations_sex_stratified"] = 99
    else:
        comparison["is_primary_model"] = True
    with pytest.raises(ValueError):
        figure_script._recorded_primary_permutation(comparison, reference)


def test_classifier_display_uses_recorded_primary_and_no_repeat_confidence_band(tmp_path, monkeypatch):
    comparison, reference = _primary_tables()
    comparison_path = tmp_path / "comparison.csv"
    reference_path = tmp_path / "reference.csv"
    comparison.to_csv(comparison_path, index=False)
    reference.to_csv(reference_path, index=False)
    for model in figure_script.CVD_MODEL_COLORS:
        pd.DataFrame({
            "repeat": [0] * 4 + [1] * 4,
            "y_true": [0, 0, 1, 1] * 2,
            "y_score": [0.1, 0.4, 0.6, 0.8, 0.2, 0.3, 0.7, 0.9],
        }).to_csv(tmp_path / f"{model}_whole_cv_predictions.csv", index=False)
    monkeypatch.setattr(figure_script, "ML_MODEL_COMPARISON", comparison_path)
    monkeypatch.setattr(figure_script, "ML_PERMUTATION_NULL", reference_path)
    monkeypatch.setattr(figure_script, "ML_MODEL_EVAL_DIR", tmp_path)
    saved = {}
    monkeypatch.setattr(figure_script, "_save_exact", lambda fig, stem: saved.setdefault(stem, fig))
    figure_script.make_cvd_classifier_figure()
    roc = saved["ml_classification_roc"].axes[0]
    reference_axis = saved["ml_classification_permutation_null"].axes[0]
    assert not roc.collections
    assert "descriptive" in roc.get_title()
    assert any("Observed 0.400" in text.get_text() and "0.500" in text.get_text() for text in reference_axis.texts)
    assert reference_axis.get_xlim()[1] > 0.8
    for fig in saved.values():
        plt.close(fig)


def test_cohort_missing_sex_is_not_counted_as_male():
    frame = pd.DataFrame({
        "gender": ["female", "male", pd.NA],
        "lesion_location_group": ["left", "right", pd.NA],
        **{column: [1.0, 2.0, np.nan] for column, _ in cohort_script.CONTINUOUS},
    })
    summary = cohort_script.build_characteristics(frame).set_index("characteristic")
    assert summary.loc["Female / male", "summary"] == "1 / 1"
    assert summary.loc["Female / male", "n_observed"] == 2
    assert summary.loc["Female / male", "n_missing"] == 1


def test_cohort_caption_uses_actual_count(tmp_path):
    target = tmp_path / "cohort.tex"
    characteristics = pd.DataFrame({"characteristic": ["Participants"], "summary": ["3"], "n_observed": [3], "n_missing": [0]})
    cohort_script.write_latex(characteristics, pd.DataFrame(), n_subjects=3, output_path=target)
    text = target.read_text(encoding="utf-8")
    assert "$n=3$" in text
    assert "$n=97$" not in text


def test_output_directory_is_required_before_loading_private_data(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["generate_manuscript_cohort_table.py"])
    monkeypatch.setattr(cohort_script, "load_cohort", lambda: pytest.fail("Private inputs must not be read before argument validation"))
    with pytest.raises(SystemExit) as result:
        cohort_script.main()
    assert result.value.code == 2


def test_missingness_percentages_use_only_observed_categories():
    cohort = pd.DataFrame({
        "age": [70., 60., 50.], "gender": ["female", pd.NA, "male"],
        "lesion_location_group": ["left", pd.NA, pd.NA],
        **{endpoint: [1., 2., 3.] for endpoint in cohort_script.ENDPOINTS},
    })
    cohort.loc[2, "moca_baseline"] = np.nan
    row = cohort_script.build_missingness(cohort).iloc[0]
    assert row["female_pct_observed"] == 100
    assert row["female_pct_missing"] == 0
    assert row["left_pct_observed"] == 100
    assert pd.isna(row["left_pct_missing"])


def test_association_figure_has_no_fixed_negative_conclusion(tmp_path, monkeypatch):
    associations = tmp_path / "associations.csv"
    performance = tmp_path / "performance.csv"
    pd.DataFrame({
        "analysis_name": ["delta_moca_association"] * 3, "spectral_block": ["1300"] * 3,
        "wavenumber": [1000., 1001., 1002.], "rho": [0.7, 0.8, 0.9],
        "fdr_q": [0.001] * 3, "cluster_p_value": [0.002] * 3,
    }).to_csv(associations, index=False)
    pd.DataFrame({"target": ["delta_moca"], "r2": [0.8], "permutation_p_value_holm": [0.003], "model_supported": [True]}).to_csv(performance, index=False)
    captured = []
    monkeypatch.setattr(journal_figures, "_save_and_close", lambda fig, path: captured.append(fig))
    assert journal_figures._fig6_moca_association_landscape(associations, performance, tmp_path / "fig6.png")
    fig = captured[0]
    assert "no wavenumber or model" not in fig._suptitle.get_text()
    assert any("3 / 3" in text.get_text() for text in fig.axes[0].texts)
    np.testing.assert_allclose(fig.axes[1].collections[0].get_offsets()[:, 0], [0.8])
    plt.close(fig)


def test_ratio_figure_uses_descriptive_recorded_groups(tmp_path, monkeypatch):
    coupling = tmp_path / "coupling.csv"
    ratios = tmp_path / "ratios.csv"
    pd.DataFrame().to_csv(coupling, index=False)
    pd.DataFrame({
        "ch_bulk_to_carbonyl_ratio": [1., 2.], "carotenoid_to_carbonyl_ratio": [0.2, 0.8],
        "ratio_group": ["lower_carotenoid_lower_ch_bulk_ratio", "upper_carotenoid_ratio"],
    }).to_csv(ratios, index=False)
    captured = []
    monkeypatch.setattr(journal_figures, "_save_and_close", lambda fig, path: captured.append(fig))
    assert journal_figures._fig5_coupling_regimes(coupling, ratios, tmp_path / "fig5.png")
    assert captured[0].axes[0].get_title() == "Descriptive ratio quadrants"
    assert "biochemical" not in captured[0]._suptitle.get_text()
    plt.close(captured[0])


def test_source_provenance_marks_missing_run_metadata_without_substitution(tmp_path):
    from scripts.source_provenance import write_source_provenance
    from src.data_preparation.manifest import hash_file

    source = tmp_path / "recorded.csv"
    source.write_bytes(b"observed,r2\n1,0.2\n")
    output = tmp_path / "rendered"
    path = write_source_provenance(output, source_files=[source], rendering_settings={"seed": 9})
    provenance = json.loads(path.read_text(encoding="utf-8"))
    assert provenance["source_files"][0]["sha256"] == hash_file(source)
    assert provenance["source_run_provenance_status"] == "unavailable"
    assert provenance["rendering_settings"] == {"seed": 9}
    assert not (output / "config_snapshot.yaml").exists()
    assert not (output / "data_manifest_snapshot.json").exists()


def test_source_provenance_copies_exact_archived_snapshots(tmp_path):
    from scripts.source_provenance import write_source_provenance

    source_root = tmp_path / "source_run"
    metadata = source_root / "00_run_metadata"
    metadata.mkdir(parents=True)
    snapshots = {
        "config_snapshot.yaml": b"setting: saved_value\r\n",
        "data_manifest_snapshot.json": b'{"manifest_hash": "recorded"}\r\n',
        "run_manifest.json": b'{"command": "recorded command"}\r\n',
    }
    for name, contents in snapshots.items():
        (metadata / name).write_bytes(contents)
    source = source_root / "results.csv"
    source.write_bytes(b"score\n0.2\n")
    output = tmp_path / "rendered"
    path = write_source_provenance(output, source_files=[source], rendering_settings={"setting": "render_only"})
    provenance = json.loads(path.read_text(encoding="utf-8"))
    assert provenance["source_run_provenance_status"] == "found"
    for snapshot in provenance["source_run_snapshots"][0]["snapshots"]:
        assert (output / snapshot["path"]).read_bytes() == snapshots[snapshot["name"]]
    assert provenance["rendering_settings"]["setting"] == "render_only"
