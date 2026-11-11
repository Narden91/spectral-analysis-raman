"""Check EDA exports and subject identities using small in-memory spectra."""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import src.analysis._single_range as runner


@pytest.fixture
def prepared_cohort(monkeypatch, tmp_path):
    wave = np.linspace(734.0, 1788.0, 64)
    rng = np.random.default_rng(73)
    groups = {
        "female_disease": rng.normal(size=(6, wave.size)),
        "female_control": rng.normal(size=(6, wave.size)),
    }
    ids = {key: [f"{key}_{i}" for i in range(6)] for key in groups}
    index = pd.DataFrame({"subject_id": [sid for values in ids.values() for sid in values]})
    summary = {"group_counts": pd.DataFrame()}
    monkeypatch.setattr(
        runner, "_load_eda_cohort",
        lambda: (index, pd.DataFrame(), summary, "disease_vs_control",
                 {key: key for key in ["group_comp", "pca", "band", "variance", "robustness"]}),
    )
    monkeypatch.setattr(runner, "group_configs_for_eda_mode", lambda mode: {key: {"group": key} for key in groups})
    monkeypatch.setattr(runner, "build_group_matrix", lambda index, *, group, **kwargs: (wave, groups[group], ids[group]))
    monkeypatch.setattr(runner, "get_eda_phase_dir", lambda spectral_range, key: tmp_path / key)
    for key in ["group_comp", "pca", "band", "variance", "robustness"]:
        (tmp_path / key).mkdir()
    monkeypatch.setattr(runner, "_save_fig", lambda fig, *args: plt.close(fig))
    monkeypatch.setattr(runner, "_estimate_variance_decomposition", lambda *args: (5.0, 90.0, 5.0))
    monkeypatch.setitem(runner.config, "analysis", {"pca_n_components": 3})
    monkeypatch.setattr(runner, "_figure_flag", lambda *args, **kwargs: False)
    return wave, groups, ids, summary


def test_single_range_preserves_pca_math_and_export_identities(prepared_cohort, tmp_path):
    wave, groups, ids, summary = prepared_cohort
    result = runner._run_single_range_eda("1300")
    expected_labels = np.concatenate([np.repeat(key, len(values)) for key, values in groups.items()])
    expected_ids = [sid for key in groups for sid in ids[key]]

    assert result["groups"] is not None
    assert result["cohort_summary"] is summary
    np.testing.assert_array_equal(result["wavenumbers"], wave)
    scores = pd.read_csv(tmp_path / "pca" / "pca_scores_1300.csv")
    assert scores["group"].tolist() == expected_labels.tolist()
    assert scores["subject_id"].tolist() == expected_ids
    np.testing.assert_allclose(scores[["PC1", "PC2", "PC3"]], result["pca_result"]["scores"])

    loadings = pd.read_csv(tmp_path / "pca" / "pca_loadings_1300.csv", index_col=0)
    reconstructed = scores[["PC1", "PC2", "PC3"]].to_numpy() @ loadings.to_numpy().T
    spectra = np.vstack(list(groups.values()))
    expected = result["pca_result"]["scores"] @ result["pca_result"]["loadings"]
    np.testing.assert_allclose(reconstructed, expected, atol=1e-12)
    assert reconstructed.shape == spectra.shape

    bands = pd.read_csv(tmp_path / "band" / "band_intensities_1300.csv")
    assert bands["group"].tolist() == expected_labels.tolist()
    variance = pd.read_csv(tmp_path / "variance" / "variance_components.csv")
    np.testing.assert_allclose(variance[["measurement_noise_pct", "inter_subject_pct", "inter_group_pct"]], [[5.0, 90.0, 5.0]])


def test_whole_range_delegates_with_matching_subject_ids(prepared_cohort, monkeypatch):
    import src.analysis._whole_spectra_runner as whole_runner

    wave, groups, ids, _ = prepared_cohort
    monkeypatch.setattr(runner, "build_whole_spectra_group_matrices", lambda *args, **kwargs: (wave, groups, ids))
    expected = {"whole_result": True}

    def run_whole(index, actual_groups, actual_wave, actual_ids, title):
        assert actual_groups is groups
        assert actual_ids is ids
        np.testing.assert_array_equal(actual_wave, wave)
        return expected

    monkeypatch.setattr(whole_runner, "_run_whole_spectra_eda", run_whole)
    assert runner._run_single_range_eda("both") is expected


def test_group_axis_mismatch_is_rejected(prepared_cohort, monkeypatch):
    wave, groups, ids, _ = prepared_cohort

    def load(index, *, group, **kwargs):
        axis = wave if group == "female_disease" else wave + 0.5
        return axis, groups[group], ids[group]

    monkeypatch.setattr(runner, "build_group_matrix", load)
    with pytest.raises(ValueError, match="Wavenumber axis mismatch"):
        runner._run_single_range_eda("1300")


def test_unavailable_sampling_variance_remains_missing(monkeypatch):
    monkeypatch.setattr(runner, "load_subject_spectra", lambda *args: (np.arange(3), np.empty((0, 3))))
    groups = {"group_a": np.array([[0., 1., 2.], [1., 2., 3.]])}
    noise, biological, between = runner._estimate_variance_decomposition(
        pd.DataFrame({"subject_id": ["s1", "s2"]}), "1300", groups,
    )
    assert np.isnan(noise)
    assert np.isnan(biological)
    assert between == pytest.approx(0.)
