from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src.analysis.biological_pca import _band_class_map
from src.analysis.clinical_feature_associations import load_disease_feature_frame_with_bc
from src.analysis.covariate_models import _ratio_predictors
from src.analysis.disease_only_pipeline import _collect_cross_phase_multiplicity
from src.analysis.disease_stratification import BINARY_SPECS
from src.analysis.minimal_panel import _build_feature_pool
from src.analysis.preregistered_panel import build_panel_matrix
from src.analysis.raman_bands import biochemical_class_map
from src.analysis.statistics import extract_band_intensities
from src.analysis.whole_spectra import _safe_ratio


def test_cluster_identity_keeps_distinct_endpoints_and_contrasts(tmp_path) -> None:
    path = tmp_path / "clusters.csv"
    pd.DataFrame({
        "analysis_name": ["correlation", "correlation", "correlation", "binary"],
        "target": ["baseline", "baseline", "change", pd.NA],
        "contrast_name": [pd.NA, pd.NA, pd.NA, "decline_vs_stable"],
        "spectral_block": ["1300"] * 4,
        "cluster_id": ["1300_C1"] * 4,
        "cluster_p_value": [0.01, 0.01, 0.02, 0.04],
    }).to_csv(path, index=False)
    result = _collect_cross_phase_multiplicity({"clinical_correlation_matrix": {"clinical_correlation_matrix": path}})
    assert len(result) == 3
    np.testing.assert_allclose(result["family_adjusted_p"], [0.03, 0.04, 0.04])
    assert set(result["analysis_name"]) == {"correlation", "binary"}


def test_cluster_identity_rejects_conflicting_repeated_p_values(tmp_path) -> None:
    path = tmp_path / "clusters.csv"
    pd.DataFrame({"target": ["change", "change"], "cluster_id": ["C1", "C1"],
                  "cluster_p_value": [0.01, 0.02]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="inconsistent p-values"):
        _collect_cross_phase_multiplicity({"clinical_correlation_matrix": {"clinical_correlation_matrix": path}})


def test_decline_comparison_has_one_owner() -> None:
    assert not any(spec.variable == "delta_moca_group" for spec in BINARY_SPECS)


def test_companion_join_matches_subjects_and_preserves_filtered_index(tmp_path) -> None:
    normal = pd.DataFrame({"subject_id": ["C1", "D1", "D2"], "is_control": [True, False, False],
                           "1300__1000.0": [9.0, -1.0, 1.0]})
    normal_path = tmp_path / "subject_feature_matrix_whole.parquet"
    normal.to_parquet(normal_path, index=False)
    bc_path = tmp_path / "subject_feature_matrix_whole_bc.parquet"
    companion = pd.DataFrame({"subject_id": ["D2", "C1", "D1"], "1300_bc__1000.0": [20., 90., 10.]})
    companion.to_parquet(bc_path, index=False)
    joined, _, columns = load_disease_feature_frame_with_bc(feature_matrix_path=normal_path)
    assert joined.index.tolist() == [1, 2]
    assert joined["subject_id"].tolist() == ["D1", "D2"]
    assert joined[columns[0]].tolist() == [10., 20.]
    companion.iloc[:1].to_parquet(bc_path, index=False)
    with pytest.raises(ValueError, match="missing requested subjects"):
        load_disease_feature_frame_with_bc(feature_matrix_path=normal_path)


def test_biochemical_means_and_ratios_ignore_normalized_shape_values() -> None:
    frame = pd.DataFrame({"1300__1000.0": [-10., 3.], "1300__1010.0": [8., 9.],
                          "2850__2850.0": [0.2, -0.4], "2850__2860.0": [-8., 3.],
                          "1300_bc__1000.0": [2., 6.], "1300_bc__1010.0": [4., 8.],
                          "2850_bc__2850.0": [1., 2.], "2850_bc__2860.0": [2., 5.]})
    normal = [name for name in frame if name.startswith(("1300__", "2850__"))]
    bc = [name for name in frame if name.startswith(("1300_bc__", "2850_bc__"))]
    cfg = {"whole_spectra": {"bands": {"a": [1000, 1010], "b": [2850, 2860]}, "ratio_pairs": [["a", "b"]]}}
    changed = frame.copy()
    changed[normal] *= -1000
    pd.testing.assert_frame_equal(_ratio_predictors(frame, bc, cfg), _ratio_predictors(changed, bc, cfg))
    pd.testing.assert_frame_equal(_build_feature_pool(frame, normal, None, cfg),
                                  _build_feature_pool(changed, normal, None, cfg))
    np.testing.assert_allclose(_ratio_predictors(frame, bc, cfg).iloc[:, 0], [2., 2.])


def test_ratio_does_not_shift_when_unrelated_low_signal_subject_is_added() -> None:
    numerator = pd.Series([-2., 6.])
    denominator = pd.Series([1., 3.])
    before = _safe_ratio(numerator, denominator)
    after = _safe_ratio(pd.concat([numerator, pd.Series([-100.], index=[2])]),
                        pd.concat([denominator, pd.Series([10.], index=[2])]))
    pd.testing.assert_series_equal(before, after.iloc[:2])
    assert before.tolist() == [-2., 2.]


def test_fixed_panel_uses_registered_windows_and_pre_normalization_values() -> None:
    cfg = {"analysis": {"preregistered_panel": {
        "integrals": {"carotenoid_1525": ["1300", 1510, 1535], "lipid_carbonyl_1721": ["1300", 1710, 1730]},
        "indices": {"carotenoid_to_carbonyl_ratio": ["carotenoid_1525", "lipid_carbonyl_1721"]},
    }}}
    frame = pd.DataFrame({f"1300_bc__{wave:.1f}": [value] for wave, value in
                          [(1510, 2), (1520, 4), (1535, 2), (1710, 3), (1720, 6), (1730, 3)]})
    result, _ = build_panel_matrix(frame, list(frame), cfg=cfg)
    expected = np.trapezoid([2, 4, 2], [1510, 1520, 1535]) / np.trapezoid([3, 6, 3], [1710, 1720, 1730])
    assert result.loc[0, "carotenoid_to_carbonyl_ratio"] == expected
    invalid = deepcopy(cfg)
    invalid["analysis"]["preregistered_panel"]["integrals"]["lipid_carbonyl_1721"][2] = 1735
    with pytest.raises(ValueError, match="differs from the Raman registry"):
        build_panel_matrix(frame, list(frame), cfg=invalid)


def test_registry_drives_classes_and_default_band_windows() -> None:
    classes = biochemical_class_map()
    assert _band_class_map({}) == classes
    assert classes["amide_iii_1258"] == "mixed"
    waves = np.array([836., 850., 851., 860., 874.])
    spectra = np.array([[10., 1., 2., 3., 20.]])
    result = extract_band_intensities(spectra, waves)
    assert result.loc[0, "tyrosine_851"] == spectra.mean()
    custom = extract_band_intensities(spectra, waves, {"custom": 851.}, tolerance=1.)
    assert custom.loc[0, "custom"] == 1.5


def test_nmf_annotations_do_not_expand_registered_windows() -> None:
    from src.analysis.nmf_decomposition import _annotate_component

    assert _annotate_component(np.array([1.]), np.array([1734.])) == "unassigned"
    assert "carbonyl-overlapping" in _annotate_component(np.array([1.]), np.array([1725.]))
