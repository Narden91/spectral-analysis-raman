from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analysis.disease_subgroups import _select_gap_k, run_disease_subgroups


def _cfg() -> dict:
    return {
        "analysis": {
            "disease_subgroups": {
                "enabled": True,
                "min_subjects": 8,
                "max_clusters": 3,
                "n_bootstrap": 6,
                "random_state": 42,
            }
        },
        "whole_spectra": {
            "bands": {
                "amide_iii_1258": [1238.0, 1252.0],
                "ch2_sym_stretch_2853": [2838.0, 2852.0],
            }
        },
    }


def _feature_matrix() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(16):
        high = index < 8
        rows.append(
            {
                "subject_id": f"D{index:02d}",
                "condition": "Disease",
                "is_control": False,
                "1300__1240.0": 8.0 if high else 1.0,
                "1300__1245.0": 8.5 if high else 1.2,
                "1300__1250.0": 9.0 if high else 1.4,
                "2850__2840.0": 1.0 if high else 7.0,
                "2850__2845.0": 1.2 if high else 7.5,
                "2850__2850.0": 1.4 if high else 8.0,
            }
        )
    return pd.DataFrame(rows)


def _integrated_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_region_id": ["1300_R1", "2850_R1"],
            "spectral_block": ["1300", "2850"],
            "start_cm-1": [1238.0, 2838.0],
            "end_cm-1": [1252.0, 2852.0],
            "evidence_status": [
                "cluster_supported_exploratory",
                "cluster_supported_exploratory",
            ],
        }
    )


def test_run_disease_subgroups_writes_expected_outputs(tmp_path: Path) -> None:
    matrix_path = tmp_path / "subject_feature_matrix_whole.parquet"
    integrated_path = tmp_path / "integrated_candidate_regions.csv"
    _feature_matrix().to_parquet(matrix_path, index=False)
    _integrated_candidates().to_csv(integrated_path, index=False)

    outputs = run_disease_subgroups(
        feature_matrix_path=matrix_path,
        integrated_candidates_path=integrated_path,
        output_dir=tmp_path / "12_disease_subgroups",
        cfg=_cfg(),
    )

    assignments = pd.read_csv(outputs["disease_subgroup_assignments"])
    gap = pd.read_csv(outputs["disease_subgroup_gap_statistic"])
    stability = pd.read_csv(outputs["disease_subgroup_stability"])
    features = pd.read_csv(outputs["disease_subgroup_feature_matrix"])

    assert {
        "disease_subgroup_assignments",
        "disease_subgroup_gap_statistic",
        "disease_subgroup_stability",
        "disease_subgroup_feature_matrix",
        "subgroup_mean_spectra",
    }.issubset(set(outputs))
    assert (tmp_path / "12_disease_subgroups" / "subgroup_mean_spectra.csv").exists()
    assert len(assignments) == 16
    assert assignments["subgroup_label"].nunique() >= 2
    assert set(gap["n_clusters"]) == {1, 2, 3}
    assert "reference_log_dispersion_se" in gap.columns
    assert int(stability["n_bootstrap_requested"].iloc[0]) == 6
    assert "region__1300_R1" in features.columns


def test_run_disease_subgroups_disabled_returns_empty(tmp_path: Path) -> None:
    outputs = run_disease_subgroups(
        output_dir=tmp_path,
        cfg={"analysis": {"disease_subgroups": {"enabled": False}}},
    )
    assert outputs == {}


def test_run_disease_subgroups_skips_writes_when_insufficient_subjects(tmp_path: Path) -> None:
    matrix_path = tmp_path / "subject_feature_matrix_whole.parquet"
    integrated_path = tmp_path / "integrated_candidate_regions.csv"
    # Fewer subjects than min_subjects=8 in _cfg().
    _feature_matrix().iloc[:4].to_parquet(matrix_path, index=False)
    _integrated_candidates().to_csv(integrated_path, index=False)

    out_dir = tmp_path / "12_disease_subgroups"
    outputs = run_disease_subgroups(
        feature_matrix_path=matrix_path,
        integrated_candidates_path=integrated_path,
        output_dir=out_dir,
        cfg=_cfg(),
    )

    assert set(outputs) == {"_status"}
    assert outputs["_status"]["phase_key"] == "disease_subgroups"
    assert outputs["_status"]["status"] == "insufficient_n"
    assert not out_dir.exists() or not any(out_dir.glob("*.csv"))
    assert not out_dir.exists() or not any(out_dir.glob("figures/*.png"))


def test_run_disease_subgroups_cleans_stale_figures_on_k1_fallback(tmp_path: Path) -> None:
    """If a prior run wrote per-subgroup mean-spectra figures (k>1) and a later
    rerun resolves to k=1 (no supported structure), the stale figures from the
    old k must not survive -- they'd otherwise show subgroups that were never
    actually derived under the corrected result."""
    matrix_path = tmp_path / "subject_feature_matrix_whole.parquet"
    integrated_path = tmp_path / "integrated_candidate_regions.csv"

    # Flat/no-signal feature matrix: all subjects identical -> zero variance
    # columns get dropped by _prepare_feature_matrix, forcing the
    # insufficient-features early-return. Use tiny random noise instead so the
    # matrix has usable variance but no real cluster structure, so the gap
    # statistic falls back to k=1 rather than exercising the early-return path.
    rows: list[dict[str, object]] = []
    for index in range(16):
        rows.append(
            {
                "subject_id": f"D{index:02d}",
                "condition": "Disease",
                "is_control": False,
                "1300__1240.0": 5.0 + 0.001 * (index % 2),
                "1300__1245.0": 5.0 + 0.001 * ((index + 1) % 2),
                "2850__2840.0": 5.0 + 0.001 * (index % 3),
            }
        )
    pd.DataFrame(rows).to_parquet(matrix_path, index=False)
    _integrated_candidates().to_csv(integrated_path, index=False)

    out_dir = tmp_path / "12_disease_subgroups"
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True)
    stale_1300 = figures_dir / "subgroup_mean_spectra_1300.png"
    stale_2850 = figures_dir / "subgroup_mean_spectra_2850.png"
    stale_1300.write_bytes(b"stale")
    stale_2850.write_bytes(b"stale")

    outputs = run_disease_subgroups(
        feature_matrix_path=matrix_path,
        integrated_candidates_path=integrated_path,
        output_dir=out_dir,
        cfg=_cfg(),
    )

    assert outputs["_status"]["reason"] == "gap statistic selected k=1; no disease subgroups were emitted"
    assert not stale_1300.exists()
    assert not stale_2850.exists()


def test_select_gap_k_falls_back_to_one_when_gap_never_turns_over() -> None:
    """A monotonically rising gap (no interior k satisfies the one-SE rule)
    must resolve to k=1, not to the search ceiling. Reporting the ceiling here
    would misrepresent an unresolved search as a supported subgroup count."""
    gap = pd.DataFrame(
        {
            "n_clusters": [1, 2, 3, 4],
            "gap": [0.959, 1.334, 1.539, 1.758],
            "reference_log_dispersion_se": [0.021, 0.025, 0.021, 0.022],
        }
    )
    assert _select_gap_k(gap) == 1


def test_select_gap_k_returns_interior_k_when_gap_turns_over() -> None:
    """Sanity check: a gap that plateaus/turns over at an interior k is still
    selected correctly (fall-through fix must not break the normal case)."""
    gap = pd.DataFrame(
        {
            "n_clusters": [1, 2, 3, 4],
            "gap": [0.5, 2.0, 2.05, 2.06],
            "reference_log_dispersion_se": [0.02, 0.02, 0.5, 0.02],
        }
    )
    assert _select_gap_k(gap) == 2


def test_cluster_labels_k1_fast_path() -> None:
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering

    from src.analysis.disease_subgroups import _cluster_labels

    X = np.random.default_rng(1).normal(size=(18, 6))
    got = _cluster_labels(X, 1)
    ref = AgglomerativeClustering(n_clusters=1, linkage="ward").fit_predict(X)
    np.testing.assert_array_equal(got, ref)


def test_gap_statistic_and_bootstrap_ari_serial_equals_parallel() -> None:
    import numpy as np

    from src.analysis.disease_subgroups import (
        _bootstrap_ari,
        _cluster_labels,
        _gap_statistic,
    )

    X = np.random.default_rng(10).normal(size=(24, 8))
    gap_s = _gap_statistic(X, k_values=[1, 2, 3], n_bootstrap=12, random_state=42, n_jobs=1)
    gap_p = _gap_statistic(X, k_values=[1, 2, 3], n_bootstrap=12, random_state=42, n_jobs=2)
    np.testing.assert_allclose(gap_p["gap"], gap_s["gap"], atol=1e-14)
    np.testing.assert_allclose(
        gap_p["reference_log_dispersion_se"], gap_s["reference_log_dispersion_se"], atol=1e-14
    )

    labels = _cluster_labels(X, 2)
    ari_s = _bootstrap_ari(X, labels, n_clusters=2, n_bootstrap=12, random_state=43, n_jobs=1)
    ari_p = _bootstrap_ari(X, labels, n_clusters=2, n_bootstrap=12, random_state=43, n_jobs=2)
    np.testing.assert_allclose(ari_p["mean_bootstrap_ari"], ari_s["mean_bootstrap_ari"], atol=1e-14)
    np.testing.assert_allclose(ari_p["sd_bootstrap_ari"], ari_s["sd_bootstrap_ari"], atol=1e-14)


def test_disease_subgroups_nested_pool_guard() -> None:
    import numpy as np
    from joblib import Parallel, delayed

    from src.analysis.disease_subgroups import (
        _bootstrap_ari,
        _cluster_labels,
        _gap_statistic,
    )

    X = np.random.default_rng(20).normal(size=(20, 6))
    gap_s = _gap_statistic(X, k_values=[1, 2], n_bootstrap=8, random_state=5, n_jobs=1)
    labels = _cluster_labels(X, 2)
    ari_s = _bootstrap_ari(X, labels, n_clusters=2, n_bootstrap=8, random_state=6, n_jobs=1)

    nested_gap = Parallel(n_jobs=2)(
        delayed(_gap_statistic)(X, k_values=[1, 2], n_bootstrap=8, random_state=5, n_jobs=2)
        for _ in range(2)
    )
    nested_ari = Parallel(n_jobs=2)(
        delayed(_bootstrap_ari)(X, labels, n_clusters=2, n_bootstrap=8, random_state=6, n_jobs=2)
        for _ in range(2)
    )
    np.testing.assert_allclose(nested_gap[0]["gap"], gap_s["gap"], atol=1e-14)
    np.testing.assert_allclose(nested_ari[0]["mean_bootstrap_ari"], ari_s["mean_bootstrap_ari"], atol=1e-14)


def test_gap_statistic_zero_bootstrap_handles_gt1_clusters() -> None:
    import numpy as np

    from src.analysis.disease_subgroups import _gap_statistic

    X = np.random.default_rng(31).normal(size=(16, 4))
    gap = _gap_statistic(X, k_values=[1, 2], n_bootstrap=0, random_state=1, n_jobs=1)
    assert list(gap["n_clusters"]) == [1, 2]
    assert list(gap["n_bootstrap"]) == [0, 0]
    assert np.all(np.isfinite(gap["observed_log_dispersion"]))
    assert np.all(np.isnan(gap["reference_log_dispersion_mean"]))
    assert np.all(np.isnan(gap["gap"]))
    np.testing.assert_array_equal(gap["reference_log_dispersion_sd"].to_numpy(), np.zeros(2))
    np.testing.assert_array_equal(gap["reference_log_dispersion_se"].to_numpy(), np.zeros(2))
