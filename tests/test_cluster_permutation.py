from __future__ import annotations

import pandas as pd

from src.analysis.clinical_feature_associations import (
    compute_binary_group_feature_tests,
    compute_spearman_feature_tests,
    feature_metadata,
)
from src.statistics.cluster_permutation import (
    apply_binary_cluster_permutation,
    apply_spearman_cluster_permutation,
    find_signed_clusters,
)


def _spearman_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(12):
        delta = float(index - 5)
        rows.append(
            {
                "subject_id": f"S{index:02d}",
                "delta_moca": delta,
                "1300__1240.0": 10.0 + delta * 2.0,
                "1300__1245.0": 11.0 + delta * 2.1,
                "1300__1250.0": 12.0 + delta * 2.2,
                "2850__2840.0": float(index % 3),
            }
        )
    return pd.DataFrame(rows)


def _binary_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(12):
        group = "decline" if index < 6 else "stable_or_improved"
        signal = 3.0 if group == "decline" else 0.5
        rows.append(
            {
                "subject_id": f"B{index:02d}",
                "delta_moca_group": group,
                "1300__1240.0": signal,
                "1300__1245.0": signal + 0.2,
                "1300__1250.0": signal + 0.4,
                "2850__2840.0": float(index % 2),
            }
        )
    return pd.DataFrame(rows)


def test_find_signed_clusters_returns_expected_interval() -> None:
    clusters = find_signed_clusters(
        effects=pd.Series([0.1, 1.5, 1.7, 1.6, -0.2]).to_numpy(dtype=float),
        p_values=pd.Series([0.5, 0.01, 0.01, 0.02, 0.4]).to_numpy(dtype=float),
        alpha=0.05,
    )

    assert len(clusters) == 1
    start, stop, mass = clusters[0]
    assert (start, stop) == (1, 3)
    assert mass > 4.7


def test_spearman_cluster_permutation_is_deterministic() -> None:
    frame = _spearman_frame()
    meta = feature_metadata([column for column in frame.columns if "__" in column])
    tests = compute_spearman_feature_tests(
        frame,
        meta,
        target="delta_moca",
        min_subjects=5,
        fdr_alpha=0.05,
    )

    first = apply_spearman_cluster_permutation(
        tests,
        frame,
        target="delta_moca",
        n_permutations=16,
        cluster_alpha=0.1,
        random_state=42,
    )
    second = apply_spearman_cluster_permutation(
        tests,
        frame,
        target="delta_moca",
        n_permutations=16,
        cluster_alpha=0.1,
        random_state=42,
    )

    assert first["cluster_p_value"].equals(second["cluster_p_value"])
    assert first["cluster_id"].notna().any()


def test_binary_cluster_permutation_tracks_subject_rows() -> None:
    frame = _binary_frame()
    meta = feature_metadata([column for column in frame.columns if "__" in column])
    tests, _counts = compute_binary_group_feature_tests(
        frame,
        meta,
        group_column="delta_moca_group",
        ordered_groups=("decline", "stable_or_improved"),
        contrast_name="decline_vs_stable_or_improved",
        min_subjects_per_group=5,
        fdr_alpha=0.05,
    )

    ordered = apply_binary_cluster_permutation(
        tests,
        frame,
        group_column="delta_moca_group",
        ordered_groups=("decline", "stable_or_improved"),
        n_permutations=16,
        cluster_alpha=0.1,
        random_state=7,
    )
    shuffled_frame = frame.sample(frac=1.0, random_state=9).reset_index(drop=True)
    shuffled = apply_binary_cluster_permutation(
        tests,
        shuffled_frame,
        group_column="delta_moca_group",
        ordered_groups=("decline", "stable_or_improved"),
        n_permutations=16,
        cluster_alpha=0.1,
        random_state=7,
    )

    assert ordered["cluster_p_value"].equals(shuffled["cluster_p_value"])
    assert ordered["cluster_id"].notna().any()


def test_spearman_cluster_maxima_serial_equals_parallel() -> None:
    frame = _spearman_frame()
    meta = feature_metadata([column for column in frame.columns if "__" in column])
    tests = compute_spearman_feature_tests(
        frame,
        meta,
        target="delta_moca",
        min_subjects=5,
        fdr_alpha=0.05,
    )

    serial = apply_spearman_cluster_permutation(
        tests,
        frame,
        target="delta_moca",
        n_permutations=16,
        cluster_alpha=0.1,
        random_state=42,
        n_jobs=1,
    )
    parallel = apply_spearman_cluster_permutation(
        tests,
        frame,
        target="delta_moca",
        n_permutations=16,
        cluster_alpha=0.1,
        random_state=42,
        n_jobs=2,
    )

    assert serial["cluster_p_value"].equals(parallel["cluster_p_value"])
    assert serial["cluster_mass"].equals(parallel["cluster_mass"])


def test_binary_cluster_maxima_serial_equals_parallel() -> None:
    frame = _binary_frame()
    meta = feature_metadata([column for column in frame.columns if "__" in column])
    tests, _counts = compute_binary_group_feature_tests(
        frame,
        meta,
        group_column="delta_moca_group",
        ordered_groups=("decline", "stable_or_improved"),
        contrast_name="decline_vs_stable_or_improved",
        min_subjects_per_group=5,
        fdr_alpha=0.05,
    )

    serial = apply_binary_cluster_permutation(
        tests,
        frame,
        group_column="delta_moca_group",
        ordered_groups=("decline", "stable_or_improved"),
        n_permutations=16,
        cluster_alpha=0.1,
        random_state=7,
        n_jobs=1,
    )
    parallel = apply_binary_cluster_permutation(
        tests,
        frame,
        group_column="delta_moca_group",
        ordered_groups=("decline", "stable_or_improved"),
        n_permutations=16,
        cluster_alpha=0.1,
        random_state=7,
        n_jobs=2,
    )

    assert serial["cluster_p_value"].equals(parallel["cluster_p_value"])
    assert serial["cluster_mass"].equals(parallel["cluster_mass"])


def test_binary_cluster_maxima_matches_scipy_with_ties_nan_and_inf() -> None:
    import numpy as np
    from scipy import stats

    from src.statistics.cluster_permutation import (
        _binary_cluster_maxima,
        rank_biserial_effect,
    )

    rng = np.random.default_rng(301)
    values = np.round(rng.uniform(0.0, 4.0, size=(24, 30)), 0)
    values[:, 4] = 1.0
    values[:, 9] = np.nan
    values[0, 12] = np.inf
    values[1, 12] = -np.inf
    labels = np.array([0] * 12 + [1] * 12, dtype=int)

    got = _binary_cluster_maxima(
        values,
        labels,
        n_permutations=20,
        cluster_alpha=0.1,
        rng=np.random.default_rng(55),
        n_jobs=1,
    )
    seeds = np.random.default_rng(55).integers(0, np.iinfo(np.int32).max, size=20, dtype=np.int64)
    ref = np.zeros(20, dtype=float)
    for i, seed in enumerate(seeds):
        order = np.random.default_rng(int(seed)).permutation(len(labels))
        left = values[order[:12]]
        right = values[order[12:]]
        res = stats.mannwhitneyu(left, right, axis=0, alternative="two-sided", method="auto")
        eff = np.asarray(rank_biserial_effect(left, right, np.asarray(res.statistic, dtype=float)), dtype=float)
        cl = find_signed_clusters(eff, np.asarray(res.pvalue, dtype=float), alpha=0.1)
        ref[i] = max((m for _, _, m in cl), default=0.0)
    np.testing.assert_allclose(got, ref, atol=1e-13)


def test_binary_cluster_maxima_small_n_exact_and_empty_group() -> None:
    import warnings

    import numpy as np
    from scipy import stats

    from src.statistics.cluster_permutation import (
        _binary_cluster_maxima,
        rank_biserial_effect,
    )

    rng = np.random.default_rng(302)
    # Mixed tied + NaN + untied small-N (3 vs 3) matches SciPy column-wise exact/asymptotic dispatch
    values = rng.normal(size=(6, 6))
    values[:, 0] = [1.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    values[2, 1] = np.nan
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=int)
    got = _binary_cluster_maxima(
        values, labels, n_permutations=12, cluster_alpha=0.2, rng=np.random.default_rng(11), n_jobs=1
    )
    seeds = np.random.default_rng(11).integers(0, np.iinfo(np.int32).max, size=12, dtype=np.int64)
    ref = np.zeros(12, dtype=float)
    for i, seed in enumerate(seeds):
        order = np.random.default_rng(int(seed)).permutation(len(labels))
        left = values[order[:3]]
        right = values[order[3:]]
        res = stats.mannwhitneyu(left, right, axis=0, alternative="two-sided", method="auto")
        eff = np.asarray(rank_biserial_effect(left, right, np.asarray(res.statistic, dtype=float)), dtype=float)
        cl = find_signed_clusters(eff, np.asarray(res.pvalue, dtype=float), alpha=0.2)
        ref[i] = max((m for _, _, m in cl), default=0.0)
    np.testing.assert_allclose(got, ref, atol=1e-13)

    # Empty group matches SciPy fallback behavior cleanly (zero cluster mass)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got_empty = _binary_cluster_maxima(
            values, np.ones(6, dtype=int), n_permutations=2, cluster_alpha=0.1, rng=np.random.default_rng(1)
        )
    np.testing.assert_array_equal(got_empty, np.zeros(2, dtype=float))


def test_cluster_maxima_serial_equals_parallel_above_threshold_and_nested_guard() -> None:
    import numpy as np
    from joblib import Parallel, delayed

    from src.statistics.cluster_permutation import (
        _binary_cluster_maxima,
        _spearman_cluster_maxima,
    )

    rng = np.random.default_rng(303)
    values = rng.normal(size=(24, 40))
    labels = np.array([0] * 12 + [1] * 12, dtype=int)
    target = rng.normal(size=24)

    bin_s = _binary_cluster_maxima(
        values, labels, n_permutations=550, cluster_alpha=0.05, rng=np.random.default_rng(77), n_jobs=1
    )
    bin_p = _binary_cluster_maxima(
        values, labels, n_permutations=550, cluster_alpha=0.05, rng=np.random.default_rng(77), n_jobs=2
    )
    np.testing.assert_array_equal(bin_s, bin_p)

    sp_s = _spearman_cluster_maxima(
        values, target, n_permutations=550, cluster_alpha=0.05, rng=np.random.default_rng(88), n_jobs=1
    )
    sp_p = _spearman_cluster_maxima(
        values, target, n_permutations=550, cluster_alpha=0.05, rng=np.random.default_rng(88), n_jobs=2
    )
    np.testing.assert_array_equal(sp_s, sp_p)

    # Nested joblib pool guard must avoid deadlock and preserve exact output
    nested = Parallel(n_jobs=2)(
        delayed(_binary_cluster_maxima)(
            values, labels, n_permutations=550, cluster_alpha=0.05, rng=np.random.default_rng(77), n_jobs=2
        )
        for _ in range(2)
    )
    np.testing.assert_array_equal(nested[0], bin_s)


def test_spearman_cluster_maxima_sub_epsilon_alpha_matches_reference() -> None:
    import warnings

    import numpy as np
    from scipy import stats

    from src.statistics.cluster_permutation import (
        _spearman_cluster_maxima,
        _spearman_cluster_maximum,
        prepare_ranked_features,
    )

    values = np.tile(np.arange(5, dtype=float)[:, None], (1, 4))
    target = np.arange(5, dtype=float)
    prepared_columns, centered_features, feature_denoms = prepare_ranked_features(values)
    ranked_target = stats.rankdata(target, method="average")
    centered_target = ranked_target - float(np.mean(ranked_target))

    for alpha in (1e-17, 1e-320):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            got = _spearman_cluster_maxima(
                values,
                target,
                n_permutations=4,
                cluster_alpha=alpha,
                rng=np.random.default_rng(19),
                n_jobs=1,
            )
        runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        assert not runtime_warnings

        seeds = np.random.default_rng(19).integers(0, np.iinfo(np.int32).max, size=4, dtype=np.int64)
        ref = np.array(
            [
                _spearman_cluster_maximum(
                    prepared_columns,
                    centered_features,
                    feature_denoms,
                    centered_target,
                    alpha,
                    int(seed),
                )
                for seed in seeds
            ],
            dtype=float,
        )
        np.testing.assert_allclose(got, ref, atol=1e-12)


def test_binary_cluster_permutation_coerces_non_string_groups_and_rejects_empty() -> None:
    import numpy as np
    import pytest

    frame_int = pd.DataFrame(
        {
            "grp": pd.array([0] * 8 + [1] * 8 + [pd.NA], dtype="Int64"),
            "1300__1000.0": np.r_[np.arange(8, dtype=float), np.arange(8, dtype=float) + 2.0, 5.0],
            "1300__1001.0": np.r_[np.arange(8, dtype=float), np.arange(8, dtype=float) + 2.0, 5.0],
        }
    )
    meta = feature_metadata(["1300__1000.0", "1300__1001.0"])
    tests, _ = compute_binary_group_feature_tests(
        frame_int,
        meta,
        group_column="grp",
        ordered_groups=("0", "1"),
        contrast_name="zero_vs_one",
        min_subjects_per_group=5,
        fdr_alpha=0.05,
    )
    frame_plain_int = frame_int.dropna(subset=["grp"]).copy()
    frame_plain_int["grp"] = frame_plain_int["grp"].astype(int)
    frame_str = frame_plain_int.copy()
    frame_str["grp"] = frame_str["grp"].astype(str)

    res_int_na = apply_binary_cluster_permutation(
        tests,
        frame_int,
        group_column="grp",
        ordered_groups=("0", "1"),
        n_permutations=40,
        cluster_alpha=0.5,
        random_state=42,
    )
    res_plain_int = apply_binary_cluster_permutation(
        tests,
        frame_plain_int,
        group_column="grp",
        ordered_groups=("0", "1"),
        n_permutations=40,
        cluster_alpha=0.5,
        random_state=42,
    )
    res_str = apply_binary_cluster_permutation(
        tests,
        frame_str,
        group_column="grp",
        ordered_groups=("0", "1"),
        n_permutations=40,
        cluster_alpha=0.5,
        random_state=42,
    )
    assert res_int_na["cluster_p_value"].equals(res_str["cluster_p_value"])
    assert res_plain_int["cluster_p_value"].equals(res_str["cluster_p_value"])
    assert res_plain_int["cluster_mass"].equals(res_str["cluster_mass"])
    assert float(res_plain_int["cluster_p_value"].iloc[0]) > 1.0 / 41.0

    frame_one_group = frame_str.loc[frame_str["grp"] == "0"].copy()
    with pytest.raises(ValueError, match="non-empty groups"):
        apply_binary_cluster_permutation(
            tests,
            frame_one_group,
            group_column="grp",
            ordered_groups=("0", "1"),
            n_permutations=10,
            cluster_alpha=0.5,
            random_state=42,
        )
