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
