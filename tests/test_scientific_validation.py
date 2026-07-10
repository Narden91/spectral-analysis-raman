from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.clinical_feature_associations import compute_spearman_feature_tests, feature_metadata
from src.statistics.cluster_permutation import apply_spearman_cluster_permutation


def _cluster_scan(seed: int, *, planted: bool) -> bool:
    rng = np.random.default_rng(seed)
    n_subjects = 40
    n_features = 20
    endpoint = rng.normal(size=n_subjects)
    spectra = rng.normal(size=(n_subjects, n_features))
    if planted:
        spectra[:, 7:12] += endpoint[:, None] * 1.5
    feature_names = [f"1300__{1000 + index:.1f}" for index in range(n_features)]
    frame = pd.DataFrame(spectra, columns=feature_names)
    frame.insert(0, "delta_moca", endpoint)
    frame.insert(0, "subject_id", [f"S{index:03d}" for index in range(n_subjects)])
    metadata = feature_metadata(feature_names)
    tests = compute_spearman_feature_tests(
        frame,
        metadata,
        target="delta_moca",
        min_subjects=20,
        fdr_alpha=0.05,
    )
    result = apply_spearman_cluster_permutation(
        tests,
        frame,
        target="delta_moca",
        n_permutations=99,
        cluster_alpha=0.05,
        random_state=seed + 10_000,
    )
    return bool(pd.to_numeric(result["cluster_p_value"], errors="coerce").lt(0.05).any())


@pytest.mark.scientific_validation
def test_cluster_permutation_null_false_positive_rate_is_bounded() -> None:
    false_positives = sum(_cluster_scan(seed, planted=False) for seed in range(20))
    assert false_positives <= 4


@pytest.mark.scientific_validation
def test_cluster_permutation_recovers_planted_contiguous_signal() -> None:
    assert _cluster_scan(42, planted=True)
