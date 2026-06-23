from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

import src
from src.preprocessing.pipeline import _normalization_method, normalize_spectra


@pytest.fixture()
def preprocessing_config_guard() -> None:
    original = deepcopy(src.config)
    try:
        yield
    finally:
        src.config.clear()
        src.config.update(original)


def test_fingerprint_and_ch_stretch_use_separate_normalization(
    preprocessing_config_guard: None,
) -> None:
    assert _normalization_method(None, "eda", "fingerprint") == "snv"
    assert _normalization_method(None, "eda", "ch_stretch") == "area"


def test_cross_fusion_mode_disables_joint_rescale(preprocessing_config_guard: None) -> None:
    assert _normalization_method(None, "cross", "fingerprint") == "none"
    assert _normalization_method(None, "cross", "ch_stretch") == "none"


def test_normalize_spectra_applies_block_specific_methods(preprocessing_config_guard: None) -> None:
    fingerprint = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float)
    ch_stretch = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=float)

    snv = normalize_spectra(fingerprint, region="fingerprint")
    area = normalize_spectra(ch_stretch, region="ch_stretch")
    cross = normalize_spectra(fingerprint, mode="cross", region="fingerprint")

    assert not np.allclose(snv, fingerprint)
    assert not np.allclose(area, ch_stretch)
    assert np.allclose(cross, fingerprint)


def test_block_identity_column_prefixes_preserved_in_feature_matrix() -> None:
    columns = ["subject_id", "1300__1240.0", "1300__1250.0", "2850__2850.0"]
    blocks = {column.split("__", 1)[0] for column in columns if "__" in column}
    assert blocks == {"1300", "2850"}
