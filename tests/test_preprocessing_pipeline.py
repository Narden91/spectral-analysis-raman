from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

import src
from src.preprocessing.pipeline import (
    _detect_cosmic_spikes,
    compute_qc_metrics,
    interpolate_spectra,
    normalize_spectra,
    run_preprocessing,
    sample_baseline_qc_artifacts,
    smooth_spectra,
    spectral_quality_mask,
)


@pytest.fixture()
def preprocessing_config_guard() -> None:
    original = deepcopy(src.config)
    try:
        yield
    finally:
        src.config.clear()
        src.config.update(original)


def test_spectral_quality_uses_trailing_noise_channels(preprocessing_config_guard: None) -> None:
    src.config["preprocessing"]["quality_filter"]["spectral_quality"]["noise_channels"] = 5

    wavenumbers = np.arange(100.0, 130.0, dtype=float)
    spectra = np.zeros((2, 30), dtype=float)
    spectra[0, :5] = np.array([0.0, 10.0, 0.0, 10.0, 0.0])
    spectra[1, :5] = np.array([0.0, 8.0, 0.0, 8.0, 0.0])
    spectra[:, -5:] = 1.0
    spectra[0, 20] = 100.0
    spectra[1, 20] = 80.0

    metrics = compute_qc_metrics(spectra)
    assert float(metrics.loc[0, "snr"]) > 30.0

    mask, _ = spectral_quality_mask(
        spectra,
        snr_threshold=30.0,
        peak_window=1,
        noise_channels=5,
        min_area_percentile=10.0,
        wavenumbers=wavenumbers,
        region="fingerprint",
    )
    assert bool(mask[0])


def test_frozen_interpolation_rejects_any_extrapolation() -> None:
    axis = np.array([100.0, 101.0, 102.0])
    spectra = np.array([[1.0, 2.0, 3.0]])
    aligned, target = interpolate_spectra(spectra, axis, target_wavenumbers=np.array([100.0, 101.0, 102.0]))
    np.testing.assert_allclose(aligned, spectra)
    np.testing.assert_allclose(target, axis)
    with pytest.raises(ValueError, match="extrapolation is forbidden"):
        interpolate_spectra(spectra, axis, target_wavenumbers=np.array([99.0, 100.0, 101.0]))


def test_baseline_qc_artifacts_use_consistent_trimmed_shapes(tmp_path: Path, preprocessing_config_guard: None) -> None:
    wavenumbers = np.arange(1200.0, 1212.0, dtype=float)
    spectra = np.vstack(
        [
            np.linspace(1.0, 6.0, 12),
            np.linspace(2.0, 7.0, 12),
            np.linspace(3.0, 8.0, 12),
        ]
    )

    result = run_preprocessing(
        spectra,
        wavenumbers,
        region="fingerprint",
        screen_dead=False,
        suppress_cosmic=False,
        correct_bl=True,
        smooth=False,
        interpolate=False,
        filter_quality=False,
        normalize=False,
        trim=2,
        record_stages=True,
    )
    written = sample_baseline_qc_artifacts(result, tmp_path, "baseline_case", n_samples=1, seed=0)

    assert written
    artifact_path = Path(next(iter(written.values())))
    with np.load(artifact_path, allow_pickle=False) as data:
        expected_shape = np.asarray(data["wavenumbers"]).shape
        assert np.asarray(data["raw"]).shape == expected_shape
        assert np.asarray(data["baseline"]).shape == expected_shape
        assert np.asarray(data["corrected"]).shape == expected_shape
        assert np.asarray(data["residual"]).shape == expected_shape


def test_smooth_spectra_skips_tiny_channel_counts() -> None:
    one_channel = np.array([[1.0]], dtype=float)
    two_channels = np.array([[1.0, 2.0]], dtype=float)

    assert np.array_equal(smooth_spectra(one_channel), one_channel)
    assert np.array_equal(smooth_spectra(two_channels), two_channels)


def test_baseline_correction_preserves_signed_residuals(
    monkeypatch: pytest.MonkeyPatch,
    preprocessing_config_guard: None,
) -> None:
    corrected = np.array([[1.0, -0.5, 2.0, -0.25, 1.0]], dtype=float)

    def fake_baseline(spectra, **kwargs):
        baseline = np.asarray(spectra, dtype=float) - corrected
        return corrected.copy(), baseline

    monkeypatch.setattr("src.preprocessing.pipeline.correct_baseline", fake_baseline)
    result = run_preprocessing(
        np.full((1, 5), 10.0),
        np.arange(1000.0, 1005.0),
        region="fingerprint",
        screen_dead=False,
        suppress_cosmic=False,
        correct_bl=True,
        smooth=False,
        filter_quality=False,
        normalize=False,
        trim=0,
    )

    assert np.array_equal(result.spectra, corrected)
    assert np.isclose(float(result.qc_metrics.loc[0, "negative_fraction"]), 0.4)
    assert float(result.qc_metrics.loc[0, "negative_area_fraction"]) > 0


def test_spectral_quality_has_no_batch_relative_area_rejection() -> None:
    wavenumbers = np.arange(1000.0, 1010.0)
    spectra = np.vstack([np.linspace(0.0, 1.0, 10), np.linspace(0.0, 100.0, 10)])
    mask, stats = spectral_quality_mask(
        spectra,
        snr_threshold=-1.0,
        min_area_percentile=99.0,
        min_integrated_signal=None,
        wavenumbers=wavenumbers,
        region="fingerprint",
    )
    assert mask.all()
    assert np.isnan(stats["area_threshold"])


def test_pre_normalization_basis_uses_the_same_preprocessing_pass(
    preprocessing_config_guard: None,
) -> None:
    wavenumbers = np.linspace(1200.0, 1280.0, 81)
    spectra = np.vstack((20.0 + np.exp(-((wavenumbers - 1240.0) ** 2) / 20.0), 22.0 + np.exp(-((wavenumbers - 1240.0) ** 2) / 20.0)))
    result = run_preprocessing(
        spectra, wavenumbers, region="fingerprint", screen_dead=False,
        suppress_cosmic=False, correct_bl=False, smooth=False, filter_quality=False,
        trim=0, normalize=True, norm_for="eda", return_pre_normalization=True,
    )
    pre = result.stage_arrays["pre_normalization"]
    assert np.allclose(result.spectra, normalize_spectra(pre, mode="eda", region="fingerprint"))


def test_cosmic_detection_keeps_only_positive_narrow_events() -> None:
    spectra = np.zeros((2, 31), dtype=float)
    spectra[0, 15] = 100.0
    spectra[1, 15] = -100.0
    mask, _replacement = _detect_cosmic_spikes(
        spectra,
        pixel_coords=None,
        cr_cfg={"method": "local", "z_threshold": 8.0, "window": 2, "max_width": 3},
    )
    assert bool(mask[0, 15])
    assert not mask[1].any()


def test_excessive_cosmic_correction_rejects_and_keeps_qc_lineage(
    preprocessing_config_guard: None,
) -> None:
    spectra = np.zeros((1, 101), dtype=float)
    spectra[0, [20, 40, 60]] = 100.0
    result = run_preprocessing(
        spectra,
        np.arange(1000.0, 1101.0),
        region="fingerprint",
        screen_dead=False,
        correct_bl=False,
        smooth=False,
        filter_quality=False,
        normalize=False,
        trim=0,
    )
    assert result.spectra.shape == (0, 101)
    assert not bool(result.retained_mask[0])
    assert result.qc_flags.loc[0, "qc_reason"] == "excessive_cosmic_correction"
    assert len(result.qc_metrics) == 1
    assert float(result.qc_metrics.loc[0, "cosmic_spike_fraction"]) > 0.02


def test_post_baseline_spike_repair_is_never_applied(
    monkeypatch: pytest.MonkeyPatch,
    preprocessing_config_guard: None,
) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("post-baseline spike repair must not run")

    monkeypatch.setattr("src.preprocessing.pipeline.repair_isolated_spikes_with_mask", fail_if_called)
    run_preprocessing(
        np.tile(np.linspace(1.0, 2.0, 31), (2, 1)),
        np.arange(1000.0, 1031.0),
        region="fingerprint",
        screen_dead=False,
        suppress_cosmic=False,
        correct_bl=False,
        smooth=False,
        filter_quality=False,
        normalize=False,
        trim=0,
    )


def test_area_normalization_uses_supplied_wavenumber_axis() -> None:
    wavenumbers = np.array([0.0, 1.0, 3.0])
    normalized = normalize_spectra(
        np.array([[1.0, 1.0, 1.0]]),
        method="area",
        wavenumbers=wavenumbers,
    )
    assert np.isclose(np.trapezoid(np.abs(normalized[0]), x=wavenumbers), 1.0)
