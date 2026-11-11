from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

import src
from src.preprocessing.pipeline import (
    _compute_qc_metrics,
    _detect_cosmic_spikes,
    interpolate_spectra,
    normalize_spectra,
    run_preprocessing,
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

    metrics = _compute_qc_metrics(spectra)
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


@pytest.mark.parametrize("spacing", [0.3, 0.6, 1.0])
def test_generated_interpolation_grid_stays_inside_native_axis(spacing: float) -> None:
    axis = np.array([100.0, 101.0, 102.0])
    spectra = 2.0 * axis[None, :] + 1.0
    aligned, target = interpolate_spectra(spectra, axis, target_spacing=spacing, method="linear")
    assert target[0] >= axis[0]
    assert target[-1] <= axis[-1]
    np.testing.assert_allclose(np.diff(target), spacing)
    np.testing.assert_allclose(aligned, 2.0 * target[None, :] + 1.0)


@pytest.mark.parametrize("spacing", [0.0, -1.0, np.nan, np.inf])
def test_generated_interpolation_grid_rejects_invalid_spacing(spacing: float) -> None:
    axis = np.array([100.0, 101.0, 102.0])
    with pytest.raises(ValueError, match="target_spacing must be finite and positive"):
        interpolate_spectra(axis[None, :], axis, target_spacing=spacing)


def test_preprocessing_uses_supplied_config_without_changing_defaults() -> None:
    original = deepcopy(src.config)
    override = deepcopy(original)
    override["preprocessing"]["ranges"]["fingerprint"]["normalization"] = {"method": "none"}
    override["preprocessing"]["dead_pixel"]["min_spectral_std"] = 1e6
    spectra = np.tile(np.linspace(1.0, 2.0, 31), (2, 1))
    axis = np.arange(1000.0, 1031.0)
    options = dict(
        region="fingerprint", suppress_cosmic=False, correct_bl=False, smooth=False,
        filter_quality=False, trim=0, cfg=override,
    )
    kept = run_preprocessing(spectra, axis, screen_dead=False, **options)
    np.testing.assert_array_equal(kept.spectra, spectra)
    assert kept.metadata["normalization_method"] == "none"
    rejected = run_preprocessing(spectra, axis, screen_dead=True, **options)
    assert not rejected.retained_mask.any()
    assert src.config == original


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
    preprocessing_config_guard: None,
) -> None:
    spectra = np.tile(np.linspace(1.0, 2.0, 31), (2, 1))
    result = run_preprocessing(
        spectra,
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
    np.testing.assert_array_equal(result.spectra, spectra)
    assert result.metadata["isolated_spike_repair_enabled"] is False
    assert not result.qc_metrics["n_spike_repaired_channels"].any()


def test_area_normalization_uses_supplied_wavenumber_axis() -> None:
    wavenumbers = np.array([0.0, 1.0, 3.0])
    normalized = normalize_spectra(
        np.array([[1.0, 1.0, 1.0]]),
        method="area",
        wavenumbers=wavenumbers,
    )
    assert np.isclose(np.trapezoid(np.abs(normalized[0]), x=wavenumbers), 1.0)


@pytest.mark.parametrize("parameter,value", [("lam", -1), ("ratio", 0), ("max_iter", 0)])
def test_nested_arpls_validation_checks_actual_parameters(parameter, value):
    from src.preprocessing.pipeline import _validate_preprocessing_config

    arpls = {"lam": 1000.0, "ratio": 0.001, "max_iter": 20, parameter: value}
    cfg = {"preprocessing": {"normalization": {"method": "none"}, "baseline": {"method": "arpls", "arpls": arpls}}}
    with pytest.raises(ValueError, match="arPLS"):
        _validate_preprocessing_config("fingerprint", cfg)


def test_nested_arpls_values_override_unused_top_level_parameters():
    from src.preprocessing.pipeline import _validate_preprocessing_config

    cfg = {"preprocessing": {"normalization": {"method": "none"}, "baseline": {
        "method": "arpls", "lam": -1, "ratio": 0, "max_iter": 0,
        "arpls": {"lam": 1000.0, "ratio": 0.001, "max_iter": 20},
    }}}
    _validate_preprocessing_config("fingerprint", cfg)


@pytest.mark.parametrize("parameter,value", [("lam", -1), ("p", 0), ("p", 1)])
def test_als_validation_uses_als_parameters(parameter, value):
    from src.preprocessing.pipeline import _validate_preprocessing_config

    cfg = {"preprocessing": {"normalization": {"method": "none"}, "baseline": {"method": "als", "lam": 1000.0, "p": 0.01, parameter: value}}}
    with pytest.raises(ValueError, match="ALS"):
        _validate_preprocessing_config("fingerprint", cfg)


@pytest.mark.parametrize("coords", [np.ones((2, 3)), np.ones((2, 2)), np.full((2, 2), np.nan)])
def test_spatial_cosmic_fallback_uses_configured_local_window(coords):
    from src.preprocessing.pipeline import _local_cosmic_detection

    spectra = np.array([[0., 0., 0., 0., 8., 8., 8., 0., 0., 0., 0.], [0., 1., 0., 1., 0., 9., 0., 1., 0., 1., 0.]])
    cfg = {"method": "spatial", "window": 4, "z_threshold": 2.0, "max_width": 3}
    actual = _detect_cosmic_spikes(spectra, coords, cfg)
    expected = _local_cosmic_detection(spectra, z_threshold=2.0, window=4, max_width=3)
    for actual_array, expected_array in zip(actual, expected):
        np.testing.assert_array_equal(actual_array, expected_array)


def test_rejected_quality_rows_keep_computed_stage_metrics():
    cfg = deepcopy(src.config)
    cfg["preprocessing"]["quality_filter"]["spectral_quality"]["snr_threshold"] = 1e12
    cfg["preprocessing"]["cosmic_ray"]["max_corrected_fraction"] = 1.0
    spectra = np.vstack([np.linspace(1., 2., 31), np.linspace(2., 3., 31)])
    result = run_preprocessing(
        spectra, np.arange(1000., 1031.), region="fingerprint", cfg=cfg,
        screen_dead=False, suppress_cosmic=False, correct_bl=False,
        smooth=False, filter_quality=True, normalize=False, trim=0,
    )
    assert not result.retained_mask.any()
    assert np.isfinite(result.qc_metrics["post_baseline_snr"]).all()
    assert np.isfinite(result.qc_metrics["post_baseline_signal_area"]).all()
