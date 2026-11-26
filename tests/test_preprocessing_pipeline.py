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


@pytest.mark.parametrize("lam_val", [1e4, 1e6])
def test_correct_baseline_fast_whittaker_parity_and_edge_cases(lam_val: float) -> None:
    from pybaselines.whittaker import arpls, asls

    from src.preprocessing.pipeline import correct_baseline

    rng = np.random.default_rng(123)
    x = np.linspace(0.0, 1.0, 120)
    spectra = (
        25.0
        + 10.0 * x[None, :]
        + 80.0 * np.exp(-0.5 * ((x[None, :] - 0.5) / 0.05) ** 2)
        + rng.normal(0.0, 0.5, size=(6, 120))
    )
    corr_ar, bl_ar = correct_baseline(
        spectra, method="arpls", arpls_lam=lam_val, arpls_ratio=1e-3, arpls_max_iter=50, return_baseline=True
    )
    ref_ar = np.vstack([arpls(row, lam=lam_val, diff_order=2, max_iter=50, tol=1e-3)[0] for row in spectra])
    np.testing.assert_allclose(bl_ar, ref_ar, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(corr_ar, spectra - ref_ar, atol=1e-10, rtol=1e-10)

    corr_als, bl_als = correct_baseline(spectra, method="als", lam=lam_val, p=0.01, return_baseline=True)
    ref_als = np.vstack([asls(row, lam=lam_val, p=0.01)[0] for row in spectra])
    np.testing.assert_allclose(bl_als, ref_als, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(corr_als, spectra - ref_als, atol=1e-10, rtol=1e-10)

    # Empty, constant/zero, and short-channel W=3, 4 edge cases
    empty_corr, empty_bl = correct_baseline(np.empty((0, 10), dtype=float), method="arpls", return_baseline=True)
    assert empty_corr.shape == (0, 10) and empty_bl.shape == (0, 10)

    const_spec = np.vstack([np.zeros(40), np.full(40, 42.0)])
    const_corr, const_bl = correct_baseline(const_spec, method="arpls", arpls_lam=1e4, return_baseline=True)
    ref_const_bl = np.vstack([arpls(r, lam=1e4, diff_order=2, max_iter=100, tol=1e-3)[0] for r in const_spec])
    np.testing.assert_allclose(const_bl, ref_const_bl, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(const_corr, 0.0, atol=1e-9)

    for width in (3, 4):
        short_spec = rng.normal(10.0, 1.0, size=(3, width))
        _, short_bl_ar = correct_baseline(short_spec, method="arpls", arpls_lam=100.0, return_baseline=True)
        _, short_bl_als = correct_baseline(short_spec, method="als", lam=100.0, p=0.01, return_baseline=True)
        ref_short_ar = np.vstack([arpls(r, lam=100.0, diff_order=2, max_iter=100, tol=1e-3)[0] for r in short_spec])
        ref_short_als = np.vstack([asls(r, lam=100.0, p=0.01)[0] for r in short_spec])
        np.testing.assert_allclose(short_bl_ar, ref_short_ar, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(short_bl_als, ref_short_als, atol=1e-10, rtol=1e-10)

    with pytest.raises(ValueError):
        correct_baseline(np.ones((2, 2)), method="arpls")
    with pytest.raises(ValueError):
        correct_baseline(np.array([[1.0, np.nan, 2.0, 3.0]]), method="arpls")
    with pytest.raises(ValueError):
        correct_baseline(np.array([[1.0, np.inf, 2.0, 3.0]]), method="als")
    with pytest.raises(ValueError):
        correct_baseline(np.ones((2, 10)), method="arpls", arpls_lam=-1.0)
    with pytest.raises(ValueError):
        correct_baseline(np.ones((2, 10)), method="als", lam=100.0, p=1.0)


def test_spatial_cosmic_detection_vectorized_parity_and_width_limit() -> None:
    from src.preprocessing.pipeline import _spatial_cosmic_detection

    rng = np.random.default_rng(9)
    spectra = rng.normal(10.0, 0.2, size=(12, 40))
    spectra[1, 15] += 50.0  # narrow spike (width 1 <= 2)
    spectra[3, 20:25] += 50.0  # wide excursion (width 5 > 2)
    coords = np.column_stack([np.repeat(np.arange(3), 4), np.tile(np.arange(4), 3)]).astype(float)

    mask, repl = _spatial_cosmic_detection(spectra, coords, z_threshold=6.0, n_neighbors=4, max_width=2)
    assert bool(mask[1, 15])
    assert not mask[3].any()
    assert not mask[0].any()

    diffs = coords[:, None, :] - coords[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    nearest = np.argpartition(dists, kth=3, axis=1)[:, :4]
    ref_repl = np.vstack([np.median(spectra[nearest[i]], axis=0) for i in range(len(spectra))])
    np.testing.assert_array_equal(repl, ref_repl)


def test_compute_qc_metrics_partial_nan_fluorescence_and_linear_interp() -> None:
    from scipy.interpolate import interp1d

    spectra = np.array(
        [
            [1.0, 3.0, 5.0, 7.0, 9.0, 11.0],
            [2.0, np.nan, 6.0, 10.0, np.nan, 14.0],
        ],
        dtype=float,
    )
    metrics = _compute_qc_metrics(spectra)
    assert float(metrics.loc[0, "fluorescence_level"]) == 6.0
    assert float(metrics.loc[1, "fluorescence_level"]) == 8.0

    axis = np.array([100.0, 101.2, 102.5, 104.0, 105.5, 107.0])
    target = np.linspace(100.5, 106.5, 13)
    aligned, out_axis = interpolate_spectra(spectra[:1], axis, method="linear", target_wavenumbers=target)
    ref = interp1d(axis, spectra[:1], kind="linear", axis=1, bounds_error=True)(target)
    np.testing.assert_allclose(aligned, ref, atol=1e-12)
    np.testing.assert_array_equal(out_axis, target)
    with pytest.raises(ValueError):
        interpolate_spectra(spectra[:1], axis, method="linear", target_wavenumbers=np.array([99.99999999, 101.0]))


def test_interpolate_spectra_linear_matches_scipy_interp1d_edge_cases() -> None:
    from scipy.interpolate import interp1d

    rng = np.random.default_rng(1234)
    wn_src = np.sort(rng.uniform(700.0, 1820.0, 611))
    wn_tgt = np.arange(734.0, 1789.0, 1.0)
    spectra = rng.normal(50.0, 10.0, size=(8, 611))
    got, out_axis = interpolate_spectra(spectra, wn_src, method="linear", target_wavenumbers=wn_tgt)
    ref = interp1d(wn_src, spectra, kind="linear", axis=1, bounds_error=True)(wn_tgt)
    np.testing.assert_array_equal(out_axis, wn_tgt)
    np.testing.assert_allclose(got, ref, atol=0.0, rtol=0.0)

    knot_src = np.array([100.0, 101.2, 102.5, 104.0, 105.5, 107.0])
    knot_tgt = np.array([100.0, 101.2, 101.8, 102.5, 104.0, 105.5, 107.0])
    knot_spec = rng.normal(10.0, 3.0, size=(4, 6))
    got_knot, _ = interpolate_spectra(knot_spec, knot_src, method="linear", target_wavenumbers=knot_tgt)
    ref_knot = interp1d(knot_src, knot_spec, kind="linear", axis=1, bounds_error=True)(knot_tgt)
    np.testing.assert_allclose(got_knot, ref_knot, atol=0.0, rtol=0.0)

    nan_spec = knot_spec.copy()
    nan_spec[0, 0] = np.nan
    nan_spec[1, 1] = np.nan
    nan_spec[2, 2] = np.nan
    nan_spec[3, 5] = np.nan
    got_nan, _ = interpolate_spectra(nan_spec, knot_src, method="linear", target_wavenumbers=knot_tgt)
    ref_nan = interp1d(knot_src, nan_spec, kind="linear", axis=1, bounds_error=True)(knot_tgt)
    assert np.array_equal(got_nan, ref_nan, equal_nan=True)

    two_src = np.array([100.0, 110.0])
    two_tgt = np.array([100.0, 105.0, 110.0])
    two_spec = np.array([[1.0, 11.0], [2.0, 22.0]])
    got_two, _ = interpolate_spectra(two_spec, two_src, method="linear", target_wavenumbers=two_tgt)
    ref_two = interp1d(two_src, two_spec, kind="linear", axis=1, bounds_error=True)(two_tgt)
    np.testing.assert_allclose(got_two, ref_two, atol=0.0, rtol=0.0)

    dup_src = np.array([100.0, 102.0, 102.0, 105.0])
    dup_tgt = np.array([100.5, 103.5])
    dup_spec = np.array([[1.0, 2.0, 4.0, 7.0]])
    got_dup, _ = interpolate_spectra(dup_spec, dup_src, method="linear", target_wavenumbers=dup_tgt)
    ref_dup = interp1d(dup_src, dup_spec, kind="linear", axis=1, bounds_error=True)(dup_tgt)
    np.testing.assert_allclose(got_dup, ref_dup, atol=0.0, rtol=0.0)

    with pytest.raises(ValueError):
        interpolate_spectra(knot_spec, knot_src, method="linear", target_wavenumbers=np.array([99.999999995, 106.0]))


def test_run_preprocessing_fingerprint_linear_arpls_matches_interp1d_pybaselines() -> None:
    from pybaselines.whittaker import arpls
    from scipy.interpolate import interp1d

    from src.preprocessing.pipeline import (
        _apply_edge_trim,
        _apply_initial_qc,
        _detect_cosmic_spikes,
        _suppress_cosmic_spikes,
    )

    rng = np.random.default_rng(1234)
    wn_full = np.linspace(710.0, 1815.0, 600)
    tgt_full = np.arange(734.0, 1789.0, 1.0)
    spectra = rng.normal(50.0, 2.0, size=(20, 600))
    spectra += 300.0 * np.exp(-0.5 * ((wn_full - 1005.0) / 10.0) ** 2)
    spectra += 250.0 * np.exp(-0.5 * ((wn_full - 1450.0) / 15.0) ** 2)
    spectra += 200.0 * np.exp(-0.5 * ((wn_full - 1655.0) / 15.0) ** 2)

    cfg = deepcopy(src.config)
    res = run_preprocessing(
        spectra,
        wn_full,
        region="fingerprint",
        interpolate=True,
        target_wavenumbers=tgt_full,
        record_stages=True,
        return_pre_normalization=True,
        cfg=cfg,
    )

    metrics_df = _compute_qc_metrics(spectra, cfg=cfg)
    arr, _, _ = _apply_initial_qc(spectra.copy(), metrics_df, cfg=cfg)
    spike_mask, repl = _detect_cosmic_spikes(arr, None, cfg["preprocessing"]["cosmic_ray"])
    arr = _suppress_cosmic_spikes(arr, spike_mask, repl)
    arr, wn_trim = _apply_edge_trim(arr, wn_full, trim=None, trim_cm=None, cfg=cfg)
    arr_interp = interp1d(wn_trim, arr, kind="linear", axis=1, bounds_error=True)(tgt_full)

    bl_cfg = cfg["preprocessing"]["ranges"]["fingerprint"]["baseline"]
    lam = float(bl_cfg["arpls"]["lam"])
    max_iter = int(bl_cfg["arpls"]["max_iter"])
    tol = float(bl_cfg["arpls"]["ratio"])
    bl = np.vstack([arpls(row, lam=lam, diff_order=2, max_iter=max_iter, tol=tol)[0] for row in arr_interp])
    arr_bl = arr_interp - bl
    q_mask, _ = spectral_quality_mask(arr_bl, wavenumbers=tgt_full, region="fingerprint", cfg=cfg)
    arr_q = arr_bl[q_mask]

    sm_cfg = {**cfg["preprocessing"]["smoothing"], **cfg["preprocessing"]["ranges"]["fingerprint"].get("smoothing", {})}
    arr_sm = (
        smooth_spectra(
            arr_q,
            method=sm_cfg.get("method"),
            window_length=sm_cfg.get("window_length"),
            polyorder=sm_cfg.get("polyorder"),
        )
        if sm_cfg.get("enabled", False)
        else arr_q
    )
    arr_norm = normalize_spectra(arr_sm, region="fingerprint", wavenumbers=tgt_full, cfg=cfg)

    assert res.spectra.shape[0] == 20
    assert float(np.max(np.abs(res.stage_arrays["baseline_input"] - arr_interp))) <= 1e-12
    assert float(np.max(np.abs(res.stage_arrays["pre_normalization"] - arr_sm))) <= 1e-12
    assert float(np.max(np.abs(res.spectra - arr_norm))) <= 1e-12


def test_linear_interpolation_exact_interp1d_parity_and_adjacent_nan_knots() -> None:
    from pybaselines.whittaker import arpls
    from scipy.interpolate import interp1d

    rng = np.random.default_rng(20261126)
    wn_src = np.cumsum(rng.uniform(0.6, 1.4, size=60)) + 709.0
    spectra = rng.normal(25.0, 4.0, size=(6, wn_src.size))

    # Row 0: NaN at index 1 (adjacent to left endpoint wn_src[0])
    spectra[0, 1] = np.nan
    # Row 1: NaN at k-1 (index 14) when target hits interior knot wn_src[15] -> NaN under interp1d side="left"
    spectra[1, 14] = np.nan
    # Row 2: NaN at k+1 (index 16) when target hits interior knot wn_src[15] -> finite under interp1d side="left"
    spectra[2, 16] = np.nan
    # Row 3: NaN at index -2 (adjacent to right endpoint wn_src[-1])
    spectra[3, -2] = np.nan
    # Row 4: NaN directly on interior knot wn_src[30]
    spectra[4, 30] = np.nan

    random_interior = rng.uniform(wn_src[0], wn_src[-1], size=40)
    target = np.unique(
        np.concatenate([[wn_src[0], wn_src[15], wn_src[30], wn_src[-1]], random_interior])
    )

    aligned, out_axis = interpolate_spectra(spectra, wn_src, method="linear", target_wavenumbers=target)
    ref = interp1d(wn_src, spectra, kind="linear", axis=1, bounds_error=True)(target)
    assert np.array_equal(aligned, ref, equal_nan=True)
    assert np.array_equal(out_axis, target)
    assert np.isnan(aligned[0, 0])
    assert np.isnan(aligned[1, np.searchsorted(target, wn_src[15])])
    assert np.isfinite(aligned[2, np.searchsorted(target, wn_src[15])])
    assert np.isnan(aligned[3, -1])
    assert np.isnan(aligned[4, np.searchsorted(target, wn_src[30])])

    with pytest.raises(ValueError):
        interpolate_spectra(spectra[:, :-1], wn_src, method="linear", target_wavenumbers=target)

    # End-to-end run_preprocessing (linear interp + arPLS lam=1e6, tol=1e-3) exact parity
    cfg = deepcopy(src.config)
    cfg["preprocessing"]["interpolation"]["method"] = "linear"
    cfg["preprocessing"]["baseline"]["method"] = "arpls"
    cfg["preprocessing"]["baseline"]["arpls"] = {"lam": 1e6, "ratio": 1e-3, "max_iter": 100}
    wn_full = np.linspace(710.0, 1815.0, 600)
    tgt_full = np.arange(734.0, 1789.0, 1.0)
    raw = (
        20.0
        + 0.01 * wn_full[None, :]
        + 15.0 * np.exp(-0.5 * ((wn_full[None, :] - 1440.0) / 25.0) ** 2)
        + np.random.default_rng(1234).normal(0.0, 0.5, size=(20, 600))
    )
    res = run_preprocessing(
        raw,
        wn_full,
        region="fingerprint",
        screen_dead=False,
        suppress_cosmic=False,
        interpolate=True,
        target_wavenumbers=tgt_full,
        correct_bl=True,
        smooth=False,
        filter_quality=False,
        normalize=True,
        trim=0,
        record_stages=True,
        return_pre_normalization=True,
        cfg=cfg,
    )
    bl_in_ref = interp1d(wn_full, raw, kind="linear", axis=1, bounds_error=True)(tgt_full)
    bl_ref = np.vstack([arpls(row, lam=1e6, diff_order=2, max_iter=100, tol=1e-3)[0] for row in bl_in_ref])
    pre_ref = bl_in_ref - bl_ref
    assert np.array_equal(res.stage_arrays["baseline_input"], bl_in_ref)
    assert np.array_equal(res.stage_arrays["pre_normalization"], pre_ref)
