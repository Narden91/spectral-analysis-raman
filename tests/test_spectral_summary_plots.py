from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src import load_config
from src.analysis.raman_bands import RAMAN_BANDS, validate_configured_bands
from src.visualization.spectral_summary_plots import (
    BAND_CATEGORY_LABELS,
    COHORT_COLORS,
    _nearest_band_label,
    _peak_indices,
    plot_cohort_mean_spectra_grid,
    plot_condition_region,
    plot_global_comparison,
    plot_region_mean_spectrum,
)
from src.visualization.theme import GROUP_COLORS


def _plot_cfg(*, band_style: bool = False) -> dict:
    return {
        "bands": {name: band.bounds for name, band in RAMAN_BANDS.items()},
        "peak_detection": {"enabled": True, "prominence": 0.05, "distance": 1, "max_peaks": 4},
        "raman_band_style": {"enabled": band_style},
    }


def test_peak_detection_excludes_negative_local_maxima() -> None:
    signal = np.array([0.0, 0.8, 0.0, -1.0, -0.1, -1.0, 0.0])

    peaks = _peak_indices(signal, {"enabled": True, "prominence": 0.05, "distance": 1, "max_peaks": 4})

    assert peaks.tolist() == [1]


def test_nearest_band_label_uses_publication_display_names() -> None:
    label = _nearest_band_label(786.0, {"cytosine_786": [773, 796]})

    assert label == "Cytosine"


def test_plotting_profile_inherits_curated_band_definitions() -> None:
    plot_config = load_config("config/plots.yaml")

    assert len(plot_config["whole_spectra"]["bands"]) == 21
    validate_configured_bands(plot_config["whole_spectra"]["bands"])
    assert {name: band.bounds for name, band in RAMAN_BANDS.items()} == {
        name: tuple(bounds) for name, bounds in plot_config["whole_spectra"]["bands"].items()
    }


def test_condition_aggregate_combines_coloured_bands_and_peak_leader_labels() -> None:
    wavenumbers = np.arange(770.0, 802.0)
    peak = np.exp(-0.5 * ((wavenumbers - 786.0) / 2.0) ** 2)

    fig, ax = plot_condition_region(
        wavenumbers,
        np.vstack((peak, peak)),
        np.vstack((2.0 * peak, 2.0 * peak)),
        "1300",
        _plot_cfg(),
    )

    assert [text.get_text() for text in fig.legends[0].get_texts()] == ["Control", "Disease"]
    annotations = [text for text in ax.texts if text.get_text() == "Cytosine"]
    assert annotations
    plt.close(fig)


def test_cohort_colors_reuse_central_theme_palette() -> None:
    assert COHORT_COLORS == {
        ("control", "female"): GROUP_COLORS["female_control"],
        ("control", "male"): GROUP_COLORS["male_control"],
        ("disease", "female"): GROUP_COLORS["female_disease"],
        ("disease", "male"): GROUP_COLORS["male_disease"],
    }


def test_global_band_legend_accumulates_categories_across_regions() -> None:
    axes = {
        "1300": np.arange(770.0, 1789.0),
        "2850": np.arange(2451.0, 3198.0),
    }
    segments = {
        region: np.exp(-0.5 * ((wavenumbers - float(wavenumbers.mean())) / 10.0) ** 2)
        for region, wavenumbers in axes.items()
    }

    fig, ax = plot_global_comparison(
        axes,
        [{"label": "Control", "color": "#0072B2", "segments": segments}],
        "Global",
        _plot_cfg(band_style=True),
    )

    assert ax.get_legend() is not None
    assert {text.get_text() for text in ax.get_legend().get_texts()} == set(BAND_CATEGORY_LABELS.values())
    plt.close(fig)


def test_qc_mean_and_cohort_grid_annotate_peaks() -> None:
    wavenumbers = np.arange(770.0, 802.0)
    peak = np.exp(-0.5 * ((wavenumbers - 786.0) / 2.0) ** 2)
    plot_cfg = _plot_cfg(band_style=True)

    fig, ax = plot_region_mean_spectrum(wavenumbers, np.vstack((peak, peak)), "1300", plot_cfg)
    assert any(text.get_text() == "Cytosine" for text in ax.texts)
    assert ax.patches
    fig.canvas.draw()
    legend = ax.get_legend()
    assert legend is not None
    renderer = fig.canvas.get_renderer()
    legend_bottom = legend.get_window_extent(renderer=renderer).y0
    annotation = next(text for text in ax.texts if text.get_text() == "Cytosine")
    assert annotation.get_window_extent(renderer=renderer).y1 < legend_bottom
    plt.close(fig)

    fig, axes = plot_cohort_mean_spectra_grid(
        {"1300": wavenumbers},
        {("control", "female"): {"1300": peak}},
        plot_cfg,
    )
    assert any(text.get_text() == "Cytosine" for ax in axes.flat for text in ax.texts)
    plt.close(fig)


def test_global_individual_traces_can_skip_fill_but_keep_mean_fill() -> None:
    wavenumbers = np.arange(770.0, 802.0)
    peak = np.exp(-0.5 * ((wavenumbers - 786.0) / 2.0) ** 2)
    fig, ax = plot_global_comparison(
        {"1300": wavenumbers},
        [
            {"color": "#0072B2", "annotate": False, "shade": False, "segments": {"1300": peak}},
            {"label": "Control mean", "color": "#0072B2", "segments": {"1300": peak}},
        ],
        "Individual global",
        _plot_cfg(),
    )

    assert len(ax.collections) == 1
    assert any(text.get_text() == "Cytosine" for text in ax.texts)
    plt.close(fig)
