from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from src import load_config
from src.analysis.raman_bands import RAMAN_BANDS, validate_configured_bands
from src.visualization.spectral_summary_plots import (
    _nearest_band_label,
    _peak_indices,
    plot_condition_region,
)


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


def test_condition_aggregate_uses_coloured_bands_not_peak_leader_labels() -> None:
    wavenumbers = np.arange(770.0, 802.0)
    peak = np.exp(-0.5 * ((wavenumbers - 786.0) / 2.0) ** 2)
    plot_cfg = {
        "peak_detection": {"enabled": True, "prominence": 0.05, "distance": 1, "max_peaks": 4},
        "raman_band_style": {"enabled": False},
    }

    fig, ax = plot_condition_region(
        wavenumbers,
        np.vstack((peak, peak)),
        np.vstack((2.0 * peak, 2.0 * peak)),
        "1300",
        plot_cfg,
    )

    assert [text.get_text() for text in fig.legends[0].get_texts()] == ["Control", "Disease"]
    annotations = [text for text in ax.texts if text.get_text() == "Cytosine"]
    assert not annotations
    plt.close(fig)
