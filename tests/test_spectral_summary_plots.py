from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from src import load_config
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


def test_condition_legend_excludes_subject_counts_and_band_labels_are_unique() -> None:
    wavenumbers = np.arange(770.0, 802.0)
    peak = np.exp(-0.5 * ((wavenumbers - 786.0) / 2.0) ** 2)
    plot_cfg = {
        "peak_detection": {"enabled": True, "prominence": 0.05, "distance": 1, "max_peaks": 4},
        "bands": {"cytosine_786": [773, 796]},
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
    assert len(annotations) == 1
    assert np.isclose(annotations[0].xy[1], 2.0)
    assert annotations[0].get_position()[1] > annotations[0].xy[1]
    plt.close(fig)
