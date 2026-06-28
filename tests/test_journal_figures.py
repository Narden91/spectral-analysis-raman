"""Tests for Phase U journal figure assembly."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.journal_figures import _fig4_nmf_entropy, _plot_nmf_components


def test_plot_nmf_components_long_format():
    wn = np.linspace(1200.0, 1300.0, 5)
    rows = []
    for comp in ("component_1", "component_2"):
        for w, loading in zip(wn, np.arange(5, dtype=float)):
            rows.append({"component": comp, "wavenumber": w, "loading": loading})
    components = pd.DataFrame(rows)

    fig, ax = plt.subplots()
    assert _plot_nmf_components(ax, components) is True
    plt.close(fig)


def test_fig4_nmf_entropy_writes_from_long_format_components(tmp_path: Path):
    wn = np.linspace(1200.0, 1300.0, 4)
    components = pd.DataFrame(
        {
            "component": np.repeat(["component_1", "component_2"], len(wn)),
            "wavenumber": np.tile(wn, 2),
            "loading": np.linspace(0.1, 0.8, len(wn) * 2),
        }
    )
    entropy = pd.DataFrame(
        {
            "subject_id": ["S01", "S02", "S03"],
            "spectral_entropy": [1.0, 1.2, 0.9],
            "delta_moca": [-2.0, 1.0, 0.5],
        }
    )
    components_path = tmp_path / "nmf_components.csv"
    entropy_path = tmp_path / "spectral_entropy.csv"
    output_path = tmp_path / "fig4.png"
    components.to_csv(components_path, index=False)
    entropy.to_csv(entropy_path, index=False)

    assert _fig4_nmf_entropy(components_path, entropy_path, output_path) is True
    assert output_path.exists()
