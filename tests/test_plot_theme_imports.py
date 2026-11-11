from __future__ import annotations

import importlib

import matplotlib
import pytest


@pytest.mark.parametrize("module_name", [
    "src.visualization.eda_plots",
    "src.visualization.spectral_summary_plots",
    "src.analysis.control_vs_disease_ml_report",
])
def test_plot_module_import_preserves_callers_theme(module_name: str) -> None:
    with matplotlib.rc_context({"axes.facecolor": "#123456", "font.size": 8.25}):
        importlib.reload(importlib.import_module(module_name))
        assert matplotlib.rcParams["axes.facecolor"] == "#123456"
        assert matplotlib.rcParams["font.size"] == 8.25
