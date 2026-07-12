from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.analysis._whole_spectra_runner as runner


@pytest.fixture()
def whole_spectra_output_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    base = tmp_path / "whole"
    monkeypatch.setattr(
        runner,
        "get_whole_spectra_phase_dir",
        lambda key: (base / key).mkdir(parents=True, exist_ok=True) or (base / key),
    )
    return base


def test_disease_only_whole_spectra_skips_ratio_summary(
    whole_spectra_output_guard: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(runner.config, "analysis", {"mode": "disease_only"})
    monkeypatch.setitem(
        runner.config,
        "whole_spectra",
        {
            "bands": {
                "beta_sheet_980": [995, 1015],
                "amide_i_1657": [1630, 1680],
                "amide_iii_1258": [1240, 1280],
                "dna_po2_sym_1090": [1080, 1100],
                "ch2_sym_stretch_2853": [2840, 2865],
                "ch3_asym_stretch_2937": [2920, 2945],
            },
            "ratio_pairs": [
                ["amide_i_1657", "ch2_sym_stretch_2853"],
                ["amide_iii_1258", "amide_i_1657"],
            ],
            "correlation": {"method": "spearman", "n_bootstrap": 0},
        },
    )
    monkeypatch.setattr(runner, "whole_spectra_split_index", lambda _: 5)

    wavenumbers = np.array(
        [1000.0, 1090.0, 1260.0, 1655.0, 1680.0, 2840.0, 2850.0, 2865.0, 2930.0, 2945.0],
        dtype=float,
    )
    groups = {
        "female_disease": np.array(
            [
                [1.0, 1.1, 1.2, 2.0, 1.95, 0.79, 0.8, 0.82, 0.7, 0.71],
                [1.1, 1.0, 1.3, 2.1, 2.0, 0.89, 0.9, 0.92, 0.8, 0.81],
                [1.2, 1.2, 1.1, 1.9, 1.85, 0.84, 0.85, 0.87, 0.75, 0.76],
            ],
            dtype=float,
        ),
        "male_disease": np.array(
            [
                [0.9, 0.95, 1.0, 1.8, 1.78, 0.69, 0.7, 0.72, 0.65, 0.66],
                [1.0, 1.0, 1.05, 1.85, 1.82, 0.71, 0.72, 0.74, 0.66, 0.67],
                [1.05, 1.02, 1.08, 1.9, 1.88, 0.73, 0.74, 0.76, 0.68, 0.69],
            ],
            dtype=float,
        ),
    }
    group_ids = {
        "female_disease": ["FD1", "FD2", "FD3"],
        "male_disease": ["MD1", "MD2", "MD3"],
    }
    df_index = pd.DataFrame(
        {
            "subject_id": ["FD1", "FD2", "FD3", "MD1", "MD2", "MD3"],
            "is_control": [False, False, False, False, False, False],
        }
    )

    result = runner._run_whole_spectra_eda(
        df_index=df_index,
        groups=groups,
        wavenumbers=wavenumbers,
        group_ids=group_ids,
        range_title="whole spectra",
    )

    drivers_dir = whole_spectra_output_guard / "global_drivers"
    assert (drivers_dir / "cross_region_ratios.csv").exists()
    assert (drivers_dir / "ratio_boxplots.png").exists()
    assert not (drivers_dir / "ratio_group_tests.csv").exists()
    assert not (drivers_dir / "ratio_summary.png").exists()
    assert result["ratio_tests"].empty
