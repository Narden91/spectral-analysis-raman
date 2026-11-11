from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis.robust_contrasts import (
    _permutation_p_value,
    _permutation_partial_spearman_p_value,
    _permuted_partial_spearman_exceeds,
    run_anchor_test,
    run_confirmatory_panel_statistics,
    run_panel_contrasts,
)


def _cfg() -> dict:
    return {
        "analysis": {
            "preregistered_panel": {
                "enabled": True,
                "fdr_alpha": 0.05,
                "n_permutations": 19,
                "n_bootstrap": 20,
                "random_state": 7,
                "min_subjects_per_group": 3,
                "min_subjects_continuous": 5,
                "indices": {
                    "oxidative_index": ["carotenoid_1525", "lipid_carbonyl_1721"],
                    "membrane_integrity_index": ["ch_bulk_2885", "lipid_carbonyl_1721"],
                },
                "integrals": {
                    "carotenoid_1525": ["1300", 1510, 1535],
                    "lipid_carbonyl_1721": ["1300", 1710, 1735],
                    "ch_bulk_2885": ["2850", 2850, 2950],
                },
                "primary_hypothesis": {
                    "feature": "oxidative_index",
                    "endpoint": "delta_moca",
                    "direction": "negative",
                    "test": "partial_spearman",
                    "covariates": ["age", "sex"],
                },
                "endpoints": {
                    "continuous": ["moca_baseline", "moca_3m", "delta_moca"],
                    "contrasts": [
                        {
                            "name": "severity_extreme",
                            "group_variable": "moca_severity_group_extreme",
                            "ordered_groups": ["severe", "mild"],
                        },
                        {
                            "name": "trajectory_extreme",
                            "group_variable": "delta_moca_group_extreme",
                            "ordered_groups": ["decline", "recovery"],
                        },
                    ],
                },
                "covariates": ["age", "sex"],
            }
        }
    }


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": [f"S{i:02d}" for i in range(12)],
            "condition": ["Disease"] * 12,
            "is_control": [False] * 12,
            "oxidative_index": [0.1, 0.2, 0.15, 0.25, 0.2, 0.18, 1.1, 1.2, 1.15, 1.25, 1.2, 1.18],
            "membrane_integrity_index": [1.0, 1.1, 0.9, 1.2, 1.1, 1.0, 1.8, 1.9, 2.0, 1.7, 1.8, 1.9],
            "carotenoid_1525": np.linspace(10, 21, 12),
            "lipid_carbonyl_1721": np.linspace(20, 31, 12),
            "ch_bulk_2885": np.linspace(30, 41, 12),
            "moca_baseline": [18, 17, 19, 18, 16, 19, 27, 28, 29, 27, 28, 29],
            "moca_3m": [17, 16, 18, 17, 15, 18, 29, 30, 31, 29, 30, 31],
            "delta_moca": [-3, -2, -4, -3, -2, -4, 3, 4, 2, 3, 4, 2],
            "age": [70, 71, 69, 72, 68, 70, 65, 66, 64, 67, 65, 66],
            "sex": ["F", "M", "F", "M", "F", "M", "F", "M", "F", "M", "F", "M"],
            "moca_severity_group_extreme": ["severe"] * 6 + ["mild"] * 6,
            "delta_moca_group_extreme": ["decline"] * 6 + ["recovery"] * 6,
        }
    )


def test_anchor_runs_before_panel_fdr() -> None:
    frame = _frame()

    anchor = run_anchor_test(frame, cfg=_cfg())
    contrasts = run_panel_contrasts(frame, cfg=_cfg())

    assert anchor.loc[0, "analysis_order"] == 1
    assert "fdr_q" not in anchor.columns
    assert {"p_value", "fdr_q", "perm_p_value", "cliffs_delta"}.issubset(contrasts.columns)


def test_permutation_pvalue_bounds() -> None:
    contrasts = run_panel_contrasts(_frame(), cfg=_cfg())
    ok = contrasts.loc[contrasts["status"] == "ok"]

    assert ok["perm_p_value"].between(1 / 20, 1).all()
    separated = ok.loc[ok["feature_name"] == "oxidative_index", "perm_p_value"]
    assert separated.min() < 0.2


def test_permutation_p_value_respects_permutation_max() -> None:
    rng = np.random.default_rng(41)
    # Fully overlapping distributions: no separation, so the tail probability
    # should be large (not significant) and the loop must never exceed the
    # configured ceiling, resolved early or not.
    left = rng.normal(size=15)
    right = rng.normal(size=15)

    p_value, draws_used = _permutation_p_value(
        left,
        right,
        n_permutations=200,
        random_state=42,
        n_jobs=1,
        stopping_cfg={"permutation_min": 20, "permutation_max": 40, "permutation_check_interval": 10},
    )
    assert 0.0 < p_value <= 1.0
    assert 0 < draws_used <= 40


def test_permutation_p_value_full_budget_when_unresolved() -> None:
    rng = np.random.default_rng(42)
    left = rng.normal(size=15)
    right = rng.normal(size=15)

    p_value, draws_used = _permutation_p_value(
        left,
        right,
        n_permutations=15,
        random_state=42,
        n_jobs=1,
        stopping_cfg={
            "permutation_min": 15,
            "permutation_max": 15,
            "permutation_check_interval": 15,
            "permutation_confidence": 0.999999,
        },
    )
    assert draws_used == 15
    assert np.isfinite(p_value)
    assert 0.0 < p_value <= 1.0


def test_permutation_partial_spearman_p_value_respects_permutation_max() -> None:
    rng = np.random.default_rng(43)
    n = 20
    frame = pd.DataFrame(
        {
            "feature": rng.normal(size=n),
            "endpoint": rng.normal(size=n),  # unrelated to feature: weak/no partial correlation
            "age": rng.normal(50, 10, size=n),
            "sex": rng.choice(["male", "female"], size=n),
        }
    )

    p_value, draws_used = _permutation_partial_spearman_p_value(
        frame,
        feature="feature",
        endpoint="endpoint",
        covariates=["age"],
        n_permutations=200,
        random_state=42,
        n_jobs=1,
        stopping_cfg={"permutation_min": 20, "permutation_max": 40, "permutation_check_interval": 10},
    )
    assert 0.0 < p_value <= 1.0


def test_partial_spearman_permutation_rejects_undefined_correlation() -> None:
    with np.errstate(invalid="ignore", divide="ignore"):
        with pytest.raises(ValueError, match="Permutation score must be finite"):
            _permuted_partial_spearman_exceeds(np.ones(4), np.arange(4.0), 0.5, 42)


def test_ineligible_contrasts_keep_normalized_schema() -> None:
    cfg = _cfg()
    cfg["analysis"]["preregistered_panel"]["min_subjects_per_group"] = 10

    contrasts = run_panel_contrasts(_frame(), cfg=cfg)

    assert set(contrasts["status"]) == {"insufficient_n"}
    assert {"raw_p", "within_scan_p", "family_adjusted_p", "family_id", "inference_role"}.issubset(
        contrasts.columns
    )
    assert contrasts["raw_p"].isna().all()


def test_run_confirmatory_panel_statistics_writes_outputs(tmp_path: Path) -> None:
    frame = _frame()
    feature_matrix_path = tmp_path / "subject_feature_matrix_whole.parquet"
    panel_matrix_path = tmp_path / "panel_feature_matrix.csv"
    frame.drop(columns=["oxidative_index", "membrane_integrity_index", "carotenoid_1525", "lipid_carbonyl_1721", "ch_bulk_2885"]).to_parquet(feature_matrix_path, index=False)
    frame[["subject_id", "oxidative_index", "membrane_integrity_index", "carotenoid_1525", "lipid_carbonyl_1721", "ch_bulk_2885"]].to_csv(panel_matrix_path, index=False)

    outputs = run_confirmatory_panel_statistics(
        feature_matrix_path=feature_matrix_path,
        panel_matrix_path=panel_matrix_path,
        output_dir=tmp_path / "panel",
        cfg=_cfg(),
    )

    assert {
        "primary_anchor_result",
        "panel_contrast_tests",
        "panel_continuous_associations",
        "panel_variance_homogeneity",
    }.issubset(outputs)
    assert pd.read_csv(outputs["panel_continuous_associations"])["partial_spearman_p_value"].notna().any()
    assert "brown_forsythe_p_value" in pd.read_csv(outputs["panel_variance_homogeneity"]).columns


def test_effect_plots_skip_empty_tables(tmp_path):
    from src.analysis.robust_contrasts import _plot_effect_sizes, _plot_raincloud_surrogate

    assert _plot_effect_sizes(pd.DataFrame(), tmp_path) is None
    assert _plot_raincloud_surrogate(pd.DataFrame(), pd.DataFrame(), tmp_path) is None
    assert not list(tmp_path.iterdir())
