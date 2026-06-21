from __future__ import annotations

import sys

import pandas as pd
import pytest

from src import PROJECT_ROOT, _load_yaml_with_extends
from src.analysis.eda_mode import (
    apply_eda_analysis_mode,
    eda_output_root,
    filter_index_for_eda_mode,
    group_configs_for_eda_mode,
    resolve_eda_analysis_mode,
    validate_eda_mode_config,
)


def _eda_cfg(path: str) -> dict:
    return _load_yaml_with_extends(PROJECT_ROOT / path)


def _subject_index() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": ["C01", "D01", "D02"],
            "is_control": [True, False, False],
        }
    )


def test_default_eda_config_resolves_case_control_mode() -> None:
    cfg = _eda_cfg("config/eda.yaml")
    mode = resolve_eda_analysis_mode(cfg)

    assert mode == "disease_vs_control"
    validate_eda_mode_config(cfg, mode)
    assert eda_output_root(mode) == "Results_control_vs_disease"


def test_wrapper_configs_resolve_expected_modes() -> None:
    control_cfg = _eda_cfg("config/eda_disease_control.yaml")
    disease_cfg = _eda_cfg("config/eda_disease_only.yaml")

    assert resolve_eda_analysis_mode(control_cfg) == "disease_vs_control"
    assert resolve_eda_analysis_mode(disease_cfg) == "disease_only"


def test_cli_override_wins_for_mode_resolution() -> None:
    cfg = _eda_cfg("config/eda_disease_control.yaml")

    mode = resolve_eda_analysis_mode(cfg, override="disease_only")

    assert mode == "disease_only"


def test_cli_parser_accepts_analysis_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_eda.py", "--analysis-mode", "disease_only"])
    import scripts.run_eda as run_eda_script

    args = run_eda_script._parse_args()

    assert args.analysis_mode == "disease_only"


def test_disease_only_requires_intra_disease_phase() -> None:
    cfg = _eda_cfg("config/eda.yaml")
    apply_eda_analysis_mode(cfg, "disease_only")

    with pytest.raises(ValueError, match="requires analysis.advanced.intra_disease.enabled=true"):
        validate_eda_mode_config(cfg, "disease_only")


def test_disease_vs_control_rejects_intra_disease_phase() -> None:
    cfg = _eda_cfg("config/eda_disease_only.yaml")
    apply_eda_analysis_mode(cfg, "disease_vs_control")

    with pytest.raises(ValueError, match="requires analysis.advanced.intra_disease.enabled=false"):
        validate_eda_mode_config(cfg, "disease_vs_control")


def test_disease_vs_control_requires_both_cohorts_in_data_source() -> None:
    cfg = _eda_cfg("config/eda_disease_control.yaml")
    df_index = pd.DataFrame({"subject_id": ["D01", "D02"], "is_control": [False, False]})

    with pytest.raises(ValueError, match="requires both control and disease subjects"):
        validate_eda_mode_config(cfg, "disease_vs_control", df_index)


def test_disease_only_requires_disease_subjects_in_data_source() -> None:
    cfg = _eda_cfg("config/eda_disease_only.yaml")
    df_index = pd.DataFrame({"subject_id": ["C01"], "is_control": [True]})

    with pytest.raises(ValueError, match="requires at least one disease subject"):
        validate_eda_mode_config(cfg, "disease_only", df_index)


def test_disease_only_routing_filters_controls_and_uses_intra_disease_root() -> None:
    df_index = _subject_index()

    filtered = filter_index_for_eda_mode(df_index, "disease_only")
    group_configs = group_configs_for_eda_mode("disease_only")

    assert filtered["subject_id"].tolist() == ["D01", "D02"]
    assert not any(bool(group["is_control"]) for group in group_configs.values())
    assert eda_output_root("disease_only") == "Results_intra_disease"


def test_disease_vs_control_routing_keeps_both_cohorts() -> None:
    df_index = _subject_index()

    filtered = filter_index_for_eda_mode(df_index, "disease_vs_control")
    group_configs = group_configs_for_eda_mode("disease_vs_control")

    assert filtered.equals(df_index)
    assert any(bool(group["is_control"]) for group in group_configs.values())
    assert any(not bool(group["is_control"]) for group in group_configs.values())
