from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.analysis.clinical_feature_associations import filter_disease_subjects


def test_filter_disease_subjects_drops_missing_is_control(caplog) -> None:
    frame = pd.DataFrame(
        {
            "subject_id": ["S1", "S2", "S3"],
            "is_control": [True, False, np.nan],
        }
    )

    with caplog.at_level(logging.WARNING):
        result = filter_disease_subjects(frame)

    assert result["subject_id"].tolist() == ["S2"]
    assert "S3" in caplog.text


def test_filter_disease_subjects_condition_fallback_drops_unknown() -> None:
    frame = pd.DataFrame(
        {
            "subject_id": ["S1", "S2", "S3", "S4"],
            "condition": ["disease", "control", "unknown", np.nan],
        }
    )

    result = filter_disease_subjects(frame)

    assert result["subject_id"].tolist() == ["S1"]


def test_filter_disease_subjects_routine_control_exclusion_does_not_warn(caplog) -> None:
    frame = pd.DataFrame(
        {
            "subject_id": ["S1", "S2", "S3"],
            "is_control": [True, True, False],
        }
    )

    with caplog.at_level(logging.WARNING):
        result = filter_disease_subjects(frame)

    assert result["subject_id"].tolist() == ["S3"]
    assert caplog.text == ""
