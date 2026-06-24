from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from src.analysis.sklearn_utils import (
    covariate_logistic_model,
    fit_with_convergence_capture,
    l1_logistic_model,
)


def test_l1_logistic_model_uses_high_max_iter() -> None:
    model = l1_logistic_model(c=1.0, random_state=0)
    assert model.max_iter == 10_000


def test_covariate_logistic_model_uses_high_max_iter() -> None:
    model = covariate_logistic_model(random_state=0)
    assert model.max_iter == 10_000


def test_fit_with_convergence_capture_suppresses_warning_output(capsys) -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 5))
    y = (rng.normal(size=20) > 0).astype(int)
    model = LogisticRegression(solver="liblinear", max_iter=1, random_state=0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fit_with_convergence_capture(model, X, y, context="test logistic")

    captured = capsys.readouterr()
    assert "ConvergenceWarning" not in captured.err
