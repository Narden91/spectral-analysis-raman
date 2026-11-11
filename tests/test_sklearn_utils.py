from __future__ import annotations

import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.analysis.sklearn_utils import (
    fit_with_convergence_capture,
    l1_logistic_model,
)


def test_l1_logistic_model_uses_high_max_iter() -> None:
    model = l1_logistic_model(c=1.0, random_state=0)
    assert model.max_iter == 10_000


def test_l1_logistic_model_is_l1_penalty() -> None:
    model = l1_logistic_model(c=1.0, random_state=0)
    assert model.penalty == "l1"
    assert model.solver == "liblinear"
    assert model.dual is False
    assert model.class_weight == "balanced"


def test_l1_logistic_model_produces_sparse_coefficients() -> None:
    rng = np.random.default_rng(0)
    n_samples, n_informative, n_noise = 200, 3, 47
    X_informative = rng.normal(size=(n_samples, n_informative))
    weights = np.array([2.5, -2.0, 1.8])
    logits = X_informative @ weights
    y = (logits > 0).astype(int)
    X_noise = rng.normal(size=(n_samples, n_noise))
    X = np.hstack([X_informative, X_noise])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = l1_logistic_model(c=0.1, random_state=0)
    fit_with_convergence_capture(model, X_scaled, y, context="test l1 sparsity")

    coef = model.coef_.ravel()
    assert coef.shape[0] == n_informative + n_noise
    assert (coef == 0).mean() > 0.5


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
