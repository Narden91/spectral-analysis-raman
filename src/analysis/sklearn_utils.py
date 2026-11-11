"""Shared sklearn fit helpers with convergence warning handling."""

from __future__ import annotations

import logging
import warnings
from typing import Any

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

LOGISTIC_MAX_ITER = 10_000
LOGISTIC_TOL = 1e-4


def l1_logistic_model(*, c: float, random_state: int) -> LogisticRegression:
    return LogisticRegression(
        solver="liblinear",
        penalty="l1",
        dual=False,
        C=c,
        class_weight="balanced",
        random_state=random_state,
        max_iter=LOGISTIC_MAX_ITER,
        tol=LOGISTIC_TOL,
    )


def fit_with_convergence_capture(
    model: Any,
    X: Any,
    y: Any,
    *,
    context: str = "model",
) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X, y)
    for warning in caught:
        logger.debug("%s convergence warning: %s", context, warning.message)
