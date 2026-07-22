"""Adaptive early-stopping for single-statistic Monte Carlo permutation tests.

Applies only where one permutation null is compared against one observed
statistic per call (independent p-value per invocation). It does not apply to
shared family-wise nulls compared against many simultaneous statistics (e.g.
cluster-mass maxima checked against multiple clusters in one block) — that
case needs the tail resolved against the most extreme of several statistics,
which this curtailment rule does not address and must not be used for.

Method: after each batch of new draws, form the exact Clopper-Pearson
confidence interval for the true one-sided tail probability p = P(null >=
observed) from x = #{null >= observed} successes in k draws, at level
`confidence`. Stop once the interval no longer straddles `alpha` (the
decision "reject" vs "fail to reject" at that alpha is already certain at the
requested confidence) or once `permutation_max` draws are reached. Never
stops before `permutation_min` draws, since Monte Carlo p-value validity
requires an a-priori draw count and the observed statistic must always be
included in the tail count.

Reference: Besag & Clifford (1991), "Sequential Monte Carlo p-values";
this is the standard curtailment construction for permutation testing, not a
novel or experimental method.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from scipy import stats


@dataclass(frozen=True)
class SequentialStoppingConfig:
    permutation_min: int
    permutation_max: int
    check_interval: int
    confidence: float = 0.99
    alpha: float = 0.05

    def __post_init__(self) -> None:
        if self.permutation_min < 1:
            raise ValueError("permutation_min must be >= 1")
        if self.permutation_max < self.permutation_min:
            raise ValueError("permutation_max must be >= permutation_min")
        if self.check_interval < 1:
            raise ValueError("check_interval must be >= 1")
        if not (0.0 < self.confidence < 1.0):
            raise ValueError("confidence must be in (0, 1)")
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")


def clopper_pearson_interval(successes: int, draws: int, *, confidence: float) -> tuple[float, float]:
    """Exact two-sided Clopper-Pearson CI for a binomial proportion."""
    if draws <= 0:
        return 0.0, 1.0
    tail = (1.0 - confidence) / 2.0
    lower = 0.0 if successes == 0 else float(stats.beta.ppf(tail, successes, draws - successes + 1))
    upper = 1.0 if successes == draws else float(stats.beta.ppf(1.0 - tail, successes + 1, draws - successes))
    return lower, upper


def is_resolved(successes: int, draws: int, *, alpha: float, confidence: float) -> bool:
    """True once the CI for the Monte Carlo p-value's tail probability clears alpha."""
    lower, upper = clopper_pearson_interval(successes, draws, confidence=confidence)
    return upper < alpha or lower > alpha


def should_stop(successes: int, draws: int, config: SequentialStoppingConfig) -> bool:
    """Whether a permutation loop may stop after `draws` completed draws.

    `successes` counts draws so far with null >= observed (the tail count
    feeding the eventual Monte Carlo p-value `(1 + successes) / (draws + 1)`).
    Stopping is only permitted at or after `permutation_min` draws, and only
    on a `check_interval` boundary or at `permutation_max`.
    """
    if draws < config.permutation_min:
        return False
    if draws >= config.permutation_max:
        return True
    if draws % config.check_interval != 0:
        return False
    return is_resolved(successes, draws, alpha=config.alpha, confidence=config.confidence)


_OPTIONAL_INT_FIELDS = ("permutation_min", "permutation_max", "permutation_check_interval")
_OPTIONAL_FLOAT_FIELDS = ("permutation_confidence", "permutation_alpha")


def validate_sequential_stopping_config(cfg: dict, *, section: str) -> None:
    """Validate the optional adaptive-stopping keys on a permutation-test config section.

    All keys are optional; only type/range is checked when present. This does
    not touch the section's own `n_permutations` validation (each caller
    validates that separately, since some sections don't validate it at all).
    """
    for field in _OPTIONAL_INT_FIELDS:
        if field not in cfg:
            continue
        value = cfg[field]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{section}.{field} must be a positive integer, got {value!r}")

    for field in _OPTIONAL_FLOAT_FIELDS:
        if field not in cfg:
            continue
        value = cfg[field]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not (0.0 < float(value) < 1.0):
            raise ValueError(f"{section}.{field} must be a number strictly between 0 and 1, got {value!r}")

    if "permutation_min" in cfg and "permutation_max" in cfg and int(cfg["permutation_min"]) > int(cfg["permutation_max"]):
        raise ValueError(f"{section}.permutation_min must be <= {section}.permutation_max")


def run_curtailed_permutation_draws(
    draw_fn: Callable[..., float],
    seed_args: Sequence[tuple[Any, ...]],
    *,
    observed: float,
    config: SequentialStoppingConfig,
    n_jobs: int,
) -> list[float]:
    """Run `draw_fn(*args)` over `seed_args` in chunks, stopping early per `config`.

    `seed_args` must already be truncated/generated to `config.permutation_max`
    entries. Chunked (not a single streamed generator) because interrupting a
    live `joblib.Parallel(return_as="generator_unordered")` mid-stream leaves
    in-flight futures whose callbacks fire after the executor tears down,
    raising `ShutdownExecutorError` on every cancelled task; submitting fixed
    `config.check_interval`-sized batches and checking the stopping condition
    only between batches avoids that entirely.
    """
    values: list[float] = []
    successes = 0
    offset = 0
    while offset < len(seed_args) and not should_stop(successes=successes, draws=len(values), config=config):
        chunk = seed_args[offset : offset + config.check_interval]
        offset += config.check_interval
        if n_jobs != 1 and len(chunk) > 1:
            chunk_values = Parallel(n_jobs=n_jobs)(delayed(draw_fn)(*args) for args in chunk)
        else:
            chunk_values = [draw_fn(*args) for args in chunk]
        values.extend(chunk_values)
        successes += sum(1 for value in chunk_values if np.isfinite(value) and value >= observed)
    return values


def resolve_sequential_stopping_config(cfg: dict, *, default_max: int) -> SequentialStoppingConfig:
    """Build a stopping config from a permutation-test config section.

    `permutation_max` defaults to the section's existing `n_permutations`
    value so a config that has not opted into the new keys keeps running the
    full previously-configured budget in the worst case (fully resolved
    early only when the data allows it, never more draws than before).
    `permutation_min` and `check_interval` are clamped to `permutation_max`
    so a deliberately small `n_permutations` (fast/dev profiles) still runs
    exactly that many draws instead of raising or being forced upward.
    """
    permutation_max = int(cfg.get("permutation_max", default_max))
    permutation_min = min(int(cfg.get("permutation_min", 200)), permutation_max)
    check_interval = min(int(cfg.get("permutation_check_interval", 100)), permutation_max)
    return SequentialStoppingConfig(
        permutation_min=permutation_min,
        permutation_max=permutation_max,
        check_interval=check_interval,
        confidence=float(cfg.get("permutation_confidence", 0.99)),
        alpha=float(cfg.get("permutation_alpha", 0.05)),
    )
