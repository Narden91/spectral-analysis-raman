from __future__ import annotations

import numpy as np
import pytest

from src.statistics.sequential_permutation import (
    SequentialStoppingConfig,
    clopper_pearson_interval,
    is_resolved,
    resolve_sequential_stopping_config,
    run_curtailed_permutation_draws,
    should_stop,
)


def test_clopper_pearson_interval_matches_independent_implementation() -> None:
    statsmodels = pytest.importorskip("statsmodels.stats.proportion")
    for successes, draws in ((10, 100), (0, 50), (50, 50), (1, 30), (29, 30)):
        lower, upper = clopper_pearson_interval(successes, draws, confidence=0.95)
        ref_lower, ref_upper = statsmodels.proportion_confint(successes, draws, alpha=0.05, method="beta")
        assert lower == pytest.approx(ref_lower, abs=1e-8)
        assert upper == pytest.approx(ref_upper, abs=1e-8)


def test_clopper_pearson_interval_zero_successes() -> None:
    lower, upper = clopper_pearson_interval(0, 50, confidence=0.95)
    assert lower == 0.0
    assert upper < 1.0


def test_clopper_pearson_interval_all_successes() -> None:
    lower, upper = clopper_pearson_interval(50, 50, confidence=0.95)
    assert upper == 1.0
    assert lower > 0.0


def test_clopper_pearson_interval_zero_draws_is_maximally_uncertain() -> None:
    assert clopper_pearson_interval(0, 0, confidence=0.95) == (0.0, 1.0)


def test_is_resolved_false_when_ci_straddles_alpha() -> None:
    # Small sample, count near the boundary: CI must still contain 0.05.
    assert not is_resolved(2, 20, alpha=0.05, confidence=0.99)


def test_is_resolved_true_when_clearly_significant() -> None:
    # Strong, unambiguous signal: null essentially never beats observed.
    assert is_resolved(0, 500, alpha=0.05, confidence=0.99)


def test_is_resolved_true_when_clearly_not_significant() -> None:
    # Tail probability far above alpha with tight CI.
    assert is_resolved(400, 500, alpha=0.05, confidence=0.99)


def test_should_stop_never_before_permutation_min() -> None:
    config = SequentialStoppingConfig(permutation_min=200, permutation_max=1000, check_interval=50)
    # Even a maximally resolved outcome (0 successes) must not stop early.
    assert not should_stop(successes=0, draws=100, config=config)


def test_should_stop_always_at_permutation_max() -> None:
    config = SequentialStoppingConfig(permutation_min=10, permutation_max=100, check_interval=1000)
    # Ambiguous count, but budget exhausted.
    assert should_stop(successes=5, draws=100, config=config)


def test_should_stop_only_fires_on_check_interval_boundary() -> None:
    config = SequentialStoppingConfig(permutation_min=10, permutation_max=1000, check_interval=50)
    # 60 is past permutation_min but not a multiple of check_interval (50).
    assert not should_stop(successes=0, draws=60, config=config)


def test_should_stop_fires_on_boundary_when_resolved() -> None:
    config = SequentialStoppingConfig(permutation_min=10, permutation_max=1000, check_interval=50, confidence=0.99)
    assert should_stop(successes=0, draws=200, config=config)


def test_should_stop_does_not_fire_on_boundary_when_unresolved() -> None:
    config = SequentialStoppingConfig(permutation_min=10, permutation_max=1000, check_interval=50, confidence=0.999)
    # A count right at the alpha boundary should remain unresolved at low draw counts.
    assert not should_stop(successes=10, draws=200, config=config)


def test_sequential_stopping_config_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="permutation_min"):
        SequentialStoppingConfig(permutation_min=0, permutation_max=100, check_interval=10)
    with pytest.raises(ValueError, match="permutation_max"):
        SequentialStoppingConfig(permutation_min=100, permutation_max=50, check_interval=10)
    with pytest.raises(ValueError, match="check_interval"):
        SequentialStoppingConfig(permutation_min=10, permutation_max=100, check_interval=0)
    with pytest.raises(ValueError, match="confidence"):
        SequentialStoppingConfig(permutation_min=10, permutation_max=100, check_interval=10, confidence=1.5)
    with pytest.raises(ValueError, match="alpha"):
        SequentialStoppingConfig(permutation_min=10, permutation_max=100, check_interval=10, alpha=0.0)


def test_resolve_sequential_stopping_config_defaults_max_to_n_permutations() -> None:
    config = resolve_sequential_stopping_config({}, default_max=1000)
    assert config.permutation_max == 1000
    assert config.permutation_min == 200
    assert config.check_interval == 100
    assert config.confidence == 0.99
    assert config.alpha == 0.05


def test_resolve_sequential_stopping_config_clamps_min_and_interval_to_small_max() -> None:
    # Fast/dev profiles configure n_permutations far below the 200-draw default
    # minimum; the resolved config must still be internally valid and run
    # exactly that many draws, not raise or silently inflate the budget.
    config = resolve_sequential_stopping_config({}, default_max=5)
    assert config.permutation_max == 5
    assert config.permutation_min == 5
    assert config.check_interval == 5


def test_resolve_sequential_stopping_config_honors_explicit_overrides() -> None:
    cfg = {
        "permutation_min": 50,
        "permutation_max": 500,
        "permutation_check_interval": 25,
        "permutation_confidence": 0.95,
        "permutation_alpha": 0.01,
    }
    config = resolve_sequential_stopping_config(cfg, default_max=1000)
    assert config == SequentialStoppingConfig(
        permutation_min=50, permutation_max=500, check_interval=25, confidence=0.95, alpha=0.01
    )


def test_run_curtailed_permutation_draws_stops_early_when_resolved() -> None:
    config = SequentialStoppingConfig(permutation_min=50, permutation_max=2000, check_interval=25)
    seed_args = [(seed,) for seed in range(config.permutation_max)]

    # Every draw scores 0.0, observed is 1.0: unambiguously never in the tail.
    values = run_curtailed_permutation_draws(
        lambda _seed: 0.0, seed_args, observed=1.0, config=config, n_jobs=1
    )
    assert len(values) < config.permutation_max
    assert len(values) >= config.permutation_min


def test_run_curtailed_permutation_draws_exhausts_budget_when_ambiguous() -> None:
    config = SequentialStoppingConfig(permutation_min=20, permutation_max=20, check_interval=20)
    seed_args = [(seed,) for seed in range(config.permutation_max)]

    values = run_curtailed_permutation_draws(
        lambda _seed: 0.5, seed_args, observed=0.5, config=config, n_jobs=1
    )
    assert len(values) == 20


def test_run_curtailed_permutation_draws_parallel_matches_serial() -> None:
    config = SequentialStoppingConfig(permutation_min=30, permutation_max=30, check_interval=30)
    seed_args = [(seed,) for seed in range(config.permutation_max)]

    def draw(seed: int) -> float:
        return float(np.random.default_rng(seed).normal())

    serial = run_curtailed_permutation_draws(draw, seed_args, observed=0.0, config=config, n_jobs=1)
    parallel = run_curtailed_permutation_draws(draw, seed_args, observed=0.0, config=config, n_jobs=2)
    assert serial == parallel


def test_never_stops_before_min_even_under_repeated_simulation() -> None:
    # Monte Carlo sanity: across many synthetic null-vs-observed comparisons,
    # the rule must never report a stop decision before permutation_min draws.
    config = SequentialStoppingConfig(permutation_min=150, permutation_max=2000, check_interval=25)
    rng = np.random.default_rng(7)
    for _ in range(200):
        p_true = float(rng.uniform(0.0, 1.0))
        draws = int(rng.integers(1, config.permutation_min))
        successes = int(rng.binomial(draws, p_true))
        assert not should_stop(successes, draws, config)
