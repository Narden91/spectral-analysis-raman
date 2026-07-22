"""Shared statistical utilities for cluster-based and sequential inference."""

from .cluster_permutation import (
    apply_binary_cluster_permutation,
    apply_spearman_cluster_permutation,
    find_signed_clusters,
)
from .sequential_permutation import (
    SequentialStoppingConfig,
    clopper_pearson_interval,
    is_resolved,
    resolve_sequential_stopping_config,
    run_curtailed_permutation_draws,
    should_stop,
    validate_sequential_stopping_config,
)

__all__ = [
    "apply_binary_cluster_permutation",
    "apply_spearman_cluster_permutation",
    "find_signed_clusters",
    "SequentialStoppingConfig",
    "clopper_pearson_interval",
    "is_resolved",
    "resolve_sequential_stopping_config",
    "run_curtailed_permutation_draws",
    "should_stop",
    "validate_sequential_stopping_config",
]
