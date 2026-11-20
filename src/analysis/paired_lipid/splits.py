"""Deterministic participant-grouped outer and inner cross-validation splits."""

from __future__ import annotations

import numpy as np

from src.analysis.paired_lipid.cohort import CohortDataset, PairedLipidConfig


def make_participant_splits(
    n_participants: int,
    paired_mask: np.ndarray,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create deterministic participant-grouped folds stratified by pair availability."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    rng = np.random.default_rng(seed)
    paired_idx = np.flatnonzero(paired_mask)
    single_idx = np.flatnonzero(~paired_mask)
    paired_perm = rng.permutation(paired_idx)
    single_perm = rng.permutation(single_idx)

    folds: list[list[int]] = [[] for _ in range(n_splits)]
    for i, idx in enumerate(paired_perm):
        folds[i % n_splits].append(int(idx))
    for i, idx in enumerate(single_perm):
        folds[(i + paired_perm.size) % n_splits].append(int(idx))

    all_idx = np.arange(n_participants, dtype=int)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for f in folds:
        test_idx = np.asarray(sorted(f), dtype=int)
        train_idx = np.setdiff1d(all_idx, test_idx, assume_unique=True)
        splits.append((train_idx, test_idx))
    return splits


def make_nested_splits(
    dataset: CohortDataset,
    cfg: PairedLipidConfig,
    seed: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[list[tuple[np.ndarray, np.ndarray]]]]:
    """Locked outer folds and, per outer fold f, inner folds seeded with seed + 1000 * (f + 1)."""
    n_p = dataset.n_participants
    paired_mask = np.isin(np.arange(n_p), dataset.paired_participant_indices)
    outer = make_participant_splits(n_p, paired_mask, cfg.n_outer_splits, seed)
    inner = [
        make_participant_splits(
            tr_idx.size,
            paired_mask[tr_idx],
            cfg.n_inner_splits,
            seed + 1000 * (fold_idx + 1),
        )
        for fold_idx, (tr_idx, _) in enumerate(outer)
    ]
    return outer, inner
