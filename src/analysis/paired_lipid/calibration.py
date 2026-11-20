"""Bootstrap interval calibration, feasibility gate checks, and synthetic scenario suites."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Literal, NamedTuple

import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.io_utils import write_csv_atomic
from src.analysis.paired_lipid.batched import resolve_array_backend
from src.analysis.paired_lipid.cohort import CohortDataset, PairedLipidConfig
from src.analysis.paired_lipid.nested_cv import (
    INTERVAL_METHODS,
    IntervalMethod,
    PreparedExternalTest,
    evaluate_nested_cohort,
    run_multiplicity_bootstrap,
    score_external_primary_pair,
)
from src.analysis.paired_lipid.synthetic import (
    DEFAULT_N_SINGLE_SAMPLE,
    generate_synthetic_cohort,
)

SeedSet = Literal["development", "validation"]
# Disjoint replicate-seed ranges; the independent test cohort uses base_seed (< 10_000).
CALIBRATION_SEED_RANGES: dict[SeedSet, tuple[int, int]] = {
    "development": (10_000, 5_000_000),
    "validation": (5_000_000, 10_000_000),
}
_SCENARIO_RESERVED_KEYS: frozenset[str] = frozenset(
    {"scenario", "covariate_preset", "n_participants", "seed", "compact_grid"}
)
_SCENARIO_META_KEYS: tuple[str, ...] = ("scenario", "covariate_preset")


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Compute the Wilson score interval for a binomial proportion."""
    if total <= 0:
        return (float("nan"), float("nan"))
    z = float(stats.norm.ppf(0.5 + 0.5 * confidence))
    phat = float(successes) / float(total)
    denom = 1.0 + (z**2) / float(total)
    center = (phat + (z**2) / (2.0 * float(total))) / denom
    half = (z / denom) * math.sqrt(
        (phat * (1.0 - phat) / float(total)) + ((z**2) / (4.0 * (float(total) ** 2)))
    )
    return (max(0.0, center - half), min(1.0, center + half))


def min_reps_for_gate(wilson_lower_min: float, confidence: float = 0.95) -> int:
    """Smallest repetition count whose Wilson lower bound at 100% coverage reaches the gate."""
    # At n/n coverage the Wilson lower bound is n / (n + z^2).
    z_sq = float(stats.norm.ppf(0.5 + 0.5 * confidence)) ** 2
    return math.ceil(wilson_lower_min * z_sq / (1.0 - wilson_lower_min))


def check_gate_attainable(n_repetitions: int, seed_set: str, wilson_lower_min: float) -> tuple[bool, int]:
    """Return (attainable, minimum reps); raise for a validation run that cannot pass the gate."""
    if seed_set not in CALIBRATION_SEED_RANGES:
        raise ValueError(
            f"Unknown seed_set: {seed_set!r}. Expected one of {tuple(CALIBRATION_SEED_RANGES)}."
        )
    min_reps = min_reps_for_gate(wilson_lower_min)
    if seed_set == "validation" and n_repetitions < min_reps:
        # Even 100% coverage cannot reach the Wilson lower bound, so refuse before any compute.
        raise ValueError(
            f"Validation calibration needs >= {min_reps} repetitions to be able to pass the gate "
            f"(Wilson lower >= {wilson_lower_min}); got {n_repetitions}."
        )
    return n_repetitions >= min_reps, min_reps


class _CalibrationRepResult(NamedTuple):
    """Per-replicate bootstrap intervals and external target scores for primary and comparator endpoints."""

    point_delta_mse: float
    ci_lo_delta_mse: float
    ci_hi_delta_mse: float
    succeeded: bool
    ext_delta_mse: float
    ext_part_deltas_mse: np.ndarray
    point_vs_comp: float
    ci_lo_vs_comp: float
    ci_hi_vs_comp: float
    ext_delta_vs_comp: float
    ext_part_deltas_vs_comp: np.ndarray


def _eval_single_calibration_rep(
    rep_seed: int,
    n_participants: int,
    gen_kwargs: dict[str, Any],
    n_bootstraps: int,
    cfg: PairedLipidConfig,
    use_gpu: bool | None,
    test_ds: CohortDataset | PreparedExternalTest,
    compact_grid: bool,
    interval_method: IntervalMethod = "studentized_t",
) -> _CalibrationRepResult:
    """Run one synthetic calibration replicate: bootstrap intervals and external target scores."""
    ds, _ = generate_synthetic_cohort(
        n_participants=n_participants,
        seed=rep_seed,
        compact_grid=compact_grid,
        **gen_kwargs,
    )
    boot = run_multiplicity_bootstrap(
        ds,
        cfg=cfg,
        n_bootstraps=n_bootstraps,
        bootstrap_seed=rep_seed + 1,
        use_gpu=use_gpu,
        interval_method=interval_method,
    )
    ext_eval = score_external_primary_pair(ds, test_ds, cfg=cfg)
    ext_part_comp = ext_eval.get("external_participant_deltas_vs_comparator")
    return _CalibrationRepResult(
        point_delta_mse=float(boot["point_estimate_delta_mse"]),
        ci_lo_delta_mse=float(boot["ci_95_delta_mse"][0]),
        ci_hi_delta_mse=float(boot["ci_95_delta_mse"][1]),
        succeeded=bool(boot["interval_procedure_succeeded"]),
        ext_delta_mse=float(ext_eval["external_delta_mse"]),
        ext_part_deltas_mse=np.asarray(ext_eval["external_participant_deltas"], dtype=float),
        point_vs_comp=float(boot["point_estimate_vs_comparator"]),
        ci_lo_vs_comp=float(boot["ci_95_vs_comparator"][0]),
        ci_hi_vs_comp=float(boot["ci_95_vs_comparator"][1]),
        ext_delta_vs_comp=float(ext_eval.get("external_delta_vs_comparator", float("nan"))),
        ext_part_deltas_vs_comp=(
            np.asarray(ext_part_comp, dtype=float) if ext_part_comp is not None else np.empty(0, dtype=float)
        ),
    )


def _two_way_anova_ms_from_sufficient_stats(
    row_means: np.ndarray,
    col_sums: np.ndarray,
    row_ss_within: np.ndarray,
    n_ext: int,
) -> tuple[float, float, float]:
    """Compute exact two-way ANOVA mean squares in O(R + C) memory from per-rep sufficient stats."""
    n_rows = int(row_means.size)
    if n_rows == 0 or n_ext == 0:
        return float("nan"), float("nan"), float("nan")
    grand_mean = float(np.mean(row_means))
    ss_row = float(n_ext * np.sum((row_means - grand_mean) ** 2))
    col_means = col_sums / float(n_rows)
    ss_col = float(n_rows * np.sum((col_means - grand_mean) ** 2))
    ss_err = max(0.0, float(np.sum(row_ss_within)) - ss_col)
    ms_row = ss_row / float(n_rows - 1) if n_rows >= 2 else float("nan")
    ms_col = ss_col / float(n_ext - 1) if n_ext >= 2 else float("nan")
    ms_err = (
        ss_err / float((n_rows - 1) * (n_ext - 1)) if (n_rows >= 2 and n_ext >= 2) else float("nan")
    )
    return ms_row, ms_col, ms_err


def _calibration_fingerprint(
    *,
    n_repetitions: int,
    n_bootstraps: int,
    n_participants: int,
    n_test_participants: int,
    base_seed: int,
    seed_set: SeedSet,
    compact_grid: bool,
    interval_method: IntervalMethod,
    cfg: PairedLipidConfig,
    gen_kwargs: Mapping[str, Any],
    rep_seeds: np.ndarray,
) -> str:
    """Build a deterministic SHA-256 fingerprint guarding calibration checkpoint compatibility."""
    payload = {
        "n_repetitions": int(n_repetitions),
        "n_bootstraps": int(n_bootstraps),
        "n_participants": int(n_participants),
        "n_test_participants": int(n_test_participants),
        "base_seed": int(base_seed),
        "seed_set": str(seed_set),
        "compact_grid": bool(compact_grid),
        "interval_method": str(interval_method),
        "cfg": asdict(cfg),
        "gen_kwargs": {str(k): v for k, v in sorted(gen_kwargs.items())},
        "rep_seeds_sha256": hashlib.sha256(np.ascontiguousarray(rep_seeds, dtype=np.int64).tobytes()).hexdigest(),
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _save_calibration_checkpoint(path: Path, state: Mapping[str, Any]) -> None:
    """Atomically write compact O(R + C) calibration checkpoint state to an ``.npz`` archive."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with tmp_path.open("wb") as handle:
            np.savez(handle, **state)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _load_calibration_checkpoint(path: Path, expected_fingerprint: str) -> dict[str, Any]:
    """Load and verify a saved calibration checkpoint against ``expected_fingerprint``."""
    with np.load(path, allow_pickle=False) as data:
        stored_fp = str(data["fingerprint"].item())
        if stored_fp != expected_fingerprint:
            raise ValueError(
                f"Calibration checkpoint {path} does not match the current configuration/seed fingerprint; "
                "use a separate output directory or remove the stale checkpoint."
            )
        return {key: data[key].copy() for key in data.files if key != "fingerprint"}


def _summarize_interval_calibration(
    *,
    point_estimates: Sequence[float],
    ci_lows: Sequence[float],
    ci_highs: Sequence[float],
    succeeded_flags: Sequence[bool],
    external_deltas: np.ndarray,
    ext_col_sums: np.ndarray | None,
    ext_row_ss_within: np.ndarray | None,
    n_ext: int,
    gate_wilson_lower_min: float,
    gate_max_failure_rate: float,
) -> dict[str, Any]:
    """Summarize target uncertainty, bias, empirical coverage, tail asymmetry, and target-SE sensitivity."""
    n_repetitions = len(succeeded_flags)
    finite_ext = external_deltas[np.isfinite(external_deltas)]
    target_delta = float(np.mean(finite_ext)) if finite_ext.size > 0 else float("nan")
    target_mc_se = (
        float(np.std(finite_ext, ddof=1) / math.sqrt(finite_ext.size))
        if finite_ext.size > 1
        else float("nan")
    )
    if (
        n_repetitions > 0
        and n_ext > 1
        and ext_col_sums is not None
        and ext_row_ss_within is not None
        and ext_col_sums.size == n_ext
        and ext_row_ss_within.size == n_repetitions
        and np.all(np.isfinite(external_deltas))
        and np.all(np.isfinite(ext_col_sums))
        and np.all(np.isfinite(ext_row_ss_within))
    ):
        ms_row, ms_col, ms_err = _two_way_anova_ms_from_sufficient_stats(
            external_deltas, ext_col_sums, ext_row_ss_within, n_ext
        )
        denom_scale = float(n_repetitions * n_ext)
        target_test_se = float(math.sqrt(max(0.0, ms_col) / denom_scale))
        if n_repetitions > 1:
            var_total = max(0.0, (ms_row + ms_col - ms_err) / denom_scale)
            target_total_se = float(math.sqrt(var_total))
        else:
            target_total_se = float("nan")
    else:
        target_test_se = float("nan")
        target_total_se = float("nan")

    valid_intervals: list[tuple[float, float]] = []
    for lo, hi, ok in zip(ci_lows, ci_highs, succeeded_flags):
        if ok and np.isfinite(lo) and np.isfinite(hi):
            valid_intervals.append((float(lo), float(hi)))

    n_success = len(valid_intervals)
    widths = [hi - lo for lo, hi in valid_intervals]

    def _count_covered(target_val: float) -> int:
        if not np.isfinite(target_val):
            return 0
        return sum(1 for lo, hi in valid_intervals if lo <= target_val <= hi)

    uncond_covered = _count_covered(target_delta)
    uncond_cov = float(uncond_covered) / float(n_repetitions) if n_repetitions > 0 else float("nan")
    cond_cov = float(uncond_covered) / float(n_success) if n_success > 0 else float("nan")
    if n_success > 0 and np.isfinite(target_delta):
        # Lower noncoverage: target falls below CI lower bound (interval too high).
        # Upper noncoverage: target falls above CI upper bound (interval too low).
        lower_noncov = float(sum(1 for lo, _ in valid_intervals if target_delta < lo)) / float(n_success)
        upper_noncov = float(sum(1 for _, hi in valid_intervals if target_delta > hi)) / float(n_success)
    else:
        lower_noncov = float("nan")
        upper_noncov = float("nan")
    fail_rate = float(n_repetitions - n_success) / float(n_repetitions) if n_repetitions > 0 else float("nan")
    wilson_lo, wilson_hi = wilson_score_interval(uncond_covered, n_repetitions, confidence=0.95)
    passes_gate = bool(wilson_lo >= gate_wilson_lower_min and fail_rate < gate_max_failure_rate)
    detected_positive = sum(1 for lo, _ in valid_intervals if lo > 0.0)
    detection_rate = float(detected_positive) / float(n_repetitions) if n_repetitions > 0 else float("nan")
    det_wilson_lo, det_wilson_hi = wilson_score_interval(detected_positive, n_repetitions, confidence=0.95)

    if n_repetitions > 0 and np.isfinite(target_delta) and np.isfinite(target_total_se):
        z95 = float(stats.norm.ppf(0.975))
        cand_covs = [
            float(_count_covered(t_val)) / float(n_repetitions)
            for t_val in (
                target_delta,
                target_delta - z95 * target_total_se,
                target_delta + z95 * target_total_se,
            )
        ]
        target_sens_min_cov = float(min(cand_covs))
    else:
        target_sens_min_cov = float("nan")

    finite_points = [v for v in point_estimates if np.isfinite(v)]
    mean_point_est = float(np.mean(finite_points)) if finite_points else float("nan")
    emp_sd_point = float(np.std(finite_points, ddof=1)) if len(finite_points) > 1 else float("nan")
    return {
        "target_delta": target_delta,
        "target_mc_se": target_mc_se,
        "target_test_se": target_test_se,
        "target_total_se": target_total_se,
        "mean_point_estimate": mean_point_est,
        "empirical_sd_point_estimate": emp_sd_point,
        "bias": float(mean_point_est - target_delta),
        "unconditional_coverage": uncond_cov,
        "conditional_coverage": cond_cov,
        "lower_noncoverage_rate": lower_noncov,
        "upper_noncoverage_rate": upper_noncov,
        "target_sensitivity_min_coverage": target_sens_min_cov,
        "wilson_95_lower": wilson_lo,
        "wilson_95_upper": wilson_hi,
        "dataset_failure_rate": fail_rate,
        "mean_interval_width": float(np.mean(widths)) if widths else float("nan"),
        "detection_rate": detection_rate,
        "detection_wilson_95_lower": det_wilson_lo,
        "detection_wilson_95_upper": det_wilson_hi,
        "passes_provisional_gate": passes_gate,
    }


def evaluate_bootstrap_calibration(
    *,
    n_repetitions: int = 20,
    n_bootstraps: int = 100,
    n_participants: int = 60,
    signal_strength: float = 0.8,
    base_seed: int = 9000,
    cfg: PairedLipidConfig = PairedLipidConfig(),
    n_jobs: int = -1,
    use_gpu: bool | None = None,
    n_test_participants: int = 25000,
    seed_set: SeedSet = "development",
    compact_grid: bool = True,
    interval_method: IntervalMethod = "studentized_t",
    gate_wilson_lower_min: float = 0.925,
    gate_max_failure_rate: float = 0.01,
    scenario_kwargs: Mapping[str, Any] | None = None,
    checkpoint_path: str | Path | None = None,
    checkpoint_chunk_size: int | None = None,
) -> dict[str, Any]:
    """Evaluate bootstrap interval coverage, width, bias, and failure rate in synthetic repetitions.

    Target: the expected held-out risk difference MSE_cov - MSE_paired (and strongest_comparator -
    paired) of the procedure at this training size, estimated as the replicate mean of each
    replicate's outer-fold models scored on one independent test cohort (seed ``base_seed``, below
    the replicate seed range). Development and validation replicates draw seeds from disjoint
    ranges; only the validation set gives a confirmatory gate result.

    When ``checkpoint_path`` is provided, replicate sufficient statistics are saved atomically after
    each chunk and resumed automatically if the checkpoint matches the configuration fingerprint.
    """
    if interval_method not in INTERVAL_METHODS:
        raise ValueError(
            f"Unknown interval_method: {interval_method!r}. Expected one of {INTERVAL_METHODS}."
        )
    min_rep_seed = CALIBRATION_SEED_RANGES["development"][0]
    if base_seed < 0 or base_seed >= min_rep_seed:
        raise ValueError(
            f"base_seed ({base_seed}) must satisfy 0 <= base_seed < {min_rep_seed} "
            "to stay disjoint from replicate seeds."
        )
    if n_bootstraps < 1:
        raise ValueError(f"n_bootstraps must be >= 1 for bootstrap interval calibration; got {n_bootstraps}.")
    if n_test_participants < 2:
        raise ValueError(f"n_test_participants must be >= 2; got {n_test_participants}.")
    gate_attainable, min_reps = check_gate_attainable(n_repetitions, seed_set, gate_wilson_lower_min)
    lo_seed, hi_seed = CALIBRATION_SEED_RANGES[seed_set]
    n_slots = (hi_seed - lo_seed) // 2
    if n_repetitions < 1 or n_repetitions > n_slots:
        raise ValueError(
            f"n_repetitions ({n_repetitions}) must be between 1 and {n_slots} for seed_set={seed_set!r}."
        )
    eff_cfg = (
        replace(cfg, covariate_preset=str(scenario_kwargs["covariate_preset"]))
        if scenario_kwargs is not None and "covariate_preset" in scenario_kwargs
        else cfg
    )
    gen_kwargs: dict[str, Any] = {"signal_strength": float(signal_strength)}
    if scenario_kwargs is not None:
        for k, v in scenario_kwargs.items():
            if k not in _SCENARIO_RESERVED_KEYS:
                gen_kwargs[k] = v

    rng = np.random.default_rng(base_seed)
    # Draw unique even offsets so rep_seed and bootstrap_seed = rep_seed + 1 can never collide.
    rep_seeds = lo_seed + 2 * rng.choice(n_slots, size=n_repetitions, replace=False)

    ckpt_file = Path(checkpoint_path) if checkpoint_path is not None else None
    fingerprint = (
        _calibration_fingerprint(
            n_repetitions=n_repetitions,
            n_bootstraps=n_bootstraps,
            n_participants=n_participants,
            n_test_participants=n_test_participants,
            base_seed=base_seed,
            seed_set=seed_set,
            compact_grid=compact_grid,
            interval_method=interval_method,
            cfg=eff_cfg,
            gen_kwargs=gen_kwargs,
            rep_seeds=rep_seeds,
        )
        if ckpt_file is not None
        else ""
    )

    # Synthetic cohorts have a flat white-noise tail beyond 3 latent factors and bypass GPU CheFSI,
    # so parallel multi-core CPU LAPACK dsyevr (~8.7 s/rep at 2,000 boots) outperforms serial GPU
    # eigh (~34.4 s/rep) when >= 4 CPU cores are available. Explicit use_gpu=True is always honored.
    prefer_parallel_cpu = (
        use_gpu is None
        and n_repetitions >= 4
        and n_jobs != 1
        and (os.cpu_count() or 1) >= 4
    )
    effective_use_gpu: bool | None = False if prefer_parallel_cpu else use_gpu
    _, gpu_meta = resolve_array_backend(effective_use_gpu, n_bootstraps)
    gpu_active = bool(gpu_meta["gpu_active"])

    n_ext = int(n_test_participants)
    if ckpt_file is not None and ckpt_file.is_file():
        state = _load_calibration_checkpoint(ckpt_file, fingerprint)
    else:
        state = {
            "completed": np.zeros(n_repetitions, dtype=bool),
            "point_delta_mse": np.full(n_repetitions, np.nan, dtype=np.float64),
            "ci_lo_delta_mse": np.full(n_repetitions, np.nan, dtype=np.float64),
            "ci_hi_delta_mse": np.full(n_repetitions, np.nan, dtype=np.float64),
            "succeeded": np.zeros(n_repetitions, dtype=bool),
            "ext_delta_mse": np.full(n_repetitions, np.nan, dtype=np.float64),
            "ext_row_ss_mse": np.full(n_repetitions, np.nan, dtype=np.float64),
            "ext_col_sum_mse": np.zeros(n_ext, dtype=np.float64),
            "ext_part_valid_mse": np.array(True, dtype=bool),
            "point_vs_comp": np.full(n_repetitions, np.nan, dtype=np.float64),
            "ci_lo_vs_comp": np.full(n_repetitions, np.nan, dtype=np.float64),
            "ci_hi_vs_comp": np.full(n_repetitions, np.nan, dtype=np.float64),
            "ext_delta_vs_comp": np.full(n_repetitions, np.nan, dtype=np.float64),
            "ext_row_ss_comp": np.full(n_repetitions, np.nan, dtype=np.float64),
            "ext_col_sum_comp": np.zeros(n_ext, dtype=np.float64),
            "ext_part_valid_comp": np.array(True, dtype=bool),
        }

    pending_indices = [int(i) for i in np.flatnonzero(~state["completed"])]
    if pending_indices:
        # Match the training single-sample proportion so the external test cohort shares the same
        # one-sample/two-sample mixture as each calibration replicate.
        train_n_single = int(gen_kwargs.get("n_single_sample", DEFAULT_N_SINGLE_SAMPLE))
        test_gen_kwargs = dict(gen_kwargs)
        test_gen_kwargs["n_single_sample"] = min(
            n_test_participants,
            max(0, int(round(n_test_participants * (train_n_single / float(n_participants))))),
        )
        test_ds, _ = generate_synthetic_cohort(
            n_participants=n_test_participants,
            seed=base_seed,
            compact_grid=compact_grid,
            **test_gen_kwargs,
        )
        prepared_test = PreparedExternalTest.from_dataset(
            test_ds,
            eff_cfg.primary_endpoint,
            preset=eff_cfg.covariate_preset,
        )
        n_ext = prepared_test.n_participants
        del test_ds

        use_parallel = (
            not gpu_active
            and n_jobs != 1
            and len(pending_indices) > 1
            and (len(pending_indices) * max(n_bootstraps, 1) >= 200)
        )
        if ckpt_file is None:
            chunk_size = len(pending_indices)
        elif checkpoint_chunk_size is not None and checkpoint_chunk_size >= 1:
            chunk_size = int(checkpoint_chunk_size)
        else:
            chunk_size = max(16, (os.cpu_count() or 4) * 2) if use_parallel else 10

        parallel_cls: Any = None
        delayed_fn: Any = None
        if use_parallel:
            from joblib import Parallel, delayed

            parallel_cls = Parallel
            delayed_fn = delayed

        for c_start in range(0, len(pending_indices), chunk_size):
            chunk_idxs = pending_indices[c_start : c_start + chunk_size]
            if use_parallel and len(chunk_idxs) > 1:
                chunk_results: list[_CalibrationRepResult] = parallel_cls(n_jobs=n_jobs)(
                    delayed_fn(_eval_single_calibration_rep)(
                        int(rep_seeds[rep_idx]),
                        n_participants,
                        gen_kwargs,
                        n_bootstraps,
                        eff_cfg,
                        False,
                        prepared_test,
                        compact_grid,
                        interval_method,
                    )
                    for rep_idx in chunk_idxs
                )
            else:
                chunk_results = [
                    _eval_single_calibration_rep(
                        int(rep_seeds[rep_idx]),
                        n_participants,
                        gen_kwargs,
                        n_bootstraps,
                        eff_cfg,
                        effective_use_gpu,
                        prepared_test,
                        compact_grid,
                        interval_method,
                    )
                    for rep_idx in chunk_idxs
                ]

            for rep_idx, rep_res in zip(chunk_idxs, chunk_results):
                state["completed"][rep_idx] = True
                state["point_delta_mse"][rep_idx] = rep_res.point_delta_mse
                state["ci_lo_delta_mse"][rep_idx] = rep_res.ci_lo_delta_mse
                state["ci_hi_delta_mse"][rep_idx] = rep_res.ci_hi_delta_mse
                state["succeeded"][rep_idx] = rep_res.succeeded
                state["ext_delta_mse"][rep_idx] = rep_res.ext_delta_mse
                if rep_res.ext_part_deltas_mse.size == n_ext:
                    row_m = rep_res.ext_part_deltas_mse
                    state["ext_col_sum_mse"] += row_m
                    state["ext_row_ss_mse"][rep_idx] = float(np.sum((row_m - float(np.mean(row_m))) ** 2))
                else:
                    state["ext_part_valid_mse"] = np.array(False, dtype=bool)

                state["point_vs_comp"][rep_idx] = rep_res.point_vs_comp
                state["ci_lo_vs_comp"][rep_idx] = rep_res.ci_lo_vs_comp
                state["ci_hi_vs_comp"][rep_idx] = rep_res.ci_hi_vs_comp
                state["ext_delta_vs_comp"][rep_idx] = rep_res.ext_delta_vs_comp
                if rep_res.ext_part_deltas_vs_comp.size == n_ext:
                    row_c = rep_res.ext_part_deltas_vs_comp
                    state["ext_col_sum_comp"] += row_c
                    state["ext_row_ss_comp"][rep_idx] = float(np.sum((row_c - float(np.mean(row_c))) ** 2))
                else:
                    state["ext_part_valid_comp"] = np.array(False, dtype=bool)

            if ckpt_file is not None:
                _save_calibration_checkpoint(
                    ckpt_file,
                    {"fingerprint": np.array(fingerprint), **state},
                )

    succeeded_flags = [bool(v) for v in state["succeeded"]]
    prim = _summarize_interval_calibration(
        point_estimates=state["point_delta_mse"].tolist(),
        ci_lows=state["ci_lo_delta_mse"].tolist(),
        ci_highs=state["ci_hi_delta_mse"].tolist(),
        succeeded_flags=succeeded_flags,
        external_deltas=state["ext_delta_mse"],
        ext_col_sums=state["ext_col_sum_mse"] if bool(state["ext_part_valid_mse"].item()) else None,
        ext_row_ss_within=state["ext_row_ss_mse"] if bool(state["ext_part_valid_mse"].item()) else None,
        n_ext=n_ext,
        gate_wilson_lower_min=gate_wilson_lower_min,
        gate_max_failure_rate=gate_max_failure_rate,
    )
    comp = _summarize_interval_calibration(
        point_estimates=state["point_vs_comp"].tolist(),
        ci_lows=state["ci_lo_vs_comp"].tolist(),
        ci_highs=state["ci_hi_vs_comp"].tolist(),
        succeeded_flags=succeeded_flags,
        external_deltas=state["ext_delta_vs_comp"],
        ext_col_sums=state["ext_col_sum_comp"] if bool(state["ext_part_valid_comp"].item()) else None,
        ext_row_ss_within=state["ext_row_ss_comp"] if bool(state["ext_part_valid_comp"].item()) else None,
        n_ext=n_ext,
        gate_wilson_lower_min=gate_wilson_lower_min,
        gate_max_failure_rate=gate_max_failure_rate,
    )

    return {
        "seed_set": seed_set,
        "interval_method": str(interval_method),
        "gate_is_confirmatory": seed_set == "validation",
        "n_repetitions": n_repetitions,
        "n_bootstraps": n_bootstraps,
        "gpu_requested": use_gpu,
        "gpu_active": gpu_active,
        "gpu_fallback": bool(gpu_meta["gpu_fallback"]),
        "target_delta_mse": prim["target_delta"],
        "target_monte_carlo_se": prim["target_mc_se"],
        "target_test_cohort_se": prim["target_test_se"],
        "target_total_se": prim["target_total_se"],
        "n_test_participants": n_test_participants,
        "compact_grid": compact_grid,
        "mean_point_estimate": prim["mean_point_estimate"],
        "empirical_sd_point_estimate": prim["empirical_sd_point_estimate"],
        "bias": prim["bias"],
        "unconditional_coverage": prim["unconditional_coverage"],
        "conditional_coverage": prim["conditional_coverage"],
        "lower_noncoverage_rate": prim["lower_noncoverage_rate"],
        "upper_noncoverage_rate": prim["upper_noncoverage_rate"],
        "target_sensitivity_min_coverage": prim["target_sensitivity_min_coverage"],
        "wilson_95_lower": prim["wilson_95_lower"],
        "wilson_95_upper": prim["wilson_95_upper"],
        "dataset_failure_rate": prim["dataset_failure_rate"],
        "mean_interval_width": prim["mean_interval_width"],
        "detection_rate": prim["detection_rate"],
        "detection_wilson_95_lower": prim["detection_wilson_95_lower"],
        "detection_wilson_95_upper": prim["detection_wilson_95_upper"],
        "passes_provisional_gate": prim["passes_provisional_gate"],
        "target_delta_vs_comparator": comp["target_delta"],
        "target_monte_carlo_se_vs_comparator": comp["target_mc_se"],
        "target_test_cohort_se_vs_comparator": comp["target_test_se"],
        "target_total_se_vs_comparator": comp["target_total_se"],
        "mean_point_estimate_vs_comparator": comp["mean_point_estimate"],
        "empirical_sd_point_estimate_vs_comparator": comp["empirical_sd_point_estimate"],
        "bias_vs_comparator": comp["bias"],
        "unconditional_coverage_vs_comparator": comp["unconditional_coverage"],
        "conditional_coverage_vs_comparator": comp["conditional_coverage"],
        "lower_noncoverage_rate_vs_comparator": comp["lower_noncoverage_rate"],
        "upper_noncoverage_rate_vs_comparator": comp["upper_noncoverage_rate"],
        "target_sensitivity_min_coverage_vs_comparator": comp["target_sensitivity_min_coverage"],
        "wilson_95_lower_vs_comparator": comp["wilson_95_lower"],
        "wilson_95_upper_vs_comparator": comp["wilson_95_upper"],
        "mean_interval_width_vs_comparator": comp["mean_interval_width"],
        "detection_rate_vs_comparator": comp["detection_rate"],
        "passes_provisional_gate_vs_comparator": comp["passes_provisional_gate"],
        "gate_attainable": gate_attainable,
        "min_reps_for_gate": min_reps,
    }


def evaluate_scenario_calibration_suite(
    *,
    scenarios: Sequence[Mapping[str, Any]],
    n_repetitions: int = 20,
    n_bootstraps: int = 100,
    n_participants: int = 92,
    base_seed: int = 9000,
    cfg: PairedLipidConfig = PairedLipidConfig(),
    n_jobs: int = -1,
    use_gpu: bool | None = None,
    n_test_participants: int = 25000,
    seed_set: SeedSet = "development",
    compact_grid: bool = True,
    interval_method: IntervalMethod = "studentized_t",
    gate_wilson_lower_min: float = 0.925,
    gate_max_failure_rate: float = 0.01,
    checkpoint_dir: str | Path | None = None,
    checkpoint_chunk_size: int | None = None,
    base_seed_offsets: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Evaluate bootstrap interval calibration across multiple synthetic data-generating scenarios."""
    min_rep_seed = CALIBRATION_SEED_RANGES["development"][0]
    offsets = list(range(len(scenarios))) if base_seed_offsets is None else [int(o) for o in base_seed_offsets]
    if len(offsets) != len(scenarios):
        raise ValueError(
            f"base_seed_offsets length ({len(offsets)}) must match scenarios length ({len(scenarios)})."
        )
    if base_seed < 0 or any(base_seed + off < 0 or base_seed + off >= min_rep_seed for off in offsets):
        raise ValueError(
            f"All scenario base seeds from base_seed={base_seed} and offsets={offsets} must lie in "
            f"[0, {min_rep_seed}) to stay disjoint from replicate seeds."
        )
    ckpt_root = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if ckpt_root is not None:
        ckpt_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for s_idx, (spec, seed_off) in enumerate(zip(scenarios, offsets)):
        scenario_name = str(spec.get("scenario", f"scenario_{seed_off}"))
        preset_name = str(spec.get("covariate_preset", cfg.covariate_preset))
        sig = float(spec.get("signal_strength", 0.8))
        scenario_ckpt: Path | None = None
        if ckpt_root is not None:
            safe_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", scenario_name).strip("_") or f"scenario_{seed_off}"
            scenario_ckpt = ckpt_root / f"scenario_{seed_off:02d}_{safe_slug}.npz"
        summary = evaluate_bootstrap_calibration(
            n_repetitions=n_repetitions,
            n_bootstraps=n_bootstraps,
            n_participants=n_participants,
            signal_strength=sig,
            base_seed=base_seed + seed_off,
            cfg=cfg,
            n_jobs=n_jobs,
            use_gpu=use_gpu,
            n_test_participants=n_test_participants,
            seed_set=seed_set,
            compact_grid=compact_grid,
            interval_method=interval_method,
            gate_wilson_lower_min=gate_wilson_lower_min,
            gate_max_failure_rate=gate_max_failure_rate,
            scenario_kwargs=spec,
            checkpoint_path=scenario_ckpt,
            checkpoint_chunk_size=checkpoint_chunk_size,
        )
        rows.append(
            {
                "scenario": scenario_name,
                "covariate_preset": preset_name,
                **summary,
            }
        )
        if ckpt_root is not None:
            write_csv_atomic(pd.DataFrame(rows), ckpt_root / "partial_calibration_report.csv", index=False)
    return pd.DataFrame(rows)


def evaluate_mde_signal_curve(
    *,
    signal_grid: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    n_repetitions: int = 48,
    n_bootstraps: int = 250,
    n_participants: int = 92,
    base_seed: int = 9500,
    cfg: PairedLipidConfig = PairedLipidConfig(),
    n_jobs: int = -1,
    use_gpu: bool | None = None,
    n_test_participants: int = 25000,
    seed_set: SeedSet = "development",
    compact_grid: bool = True,
    interval_method: IntervalMethod = "studentized_t",
    gate_wilson_lower_min: float = 0.925,
    gate_max_failure_rate: float = 0.01,
    scenario_kwargs: Mapping[str, Any] | None = None,
    checkpoint_dir: str | Path | None = None,
    checkpoint_chunk_size: int | None = None,
) -> pd.DataFrame:
    """Evaluate pre-specified detectable-effect (MDE) curve over a locked signal-strength grid."""
    if not signal_grid:
        raise ValueError("signal_grid must be non-empty.")
    min_rep_seed = CALIBRATION_SEED_RANGES["development"][0]
    if base_seed < 0 or base_seed + len(signal_grid) > min_rep_seed:
        raise ValueError(
            f"MDE curve base_seed ({base_seed}) + grid length ({len(signal_grid)}) must stay in [0, {min_rep_seed})."
        )
    ckpt_root = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if ckpt_root is not None:
        ckpt_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for idx, sig_raw in enumerate(signal_grid):
        sig = float(sig_raw)
        if not math.isfinite(sig) or sig < 0.0:
            raise ValueError(f"signal_grid entries must be finite and non-negative; got {sig_raw!r}.")
        spec = dict(scenario_kwargs) if scenario_kwargs is not None else {}
        spec["signal_strength"] = sig
        mde_ckpt = (ckpt_root / f"mde_{idx:02d}_sig_{str(sig).replace('.', 'p')}.npz") if ckpt_root else None
        cal = evaluate_bootstrap_calibration(
            n_repetitions=n_repetitions,
            n_bootstraps=n_bootstraps,
            n_participants=n_participants,
            signal_strength=sig,
            base_seed=base_seed + idx,
            cfg=cfg,
            n_jobs=n_jobs,
            use_gpu=use_gpu,
            n_test_participants=n_test_participants,
            seed_set=seed_set,
            compact_grid=compact_grid,
            interval_method=interval_method,
            gate_wilson_lower_min=gate_wilson_lower_min,
            gate_max_failure_rate=gate_max_failure_rate,
            scenario_kwargs=spec,
            checkpoint_path=mde_ckpt,
            checkpoint_chunk_size=checkpoint_chunk_size,
        )
        rows.append(
            {
                "signal_strength": sig,
                "n_participants": int(n_participants),
                "n_repetitions": int(n_repetitions),
                "n_bootstraps": int(n_bootstraps),
                "interval_method": str(interval_method),
                "seed_set": str(seed_set),
                "compact_grid": bool(compact_grid),
                "target_delta_mse": cal["target_delta_mse"],
                "target_total_se": cal["target_total_se"],
                "mean_delta_mse": cal["mean_point_estimate"],
                "sd_delta_mse": cal["empirical_sd_point_estimate"],
                "unconditional_coverage": cal["unconditional_coverage"],
                "detection_rate_delta_mse": cal["detection_rate"],
                "detection_wilson_95_lower": cal["detection_wilson_95_lower"],
                "detection_wilson_95_upper": cal["detection_wilson_95_upper"],
                "mean_interval_width_delta_mse": cal["mean_interval_width"],
                "target_delta_vs_comparator": cal["target_delta_vs_comparator"],
                "mean_delta_vs_comparator": cal["mean_point_estimate_vs_comparator"],
                "detection_rate_vs_comparator": cal["detection_rate_vs_comparator"],
                "dataset_failure_rate": cal["dataset_failure_rate"],
            }
        )
    return pd.DataFrame(rows)


def _ldl_factor_recovery_r2(eval_res: dict[str, Any], z_ldl: np.ndarray) -> float:
    """Mean over outer folds of the held-out R^2 of the true LDL-C factor on the learned paired scores."""
    r2s: list[float] = []
    for te_idx, scores in eval_res["fold_test_paired_scores"]:
        if te_idx.size == 0:
            continue
        y = z_ldl[te_idx]
        X = np.column_stack([np.ones(te_idx.size, dtype=float), scores])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2s.append(1.0 - float(np.sum((y - X @ beta) ** 2)) / ss_tot if ss_tot > 0.0 else float("nan"))
    if not r2s:
        return float("nan")
    arr = np.asarray(r2s, dtype=float)
    finite = arr[np.isfinite(arr)]
    return float(np.mean(finite)) if finite.size > 0 else float("nan")


class _ScenarioRepResult(NamedTuple):
    """Per-repetition summary metrics for a synthetic scenario evaluation."""

    delta_mse_paired: float
    delta_mse_pca: float
    delta_mse_pls: float
    delta_mse_fixed_band: float
    delta_vs_strongest: float
    mean_spectral_icc: float | None
    recovery_r2_ldl_factor: float
    corr_ldl_tg: float


def _eval_single_scenario_rep(
    seed: int,
    n_participants: int,
    compact_grid: bool,
    spec: Mapping[str, Any],
    cfg: PairedLipidConfig,
) -> _ScenarioRepResult:
    """Evaluate one synthetic scenario replicate on the locked nested-CV pipeline."""
    gen_kwargs = {k: v for k, v in spec.items() if k not in _SCENARIO_META_KEYS}
    ds, truth = generate_synthetic_cohort(
        n_participants=n_participants, seed=seed, compact_grid=compact_grid, **gen_kwargs
    )
    res = evaluate_nested_cohort(ds, cfg=cfg)
    comp = res["comparisons_vs_covariate_ridge"]
    fold_iccs = [
        float(r["paired_spectral_icc"]["icc_a1"])
        for r in res["fold_repeatability"]
        if "paired_spectral_icc" in r and r["paired_spectral_icc"]["estimable"]
    ]
    mean_icc = float(np.mean(fold_iccs)) if fold_iccs else None
    return _ScenarioRepResult(
        delta_mse_paired=float(comp["paired_linear_ridge"]["delta_mse"]),
        delta_mse_pca=float(comp["pca_ridge"]["delta_mse"]),
        delta_mse_pls=float(comp["pls_regression"]["delta_mse"]),
        delta_mse_fixed_band=float(comp["fixed_band_ridge"]["delta_mse"]),
        delta_vs_strongest=float(res["paired_vs_strongest_comparator"]["delta_mse"]),
        mean_spectral_icc=mean_icc,
        recovery_r2_ldl_factor=_ldl_factor_recovery_r2(res, truth["z_bio"][:, 0]),
        corr_ldl_tg=float(np.corrcoef(ds.outcomes["ldl_mmol_l"], ds.outcomes["triglycerides_mmol_l"])[0, 1]),
    )


def run_synthetic_scenario_suite(
    *,
    scenarios: Sequence[Mapping[str, Any]],
    n_participants_grid: Sequence[int],
    n_repetitions: int = 5,
    compact_grid: bool = True,
    base_seed: int = 1000,
    n_jobs: int = -1,
    cfg: PairedLipidConfig = PairedLipidConfig(),
) -> pd.DataFrame:
    """Evaluate every configured synthetic scenario at every participant count across repetitions."""
    tasks = [
        (
            s_idx,
            n_idx,
            base_seed + 1_000_000 * n_idx + 10_000 * s_idx + rep,
            int(n_participants_grid[n_idx]),
            dict(spec),
            replace(cfg, covariate_preset=str(spec["covariate_preset"])),
        )
        for n_idx in range(len(n_participants_grid))
        for s_idx, spec in enumerate(scenarios)
        for rep in range(n_repetitions)
    ]

    if n_jobs > 1 and len(tasks) > 1:
        eff_jobs = min(int(n_jobs), len(tasks))
    elif n_jobs < 0 and (len(tasks) >= 32 or (not compact_grid and len(tasks) >= 8)):
        eff_jobs = min(8, max(1, (os.cpu_count() or 4) // 2), len(tasks))
    else:
        eff_jobs = 1

    if eff_jobs > 1:
        from joblib import Parallel, delayed

        task_results: list[_ScenarioRepResult] = Parallel(n_jobs=eff_jobs)(
            delayed(_eval_single_scenario_rep)(seed, n_p, compact_grid, spec, task_cfg)
            for _, _, seed, n_p, spec, task_cfg in tasks
        )
    else:
        task_results = [
            _eval_single_scenario_rep(seed, n_p, compact_grid, spec, task_cfg)
            for _, _, seed, n_p, spec, task_cfg in tasks
        ]

    grouped: dict[tuple[int, int], list[_ScenarioRepResult]] = {}
    for (s_idx, n_idx, *_), result in zip(tasks, task_results):
        grouped.setdefault((n_idx, s_idx), []).append(result)

    rows: list[dict[str, Any]] = []
    for (n_idx, s_idx), s_res in grouped.items():
        spec = scenarios[s_idx]
        deltas_paired = [r.delta_mse_paired for r in s_res]
        spec_iccs = [r.mean_spectral_icc for r in s_res if r.mean_spectral_icc is not None]
        rec_r2s = [r.recovery_r2_ldl_factor for r in s_res if np.isfinite(r.recovery_r2_ldl_factor)]
        rows.append(
            {
                "scenario": spec["scenario"],
                "covariate_preset": spec["covariate_preset"],
                "n_repetitions": n_repetitions,
                "n_participants": int(n_participants_grid[n_idx]),
                "mean_delta_mse_paired": float(np.mean(deltas_paired)),
                "sd_delta_mse_paired": float(np.std(deltas_paired, ddof=1)) if n_repetitions > 1 else 0.0,
                "mean_delta_vs_strongest_comparator": float(np.mean([r.delta_vs_strongest for r in s_res])),
                "mean_delta_mse_pca": float(np.mean([r.delta_mse_pca for r in s_res])),
                "mean_delta_mse_pls": float(np.mean([r.delta_mse_pls for r in s_res])),
                "mean_delta_mse_fixed_band": float(np.mean([r.delta_mse_fixed_band for r in s_res])),
                "mean_spectral_icc_a1": float(np.mean(spec_iccs)) if spec_iccs else float("nan"),
                "mean_recovery_r2_ldl_factor": float(np.mean(rec_r2s)) if rec_r2s else float("nan"),
                "mean_corr_ldl_tg": float(np.mean([r.corr_ldl_tg for r in s_res])),
            }
        )
    return pd.DataFrame(rows)
