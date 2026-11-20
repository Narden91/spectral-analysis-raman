"""Clinical-master linkage, paired cohort construction, band extraction and limitations table."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, get_args

import numpy as np
import pandas as pd

from src.analysis.raman_bands import RAMAN_BANDS
from src.metadata.clinical_master import (
    IDENTITY_CONFLICT_FLAGS as IDENTITY_CONFLICT_FLAGS,
)
from src.metadata.clinical_master import (
    MISSING_TOKENS as MISSING_TOKENS,
)
from src.metadata.clinical_master import (
    load_clinical_master_verified as load_clinical_master_verified,
)

SubsetMode = Literal["all_eligible", "paired_only", "single_sample", "three_map_complete"]
SUBSET_MODES: tuple[SubsetMode, ...] = get_args(SubsetMode)

COVARIATE_PRESETS: tuple[str, ...] = (
    "primary",
    "no_label_trend",
    "quadratic_label_trend",
    "quadratic_age_and_label",
    "pre_event_risk_factors",
)

PRE_EVENT_RISK_FACTORS: tuple[str, ...] = (
    "hypertension",
    "diabetes",
    "atrial_fibrillation",
    "coronary_artery_disease",
    "smoking_status",
    "antiplatelet",
    "anticoagulant",
)

_SAMPLE_PREFIX_RE = re.compile(r"^(?:sample[_\-\s]*)?", re.IGNORECASE)
_NUMERIC_TOKEN_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _coerce_int_tuple(values: Any, field_name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{field_name} must be a sequence of integers, not a string.")
    out: list[int] = []
    for item in values:
        if isinstance(item, bool):
            raise ValueError(f"{field_name} must contain finite integers.")
        if isinstance(item, float) and (not math.isfinite(item) or not item.is_integer()):
            raise ValueError(f"{field_name} must contain finite integers.")
        out.append(int(item))
    return tuple(out)


def _coerce_float_tuple(values: Any, field_name: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{field_name} must be a sequence of floats, not a string.")
    out: list[float] = []
    for item in values:
        if isinstance(item, bool):
            raise ValueError(f"{field_name} must contain finite numeric values.")
        val = float(item)
        if not math.isfinite(val):
            raise ValueError(f"{field_name} must contain finite numeric values.")
        out.append(val)
    return tuple(out)


def _coerce_str_tuple(values: Any, field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{field_name} must be a sequence of strings, not a string.")
    return tuple(str(item) for item in values)


def _is_strictly_ascending(values: Sequence[int] | Sequence[float]) -> bool:
    return all(a < b for a, b in zip(values, values[1:]))


@dataclass(frozen=True, slots=True)
class PairedLipidConfig:
    """Locked configuration for paired spectral representation learning."""

    pca_dimensions: tuple[int, ...] = (5, 10, 15)
    retained_directions: tuple[int, ...] = (1, 2, 3)
    lambda_grid: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0, 10.0)
    ridge_alpha_grid: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
    pls_component_grid: tuple[int, ...] = (1, 2, 3, 4, 5)
    n_outer_splits: int = 5
    n_inner_splits: int = 3
    primary_seed: int = 42
    sensitivity_seeds: tuple[int, ...] = (101, 202, 303)
    min_maps_per_block: int = 2
    rank_tolerance: float = 1e-8
    eigenvalue_tolerance: float = 1e-10
    primary_endpoint: str = "ldl_mmol_l"
    secondary_endpoints: tuple[str, ...] = (
        "hdl_mmol_l",
        "triglycerides_mmol_l",
        "plt_10e9_l",
    )
    covariate_preset: str = "primary"
    local_candidate_window: int = 5

    def __post_init__(self) -> None:
        pca_dims = _coerce_int_tuple(self.pca_dimensions, "pca_dimensions")
        ret_dirs = _coerce_int_tuple(self.retained_directions, "retained_directions")
        lam_grid = _coerce_float_tuple(self.lambda_grid, "lambda_grid")
        alpha_grid = _coerce_float_tuple(self.ridge_alpha_grid, "ridge_alpha_grid")
        pls_grid = _coerce_int_tuple(self.pls_component_grid, "pls_component_grid")
        sens_seeds = _coerce_int_tuple(self.sensitivity_seeds, "sensitivity_seeds")
        sec_eps = _coerce_str_tuple(self.secondary_endpoints, "secondary_endpoints")

        object.__setattr__(self, "pca_dimensions", pca_dims)
        object.__setattr__(self, "retained_directions", ret_dirs)
        object.__setattr__(self, "lambda_grid", lam_grid)
        object.__setattr__(self, "ridge_alpha_grid", alpha_grid)
        object.__setattr__(self, "pls_component_grid", pls_grid)
        object.__setattr__(self, "sensitivity_seeds", sens_seeds)
        object.__setattr__(self, "secondary_endpoints", sec_eps)

        if self.covariate_preset not in COVARIATE_PRESETS:
            raise ValueError(f"Unknown covariate preset: {self.covariate_preset!r}")
        if not pca_dims or any(d < 1 for d in pca_dims):
            raise ValueError("pca_dimensions must contain positive integers.")
        if not _is_strictly_ascending(pca_dims):
            raise ValueError("pca_dimensions must be strictly ascending without duplicates.")
        if not ret_dirs or any(r < 1 for r in ret_dirs):
            raise ValueError("retained_directions must contain positive integers.")
        if not _is_strictly_ascending(ret_dirs):
            raise ValueError("retained_directions must be strictly ascending without duplicates.")
        if not lam_grid or any(lam < 0.0 for lam in lam_grid):
            raise ValueError("lambda_grid must contain finite non-negative values.")
        if not _is_strictly_ascending(lam_grid):
            raise ValueError("lambda_grid must be strictly ascending without duplicates.")
        if not alpha_grid or any(a <= 0.0 for a in alpha_grid):
            raise ValueError("ridge_alpha_grid must contain finite strictly positive values.")
        if not _is_strictly_ascending(alpha_grid):
            raise ValueError("ridge_alpha_grid must be strictly ascending without duplicates.")
        if not pls_grid or any(c < 1 for c in pls_grid):
            raise ValueError("pls_component_grid must contain positive integers.")
        if not _is_strictly_ascending(pls_grid):
            raise ValueError("pls_component_grid must be strictly ascending without duplicates.")
        if self.n_outer_splits < 2 or self.n_inner_splits < 2:
            raise ValueError("n_outer_splits and n_inner_splits must be at least 2.")
        if self.min_maps_per_block < 1 or self.local_candidate_window < 1:
            raise ValueError("min_maps_per_block and local_candidate_window must be at least 1.")
        if (
            not math.isfinite(self.rank_tolerance)
            or self.rank_tolerance <= 0.0
            or not math.isfinite(self.eigenvalue_tolerance)
            or self.eigenvalue_tolerance <= 0.0
        ):
            raise ValueError("rank_tolerance and eigenvalue_tolerance must be finite and positive.")


@dataclass(frozen=True, slots=True)
class SampleRecord:
    """Aggregated spectral summary for one eligible technical sample."""

    participant_id: str
    sample_id: str
    parsed_label: float
    spec_1300: np.ndarray
    spec_2850: np.ndarray


@dataclass(slots=True)
class CohortDataset:
    """Structured participant- and sample-level dataset ready for nested evaluation."""

    participant_ids: np.ndarray
    outcomes: dict[str, np.ndarray]
    covariate_frame: pd.DataFrame
    wavenumbers_1300: np.ndarray
    wavenumbers_2850: np.ndarray
    sample_1300: np.ndarray
    sample_2850: np.ndarray
    sample_participant_idx: np.ndarray
    sample_ids: np.ndarray
    sample_parsed_labels: np.ndarray
    participant_sample_counts: np.ndarray
    paired_participant_indices: np.ndarray
    pair_sample_a_indices: np.ndarray
    pair_sample_b_indices: np.ndarray
    manifest_summary: dict[str, Any] = field(default_factory=dict)
    exclusions: list[dict[str, Any]] = field(default_factory=list)
    _cache: dict[str, Any] = field(default_factory=dict, init=False, repr=False, compare=False)

    @property
    def n_participants(self) -> int:
        return int(self.participant_ids.size)

    @property
    def n_samples(self) -> int:
        return int(self.sample_participant_idx.size)


@dataclass(slots=True)
class _AssembledSamples:
    """Intermediate participant-aligned arrays produced during cohort assembly."""

    kept_clin_indices: list[int]
    flat_1300: list[np.ndarray]
    flat_2850: list[np.ndarray]
    flat_p_idx: list[int]
    flat_s_ids: list[str]
    flat_lbls: list[float]
    sample_counts: list[int]
    paired_p_indices: list[int]
    pair_a_indices: list[int]
    pair_b_indices: list[int]
    mean_labels: list[float]


@lru_cache(maxsize=4096)
def parse_sample_label(raw_label: object) -> float:
    """Deterministically parse a numeric sample label from a slide/sample identifier."""
    if raw_label is None or (isinstance(raw_label, float) and np.isnan(raw_label)):
        return float("nan")
    cleaned = (
        str(raw_label)
        .strip()
        .replace("\u2019", "")
        .replace("\u2018", "")
        .replace("''", "")
        .replace("'", "")
    )
    cleaned = _SAMPLE_PREFIX_RE.sub("", cleaned).strip()
    if not cleaned:
        return float("nan")
    try:
        return float(cleaned)
    except ValueError:
        match = _NUMERIC_TOKEN_RE.search(cleaned)
        return float(match.group(0)) if match is not None else float("nan")


def _filter_clinical_records(
    clinical_df: pd.DataFrame,
    target_col: str,
    exclusions: list[dict[str, Any]],
) -> tuple[pd.DataFrame, str]:
    """Filter eligible stroke clinical records in one boolean pass while recording exclusions."""
    n_rows = len(clinical_df)
    keep = np.ones(n_rows, dtype=bool)
    if "cohort" in clinical_df.columns:
        keep &= (clinical_df["cohort"].astype(str) == "stroke").to_numpy()

    id_col = "spectral_subject_id" if "spectral_subject_id" in clinical_df.columns else "participant_id"
    ids_str = clinical_df[id_col].astype(str)
    ids_arr = ids_str.to_numpy()

    if "identity_verified" in clinical_df.columns:
        verified = clinical_df["identity_verified"].astype(bool).to_numpy()
        rejected_id = keep & ~verified
        if np.any(rejected_id):
            if "has_spectra" in clinical_df.columns:
                has_spec = (clinical_df["has_spectra"].fillna(0).astype(int) == 1).to_numpy()
            else:
                has_spec = np.ones(n_rows, dtype=bool)
            keys_arr = (
                clinical_df["participant_key"].to_numpy()
                if "participant_key" in clinical_df.columns
                else ids_arr
            )
            for pid, key, spec_ok in zip(ids_arr[rejected_id], keys_arr[rejected_id], has_spec[rejected_id]):
                # Records without spectra have no spectral id; report the clinical key instead.
                exclusions.append(
                    {
                        "participant_id": str(pid) if spec_ok else str(key),
                        "level": "participant",
                        "reason": "identity_review_required" if spec_ok else "no_spectra",
                    }
                )
        keep &= verified
    elif "spectral_link_status" in clinical_df.columns:
        matched = (clinical_df["spectral_link_status"].astype(str) == "matched").to_numpy()
        rejected_link = keep & ~matched
        if np.any(rejected_link):
            for pid in ids_arr[rejected_link]:
                exclusions.append(
                    {"participant_id": str(pid), "level": "participant", "reason": "unmatched_spectral_link"}
                )
        keep &= matched

    target_valid = pd.to_numeric(clinical_df[target_col], errors="coerce").notna().to_numpy()
    rejected_target = keep & ~target_valid
    if np.any(rejected_target):
        for pid in ids_arr[rejected_target]:
            exclusions.append({"participant_id": str(pid), "level": "participant", "reason": f"missing_{target_col}"})
    keep &= target_valid

    clin = clinical_df.loc[keep].copy()
    clin[id_col] = ids_arr[keep]
    clin = clin.sort_values(id_col, kind="mergesort").reset_index(drop=True)
    return clin, id_col


def _fast_block_median(spectra: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Compute column-wise median, taking the fast np.median path when all entries are finite."""
    sub = spectra[idx]
    if np.isfinite(sub).all():
        return np.median(sub, axis=0)
    return np.nanmedian(sub, axis=0)


def _aggregate_sample_blocks(
    map_records: pd.DataFrame,
    spectra_1300: np.ndarray,
    spectra_2850: np.ndarray,
    *,
    min_maps: int,
    exclusions: list[dict[str, Any]],
) -> dict[str, list[SampleRecord]]:
    """Aggregate map-level spectra into deterministic per-participant sample records."""
    p_col = "spectral_subject_id" if "spectral_subject_id" in map_records.columns else "participant_id"
    s_col = "slide_number" if "slide_number" in map_records.columns else "sample_id"
    if "retained" in map_records.columns:
        retained_mask = map_records["retained"].astype(bool).to_numpy()
        active_maps = map_records.iloc[retained_mask]
    else:
        active_maps = map_records

    maps_keyed = pd.DataFrame(
        {
            "participant_id": active_maps[p_col].astype(str).to_numpy(),
            "sample_id": active_maps[s_col].astype(str).to_numpy(),
            "spectral_range": active_maps["spectral_range"].astype(str).to_numpy(),
            "row_index": active_maps["row_index"].to_numpy(dtype=int),
        }
    )

    sample_blocks: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for (pid, sid, rng), grp in maps_keyed.groupby(["participant_id", "sample_id", "spectral_range"], sort=False):
        sample_blocks.setdefault((str(pid), str(sid)), {})[f"idx_{rng}"] = grp["row_index"].to_numpy(dtype=int)

    empty_idx = np.empty(0, dtype=int)
    participant_samples: dict[str, list[SampleRecord]] = {}
    for (pid, sid), info in sorted(sample_blocks.items()):
        idx_1300 = info.get("idx_1300", empty_idx)
        idx_2850 = info.get("idx_2850", empty_idx)
        if idx_1300.size == 0 or idx_2850.size == 0:
            exclusions.append(
                {"participant_id": pid, "sample_id": sid, "level": "sample", "reason": "missing_spectral_block"}
            )
            continue
        if idx_1300.size < min_maps or idx_2850.size < min_maps:
            exclusions.append(
                {
                    "participant_id": pid,
                    "sample_id": sid,
                    "level": "sample",
                    "reason": f"fewer_than_{min_maps}_maps",
                }
            )
            continue
        med_1300 = _fast_block_median(spectra_1300, idx_1300)
        med_2850 = _fast_block_median(spectra_2850, idx_2850)
        if not (np.all(np.isfinite(med_1300)) and np.all(np.isfinite(med_2850))):
            exclusions.append(
                {"participant_id": pid, "sample_id": sid, "level": "sample", "reason": "nonfinite_sample_median"}
            )
            continue
        parsed_lbl = parse_sample_label(sid)
        participant_samples.setdefault(pid, []).append(
            SampleRecord(
                participant_id=pid,
                sample_id=sid,
                parsed_label=parsed_lbl,
                spec_1300=med_1300,
                spec_2850=med_2850,
            )
        )

    # Order samples within each participant deterministically by (parsed_label, sample_id).
    for pid, s_list in participant_samples.items():
        s_list.sort(key=lambda r: (float("inf") if np.isnan(r.parsed_label) else r.parsed_label, r.sample_id))
        if len(s_list) > 2:
            # Retain the first two deterministic technical samples if more than two exist.
            for dropped in s_list[2:]:
                exclusions.append(
                    {
                        "participant_id": pid,
                        "sample_id": dropped.sample_id,
                        "level": "sample",
                        "reason": "beyond_first_two_samples",
                    }
                )
            participant_samples[pid] = s_list[:2]

    return participant_samples


def _assemble_participant_samples(
    clin: pd.DataFrame,
    id_col: str,
    participant_samples: dict[str, list[SampleRecord]],
    *,
    subset_mode: SubsetMode,
    exclusions: list[dict[str, Any]],
) -> _AssembledSamples:
    """Join filtered clinical rows with eligible sample records and pair indices."""
    kept_clin_indices: list[int] = []
    flat_1300: list[np.ndarray] = []
    flat_2850: list[np.ndarray] = []
    flat_p_idx: list[int] = []
    flat_s_ids: list[str] = []
    flat_lbls: list[float] = []
    sample_counts: list[int] = []
    paired_p_indices: list[int] = []
    pair_a_indices: list[int] = []
    pair_b_indices: list[int] = []
    mean_labels: list[float] = []

    for row_pos, raw_pid in enumerate(clin[id_col].tolist()):
        pid = str(raw_pid)
        s_list = participant_samples.get(pid, [])
        if not s_list:
            exclusions.append({"participant_id": pid, "level": "participant", "reason": "no_eligible_samples"})
            continue
        if subset_mode == "paired_only" and len(s_list) < 2:
            exclusions.append(
                {"participant_id": pid, "level": "participant", "reason": "unpaired_in_paired_only_mode"}
            )
            continue

        p_idx = len(kept_clin_indices)
        kept_clin_indices.append(row_pos)
        sample_counts.append(len(s_list))

        first_sample_global_idx = len(flat_1300)
        for s_rec in s_list:
            flat_1300.append(s_rec.spec_1300)
            flat_2850.append(s_rec.spec_2850)
            flat_p_idx.append(p_idx)
            flat_s_ids.append(s_rec.sample_id)
            flat_lbls.append(s_rec.parsed_label)

        if len(s_list) == 2:
            paired_p_indices.append(p_idx)
            pair_a_indices.append(first_sample_global_idx)
            pair_b_indices.append(first_sample_global_idx + 1)

        # In single_sample sensitivity mode, the participant-level sample-label covariate uses the
        # first sample (A); paired samples remain in flat arrays so training divisions can still
        # fit W and T on paired training participants.
        active_samples = [s_list[0]] if subset_mode == "single_sample" else s_list
        finite_lbls = [sr.parsed_label for sr in active_samples if np.isfinite(sr.parsed_label)]
        mean_labels.append(float(np.mean(finite_lbls)) if finite_lbls else float("nan"))

    if not kept_clin_indices:
        raise ValueError("No eligible stroke participants remained after linkage and spectral checks.")

    return _AssembledSamples(
        kept_clin_indices=kept_clin_indices,
        flat_1300=flat_1300,
        flat_2850=flat_2850,
        flat_p_idx=flat_p_idx,
        flat_s_ids=flat_s_ids,
        flat_lbls=flat_lbls,
        sample_counts=sample_counts,
        paired_p_indices=paired_p_indices,
        pair_a_indices=pair_a_indices,
        pair_b_indices=pair_b_indices,
        mean_labels=mean_labels,
    )


def _build_covariate_and_outcome_frames(
    kept_df: pd.DataFrame,
    mean_labels: Sequence[float],
    cfg: PairedLipidConfig,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Construct the participant-level covariate DataFrame and outcome arrays."""
    cov_df = pd.DataFrame(index=kept_df.index)
    cov_df["age_years"] = pd.to_numeric(kept_df.get("age_years"), errors="coerce")
    cov_df["sex"] = kept_df.get("sex", pd.Series(np.nan, index=kept_df.index)).astype(object)
    cov_df["statin"] = pd.to_numeric(kept_df.get("statin"), errors="coerce")
    cov_df["sample_label_trend"] = np.asarray(mean_labels, dtype=float)

    for rf in PRE_EVENT_RISK_FACTORS:
        if rf in kept_df.columns:
            if rf == "smoking_status":
                cov_df[rf] = kept_df[rf].astype(object)
            else:
                cov_df[rf] = pd.to_numeric(kept_df[rf], errors="coerce")

    outcomes: dict[str, np.ndarray] = {}
    for ep in (cfg.primary_endpoint, *cfg.secondary_endpoints):
        if ep in kept_df.columns:
            outcomes[ep] = pd.to_numeric(kept_df[ep], errors="coerce").to_numpy(dtype=float)

    return cov_df, outcomes


def build_paired_cohort_dataset(
    clinical_df: pd.DataFrame,
    map_records: pd.DataFrame,
    spectra_1300: np.ndarray,
    spectra_2850: np.ndarray,
    wavenumbers_1300: np.ndarray,
    wavenumbers_2850: np.ndarray,
    *,
    cfg: PairedLipidConfig = PairedLipidConfig(),
    subset_mode: SubsetMode = "all_eligible",
) -> CohortDataset:
    """Construct the participant -> sample -> map hierarchy and covariate table.

    Parameters
    ----------
    clinical_df : pd.DataFrame
        Verified clinical master dataframe (or synthetic equivalent).
    map_records : pd.DataFrame
        Map-level metadata with columns: ``participant_id`` (or ``spectral_subject_id``),
        ``sample_id`` (or ``slide_number``), ``spectral_range`` ('1300' or '2850'),
        ``row_index`` (index into ``spectra_1300`` or ``spectra_2850``), and optional
        ``retained`` (bool). Acquisition modes (FP/SS) are pooled as equivalent (investigator
        decision, 2026-11-09), so they enter no covariate.
    """
    if subset_mode not in SUBSET_MODES:
        raise ValueError(f"Unknown subset_mode: {subset_mode!r}. Expected one of {SUBSET_MODES}.")

    target_col = cfg.primary_endpoint
    min_maps = 3 if subset_mode == "three_map_complete" else cfg.min_maps_per_block
    exclusions: list[dict[str, Any]] = []

    clin, id_col = _filter_clinical_records(clinical_df, target_col, exclusions)
    participant_samples = _aggregate_sample_blocks(
        map_records,
        spectra_1300,
        spectra_2850,
        min_maps=min_maps,
        exclusions=exclusions,
    )
    assembled = _assemble_participant_samples(
        clin,
        id_col,
        participant_samples,
        subset_mode=subset_mode,
        exclusions=exclusions,
    )

    kept_df = clin.iloc[assembled.kept_clin_indices].reset_index(drop=True)
    cov_df, outcomes = _build_covariate_and_outcome_frames(kept_df, assembled.mean_labels, cfg)
    counts_arr = np.asarray(assembled.sample_counts, dtype=int)

    return CohortDataset(
        participant_ids=kept_df[id_col].astype(str).to_numpy(),
        outcomes=outcomes,
        covariate_frame=cov_df,
        wavenumbers_1300=np.asarray(wavenumbers_1300, dtype=float),
        wavenumbers_2850=np.asarray(wavenumbers_2850, dtype=float),
        sample_1300=np.vstack(assembled.flat_1300),
        sample_2850=np.vstack(assembled.flat_2850),
        sample_participant_idx=np.asarray(assembled.flat_p_idx, dtype=int),
        sample_ids=np.asarray(assembled.flat_s_ids, dtype=str),
        sample_parsed_labels=np.asarray(assembled.flat_lbls, dtype=float),
        participant_sample_counts=counts_arr,
        paired_participant_indices=np.asarray(assembled.paired_p_indices, dtype=int),
        pair_sample_a_indices=np.asarray(assembled.pair_a_indices, dtype=int),
        pair_sample_b_indices=np.asarray(assembled.pair_b_indices, dtype=int),
        manifest_summary={
            "subset_mode": subset_mode,
            "min_maps_per_block": min_maps,
            "n_eligible_participants": len(assembled.kept_clin_indices),
            "n_paired_participants": len(assembled.paired_p_indices),
            "n_single_sample_participants": int(np.sum(counts_arr == 1)),
            "n_eligible_samples": len(assembled.flat_1300),
            "n_exclusions": len(exclusions),
        },
        exclusions=exclusions,
    )


@lru_cache(maxsize=128)
def _cached_band_slices(
    wn_1300_bytes: bytes,
    wn_2850_bytes: bytes,
) -> tuple[tuple[str, str, slice | np.ndarray], ...]:
    """Memoize channel selectors (contiguous slices where possible) for all 21 pre-specified Raman bands."""
    wn_1300 = np.frombuffer(wn_1300_bytes, dtype=np.float64)
    wn_2850 = np.frombuffer(wn_2850_bytes, dtype=np.float64)
    specs: list[tuple[str, str, slice | np.ndarray]] = []
    for name, band in RAMAN_BANDS.items():
        lower, upper = float(band.bounds[0]), float(band.bounds[1])
        block = "1300" if upper < 2000.0 else "2850"
        axis = wn_1300 if block == "1300" else wn_2850
        idx = np.flatnonzero((axis >= lower) & (axis <= upper))
        if idx.size > 0 and int(idx[-1]) - int(idx[0]) + 1 == idx.size:
            selector: slice | np.ndarray = slice(int(idx[0]), int(idx[-1]) + 1)
        else:
            idx.setflags(write=False)
            selector = idx
        specs.append((name, block, selector))
    return tuple(specs)


def extract_sample_band_matrix(dataset: CohortDataset) -> tuple[np.ndarray, list[str]]:
    """Compute the 21-band summary matrix for all samples with memoized zero-copy channel slicing."""
    cached = dataset._cache.get("sample_band_matrix")
    if cached is not None:
        mat_cached, names_cached = cached
        return mat_cached, list(names_cached)

    w1 = np.ascontiguousarray(dataset.wavenumbers_1300, dtype=np.float64).tobytes()
    w2 = np.ascontiguousarray(dataset.wavenumbers_2850, dtype=np.float64).tobytes()
    slices = _cached_band_slices(w1, w2)

    mat = np.empty((dataset.n_samples, len(slices)), dtype=np.float64)
    names: list[str] = []
    for j, (name, block, selector) in enumerate(slices):
        if isinstance(selector, np.ndarray) and selector.size == 0:
            raise ValueError(f"Band '{name}' has no channels in block {block} axis.")
        source = dataset.sample_1300 if block == "1300" else dataset.sample_2850
        np.mean(source[:, selector], axis=1, out=mat[:, j])
        names.append(name)

    mat.setflags(write=False)
    names_tuple = tuple(names)
    dataset._cache["sample_band_matrix"] = (mat, names_tuple)
    return mat, list(names_tuple)


def _get_participant_aggregation_plan(dataset: CohortDataset) -> dict[str, Any]:
    """Memoize sample-to-participant index gathers so repeated aggregations avoid unbuffered scatter."""
    plan = dataset._cache.get("_agg_plan")
    if plan is not None:
        return plan

    n_p = dataset.n_participants
    p_idx = dataset.sample_participant_idx
    counts = dataset.participant_sample_counts
    _, first_indices = np.unique(p_idx, return_index=True)
    single_p = np.flatnonzero(counts == 1)
    paired_p = dataset.paired_participant_indices
    fast_one_two = (
        first_indices.size == n_p
        and int(single_p.size + paired_p.size) == n_p
        and np.all((counts == 1) | (counts == 2))
    )
    plan = {
        "first_indices": first_indices,
        "fast_one_two": bool(fast_one_two),
        "single_p": single_p,
        "single_s": first_indices[single_p] if first_indices.size == n_p else np.empty(0, dtype=int),
        "paired_p": paired_p,
        "pair_a": dataset.pair_sample_a_indices,
        "pair_b": dataset.pair_sample_b_indices,
        "inv_counts": (1.0 / counts.astype(float))[:, None],
    }
    dataset._cache["_agg_plan"] = plan
    return plan


def aggregate_samples_to_participants(
    sample_matrix: np.ndarray,
    dataset: CohortDataset,
    *,
    single_sample_only: bool = False,
) -> np.ndarray:
    """Vectorized participant-level equal average of eligible sample representations."""
    plan = _get_participant_aggregation_plan(dataset)
    if single_sample_only:
        return sample_matrix[plan["first_indices"]].copy()

    if plan["fast_one_two"]:
        out = np.empty((dataset.n_participants, sample_matrix.shape[1]), dtype=float)
        if plan["single_p"].size > 0:
            out[plan["single_p"]] = sample_matrix[plan["single_s"]]
        if plan["paired_p"].size > 0:
            out[plan["paired_p"]] = 0.5 * (sample_matrix[plan["pair_a"]] + sample_matrix[plan["pair_b"]])
        return out

    sums = np.zeros((dataset.n_participants, sample_matrix.shape[1]), dtype=float)
    np.add.at(sums, dataset.sample_participant_idx, sample_matrix)
    return sums * plan["inv_counts"]


def _lookup_row_indices(query_ids: pd.Series, target_ids: np.ndarray) -> np.ndarray:
    """Map string measurement IDs to row positions in a summary table (-1 if absent)."""
    target_index = pd.Index(target_ids).astype(str)
    if target_index.is_unique:
        return np.asarray(target_index.get_indexer(query_ids), dtype=int)
    idx_map = {mid: i for i, mid in enumerate(target_index)}
    return query_ids.map(idx_map).fillna(-1).to_numpy(dtype=int)


def load_paired_cohort_from_store(
    clinical_master_path: str | Path = "data/clinical_master.csv",
    *,
    data_source: Any | None = None,
    cfg: PairedLipidConfig = PairedLipidConfig(),
    subset_mode: SubsetMode = "all_eligible",
) -> tuple[CohortDataset, dict[str, Any]]:
    """Load the stroke cohort hierarchy from the canonical preprocessed store and clinical master."""
    if subset_mode not in SUBSET_MODES:
        raise ValueError(f"Unknown subset_mode: {subset_mode!r}. Expected one of {SUBSET_MODES}.")

    from src.data_preparation.loaders import load_area_summaries, load_file_index

    clin_df, clin_audit = load_clinical_master_verified(clinical_master_path)
    file_index = load_file_index(data_source)
    if "is_control" in file_index.columns:
        non_control_mask = (~file_index["is_control"].fillna(False).astype(bool)).to_numpy()
        file_index = file_index.iloc[non_control_mask]

    raw_id_col = "subject_id_raw" if "subject_id_raw" in file_index.columns else "subject_id"
    participant_ids = file_index[raw_id_col].astype(str).str.strip()
    sample_ids = file_index["slide_number"].astype(str).str.strip()
    mid_series = file_index["measurement_id"].astype(str)
    rng_series = file_index["spectral_range"].astype(str)

    sum_1300 = load_area_summaries(data_source, "1300")
    sum_2850 = load_area_summaries(data_source, "2850")

    def _extract_wn(feature_names: np.ndarray) -> np.ndarray:
        return np.fromiter(
            (float(str(fn).rsplit("__", 1)[-1]) for fn in feature_names),
            dtype=float,
            count=len(feature_names),
        )

    wn_1300 = _extract_wn(sum_1300["feature_names"])
    wn_2850 = _extract_wn(sum_2850["feature_names"])

    row_idx = np.full(len(file_index), -1, dtype=int)
    mask_1300 = (rng_series == "1300").to_numpy()
    mask_2850 = (rng_series == "2850").to_numpy()
    if np.any(mask_1300):
        row_idx[mask_1300] = _lookup_row_indices(mid_series.iloc[mask_1300], sum_1300["measurement_ids"])
    if np.any(mask_2850):
        row_idx[mask_2850] = _lookup_row_indices(mid_series.iloc[mask_2850], sum_2850["measurement_ids"])

    valid_rows = row_idx >= 0
    map_df = pd.DataFrame(
        {
            "participant_id": participant_ids.iloc[valid_rows].to_numpy(dtype=str),
            "sample_id": sample_ids.iloc[valid_rows].to_numpy(dtype=str),
            "spectral_range": rng_series.iloc[valid_rows].to_numpy(dtype=str),
            "row_index": row_idx[valid_rows],
            "retained": np.ones(int(np.count_nonzero(valid_rows)), dtype=bool),
        }
    )

    dataset = build_paired_cohort_dataset(
        clin_df,
        map_df,
        np.asarray(sum_1300["area_summary"], dtype=float),
        np.asarray(sum_2850["area_summary"], dtype=float),
        wn_1300,
        wn_2850,
        cfg=cfg,
        subset_mode=subset_mode,
    )
    return dataset, clin_audit


def build_limitations_table(clin_audit: Mapping[str, Any] | None = None) -> pd.DataFrame:
    """Return a structured limitations table tied to source metadata and MISEV2023 requirements."""
    rows = [
        {
            "domain": "retrospective_design",
            "status": "documented_constraint",
            "source_reference": "docs/research_design.md; AGENT_LEARNINGS.md",
            "implication": (
                "Earlier cohort results were known when the design was written; nested validation "
                "reduces internal overfitting but cannot remove prior researcher adaptation."
            ),
        },
        {
            "domain": "sample_identity_and_linkage",
            "status": (
                f"{clin_audit.get('stroke_identity_excluded', 'unverified')} stroke records excluded for identity review"
                if clin_audit
                else "requires_clinical_master_verification"
            ),
            "source_reference": "data/clinical_master.csv; data/clinical_master_notes.md",
            "implication": (
                "Records with SEX_FOLDER_CONFLICT or filename/folder mismatches are excluded rather "
                "than silently relinked."
            ),
        },
        {
            "domain": "laboratory_units_and_assay",
            "status": "measurement_definition_requires_source_documentation",
            "source_reference": "data/clinical_master_dictionary.csv",
            "implication": (
                "Interpret outcomes in their recorded source units. Assay methods, fasting status, "
                "treatment timing and any calculated-LDL formula require source documentation; "
                "this code does not infer them."
            ),
        },
        {
            "domain": "misev2023_fraction_and_repeat_scope",
            "status": "shared_isolation_possible",
            "source_reference": "MISEV2023 (doi:10.1002/jev2.12404); docs/preprocessing.md",
            "implication": (
                "SNV and area-normalized spectra capture relative shape, not particle concentration or "
                "fraction lipid mass. Repeat agreement measures only stages independently repeated; "
                "co-isolated lipoproteins or platelets cannot be separated from EVs by Raman alone."
            ),
        },
        {
            "domain": "conditional_independence_and_confounding",
            "status": "model_relative_comparison_only",
            "source_reference": "docs/research_design.md",
            "implication": (
                "Observed error differences compare fitted procedures. They do not establish "
                "biological specificity or conditional association, and spectra may proxy "
                "nonlinear covariate effects or unmeasured technical preparation structure."
            ),
        },
        {
            "domain": "control_cohort_absence_of_lipids",
            "status": "controls_excluded_from_primary_model",
            "source_reference": "data/clinical_master.csv",
            "implication": (
                "Controls are outside the primary within-stroke analysis. This design does not "
                "establish case-control disease specificity."
            ),
        },
    ]
    return pd.DataFrame(rows)
