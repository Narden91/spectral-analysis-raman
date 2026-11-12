"""Clinical-master linkage, paired cohort construction, band extraction and limitations table."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

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


@dataclass(frozen=True)
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
        if self.covariate_preset not in COVARIATE_PRESETS:
            raise ValueError(f"Unknown covariate preset: {self.covariate_preset!r}")
        if not self.pca_dimensions or any(int(d) < 1 for d in self.pca_dimensions):
            raise ValueError("pca_dimensions must contain positive integers.")
        if not self.retained_directions or any(int(r) < 1 for r in self.retained_directions):
            raise ValueError("retained_directions must contain positive integers.")
        if not self.lambda_grid or any(float(lam) < 0.0 for lam in self.lambda_grid):
            raise ValueError("lambda_grid must contain non-negative values.")
        if not self.ridge_alpha_grid or any(float(a) <= 0.0 for a in self.ridge_alpha_grid):
            raise ValueError("ridge_alpha_grid must contain strictly positive values.")
        if not self.pls_component_grid or any(int(c) < 1 for c in self.pls_component_grid):
            raise ValueError("pls_component_grid must contain positive integers.")
        if self.n_outer_splits < 2 or self.n_inner_splits < 2:
            raise ValueError("n_outer_splits and n_inner_splits must be at least 2.")
        if self.min_maps_per_block < 1 or self.local_candidate_window < 1:
            raise ValueError("min_maps_per_block and local_candidate_window must be at least 1.")
        if self.rank_tolerance <= 0.0 or self.eigenvalue_tolerance <= 0.0:
            raise ValueError("rank_tolerance and eigenvalue_tolerance must be positive.")


@dataclass(frozen=True)
class SampleRecord:
    """Aggregated spectral summary for one eligible technical sample."""

    participant_id: str
    sample_id: str
    parsed_label: float
    spec_1300: np.ndarray
    spec_2850: np.ndarray


@dataclass
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
    _cache: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def n_participants(self) -> int:
        return int(self.participant_ids.size)

    @property
    def n_samples(self) -> int:
        return int(self.sample_participant_idx.size)


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
    target_col = cfg.primary_endpoint
    min_maps = 3 if subset_mode == "three_map_complete" else cfg.min_maps_per_block

    clin = clinical_df.copy()
    if "cohort" in clin.columns:
        clin = clin.loc[clin["cohort"].astype(str) == "stroke"].copy()
    id_col = "spectral_subject_id" if "spectral_subject_id" in clin.columns else "participant_id"
    clin[id_col] = clin[id_col].astype(str)

    exclusions: list[dict[str, Any]] = []
    if "identity_verified" in clin.columns:
        verified = clin["identity_verified"].astype(bool)
        has_spec = clin.get("has_spectra", pd.Series(1, index=clin.index)).fillna(0).astype(int) == 1
        keys = clin.get("participant_key", clin[id_col])
        for pid, key, spec_ok in zip(clin.loc[~verified, id_col], keys[~verified], has_spec[~verified]):
            # Records without spectra have no spectral id; report the clinical key instead.
            exclusions.append(
                {
                    "participant_id": str(pid) if spec_ok else str(key),
                    "level": "participant",
                    "reason": "identity_review_required" if spec_ok else "no_spectra",
                }
            )
        clin = clin.loc[verified].copy()
    elif "spectral_link_status" in clin.columns:
        matched = clin["spectral_link_status"].astype(str) == "matched"
        for pid in clin.loc[~matched, id_col]:
            exclusions.append({"participant_id": str(pid), "level": "participant", "reason": "unmatched_spectral_link"})
        clin = clin.loc[matched].copy()

    target_vals = pd.to_numeric(clin[target_col], errors="coerce")
    for pid in clin.loc[target_vals.isna(), id_col]:
        exclusions.append({"participant_id": str(pid), "level": "participant", "reason": f"missing_{target_col}"})
    clin = clin.loc[target_vals.notna()].copy()
    clin = clin.sort_values(id_col, kind="mergesort").reset_index(drop=True)

    maps = map_records.copy()
    p_col = "spectral_subject_id" if "spectral_subject_id" in maps.columns else "participant_id"
    s_col = "slide_number" if "slide_number" in maps.columns else "sample_id"
    maps["participant_id"] = maps[p_col].astype(str)
    maps["sample_id"] = maps[s_col].astype(str)
    maps["spectral_range"] = maps["spectral_range"].astype(str)
    if "retained" in maps.columns:
        maps = maps.loc[maps["retained"].astype(bool)].copy()

    # Group maps by (participant_id, sample_id, spectral_range)
    grouped = maps.groupby(["participant_id", "sample_id", "spectral_range"], sort=False)
    sample_blocks: dict[tuple[str, str], dict[str, Any]] = {}
    for (pid, sid, rng), grp in grouped:
        sample_blocks.setdefault((str(pid), str(sid)), {})[f"idx_{rng}"] = grp["row_index"].to_numpy(dtype=int)

    participant_samples: dict[str, list[SampleRecord]] = {}
    for (pid, sid), info in sorted(sample_blocks.items()):
        idx_1300 = info.get("idx_1300", np.empty(0, dtype=int))
        idx_2850 = info.get("idx_2850", np.empty(0, dtype=int))
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
        med_1300 = np.nanmedian(spectra_1300[idx_1300], axis=0)
        med_2850 = np.nanmedian(spectra_2850[idx_2850], axis=0)
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

    # Order samples within each participant deterministically by (parsed_label, sample_id)
    for pid, s_list in participant_samples.items():
        s_list.sort(key=lambda r: (float("inf") if np.isnan(r.parsed_label) else r.parsed_label, r.sample_id))
        if len(s_list) > 2:
            # Retain the first two deterministic technical samples if more than two exist
            for dropped in s_list[2:]:
                exclusions.append(
                    {"participant_id": pid, "sample_id": dropped.sample_id, "level": "sample", "reason": "beyond_first_two_samples"}
                )
            participant_samples[pid] = s_list[:2]

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
            exclusions.append({"participant_id": pid, "level": "participant", "reason": "unpaired_in_paired_only_mode"})
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

    kept_df = clin.iloc[kept_clin_indices].reset_index(drop=True)
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

    dataset = CohortDataset(
        participant_ids=kept_df[id_col].astype(str).to_numpy(),
        outcomes=outcomes,
        covariate_frame=cov_df,
        wavenumbers_1300=np.asarray(wavenumbers_1300, dtype=float),
        wavenumbers_2850=np.asarray(wavenumbers_2850, dtype=float),
        sample_1300=np.vstack(flat_1300),
        sample_2850=np.vstack(flat_2850),
        sample_participant_idx=np.asarray(flat_p_idx, dtype=int),
        sample_ids=np.asarray(flat_s_ids, dtype=str),
        sample_parsed_labels=np.asarray(flat_lbls, dtype=float),
        participant_sample_counts=np.asarray(sample_counts, dtype=int),
        paired_participant_indices=np.asarray(paired_p_indices, dtype=int),
        pair_sample_a_indices=np.asarray(pair_a_indices, dtype=int),
        pair_sample_b_indices=np.asarray(pair_b_indices, dtype=int),
        manifest_summary={
            "subset_mode": subset_mode,
            "min_maps_per_block": min_maps,
            "n_eligible_participants": len(kept_clin_indices),
            "n_paired_participants": len(paired_p_indices),
            "n_single_sample_participants": int(np.sum(np.asarray(sample_counts) == 1)),
            "n_eligible_samples": len(flat_1300),
            "n_exclusions": len(exclusions),
        },
        exclusions=exclusions,
    )
    return dataset


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
            selector = idx
        specs.append((name, block, selector))
    return tuple(specs)


def extract_sample_band_matrix(dataset: CohortDataset) -> tuple[np.ndarray, list[str]]:
    """Compute the 21-band summary matrix for all samples with memoized zero-copy channel slicing."""
    cached = dataset._cache.get("sample_band_matrix")
    if cached is not None:
        return cached

    w1 = np.ascontiguousarray(dataset.wavenumbers_1300, dtype=np.float64).tobytes()
    w2 = np.ascontiguousarray(dataset.wavenumbers_2850, dtype=np.float64).tobytes()
    slices = _cached_band_slices(w1, w2)

    cols: list[np.ndarray] = []
    names: list[str] = []
    for name, block, selector in slices:
        if isinstance(selector, np.ndarray) and selector.size == 0:
            raise ValueError(f"Band '{name}' has no channels in block {block} axis.")
        source = dataset.sample_1300 if block == "1300" else dataset.sample_2850
        cols.append(np.mean(source[:, selector], axis=1))
        names.append(name)

    mat = np.column_stack(cols)
    result = (mat, names)
    dataset._cache["sample_band_matrix"] = result
    return result


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


def load_paired_cohort_from_store(
    clinical_master_path: str | Path = "data/clinical_master.csv",
    *,
    data_source: Any | None = None,
    cfg: PairedLipidConfig = PairedLipidConfig(),
    subset_mode: SubsetMode = "all_eligible",
) -> tuple[CohortDataset, dict[str, Any]]:
    """Load the stroke cohort hierarchy from the canonical preprocessed store and clinical master."""
    from src.data_preparation.loaders import load_area_summaries, load_file_index

    clin_df, clin_audit = load_clinical_master_verified(clinical_master_path)
    file_index = load_file_index(data_source).copy()
    if "is_control" in file_index.columns:
        file_index = file_index.loc[~file_index["is_control"].fillna(False).astype(bool)].copy()

    raw_id_col = "subject_id_raw" if "subject_id_raw" in file_index.columns else "subject_id"
    file_index["participant_id"] = file_index[raw_id_col].astype(str).str.strip()
    file_index["sample_id"] = file_index["slide_number"].astype(str).str.strip()
    file_index["measurement_id"] = file_index["measurement_id"].astype(str)

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

    idx_map_1300 = {str(mid): i for i, mid in enumerate(sum_1300["measurement_ids"])}
    idx_map_2850 = {str(mid): i for i, mid in enumerate(sum_2850["measurement_ids"])}

    rng_series = file_index["spectral_range"].astype(str)
    mid_series = file_index["measurement_id"]
    row_idx = pd.Series(-1, index=file_index.index, dtype=int)
    mask_1300 = rng_series == "1300"
    mask_2850 = rng_series == "2850"
    if mask_1300.any():
        row_idx.loc[mask_1300] = mid_series.loc[mask_1300].map(idx_map_1300).fillna(-1).astype(int)
    if mask_2850.any():
        row_idx.loc[mask_2850] = mid_series.loc[mask_2850].map(idx_map_2850).fillna(-1).astype(int)

    valid_rows = row_idx >= 0
    map_df = pd.DataFrame(
        {
            "participant_id": file_index.loc[valid_rows, "participant_id"].to_numpy(dtype=str),
            "sample_id": file_index.loc[valid_rows, "sample_id"].to_numpy(dtype=str),
            "spectral_range": rng_series.loc[valid_rows].to_numpy(dtype=str),
            "row_index": row_idx.loc[valid_rows].to_numpy(dtype=int),
            "retained": np.ones(int(valid_rows.sum()), dtype=bool),
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
