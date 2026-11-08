"""Reproducible Raman representation learning for circulating lipid prediction.

Implements the study design in ``docs/research_design.md`` using DRY, KISS,
vectorized linear algebra, and memoized preprocessing/feature extraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import math
import re
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import linalg, stats

from src.analysis.raman_bands import RAMAN_BANDS, RamanBand

MISSING_TOKENS: tuple[str, ...] = ("unknown", "not_applicable", "not_available")
IDENTITY_CONFLICT_FLAGS: frozenset[str] = frozenset(
    {
        "SEX_FOLDER_CONFLICT",
        "SPECTRAL_FILENAME_FOLDER_CONFLICT",
        "POSSIBLE_SPECTRA_IN_ANOTHER_FOLDER",
        "MULTIPLE_SEX_FOLDERS",
    }
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
ESTABLISHED_COMPARATORS: tuple[str, ...] = (
    "fixed_band_ridge",
    "pca_ridge",
    "pls_regression",
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
    mode_reference_category: str = "NONE"
    local_candidate_window: int = 5


@dataclass(frozen=True)
class SampleRecord:
    """Aggregated spectral summary for one eligible technical sample."""

    participant_id: str
    sample_id: str
    parsed_label: float
    spec_1300: np.ndarray
    spec_2850: np.ndarray
    n_maps_1300: int
    n_maps_2850: int
    mode_proportions: dict[str, float]


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


def load_clinical_master_verified(master_path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load and audit the standalone clinical master table per study readiness rules."""
    path = Path(master_path)
    raw_bytes = path.read_bytes()
    sha256_hash = hashlib.sha256(raw_bytes).hexdigest()
    frame = pd.read_csv(
        path,
        na_values=list(MISSING_TOKENS),
        keep_default_na=True,
        encoding="utf-8-sig",
    )
    if "participant_key" not in frame.columns or "cohort" not in frame.columns:
        raise ValueError("Clinical master must contain 'participant_key' and 'cohort' columns.")
    if frame["participant_key"].duplicated().any():
        raise ValueError("Duplicate participant_key entries detected in clinical master.")

    has_spec_mask = frame.get("has_spectra", pd.Series(0, index=frame.index)).fillna(0).astype(int) == 1
    spec_subset = frame.loc[has_spec_mask]
    if spec_subset.duplicated(subset=["cohort", "spectral_subject_id"]).any():
        raise ValueError("Duplicate (cohort, spectral_subject_id) among participants with spectra.")

    def _has_identity_conflict(flags_val: object) -> bool:
        if pd.isna(flags_val):
            return False
        tokens = {tok.strip() for tok in str(flags_val).split(";") if tok.strip()}
        return bool(tokens & IDENTITY_CONFLICT_FLAGS)

    conflict_series = frame.get("qc_flags", pd.Series("none", index=frame.index)).map(_has_identity_conflict)
    link_matched = frame.get("spectral_link_status", pd.Series("", index=frame.index)).astype(str) == "matched"
    frame = frame.copy()
    frame["identity_verified"] = has_spec_mask & link_matched & (~conflict_series)

    stroke_mask = frame["cohort"].astype(str) == "stroke"
    stroke_verified = frame.loc[stroke_mask & frame["identity_verified"]]
    audit = {
        "clinical_master_sha256": sha256_hash,
        "total_rows": int(len(frame)),
        "stroke_with_spectra": int((stroke_mask & has_spec_mask).sum()),
        "stroke_identity_verified": int(len(stroke_verified)),
        "stroke_identity_excluded": int((stroke_mask & has_spec_mask & ~frame["identity_verified"]).sum()),
        "control_with_spectra": int(((~stroke_mask) & has_spec_mask).sum()),
    }
    return frame, audit


def build_paired_cohort_dataset(
    clinical_df: pd.DataFrame,
    map_records: pd.DataFrame,
    spectra_1300: np.ndarray,
    spectra_2850: np.ndarray,
    wavenumbers_1300: np.ndarray,
    wavenumbers_2850: np.ndarray,
    *,
    cfg: PairedLipidConfig = PairedLipidConfig(),
    subset_mode: Literal["all_eligible", "paired_only", "single_sample", "three_map_complete"] = "all_eligible",
    endpoint: str | None = None,
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
        ``acquisition_mode`` and ``retained`` (bool).
    """
    target_col = endpoint or cfg.primary_endpoint
    min_maps = 3 if subset_mode == "three_map_complete" else cfg.min_maps_per_block

    clin = clinical_df.copy()
    if "cohort" in clin.columns:
        clin = clin.loc[clin["cohort"].astype(str) == "stroke"].copy()
    id_col = "spectral_subject_id" if "spectral_subject_id" in clin.columns else "participant_id"
    clin[id_col] = clin[id_col].astype(str)

    exclusions: list[dict[str, Any]] = []
    if "identity_verified" in clin.columns:
        bad_id = clin.loc[~clin["identity_verified"].astype(bool)]
        for pid in bad_id[id_col]:
            exclusions.append({"participant_id": str(pid), "level": "participant", "reason": "identity_review_required"})
        clin = clin.loc[clin["identity_verified"].astype(bool)].copy()
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
    maps["acquisition_mode"] = (
        maps["acquisition_mode"].fillna("NONE").astype(str).str.upper()
        if "acquisition_mode" in maps.columns
        else "NONE"
    )

    all_modes = sorted(maps["acquisition_mode"].unique().tolist())
    if not all_modes:
        all_modes = [cfg.mode_reference_category]

    # Group maps by (participant_id, sample_id, spectral_range)
    grouped = maps.groupby(["participant_id", "sample_id", "spectral_range"], sort=False)
    sample_blocks: dict[tuple[str, str], dict[str, Any]] = {}
    for (pid, sid, rng), grp in grouped:
        entry = sample_blocks.setdefault((str(pid), str(sid)), {"modes": []})
        idx = grp["row_index"].to_numpy(dtype=int)
        entry[f"idx_{rng}"] = idx
        entry["modes"].extend(grp["acquisition_mode"].tolist())

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
        modes_list = info["modes"]
        n_m = max(len(modes_list), 1)
        mode_props = {m: modes_list.count(m) / float(n_m) for m in all_modes}
        parsed_lbl = parse_sample_label(sid)
        participant_samples.setdefault(pid, []).append(
            SampleRecord(
                participant_id=pid,
                sample_id=sid,
                parsed_label=parsed_lbl,
                spec_1300=med_1300,
                spec_2850=med_2850,
                n_maps_1300=int(idx_1300.size),
                n_maps_2850=int(idx_2850.size),
                mode_proportions=mode_props,
            )
        )

    # Order samples within each participant deterministically by (parsed_label, sample_id)
    for pid, s_list in participant_samples.items():
        s_list.sort(key=lambda r: (float("inf") if np.isnan(r.parsed_label) else r.parsed_label, r.sample_id))
        if len(s_list) > 2:
            # Retain the first two deterministic technical samples if more than two exist
            participant_samples[pid] = s_list[:2]

    kept_clin_rows: list[pd.Series] = []
    flat_1300: list[np.ndarray] = []
    flat_2850: list[np.ndarray] = []
    flat_p_idx: list[int] = []
    flat_s_ids: list[str] = []
    flat_lbls: list[float] = []
    sample_counts: list[int] = []
    paired_p_indices: list[int] = []
    pair_a_indices: list[int] = []
    pair_b_indices: list[int] = []
    mode_rows: list[dict[str, float]] = []
    mean_labels: list[float] = []

    for _, row in clin.iterrows():
        pid = str(row[id_col])
        s_list = participant_samples.get(pid, [])
        if not s_list:
            exclusions.append({"participant_id": pid, "level": "participant", "reason": "no_eligible_samples"})
            continue
        if subset_mode == "paired_only" and len(s_list) < 2:
            exclusions.append({"participant_id": pid, "level": "participant", "reason": "unpaired_in_paired_only_mode"})
            continue

        p_idx = len(kept_clin_rows)
        kept_clin_rows.append(row)
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

        # In single_sample sensitivity mode, participant-level regression covariates/summaries
        # for sample label and mode use the first sample (A); paired samples remain in flat arrays
        # so training divisions can still fit W and T on paired training participants.
        active_samples = [s_list[0]] if subset_mode == "single_sample" else s_list
        avg_modes = {
            f"mode_prop_{m}": float(np.mean([sr.mode_proportions.get(m, 0.0) for sr in active_samples]))
            for m in all_modes
            if m != cfg.mode_reference_category
        }
        mode_rows.append(avg_modes)
        finite_lbls = [sr.parsed_label for sr in active_samples if np.isfinite(sr.parsed_label)]
        mean_labels.append(float(np.mean(finite_lbls)) if finite_lbls else float("nan"))

    if not kept_clin_rows:
        raise ValueError("No eligible stroke participants remained after linkage and spectral checks.")

    kept_df = pd.DataFrame(kept_clin_rows).reset_index(drop=True)
    mode_df = pd.DataFrame(mode_rows, index=kept_df.index)
    cov_df = pd.DataFrame(index=kept_df.index)
    cov_df["age_years"] = pd.to_numeric(kept_df.get("age_years"), errors="coerce")
    cov_df["sex"] = kept_df.get("sex", pd.Series(np.nan, index=kept_df.index)).astype(object)
    cov_df["statin"] = pd.to_numeric(kept_df.get("statin"), errors="coerce")
    for m_col in mode_df.columns:
        cov_df[m_col] = mode_df[m_col]
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
            "n_eligible_participants": len(kept_clin_rows),
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
) -> tuple[tuple[str, str, np.ndarray], ...]:
    """Memoize channel index arrays for all 21 pre-specified Raman bands."""
    wn_1300 = np.frombuffer(wn_1300_bytes, dtype=np.float64)
    wn_2850 = np.frombuffer(wn_2850_bytes, dtype=np.float64)
    specs: list[tuple[str, str, np.ndarray]] = []
    for name, band in RAMAN_BANDS.items():
        lower, upper = float(band.bounds[0]), float(band.bounds[1])
        block = "1300" if upper < 2000.0 else "2850"
        axis = wn_1300 if block == "1300" else wn_2850
        idx = np.flatnonzero((axis >= lower) & (axis <= upper))
        specs.append((name, block, idx))
    return tuple(specs)


def extract_sample_band_matrix(dataset: CohortDataset) -> tuple[np.ndarray, list[str]]:
    """Compute the 21-band summary matrix for all samples with memoized channel slicing."""
    cached = dataset._cache.get("sample_band_matrix")
    if cached is not None:
        return cached

    w1 = np.ascontiguousarray(dataset.wavenumbers_1300, dtype=np.float64).tobytes()
    w2 = np.ascontiguousarray(dataset.wavenumbers_2850, dtype=np.float64).tobytes()
    slices = _cached_band_slices(w1, w2)

    cols: list[np.ndarray] = []
    names: list[str] = []
    for name, block, idx in slices:
        if idx.size == 0:
            raise ValueError(f"Band '{name}' has no channels in block {block} axis.")
        source = dataset.sample_1300 if block == "1300" else dataset.sample_2850
        cols.append(np.mean(source[:, idx], axis=1))
        names.append(name)

    mat = np.column_stack(cols)
    result = (mat, names)
    dataset._cache["sample_band_matrix"] = result
    return result


def aggregate_samples_to_participants(
    sample_matrix: np.ndarray,
    dataset: CohortDataset,
    *,
    single_sample_only: bool = False,
) -> np.ndarray:
    """Vectorized participant-level equal average of eligible sample representations."""
    n_p = dataset.n_participants
    p_idx = dataset.sample_participant_idx
    if single_sample_only:
        # Select first eligible sample (A) for each participant via unique first-occurrence index
        _, first_indices = np.unique(p_idx, return_index=True)
        return sample_matrix[first_indices].copy()

    # Fast vectorized sum by participant using np.add.at
    sums = np.zeros((n_p, sample_matrix.shape[1]), dtype=float)
    np.add.at(sums, p_idx, sample_matrix)
    counts = dataset.participant_sample_counts[:, None].astype(float)
    return sums / counts


@dataclass
class CovariatePreprocessor:
    """Training-only covariate imputation, missingness indicator, and standardization."""

    feature_names: list[str]
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    dropped_columns: list[str]
    include_missing_indicators_for: list[str]
    preset: str

    @classmethod
    def fit(
        cls,
        cov_frame: pd.DataFrame,
        train_indices: np.ndarray,
        participant_weights: np.ndarray,
        *,
        preset: str = "primary",
    ) -> "CovariatePreprocessor":
        raw_df = cls._build_raw_design(cov_frame.iloc[train_indices], preset=preset)
        w = np.asarray(participant_weights, dtype=float)
        w = w / np.sum(w)

        kept_cols: list[str] = []
        dropped_cols: list[str] = []
        miss_indicator_cols: list[str] = []
        medians_list: list[float] = []

        for col in raw_df.columns:
            vals = raw_df[col].to_numpy(dtype=float)
            finite = np.isfinite(vals)
            if not np.any(finite):
                dropped_cols.append(col)
                continue
            # Weighted median on finite values
            f_vals = vals[finite]
            f_w = w[finite]
            order = np.argsort(f_vals)
            cdf = np.cumsum(f_w[order]) / np.sum(f_w)
            med = float(f_vals[order[np.searchsorted(cdf, 0.5)]])
            imputed = np.where(finite, vals, med)
            if np.ptp(imputed) <= 1e-12:
                dropped_cols.append(col)
                continue
            kept_cols.append(col)
            medians_list.append(med)
            if not np.all(finite):
                miss_indicator_cols.append(col)

        if not kept_cols:
            return cls(
                feature_names=[],
                medians=np.empty(0, dtype=float),
                means=np.empty(0, dtype=float),
                scales=np.empty(0, dtype=float),
                dropped_columns=dropped_cols,
                include_missing_indicators_for=[],
                preset=preset,
            )

        medians = np.asarray(medians_list, dtype=float)
        design_list: list[np.ndarray] = []
        names: list[str] = []
        for idx, col in enumerate(kept_cols):
            vals = raw_df[col].to_numpy(dtype=float)
            finite = np.isfinite(vals)
            design_list.append(np.where(finite, vals, medians[idx]))
            names.append(col)
        for col in miss_indicator_cols:
            vals = raw_df[col].to_numpy(dtype=float)
            ind = (~np.isfinite(vals)).astype(float)
            if np.ptp(ind) > 1e-12:
                design_list.append(ind)
                names.append(f"{col}__missing")

        X = np.column_stack(design_list)
        means = np.sum(w[:, None] * X, axis=0)
        centered = X - means
        variances = np.sum(w[:, None] * (centered**2), axis=0)
        scales = np.sqrt(np.maximum(variances, 1e-12))
        return cls(
            feature_names=names,
            medians=medians,
            means=means,
            scales=scales,
            dropped_columns=dropped_cols,
            include_missing_indicators_for=miss_indicator_cols,
            preset=preset,
        )

    def transform(self, cov_frame: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
        if not self.feature_names:
            return np.zeros((len(indices), 0), dtype=float)
        raw_df = self._build_raw_design(cov_frame.iloc[indices], preset=self.preset)
        base_cols = [c for c in self.feature_names if not c.endswith("__missing")]
        cols_out: list[np.ndarray] = []
        for idx, col in enumerate(base_cols):
            vals = (
                raw_df[col].to_numpy(dtype=float)
                if col in raw_df.columns
                else np.full(len(indices), np.nan, dtype=float)
            )
            finite = np.isfinite(vals)
            cols_out.append(np.where(finite, vals, self.medians[idx]))
        for col in self.include_missing_indicators_for:
            if f"{col}__missing" in self.feature_names:
                vals = (
                    raw_df[col].to_numpy(dtype=float)
                    if col in raw_df.columns
                    else np.full(len(indices), np.nan, dtype=float)
                )
                cols_out.append((~np.isfinite(vals)).astype(float))
        X = np.column_stack(cols_out)
        return (X - self.means) / self.scales

    @staticmethod
    def _build_raw_design(df: pd.DataFrame, *, preset: str) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        out["age_years"] = pd.to_numeric(df.get("age_years"), errors="coerce")
        if preset == "quadratic_age_and_label":
            out["age_years_sq"] = out["age_years"] ** 2

        sex_s = df.get("sex", pd.Series(np.nan, index=df.index)).astype(str).str.lower()
        sex_num = np.where(sex_s == "female", 1.0, np.where(sex_s == "male", 0.0, np.nan))
        out["sex_female"] = sex_num
        out["statin"] = pd.to_numeric(df.get("statin"), errors="coerce")

        for col in sorted(c for c in df.columns if c.startswith("mode_prop_")):
            out[col] = pd.to_numeric(df[col], errors="coerce")

        if preset != "no_label_trend":
            lbl = pd.to_numeric(df.get("sample_label_trend"), errors="coerce")
            out["sample_label_trend"] = lbl
            if preset in ("quadratic_label_trend", "quadratic_age_and_label"):
                out["sample_label_trend_sq"] = lbl**2

        if preset == "pre_event_risk_factors":
            for rf in PRE_EVENT_RISK_FACTORS:
                if rf not in df.columns:
                    continue
                if rf == "smoking_status":
                    sm = df[rf].astype(str).str.lower()
                    out["smoking_current"] = np.where(
                        sm == "current", 1.0, np.where(sm.isin(["never", "former"]), 0.0, np.nan)
                    )
                    out["smoking_former"] = np.where(
                        sm == "former", 1.0, np.where(sm.isin(["never", "current"]), 0.0, np.nan)
                    )
                else:
                    out[rf] = pd.to_numeric(df[rf], errors="coerce")
        return out


@dataclass
class WeightedSpectralPreprocessor:
    """Participant-weighted MFA block scaling and truncated SVD for training divisions."""

    mean_1300: np.ndarray
    mean_2850: np.ndarray
    scale_1300: float
    scale_2850: float
    eig1_1300: float
    eig1_2850: float
    pca_components: np.ndarray
    pca_singular_values: np.ndarray
    numerical_rank: int
    n_distinct_paired_train: int

    @classmethod
    def fit(
        cls,
        dataset: CohortDataset,
        train_participant_indices: np.ndarray,
        participant_multiplicities: np.ndarray,
        *,
        cfg: PairedLipidConfig = PairedLipidConfig(),
    ) -> "WeightedSpectralPreprocessor":
        train_p = np.asarray(train_participant_indices, dtype=int)
        mult = np.asarray(participant_multiplicities, dtype=float)
        if np.any(mult <= 0):
            keep = mult > 0
            train_p = train_p[keep]
            mult = mult[keep]

        n_eff = float(np.sum(mult))
        p_to_mult = np.zeros(dataset.n_participants, dtype=float)
        p_to_mult[train_p] = mult

        sample_p = dataset.sample_participant_idx
        in_train = p_to_mult[sample_p] > 0
        s_indices = np.flatnonzero(in_train)
        s_p = sample_p[s_indices]

        m_i = dataset.participant_sample_counts[s_p].astype(float)
        w_samples = p_to_mult[s_p] / (n_eff * m_i)
        w_sum = float(np.sum(w_samples))
        if not np.isfinite(w_sum) or w_sum <= 0:
            raise ValueError("Invalid sample weights in spectral preprocessing.")
        w_samples = w_samples / w_sum

        X1 = dataset.sample_1300[s_indices]
        X2 = dataset.sample_2850[s_indices]

        mean1 = np.sum(w_samples[:, None] * X1, axis=0)
        mean2 = np.sum(w_samples[:, None] * X2, axis=0)
        C1 = X1 - mean1
        C2 = X2 - mean2

        # Effective total sample count for sample covariance denominator
        m_total = float(np.sum(p_to_mult[s_p]))
        denom_factor = m_total / max(m_total - 1.0, 1.0)
        sqrt_w = np.sqrt(w_samples * denom_factor)[:, None]

        eig1_1300 = cls._leading_eigenvalue(sqrt_w * C1, cfg.eigenvalue_tolerance)
        eig1_2850 = cls._leading_eigenvalue(sqrt_w * C2, cfg.eigenvalue_tolerance)
        scale1 = float(1.0 / np.sqrt(eig1_1300))
        scale2 = float(1.0 / np.sqrt(eig1_2850))

        combined_scaled = np.hstack([C1 * scale1, C2 * scale2])
        weighted_combined = sqrt_w * combined_scaled
        _, s_vals, vt = linalg.svd(weighted_combined, full_matrices=False, check_finite=True)

        tol = max(weighted_combined.shape) * np.finfo(float).eps * max(float(s_vals[0]) if s_vals.size else 1.0, 1.0)
        rank = int(np.sum(s_vals > max(tol, cfg.rank_tolerance)))

        max_k = min(max(cfg.pca_dimensions), rank, vt.shape[0])
        comps = vt[:max_k].copy()
        s_kept = s_vals[:max_k].copy()

        # Deterministic sign rule: largest-magnitude loading element in each component is positive
        max_abs_idx = np.argmax(np.abs(comps), axis=1)
        signs = np.sign(comps[np.arange(max_k), max_abs_idx])
        signs[signs == 0] = 1.0
        comps *= signs[:, None]

        paired_in_train = np.intersect1d(train_p, dataset.paired_participant_indices, assume_unique=False)
        return cls(
            mean_1300=mean1,
            mean_2850=mean2,
            scale_1300=scale1,
            scale_2850=scale2,
            eig1_1300=eig1_1300,
            eig1_2850=eig1_2850,
            pca_components=comps,
            pca_singular_values=s_kept,
            numerical_rank=rank,
            n_distinct_paired_train=int(paired_in_train.size),
        )

    @staticmethod
    def _leading_eigenvalue(weighted_matrix: np.ndarray, tol: float) -> float:
        if weighted_matrix.shape[0] < 2:
            raise ValueError("Block requires at least two weighted observations.")
        # Gram matrix is n_samples x n_samples (typically <= 200), much smaller than p_b x p_b
        gram = weighted_matrix @ weighted_matrix.T
        gram = 0.5 * (gram + gram.T)
        eigvals = linalg.eigvalsh(gram)
        top_eig = float(eigvals[-1])
        if not np.isfinite(top_eig) or top_eig <= tol:
            raise ValueError(f"Degenerate leading block eigenvalue: {top_eig}")
        return top_eig

    def transform_scaled_blocks(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Center and block-scale spectra without PCA truncation."""
        return np.hstack(
            [
                (X1 - self.mean_1300) * self.scale_1300,
                (X2 - self.mean_2850) * self.scale_2850,
            ]
        )

    def transform_pca(self, X1: np.ndarray, X2: np.ndarray, k: int | None = None) -> np.ndarray:
        """Project spectra into top-k weighted MFA-PCA coordinates."""
        scaled = self.transform_scaled_blocks(X1, X2)
        comps = self.pca_components if k is None else self.pca_components[:k]
        return scaled @ comps.T

    def candidate_pca_dims(self, cfg: PairedLipidConfig) -> list[int]:
        """Return deduplicated valid PCA dimensions capped by rank and n_p - 1."""
        cap = min(self.numerical_rank, self.n_distinct_paired_train - 1, self.pca_components.shape[0])
        if cap < 1:
            return []
        dims: list[int] = []
        for d in cfg.pca_dimensions:
            capped = min(int(d), cap)
            if capped >= 1 and capped not in dims:
                dims.append(capped)
        return dims


@dataclass
class PairedMoments:
    """Precomputed pair disagreement W and between-participant T in max-PCA or band space."""

    W_max: np.ndarray
    T_max: np.ndarray
    mean_ab_shift: np.ndarray
    n_paired_multiplicity: float
    n_paired_distinct: int

    @classmethod
    def from_sample_coordinates(
        cls,
        sample_coords: np.ndarray,
        dataset: CohortDataset,
        train_participant_indices: np.ndarray,
        participant_multiplicities: np.ndarray,
    ) -> "PairedMoments":
        p_to_mult = np.zeros(dataset.n_participants, dtype=float)
        p_to_mult[train_participant_indices] = participant_multiplicities

        paired_p = dataset.paired_participant_indices
        mult_p = p_to_mult[paired_p]
        active = mult_p > 0
        if int(np.sum(active)) < 2:
            raise ValueError("At least two distinct paired training participants are required.")

        mult_active = mult_p[active]
        idx_a = dataset.pair_sample_a_indices[active]
        idx_b = dataset.pair_sample_b_indices[active]

        x_a = sample_coords[idx_a]
        x_b = sample_coords[idx_b]
        diffs = x_a - x_b
        means = 0.5 * (x_a + x_b)

        n_p = float(np.sum(mult_active))
        if n_p <= 1.0:
            raise ValueError("Total paired multiplicity must exceed 1.")

        w_diff = np.sqrt(mult_active / (2.0 * n_p))[:, None] * diffs
        W = w_diff.T @ w_diff
        W = 0.5 * (W + W.T)

        u_bar = np.sum(mult_active[:, None] * means, axis=0) / n_p
        u_centered = means - u_bar
        w_u = np.sqrt(mult_active / (n_p - 1.0))[:, None] * u_centered
        T = w_u.T @ w_u
        T = 0.5 * (T + T.T)

        mean_shift = np.sum(mult_active[:, None] * diffs, axis=0) / n_p
        return cls(
            W_max=W,
            T_max=T,
            mean_ab_shift=mean_shift,
            n_paired_multiplicity=n_p,
            n_paired_distinct=int(np.sum(active)),
        )

    def solve_directions(
        self,
        k: int,
        lam: float,
        *,
        eigenvalue_tolerance: float = 1e-10,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Solve T v = rho (W + lam * tau * I) v for the top-k subspace in O(k^3)."""
        W_k = self.W_max[:k, :k]
        T_k = self.T_max[:k, :k]
        if not (np.all(np.isfinite(W_k)) and np.all(np.isfinite(T_k))):
            raise ValueError("Non-finite covariance matrices in paired eigensolver.")

        tr_t = float(np.trace(T_k))
        tau = tr_t / float(k)
        if not np.isfinite(tau) or tau <= eigenvalue_tolerance:
            raise ValueError(f"Non-positive or degenerate tau={tau} in paired eigensolver.")

        denom = W_k + float(lam) * tau * np.eye(k, dtype=float)
        denom = 0.5 * (denom + denom.T)

        eigvals, eigvecs = linalg.eigh(T_k, denom, check_finite=True)
        order = np.argsort(eigvals)[::-1]
        eigvals = np.asarray(eigvals[order], dtype=float)
        eigvecs = np.asarray(eigvecs[:, order], dtype=float)

        # Normalize in the regularized denominator metric and enforce deterministic signs
        metric_norms = np.sqrt(np.maximum(np.sum(eigvecs * (denom @ eigvecs), axis=0), 1e-15))
        eigvecs = eigvecs / metric_norms[None, :]
        max_abs_idx = np.argmax(np.abs(eigvecs), axis=0)
        signs = np.sign(eigvecs[max_abs_idx, np.arange(k)])
        signs[signs == 0] = 1.0
        eigvecs *= signs[None, :]
        return eigvecs, eigvals, tau


@dataclass
class MultiAlphaRidgeResult:
    """Vectorized multi-penalty ridge fit for a single representation."""

    alphas: np.ndarray
    intercepts: np.ndarray
    coef_cov: np.ndarray
    coef_spec: np.ndarray
    z_mean: np.ndarray
    z_scale: np.ndarray


def fit_ridge_grid(
    X_cov_train: np.ndarray,
    Z_spec_train: np.ndarray | None,
    y_train: np.ndarray,
    participant_weights: np.ndarray,
    alphas: Sequence[float],
) -> MultiAlphaRidgeResult:
    """Fit weighted ridge regression across all candidate alphas simultaneously via thin SVD.

    Solves ``min_{b0, b} (1 / N_eff) sum_i w_i (y_i - b0 - x_i^T b)^2 + alpha ||b||^2``
    with an unpenalized intercept.
    """
    w = np.asarray(participant_weights, dtype=float)
    w = w / np.sum(w)
    p_cov = X_cov_train.shape[1]

    if Z_spec_train is not None and Z_spec_train.shape[1] > 0:
        z_mean = np.sum(w[:, None] * Z_spec_train, axis=0)
        z_cent = Z_spec_train - z_mean
        z_var = np.sum(w[:, None] * (z_cent**2), axis=0)
        z_scale = np.sqrt(np.maximum(z_var, 1e-12))
        Z_std = z_cent / z_scale
        X = np.hstack([X_cov_train, Z_std])
    else:
        z_mean = np.empty(0, dtype=float)
        z_scale = np.empty(0, dtype=float)
        X = X_cov_train

    y_mean = float(np.sum(w * y_train))
    y_cent = y_train - y_mean
    alpha_arr = np.asarray(alphas, dtype=float)

    if X.shape[1] == 0:
        return MultiAlphaRidgeResult(
            alphas=alpha_arr,
            intercepts=np.full(alpha_arr.size, y_mean, dtype=float),
            coef_cov=np.zeros((alpha_arr.size, 0), dtype=float),
            coef_spec=np.zeros((alpha_arr.size, 0), dtype=float),
            z_mean=z_mean,
            z_scale=z_scale,
        )

    x_mean = np.sum(w[:, None] * X, axis=0)
    X_cent = X - x_mean
    sqrt_w = np.sqrt(w)
    Xw = X_cent * sqrt_w[:, None]
    yw = y_cent * sqrt_w

    U, s, Vt = linalg.svd(Xw, full_matrices=False, check_finite=True)
    uty = U.T @ yw
    # Shrinkage factors for each alpha: shape (n_alphas, rank)
    shrink = s[None, :] / (s[None, :] ** 2 + alpha_arr[:, None])
    betas = (shrink * uty[None, :]) @ Vt
    intercepts = y_mean - betas @ x_mean

    return MultiAlphaRidgeResult(
        alphas=alpha_arr,
        intercepts=intercepts,
        coef_cov=betas[:, :p_cov],
        coef_spec=betas[:, p_cov:],
        z_mean=z_mean,
        z_scale=z_scale,
    )


def predict_ridge_grid(
    fit: MultiAlphaRidgeResult,
    X_cov: np.ndarray,
    Z_spec: np.ndarray | None,
    *,
    alpha_idx: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return full predictions and spectral contributions s = beta_Z^T ((z - mean)/scale)."""
    if Z_spec is not None and Z_spec.shape[1] > 0:
        Z_std = (Z_spec - fit.z_mean) / fit.z_scale
    else:
        Z_std = np.zeros((X_cov.shape[0], 0), dtype=float)

    if alpha_idx is not None:
        b0 = float(fit.intercepts[alpha_idx])
        b_cov = fit.coef_cov[alpha_idx]
        b_spec = fit.coef_spec[alpha_idx]
        spec_contrib = Z_std @ b_spec if Z_std.shape[1] > 0 else np.zeros(X_cov.shape[0], dtype=float)
        cov_contrib = X_cov @ b_cov if X_cov.shape[1] > 0 else np.zeros(X_cov.shape[0], dtype=float)
        return b0 + cov_contrib + spec_contrib, spec_contrib

    spec_mat = Z_std @ fit.coef_spec.T if Z_std.shape[1] > 0 else np.zeros((X_cov.shape[0], fit.alphas.size))
    cov_mat = X_cov @ fit.coef_cov.T if X_cov.shape[1] > 0 else np.zeros((X_cov.shape[0], fit.alphas.size))
    return fit.intercepts[None, :] + cov_mat + spec_mat, spec_mat


@dataclass
class WeightedPLSResult:
    """Multi-component 1D weighted PLS regression model."""

    max_components: int
    x_mean: np.ndarray
    y_mean: float
    p_cov: int
    coefs_by_comp: np.ndarray  # shape (max_components, p_total)

    @classmethod
    def fit(
        cls,
        X_cov_train: np.ndarray,
        X_spec_train: np.ndarray,
        y_train: np.ndarray,
        participant_weights: np.ndarray,
        max_components: int,
    ) -> "WeightedPLSResult":
        w = np.asarray(participant_weights, dtype=float)
        w = w / np.sum(w)
        X = np.hstack([X_cov_train, X_spec_train])
        p_cov = X_cov_train.shape[1]

        x_mean = np.sum(w[:, None] * X, axis=0)
        y_mean = float(np.sum(w * y_train))
        Xk = X - x_mean
        yk = y_train - y_mean

        n_comp = max(1, min(int(max_components), X.shape[0] - 2, X.shape[1]))
        W_list: list[np.ndarray] = []
        P_list: list[np.ndarray] = []
        q_list: list[float] = []
        coefs = np.zeros((n_comp, X.shape[1]), dtype=float)

        for comp_idx in range(n_comp):
            wk = Xk.T @ (w * yk)
            w_norm = float(linalg.norm(wk))
            if not np.isfinite(w_norm) or w_norm <= 1e-12:
                if comp_idx > 0:
                    coefs[comp_idx:] = coefs[comp_idx - 1]
                break
            wk = wk / w_norm
            tk = Xk @ wk
            t_sq = float(np.sum(w * (tk**2)))
            if not np.isfinite(t_sq) or t_sq <= 1e-12:
                if comp_idx > 0:
                    coefs[comp_idx:] = coefs[comp_idx - 1]
                break
            pk = (Xk.T @ (w * tk)) / t_sq
            qk = float(np.sum(w * tk * yk) / t_sq)

            Xk = Xk - np.outer(tk, pk)
            yk = yk - tk * qk

            W_list.append(wk)
            P_list.append(pk)
            q_list.append(qk)

            W_mat = np.column_stack(W_list)
            P_mat = np.column_stack(P_list)
            q_vec = np.asarray(q_list, dtype=float)
            # R = W (P^T W)^{-1}, beta = R q
            ptw = P_mat.T @ W_mat
            r_q = linalg.solve(ptw, q_vec)
            coefs[comp_idx] = W_mat @ r_q

        return cls(
            max_components=n_comp,
            x_mean=x_mean,
            y_mean=y_mean,
            p_cov=p_cov,
            coefs_by_comp=coefs,
        )

    def predict(
        self,
        X_cov: np.ndarray,
        X_spec: np.ndarray,
        *,
        n_components: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        X_cent = np.hstack([X_cov, X_spec]) - self.x_mean
        X_cov_cent = X_cent[:, : self.p_cov]
        X_spec_cent = X_cent[:, self.p_cov :]

        if n_components is not None:
            idx = min(max(int(n_components), 1), self.max_components) - 1
            beta = self.coefs_by_comp[idx]
            cov_part = X_cov_cent @ beta[: self.p_cov] if self.p_cov > 0 else 0.0
            spec_part = X_spec_cent @ beta[self.p_cov :]
            return self.y_mean + cov_part + spec_part, spec_part

        cov_part = X_cov_cent @ self.coefs_by_comp[:, : self.p_cov].T if self.p_cov > 0 else 0.0
        spec_part = X_spec_cent @ self.coefs_by_comp[:, self.p_cov :].T
        return self.y_mean + cov_part + spec_part, spec_part


def make_participant_splits(
    n_participants: int,
    paired_mask: np.ndarray,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create deterministic participant-grouped folds stratified by pair availability."""
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


def compute_icc_a1(y_a: np.ndarray, y_b: np.ndarray) -> dict[str, Any]:
    """Two-way mixed-effects, single-measure, absolute-agreement ICC(A,1) per Koo & Li (2016).

    Formula:
        ICC(A,1) = (MS_R - MS_E) / (MS_R + (k - 1)*MS_E + (k / n)*(MS_C - MS_E))
    with k = 2 fixed positions and n random participants. Zero and negative estimates
    are retained; marked non-estimable only for n < 2 or non-positive/degenerate denominator.
    """
    a = np.asarray(y_a, dtype=float)
    b = np.asarray(y_b, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    a = a[valid]
    b = b[valid]
    n = int(a.size)
    if n < 2:
        return {
            "icc_a1": float("nan"),
            "estimable": False,
            "reason": "fewer_than_two_paired_participants",
            "n_pairs": n,
            "mean_diff": float("nan"),
            "rmsd": float("nan"),
        }

    Y = np.column_stack([a, b])
    k = 2
    grand_mean = float(np.mean(Y))
    row_means = np.mean(Y, axis=1)
    col_means = np.mean(Y, axis=0)

    ss_r = float(k * np.sum((row_means - grand_mean) ** 2))
    ss_c = float(n * np.sum((col_means - grand_mean) ** 2))
    ss_tot = float(np.sum((Y - grand_mean) ** 2))
    ss_e = max(0.0, ss_tot - ss_r - ss_c)

    ms_r = ss_r / float(n - 1)
    ms_c = ss_c / float(k - 1)
    ms_e = ss_e / float((n - 1) * (k - 1))

    denom = ms_r + (k - 1) * ms_e + (float(k) / float(n)) * (ms_c - ms_e)
    diffs = a - b
    mean_diff = float(np.mean(diffs))
    rmsd = float(np.sqrt(np.mean(diffs**2)))

    if not np.isfinite(denom) or denom <= 1e-15:
        return {
            "icc_a1": float("nan"),
            "estimable": False,
            "reason": "degenerate_denominator",
            "n_pairs": n,
            "mean_diff": mean_diff,
            "rmsd": rmsd,
        }

    icc = float((ms_r - ms_e) / denom)
    return {
        "icc_a1": icc,
        "estimable": True,
        "reason": "ok",
        "n_pairs": n,
        "mean_diff": mean_diff,
        "rmsd": rmsd,
        "ms_between_participants": ms_r,
        "ms_between_positions": ms_c,
        "ms_error": ms_e,
    }


def compute_discriminability_and_retrieval(
    z_a: np.ndarray,
    z_b: np.ndarray,
    labels_a: np.ndarray | None = None,
    labels_b: np.ndarray | None = None,
    *,
    local_window: int = 5,
) -> dict[str, float]:
    """Compute within-fold Bridgeford discriminability and bidirectional retrieval accuracy."""
    n = int(z_a.shape[0])
    if n < 2:
        return {
            "n_pairs": float(n),
            "discriminability": float("nan"),
            "retrieval_top1": float("nan"),
            "retrieval_chance": float("nan"),
            "local_retrieval_top1": float("nan"),
            "local_retrieval_chance": float("nan"),
        }

    Z = np.vstack([z_a, z_b])  # 2n x r
    diffs = Z[:, None, :] - Z[None, :, :]
    dist_mat = np.linalg.norm(diffs, axis=2)

    # Within-participant distances: d(iA, iB) = dist_mat[i, n + i]
    within_d = dist_mat[np.arange(n), np.arange(n) + n]

    # Discriminability across all 2n samples: for each sample s of participant i,
    # compare its distance to its partner s' against distances to all 2(n-1) other samples
    Wins = 0.0
    total_comparisons = 0
    for i in range(n):
        d_same = within_d[i]
        other_mask = np.ones(2 * n, dtype=bool)
        other_mask[i] = False
        other_mask[n + i] = False
        for anchor_idx in (i, n + i):
            d_others = dist_mat[anchor_idx, other_mask]
            Wins += float(np.sum(d_same < d_others) + 0.5 * np.sum(d_same == d_others))
            total_comparisons += int(d_others.size)

    discrim = Wins / float(total_comparisons) if total_comparisons > 0 else float("nan")

    # Bidirectional cross-position retrieval (A -> B and B -> A)
    cross_ab = dist_mat[:n, n:]  # n x n, diagonal is true match
    diag_d = np.diag(cross_ab)
    # Fractional tie credit if minimum distance is shared
    min_ab = np.min(cross_ab, axis=1)
    hits_ab = np.where(
        diag_d == min_ab,
        1.0 / np.maximum(np.sum(cross_ab == min_ab[:, None], axis=1), 1),
        0.0,
    )
    min_ba = np.min(cross_ab, axis=0)
    hits_ba = np.where(
        diag_d == min_ba,
        1.0 / np.maximum(np.sum(cross_ab == min_ba[None, :], axis=0), 1),
        0.0,
    )
    retrieval_top1 = float(0.5 * (np.mean(hits_ab) + np.mean(hits_ba)))
    retrieval_chance = 1.0 / float(n)

    # Local neighbor-label candidate sensitivity
    local_hits: list[float] = []
    local_chances: list[float] = []
    if labels_a is not None and labels_b is not None:
        mean_lbl = 0.5 * (np.asarray(labels_a, dtype=float) + np.asarray(labels_b, dtype=float))
        order = np.argsort(np.where(np.isfinite(mean_lbl), mean_lbl, np.inf))
        rank_pos = np.empty(n, dtype=int)
        rank_pos[order] = np.arange(n)
        for i in range(n):
            cand = np.flatnonzero(np.abs(rank_pos - rank_pos[i]) <= local_window)
            if cand.size < 2:
                continue
            d_cand_ab = cross_ab[i, cand]
            m_ab = np.min(d_cand_ab)
            local_hits.append(float(1.0 / np.sum(d_cand_ab == m_ab)) if diag_d[i] == m_ab else 0.0)
            local_chances.append(1.0 / float(cand.size))

    return {
        "n_pairs": float(n),
        "discriminability": discrim,
        "retrieval_top1": retrieval_top1,
        "retrieval_chance": retrieval_chance,
        "local_retrieval_top1": float(np.mean(local_hits)) if local_hits else float("nan"),
        "local_retrieval_chance": float(np.mean(local_chances)) if local_chances else float("nan"),
    }


def map_subspace_to_metric_orthonormal_basis(
    spec_prep: WeightedSpectralPreprocessor,
    pca_k: int,
    eigvecs_V: np.ndarray,
    p_1300: int,
    p_2850: int,
) -> np.ndarray:
    """Map paired directions back to canonical channels and orthonormalize under declared metric.

    Declared metric: equal block weight (1/2 per block) and equal channel weight within each block
    (1 / (2 * p_b)). Returns Q of shape (p_1300 + p_2850, r) with Q^T Q = I_r in metric coordinates.
    """
    # Raw channel projection matrix: (p_1300 + p_2850, r)
    pca_loadings_k = spec_prep.pca_components[:pca_k].T  # (p_total, k)
    block_scales = np.concatenate(
        [
            np.full(p_1300, spec_prep.scale_1300, dtype=float),
            np.full(p_2850, spec_prep.scale_2850, dtype=float),
        ]
    )
    raw_dirs = block_scales[:, None] * (pca_loadings_k @ eigvecs_V)

    metric_sqrt = np.concatenate(
        [
            np.full(p_1300, 1.0 / np.sqrt(2.0 * max(p_1300, 1)), dtype=float),
            np.full(p_2850, 1.0 / np.sqrt(2.0 * max(p_2850, 1)), dtype=float),
        ]
    )
    metric_dirs = metric_sqrt[:, None] * raw_dirs
    Q, _ = linalg.qr(metric_dirs, mode="economic")
    return Q


def compute_subspace_principal_angles(bases: Sequence[np.ndarray]) -> pd.DataFrame:
    """Compute pairwise principal angles (degrees) between fold-specific metric-orthonormal bases."""
    rows: list[dict[str, Any]] = []
    for i in range(len(bases)):
        for j in range(i + 1, len(bases)):
            Qi, Qj = bases[i], bases[j]
            if Qi.size == 0 or Qj.size == 0:
                continue
            s_vals = linalg.svdvals(Qi.T @ Qj)
            cosines = np.clip(s_vals, 0.0, 1.0)
            angles_deg = np.degrees(np.arccos(cosines))
            rows.append(
                {
                    "fold_a": i,
                    "fold_b": j,
                    "dim_a": int(Qi.shape[1]),
                    "dim_b": int(Qj.shape[1]),
                    "max_principal_angle_deg": float(np.max(angles_deg)),
                    "mean_principal_angle_deg": float(np.mean(angles_deg)),
                    "min_cosine": float(np.min(cosines)),
                }
            )
    return pd.DataFrame(rows)


def compute_endpoint_summary(
    y_true: np.ndarray,
    y_pred_cov: np.ndarray,
    y_pred_model: np.ndarray,
    weights: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute participant-weighted held-out LDL-C prediction metrics and descriptive calibration."""
    y = np.asarray(y_true, dtype=float)
    yc = np.asarray(y_pred_cov, dtype=float)
    ym = np.asarray(y_pred_model, dtype=float)
    w = np.ones_like(y) if weights is None else np.asarray(weights, dtype=float)
    valid = np.isfinite(y) & np.isfinite(yc) & np.isfinite(ym) & (w > 0)
    if not np.any(valid):
        return {
            "delta_mse": float("nan"),
            "relative_mse_reduction": float("nan"),
            "mse_covariate": float("nan"),
            "mse_model": float("nan"),
            "rmse_model": float("nan"),
            "mae_model": float("nan"),
            "calibration_intercept": float("nan"),
            "calibration_slope": float("nan"),
        }

    y, yc, ym, w = y[valid], yc[valid], ym[valid], w[valid]
    w_norm = w / np.sum(w)

    sq_err_c = (y - yc) ** 2
    sq_err_m = (y - ym) ** 2
    mse_c = float(np.sum(w_norm * sq_err_c))
    mse_m = float(np.sum(w_norm * sq_err_m))
    delta = float(np.sum(w_norm * (sq_err_c - sq_err_m)))
    rel_red = delta / mse_c if mse_c > 1e-15 else float("nan")
    rmse = float(np.sqrt(mse_m))
    mae = float(np.sum(w_norm * np.abs(y - ym)))

    ym_bar = float(np.sum(w_norm * ym))
    y_bar = float(np.sum(w_norm * y))
    var_ym = float(np.sum(w_norm * ((ym - ym_bar) ** 2)))
    if var_ym > 1e-12:
        slope = float(np.sum(w_norm * (ym - ym_bar) * (y - y_bar)) / var_ym)
        intercept = y_bar - slope * ym_bar
    else:
        slope = float("nan")
        intercept = float("nan")

    return {
        "delta_mse": delta,
        "relative_mse_reduction": rel_red,
        "mse_covariate": mse_c,
        "mse_model": mse_m,
        "rmse_model": rmse,
        "mae_model": mae,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def _select_inner_models_and_fit_outer(
    dataset: CohortDataset,
    y_all: np.ndarray,
    outer_train_idx: np.ndarray,
    outer_test_idx: np.ndarray,
    inner_splits: list[tuple[np.ndarray, np.ndarray]],
    participant_multiplicities: np.ndarray,
    cfg: PairedLipidConfig,
    *,
    single_sample_downstream: bool = False,
    include_band_paired: bool = False,
) -> dict[str, Any]:
    """Fit inner CV selection and outer-fold models with memoized spectral SVD & multi-alpha ridge."""
    mult_all = np.asarray(participant_multiplicities, dtype=float)
    sample_band_mat, _ = extract_sample_band_matrix(dataset)
    part_band_mat = aggregate_samples_to_participants(
        sample_band_mat, dataset, single_sample_only=single_sample_downstream
    )

    # Precompute inner division preprocessors once per inner fold
    inner_cache: list[dict[str, Any]] = []
    for in_tr_rel, in_val_rel in inner_splits:
        in_tr = outer_train_idx[in_tr_rel]
        in_val = outer_train_idx[in_val_rel]
        w_tr = mult_all[in_tr]
        w_val = mult_all[in_val]
        if np.sum(w_tr > 0) < 3 or np.sum(w_val > 0) < 1:
            raise ValueError("Inner division lacks sufficient distinct weighted participants.")

        cov_prep = CovariatePreprocessor.fit(dataset.covariate_frame, in_tr, w_tr, preset=cfg.covariate_preset)
        X_cov_tr = cov_prep.transform(dataset.covariate_frame, in_tr)
        X_cov_val = cov_prep.transform(dataset.covariate_frame, in_val)

        spec_prep = WeightedSpectralPreprocessor.fit(dataset, in_tr, w_tr, cfg=cfg)
        sample_pca_max = spec_prep.transform_pca(dataset.sample_1300, dataset.sample_2850)
        part_pca_max = aggregate_samples_to_participants(
            sample_pca_max, dataset, single_sample_only=single_sample_downstream
        )
        paired_moments = PairedMoments.from_sample_coordinates(sample_pca_max, dataset, in_tr, w_tr)

        sample_scaled = spec_prep.transform_scaled_blocks(dataset.sample_1300, dataset.sample_2850)
        part_scaled = aggregate_samples_to_participants(
            sample_scaled, dataset, single_sample_only=single_sample_downstream
        )

        entry: dict[str, Any] = {
            "in_tr": in_tr,
            "in_val": in_val,
            "w_tr": w_tr,
            "w_val": w_val,
            "y_tr": y_all[in_tr],
            "y_val": y_all[in_val],
            "X_cov_tr": X_cov_tr,
            "X_cov_val": X_cov_val,
            "spec_prep": spec_prep,
            "part_pca_max": part_pca_max,
            "paired_moments": paired_moments,
            "part_scaled": part_scaled,
        }
        if include_band_paired:
            entry["band_moments"] = PairedMoments.from_sample_coordinates(
                sample_band_mat, dataset, in_tr, w_tr
            )
        inner_cache.append(entry)

    alphas = cfg.ridge_alpha_grid
    n_alphas = len(alphas)
    total_val_weight = sum(float(np.sum(ic["w_val"])) for ic in inner_cache)

    # 1. Covariate-only ridge inner selection
    cov_sse = np.zeros(n_alphas, dtype=float)
    for ic in inner_cache:
        fit_c = fit_ridge_grid(ic["X_cov_tr"], None, ic["y_tr"], ic["w_tr"], alphas)
        preds_c, _ = predict_ridge_grid(fit_c, ic["X_cov_val"], None)
        cov_sse += np.sum(ic["w_val"][:, None] * ((ic["y_val"][:, None] - preds_c) ** 2), axis=0)
    cov_mses = cov_sse / total_val_weight
    # Tie rule: stronger regularization (larger alpha) wins exact ties
    best_cov_alpha_idx = int(np.lexsort((-np.arange(n_alphas), np.round(cov_mses, 12)))[0])

    # 2. Fixed band summaries + ridge inner selection
    band_sse = np.zeros(n_alphas, dtype=float)
    for ic in inner_cache:
        fit_b = fit_ridge_grid(
            ic["X_cov_tr"],
            part_band_mat[ic["in_tr"]],
            ic["y_tr"],
            ic["w_tr"],
            alphas,
        )
        preds_b, _ = predict_ridge_grid(fit_b, ic["X_cov_val"], part_band_mat[ic["in_val"]])
        band_sse += np.sum(ic["w_val"][:, None] * ((ic["y_val"][:, None] - preds_b) ** 2), axis=0)
    band_mses = band_sse / total_val_weight
    best_band_alpha_idx = int(np.lexsort((-np.arange(n_alphas), np.round(band_mses, 12)))[0])
    best_band_mse = float(band_mses[best_band_alpha_idx])

    # Determine common valid PCA dimensions across inner and outer divisions
    outer_w_tr = mult_all[outer_train_idx]
    outer_spec_prep = WeightedSpectralPreprocessor.fit(dataset, outer_train_idx, outer_w_tr, cfg=cfg)
    valid_k_sets = [set(ic["spec_prep"].candidate_pca_dims(cfg)) for ic in inner_cache]
    valid_k_sets.append(set(outer_spec_prep.candidate_pca_dims(cfg)))
    candidate_k = sorted(set.intersection(*valid_k_sets)) if valid_k_sets else []
    if not candidate_k:
        raise ValueError("No valid candidate PCA dimensions available across training divisions.")

    # 3. Block-scaled PCA + ridge inner selection
    best_pca_key: tuple[float, int, float] | None = None
    best_pca_setting: tuple[int, int] = (candidate_k[0], 0)
    for k in candidate_k:
        pca_sse = np.zeros(n_alphas, dtype=float)
        for ic in inner_cache:
            Z_tr = ic["part_pca_max"][ic["in_tr"], :k]
            Z_val = ic["part_pca_max"][ic["in_val"], :k]
            fit_p = fit_ridge_grid(ic["X_cov_tr"], Z_tr, ic["y_tr"], ic["w_tr"], alphas)
            preds_p, _ = predict_ridge_grid(fit_p, ic["X_cov_val"], Z_val)
            pca_sse += np.sum(ic["w_val"][:, None] * ((ic["y_val"][:, None] - preds_p) ** 2), axis=0)
        pca_mses = pca_sse / total_val_weight
        for a_idx, mse_val in enumerate(pca_mses):
            # Tie rule: lower MSE, fewer PCA dimensions k, stronger ridge penalty alpha
            cand_key = (round(float(mse_val), 12), int(k), -float(alphas[a_idx]))
            if best_pca_key is None or cand_key < best_pca_key:
                best_pca_key = cand_key
                best_pca_setting = (int(k), int(a_idx))
    best_pca_mse = float(best_pca_key[0]) if best_pca_key is not None else float("inf")

    # 4. PLS regression inner selection
    max_pls_c = max(cfg.pls_component_grid)
    pls_sse = np.zeros(max_pls_c, dtype=float)
    valid_pls_c = max_pls_c
    for ic in inner_cache:
        pls_fit = WeightedPLSResult.fit(
            ic["X_cov_tr"],
            ic["part_scaled"][ic["in_tr"]],
            ic["y_tr"],
            ic["w_tr"],
            max_pls_c,
        )
        valid_pls_c = min(valid_pls_c, pls_fit.max_components)
        preds_pls, _ = pls_fit.predict(ic["X_cov_val"], ic["part_scaled"][ic["in_val"]])
        pls_sse[:valid_pls_c] += np.sum(
            ic["w_val"][:, None] * ((ic["y_val"][:, None] - preds_pls[:, :valid_pls_c]) ** 2),
            axis=0,
        )
    pls_mses = pls_sse[:valid_pls_c] / total_val_weight
    best_pls_c = int(np.lexsort((np.arange(1, valid_pls_c + 1), np.round(pls_mses, 12)))[0]) + 1
    best_pls_mse = float(pls_mses[best_pls_c - 1])

    # 5. Paired linear representation + ridge inner selection
    best_paired_key: tuple[float, int, int, float, float] | None = None
    best_paired_setting: tuple[int, float, int, int] | None = None
    invalid_candidates: list[dict[str, Any]] = []

    for k in candidate_k:
        for lam in cfg.lambda_grid:
            # Solve generalized eigenproblem once per inner fold for (k, lam)
            fold_V: list[np.ndarray] = []
            failed_reason: str | None = None
            for ic in inner_cache:
                try:
                    V_k, _, _ = ic["paired_moments"].solve_directions(
                        k, lam, eigenvalue_tolerance=cfg.eigenvalue_tolerance
                    )
                    fold_V.append(V_k)
                except ValueError as exc:
                    failed_reason = str(exc)
                    break
            if failed_reason is not None:
                invalid_candidates.append({"k": k, "lambda": lam, "reason": failed_reason})
                continue

            for r in cfg.retained_directions:
                if r > k:
                    continue
                pair_sse = np.zeros(n_alphas, dtype=float)
                for ic, V_k in zip(inner_cache, fold_V):
                    V_r = V_k[:, :r]
                    Z_tr = ic["part_pca_max"][ic["in_tr"], :k] @ V_r
                    Z_val = ic["part_pca_max"][ic["in_val"], :k] @ V_r
                    fit_pr = fit_ridge_grid(ic["X_cov_tr"], Z_tr, ic["y_tr"], ic["w_tr"], alphas)
                    preds_pr, _ = predict_ridge_grid(fit_pr, ic["X_cov_val"], Z_val)
                    pair_sse += np.sum(
                        ic["w_val"][:, None] * ((ic["y_val"][:, None] - preds_pr) ** 2), axis=0
                    )
                pair_mses = pair_sse / total_val_weight
                for a_idx, mse_val in enumerate(pair_mses):
                    # Exact tie rule: lowest MSE, fewer retained directions r, fewer PCA dims k,
                    # stronger lambda, stronger alpha
                    cand_key = (
                        round(float(mse_val), 12),
                        int(r),
                        int(k),
                        -float(lam),
                        -float(alphas[a_idx]),
                    )
                    if best_paired_key is None or cand_key < best_paired_key:
                        best_paired_key = cand_key
                        best_paired_setting = (int(k), float(lam), int(r), int(a_idx))

    if best_paired_setting is None:
        raise ValueError("No valid paired linear candidate found in inner selection.")

    # Refit all selected models on the complete outer training division
    outer_cov_prep = CovariatePreprocessor.fit(
        dataset.covariate_frame, outer_train_idx, outer_w_tr, preset=cfg.covariate_preset
    )
    X_cov_otr = outer_cov_prep.transform(dataset.covariate_frame, outer_train_idx)
    X_cov_ote = outer_cov_prep.transform(dataset.covariate_frame, outer_test_idx)
    y_otr = y_all[outer_train_idx]

    # Outer Covariate-only Ridge
    fit_cov_outer = fit_ridge_grid(X_cov_otr, None, y_otr, outer_w_tr, [alphas[best_cov_alpha_idx]])
    pred_cov_ote, _ = predict_ridge_grid(fit_cov_outer, X_cov_ote, None, alpha_idx=0)

    # Outer Fixed Band + Ridge
    fit_band_outer = fit_ridge_grid(
        X_cov_otr,
        part_band_mat[outer_train_idx],
        y_otr,
        outer_w_tr,
        [alphas[best_band_alpha_idx]],
    )
    pred_band_ote, spec_band_ote = predict_ridge_grid(
        fit_band_outer, X_cov_ote, part_band_mat[outer_test_idx], alpha_idx=0
    )

    # Outer PCA + Ridge
    sel_pca_k, sel_pca_a_idx = best_pca_setting
    sample_pca_outer = outer_spec_prep.transform_pca(dataset.sample_1300, dataset.sample_2850)
    part_pca_outer = aggregate_samples_to_participants(
        sample_pca_outer, dataset, single_sample_only=single_sample_downstream
    )
    fit_pca_outer = fit_ridge_grid(
        X_cov_otr,
        part_pca_outer[outer_train_idx, :sel_pca_k],
        y_otr,
        outer_w_tr,
        [alphas[sel_pca_a_idx]],
    )
    pred_pca_ote, spec_pca_ote = predict_ridge_grid(
        fit_pca_outer, X_cov_ote, part_pca_outer[outer_test_idx, :sel_pca_k], alpha_idx=0
    )

    # Outer PLS Regression
    sample_scaled_outer = outer_spec_prep.transform_scaled_blocks(dataset.sample_1300, dataset.sample_2850)
    part_scaled_outer = aggregate_samples_to_participants(
        sample_scaled_outer, dataset, single_sample_only=single_sample_downstream
    )
    fit_pls_outer = WeightedPLSResult.fit(
        X_cov_otr,
        part_scaled_outer[outer_train_idx],
        y_otr,
        outer_w_tr,
        best_pls_c,
    )
    pred_pls_ote, spec_pls_ote = fit_pls_outer.predict(
        X_cov_ote, part_scaled_outer[outer_test_idx], n_components=best_pls_c
    )

    # Outer Paired Linear + Ridge
    sel_k, sel_lam, sel_r, sel_a_idx = best_paired_setting
    outer_moments = PairedMoments.from_sample_coordinates(
        sample_pca_outer, dataset, outer_train_idx, outer_w_tr
    )
    V_outer_k, eigvals_outer, tau_outer = outer_moments.solve_directions(
        sel_k, sel_lam, eigenvalue_tolerance=cfg.eigenvalue_tolerance
    )
    V_outer_r = V_outer_k[:, :sel_r]
    sample_paired_z = sample_pca_outer[:, :sel_k] @ V_outer_r
    part_paired_z = aggregate_samples_to_participants(
        sample_paired_z, dataset, single_sample_only=single_sample_downstream
    )
    fit_paired_outer = fit_ridge_grid(
        X_cov_otr,
        part_paired_z[outer_train_idx],
        y_otr,
        outer_w_tr,
        [alphas[sel_a_idx]],
    )
    pred_paired_ote, spec_paired_ote = predict_ridge_grid(
        fit_paired_outer, X_cov_ote, part_paired_z[outer_test_idx], alpha_idx=0
    )

    # Select strongest established comparator using inner validation MSE only
    comp_inner_mses = {
        "fixed_band_ridge": best_band_mse,
        "pca_ridge": best_pca_mse,
        "pls_regression": best_pls_mse,
    }
    strongest_comp_name = min(comp_inner_mses, key=lambda name: comp_inner_mses[name])
    comp_preds_map = {
        "fixed_band_ridge": pred_band_ote,
        "pca_ridge": pred_pca_ote,
        "pls_regression": pred_pls_ote,
    }

    # Evaluate per-sample (A vs B) repeatability on paired outer-test participants
    paired_test_pos = np.isin(dataset.paired_participant_indices, outer_test_idx)
    paired_test_p = dataset.paired_participant_indices[paired_test_pos]
    idx_a_te = dataset.pair_sample_a_indices[paired_test_pos]
    idx_b_te = dataset.pair_sample_b_indices[paired_test_pos]

    repeatability_info: dict[str, Any] = {"n_paired_test": int(paired_test_p.size)}
    if paired_test_p.size >= 2:
        X_cov_pair_te = outer_cov_prep.transform(dataset.covariate_frame, paired_test_p)
        y_pair_a, s_pair_a = predict_ridge_grid(
            fit_paired_outer, X_cov_pair_te, sample_paired_z[idx_a_te], alpha_idx=0
        )
        y_pair_b, s_pair_b = predict_ridge_grid(
            fit_paired_outer, X_cov_pair_te, sample_paired_z[idx_b_te], alpha_idx=0
        )
        y_band_a, s_band_a = predict_ridge_grid(
            fit_band_outer, X_cov_pair_te, sample_band_mat[idx_a_te], alpha_idx=0
        )
        y_band_b, s_band_b = predict_ridge_grid(
            fit_band_outer, X_cov_pair_te, sample_band_mat[idx_b_te], alpha_idx=0
        )
        repeatability_info.update(
            {
                "paired_full_icc": compute_icc_a1(y_pair_a, y_pair_b),
                "paired_spectral_icc": compute_icc_a1(s_pair_a, s_pair_b),
                "fixed_band_full_icc": compute_icc_a1(y_band_a, y_band_b),
                "fixed_band_spectral_icc": compute_icc_a1(s_band_a, s_band_b),
                "paired_discriminability": compute_discriminability_and_retrieval(
                    sample_paired_z[idx_a_te],
                    sample_paired_z[idx_b_te],
                    dataset.sample_parsed_labels[idx_a_te],
                    dataset.sample_parsed_labels[idx_b_te],
                    local_window=cfg.local_candidate_window,
                ),
                "pca_discriminability": compute_discriminability_and_retrieval(
                    sample_pca_outer[idx_a_te, :sel_pca_k],
                    sample_pca_outer[idx_b_te, :sel_pca_k],
                    dataset.sample_parsed_labels[idx_a_te],
                    dataset.sample_parsed_labels[idx_b_te],
                    local_window=cfg.local_candidate_window,
                ),
                "band_discriminability": compute_discriminability_and_retrieval(
                    sample_band_mat[idx_a_te],
                    sample_band_mat[idx_b_te],
                    dataset.sample_parsed_labels[idx_a_te],
                    dataset.sample_parsed_labels[idx_b_te],
                    local_window=cfg.local_candidate_window,
                ),
            }
        )

    subspace_basis = map_subspace_to_metric_orthonormal_basis(
        outer_spec_prep,
        sel_k,
        V_outer_r,
        dataset.wavenumbers_1300.size,
        dataset.wavenumbers_2850.size,
    )

    out = {
        "predictions": {
            "covariate_ridge": pred_cov_ote,
            "fixed_band_ridge": pred_band_ote,
            "pca_ridge": pred_pca_ote,
            "pls_regression": pred_pls_ote,
            "paired_linear_ridge": pred_paired_ote,
            "strongest_comparator": comp_preds_map[strongest_comp_name],
        },
        "spectral_contributions": {
            "fixed_band_ridge": spec_band_ote,
            "pca_ridge": spec_pca_ote,
            "pls_regression": spec_pls_ote,
            "paired_linear_ridge": spec_paired_ote,
        },
        "selected_settings": {
            "covariate_ridge_alpha": float(alphas[best_cov_alpha_idx]),
            "fixed_band_ridge_alpha": float(alphas[best_band_alpha_idx]),
            "pca_ridge_k": int(sel_pca_k),
            "pca_ridge_alpha": float(alphas[sel_pca_a_idx]),
            "pls_components": int(best_pls_c),
            "paired_k": int(sel_k),
            "paired_lambda": float(sel_lam),
            "paired_r": int(sel_r),
            "paired_alpha": float(alphas[sel_a_idx]),
            "strongest_comparator_name": strongest_comp_name,
            "n_paired_distinct_train": int(outer_moments.n_paired_distinct),
            "n_paired_multiplicity_train": float(outer_moments.n_paired_multiplicity),
            "numerical_rank_train": int(outer_spec_prep.numerical_rank),
            "mean_ab_shift_l2": float(linalg.norm(outer_moments.mean_ab_shift[:sel_k])),
            "tau": float(tau_outer),
            "top_rho": float(eigvals_outer[0]),
            "dropped_covariates": outer_cov_prep.dropped_columns,
            "n_invalid_candidates": len(invalid_candidates),
        },
        "repeatability": repeatability_info,
        "subspace_basis": subspace_basis,
        "sample_paired_z": sample_paired_z,
        "part_paired_z": part_paired_z,
    }

    if include_band_paired:
        # Secondary band-input paired ablation
        outer_band_moments = PairedMoments.from_sample_coordinates(
            sample_band_mat, dataset, outer_train_idx, outer_w_tr
        )
        k_band = sample_band_mat.shape[1]
        V_b, _, _ = outer_band_moments.solve_directions(
            k_band, sel_lam, eigenvalue_tolerance=cfg.eigenvalue_tolerance
        )
        Z_b_sample = sample_band_mat @ V_b[:, : min(sel_r, k_band)]
        Z_b_part = aggregate_samples_to_participants(
            Z_b_sample, dataset, single_sample_only=single_sample_downstream
        )
        fit_bp = fit_ridge_grid(
            X_cov_otr,
            Z_b_part[outer_train_idx],
            y_otr,
            outer_w_tr,
            [alphas[sel_a_idx]],
        )
        pred_bp, _ = predict_ridge_grid(fit_bp, X_cov_ote, Z_b_part[outer_test_idx], alpha_idx=0)
        out["predictions"]["band_paired_ridge"] = pred_bp

    return out


def evaluate_nested_cohort(
    dataset: CohortDataset,
    *,
    cfg: PairedLipidConfig = PairedLipidConfig(),
    endpoint: str | None = None,
    outer_seed: int | None = None,
    participant_multiplicities: np.ndarray | None = None,
    precomputed_splits: tuple[
        list[tuple[np.ndarray, np.ndarray]],
        list[list[tuple[np.ndarray, np.ndarray]]],
    ]
    | None = None,
    single_sample_downstream: bool = False,
    include_band_paired: bool = False,
) -> dict[str, Any]:
    """Execute the locked 5x3 nested cross-validation procedure on the cohort."""
    target_col = endpoint or cfg.primary_endpoint
    if target_col not in dataset.outcomes:
        raise ValueError(f"Endpoint '{target_col}' not present in dataset outcomes.")
    y_all = dataset.outcomes[target_col]
    if not np.all(np.isfinite(y_all)):
        raise ValueError(f"Endpoint '{target_col}' contains missing values among eligible participants.")

    n_p = dataset.n_participants
    mult = (
        np.ones(n_p, dtype=float)
        if participant_multiplicities is None
        else np.asarray(participant_multiplicities, dtype=float)
    )

    paired_mask = np.isin(np.arange(n_p), dataset.paired_participant_indices)
    seed_to_use = cfg.primary_seed if outer_seed is None else int(outer_seed)

    if precomputed_splits is None:
        outer_splits = make_participant_splits(n_p, paired_mask, cfg.n_outer_splits, seed_to_use)
        inner_splits_per_fold = [
            make_participant_splits(
                tr_idx.size,
                paired_mask[tr_idx],
                cfg.n_inner_splits,
                seed_to_use + 1000 * (fold_idx + 1),
            )
            for fold_idx, (tr_idx, _) in enumerate(outer_splits)
        ]
    else:
        outer_splits, inner_splits_per_fold = precomputed_splits

    model_names = [
        "covariate_ridge",
        "fixed_band_ridge",
        "pca_ridge",
        "pls_regression",
        "paired_linear_ridge",
        "strongest_comparator",
    ]
    if include_band_paired:
        model_names.append("band_paired_ridge")

    oof_preds = {m: np.full(n_p, np.nan, dtype=float) for m in model_names}
    oof_spec = {
        m: np.full(n_p, np.nan, dtype=float)
        for m in ("fixed_band_ridge", "pca_ridge", "pls_regression", "paired_linear_ridge")
    }
    fold_settings: list[dict[str, Any]] = []
    fold_repeatability: list[dict[str, Any]] = []
    subspace_bases: list[np.ndarray] = []

    for fold_idx, ((tr_idx, te_idx), in_splits) in enumerate(zip(outer_splits, inner_splits_per_fold)):
        if np.sum(mult[te_idx] > 0) == 0:
            raise ValueError(f"Outer fold {fold_idx} has zero weighted test participants.")
        fold_out = _select_inner_models_and_fit_outer(
            dataset,
            y_all,
            tr_idx,
            te_idx,
            in_splits,
            mult,
            cfg,
            single_sample_downstream=single_sample_downstream,
            include_band_paired=include_band_paired,
        )
        for m in model_names:
            oof_preds[m][te_idx] = fold_out["predictions"][m]
        for m in oof_spec:
            oof_spec[m][te_idx] = fold_out["spectral_contributions"][m]

        settings_row = {"outer_fold": fold_idx, **fold_out["selected_settings"]}
        fold_settings.append(settings_row)
        rep_row = {"outer_fold": fold_idx, **fold_out["repeatability"]}
        fold_repeatability.append(rep_row)
        subspace_bases.append(fold_out["subspace_basis"])

    # Compute endpoint summaries for every model relative to covariate-only ridge
    comparisons: dict[str, dict[str, float]] = {}
    for m in model_names:
        comparisons[m] = compute_endpoint_summary(
            y_all, oof_preds["covariate_ridge"], oof_preds[m], weights=mult
        )

    # Also compute direct paired_linear_ridge vs strongest_comparator difference
    vs_strongest = compute_endpoint_summary(
        y_all, oof_preds["strongest_comparator"], oof_preds["paired_linear_ridge"], weights=mult
    )

    angles_df = compute_subspace_principal_angles(subspace_bases)
    return {
        "endpoint": target_col,
        "outer_seed": seed_to_use,
        "oof_predictions": oof_preds,
        "oof_spectral_contributions": oof_spec,
        "comparisons_vs_covariate_ridge": comparisons,
        "paired_vs_strongest_comparator": vs_strongest,
        "fold_settings": pd.DataFrame(fold_settings),
        "fold_repeatability": fold_repeatability,
        "subspace_angles": angles_df,
        "splits": (outer_splits, inner_splits_per_fold),
    }


def run_multiplicity_bootstrap(
    dataset: CohortDataset,
    *,
    cfg: PairedLipidConfig = PairedLipidConfig(),
    n_bootstraps: int = 2000,
    bootstrap_seed: int = 20261106,
    endpoint: str | None = None,
) -> dict[str, Any]:
    """Participant-level multiplicity bootstrap keeping original fold memberships fixed."""
    base_eval = evaluate_nested_cohort(dataset, cfg=cfg, endpoint=endpoint)
    splits = base_eval["splits"]
    n_p = dataset.n_participants
    rng = np.random.default_rng(bootstrap_seed)

    delta_paired: list[float] = []
    delta_vs_comp: list[float] = []
    distinct_counts: list[int] = []
    failed_draws = 0
    failure_reasons: list[str] = []

    for _ in range(n_bootstraps):
        drawn = rng.integers(0, n_p, size=n_p)
        mult = np.bincount(drawn, minlength=n_p).astype(float)
        distinct_counts.append(int(np.sum(mult > 0)))
        try:
            b_res = evaluate_nested_cohort(
                dataset,
                cfg=cfg,
                endpoint=endpoint,
                participant_multiplicities=mult,
                precomputed_splits=splits,
            )
            d_pair = b_res["comparisons_vs_covariate_ridge"]["paired_linear_ridge"]["delta_mse"]
            d_comp = b_res["paired_vs_strongest_comparator"]["delta_mse"]
            if not (np.isfinite(d_pair) and np.isfinite(d_comp)):
                failed_draws += 1
                failure_reasons.append("nonfinite_bootstrap_estimate")
            else:
                delta_paired.append(float(d_pair))
                delta_vs_comp.append(float(d_comp))
        except Exception as exc:
            failed_draws += 1
            failure_reasons.append(str(exc))

    # Per study design: if any bootstrap refit cannot produce the locked comparison,
    # mark that dataset's interval procedure as failed rather than discarding resamples.
    interval_valid = failed_draws == 0 and len(delta_paired) == n_bootstraps
    if interval_valid:
        arr_p = np.asarray(delta_paired, dtype=float)
        arr_c = np.asarray(delta_vs_comp, dtype=float)
        ci_paired = (float(np.percentile(arr_p, 2.5)), float(np.percentile(arr_p, 97.5)))
        ci_vs_comp = (float(np.percentile(arr_c, 2.5)), float(np.percentile(arr_c, 97.5)))
        mc_se = float(np.std(arr_p, ddof=1) / np.sqrt(n_bootstraps))
    else:
        ci_paired = (float("nan"), float("nan"))
        ci_vs_comp = (float("nan"), float("nan"))
        mc_se = float("nan")

    return {
        "point_estimate_delta_mse": base_eval["comparisons_vs_covariate_ridge"]["paired_linear_ridge"]["delta_mse"],
        "point_estimate_vs_comparator": base_eval["paired_vs_strongest_comparator"]["delta_mse"],
        "ci_95_delta_mse": ci_paired,
        "ci_95_vs_comparator": ci_vs_comp,
        "interval_procedure_succeeded": interval_valid,
        "n_bootstraps_requested": n_bootstraps,
        "n_bootstraps_succeeded": len(delta_paired),
        "n_failed_refits": failed_draws,
        "failure_reasons": failure_reasons[:10],
        "mean_distinct_participants": float(np.mean(distinct_counts)) if distinct_counts else float("nan"),
        "monte_carlo_se": mc_se,
        "bootstrap_draws_delta_mse": np.asarray(delta_paired, dtype=float),
    }


def generate_synthetic_cohort(
    *,
    n_participants: int = 92,
    n_single_sample: int = 2,
    maps_per_sample: int = 3,
    seed: int = 42,
    compact_grid: bool = False,
    signal_strength: float = 0.8,
    ab_shift_scale: float = 0.15,
    nonlinear_covariate_proxy: bool = False,
    formula_linked_ldl: bool = False,
    shared_prep_scale: float = 0.3,
    label_drift_scale: float = 0.1,
) -> tuple[CohortDataset, dict[str, Any]]:
    """Generate a synthetic paired Raman cohort with known latent biological and technical factors."""
    rng = np.random.default_rng(seed)
    step = 10.0 if compact_grid else 1.0
    wn_1300 = np.arange(734.0, 1788.0 + 0.5 * step, step, dtype=float)
    wn_2850 = np.arange(2451.0, 3197.0 + 0.5 * step, step, dtype=float)

    def _gaussian_peak(axis: np.ndarray, center: float, width: float) -> np.ndarray:
        v = np.exp(-0.5 * ((axis - center) / width) ** 2)
        return v / max(float(linalg.norm(v)), 1e-12)

    # Smooth biological basis functions overlapping lipid-associated and protein-associated regions
    bio_1300_1 = _gaussian_peak(wn_1300, 1440.0, 14.0) + 0.7 * _gaussian_peak(wn_1300, 1301.0, 16.0)
    bio_2850_1 = _gaussian_peak(wn_2850, 2853.0, 12.0) + 0.6 * _gaussian_peak(wn_2850, 2890.0, 18.0)

    bio_1300_2 = _gaussian_peak(wn_1300, 1657.0, 18.0) - 0.5 * _gaussian_peak(wn_1300, 1089.0, 12.0)
    bio_2850_2 = _gaussian_peak(wn_2850, 2937.0, 15.0) - 0.4 * _gaussian_peak(wn_2850, 2853.0, 12.0)

    # Technical nuisance/disagreement basis functions (orthogonalizable or distinct peaks)
    tech_1300 = _gaussian_peak(wn_1300, 1156.0, 20.0) - 0.6 * _gaussian_peak(wn_1300, 980.0, 15.0)
    tech_2850 = _gaussian_peak(wn_2850, 3015.0, 22.0)

    age = rng.normal(68.0, 11.0, size=n_participants)
    sex_female = rng.binomial(1, 0.48, size=n_participants)
    statin = rng.binomial(1, 0.35, size=n_participants)

    # Latent biological factors z_bio (n_participants x 2)
    z_bio = rng.normal(0.0, 1.0, size=(n_participants, 2))
    age_std = (age - np.mean(age)) / max(float(np.std(age)), 1e-8)
    if nonlinear_covariate_proxy:
        # Failure mode test: spectral factor proxies nonlinear age^2 without independent biological effect on y
        z_bio[:, 0] = 0.85 * (age_std**2 - 1.0) + 0.5 * rng.normal(0.0, 1.0, size=n_participants)

    shared_prep = shared_prep_scale * rng.normal(0.0, 1.0, size=n_participants)

    hdl = np.clip(1.3 + 0.25 * z_bio[:, 1] + 0.1 * sex_female + rng.normal(0.0, 0.2, size=n_participants), 0.5, 3.0)
    tg = np.clip(1.5 + 0.35 * z_bio[:, 0] - 0.2 * z_bio[:, 1] + rng.normal(0.0, 0.3, size=n_participants), 0.4, 5.0)
    plt_cnt = np.clip(240.0 + 25.0 * shared_prep + rng.normal(0.0, 35.0, size=n_participants), 100.0, 500.0)

    linear_cov = 3.1 - 0.18 * age_std - 0.45 * statin + 0.12 * sex_female
    if nonlinear_covariate_proxy:
        ldl = linear_cov + 0.45 * (age_std**2 - 1.0) + rng.normal(0.0, 0.45, size=n_participants)
    elif formula_linked_ldl:
        total_chol = 4.8 + signal_strength * z_bio[:, 0] - 0.4 * statin + rng.normal(0.0, 0.4, size=n_participants)
        ldl = total_chol - hdl - tg / 2.2
    else:
        ldl = (
            linear_cov
            + signal_strength * z_bio[:, 0]
            + 0.25 * signal_strength * z_bio[:, 1]
            + label_drift_scale * shared_prep
            + rng.normal(0.0, 0.45, size=n_participants)
        )

    clin_rows: list[dict[str, Any]] = []
    map_rows: list[dict[str, Any]] = []
    spec_1300_list: list[np.ndarray] = []
    spec_2850_list: list[np.ndarray] = []

    single_set = set(range(n_participants - n_single_sample, n_participants))
    sample_counter = 1

    for i in range(n_participants):
        pid = str(100 + i)
        clin_rows.append(
            {
                "participant_key": f"stroke_{pid}",
                "cohort": "stroke",
                "spectral_subject_id": pid,
                "has_spectra": 1,
                "spectral_link_status": "matched",
                "qc_flags": "none",
                "age_years": float(age[i]),
                "sex": "female" if sex_female[i] == 1 else "male",
                "statin": int(statin[i]),
                "hypertension": int(rng.binomial(1, 0.6)),
                "diabetes": int(rng.binomial(1, 0.25)),
                "atrial_fibrillation": int(rng.binomial(1, 0.2)),
                "coronary_artery_disease": int(rng.binomial(1, 0.18)),
                "smoking_status": rng.choice(["never", "former", "current"], p=[0.5, 0.3, 0.2]),
                "antiplatelet": int(rng.binomial(1, 0.3)),
                "anticoagulant": int(rng.binomial(1, 0.15)),
                "ldl_mmol_l": float(ldl[i]),
                "hdl_mmol_l": float(hdl[i]),
                "triglycerides_mmol_l": float(tg[i]),
                "plt_10e9_l": float(plt_cnt[i]),
            }
        )
        m_i = 1 if i in single_set else 2
        for s_pos in range(m_i):
            sid = f"sample_{sample_counter}"
            lbl_drift = label_drift_scale * (sample_counter / float(2 * n_participants))
            sample_counter += 1

            ab_shift = (ab_shift_scale if s_pos == 1 else -ab_shift_scale)
            sample_err = rng.normal(0.0, 0.65) + ab_shift + lbl_drift
            base_1300 = (
                (z_bio[i, 0] + shared_prep[i]) * bio_1300_1
                + z_bio[i, 1] * bio_1300_2
                + sample_err * tech_1300
            )
            base_2850 = (
                (z_bio[i, 0] + shared_prep[i]) * bio_2850_1
                + z_bio[i, 1] * bio_2850_2
                + sample_err * tech_2850
            )

            mode = "SS" if (i + s_pos) % 3 != 0 else "FP"
            for map_idx in range(maps_per_sample):
                row_idx = len(spec_1300_list)
                noise_1300 = rng.normal(0.0, 0.04, size=wn_1300.size)
                noise_2850 = rng.normal(0.0, 0.04, size=wn_2850.size)
                spec_1300_list.append(base_1300 + noise_1300)
                spec_2850_list.append(base_2850 + noise_2850)
                for rng_code in ("1300", "2850"):
                    map_rows.append(
                        {
                            "participant_id": pid,
                            "sample_id": sid,
                            "spectral_range": rng_code,
                            "row_index": row_idx,
                            "acquisition_mode": mode,
                            "retained": True,
                            "map_number": map_idx + 1,
                        }
                    )

    dataset = build_paired_cohort_dataset(
        pd.DataFrame(clin_rows),
        pd.DataFrame(map_rows),
        np.vstack(spec_1300_list),
        np.vstack(spec_2850_list),
        wn_1300,
        wn_2850,
    )
    truth = {
        "bio_1300": np.column_stack([bio_1300_1, bio_1300_2]),
        "bio_2850": np.column_stack([bio_2850_1, bio_2850_2]),
        "z_bio": z_bio,
        "signal_strength": signal_strength,
        "nonlinear_covariate_proxy": nonlinear_covariate_proxy,
    }
    return dataset, truth


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Compute the Wilson score interval for a binomial proportion."""
    if total <= 0:
        return (float("nan"), float("nan"))
    z = float(stats.norm.ppf(0.5 + 0.5 * confidence))
    phat = float(successes) / float(total)
    denom = 1.0 + (z**2) / float(total)
    center = (phat + (z**2) / (2.0 * float(total))) / denom
    half = (z / denom) * math.sqrt((phat * (1.0 - phat) / float(total)) + ((z**2) / (4.0 * (float(total) ** 2))))
    return (max(0.0, center - half), min(1.0, center + half))


def evaluate_bootstrap_calibration(
    *,
    n_repetitions: int = 20,
    n_bootstraps: int = 100,
    n_participants: int = 60,
    signal_strength: float = 0.8,
    base_seed: int = 9000,
    cfg: PairedLipidConfig = PairedLipidConfig(),
) -> dict[str, Any]:
    """Evaluate bootstrap interval coverage, width, bias, and failure rate in synthetic repetitions."""
    # Estimate target risk difference from independent synthetic validation draws
    rng = np.random.default_rng(base_seed)
    rep_seeds = rng.integers(10_000, 10_000_000, size=n_repetitions)

    point_estimates: list[float] = []
    ci_lows: list[float] = []
    ci_highs: list[float] = []
    succeeded_flags: list[bool] = []

    for rep_idx in range(n_repetitions):
        ds, _ = generate_synthetic_cohort(
            n_participants=n_participants,
            seed=int(rep_seeds[rep_idx]),
            compact_grid=True,
            signal_strength=signal_strength,
        )
        boot = run_multiplicity_bootstrap(
            ds,
            cfg=cfg,
            n_bootstraps=n_bootstraps,
            bootstrap_seed=int(rep_seeds[rep_idx]) + 1,
        )
        point_estimates.append(float(boot["point_estimate_delta_mse"]))
        ci_lows.append(float(boot["ci_95_delta_mse"][0]))
        ci_highs.append(float(boot["ci_95_delta_mse"][1]))
        succeeded_flags.append(bool(boot["interval_procedure_succeeded"]))

    target_delta = float(np.nanmean(point_estimates))
    uncond_covered = 0
    cond_covered = 0
    n_success = 0
    widths: list[float] = []

    for est, lo, hi, ok in zip(point_estimates, ci_lows, ci_highs, succeeded_flags):
        if ok and np.isfinite(lo) and np.isfinite(hi):
            n_success += 1
            widths.append(hi - lo)
            if lo <= target_delta <= hi:
                uncond_covered += 1
                cond_covered += 1

    uncond_cov = float(uncond_covered) / float(n_repetitions) if n_repetitions > 0 else float("nan")
    cond_cov = float(cond_covered) / float(n_success) if n_success > 0 else float("nan")
    fail_rate = float(n_repetitions - n_success) / float(n_repetitions) if n_repetitions > 0 else float("nan")
    wilson_lo, wilson_hi = wilson_score_interval(uncond_covered, n_repetitions, confidence=0.95)
    passes_gate = bool(wilson_lo >= 0.925 and fail_rate < 0.01)

    return {
        "n_repetitions": n_repetitions,
        "n_bootstraps": n_bootstraps,
        "target_delta_mse": target_delta,
        "mean_point_estimate": float(np.nanmean(point_estimates)),
        "bias": float(np.nanmean(point_estimates) - target_delta),
        "unconditional_coverage": uncond_cov,
        "conditional_coverage": cond_cov,
        "wilson_95_lower": wilson_lo,
        "wilson_95_upper": wilson_hi,
        "dataset_failure_rate": fail_rate,
        "mean_interval_width": float(np.mean(widths)) if widths else float("nan"),
        "passes_provisional_gate": passes_gate,
    }


def summarize_anchor_associations(
    dataset: CohortDataset,
    eval_result: dict[str, Any],
    *,
    anchors: Sequence[str] = ("ldl_mmol_l", "hdl_mmol_l", "triglycerides_mmol_l", "plt_10e9_l"),
    n_resamples: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute continuous anchor associations and resample stability without categorical source claims."""
    spec_contrib = eval_result["oof_spectral_contributions"]["paired_linear_ridge"]
    cov_prep = CovariatePreprocessor.fit(
        dataset.covariate_frame,
        np.arange(dataset.n_participants),
        np.ones(dataset.n_participants),
    )
    X_cov = cov_prep.transform(dataset.covariate_frame, np.arange(dataset.n_participants))
    rng = np.random.default_rng(seed)

    rows: list[dict[str, Any]] = []
    for anchor in anchors:
        if anchor not in dataset.outcomes:
            continue
        y_a = dataset.outcomes[anchor]
        valid = np.isfinite(spec_contrib) & np.isfinite(y_a)
        if np.sum(valid) < 5:
            continue
        s_v = spec_contrib[valid]
        y_v = y_a[valid]
        C_v = X_cov[valid]

        # Residualize both s_v and y_v on standardized covariates + intercept
        design = np.column_stack([np.ones(s_v.size), C_v])
        q_mat, _ = linalg.qr(design, mode="economic")
        s_res = s_v - q_mat @ (q_mat.T @ s_v)
        y_res = y_v - q_mat @ (q_mat.T @ y_v)

        denom = float(linalg.norm(s_res) * linalg.norm(y_res))
        partial_r = float(np.dot(s_res, y_res) / denom) if denom > 1e-12 else 0.0
        spearman_rho = float(stats.spearmanr(s_res, y_res).statistic)

        boot_r: list[float] = []
        n_v = s_v.size
        for _ in range(n_resamples):
            idx = rng.integers(0, n_v, size=n_v)
            sr = s_res[idx] - np.mean(s_res[idx])
            yr = y_res[idx] - np.mean(y_res[idx])
            dn = float(linalg.norm(sr) * linalg.norm(yr))
            if dn > 1e-12:
                boot_r.append(float(np.dot(sr, yr) / dn))
        arr_r = np.asarray(boot_r, dtype=float)
        sign_stability = float(np.mean(np.sign(arr_r) == np.sign(partial_r))) if arr_r.size else float("nan")

        rows.append(
            {
                "anchor": anchor,
                "n_participants": int(n_v),
                "partial_pearson_r": partial_r,
                "partial_spearman_rho": spearman_rho,
                "resample_ci_low": float(np.percentile(arr_r, 2.5)) if arr_r.size else float("nan"),
                "resample_ci_high": float(np.percentile(arr_r, 97.5)) if arr_r.size else float("nan"),
                "sign_stability": sign_stability,
                "attribution_status": "continuous_exploratory_summary_only",
            }
        )
    return pd.DataFrame(rows)


def load_paired_cohort_from_store(
    clinical_master_path: str | Path = "data/clinical_master.csv",
    *,
    data_source: Any | None = None,
    cfg: PairedLipidConfig = PairedLipidConfig(),
    subset_mode: Literal["all_eligible", "paired_only", "single_sample", "three_map_complete"] = "all_eligible",
    endpoint: str | None = None,
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
        vals: list[float] = []
        for fn in feature_names:
            token = str(fn).split("__")[-1]
            vals.append(float(token))
        return np.asarray(vals, dtype=float)

    wn_1300 = _extract_wn(sum_1300["feature_names"])
    wn_2850 = _extract_wn(sum_2850["feature_names"])

    idx_map_1300 = {str(mid): i for i, mid in enumerate(sum_1300["measurement_ids"])}
    idx_map_2850 = {str(mid): i for i, mid in enumerate(sum_2850["measurement_ids"])}

    map_rows: list[dict[str, Any]] = []
    for _, row in file_index.iterrows():
        mid = str(row["measurement_id"])
        rng_code = str(row["spectral_range"])
        if rng_code == "1300" and mid in idx_map_1300:
            r_idx = idx_map_1300[mid]
        elif rng_code == "2850" and mid in idx_map_2850:
            r_idx = idx_map_2850[mid]
        else:
            continue
        map_rows.append(
            {
                "participant_id": str(row["participant_id"]),
                "sample_id": str(row["sample_id"]),
                "spectral_range": rng_code,
                "row_index": int(r_idx),
                "acquisition_mode": row.get("acquisition_mode", "NONE"),
                "retained": True,
            }
        )

    dataset = build_paired_cohort_dataset(
        clin_df,
        pd.DataFrame(map_rows),
        np.asarray(sum_1300["area_summary"], dtype=float),
        np.asarray(sum_2850["area_summary"], dtype=float),
        wn_1300,
        wn_2850,
        cfg=cfg,
        subset_mode=subset_mode,
        endpoint=endpoint,
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
            "status": "assumed_mmol_l_and_unconfirmed_formula",
            "source_reference": "data/clinical_master_dictionary.csv",
            "implication": (
                "LDL-C, HDL-C, TG, and platelet units are marked assumed in the dictionary; "
                "fasting status, blood-draw timing relative to EV isolation, and direct vs calculated "
                "LDL-C (e.g., Friedewald) are unconfirmed."
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
                "Positive delta MSE shows predictive improvement over linear ridge on C under the "
                "observed protocol; spectra may proxy nonlinear covariate effects or unmeasured "
                "technical preparation structure."
            ),
        },
        {
            "domain": "control_cohort_absence_of_lipids",
            "status": "controls_excluded_from_primary_model",
            "source_reference": "data/clinical_master.csv",
            "implication": (
                "Control participants have no recorded lipid values and only one sample per participant; "
                "within-stroke lipid prediction does not validate case-control disease specificity."
            ),
        },
    ]
    return pd.DataFrame(rows)

