"""Clinical-master linkage, paired cohort construction, band extraction and limitations table."""

from __future__ import annotations
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import re
from pathlib import Path
from typing import Any, Literal, Mapping
import numpy as np
import pandas as pd
from src.analysis.raman_bands import RAMAN_BANDS


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
