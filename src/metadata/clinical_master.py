"""Read clinical-master identity and source provenance without model dependencies."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

MISSING_TOKENS: tuple[str, ...] = ("unknown", "not_applicable", "not_available")

IDENTITY_CONFLICT_FLAGS: frozenset[str] = frozenset(
    {
        "SEX_FOLDER_CONFLICT",
        "SPECTRAL_FILENAME_FOLDER_CONFLICT",
        "POSSIBLE_SPECTRA_IN_ANOTHER_FOLDER",
        "MULTIPLE_SEX_FOLDERS",
    }
)


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
        # NA tokens make pandas infer float64 ids ("7.0"), which never match store ids ("7").
        dtype={"participant_key": "string", "cohort": "string", "spectral_subject_id": "string"},
    )
    if "participant_key" not in frame.columns or "cohort" not in frame.columns:
        raise ValueError("Clinical master must contain 'participant_key' and 'cohort' columns.")
    if frame["participant_key"].isna().any() or frame["participant_key"].str.strip().eq("").any():
        raise ValueError("Clinical master participant_key must be explicit and nonempty")
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


MASTER_LINK_COLUMNS = frozenset({
    "participant_key", "cohort", "spectral_subject_id", "has_spectra", "spectral_link_status", "qc_flags",
})


def is_clinical_master(columns: object) -> bool:
    """Recognize explicit master identifiers, including an incomplete master schema."""
    return bool({"participant_key", "spectral_subject_id"}.intersection(columns))


def load_master_for_linkage(path: str | Path) -> pd.DataFrame:
    """Require explicit identity metadata before linking clinical observations."""
    columns = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns
    missing = sorted(MASTER_LINK_COLUMNS.difference(columns))
    if missing:
        raise ValueError(f"Clinical master is missing identity columns: {', '.join(missing)}")
    frame, _ = load_clinical_master_verified(path)
    if not frame["cohort"].isin(["stroke", "control"]).all():
        raise ValueError("Clinical master contains an unsupported cohort")
    has_spectra = pd.to_numeric(frame["has_spectra"], errors="raise")
    if not has_spectra.isin([0, 1]).all():
        raise ValueError("Clinical master has_spectra must be explicitly 0 or 1")
    spectral_ids = frame["spectral_subject_id"].astype("string").str.strip()
    if (has_spectra.eq(1) & (spectral_ids.isna() | spectral_ids.eq(""))).any():
        raise ValueError("Clinical master has spectra without an explicit spectral subject ID")
    frame["spectral_subject_id"] = spectral_ids
    return frame


def clinical_source_paths(cfg: dict) -> dict[str, Path]:
    """Resolve the clinical table and, for master inputs, its required dictionary."""
    from src import resolve_project_path

    data_cfg = cfg.get("data", {})
    table = resolve_project_path(data_cfg.get("clinical_data_path", "data/clinical_master.csv")).resolve()
    columns = pd.read_csv(table, nrows=0, encoding="utf-8-sig").columns
    paths = {"clinical_data_path": table}
    if is_clinical_master(columns):
        dictionary = resolve_project_path(
            data_cfg.get("clinical_dictionary_path", "data/clinical_master_dictionary.csv")
        ).resolve()
        pd.read_csv(dictionary, nrows=0, encoding="utf-8-sig")
        paths["clinical_dictionary_path"] = dictionary
    return paths
