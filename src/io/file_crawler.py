import logging
from pathlib import Path
from typing import Literal

import pandas as pd

from src import config, resolve_project_path
from src.metadata.parser import parse_filename

logger = logging.getLogger(__name__)


def _is_reorganized_measurement_file(file_path: Path) -> bool:
    """Return True for canonical reorganized Raman exports only.

    Matches mode-free (_range_meas) and legacy mode-tagged
    (_range_SS_meas, _range_FP_meas) stems; the mode token is discarded.
    """
    stem = file_path.stem
    return "_range" in stem and "meas" in stem


def disambiguate_subject_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Split conflicting bare subject IDs into stable cohort-aware subject IDs."""
    if df.empty or "subject_id" not in df.columns:
        return df.copy()

    out = df.copy()
    out["subject_id"] = out["subject_id"].astype(str).str.strip()
    if "subject_id_raw" in out.columns:
        out["subject_id_raw"] = out["subject_id_raw"].astype(str).str.strip()
        return out

    out["subject_id_raw"] = out["subject_id"]

    gender_series = pd.Series("unknown", index=out.index, dtype="object")
    if "gender" in out.columns:
        gender_series = out["gender"].astype(str).str.strip().str.lower()
        gender_series = gender_series.replace({"": "unknown", "nan": "unknown"})

    if "is_control" not in out.columns:
        return out

    conflicting_ids: set[str] = set()
    probe = pd.DataFrame(
        {
            "subject_id_raw": out["subject_id_raw"].astype(str),
            "is_control": out["is_control"],
            "gender": gender_series,
        }
    )
    for subject_id_raw, subject_df in probe.groupby("subject_id_raw", sort=False):
        n_groups = subject_df["is_control"].dropna().astype(bool).nunique()
        n_sexes = subject_df["gender"].dropna().nunique()
        if n_groups > 1 or n_sexes > 1:
            conflicting_ids.add(str(subject_id_raw))

    if not conflicting_ids:
        return out

    condition_series = out["is_control"].map(
        lambda value: "control" if bool(value) else "disease" if pd.notna(value) else "unknown"
    )
    mask = out["subject_id_raw"].isin(conflicting_ids)
    out.loc[mask, "subject_id"] = (
        condition_series.loc[mask].astype(str)
        + "_"
        + gender_series.loc[mask].astype(str)
        + "_"
        + out.loc[mask, "subject_id_raw"].astype(str)
    )
    return out


def _deduplicate_measurements(
    df: pd.DataFrame,
    sort_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Renumber measurements when duplicate keys exist after ignoring mode.

    Within each (subject_id, slide_number, spectral_range_cm-1) group, if
    multiple rows share the same measurement_number, sort by ``sort_cols`` and
    assign sequential measurement numbers starting at 1 so all files are
    retained without using acquisition mode as a key.
    """
    if df.empty or "measurement_number" not in df.columns:
        return df.copy()

    sort_cols = [c for c in (sort_cols or ["filepath"]) if c in df.columns] or None
    group_cols = [
        c for c in ["subject_id", "slide_number", "spectral_range_cm-1"] if c in df.columns
    ]
    if not group_cols:
        return df.copy()

    out = df.copy().reset_index(drop=True)
    key_cols = group_cols + ["measurement_number"]
    if "acquisition_mode" in out.columns:
        key_cols.append("acquisition_mode")
    if not out.duplicated(subset=key_cols, keep=False).any():
        return out

    duplicates = out.loc[out.duplicated(subset=key_cols, keep=False), key_cols + ["filepath"]]
    raise ValueError(
        "Duplicate measurement identities in raw data; correct the provenance: "
        f"{duplicates.to_dict(orient='records')[:5]}"
    )




def build_file_index(
    data_dir: str | Path = None,
    measurement: int | Literal["all"] = 1,
) -> pd.DataFrame:
    """
    Crawl the reorganized `data/` directory and index measurement files.

    Parameters
    ----------
    data_dir : path-like, optional
        Root of the reorganized data tree. Defaults to ``config.data.root_dir``.
    measurement : int or "all", default 1
        Which measurement replicate to include.
        - 1, 2, or 3 → return only files matching ``_meas{N}.txt``
        - "all"       → return all 3 replicates

    The default of ``measurement=1`` remains available for callers that want
    exactly one replicate per (subject, slide, spectral_range). Current EDA,
    ML entrypoints explicitly request ``measurement="all"`` when they
    need the full replicate set.

    Returns
    -------
    pd.DataFrame
        Structured index of measurement files.
    """
    if data_dir is None:
        data_cfg = config.get("data", {})
        data_dir = resolve_project_path(
            data_cfg.get("raw_text_dir", data_cfg.get("root_dir", "data/raw"))
        )

    data_dir = Path(data_dir)
    ext = config["data"]["file_extension"].lower()

    all_records: list[dict] = []

    for file_path in data_dir.rglob(f"*{ext}"):
        if file_path.name.startswith("."):
            continue
        if "test measurements" in [p.lower() for p in file_path.parts]:
            continue
        if not _is_reorganized_measurement_file(file_path):
            continue

        metadata = parse_filename(file_path)

        if "subject_id" in metadata and "slide_number" in metadata:
            all_records.append(metadata)
        else:
            logger.warning(
                "Skipping file (missing subject_id or slide_number): %s", file_path
            )

    if not all_records:
        logger.warning("No measurement files found in %s", data_dir)
        return pd.DataFrame()

    df = pd.DataFrame(all_records)

    if "spectral_range" in df.columns:
        df = df.rename(columns={"spectral_range": "spectral_range_cm-1"})

    # Enforce str type at the IO boundary.
    if "subject_id" in df.columns:
        df["subject_id"] = df["subject_id"].astype(str)
    if "slide_number" in df.columns:
        df["slide_number"] = df["slide_number"].astype(str)

    # Renumber measurements when duplicate keys exist after ignoring mode.
    df = _deduplicate_measurements(df, sort_cols=["filepath"])

    # Apply measurement filter
    if measurement != "all":
        meas_int = int(measurement)
        if "measurement_number" in df.columns:
            df = df[df["measurement_number"] == meas_int].reset_index(drop=True)
        else:
            logger.warning(
                "measurement_number column missing — returning all files unfiltered"
            )

    # Derive area_id: unique per (subject, slide, measurement) across regions.
    # Format: "{slide_number}_meas{measurement_number}" — stable key for the
    # area-level hierarchy (pixels → area → slide → subject).
    if "slide_number" in df.columns and "measurement_number" in df.columns:
        nan_slide = df["slide_number"].isna()
        if nan_slide.any():
            logger.warning(
                "%d file(s) have a missing slide_number; their area_id will be 'nan_meas...' — "
                "check parser output for: %s",
                nan_slide.sum(),
                df.loc[nan_slide, "filename"].tolist()[:5],
            )
        df["area_id"] = (
            df["slide_number"].astype(str)
            + "_"
            + "meas"
            + df["measurement_number"].fillna(1).astype(int).astype(str)
        )
    else:
        df["area_id"] = "unknown"

    # Enforce subject_id as first column
    cols = df.columns.tolist()
    priority = ["subject_id", "slide_number", "area_id"]
    for col in reversed(priority):
        if col in cols:
            cols.remove(col)
            cols.insert(0, col)
    df = df[cols]

    logger.info(
        "Indexed %d files from %d subjects (measurement=%s)",
        len(df),
        df["subject_id"].nunique(),
        measurement,
    )
    return df
