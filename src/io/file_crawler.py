import logging
from pathlib import Path

import pandas as pd

from src import config
from src.metadata.parser import parse_filename

logger = logging.getLogger(__name__)


def _is_reorganized_measurement_file(file_path: Path) -> bool:
    """Return True for canonical reorganized Raman exports only.

    Matches mode-free (_range_meas) and legacy mode-tagged
    (_range_SS_meas, _range_FP_meas) stems; the mode token is kept as acquisition_mode.
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


def _deduplicate_measurements(df: pd.DataFrame) -> pd.DataFrame:
    """Raise when two files share one measurement identity.

    The identity is (subject_id, slide_number, spectral_range_cm-1,
    measurement_number) plus acquisition_mode when present. Duplicates mean the
    raw provenance is wrong, so the function never renumbers or drops files.
    """
    if df.empty or "measurement_number" not in df.columns:
        return df.copy()

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




def build_file_index(data_dir: str | Path) -> pd.DataFrame:
    """
    Crawl the reorganized raw data tree and index measurement files.

    Parameters
    ----------
    data_dir : path-like
        Root of the reorganized raw tree (``data.raw_text_dir`` in the config).

    Returns
    -------
    pd.DataFrame
        Structured index with one row per measurement file, all replicates included.
    """
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

    # Duplicate measurement identities raise instead of being renumbered.
    df = _deduplicate_measurements(df)

    # Derive area_id: unique per (subject, slide, measurement) across regions.
    # Format: "{slide_number}_meas{measurement_number}" - stable key for the
    # area-level hierarchy (pixels -> area -> slide -> subject).
    df["area_id"] = (
        df["slide_number"]
        + "_meas"
        + df["measurement_number"].fillna(1).astype(int).astype(str)
    )

    # Enforce subject_id as first column
    cols = df.columns.tolist()
    priority = ["subject_id", "slide_number", "area_id"]
    for col in reversed(priority):
        if col in cols:
            cols.remove(col)
            cols.insert(0, col)
    df = df[cols]

    logger.info("Indexed %d files from %d subjects", len(df), df["subject_id"].nunique())
    return df
