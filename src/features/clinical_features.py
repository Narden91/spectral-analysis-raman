from __future__ import annotations

import numpy as np
import pandas as pd


def build_clinical_feature_block(
    area_df: pd.DataFrame,
    clinical_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build Track E clinical covariates at subject level.

    Output columns:
    - subject_id
    - age
    - sex_binary (female=0, male=1)

    ``sex_binary`` is itself cohort-confounded; Track E remains off by default.
    """
    if "subject_id" not in area_df.columns:
        raise ValueError("area_df must contain 'subject_id'")

    subjects = pd.Index(area_df["subject_id"].astype(str).unique(), name="subject_id")

    out = pd.DataFrame(index=subjects)
    out["age"] = np.nan
    out["sex_binary"] = np.nan

    if clinical_df is not None and not clinical_df.empty and "subject_id" in clinical_df.columns:
        clinical = clinical_df.copy()
        clinical["subject_id"] = clinical["subject_id"].astype(str)
        clinical = clinical.drop_duplicates(subset=["subject_id"], keep="first")
        clinical = clinical.set_index("subject_id")

        age_col = _pick_age_column(clinical)
        if age_col is not None:
            out.loc[:, "age"] = pd.to_numeric(clinical[age_col], errors="coerce").reindex(out.index)

        sex_series = _clinical_sex_binary(clinical)
        if sex_series is not None:
            out.loc[:, "sex_binary"] = sex_series.reindex(out.index)

    if out["sex_binary"].isna().any() and "gender" in area_df.columns:
        gender_from_area = (
            area_df[["subject_id", "gender"]]
            .drop_duplicates(subset=["subject_id"], keep="first")
            .assign(subject_id=lambda d: d["subject_id"].astype(str))
            .set_index("subject_id")["gender"]
        )
        mapped = (
            gender_from_area.astype(str).str.strip().str.lower().map({"female": 0.0, "f": 0.0, "male": 1.0, "m": 1.0})
        )
        out["sex_binary"] = out["sex_binary"].fillna(mapped.reindex(out.index))

    out = out.reset_index()
    return out[["subject_id", "age", "sex_binary"]]


def _pick_age_column(clinical: pd.DataFrame) -> str | None:
    lowered = {c.lower().strip(): c for c in clinical.columns}
    if "age" in lowered:
        return lowered["age"]

    for col in clinical.columns:
        name = col.lower().strip()
        if "age" in name:
            return col

    return None


def _clinical_sex_binary(clinical: pd.DataFrame) -> pd.Series | None:
    if "sex_binary" in clinical.columns:
        return pd.to_numeric(clinical["sex_binary"], errors="coerce")

    sex_col = None
    for col in clinical.columns:
        name = col.lower().strip()
        if name in {"sex", "gender", "gender female=1"}:
            sex_col = col
            break

    if sex_col is None:
        return None

    raw = clinical[sex_col]
    if raw.dtype.kind in {"i", "u", "f"}:
        numeric = pd.to_numeric(raw, errors="coerce")
        unique = set(numeric.dropna().unique().tolist())
        if unique.issubset({0.0, 1.0}):
            if sex_col.lower().strip() == "gender female=1":
                return 1.0 - numeric
            return numeric

    sex = raw.astype(str).str.strip().str.lower()
    return sex.map({"female": 0.0, "f": 0.0, "male": 1.0, "m": 1.0})
