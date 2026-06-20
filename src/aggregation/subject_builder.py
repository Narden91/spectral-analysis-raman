from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_subject_hierarchy(area_df: pd.DataFrame) -> dict:
    """Build area->slide->subject hierarchy tables for CV ingestion.

    Aggregation uses median at every level (area→slide, slide→subject) to match
    the subject-level Biological PCA contract and resist outlier areas.
    """
    required = {"subject_id", "slide_number", "area_mean", "n_valid_pixels"}
    missing = required - set(area_df.columns)
    if missing:
        raise ValueError(f"area_df missing required columns: {sorted(missing)}")

    if area_df.empty:
        return {
            "area_table": area_df.copy(),
            "slide_table": pd.DataFrame(columns=["subject_id", "slide_number", "slide_median", "n_areas_surviving"]),
            "subject_table": pd.DataFrame(columns=["subject_id", "subject_median", "n_slides", "n_areas_surviving"]),
        }

    slide_rows: list[dict] = []
    for (subject_id, slide_number), grp in area_df.groupby(["subject_id", "slide_number"], sort=False):
        vectors = np.vstack([np.asarray(v, dtype=float) for v in grp["area_mean"].to_numpy()])
        slide_median = np.nanmedian(vectors, axis=0)
        slide_rows.append(
            {
                "subject_id": str(subject_id),
                "slide_number": str(slide_number),
                "slide_median": slide_median,
                "n_areas_surviving": int(len(grp)),
            }
        )

    slide_df = pd.DataFrame(slide_rows)

    subject_rows: list[dict] = []
    for subject_id, grp in slide_df.groupby("subject_id", sort=False):
        slide_vectors = np.vstack([np.asarray(v, dtype=float) for v in grp["slide_median"].to_numpy()])
        subject_median = np.nanmedian(slide_vectors, axis=0)
        subject_rows.append(
            {
                "subject_id": str(subject_id),
                "subject_median": subject_median,
                "n_slides": int(len(grp)),
                "n_areas_surviving": int(grp["n_areas_surviving"].sum()),
            }
        )

    subject_df = pd.DataFrame(subject_rows)
    return {
        "area_table": area_df.copy().reset_index(drop=True),
        "slide_table": slide_df.reset_index(drop=True),
        "subject_table": subject_df.reset_index(drop=True),
    }

