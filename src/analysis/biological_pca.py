"""Strict subject-level Biological PCA outputs for Raman spectra."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull

from src import (
    config,
    get_biological_pca_canonical_dir,
    get_cross_region_phase_dir,
    get_cross_region_summary_dir,
    resolve_project_path,
)
from src.analysis.joint_dr import joint_pca_mfa
from src.analysis.statistics import run_pca
from src.analysis.subject_features import write_subject_feature_exports
from src.data_preparation.loaders import (
    load_area_summaries,
    load_clinical_subjects,
    load_file_index,
)
from src.data_preparation.store import get_active_data_source
from src.io.file_crawler import build_file_index
from src.io.txt_reader import read_txt_spectrum
from src.preprocessing.pipeline import run_preprocessing
from src.visualization.eda_plots import plot_scree
from src.visualization.theme import GROUP_COLORS, GROUP_LABELS

logger = logging.getLogger(__name__)

REQUIRED_MODELS = ("PCA_1330", "PCA_2850", "PCA_whole")
BLOCKS = ("1300", "2850")
CONTRIBUTION_TOLERANCE = 1e-6
CANONICAL_PCS = ("PC1", "PC2", "PC3")
PC_COLORS = {
    "PC1": "#264653",
    "PC2": "#2A9D8F",
    "PC3": "#E76F51",
}
FORBIDDEN_INTERPRETATION_PHRASES = (
    "lipids are increased",
    "protein abundance is higher",
    "this proves disease separation",
    "this biomarker diagnoses disease",
    "diagnoses disease",
    "disease separation",
    "biomarker",
    "reflects disease",
    "disease signature",
    "associates with disease",
    "is diagnostic",
    "this predicts",
    "proves causal",
)


@dataclass(frozen=True)
class SubjectInputs:
    subject_spectra: pd.DataFrame
    subject_region_counts: pd.DataFrame
    input_provenance: pd.DataFrame
    axes: dict[str, np.ndarray]
    matrices: dict[str, np.ndarray]
    subjects: list[str]
    excluded_subjects: list[dict[str, str]] = field(default_factory=list)  # [{subject_id, reason}]


@dataclass(frozen=True)
class ModelResult:
    name: str
    scores: pd.DataFrame
    loadings: pd.DataFrame
    explained_variance: pd.DataFrame
    loading_regions: pd.DataFrame
    region_contributions: pd.DataFrame
    region_annotation_candidates: pd.DataFrame
    loading_region_summary: pd.DataFrame
    band_contributions: pd.DataFrame
    biochemical_summary: pd.DataFrame
    pc1_warning: str | None = None
    method_metadata: dict[str, object] = field(default_factory=dict)


def select_subjects(subject_ids: Iterable[object], max_subjects: int | None = None) -> list[str]:
    """Return sorted subject IDs, optionally limited deterministically."""
    selected = sorted({str(subject_id) for subject_id in subject_ids})
    if max_subjects is not None:
        if int(max_subjects) <= 0:
            raise ValueError("--max-subjects must be a positive integer")
        selected = selected[: int(max_subjects)]
    return selected


def _biological_cfg(cfg: dict) -> dict:
    bp_cfg = cfg.get("biological_pca")
    if not isinstance(bp_cfg, dict):
        raise ValueError("Missing required config section: biological_pca")
    return bp_cfg


def _band_windows(cfg: dict) -> dict[str, tuple[float, float]]:
    bands_cfg = cfg.get("whole_spectra", {}).get("bands", {})
    if not isinstance(bands_cfg, dict) or not bands_cfg:
        raise ValueError("Missing required config section: whole_spectra.bands")
    windows: dict[str, tuple[float, float]] = {}
    for band_name, limits in bands_cfg.items():
        if not isinstance(limits, (list, tuple)) or len(limits) != 2:
            raise ValueError(f"Band '{band_name}' must define exactly two limits")
        lower, upper = float(limits[0]), float(limits[1])
        if lower >= upper:
            raise ValueError(f"Band '{band_name}' must satisfy lower < upper")
        windows[str(band_name)] = (lower, upper)
    return windows


def _band_class_map(bp_cfg: dict) -> dict[str, str]:
    class_cfg = bp_cfg.get("biochemical_classes", {})
    if not isinstance(class_cfg, dict) or not class_cfg:
        raise ValueError("biological_pca.biochemical_classes must be a non-empty mapping")
    band_to_class: dict[str, str] = {}
    for class_name, band_names in class_cfg.items():
        if str(class_name) not in {"lipid", "protein", "nucleic_acid", "mixed"}:
            raise ValueError(f"Unsupported biochemical class '{class_name}'")
        for band_name in band_names or []:
            band_name = str(band_name)
            if band_name in band_to_class:
                raise ValueError(
                    f"Band '{band_name}' maps to multiple biochemical classes: "
                    f"{band_to_class[band_name]} and {class_name}"
                )
            band_to_class[band_name] = str(class_name)
    return band_to_class


def validate_biological_pca_config(cfg: dict) -> None:
    """Validate static Biological PCA config before spectra are loaded."""
    bp_cfg = _biological_cfg(cfg)
    if int(bp_cfg.get("n_components", 0)) <= 0:
        raise ValueError("biological_pca.n_components must be a positive integer")
    if str(bp_cfg.get("aggregation", "")).lower() != "median":
        raise ValueError("biological_pca.aggregation must be 'median'")

    models = bp_cfg.get("models")
    if not isinstance(models, dict):
        raise ValueError("biological_pca.models must be a mapping")
    missing = [model for model in REQUIRED_MODELS if model not in models]
    if missing:
        raise ValueError(f"Missing Biological PCA model config(s): {', '.join(missing)}")

    windows = _band_windows(cfg)
    band_to_class = _band_class_map(bp_cfg)
    for model_name in REQUIRED_MODELS:
        model_cfg = models[model_name]
        blocks = [str(block) for block in model_cfg.get("spectral_blocks", [])]
        if model_name == "PCA_whole":
            if blocks != ["1300", "2850"]:
                raise ValueError("PCA_whole must use spectral_blocks ['1300', '2850']")
        elif len(blocks) != 1 or blocks[0] not in BLOCKS:
            raise ValueError(f"{model_name} must use exactly one spectral block")

        used_bands = [str(band) for band in model_cfg.get("bands", [])]
        if not used_bands:
            raise ValueError(f"{model_name} must list at least one band")
        for band_name in used_bands:
            if band_name not in windows:
                raise ValueError(f"{model_name} references missing whole_spectra band '{band_name}'")
            if band_name not in band_to_class:
                raise ValueError(f"Band '{band_name}' used by {model_name} has no biochemical class")


def _flatten_spectra_with_coords(
    spectra: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(spectra, dtype=float)
    if arr.ndim == 3:
        flat = arr.reshape(-1, arr.shape[-1])
        x_grid, y_grid = np.meshgrid(np.asarray(x_coords), np.asarray(y_coords), indexing="ij")
        return flat, x_grid.reshape(-1), y_grid.reshape(-1)
    if arr.ndim == 2:
        n_pixels = arr.shape[0]
        x = np.asarray(x_coords, dtype=float)
        y = np.asarray(y_coords, dtype=float)
        if x.size != n_pixels:
            x = np.full(n_pixels, np.nan)
        if y.size != n_pixels:
            y = np.full(n_pixels, np.nan)
        return arr, x, y
    raise ValueError("Spectra array must be 2D or 3D")


def _spectral_range_value(row: pd.Series) -> str:
    if "spectral_range_cm-1" in row:
        value = row["spectral_range_cm-1"]
    else:
        value = row.get("spectral_range")
    return str(int(float(value))) if pd.notna(value) else ""


def _region_for_block(block: str) -> str:
    return "1300" if str(block) == "1300" else "2850"


def _preprocess_area(row: pd.Series) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, int]:
    filepath = Path(row["filepath"])
    wavenumbers, x_coords, y_coords, spectra, _ = read_txt_spectrum(filepath)
    spectra_flat, x_flat, y_flat = _flatten_spectra_with_coords(spectra, x_coords, y_coords)
    block = _spectral_range_value(row)
    result = run_preprocessing(
        spectra_flat,
        wavenumbers,
        region=_region_for_block(block),
        pixel_coords=np.column_stack([x_flat, y_flat]) if len(x_flat) == len(spectra_flat) else None,
        screen_dead=True,
        suppress_cosmic=None,
        correct_bl=True,
        smooth=None,
        interpolate=False,
        filter_quality=None,
        normalize=True,
        norm_for="eda",
        record_stages=False,
    )
    retained_indices = np.flatnonzero(result.retained_mask)
    provenance = pd.DataFrame(
        {
            "subject_id": str(row["subject_id"]),
            "sample_id": str(row.get("slide_number", "")),
            "spot_id": str(row.get("area_id", "")),
            "pixel_id": retained_indices.astype(int),
            "x_coord": x_flat[retained_indices] if x_flat.size else np.nan,
            "y_coord": y_flat[retained_indices] if y_flat.size else np.nan,
            "spectral_range": block,
            "filepath": str(filepath),
            "retained_after_qc": True,
        }
    )
    return result.spectra, result.wavenumbers, provenance, int(spectra_flat.shape[0])


def _common_axis(axes: list[np.ndarray], block: str) -> np.ndarray:
    if not axes:
        raise ValueError(f"No post-preprocessing Raman axes found for block {block}")
    lower = float(max(np.nanmin(axis) for axis in axes))
    upper = float(min(np.nanmax(axis) for axis in axes))
    if upper <= lower:
        raise ValueError(f"Post-preprocessing Raman axes have no common overlap for block {block}")
    spacing_candidates = []
    for axis in axes:
        diffs = np.diff(np.asarray(axis, dtype=float))
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size:
            spacing_candidates.append(float(np.nanmedian(diffs)))
    spacing = float(np.nanmedian(spacing_candidates)) if spacing_candidates else 1.0
    start = np.ceil(lower / spacing) * spacing
    stop = np.floor(upper / spacing) * spacing
    common = np.arange(start, stop + spacing * 0.5, spacing, dtype=float)
    common = np.round(common, 6)
    if common.size == 0:
        raise ValueError(f"Post-preprocessing Raman axes have no represented common points for block {block}")
    return common


def _resample_to_axis(spectra: np.ndarray, source_axis: np.ndarray, target_axis: np.ndarray) -> np.ndarray:
    if source_axis.size == target_axis.size and np.allclose(source_axis, target_axis, rtol=0, atol=1e-8):
        return spectra
    interpolator = interp1d(
        source_axis,
        spectra,
        kind="linear",
        axis=1,
        bounds_error=False,
        fill_value="extrapolate",
    )
    return interpolator(target_axis)


def build_subject_inputs(max_subjects: int | None = None, cfg: dict | None = None) -> SubjectInputs:
    """Preprocess retained spectra and aggregate one median spectrum per subject and block."""
    cfg = config if cfg is None else cfg
    validate_biological_pca_config(cfg)
    active_source = get_active_data_source()
    if active_source is not None and active_source.source_name != "raw_text":
        return _build_subject_inputs_from_materialized_store(max_subjects=max_subjects)
    df_index = build_file_index(measurement="all")
    if df_index.empty:
        raise ValueError("No Raman input files were found")
    required_cols = {"subject_id", "slide_number", "area_id", "filepath", "spectral_range_cm-1"}
    missing_cols = required_cols - set(df_index.columns)
    if missing_cols:
        raise ValueError(f"File index is missing required column(s): {', '.join(sorted(missing_cols))}")

    subjects = select_subjects(df_index["subject_id"], max_subjects=max_subjects)
    df_index = df_index[df_index["subject_id"].astype(str).isin(subjects)].copy()

    area_axes: dict[str, list[np.ndarray]] = {"1300": [], "2850": []}
    per_subject_block_spectra: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]] = {}
    per_subject_block_meta: dict[tuple[str, str], dict[str, set | int]] = {}
    provenance_frames: list[pd.DataFrame] = []

    for _, row in df_index.sort_values(["subject_id", "slide_number", "area_id", "filepath"]).iterrows():
        block = _spectral_range_value(row)
        if block not in BLOCKS:
            continue
        spectra, axis, provenance, n_input_pixels = _preprocess_area(row)
        area_axes[block].append(axis)
        subject_id = str(row["subject_id"])
        key = (subject_id, block)
        per_subject_block_spectra.setdefault(key, []).append((spectra, axis))
        meta = per_subject_block_meta.setdefault(
            key,
            {"samples": set(), "spots": set(), "n_pixels": 0, "n_retained_pixels": 0},
        )
        meta["samples"].add(str(row.get("slide_number", "")))
        meta["spots"].add(str(row.get("area_id", "")))
        meta["n_pixels"] = int(meta["n_pixels"]) + n_input_pixels
        meta["n_retained_pixels"] = int(meta["n_retained_pixels"]) + int(len(provenance))
        provenance_frames.append(provenance)

    input_provenance = (
        pd.concat(provenance_frames, ignore_index=True)
        if provenance_frames
        else pd.DataFrame(
            columns=[
                "subject_id",
                "sample_id",
                "spot_id",
                "pixel_id",
                "x_coord",
                "y_coord",
                "spectral_range",
                "filepath",
                "retained_after_qc",
            ]
        )
    )

    axes = {block: _common_axis(area_axes[block], block) for block in BLOCKS}
    subject_clinical = (
        df_index[["subject_id", "is_control", "gender"]]
        .copy()
        .assign(subject_id=lambda df: df["subject_id"].astype(str))
        .drop_duplicates("subject_id")
        .set_index("subject_id")
    )
    included_subjects = []
    for subject_id in subjects:
        has_both = all(
            (subject_id, block) in per_subject_block_spectra
            and sum(arr.shape[0] for arr, _ in per_subject_block_spectra[(subject_id, block)]) > 0
            for block in BLOCKS
        )
        if has_both:
            included_subjects.append(subject_id)
    missing_both = sorted(set(subjects) - set(included_subjects))
    if missing_both:
        raise ValueError(
            "Every included subject must have both 1300 and 2850 retained spectra; "
            f"missing retained data for: {', '.join(missing_both[:10])}"
        )

    matrices: dict[str, list[np.ndarray]] = {"1300": [], "2850": []}
    region_rows: list[dict[str, object]] = []
    subject_rows: list[dict[str, object]] = []
    for subject_id in included_subjects:
        subject_samples: set[str] = set()
        subject_spots: set[str] = set()
        subject_pixels = 0
        feature_values: dict[str, float] = {}
        for block in BLOCKS:
            resampled = [
                _resample_to_axis(spectra, source_axis, axes[block])
                for spectra, source_axis in per_subject_block_spectra[(subject_id, block)]
            ]
            stacked = np.vstack(resampled)
            median = np.nanmedian(stacked, axis=0)
            matrices[block].append(median)
            key = (subject_id, block)
            meta = per_subject_block_meta[key]
            samples = set(meta["samples"])
            spots = set(meta["spots"])
            n_retained = int(meta["n_retained_pixels"])
            subject_samples.update(samples)
            subject_spots.update(spots)
            subject_pixels += n_retained
            region_rows.append(
                {
                    "subject_id": subject_id,
                    "spectral_range": block,
                    "n_samples": len(samples),
                    "n_spots": len(spots),
                    "n_pixels": int(meta["n_pixels"]),
                    "n_retained_pixels": n_retained,
                }
            )
            for shift, value in zip(axes[block], median):
                feature_values[f"{block}__{float(shift):.1f}"] = float(value)
        is_control = bool(subject_clinical.loc[subject_id, "is_control"])
        gender = str(subject_clinical.loc[subject_id, "gender"]).lower()
        if gender not in {"female", "male"}:
            gender = "unknown"
        subject_rows.append(
            {
                "subject_id": subject_id,
                "condition": "control" if is_control else "disease",
                "gender": gender,
                "group": f"{gender}_{'control' if is_control else 'disease'}",
                "n_samples": len(subject_samples),
                "n_spots": len(subject_spots),
                "n_pixels": subject_pixels,
                **feature_values,
            }
        )

    matrix_arrays = {block: np.vstack(rows) for block, rows in matrices.items()}
    subject_spectra = pd.DataFrame(subject_rows)
    subject_region_counts = pd.DataFrame(region_rows)
    return SubjectInputs(
        subject_spectra=subject_spectra,
        subject_region_counts=subject_region_counts,
        input_provenance=input_provenance,
        axes=axes,
        matrices=matrix_arrays,
        subjects=included_subjects,
        excluded_subjects=[],
    )


def _build_subject_inputs_from_materialized_store(max_subjects: int | None = None) -> SubjectInputs:
    df_index = load_file_index()
    clinical = load_clinical_subjects()
    subjects = select_subjects(df_index["subject_id"], max_subjects=max_subjects)
    df_index = df_index[df_index["subject_id"].astype(str).isin(subjects)].copy()
    clinical = clinical[clinical["subject_id"].astype(str).isin(subjects)].copy()

    payloads = {
        "1300": load_area_summaries(None, "1300"),
        "2850": load_area_summaries(None, "2850"),
    }
    axes = {
        block: np.asarray([float(name) for name in payloads[block]["feature_names"].astype(str)], dtype=float)
        for block in BLOCKS
    }

    included_subjects: list[str] = []
    matrices: dict[str, list[np.ndarray]] = {"1300": [], "2850": []}
    region_rows: list[dict[str, object]] = []
    subject_rows: list[dict[str, object]] = []

    excluded_subjects: list[dict[str, str]] = []
    for subject_id in subjects:
        has_all_blocks = all(
            np.any(payloads[block]["subject_ids"].astype(str) == str(subject_id))
            for block in BLOCKS
        )
        if not has_all_blocks:
            missing_blocks = [b for b in BLOCKS if not np.any(payloads[b]["subject_ids"].astype(str) == str(subject_id))]
            excluded_subjects.append({"subject_id": str(subject_id), "reason": f"missing blocks: {', '.join(missing_blocks)}"})
            continue
        included_subjects.append(str(subject_id))

    if excluded_subjects:
        logger.warning(
            "BioPCA excluded %d subject(s) for missing spectral data: %s",
            len(excluded_subjects),
            ", ".join(d["subject_id"] for d in excluded_subjects[:10]),
        )

    for subject_id in included_subjects:
        feature_values: dict[str, float] = {}
        subject_samples: set[str] = set()
        subject_areas: set[str] = set()
        subject_pixels = 0
        clinical_row = clinical.loc[clinical["subject_id"].astype(str) == str(subject_id)]
        if clinical_row.empty:
            is_control = False
            gender = "unknown"
        else:
            is_control = bool(clinical_row["is_control"].iloc[0]) if "is_control" in clinical_row.columns else False
            gender = str(clinical_row["gender"].iloc[0]).lower() if "gender" in clinical_row.columns else "unknown"

        for block in BLOCKS:
            mask = payloads[block]["subject_ids"].astype(str) == str(subject_id)
            block_matrix = np.asarray(payloads[block]["area_summary"][mask], dtype=float)
            block_area_ids = payloads[block]["area_ids"][mask].astype(str)
            block_measurement_ids = payloads[block]["measurement_ids"][mask].astype(str)
            median = np.nanmedian(block_matrix, axis=0)
            matrices[block].append(median)
            subject_samples.update(block_measurement_ids.tolist())
            subject_areas.update(block_area_ids.tolist())
            subject_pixels += int(block_matrix.shape[0])
            region_rows.append(
                {
                    "subject_id": str(subject_id),
                    "spectral_range": block,
                    "n_samples": len(block_measurement_ids),
                    "n_spots": len(block_area_ids),
                    "n_pixels": int(block_matrix.shape[0]),
                    "n_retained_pixels": int(block_matrix.shape[0]),
                }
            )
            for shift, value in zip(axes[block], median):
                feature_values[f"{block}__{float(shift):.1f}"] = float(value)

        subject_rows.append(
            {
                "subject_id": str(subject_id),
                "condition": "control" if is_control else "disease",
                "gender": gender,
                "group": f"{gender}_{'control' if is_control else 'disease'}",
                "n_samples": len(subject_samples),
                "n_spots": len(subject_areas),
                "n_pixels": subject_pixels,
                **feature_values,
            }
        )

    matrix_arrays = {block: np.vstack(rows) if rows else np.empty((0, 0)) for block, rows in matrices.items()}
    input_provenance = df_index[
        ["subject_id", "slide_number", "area_id", "measurement_id", "filepath", "spectral_range_cm-1"]
    ].rename(
        columns={
            "slide_number": "sample_id",
            "area_id": "spot_id",
            "measurement_id": "pixel_id",
            "filepath": "filepath",
            "spectral_range_cm-1": "spectral_range",
        }
    ).copy()
    input_provenance["x_coord"] = np.nan
    input_provenance["y_coord"] = np.nan
    input_provenance["retained_after_qc"] = True

    return SubjectInputs(
        subject_spectra=pd.DataFrame(subject_rows),
        subject_region_counts=pd.DataFrame(region_rows),
        input_provenance=input_provenance[
            ["subject_id", "sample_id", "spot_id", "pixel_id", "x_coord", "y_coord", "spectral_range", "filepath", "retained_after_qc"]
        ],
        axes=axes,
        matrices=matrix_arrays,
        subjects=included_subjects,
        excluded_subjects=excluded_subjects,
    )


def _model_axis_and_matrix(
    model_name: str, model_cfg: dict, inputs: SubjectInputs
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if model_name == "PCA_whole":
        feature_blocks = ["1300", "2850"]
        axis = np.concatenate([inputs.axes["1300"], inputs.axes["2850"]])
        matrix = np.hstack([inputs.matrices["1300"], inputs.matrices["2850"]])
        return axis, matrix, feature_blocks

    block = str(model_cfg.get("spectral_blocks", [])[0])
    axis = inputs.axes[block]
    window = model_cfg.get("window")
    if str(window).lower() == "full":
        return axis, inputs.matrices[block], [block]
    if not isinstance(window, (list, tuple)) or len(window) != 2:
        raise ValueError(f"{model_name} must define a two-value window")
    lower, upper = float(window[0]), float(window[1])
    mask = (axis >= lower) & (axis <= upper)
    if not mask.any():
        raise ValueError(f"{model_name} window [{lower}, {upper}] does not overlap the {block} axis")
    return axis[mask], inputs.matrices[block][:, mask], [block]


def _validate_axes_against_models(bp_cfg: dict, inputs: SubjectInputs) -> None:
    for block in BLOCKS:
        if block not in inputs.axes or inputs.axes[block].size == 0:
            raise ValueError(f"Missing post-preprocessing Raman axis for block {block}")
    for model_name in REQUIRED_MODELS:
        axis, matrix, _ = _model_axis_and_matrix(model_name, bp_cfg["models"][model_name], inputs)
        if axis.size == 0 or matrix.shape[1] == 0:
            raise ValueError(f"{model_name} has no represented Raman axis points")


def _validate_component_count(model_name: str, matrix: np.ndarray, n_components: int) -> None:
    if min(matrix.shape[0], matrix.shape[1]) < int(n_components):
        raise ValueError(
            f"{model_name} requires n_components={n_components}, but "
            f"min(n_subjects={matrix.shape[0]}, n_features={matrix.shape[1]}) is smaller"
        )


def _scores_frame(
    model_name: str,
    subjects: list[str],
    scores: np.ndarray,
    subject_spectra: pd.DataFrame,
) -> pd.DataFrame:
    pc_cols = {f"PC{i + 1}": scores[:, i] for i in range(scores.shape[1])}
    merge_cols = ["subject_id", "n_samples", "n_spots", "n_pixels"]
    for col in ("condition", "gender", "group"):
        if col in subject_spectra.columns:
            merge_cols.append(col)
    counts = subject_spectra[merge_cols].copy()
    df = pd.DataFrame({"subject_id": subjects, **pc_cols})
    df["region_model"] = model_name
    return df.merge(counts, on="subject_id", how="left")


def _loadings_frame(
    model_name: str,
    axis: np.ndarray,
    blocks: list[str],
    loadings: np.ndarray,
    inputs: SubjectInputs,
    cfg: dict,
) -> pd.DataFrame:
    rows: dict[str, object] = {"raman_shift": axis.astype(float)}
    if model_name == "PCA_whole":
        spectral_block = np.array(
            ["1300"] * inputs.axes["1300"].size + ["2850"] * inputs.axes["2850"].size,
            dtype=object,
        )
        rows["spectral_block"] = spectral_block
    for idx in range(loadings.shape[0]):
        rows[f"PC{idx + 1}_loading"] = loadings[idx]
    rows["region_model"] = model_name
    df = pd.DataFrame(rows)
    if model_name != "PCA_whole" and blocks:
        expected = inputs.axes[blocks[0]]
        window = _biological_cfg(cfg)["models"][model_name]["window"]
        if str(window).lower() == "full":
            mask = np.ones(expected.shape[0], dtype=bool)
        else:
            mask = (expected >= float(window[0])) & (expected <= float(window[1]))
        if len(df) != int(mask.sum()) or not np.allclose(df["raman_shift"].to_numpy(), expected[mask]):
            raise ValueError(f"{model_name} loading rows do not match the selected region axis")
    if model_name == "PCA_whole":
        duplicated = df.duplicated(subset=["spectral_block", "raman_shift"]).any()
        if duplicated:
            raise ValueError("PCA_whole loadings contain duplicated (spectral_block, raman_shift) pairs")
    return df


def _explained_variance_frame(model_name: str, result: dict) -> pd.DataFrame:
    ratios = result["explained_variance_ratio"]
    cumulative = result["cumulative_variance"]
    return pd.DataFrame(
        {
            "region_model": model_name,
            "PC": [f"PC{i + 1}" for i in range(len(ratios))],
            "explained_variance_ratio": ratios,
            "cumulative_variance": cumulative,
        }
    )


def _loadings_by_block(
    model_name: str, loadings_df: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    if model_name == "PCA_whole":
        return {
            block: loadings_df[loadings_df["spectral_block"].astype(str) == block].reset_index(drop=True)
            for block in BLOCKS
        }
    return {"single": loadings_df.reset_index(drop=True)}


def _band_block_for_window(lower: float, upper: float) -> str:
    center = (float(lower) + float(upper)) / 2.0
    return "2850" if center >= 2000 else "1300"


def _loading_region_cfg(cfg: dict) -> tuple[float, float, float]:
    bp_cfg = _biological_cfg(cfg)
    return (
        float(bp_cfg.get("loading_region_quantile", 0.9)),
        float(bp_cfg.get("loading_region_min_width_cm", 8.0)),
        float(bp_cfg.get("loading_region_merge_gap_cm", 6.0)),
    )


def _region_label(row: pd.Series) -> str:
    block = str(row.get("spectral_block", "single"))
    prefix = f"{block} " if block not in {"", "single", "nan"} else ""
    return f"{prefix}{row['start_cm-1']:.1f}–{row['end_cm-1']:.1f} cm⁻¹"


def _extract_loading_regions_for_series(
    model_name: str,
    pc_name: str,
    spectral_block: str,
    axis: np.ndarray,
    values: np.ndarray,
    *,
    quantile: float,
    min_width_cm: float,
    merge_gap_cm: float,
) -> list[dict[str, object]]:
    if axis.size == 0 or values.size == 0:
        return []
    abs_values = np.abs(values)
    if not np.isfinite(abs_values).any() or np.nanmax(abs_values) == 0:
        return []

    threshold = float(np.nanquantile(abs_values, quantile))
    active = abs_values >= threshold
    intervals: list[list[int]] = []
    start: int | None = None
    for idx, is_active in enumerate(active):
        if bool(is_active) and start is None:
            start = idx
        elif not bool(is_active) and start is not None:
            intervals.append([start, idx - 1])
            start = None
    if start is not None:
        intervals.append([start, len(active) - 1])

    merged: list[list[int]] = []
    for left, right in intervals:
        if not merged:
            merged.append([left, right])
            continue
        prev_left, prev_right = merged[-1]
        gap = float(axis[left] - axis[prev_right])
        if gap <= merge_gap_cm:
            merged[-1][1] = right
        else:
            merged.append([left, right])

    rows: list[dict[str, object]] = []
    for left, right in merged:
        start_cm = float(axis[left])
        end_cm = float(axis[right])
        width = float(end_cm - start_cm)
        if width < min_width_cm:
            continue
        axis_slice = axis[left : right + 1]
        value_slice = values[left : right + 1]
        peak_idx = int(np.nanargmax(np.abs(value_slice)))
        signed = (
            float(value_slice[0])
            if axis_slice.size == 1
            else float(np.trapz(value_slice, axis_slice))
        )
        absolute = abs(signed)
        rows.append(
            {
                "region_model": model_name,
                "PC": pc_name,
                "spectral_block": spectral_block,
                "start_cm-1": start_cm,
                "end_cm-1": end_cm,
                "width_cm-1": width,
                "peak_shift_cm-1": float(axis_slice[peak_idx]),
                "peak_loading": float(value_slice[peak_idx]),
                "peak_absolute_loading": float(np.abs(value_slice[peak_idx])),
                "signed_integrated_loading": signed,
                "absolute_integrated_loading": absolute,
                "dominant_sign": "positive" if signed > 0 else ("negative" if signed < 0 else "zero"),
                "threshold_abs_loading": threshold,
            }
        )
    return rows


def compute_loading_regions(model_name: str, loadings_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    quantile, min_width_cm, merge_gap_cm = _loading_region_cfg(cfg)
    rows: list[dict[str, object]] = []
    pc_cols = [col for col in loadings_df.columns if col.startswith("PC") and col.endswith("_loading")]
    by_block = _loadings_by_block(model_name, loadings_df)

    for pc_col in pc_cols:
        pc_name = pc_col.replace("_loading", "")
        for spectral_block, block_df in by_block.items():
            axis = block_df["raman_shift"].to_numpy(dtype=float)
            values = block_df[pc_col].to_numpy(dtype=float)
            rows.extend(
                _extract_loading_regions_for_series(
                    model_name,
                    pc_name,
                    spectral_block,
                    axis,
                    values,
                    quantile=quantile,
                    min_width_cm=min_width_cm,
                    merge_gap_cm=merge_gap_cm,
                )
            )

    regions = pd.DataFrame(rows)
    if regions.empty:
        return pd.DataFrame(
            columns=[
                "region_model",
                "PC",
                "spectral_block",
                "region_id",
                "region_label",
                "start_cm-1",
                "end_cm-1",
                "width_cm-1",
                "peak_shift_cm-1",
                "peak_loading",
                "peak_absolute_loading",
                "signed_integrated_loading",
                "absolute_integrated_loading",
                "dominant_sign",
                "threshold_abs_loading",
            ]
        )

    regions = regions.sort_values(
        ["PC", "spectral_block", "absolute_integrated_loading"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    region_ids: list[str] = []
    for pc, group in regions.groupby("PC", sort=False):
        for rank, idx in enumerate(group.index, start=1):
            region_ids.append((idx, f"{pc}_R{rank}"))
    region_id_map = dict(region_ids)
    regions["region_id"] = [region_id_map[idx] for idx in regions.index]
    regions["region_label"] = regions.apply(_region_label, axis=1)
    return regions[
        [
            "region_model",
            "PC",
            "spectral_block",
            "region_id",
            "region_label",
            "start_cm-1",
            "end_cm-1",
            "width_cm-1",
            "peak_shift_cm-1",
            "peak_loading",
            "peak_absolute_loading",
            "signed_integrated_loading",
            "absolute_integrated_loading",
            "dominant_sign",
            "threshold_abs_loading",
        ]
    ]


def build_region_contributions(loading_regions: pd.DataFrame) -> pd.DataFrame:
    if loading_regions.empty:
        return pd.DataFrame(
            columns=[
                "region_model",
                "PC",
                "spectral_block",
                "region_id",
                "region_label",
                "absolute_integrated_loading",
                "relative_absolute_contribution",
                "global_rank",
                "block_rank",
                "dominant_sign",
            ]
        )

    rows: list[dict[str, object]] = []
    for pc, pc_group in loading_regions.groupby("PC", sort=False):
        total = float(pc_group["absolute_integrated_loading"].sum())
        for block, block_group in pc_group.groupby("spectral_block", sort=False):
            for block_rank, (_, row) in enumerate(
                block_group.sort_values("absolute_integrated_loading", ascending=False).iterrows(),
                start=1,
            ):
                rows.append(
                    {
                        "region_model": row["region_model"],
                        "PC": pc,
                        "spectral_block": block,
                        "region_id": row["region_id"],
                        "region_label": row["region_label"],
                        "absolute_integrated_loading": row["absolute_integrated_loading"],
                        "relative_absolute_contribution": (
                            row["absolute_integrated_loading"] / total if total > 0 else np.nan
                        ),
                        "block_rank": block_rank,
                        "dominant_sign": row["dominant_sign"],
                    }
                )
    contributions = pd.DataFrame(rows)
    for pc, group in contributions.groupby("PC", sort=False):
        ordered = group.sort_values("absolute_integrated_loading", ascending=False)
        for global_rank, idx in enumerate(ordered.index, start=1):
            contributions.loc[idx, "global_rank"] = global_rank
        rel = ordered["relative_absolute_contribution"]
        if rel.notna().any():
            rel_sum = float(rel.sum())
            if abs(rel_sum - 1.0) >= CONTRIBUTION_TOLERANCE:
                raise ValueError(f"Loading-region contributions for {pc} sum to {rel_sum}, not 1.0")
    contributions["global_rank"] = contributions["global_rank"].astype(int)
    return contributions


def annotate_loading_regions(
    model_name: str,
    loading_regions: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    windows = _band_windows(cfg)
    band_to_class = _band_class_map(_biological_cfg(cfg))
    rows: list[dict[str, object]] = []

    for _, region in loading_regions.iterrows():
        overlapping_bands: list[str] = []
        contexts: list[str] = []
        for band_name, (lower, upper) in windows.items():
            block = _band_block_for_window(lower, upper)
            if model_name == "PCA_whole" and block != str(region["spectral_block"]):
                continue
            if float(region["start_cm-1"]) <= upper and float(region["end_cm-1"]) >= lower:
                overlapping_bands.append(band_name)
                contexts.append(band_to_class.get(band_name, "mixed"))
        unique_contexts = sorted(set(contexts))
        if not unique_contexts:
            context_label = "unassigned_broad_interval"
        elif len(unique_contexts) == 1:
            context_label = f"{unique_contexts[0]}_associated_region"
        else:
            context_label = "mixed_context_region"
        rows.append(
            {
                "region_model": region["region_model"],
                "PC": region["PC"],
                "spectral_block": region["spectral_block"],
                "region_id": region["region_id"],
                "region_label": region["region_label"],
                "start_cm-1": region["start_cm-1"],
                "end_cm-1": region["end_cm-1"],
                "overlapping_bands": "; ".join(overlapping_bands),
                "biochemical_context": context_label,
                "overlap_count": len(overlapping_bands),
                "annotation_note": (
                    "Loading-derived interval overlaps broad known Raman context."
                    if overlapping_bands
                    else "Loading-derived interval has no direct overlap with configured known bands."
                ),
            }
        )
    return pd.DataFrame(rows)


def _interpretation_sentence(pc: str, dominant_context: str | None, mixed: bool) -> str:
    if dominant_context == "unassigned_broad_interval":
        sentence = (
            f"{pc} is dominated by loading-derived intervals without direct known-band overlap. "
            "Interpret this as broad spectral structure rather than a molecule-specific assignment."
        )
    elif mixed:
        sentence = (
            f"{pc} is dominated by loading-derived intervals overlapping mixed broad Raman context. "
            "This remains an exploratory regional interpretation."
        )
    else:
        sentence = (
            f"{pc} is dominated by loading-derived intervals overlapping broad {dominant_context.replace('_', ' ')}. "
            "This remains an exploratory regional interpretation."
        )
    lower = sentence.lower()
    if any(phrase in lower for phrase in FORBIDDEN_INTERPRETATION_PHRASES):
        raise ValueError(f"Forbidden interpretation wording generated: {sentence}")
    return sentence


def build_loading_region_summary(
    model_name: str,
    region_contributions: pd.DataFrame,
    annotations: pd.DataFrame,
) -> pd.DataFrame:
    if region_contributions.empty:
        return pd.DataFrame(
            columns=[
                "region_model",
                "PC",
                "dominant_context",
                "top_region_1",
                "top_region_2",
                "top_region_3",
                "interpretation_sentence",
            ]
        )

    merged = region_contributions.merge(
        annotations[["region_id", "biochemical_context"]],
        on="region_id",
        how="left",
    )
    rows: list[dict[str, object]] = []
    for pc, group in merged.groupby("PC", sort=False):
        context_sums = (
            group.groupby("biochemical_context")["relative_absolute_contribution"]
            .sum(min_count=1)
            .sort_values(ascending=False)
        )
        dominant_context = str(context_sums.index[0]) if not context_sums.empty else None
        mixed = len(context_sums) > 1 and float(context_sums.iloc[1]) > 0.25
        top_regions = group.sort_values("relative_absolute_contribution", ascending=False)["region_label"].head(3).tolist()
        rows.append(
            {
                "region_model": model_name,
                "PC": pc,
                "dominant_context": dominant_context,
                "top_region_1": top_regions[0] if len(top_regions) > 0 else None,
                "top_region_2": top_regions[1] if len(top_regions) > 1 else None,
                "top_region_3": top_regions[2] if len(top_regions) > 2 else None,
                "interpretation_sentence": _interpretation_sentence(pc, dominant_context, mixed),
            }
        )
    return pd.DataFrame(rows)


def fit_biological_pca_models(inputs: SubjectInputs, cfg: dict | None = None) -> list[ModelResult]:
    cfg = config if cfg is None else cfg
    validate_biological_pca_config(cfg)
    bp_cfg = _biological_cfg(cfg)
    _validate_axes_against_models(bp_cfg, inputs)
    n_components = int(bp_cfg.get("n_components", 5))
    scale = bool(bp_cfg.get("scale", False))
    results: list[ModelResult] = []

    for model_name in REQUIRED_MODELS:
        model_cfg = bp_cfg["models"][model_name]
        axis, matrix, blocks = _model_axis_and_matrix(model_name, model_cfg, inputs)
        _validate_component_count(model_name, matrix, n_components)
        if model_name == "PCA_whole":
            split_index = inputs.matrices["1300"].shape[1]
            pca_result = joint_pca_mfa(matrix, split_index, n_components)
            method_metadata = {
                "method": pca_result["method"],
                "method_version": 1,
                "split_index": split_index,
                "fingerprint_first_eigenvalue": pca_result["block_eigenvalues"]["fingerprint"],
                "ch_stretch_first_eigenvalue": pca_result["block_eigenvalues"]["ch_stretch"],
                "fingerprint_scaling_factor": pca_result["scaling_factors"]["fingerprint"],
                "ch_stretch_scaling_factor": pca_result["scaling_factors"]["ch_stretch"],
                "fingerprint_channels": split_index,
                "ch_stretch_channels": matrix.shape[1] - split_index,
                "fingerprint_normalization": "snv",
                "ch_stretch_normalization": "area",
            }
        else:
            pca_result = run_pca(
                matrix,
                n_components=n_components,
                scale=scale,
                log_summary=False,
                warn_on_high_pc1=False,
            )
            method_metadata = {"method": "mean_centered_pca", "method_version": 1}
        scores = _scores_frame(model_name, inputs.subjects, pca_result["scores"], inputs.subject_spectra)
        loadings = _loadings_frame(model_name, axis, blocks, pca_result["loadings"], inputs, cfg)
        explained = _explained_variance_frame(model_name, pca_result)
        loading_regions = compute_loading_regions(model_name, loadings, cfg)
        region_contributions = build_region_contributions(loading_regions)
        region_annotations = annotate_loading_regions(model_name, loading_regions, cfg)
        loading_region_summary = build_loading_region_summary(
            model_name,
            region_contributions,
            region_annotations,
        )
        compatibility_contributions = region_contributions.rename(
            columns={
                "region_id": "band_name",
                "region_label": "assignment",
            }
        ).copy()
        compatibility_summary = loading_region_summary.rename(
            columns={"dominant_context": "dominant_biochemical_class"}
        ).copy()
        pc1_ratio = float(pca_result.get("pc1_variance_ratio", np.nan))
        pc1_warning = (
            f"PC1 explains {pc1_ratio * 100:.1f}% variance. Inspect baseline/background and score structure."
            if np.isfinite(pc1_ratio) and pc1_ratio > 0.8
            else None
        )
        results.append(
            ModelResult(
                name=model_name,
                scores=scores,
                loadings=loadings,
                explained_variance=explained,
                loading_regions=loading_regions,
                region_contributions=region_contributions,
                region_annotation_candidates=region_annotations,
                loading_region_summary=loading_region_summary,
                band_contributions=compatibility_contributions,
                biochemical_summary=compatibility_summary,
                pc1_warning=pc1_warning,
                method_metadata=method_metadata,
            )
        )
    return results


def validate_unique_subject_rows(df: pd.DataFrame, table_name: str) -> None:
    if "subject_id" not in df.columns:
        raise ValueError(f"{table_name} is missing subject_id")
    if df["subject_id"].duplicated().any():
        duplicates = df.loc[df["subject_id"].duplicated(), "subject_id"].astype(str).tolist()
        raise ValueError(f"{table_name} contains duplicated subject_id values: {duplicates[:10]}")


def validate_outputs(
    inputs: SubjectInputs,
    results: list[ModelResult],
    features: pd.DataFrame,
    cfg: dict | None = None,
) -> None:
    cfg = config if cfg is None else cfg
    validate_unique_subject_rows(features, "subject_biological_pca_scores.csv")
    for result in results:
        validate_unique_subject_rows(result.scores, f"{result.name}/scores.csv")
        if result.name == "PCA_whole":
            if "spectral_block" not in result.loadings.columns:
                raise ValueError("PCA_whole loadings must include spectral_block")
            valid_blocks = set(result.loadings["spectral_block"].astype(str))
            if valid_blocks != {"1300", "2850"}:
                raise ValueError("PCA_whole spectral_block values must be exactly 1300 and 2850")
            if result.loadings.duplicated(subset=["spectral_block", "raman_shift"]).any():
                raise ValueError("PCA_whole loadings contain duplicated spectral identities")
        else:
            model_cfg = _biological_cfg(cfg)["models"][result.name]
            block = str(model_cfg["spectral_blocks"][0])
            window = model_cfg["window"]
            if str(window).lower() == "full":
                expected = inputs.axes[block]
            else:
                lower, upper = window
                expected = inputs.axes[block][(inputs.axes[block] >= lower) & (inputs.axes[block] <= upper)]
            actual = result.loadings["raman_shift"].to_numpy(dtype=float)
            if actual.size != expected.size or not np.allclose(actual, expected):
                raise ValueError(f"{result.name} loading rows do not match the selected region axis")
        for pc, group in result.region_contributions.groupby("PC", sort=False):
            rel = group["relative_absolute_contribution"]
            if rel.isna().all():
                continue
            rel_sum = float(rel.sum())
            if abs(rel_sum - 1.0) >= CONTRIBUTION_TOLERANCE:
                raise ValueError(f"{result.name} {pc} contributions sum to {rel_sum}, not 1.0")


def build_subject_biochemical_features(results: list[ModelResult]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for result in results:
        score_cols = [col for col in result.scores.columns if col.startswith("PC")]
        renamed = result.scores[["subject_id", *score_cols]].rename(
            columns={col: f"{result.name}_{col}" for col in score_cols}
        )
        frames.append(renamed)
    features = frames[0]
    for frame in frames[1:]:
        features = features.merge(frame, on="subject_id", how="inner")
    validate_unique_subject_rows(features, "subject_biological_pca_scores.csv")
    return features


def _explained_variance_pct(result: ModelResult, pc: str) -> float:
    row = result.explained_variance[result.explained_variance["PC"] == pc]
    if row.empty:
        return 0.0
    return float(row["explained_variance_ratio"].iloc[0]) * 100


def _save_plot(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _place_outside_legend(
    ax: plt.Axes,
    *,
    handles: list | None = None,
    labels: list[str] | None = None,
    title: str | None = None,
) -> None:
    if handles is None or labels is None:
        handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    ax.legend(
        handles,
        labels,
        title=title,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0,
        fontsize=9,
    )


def _scatter_scores(ax: plt.Axes, result: ModelResult, pc_x: str, pc_y: str) -> None:
    group_col = result.scores.get("group")
    emit_hull = bool(
        config.get("biological_pca", {}).get("figures", {}).get("emit_convex_hull", False)
    )
    if group_col is not None and not group_col.isna().all():
        for grp in sorted(group_col.dropna().unique()):
            mask = group_col == grp
            color = GROUP_COLORS.get(grp, "#6B7280")
            label = GROUP_LABELS.get(grp, grp)
            ax.scatter(
                result.scores.loc[mask, pc_x],
                result.scores.loc[mask, pc_y],
                s=45,
                color=color,
                alpha=0.85,
                edgecolors="#334155",
                linewidth=0.55,
                label=label,
            )
            if emit_hull and mask.sum() >= 3:
                pts = result.scores.loc[mask, [pc_x, pc_y]].to_numpy(dtype=float)
                try:
                    hull = ConvexHull(pts)
                    hull_pts = pts[hull.vertices]
                    ax.fill(hull_pts[:, 0], hull_pts[:, 1], color=color, alpha=0.12)
                except Exception:
                    pass
        _place_outside_legend(ax)
    else:
        ax.scatter(
            result.scores[pc_x],
            result.scores[pc_y],
            s=45,
            color="#2f6f73",
            alpha=0.85,
        )


def _plot_scores_pair(result: ModelResult, output_dir: Path, pc_x: str, pc_y: str, filename: str) -> None:
    if not {pc_x, pc_y}.issubset(result.scores.columns):
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    _scatter_scores(ax, result, pc_x, pc_y)
    ax.set_xlabel(f"{pc_x} ({_explained_variance_pct(result, pc_x):.1f}%)")
    ax.set_ylabel(f"{pc_y} ({_explained_variance_pct(result, pc_y):.1f}%)")
    ax.set_title(f"{result.name} subject scores", fontsize=12, loc="center", pad=8)
    fig.tight_layout()
    _save_plot(fig, output_dir / filename)


def _plot_scores_pairgrid(result: ModelResult, output_dir: Path) -> None:
    pairs = [("PC1", "PC2"), ("PC1", "PC3"), ("PC2", "PC3")]
    if not all({pc_x, pc_y}.issubset(result.scores.columns) for pc_x, pc_y in pairs):
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, (pc_x, pc_y) in zip(axes, pairs):
        _scatter_scores(ax, result, pc_x, pc_y)
        ax.set_xlabel(f"{pc_x} ({_explained_variance_pct(result, pc_x):.1f}%)")
        ax.set_ylabel(f"{pc_y} ({_explained_variance_pct(result, pc_y):.1f}%)")
        ax.set_title(f"{pc_x} vs {pc_y}", fontsize=12, loc="center")
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    if handles:
        fig.legend(handles, labels, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    fig.suptitle(f"{result.name} subject score views", y=1.02, fontsize=12, x=0.5, ha="center")
    fig.tight_layout()
    _save_plot(fig, output_dir / "scores_pc_pairgrid.png")


def _plot_scree(result: ModelResult, output_dir: Path) -> None:
    ratios = result.explained_variance["explained_variance_ratio"].to_numpy()
    fig, _ = plot_scree(ratios, title=f"{result.name} scree plot")
    _save_plot(fig, output_dir / "scree_plot.png")


def _plot_loadings(result: ModelResult, output_dir: Path, pcs: list[str] | None = None) -> None:
    if pcs is None:
        pcs = list(CANONICAL_PCS)
    available = [pc for pc in pcs if f"{pc}_loading" in result.loadings.columns]
    if not available:
        return
    is_compatibility = available == ["PC1", "PC2"]
    file_suffix = "_".join(pc.lower() for pc in available)
    if result.name == "PCA_whole":
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
        for ax, block in zip(axes, ["1300", "2850"]):
            block_df = result.loadings[result.loadings["spectral_block"].astype(str) == block]
            for pc in available:
                ax.plot(
                    block_df["raman_shift"],
                    block_df[f"{pc}_loading"],
                    color=PC_COLORS.get(pc, "#6B7280"),
                    label=pc,
                )
            ax.axhline(0, color="black", linestyle="--", linewidth=0.5)
            ax.set_xlabel(r"Raman shift (cm$^{-1}$)")
            ax.set_ylabel("Loading")
            ax.set_title(f"{result.name} {block} loadings", fontsize=12, loc="center")
            _place_outside_legend(ax, title=f"PC ({block})")
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        for pc in available:
            ax.plot(
                result.loadings["raman_shift"],
                result.loadings[f"{pc}_loading"],
                color=PC_COLORS.get(pc, "#6B7280"),
                label=pc,
            )
        ax.axhline(0, color="black", linestyle="--", linewidth=0.5)
        ax.set_xlabel("Raman shift (cm-1)")
        ax.set_ylabel("Loading")
        ax.set_title(f"{result.name} loadings", fontsize=12, loc="center")
        _place_outside_legend(ax, title="PC")
    fig.tight_layout()
    filename = "loadings_pc1_pc2.png" if is_compatibility else f"loadings_{file_suffix}.png"
    _save_plot(fig, output_dir / filename)


def _plot_band_contributions(
    result: ModelResult, output_dir: Path, pcs: list[str] | None = None
) -> None:
    if pcs is None:
        pcs = list(CANONICAL_PCS)
    df = result.region_contributions[
        result.region_contributions["PC"].isin(pcs)
        & result.region_contributions["relative_absolute_contribution"].notna()
    ]
    if df.empty:
        return
    pivot = (
        df.pivot(index="region_label", columns="PC", values="relative_absolute_contribution")
        .fillna(0)
        .reindex(columns=pcs, fill_value=0)
    )
    order = pivot.sum(axis=1).sort_values(ascending=True).index
    pivot = pivot.loc[order]
    display_labels = list(pivot.index)
    if result.name == "PCA_whole":
        block_lookup = (
            df.drop_duplicates(subset=["region_label"])
            .set_index("region_label")["spectral_block"]
            .to_dict()
        )
        display_labels = [f"{block_lookup.get(label, 'block')} | {label}" for label in pivot.index]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(pivot))))
    y = np.arange(len(pivot))
    available_pcs = [pc for pc in pcs if pc in pivot.columns]
    height = 0.22 if len(available_pcs) >= 3 else 0.35
    offsets = np.linspace(-(len(available_pcs) - 1) / 2, (len(available_pcs) - 1) / 2, len(available_pcs)) * height
    for i, pc in enumerate(available_pcs):
        ax.barh(
            y + offsets[i],
            pivot[pc],
            height,
            label=pc,
            color=PC_COLORS.get(pc, "#6B7280"),
            alpha=0.85,
            edgecolor="#334155",
            linewidth=0.3,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(display_labels, fontsize=9)
    ax.set_xlabel("Relative absolute contribution")
    ax.set_title(f"{result.name} loading-derived regions", fontsize=12, loc="center")
    _place_outside_legend(
        ax,
        handles=[Line2D([0], [0], color=PC_COLORS.get(pc, "#6B7280"), lw=6) for pc in available_pcs],
        labels=available_pcs,
        title="PC",
    )
    fig.tight_layout()
    filename = "loading_regions_pc1_pc2.png" if available_pcs == ["PC1", "PC2"] else f"loading_regions_{'_'.join(pc.lower() for pc in available_pcs)}.png"
    _save_plot(fig, output_dir / filename)

    if available_pcs == ["PC1", "PC2"]:
        fig2, ax2 = plt.subplots(figsize=(8, max(4, 0.35 * len(pivot))))
        for i, pc in enumerate(available_pcs):
            ax2.barh(
                y + np.array([-0.175, 0.175])[i],
                pivot[pc],
                0.35,
                label=pc,
                color=PC_COLORS.get(pc, "#6B7280"),
                alpha=0.85,
                edgecolor="#334155",
                linewidth=0.3,
            )
        ax2.set_yticks(y)
        ax2.set_yticklabels(display_labels, fontsize=9)
        ax2.set_xlabel("Relative absolute contribution")
        ax2.set_title(f"{result.name} loading-derived regions", fontsize=12, loc="center")
        _place_outside_legend(ax2, title="PC")
        fig2.tight_layout()
        _save_plot(fig2, output_dir / "band_contributions_pc1_pc2.png")


def _plot_biochemical_summary(result: ModelResult, output_dir: Path, pcs: list[str] | None = None) -> None:
    if pcs is None:
        pcs = list(CANONICAL_PCS)
    df = result.loading_region_summary[result.loading_region_summary["PC"].isin(pcs)]
    if df.empty:
        return
    merged = result.region_contributions.merge(
        result.region_annotation_candidates[["region_id", "biochemical_context"]],
        on="region_id",
        how="left",
    )
    merged = merged[merged["PC"].isin(pcs)]
    if merged.empty:
        return
    pivot = (
        merged.pivot_table(
            index="PC",
            columns="biochemical_context",
            values="relative_absolute_contribution",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(pcs, fill_value=0)
    )
    class_colors = {
        "lipid_associated_region": "#E69F00",
        "protein_associated_region": "#0072B2",
        "nucleic_acid_associated_region": "#009E73",
        "mixed_context_region": "#6D597A",
        "unassigned_broad_interval": "#6B7280",
    }
    fig, ax = plt.subplots(figsize=(6, 4))
    bottom = np.zeros(len(pivot))
    for cls in pivot.columns:
        vals = pivot[cls].to_numpy()
        if not vals.any():
            continue
        ax.bar(
            pivot.index,
            vals,
            bottom=bottom,
            label=cls,
            color=class_colors[cls],
            alpha=0.85,
            edgecolor="#334155",
            linewidth=0.3,
        )
        bottom += vals
    ax.set_ylabel("Relative contribution")
    ax.set_title(f"{result.name} region annotation summary", fontsize=12, loc="center")
    ax.set_ylim(0, 1)
    _place_outside_legend(ax, title="Broad context")
    fig.tight_layout()
    filename = "biochemical_summary_pc1_pc2.png" if list(pcs) == ["PC1", "PC2"] else f"annotation_summary_{'_'.join(pc.lower() for pc in pcs)}.png"
    _save_plot(fig, output_dir / filename)


def _write_interpretation_md(result: ModelResult, output_dir: Path) -> None:
    ev = result.explained_variance.set_index("PC")
    pcs = [pc for pc in CANONICAL_PCS if pc in ev.index]

    lines = [
        f"# {result.name} - Conservative Interpretation",
        "",
        "Input: subject median spectra (one median spectrum per subject per spectral region).",
        "",
        "| PC | Explained variance | Dominant class | Top regions | Interpretation |",
        "|---|---|---|---|---|",
    ]
    for pc in pcs:
        pct = ev.loc[pc, "explained_variance_ratio"] * 100
        summary_row = result.loading_region_summary[result.loading_region_summary["PC"] == pc]
        dominant = summary_row["dominant_context"].iloc[0] if not summary_row.empty else None
        sentence = summary_row["interpretation_sentence"].iloc[0] if not summary_row.empty else ""
        top_regions = (
            result.region_contributions[
                (result.region_contributions["PC"] == pc)
                & result.region_contributions["relative_absolute_contribution"].notna()
            ]
            .sort_values("relative_absolute_contribution", ascending=False)
            .head(5)
        )
        region_list = ", ".join(
            f"{row['region_label']} ({row['relative_absolute_contribution'] * 100:.1f}%)"
            for _, row in top_regions.iterrows()
        )
        lines.append(
            f"| {pc} | {pct:.1f}% | {dominant or 'N/A'} | {region_list} | {sentence} |"
        )

    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "This interpretation is exploratory and descriptive only. It does not support diagnosis, prediction, causal inference, or use as a clinical indicator.",
        ]
    )
    text = "\n".join(lines)
    lower = text.lower()
    if any(phrase in lower for phrase in FORBIDDEN_INTERPRETATION_PHRASES):
        raise ValueError(f"Forbidden interpretation wording in {result.name} markdown")
    (output_dir / "conservative_interpretation.md").write_text(text, encoding="utf-8")


def _build_pca_whole_block_contributions(result: ModelResult) -> pd.DataFrame:
    """Compute per-PC loading variance share for each spectral block in PCA_whole.

    MFA scaling equalizes each block's leading eigenvalue before joint PCA, but
    individual PCs may still have legitimate block-dominant loading shares.
    Rows: one per (PC, spectral_block). Columns: PC, spectral_block, n_features,
    sum_sq_loading, block_variance_share.
    """
    if "spectral_block" not in result.loadings.columns:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    pcs = [pc for pc in CANONICAL_PCS if f"{pc}_loading" in result.loadings.columns]
    for pc in pcs:
        col = f"{pc}_loading"
        total_ss = float((result.loadings[col] ** 2).sum())
        for block, grp in result.loadings.groupby("spectral_block"):
            block_ss = float((grp[col] ** 2).sum())
            rows.append(
                {
                    "PC": pc,
                    "spectral_block": str(block),
                    "n_features": int(len(grp)),
                    "sum_sq_loading": round(block_ss, 6),
                    "block_variance_share": round(block_ss / total_ss, 4) if total_ss > 0 else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _build_pca_comparison_summary(results: list[ModelResult]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for result in results:
        ev = result.explained_variance.set_index("PC")
        for pc in CANONICAL_PCS:
            if pc not in ev.index:
                continue
            pct = ev.loc[pc, "explained_variance_ratio"] * 100
            summary_row = result.loading_region_summary[result.loading_region_summary["PC"] == pc]
            dominant = summary_row["dominant_context"].iloc[0] if not summary_row.empty else None
            sentence = summary_row["interpretation_sentence"].iloc[0] if not summary_row.empty else ""
            top_regions = (
                result.region_contributions[
                    (result.region_contributions["PC"] == pc)
                    & result.region_contributions["relative_absolute_contribution"].notna()
                ]
                .sort_values("relative_absolute_contribution", ascending=False)
                .head(3)["region_label"]
                .tolist()
            )
            rows.append(
                {
                    "model": result.name,
                    "PC": pc,
                    "explained_variance_pct": pct,
                    "dominant_context": dominant,
                    "top_region_1": top_regions[0] if len(top_regions) > 0 else None,
                    "top_region_2": top_regions[1] if len(top_regions) > 1 else None,
                    "top_region_3": top_regions[2] if len(top_regions) > 2 else None,
                    "interpretation_sentence": sentence,
                }
            )
    return pd.DataFrame(rows)


def _write_pc1_dominance_report(result: ModelResult, output_dir: Path) -> None:
    if not result.pc1_warning:
        return
    explained = result.explained_variance.set_index("PC")
    variance_pct = float(explained.loc["PC1", "explained_variance_ratio"]) * 100 if "PC1" in explained.index else np.nan
    cumulative_pc3 = (
        float(explained.loc["PC3", "cumulative_variance"]) * 100
        if "PC3" in explained.index
        else float(explained["cumulative_variance"].max()) * 100
    )
    top_regions = result.region_contributions[
        result.region_contributions["PC"] == "PC1"
    ].sort_values("relative_absolute_contribution", ascending=False).head(5)
    model_type = "whole-spectrum Biological PCA" if result.name == "PCA_whole" else "subject-level Biological PCA"
    lines = [
        f"# {result.name} PC1 Dominance Report",
        "",
        f"- Model name: {result.name}",
        f"- Model type: {model_type}",
        f"- Subjects used: {result.scores['subject_id'].nunique()}",
        f"- PC1 variance: {variance_pct:.1f}%",
        f"- Cumulative variance (PC1-PC3): {cumulative_pc3:.1f}%",
        "- Preprocessing context: subject medians built from pixel-level preprocessed, normalized spectra",
        "- Score plots: scores_pc1_pc2.png, scores_pc1_pc3.png, scores_pc2_pc3.png, scores_pc_pairgrid.png",
        "- Loading plots: loadings_pc1_pc2_pc3.png, loading_regions_pc1_pc2_pc3.png",
        "",
        "This may reflect baseline/background structure, normalization effects, concentration differences, technical structure, between-subject structure, or true broad biological variance. Interpret PC1 cautiously and inspect scores and loading-derived regions before drawing conclusions.",
        "",
        "## Top loading-derived regions",
        "",
    ]
    for _, row in top_regions.iterrows():
        lines.append(
            f"- {row['region_label']}: {row['relative_absolute_contribution'] * 100:.1f}% of PC1 integrated absolute loading"
        )
    (output_dir / "pc1_dominance_report.md").write_text("\n".join(lines), encoding="utf-8")


def _write_biological_pca_summary_markdown(summary_df: pd.DataFrame, output_dir: Path) -> None:
    lines = [
        "# Biological PCA Summary",
        "",
        "| Model | PC | Explained variance | Dominant context | Top region | Interpretation |",
        "|---|---|---|---|---|---|",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"| {row['model']} | {row['PC']} | {row['explained_variance_pct']:.1f}% | "
            f"{row.get('dominant_context') or 'N/A'} | {row.get('top_region_1') or 'N/A'} | "
            f"{row['interpretation_sentence']} |"
        )
    (output_dir / "biological_pca_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _write_model_readme(output_dir: Path) -> None:
    (output_dir / "README.md").write_text("\n".join(
            [
                "# Biological PCA Outputs",
                "",
                "Canonical PC1-PC3 figures:",
                "",
                "- scores_pc1_pc2.png",
                "- scores_pc1_pc3.png",
                "- scores_pc2_pc3.png",
                "- scores_pc_pairgrid.png",
                "- loadings_pc1_pc2_pc3.png",
                "- loading_regions_pc1_pc2_pc3.png",
                "",
                "Compatibility figures retained for one cycle:",
                "",
                "- loadings_pc1_pc2.png",
                "- loading_regions_pc1_pc2.png",
                "- band_contributions_pc1_pc2.png",
                "- biochemical_summary_pc1_pc2.png",
            ]
        ),
        encoding="utf-8",
    )


def _write_outputs(
    inputs: SubjectInputs,
    results: list[ModelResult],
    features: pd.DataFrame,
    cfg: dict,
) -> dict[str, Path]:
    cross_region_dir = get_cross_region_summary_dir()
    subject_features_dir = get_cross_region_phase_dir("subject_features")
    summary_dir = get_cross_region_phase_dir("biological_pca_summary")
    comparison = _build_pca_comparison_summary(results)

    # Build subject-retention audit table (included + excluded with reasons).
    included_rows = [
        {"subject_id": sid, "status": "included", "reason": ""}
        for sid in inputs.subjects
    ]
    excluded_rows = [
        {"subject_id": d["subject_id"], "status": "excluded", "reason": d.get("reason", "")}
        for d in (inputs.excluded_subjects or [])
    ]
    retention_df = pd.DataFrame(included_rows + excluded_rows).sort_values("subject_id").reset_index(drop=True)
    retention_detail = inputs.subject_region_counts.merge(
        inputs.subject_spectra[[c for c in ("subject_id", "condition", "gender") if c in inputs.subject_spectra]],
        on="subject_id",
        how="left",
    )
    retention_rows: list[dict[str, object]] = []
    for stratum in ("condition", "gender"):
        if stratum not in retention_detail.columns:
            continue
        grouped = retention_detail.groupby(["spectral_range", stratum], dropna=False, sort=True)
        for (spectral_range, level), group in grouped:
            n_input = int(group["n_pixels"].sum())
            n_retained = int(group["n_retained_pixels"].sum())
            retention_rows.append(
                {
                    "spectral_block": spectral_range,
                    "stratum": stratum,
                    "level": level,
                    "n_input": n_input,
                    "n_retained": n_retained,
                    "retention_fraction": n_retained / n_input if n_input else np.nan,
                }
            )
    retention_by_stratum = pd.DataFrame(retention_rows)

    feature_tables = {
        "subject_spectra.csv": inputs.subject_spectra,
        "subject_region_counts.csv": inputs.subject_region_counts,
        "input_provenance.csv": inputs.input_provenance,
        "subject_retention.csv": retention_df,
        "qc_retention_by_stratum.csv": retention_by_stratum,
    }
    for filename, df in feature_tables.items():
        df.to_csv(subject_features_dir / filename, index=False)

    consistency_rows: list[dict[str, object]] = []
    written_paths: dict[str, Path] = {}
    for result in results:
        model_dir = get_biological_pca_canonical_dir(result.name)
        written_paths[result.name] = model_dir
        model_tables = {
            "scores.csv": result.scores,
            "loadings.csv": result.loadings,
            "explained_variance.csv": result.explained_variance,
            "loading_regions.csv": result.loading_regions,
            "region_contributions.csv": result.region_contributions,
            "region_annotation_candidates.csv": result.region_annotation_candidates,
            "loading_region_summary.csv": result.loading_region_summary,
            "band_contributions.csv": result.band_contributions,
            "biochemical_summary.csv": result.biochemical_summary,
        }
        for filename, df in model_tables.items():
            df.to_csv(model_dir / filename, index=False)
        pd.DataFrame([result.method_metadata]).to_csv(model_dir / "pca_method_metadata.csv", index=False)

        if result.name == "PCA_whole":
            sensitivity_dir = model_dir / "sensitivity_unscaled"
            sensitivity_dir.mkdir(parents=True, exist_ok=True)
            unscaled_matrix = np.hstack([inputs.matrices["1300"], inputs.matrices["2850"]])
            unscaled = run_pca(
                unscaled_matrix,
                n_components=len(CANONICAL_PCS),
                scale=False,
                log_summary=False,
                warn_on_high_pc1=False,
            )
            whole_axis = np.concatenate([inputs.axes["1300"], inputs.axes["2850"]])
            _scores_frame(
                "PCA_whole_unscaled_sensitivity",
                inputs.subjects,
                unscaled["scores"],
                inputs.subject_spectra,
            ).to_csv(sensitivity_dir / "scores.csv", index=False)
            _loadings_frame(
                "PCA_whole",
                whole_axis,
                ["1300", "2850"],
                unscaled["loadings"],
                inputs,
                cfg,
            ).to_csv(sensitivity_dir / "loadings.csv", index=False)
            _explained_variance_frame("PCA_whole_unscaled_sensitivity", unscaled).to_csv(
                sensitivity_dir / "explained_variance.csv", index=False
            )
            pd.DataFrame(
                [{"method": "unscaled_concatenation_sensitivity", "canonical": False}]
            ).to_csv(sensitivity_dir / "pca_method_metadata.csv", index=False)
            block_contrib = _build_pca_whole_block_contributions(result)
            # Always write so readers can check for block-dominance artefacts.
            block_contrib.to_csv(model_dir / "pca_whole_block_contributions.csv", index=False)
            if not block_contrib.empty:
                for pc_col in [c for c in block_contrib.columns if c.endswith("_var_share")]:
                    dominant = block_contrib.loc[block_contrib[pc_col].idxmax(), "spectral_block"] if not block_contrib.empty else "unknown"
                    share = block_contrib[pc_col].max() if not block_contrib.empty else float("nan")
                    if share > 0.75:
                        logger.warning(
                            "PCA_whole %s: block '%s' accounts for %.0f%% of variance — potential "
                            "scale-dominance artefact; inspect pca_whole_block_contributions.csv",
                            pc_col.replace("_var_share", ""),
                            dominant,
                            share * 100,
                        )

        pcs = [pc for pc in CANONICAL_PCS if pc in result.explained_variance["PC"].tolist()]
        _plot_scores_pair(result, model_dir, "PC1", "PC2", "scores_pc1_pc2.png")
        _plot_scores_pair(result, model_dir, "PC1", "PC3", "scores_pc1_pc3.png")
        _plot_scores_pair(result, model_dir, "PC2", "PC3", "scores_pc2_pc3.png")
        _plot_scores_pairgrid(result, model_dir)
        _plot_scree(result, model_dir)
        _plot_loadings(result, model_dir, pcs=pcs)
        _plot_loadings(result, model_dir, pcs=["PC1", "PC2"])
        _plot_band_contributions(result, model_dir, pcs=pcs)
        _plot_band_contributions(result, model_dir, pcs=["PC1", "PC2"])
        _plot_biochemical_summary(result, model_dir, pcs=pcs)
        _plot_biochemical_summary(result, model_dir, pcs=["PC1", "PC2"])
        _write_interpretation_md(result, model_dir)
        _write_pc1_dominance_report(result, model_dir)
        _write_model_readme(model_dir)

        for _, row in result.loading_region_summary.iterrows():
            consistency_rows.append(
                {
                    "model": result.name,
                    "PC": row["PC"],
                    "dominant_context": row["dominant_context"],
                    "top_region_1": row["top_region_1"],
                    "top_region_2": row["top_region_2"],
                    "top_region_3": row["top_region_3"],
                    "interpretation_sentence": row["interpretation_sentence"],
                }
            )

    region_consistency = pd.DataFrame(consistency_rows)
    features.to_csv(summary_dir / "subject_biological_pca_scores.csv", index=False)
    comparison.to_csv(summary_dir / "pca_comparison_summary.csv", index=False)
    region_consistency.to_csv(summary_dir / "region_consistency_summary.csv", index=False)
    _write_biological_pca_summary_markdown(comparison, summary_dir)
    written_paths["cross_region_summary"] = cross_region_dir
    written_paths["subject_features_dir"] = subject_features_dir
    written_paths["biological_pca_summary_dir"] = summary_dir
    return written_paths


def run_biological_pca(
    *,
    output_dir: str | Path | None = None,
    max_subjects: int | None = None,
    cfg: dict | None = None,
) -> dict[str, Path]:
    cfg = config if cfg is None else cfg
    if output_dir is not None:
        out = Path(output_dir)
        if not out.is_absolute():
            out = resolve_project_path(out)
        assert out is not None

    inputs = build_subject_inputs(max_subjects=max_subjects, cfg=cfg)
    written_paths = write_subject_feature_exports(inputs, cfg=cfg)
    results = fit_biological_pca_models(inputs, cfg=cfg)
    features = build_subject_biochemical_features(results)
    validate_outputs(inputs, results, features, cfg=cfg)
    written_paths.update(_write_outputs(inputs, results, features, cfg))
    logger.info("Biological PCA outputs written to %s", get_cross_region_summary_dir().parent)
    return written_paths

