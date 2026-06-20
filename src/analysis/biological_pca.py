"""Strict subject-level Biological PCA outputs for Raman spectra."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import shutil
import tempfile
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull

from src import _output_dir, config, resolve_project_path
from src.analysis.statistics import run_pca
from src.io.file_crawler import build_file_index
from src.io.txt_reader import read_txt_spectrum
from src.preprocessing.pipeline import run_preprocessing
from src.visualization.eda_plots import plot_scree
from src.visualization.theme import GROUP_COLORS, GROUP_LABELS

logger = logging.getLogger(__name__)

REQUIRED_MODELS = ("PCA_1330", "PCA_2850", "PCA_whole")
BLOCKS = ("1300", "2850")
CONTRIBUTION_TOLERANCE = 1e-6
FORBIDDEN_INTERPRETATION_PHRASES = (
    "lipids are increased",
    "protein abundance is higher",
    "this proves disease separation",
    "this biomarker diagnoses disease",
    "diagnoses disease",
    "disease separation",
    "biomarker",
)


@dataclass(frozen=True)
class SubjectInputs:
    subject_spectra: pd.DataFrame
    subject_region_counts: pd.DataFrame
    input_provenance: pd.DataFrame
    axes: dict[str, np.ndarray]
    matrices: dict[str, np.ndarray]
    subjects: list[str]


@dataclass(frozen=True)
class ModelResult:
    name: str
    scores: pd.DataFrame
    loadings: pd.DataFrame
    explained_variance: pd.DataFrame
    band_contributions: pd.DataFrame
    biochemical_summary: pd.DataFrame


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


def _assert_axis(existing: np.ndarray | None, candidate: np.ndarray, block: str) -> np.ndarray:
    candidate = np.asarray(candidate, dtype=float)
    if existing is None:
        return candidate
    if existing.size != candidate.size or not np.allclose(existing, candidate, rtol=0, atol=1e-8):
        raise ValueError(f"Post-preprocessing Raman axes are inconsistent within spectral block {block}")
    return existing


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


def _band_axis_for_model(
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


def compute_band_contributions(
    model_name: str,
    model_cfg: dict,
    loadings_df: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    windows = _band_windows(cfg)
    band_to_class = _band_class_map(_biological_cfg(cfg))
    axis_by_block = _band_axis_for_model(model_name, loadings_df)
    rows: list[dict[str, object]] = []
    pc_cols = [col for col in loadings_df.columns if col.startswith("PC") and col.endswith("_loading")]

    for pc_col in pc_cols:
        pc_name = pc_col.replace("_loading", "")
        pc_rows: list[dict[str, object]] = []
        for band_name in model_cfg.get("bands", []):
            lower, upper = windows[str(band_name)]
            axis_key = _band_block_for_window(lower, upper) if model_name == "PCA_whole" else "single"
            band_axis_df = axis_by_block.get(axis_key, pd.DataFrame())
            axis = band_axis_df["raman_shift"].to_numpy(dtype=float) if not band_axis_df.empty else np.array([])
            mask = (axis >= lower) & (axis <= upper)
            if not mask.any():
                continue
            represented_axis = axis[mask]
            loading_values = band_axis_df.loc[mask, pc_col].to_numpy(dtype=float)
            if represented_axis.size == 1:
                signed = float(loading_values[0])
            else:
                signed = float(np.trapz(loading_values, represented_axis))
            absolute = abs(signed)
            pc_rows.append(
                {
                    "region_model": model_name,
                    "PC": pc_name,
                    "band_name": str(band_name),
                    "window_min": float(lower),
                    "window_max": float(upper),
                    "represented_window_min": float(np.min(represented_axis)),
                    "represented_window_max": float(np.max(represented_axis)),
                    "assignment": str(band_name),
                    "biochemical_class": band_to_class[str(band_name)],
                    "signed_integrated_loading": signed,
                    "absolute_integrated_loading": absolute,
                    "relative_absolute_contribution": np.nan,
                    "dominant_sign": "positive" if signed > 0 else ("negative" if signed < 0 else "zero"),
                }
            )
        total = float(sum(row["absolute_integrated_loading"] for row in pc_rows))
        if total == 0 and pc_rows:
            logger.warning(
                "%s %s has zero total absolute integrated loading across included bands; "
                "relative contributions are NaN",
                model_name,
                pc_name,
            )
        elif pc_rows:
            for row in pc_rows:
                row["relative_absolute_contribution"] = row["absolute_integrated_loading"] / total
            rel_sum = sum(row["relative_absolute_contribution"] for row in pc_rows)
            if abs(rel_sum - 1.0) >= CONTRIBUTION_TOLERANCE:
                raise ValueError(
                    f"{model_name} {pc_name} band contribution normalization failed: {rel_sum}"
                )
        rows.extend(pc_rows)

    return pd.DataFrame(
        rows,
        columns=[
            "region_model",
            "PC",
            "band_name",
            "window_min",
            "window_max",
            "represented_window_min",
            "represented_window_max",
            "assignment",
            "biochemical_class",
            "signed_integrated_loading",
            "absolute_integrated_loading",
            "relative_absolute_contribution",
            "dominant_sign",
        ],
    )


def _interpretation_sentence(pc: str, dominant_class: str | None, mixed: bool) -> str:
    if dominant_class == "lipid" and not mixed:
        sentence = f"{pc} is dominated by lipid-associated Raman loading contributions."
    elif dominant_class in {"protein", "nucleic_acid"} and not mixed:
        sentence = (
            "This component reflects a regional Raman loading pattern, "
            "not a molecule-specific assignment."
        )
    else:
        sentence = f"{pc} shows mixed protein/lipid-associated loading contributions."
    lower = sentence.lower()
    if any(phrase in lower for phrase in FORBIDDEN_INTERPRETATION_PHRASES):
        raise ValueError(f"Forbidden interpretation wording generated: {sentence}")
    return sentence


def biochemical_summary(model_name: str, contributions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if contributions.empty:
        return pd.DataFrame(
            columns=[
                "region_model",
                "PC",
                "lipid",
                "protein",
                "nucleic_acid",
                "mixed",
                "dominant_biochemical_class",
                "interpretation_sentence",
            ]
        )
    for pc, group in contributions.groupby("PC", sort=False):
        class_sums = (
            group.groupby("biochemical_class")["relative_absolute_contribution"]
            .sum(min_count=1)
            .reindex(["lipid", "protein", "nucleic_acid", "mixed"])
        )
        finite = class_sums.dropna()
        dominant = str(finite.idxmax()) if not finite.empty else None
        is_mixed = (
            dominant == "mixed"
            or finite.size == 0
            or (finite.size > 1 and finite.sort_values(ascending=False).iloc[:2].min() > 0.25)
        )
        rows.append(
            {
                "region_model": model_name,
                "PC": pc,
                "lipid": class_sums.get("lipid", np.nan),
                "protein": class_sums.get("protein", np.nan),
                "nucleic_acid": class_sums.get("nucleic_acid", np.nan),
                "mixed": class_sums.get("mixed", np.nan),
                "dominant_biochemical_class": dominant,
                "interpretation_sentence": _interpretation_sentence(pc, dominant, is_mixed),
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
        pca_result = run_pca(matrix, n_components=n_components, scale=scale)
        scores = _scores_frame(model_name, inputs.subjects, pca_result["scores"], inputs.subject_spectra)
        loadings = _loadings_frame(model_name, axis, blocks, pca_result["loadings"], inputs, cfg)
        explained = _explained_variance_frame(model_name, pca_result)
        contributions = compute_band_contributions(model_name, model_cfg, loadings, cfg)
        summary = biochemical_summary(model_name, contributions)
        results.append(ModelResult(model_name, scores, loadings, explained, contributions, summary))
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
    validate_unique_subject_rows(features, "subject_biochemical_pca_features.csv")
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
            lower, upper = model_cfg["window"]
            expected = inputs.axes[block][(inputs.axes[block] >= lower) & (inputs.axes[block] <= upper)]
            actual = result.loadings["raman_shift"].to_numpy(dtype=float)
            if actual.size != expected.size or not np.allclose(actual, expected):
                raise ValueError(f"{result.name} loading rows do not match the selected region axis")
        for pc, group in result.band_contributions.groupby("PC", sort=False):
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
    validate_unique_subject_rows(features, "subject_biochemical_pca_features.csv")
    return features


def _explained_variance_pct(result: ModelResult, pc: str) -> float:
    row = result.explained_variance[result.explained_variance["PC"] == pc]
    if row.empty:
        return 0.0
    return float(row["explained_variance_ratio"].iloc[0]) * 100


def _plot_scores(result: ModelResult, output_dir: Path) -> None:
    if not {"PC1", "PC2"}.issubset(result.scores.columns):
        return
    fig, ax = plt.subplots(figsize=(6, 5))
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
                result.scores.loc[mask, "PC1"],
                result.scores.loc[mask, "PC2"],
                s=45,
                color=color,
                alpha=0.85,
                edgecolors="#334155",
                linewidth=0.55,
                label=label,
            )
            if emit_hull and mask.sum() >= 3:
                pts = result.scores.loc[mask, ["PC1", "PC2"]].to_numpy(dtype=float)
                try:
                    hull = ConvexHull(pts)
                    hull_pts = pts[hull.vertices]
                    ax.fill(hull_pts[:, 0], hull_pts[:, 1], color=color, alpha=0.12)
                except Exception:
                    pass
        ax.legend(fontsize=9)
    else:
        ax.scatter(
            result.scores["PC1"],
            result.scores["PC2"],
            s=45,
            color="#2f6f73",
            alpha=0.85,
        )
    ax.set_xlabel(f"PC1 ({_explained_variance_pct(result, 'PC1'):.1f}%)")
    ax.set_ylabel(f"PC2 ({_explained_variance_pct(result, 'PC2'):.1f}%)")
    ax.set_title(f"{result.name} subject scores")
    fig.tight_layout()
    fig.savefig(output_dir / "scores_pc1_pc2.png", dpi=200)
    plt.close(fig)


def _plot_scree(result: ModelResult, output_dir: Path) -> None:
    ratios = result.explained_variance["explained_variance_ratio"].to_numpy()
    fig, _ = plot_scree(ratios, title=f"{result.name} scree plot")
    fig.savefig(output_dir / "scree_plot.png", dpi=200)
    plt.close(fig)


def _plot_loadings(result: ModelResult, output_dir: Path, pcs: list[str] | None = None) -> None:
    if pcs is None:
        pcs = ["PC1", "PC2"]
    available = [pc for pc in pcs if f"{pc}_loading" in result.loadings.columns]
    if not available:
        return
    colors = ["#264653", "#2A9D8F", "#E76F51"]
    if result.name == "PCA_whole":
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
        for ax, block in zip(axes, ["1300", "2850"]):
            block_df = result.loadings[result.loadings["spectral_block"].astype(str) == block]
            for i, pc in enumerate(available):
                ax.plot(
                    block_df["raman_shift"],
                    block_df[f"{pc}_loading"],
                    color=colors[i % len(colors)],
                    label=f"{block}__{pc}",
                )
            ax.axhline(0, color="black", linestyle="--", linewidth=0.5)
            ax.set_xlabel("Raman shift (cm-1)")
            ax.set_ylabel("Loading")
            ax.set_title(f"{result.name} {block} loadings")
            ax.legend(fontsize=9)
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        for i, pc in enumerate(available):
            ax.plot(
                result.loadings["raman_shift"],
                result.loadings[f"{pc}_loading"],
                color=colors[i % len(colors)],
                label=pc,
            )
        ax.axhline(0, color="black", linestyle="--", linewidth=0.5)
        ax.set_xlabel("Raman shift (cm-1)")
        ax.set_ylabel("Loading")
        ax.set_title(f"{result.name} loadings")
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "loadings_pc1_pc2.png", dpi=200)
    plt.close(fig)


def _plot_band_contributions(
    result: ModelResult, output_dir: Path, pcs: list[str] | None = None
) -> None:
    if pcs is None:
        pcs = ["PC1", "PC2"]
    df = result.band_contributions[
        result.band_contributions["PC"].isin(pcs)
        & result.band_contributions["relative_absolute_contribution"].notna()
    ]
    if df.empty:
        return
    pivot = (
        df.pivot(index="band_name", columns="PC", values="relative_absolute_contribution")
        .fillna(0)
        .reindex(columns=pcs, fill_value=0)
    )
    order = pivot.sum(axis=1).sort_values(ascending=True).index
    pivot = pivot.loc[order]
    class_map = dict(zip(df["band_name"], df["biochemical_class"]))
    class_colors = {
        "lipid": "#E69F00",
        "protein": "#0072B2",
        "nucleic_acid": "#009E73",
        "mixed": "#6D597A",
    }
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(pivot))))
    y = np.arange(len(pivot))
    height = 0.35
    for i, pc in enumerate(pcs):
        if pc not in pivot.columns:
            continue
        offset = (i - 0.5) * height
        colors = [class_colors.get(class_map.get(b, "mixed"), "#6D597A") for b in pivot.index]
        ax.barh(
            y + offset,
            pivot[pc],
            height,
            label=pc,
            color=colors,
            alpha=0.85,
            edgecolor="#334155",
            linewidth=0.3,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_xlabel("Relative absolute contribution")
    ax.set_title(f"{result.name} band contributions")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "band_contributions_pc1_pc2.png", dpi=200)
    plt.close(fig)


def _plot_biochemical_summary(result: ModelResult, output_dir: Path) -> None:
    df = result.biochemical_summary[result.biochemical_summary["PC"].isin(["PC1", "PC2"])]
    if df.empty:
        return
    class_cols = ["lipid", "protein", "nucleic_acid", "mixed"]
    pivot = df.set_index("PC")[class_cols].fillna(0)
    class_colors = {
        "lipid": "#E69F00",
        "protein": "#0072B2",
        "nucleic_acid": "#009E73",
        "mixed": "#6D597A",
    }
    fig, ax = plt.subplots(figsize=(6, 4))
    bottom = np.zeros(len(pivot))
    for cls in class_cols:
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
    ax.set_title(f"{result.name} biochemical class summary")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "biochemical_summary_pc1_pc2.png", dpi=200)
    plt.close(fig)


def _write_interpretation_md(
    result: ModelResult, output_dir: Path, pc3_threshold: float = 0.05
) -> None:
    ev = result.explained_variance.set_index("PC")
    pcs = ["PC1", "PC2"]
    pc3_ratio = ev.loc["PC3", "explained_variance_ratio"] if "PC3" in ev.index else 0.0
    include_pc3 = pc3_ratio >= pc3_threshold
    if include_pc3:
        pcs.append("PC3")

    lines = [
        f"# {result.name} — Conservative Interpretation",
        "",
        "Input: subject median spectra (one median spectrum per subject per spectral region).",
        "",
        "| PC | Explained variance | Dominant class | Top bands | Interpretation |",
        "|---|---|---|---|---|",
    ]
    for pc in pcs:
        if pc not in ev.index:
            continue
        pct = ev.loc[pc, "explained_variance_ratio"] * 100
        summary_row = result.biochemical_summary[result.biochemical_summary["PC"] == pc]
        dominant = (
            summary_row["dominant_biochemical_class"].iloc[0]
            if not summary_row.empty
            else None
        )
        sentence = (
            summary_row["interpretation_sentence"].iloc[0]
            if not summary_row.empty
            else ""
        )
        top_bands = (
            result.band_contributions[
                (result.band_contributions["PC"] == pc)
                & result.band_contributions["relative_absolute_contribution"].notna()
            ]
            .sort_values("relative_absolute_contribution", ascending=False)
            .head(5)
        )
        band_list = ", ".join(
            f"{row['band_name']} ({row['relative_absolute_contribution'] * 100:.1f}%)"
            for _, row in top_bands.iterrows()
        )
        lines.append(
            f"| {pc} | {pct:.1f}% | {dominant or 'N/A'} | {band_list} | {sentence} |"
        )

    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "This interpretation is exploratory and descriptive only. It does not support "
            "diagnosis, prediction, causal inference, or use as a clinical indicator.",
        ]
    )
    text = "\n".join(lines)
    lower = text.lower()
    if any(phrase in lower for phrase in FORBIDDEN_INTERPRETATION_PHRASES):
        raise ValueError(f"Forbidden interpretation wording in {result.name} markdown")
    (output_dir / "conservative_interpretation.md").write_text(text, encoding="utf-8")


def _build_pca_comparison_summary(results: list[ModelResult]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for result in results:
        ev = result.explained_variance.set_index("PC")
        for pc in ["PC1", "PC2"]:
            if pc not in ev.index:
                continue
            pct = ev.loc[pc, "explained_variance_ratio"] * 100
            summary_row = result.biochemical_summary[result.biochemical_summary["PC"] == pc]
            dominant = (
                summary_row["dominant_biochemical_class"].iloc[0]
                if not summary_row.empty
                else None
            )
            sentence = (
                summary_row["interpretation_sentence"].iloc[0]
                if not summary_row.empty
                else ""
            )
            top_bands = (
                result.band_contributions[
                    (result.band_contributions["PC"] == pc)
                    & result.band_contributions["relative_absolute_contribution"].notna()
                ]
                .sort_values("relative_absolute_contribution", ascending=False)
                .head(3)["band_name"]
                .tolist()
            )
            rows.append(
                {
                    "model": result.name,
                    "PC": pc,
                    "explained_variance_pct": pct,
                    "dominant_class": dominant,
                    "top_band_1": top_bands[0] if len(top_bands) > 0 else None,
                    "top_band_2": top_bands[1] if len(top_bands) > 1 else None,
                    "top_band_3": top_bands[2] if len(top_bands) > 2 else None,
                    "interpretation_sentence": sentence,
                }
            )
    return pd.DataFrame(rows)


def _write_outputs(
    staging_dir: Path,
    inputs: SubjectInputs,
    results: list[ModelResult],
    features: pd.DataFrame,
) -> list[Path]:
    written: list[Path] = []
    tables = {
        "subject_spectra.csv": inputs.subject_spectra,
        "subject_region_counts.csv": inputs.subject_region_counts,
        "input_provenance.csv": inputs.input_provenance,
        "subject_biochemical_pca_features.csv": features,
    }
    for filename, df in tables.items():
        path = staging_dir / filename
        df.to_csv(path, index=False)
        written.append(Path(filename))

    pc3_threshold = float(
        config.get("biological_pca", {}).get("pc3_variance_threshold", 0.05)
    )

    for result in results:
        model_dir = staging_dir / result.name
        model_dir.mkdir(parents=True, exist_ok=True)
        model_tables = {
            "scores.csv": result.scores,
            "loadings.csv": result.loadings,
            "explained_variance.csv": result.explained_variance,
            "band_contributions.csv": result.band_contributions,
            "biochemical_summary.csv": result.biochemical_summary,
        }
        for filename, df in model_tables.items():
            df.to_csv(model_dir / filename, index=False)
            written.append(Path(result.name) / filename)

        pcs = ["PC1", "PC2"]
        ev = result.explained_variance.set_index("PC")
        pc3_ratio = ev.loc["PC3", "explained_variance_ratio"] if "PC3" in ev.index else 0.0
        if pc3_ratio >= pc3_threshold:
            pcs.append("PC3")
            logger.info(
                "%s PC3 retained (%.1f%% variance)",
                result.name,
                pc3_ratio * 100,
            )
        else:
            logger.info(
                "%s PC3 not retained (%.1f%% variance < threshold %.0f%%)",
                result.name,
                pc3_ratio * 100,
                pc3_threshold * 100,
            )

        _plot_scores(result, model_dir)
        _plot_scree(result, model_dir)
        _plot_loadings(result, model_dir, pcs=pcs)
        _plot_band_contributions(result, model_dir, pcs=pcs)
        _plot_biochemical_summary(result, model_dir)
        _write_interpretation_md(result, model_dir, pc3_threshold=pc3_threshold)
        written.extend(
            [
                Path(result.name) / "scores_pc1_pc2.png",
                Path(result.name) / "scree_plot.png",
                Path(result.name) / "loadings_pc1_pc2.png",
                Path(result.name) / "band_contributions_pc1_pc2.png",
                Path(result.name) / "biochemical_summary_pc1_pc2.png",
                Path(result.name) / "conservative_interpretation.md",
            ]
        )

    comparison = _build_pca_comparison_summary(results)
    comparison.to_csv(staging_dir / "pca_comparison_summary.csv", index=False)
    written.append(Path("pca_comparison_summary.csv"))

    return written


def _promote_staging(staging_dir: Path, output_dir: Path, relative_files: Iterable[Path]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for relative in relative_files:
        src = staging_dir / relative
        dst = output_dir / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def run_biological_pca(
    *,
    output_dir: str | Path | None = None,
    max_subjects: int | None = None,
    cfg: dict | None = None,
) -> Path:
    cfg = config if cfg is None else cfg
    bp_cfg = _biological_cfg(cfg)
    if output_dir is None:
        subdir = str(bp_cfg.get("output_subdir", "biological_pca"))
        out = _output_dir("eda_subdir") / subdir
    else:
        out = Path(output_dir)
    if not out.is_absolute():
        out = resolve_project_path(out)
    assert out is not None

    inputs = build_subject_inputs(max_subjects=max_subjects, cfg=cfg)
    results = fit_biological_pca_models(inputs, cfg=cfg)
    features = build_subject_biochemical_features(results)
    validate_outputs(inputs, results, features, cfg=cfg)

    staging_parent = out.parent
    staging_parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix=f".{out.name}_staging_", dir=staging_parent))
    try:
        relative_files = _write_outputs(tmp_dir, inputs, results, features)
        _promote_staging(tmp_dir, out, relative_files)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    logger.info("Biological PCA outputs written to %s", out)
    return out
