"""
Per-area preprocessing and fusion pipeline.

This module keeps the paired 1300/2850 area workflow on the same shared
preprocessing implementation used elsewhere in the repo.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from src import _output_dir, config
from src.constants import CANONICAL_REGION_BOUNDS
from src.fusion.area_summary import summarize_area_pixels
from src.fusion.concatenate import concatenate_regions
from src.io.txt_reader import read_txt_spectrum
from src.preprocessing.pipeline import (
    coffee_ring_mask,
    flatten_spectra_rows,
    intersect_region_masks,
    normalize_spectra,
    run_preprocessing,
    sample_baseline_qc_artifacts,
    save_preprocessing_artifacts,
)

logger = logging.getLogger(__name__)

_CANONICAL_REGION_AXES = {
    region: np.arange(start, end + 1.0, 1.0, dtype=float)
    for region, (start, end) in CANONICAL_REGION_BOUNDS.items()
}
_SKIP_LOG_SAMPLE_LIMIT = 3


def _flatten_cube(spectra: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if spectra.ndim == 2:
        n = spectra.shape[0]
        coords = np.column_stack([np.arange(n), np.zeros(n, dtype=int)])
        return flatten_spectra_rows(spectra), coords
    n_x, n_y, n_wn = spectra.shape
    flat = flatten_spectra_rows(spectra)
    xs, ys = np.meshgrid(np.arange(n_x), np.arange(n_y), indexing="ij")
    coords = np.column_stack([xs.ravel(), ys.ravel()])
    return flat, coords


def _canonical_axis_window(
    wavenumbers: np.ndarray,
    region_key: str,
) -> np.ndarray:
    target = _CANONICAL_REGION_AXES.get(region_key)
    if target is None or wavenumbers.size == 0:
        return np.asarray(wavenumbers, dtype=float)
    if not np.all(np.diff(wavenumbers) > 0):
        raise ValueError(f"Region {region_key} wavenumbers must be strictly increasing")

    atol = 1e-6
    start_index = int(np.searchsorted(target, float(wavenumbers[0]) - atol, side="left"))
    end_index = int(np.searchsorted(target, float(wavenumbers[-1]) + atol, side="right") - 1)
    if start_index >= target.size or end_index < 0 or start_index > end_index:
        raise ValueError(
            f"Region {region_key} axis {float(wavenumbers[0]):.1f}..{float(wavenumbers[-1]):.1f} "
            "does not align to the canonical axis window"
        )
    return target[start_index : end_index + 1]


def _align_region_axis(
    spectra: np.ndarray,
    wavenumbers: np.ndarray,
    region_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    target = _CANONICAL_REGION_AXES.get(region_key)
    if target is None or spectra.size == 0 or wavenumbers.size == 0:
        return spectra, wavenumbers
    if wavenumbers.shape == target.shape and np.allclose(wavenumbers, target):
        return spectra, wavenumbers

    target_window = _canonical_axis_window(np.asarray(wavenumbers, dtype=float), region_key)
    missing_points = int(target_window.size - wavenumbers.size)
    if missing_points > 2:
        logger.warning(
            "Region %s axis requires canonical realignment across %d interpolated point(s)",
            region_key,
            missing_points,
        )

    wn_min = float(np.nanmin(wavenumbers))
    wn_max = float(np.nanmax(wavenumbers))
    if float(target[0]) < wn_min or float(target[-1]) > wn_max:
        logger.warning(
            "Region %s axis %.1f..%.1f does not cover canonical range %.1f..%.1f; keeping original axis",
            region_key,
            wn_min,
            wn_max,
            float(target[0]),
            float(target[-1]),
        )
        return spectra, wavenumbers

    aligned = np.vstack([
        np.interp(target, wavenumbers, np.asarray(row, dtype=float))
        for row in np.asarray(spectra, dtype=float)
    ])
    return aligned, target.copy()


def _preprocess_region(
    spectra_flat: np.ndarray,
    wavenumbers: np.ndarray,
    pixel_coords: np.ndarray,
    region_key: str,
    artifact_dir: Path | None = None,
    artifact_stem: str | None = None,
    baseline_qc_dir: Path | None = None,
    condition_label: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    preprocess_result = run_preprocessing(
        spectra_flat,
        wavenumbers,
        region=region_key,
        pixel_coords=pixel_coords,
        screen_dead=True,
        suppress_cosmic=None,
        correct_bl=True,
        smooth=None,
        interpolate=True,
        filter_quality=None,
        normalize=False,
        trim=None,
        trim_cm=None,
        record_stages=None,
    )
    if artifact_dir is not None and artifact_stem is not None:
        save_preprocessing_artifacts(preprocess_result, artifact_dir, artifact_stem)
    if baseline_qc_dir is not None and artifact_stem is not None:
        retained_indices = np.where(preprocess_result.retained_mask)[0]
        condition_labels = (
            {int(idx): condition_label for idx in retained_indices}
            if condition_label
            else None
        )
        sample_baseline_qc_artifacts(
            preprocess_result,
            baseline_qc_dir,
            artifact_stem,
            n_samples=5,
            seed=hash(artifact_stem) % (2**31),
            condition_labels=condition_labels,
        )
    _canonical_axis_window(preprocess_result.wavenumbers, region_key)
    full_spectra = np.full(
        (len(preprocess_result.retained_mask), preprocess_result.spectra.shape[1]),
        np.nan,
        dtype=float,
    )
    full_spectra[preprocess_result.retained_mask] = preprocess_result.spectra
    return full_spectra, preprocess_result.wavenumbers, preprocess_result.retained_mask


def _record_skip(
    reason: str,
    area_id: str,
    skip_counts: dict[str, int] | None,
    skip_examples: dict[str, list[str]] | None,
) -> None:
    if skip_counts is None or skip_examples is None:
        return
    skip_counts[reason] = skip_counts.get(reason, 0) + 1
    examples = skip_examples.setdefault(reason, [])
    if len(examples) < _SKIP_LOG_SAMPLE_LIMIT:
        examples.append(str(area_id))


def _log_empty_intersection_summary(
    subject_id: str,
    skip_counts: dict[str, int],
    skip_examples: dict[str, list[str]],
) -> None:
    count = int(skip_counts.get("empty_intersection", 0))
    if count <= 0:
        return
    examples = skip_examples.get("empty_intersection", [])
    suffix = f" (examples: {', '.join(examples)})" if examples else ""
    logger.info("Subject %s: skipped %d area(s) after mask intersection%s", subject_id, count, suffix)


def process_area(
    filepath_1300: str | Path,
    filepath_2850: str | Path,
    subject_id: str,
    slide_number: str,
    area_id: str,
    enable_coffee_ring: Optional[bool] = None,
    persist_artifacts: Optional[bool] = None,
    skip_counts: dict[str, int] | None = None,
    skip_examples: dict[str, list[str]] | None = None,
    condition_label: Optional[str] = None,
) -> dict | None:
    pre_cfg = config.get("preprocessing", {})
    cr_cfg = pre_cfg.get("quality_filter", {}).get("coffee_ring", {})
    artifact_cfg = pre_cfg.get("artifacts", {})
    if enable_coffee_ring is None:
        enable_coffee_ring = bool(cr_cfg.get("enabled", False))
    keep_artifacts = (
        bool(config.get("output", {}).get("keep_preprocessing_artifacts", True))
        if persist_artifacts is None
        else bool(persist_artifacts)
    )
    keep_baseline_qc = bool(config.get("output", {}).get("keep_preprocessing_artifacts", True))
    artifact_dir = None
    baseline_qc_dir = None
    if keep_artifacts and (
        bool(artifact_cfg.get("save_qc_tables", True)) or bool(artifact_cfg.get("save_stage_arrays", False))
    ):
        artifact_dir = _output_dir("processed_subdir") / "preprocessing_artifacts"
    if keep_baseline_qc:
        baseline_qc_dir = _output_dir("eda_root") / "qc" / "baseline_artifacts"

    try:
        wn_1300, _, _, cube_1300, _ = read_txt_spectrum(filepath_1300)
        wn_2850, _, _, cube_2850, _ = read_txt_spectrum(filepath_2850)
    except (OSError, KeyError, ValueError, RuntimeError) as exc:
        logger.error("Failed to read area files for %s/%s/%s: %s", subject_id, slide_number, area_id, exc)
        return None

    flat_1300, coords_1300 = _flatten_cube(cube_1300)
    flat_2850, coords_2850 = _flatten_cube(cube_2850)
    if flat_1300.shape[0] != flat_2850.shape[0]:
        logger.warning(
            "Pixel count mismatch for area %s/%s/%s: 1300=%d, 2850=%d - skipping",
            subject_id,
            slide_number,
            area_id,
            flat_1300.shape[0],
            flat_2850.shape[0],
        )
        return None

    spectra_1300, wn_1300_out, mask_1300 = _preprocess_region(
        flat_1300,
        wn_1300,
        coords_1300,
        "fingerprint",
        artifact_dir=artifact_dir,
        artifact_stem=f"{subject_id}_{slide_number}_{area_id}_fingerprint",
        baseline_qc_dir=baseline_qc_dir,
        condition_label=condition_label,
    )
    spectra_2850, wn_2850_out, mask_2850 = _preprocess_region(
        flat_2850,
        wn_2850,
        coords_2850,
        "ch_stretch",
        artifact_dir=artifact_dir,
        artifact_stem=f"{subject_id}_{slide_number}_{area_id}_ch_stretch",
        baseline_qc_dir=baseline_qc_dir,
        condition_label=condition_label,
    )
    spectra_1300, wn_1300_out = _align_region_axis(spectra_1300, wn_1300_out, "fingerprint")
    spectra_2850, wn_2850_out = _align_region_axis(spectra_2850, wn_2850_out, "ch_stretch")

    if enable_coffee_ring:
        mask_1300 = coffee_ring_mask(
            coords_1300,
            spectra_1300,
            mask_1300,
            intensity_ratio_threshold=cr_cfg.get("intensity_ratio_threshold"),
            min_surviving_pixels=cr_cfg.get("min_surviving_pixels"),
        )
        mask_2850 = coffee_ring_mask(
            coords_2850,
            spectra_2850,
            mask_2850,
            intensity_ratio_threshold=cr_cfg.get("intensity_ratio_threshold"),
            min_surviving_pixels=cr_cfg.get("min_surviving_pixels"),
        )

    combined_mask = intersect_region_masks(mask_1300, mask_2850)
    n_valid = int(combined_mask.sum())
    if n_valid == 0:
        _record_skip("empty_intersection", area_id, skip_counts, skip_examples)
        if skip_counts is None or skip_examples is None:
            logger.info("No valid pixels after mask intersection for area %s/%s/%s - skipping", subject_id, slide_number, area_id)
        return None

    valid_1300_bc = np.asarray(spectra_1300[combined_mask], dtype=float)
    valid_2850_bc = np.asarray(spectra_2850[combined_mask], dtype=float)
    # Pass region= so each block uses its configured per-region normalization method
    # (fingerprint → SNV, ch_stretch → area) rather than the generic mode="eda" fallback,
    # which applied SNV to both blocks and corrupted CH-stretch intensity scales.
    valid_1300 = normalize_spectra(spectra_1300[combined_mask], region="fingerprint")
    valid_2850 = normalize_spectra(spectra_2850[combined_mask], region="ch_stretch")

    pixel_vectors_bc = concatenate_regions(valid_1300_bc, valid_2850_bc, weight_by_channels=False)
    pixel_vectors = concatenate_regions(
        valid_1300,
        valid_2850,
        weight_by_channels=False,
    )

    summary = summarize_area_pixels(pixel_vectors)
    summary_bc = summarize_area_pixels(pixel_vectors_bc)
    return {
        "subject_id": subject_id,
        "slide_number": slide_number,
        "area_id": area_id,
        "area_mean": summary["area_mean"],
        "area_mean_bc": summary_bc["area_mean"],
        "area_std": summary["area_std"],
        "n_valid_pixels": summary["n_valid_pixels"],
        "wn_1300": wn_1300_out,
        "wn_2850": wn_2850_out,
        "ch_pixels": valid_2850,
    }


def process_subject_areas(
    file_index_subject: "pd.DataFrame",  # noqa: F821
    enable_coffee_ring: Optional[bool] = None,
    persist_artifacts: Optional[bool] = None,
) -> list[dict]:
    results = []
    if file_index_subject.empty:
        return results

    subject_id = str(file_index_subject.iloc[0]["subject_id"])
    skip_counts: dict[str, int] = {}
    skip_examples: dict[str, list[str]] = {}

    for (slide, area), grp in file_index_subject.groupby(["slide_number", "area_id"]):
        range_col = "spectral_range_cm-1"
        if range_col not in grp.columns:
            logger.warning("spectral_range_cm-1 column missing for area %s/%s", slide, area)
            continue

        rows_1300 = grp[grp[range_col].astype(str).str.startswith("1300")]
        rows_2850 = grp[grp[range_col].astype(str).str.startswith("2850")]
        if rows_1300.empty or rows_2850.empty:
            logger.debug(
                "Area %s/%s missing one or both regions - skipping (1300=%d, 2850=%d)",
                slide,
                area,
                len(rows_1300),
                len(rows_2850),
            )
            continue

        condition_label = "control" if bool(rows_1300.iloc[0].get("is_control", False)) else "disease"
        result = process_area(
            rows_1300.iloc[0]["filepath"],
            rows_2850.iloc[0]["filepath"],
            subject_id=subject_id,
            slide_number=str(slide),
            area_id=str(area),
            enable_coffee_ring=enable_coffee_ring,
            persist_artifacts=persist_artifacts,
            skip_counts=skip_counts,
            skip_examples=skip_examples,
            condition_label=condition_label,
        )
        if result is not None:
            results.append(result)

    _log_empty_intersection_summary(subject_id, skip_counts, skip_examples)
    return results
