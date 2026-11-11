"""Generate the spectral comparison and QC plots."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
import matplotlib.figure
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config, whole_spectra_split_index
from src.analysis.spectral_loader import build_group_matrix, build_whole_spectra_group_matrices
from src.data_preparation.loaders import load_clinical_subjects, load_file_index
from src.data_preparation.store import DataSource, get_active_data_source, resolve_data_source
from src.metadata.clinical import merge_index_with_clinical
from src.visualization.spectral_summary_plots import (
    COHORT_COLORS,
    CONDITION_COLORS,
    plot_cohort_mean_spectra_grid,
    plot_condition_region,
    plot_gender_condition_region,
    plot_global_comparison,
    plot_individual_by_gender_region,
    plot_individual_region,
    plot_qc_metric_distribution,
    plot_region_mean_spectrum,
    plot_whole_spectra_per_cohort,
    split_whole_spectra,
    stack_matrices,
)

logger = logging.getLogger(__name__)
DEFAULT_PLOT_OUTPUT_ROOT = Path("Output_plot")

GROUP_FILTERS = {
    ("control", "female"): {"gender": "female", "is_control": True},
    ("control", "male"): {"gender": "male", "is_control": True},
    ("disease", "female"): {"gender": "female", "is_control": False},
    ("disease", "male"): {"gender": "male", "is_control": False},
}

WHOLE_GROUP_KEYS = {
    ("control", "female"): "female_control",
    ("control", "male"): "male_control",
    ("disease", "female"): "female_disease",
    ("disease", "male"): "male_disease",
}

QC_METRICS = [
    "snr",
    "cosmic_spike_count",
    "saturation_fraction",
    "baseline_fraction",
    "fraction_corrected",
]

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Raman QC and spectral comparison plots")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML (overrides config/plots.yaml)")
    parser.add_argument("--max-subjects", type=int, default=None, help="Limit subjects per cohort")
    parser.add_argument("--outdir", type=str, default=None, help="Override base output directory")
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True, help="Generate plot images (default: True)")
    return parser.parse_args()


def _plot_cfg() -> dict:
    defaults = {
        "regions": ["1300", "2850"],
        "global": True,
        "gap_cm": 50.0,
        "axis": {"major_tick": 200.0, "minor_tick": 50.0},
        "peak_detection": {"enabled": True, "prominence": 0.05, "distance": 10, "max_peaks": 12},
        "modes": {
            "condition": {"enabled": True},
            "condition_by_gender": {"enabled": True, "genders": ["female", "male"]},
            "individual": {"enabled": True, "line_alpha": 0.25, "mean_linewidth": 2.5},
        },
        "qc": {"enabled": True, "metrics": QC_METRICS},
    }
    cfg = dict(defaults)
    cfg.update(config.get("plotting", {}))
    cfg["axis"] = {**defaults["axis"], **cfg.get("axis", {})}
    cfg["peak_detection"] = {**defaults["peak_detection"], **cfg.get("peak_detection", {})}
    cfg["modes"] = {
        "condition": {**defaults["modes"]["condition"], **cfg.get("modes", {}).get("condition", {})},
        "condition_by_gender": {
            **defaults["modes"]["condition_by_gender"],
            **cfg.get("modes", {}).get("condition_by_gender", {}),
        },
        "individual": {**defaults["modes"]["individual"], **cfg.get("modes", {}).get("individual", {})},
    }
    cfg["qc"] = {**defaults["qc"], **cfg.get("qc", {})}
    cfg["bands"] = config.get("whole_spectra", {}).get("bands", {})
    return cfg


def _base_output_dir(explicit: str | None) -> Path:
    outdir = Path(explicit) if explicit else DEFAULT_PLOT_OUTPUT_ROOT
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def _save_fig(fig: matplotlib.figure.Figure, path: Path) -> None:
    """Write a figure as vector PDF, whatever suffix the caller supplied.

    Call sites name their targets with a ``.png`` suffix; the suffix is
    rewritten here so every spectral and QC plot is emitted as a single
    publication-ready vector file instead of a raster one.
    """
    path = path.with_suffix(".pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot: %s", path)


def _load_index(data_source: DataSource | None = None) -> pd.DataFrame:
    df_index = load_file_index(data_source)
    clinical = load_clinical_subjects(data_source)
    merged = merge_index_with_clinical(df_index, clinical)
    if "gender" in merged.columns:
        merged["gender"] = merged["gender"].astype(str).str.strip().str.lower()
    if "condition" not in merged.columns and "is_control" in merged.columns:
        merged["condition"] = merged["is_control"].map({True: "control", False: "disease"})
    return merged


def _build_single_range_groups(
    df_index: pd.DataFrame,
    region: str,
    max_subjects: int | None,
    data_source: DataSource | None = None,
) -> tuple[np.ndarray, dict[tuple[str, str], np.ndarray]]:
    group_matrices: dict[tuple[str, str], np.ndarray] = {}
    wavenumbers: np.ndarray | None = None

    for cohort, filters in GROUP_FILTERS.items():
        wave, matrix, _ = build_group_matrix(
            df_index,
            gender=filters["gender"],
            is_control=filters["is_control"],
            spectral_range=region,
            max_subjects=max_subjects,
            data_source=data_source,
        )
        if matrix.size == 0:
            continue
        group_matrices[cohort] = matrix
        if wavenumbers is None:
            wavenumbers = wave

    if wavenumbers is None:
        return np.array([]), group_matrices
    return wavenumbers, group_matrices


def _pool_condition_groups(
    group_matrices: dict[tuple[str, str], np.ndarray],
) -> dict[str, np.ndarray]:
    return {
        "control": stack_matrices(group_matrices.get(("control", gender)) for gender in ("female", "male")),
        "disease": stack_matrices(group_matrices.get(("disease", gender)) for gender in ("female", "male")),
    }


def _whole_group_filters() -> dict[str, dict[str, object]]:
    return {WHOLE_GROUP_KEYS[cohort]: dict(filters) for cohort, filters in GROUP_FILTERS.items()}


def _build_whole_groups(
    df_index: pd.DataFrame,
    max_subjects: int | None,
    data_source: DataSource | None = None,
) -> tuple[np.ndarray, dict[tuple[str, str], np.ndarray]]:
    wave, matrices_raw, _ = build_whole_spectra_group_matrices(
        df_index,
        _whole_group_filters(),
        max_subjects=max_subjects,
        data_source=data_source,
        persist_cache=False,
        persist_artifacts=False,
    )
    matrices = {
        cohort: matrices_raw[key]
        for cohort, key in WHOLE_GROUP_KEYS.items()
        if key in matrices_raw
    }
    return wave, matrices


def _infer_split_index(wavenumbers: np.ndarray) -> int | None:
    split_index = whole_spectra_split_index(wavenumbers)
    if split_index is not None:
        return split_index
    if np.asarray(wavenumbers).ndim != 1 or np.asarray(wavenumbers).size < 2:
        return None
    deltas = np.diff(np.asarray(wavenumbers, dtype=float))
    if deltas.size == 0:
        return None
    return int(np.argmax(deltas) + 1)


def _load_qc_subject_summary(df_index: pd.DataFrame, data_source: DataSource | None = None) -> pd.DataFrame:
    source = data_source or get_active_data_source() or resolve_data_source()
    if source.source_name == "preprocessed":
        qc = pd.read_parquet(source.source_path / "qc.parquet")
        required = {"subject_id", "spectral_range", *QC_METRICS}
        missing = required.difference(qc.columns)
        if missing:
            raise RuntimeError(f"Canonical QC table is missing columns: {', '.join(sorted(missing))}")
        subject_meta = (
            df_index.groupby("subject_id", sort=False)
            .agg(gender=("gender", "first"), is_control=("is_control", "first"))
            .reset_index()
        )
        metric_columns = [*QC_METRICS]
        area_df = (
            qc.groupby(["subject_id", "spectral_range"], dropna=False)[metric_columns]
            .mean()
            .reset_index()
            .merge(subject_meta, on="subject_id", how="left")
        )
        area_df["gender"] = area_df["gender"].fillna("unknown").astype(str).str.lower()
        area_df["condition"] = np.where(area_df["is_control"].fillna(False).astype(bool), "control", "disease")
        area_df = area_df.rename(columns={"spectral_range": "region_code"})
        subject_df = area_df[["subject_id", "gender", "condition", "region_code", *QC_METRICS]]
        global_df = (
            subject_df.groupby(["subject_id", "gender", "condition"], dropna=False)[QC_METRICS]
            .mean()
            .reset_index()
            .assign(region_code="global")
        )
        return pd.concat([subject_df, global_df], ignore_index=True)

    # Only the preprocessed store has a canonical QC table; parsed data has none.
    raise RuntimeError(
        f"QC plots require data.source 'preprocessed'; got '{source.source_name}'. "
        "Run `uv run scripts/manage_data_store.py prepare` and set data.source: preprocessed."
    )


def _write_qc_plots(
    qc_root: Path,
    plot_cfg: dict,
    single_range_data: dict[str, dict[str, object]],
    whole_range_data: dict[str, object],
    qc_subject_df: pd.DataFrame,
) -> None:
    if not bool(plot_cfg.get("qc", {}).get("enabled", True)):
        return

    region_axes = {
        region: np.asarray(payload["wavenumbers"], dtype=float)
        for region, payload in single_range_data.items()
        if np.asarray(payload["wavenumbers"]).size > 0
    }
    cohort_segments: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for cohort in GROUP_FILTERS:
        segments: dict[str, np.ndarray] = {}
        for region, payload in single_range_data.items():
            matrix = payload["groups"].get(cohort)
            if matrix is None or np.asarray(matrix).size == 0:
                continue
            segments[region] = np.nanmean(np.asarray(matrix, dtype=float), axis=0)
        if segments:
            cohort_segments[cohort] = segments

    if cohort_segments and len(region_axes) >= 1:
        fig, _ = plot_cohort_mean_spectra_grid(region_axes, cohort_segments, plot_cfg)
        _save_fig(fig, qc_root / "cohort_mean_spectra.png")

    for region, payload in single_range_data.items():
        pooled = _pool_condition_groups(payload["groups"])
        all_spectra = stack_matrices([pooled.get("control"), pooled.get("disease")])
        if all_spectra.size == 0:
            continue
        fig, _ = plot_region_mean_spectrum(payload["wavenumbers"], all_spectra, region, plot_cfg)
        _save_fig(fig, qc_root / region / "mean_spectrum.png")

    if not qc_subject_df.empty:
        global_qc = qc_subject_df[qc_subject_df["region_code"].eq("global")].copy()
        for metric in plot_cfg.get("qc", {}).get("metrics", QC_METRICS):
            if metric not in global_qc.columns:
                continue
            fig, _ = plot_qc_metric_distribution(global_qc, metric)
            _save_fig(fig, qc_root / f"qc_{metric}.png")

    whole_wave = np.asarray(whole_range_data.get("wavenumbers", np.array([])), dtype=float)
    whole_groups = whole_range_data.get("groups", {})
    if whole_wave.size == 0 or not whole_groups:
        return
    split_index = _infer_split_index(whole_wave)
    if split_index is None:
        logger.warning("Could not infer whole-spectrum split index; skipping whole-spectrum QC plots")
        return

    for cohort in GROUP_FILTERS:
        matrix = whole_groups.get(cohort)
        if matrix is None or np.asarray(matrix).size == 0:
            continue
        region_axes_whole, region_segments = split_whole_spectra(whole_wave, matrix, split_index)
        mean_segments = {
            region: np.nanmean(segment_matrix, axis=0)
            for region, segment_matrix in region_segments.items()
        }
        fig, _ = plot_whole_spectra_per_cohort(
            region_axes_whole,
            mean_segments,
            cohort,
            plot_cfg=plot_cfg,
        )
        condition, gender = cohort
        _save_fig(fig, qc_root / "whole_spectra" / f"{condition.capitalize()}_{gender}.png")


def _write_spectral_plots(
    plot_root: Path,
    plot_cfg: dict,
    single_range_data: dict[str, dict[str, object]],
    whole_range_data: dict[str, object],
) -> None:
    if bool(plot_cfg.get("modes", {}).get("condition", {}).get("enabled", True)):
        for region, payload in single_range_data.items():
            pooled = _pool_condition_groups(payload["groups"])
            if pooled["control"].size == 0 or pooled["disease"].size == 0:
                continue
            fig, _ = plot_condition_region(
                payload["wavenumbers"],
                pooled["control"],
                pooled["disease"],
                region,
                plot_cfg,
            )
            _save_fig(fig, plot_root / "condition" / f"{region}.png")

    genders = plot_cfg.get("modes", {}).get("condition_by_gender", {}).get("genders", ["female", "male"])
    if bool(plot_cfg.get("modes", {}).get("condition_by_gender", {}).get("enabled", True)):
        for gender in genders:
            for region, payload in single_range_data.items():
                control = payload["groups"].get(("control", gender))
                disease = payload["groups"].get(("disease", gender))
                if control is None or disease is None or np.asarray(control).size == 0 or np.asarray(disease).size == 0:
                    continue
                fig, _ = plot_gender_condition_region(
                    payload["wavenumbers"],
                    control,
                    disease,
                    gender,
                    region,
                    plot_cfg,
                )
                _save_fig(fig, plot_root / "condition_by_gender" / f"{gender}_{region}.png")

    if bool(plot_cfg.get("modes", {}).get("individual", {}).get("enabled", True)):
        for condition in ("control", "disease"):
            for region, payload in single_range_data.items():
                spectra = _pool_condition_groups(payload["groups"])[condition]
                if spectra.size == 0:
                    continue
                fig, _ = plot_individual_region(
                    payload["wavenumbers"],
                    spectra,
                    condition,
                    region,
                    plot_cfg,
                )
                _save_fig(fig, plot_root / "individual" / f"{condition.capitalize()}_{region}.png")

        for region, payload in single_range_data.items():
            if not payload["groups"]:
                continue
            fig, _ = plot_individual_by_gender_region(
                payload["wavenumbers"],
                payload["groups"],
                region,
                plot_cfg,
            )
            _save_fig(fig, plot_root / "individual" / f"by_gender_{region}.png")

    if not bool(plot_cfg.get("global", True)):
        return

    whole_wave = np.asarray(whole_range_data.get("wavenumbers", np.array([])), dtype=float)
    whole_groups = whole_range_data.get("groups", {})
    if whole_wave.size == 0 or not whole_groups:
        return

    split_index = _infer_split_index(whole_wave)
    if split_index is None:
        logger.warning("Could not infer whole-spectrum split index; skipping global spectral plots")
        return

    whole_pooled = _pool_condition_groups(whole_groups)
    whole_axes, control_segments = split_whole_spectra(whole_wave, whole_pooled["control"], split_index)
    _, disease_segments = split_whole_spectra(whole_wave, whole_pooled["disease"], split_index)

    if whole_pooled["control"].size and whole_pooled["disease"].size and bool(plot_cfg.get("modes", {}).get("condition", {}).get("enabled", True)):
        fig, _ = plot_global_comparison(
            whole_axes,
            [
                {
                    "label": "Control",
                    "color": CONDITION_COLORS["control"],
                    "segments": {region: np.nanmean(values, axis=0) for region, values in control_segments.items()},
                },
                {
                    "label": "Disease",
                    "color": CONDITION_COLORS["disease"],
                    "segments": {region: np.nanmean(values, axis=0) for region, values in disease_segments.items()},
                },
            ],
            title="Preprocessed spectra: Disease vs Control - Global",
            plot_cfg=plot_cfg,
        )
        _save_fig(fig, plot_root / "condition" / "global.png")

    if bool(plot_cfg.get("modes", {}).get("condition_by_gender", {}).get("enabled", True)):
        for gender in genders:
            control = whole_groups.get(("control", gender))
            disease = whole_groups.get(("disease", gender))
            if control is None or disease is None or np.asarray(control).size == 0 or np.asarray(disease).size == 0:
                continue
            _, control_split = split_whole_spectra(whole_wave, control, split_index)
            _, disease_split = split_whole_spectra(whole_wave, disease, split_index)
            fig, _ = plot_global_comparison(
                whole_axes,
                [
                    {
                        "label": "Control",
                        "color": COHORT_COLORS[("control", gender)],
                        "segments": {region: np.nanmean(values, axis=0) for region, values in control_split.items()},
                    },
                    {
                        "label": "Disease",
                        "color": COHORT_COLORS[("disease", gender)],
                        "segments": {region: np.nanmean(values, axis=0) for region, values in disease_split.items()},
                    },
                ],
                title=f"Preprocessed spectra: {gender.capitalize()} Control vs Disease - Global",
                plot_cfg=plot_cfg,
            )
            _save_fig(fig, plot_root / "condition_by_gender" / f"{gender}_global.png")

    if bool(plot_cfg.get("modes", {}).get("individual", {}).get("enabled", True)):
        line_alpha = float(plot_cfg.get("modes", {}).get("individual", {}).get("line_alpha", 0.25))
        mean_linewidth = float(plot_cfg.get("modes", {}).get("individual", {}).get("mean_linewidth", 2.5))
        for condition in ("control", "disease"):
            matrix = whole_pooled[condition]
            if matrix.size == 0:
                continue
            _, split_matrix = split_whole_spectra(whole_wave, matrix, split_index)
            color = CONDITION_COLORS[condition]
            series = []
            for row_idx in range(matrix.shape[0]):
                series.append(
                    {
                        "label": None,
                        "color": color,
                        "linewidth": 0.6,
                        "alpha": line_alpha,
                        "annotate": False,
                        "shade": False,
                        "segments": {region: values[row_idx] for region, values in split_matrix.items()},
                    }
                )
            series.append(
                {
                    "label": f"{condition.capitalize()} mean",
                    "color": color,
                    "linewidth": mean_linewidth,
                    "segments": {region: np.nanmean(values, axis=0) for region, values in split_matrix.items()},
                    "annotate": True,
                }
            )
            fig, _ = plot_global_comparison(
                whole_axes,
                series,
                title=f"Preprocessed spectra: {condition.capitalize()} individual spectra - Global",
                plot_cfg=plot_cfg,
                show_band_spans=False,
            )
            _save_fig(fig, plot_root / "individual" / f"{condition.capitalize()}_global.png")


def run_spectral_plots(
    *,
    outdir: str | None = None,
    max_subjects: int | None = None,
    generate_plots: bool = True,
    data_source: DataSource | None = None,
) -> dict[str, Path]:
    plot_cfg = _plot_cfg()
    base_out = _base_output_dir(outdir)
    qc_root = base_out / "qc"
    plot_root = base_out / "spectral_plots"

    df_index = _load_index(data_source)
    requested_regions = [str(region) for region in plot_cfg.get("regions", ["1300", "2850"])]
    single_range_data: dict[str, dict[str, object]] = {}
    for region in requested_regions:
        wave, groups = _build_single_range_groups(df_index, region, max_subjects, data_source)
        single_range_data[region] = {"wavenumbers": wave, "groups": groups}

    whole_wave = np.array([])
    whole_groups: dict[tuple[str, str], np.ndarray] = {}
    if bool(plot_cfg.get("global", True)) or bool(plot_cfg.get("qc", {}).get("enabled", True)):
        whole_wave, whole_groups = _build_whole_groups(df_index, max_subjects, data_source)

    qc_subject_df = (
        _load_qc_subject_summary(df_index, data_source)
        if bool(plot_cfg.get("qc", {}).get("enabled", True))
        else pd.DataFrame()
    )
    whole_range_data = {"wavenumbers": whole_wave, "groups": whole_groups}

    if generate_plots:
        _write_qc_plots(qc_root, plot_cfg, single_range_data, whole_range_data, qc_subject_df)
        _write_spectral_plots(plot_root, plot_cfg, single_range_data, whole_range_data)

    manifest = {
        "base_output_dir": str(base_out),
        "qc_dir": str(qc_root),
        "spectral_plot_dir": str(plot_root),
        "regions": requested_regions,
        "global": bool(plot_cfg.get("global", True)),
        "plots_generated": generate_plots,
    }
    manifest_path = base_out / "spectral_plot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "base_output_dir": base_out,
        "qc_dir": qc_root,
        "spectral_plot_dir": plot_root,
        "manifest": manifest_path,
    }
