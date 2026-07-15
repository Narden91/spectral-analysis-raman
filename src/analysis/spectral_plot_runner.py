"""Generate the spectral comparison and QC plots described in plots.md."""

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

from src import config, get_output_root, load_config, whole_spectra_split_index
from src.analysis.spectral_loader import build_group_matrix, build_whole_spectra_group_matrices
from src.constants import CANONICAL_REGION_CHANNELS
from src.data_preparation.loaders import load_clinical_subjects, load_file_index
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
DEFAULT_DATA_OUTPUT_ROOT = Path("output_plot_data")

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
    parser.add_argument("--source", type=str, default=None, help="Override data.source with one of: parsed, preprocessed")
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True, help="Generate plot images (default: True)")
    parser.add_argument("--export-data", action=argparse.BooleanOptionalAction, default=True, help="Export CSV data files (default: True)")
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
        "qc": {"enabled": True, "metrics": QC_METRICS, "band_images": {"enabled": False}},
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
    if explicit:
        outdir = Path(explicit)
        outdir.mkdir(parents=True, exist_ok=True)
        return outdir
    outdir = DEFAULT_PLOT_OUTPUT_ROOT
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def _save_fig(fig: matplotlib.figure.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot: %s", path)


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.6f")
    logger.info("Saved data: %s", path)


def _data_output_dir(plot_base: Path) -> Path:
    """Derive data output root parallel to plot output root."""
    return plot_base.parent / DEFAULT_DATA_OUTPUT_ROOT.name


def _load_index() -> pd.DataFrame:
    df_index = load_file_index()
    clinical = load_clinical_subjects()
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
) -> tuple[np.ndarray, dict[tuple[str, str], np.ndarray], dict[tuple[str, str], list[str]]]:
    group_matrices: dict[tuple[str, str], np.ndarray] = {}
    group_ids: dict[tuple[str, str], list[str]] = {}
    wavenumbers: np.ndarray | None = None

    for cohort, filters in GROUP_FILTERS.items():
        wave, matrix, subject_ids = build_group_matrix(
            df_index,
            gender=filters["gender"],
            is_control=filters["is_control"],
            spectral_range=region,
            max_subjects=max_subjects,
        )
        group_ids[cohort] = subject_ids
        if matrix.size == 0:
            continue
        group_matrices[cohort] = matrix
        if wavenumbers is None:
            wavenumbers = wave

    if wavenumbers is None:
        return np.array([]), group_matrices, group_ids
    return wavenumbers, group_matrices, group_ids


def _pool_condition_groups(
    group_matrices: dict[tuple[str, str], np.ndarray],
) -> dict[str, np.ndarray]:
    return {
        "control": stack_matrices(group_matrices.get(("control", gender)) for gender in ("female", "male")),
        "disease": stack_matrices(group_matrices.get(("disease", gender)) for gender in ("female", "male")),
    }


def _whole_group_filters() -> dict[str, dict[str, object]]:
    return {
        "female_control": {"gender": "female", "is_control": True},
        "male_control": {"gender": "male", "is_control": True},
        "female_disease": {"gender": "female", "is_control": False},
        "male_disease": {"gender": "male", "is_control": False},
    }


def _build_whole_groups(
    df_index: pd.DataFrame,
    max_subjects: int | None,
) -> tuple[np.ndarray, dict[tuple[str, str], np.ndarray], dict[tuple[str, str], list[str]]]:
    wave, matrices_raw, ids_raw = build_whole_spectra_group_matrices(
        df_index,
        _whole_group_filters(),
        max_subjects=max_subjects,
        persist_cache=False,
        persist_artifacts=False,
    )
    matrices = {
        cohort: matrices_raw[key]
        for cohort, key in WHOLE_GROUP_KEYS.items()
        if key in matrices_raw
    }
    group_ids = {
        cohort: ids_raw.get(key, [])
        for cohort, key in WHOLE_GROUP_KEYS.items()
    }
    return wave, matrices, group_ids


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


def _existing_qc_artifact_dir() -> Path | None:
    artifact_dir = get_output_root("eda") / config.get("output", {}).get("processed_subdir", "processed") / "preprocessing_artifacts"
    qc_files = list(artifact_dir.glob("*_qc_metrics.csv"))
    if qc_files:
        return artifact_dir

    logger.info(
        "No preprocessing QC artifacts found under %s; "
        "skipping QC metric distribution plots during plot-only execution",
        artifact_dir,
    )
    return None


def _artifact_index(df_index: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    base = df_index.copy()
    for (subject_id, slide_number, area_id), grp in base.groupby(["subject_id", "slide_number", "area_id"], sort=False):
        values = grp["spectral_range_cm-1"].astype(float).tolist()
        if 1300.0 not in values or 2850.0 not in values:
            continue
        first = grp.iloc[0]
        for region_key, suffix in (("fingerprint", "1300"), ("ch_stretch", "2850")):
            stem = f"{subject_id}_{slide_number}_{area_id}_{region_key}"
            records.append(
                {
                    "stem": stem,
                    "region_code": suffix,
                    "region_key": region_key,
                    "subject_id": str(subject_id),
                    "slide_number": str(slide_number),
                    "area_id": str(area_id),
                    "gender": str(first.get("gender", "unknown")).strip().lower(),
                    "condition": "control" if bool(first.get("is_control", False)) else "disease",
                }
            )
    return pd.DataFrame(records)


def _baseline_fraction_proxy(metrics_df: pd.DataFrame) -> pd.Series:
    fluorescence = metrics_df.get("fluorescence_level")
    mean_intensity = metrics_df.get("mean_intensity")
    if fluorescence is None or mean_intensity is None:
        return pd.Series(np.nan, index=metrics_df.index, dtype=float)
    numerator = pd.to_numeric(fluorescence, errors="coerce").abs()
    denominator = numerator + pd.to_numeric(mean_intensity, errors="coerce").abs()
    with np.errstate(divide="ignore", invalid="ignore"):
        values = numerator / denominator.replace(0, np.nan)
    return values.clip(lower=0.0, upper=1.0)


def _fraction_corrected_proxy(metrics_df: pd.DataFrame, region_key: str) -> pd.Series:
    channels = float(CANONICAL_REGION_CHANNELS.get(region_key, np.nan))
    if "n_cosmic_channels" not in metrics_df.columns or not np.isfinite(channels) or channels <= 0:
        return pd.Series(np.nan, index=metrics_df.index, dtype=float)
    values = pd.to_numeric(metrics_df["n_cosmic_channels"], errors="coerce") / channels
    return values.clip(lower=0.0, upper=1.0)


def _summarize_qc_artifact(meta_row: pd.Series, metrics_path: Path, flags_path: Path) -> dict[str, object]:
    metrics_df = pd.read_csv(metrics_path)
    flags_df = pd.read_csv(flags_path) if flags_path.is_file() else pd.DataFrame()

    if "cosmic_spike_count" not in metrics_df.columns and "n_cosmic_channels" in metrics_df.columns:
        metrics_df["cosmic_spike_count"] = pd.to_numeric(metrics_df["n_cosmic_channels"], errors="coerce")
    if "baseline_fraction" not in metrics_df.columns:
        metrics_df["baseline_fraction"] = _baseline_fraction_proxy(metrics_df)
    if "fraction_corrected" not in metrics_df.columns:
        metrics_df["fraction_corrected"] = _fraction_corrected_proxy(metrics_df, str(meta_row["region_key"]))

    saturation_fraction = np.nan
    if not flags_df.empty and "qc_reason" in flags_df.columns:
        reasons = flags_df["qc_reason"].astype(str).str.lower()
        saturation_fraction = float(reasons.eq("saturated").mean())

    return {
        "subject_id": str(meta_row["subject_id"]),
        "gender": str(meta_row["gender"]),
        "condition": str(meta_row["condition"]),
        "region_code": str(meta_row["region_code"]),
        "snr": float(pd.to_numeric(metrics_df.get("snr"), errors="coerce").median()) if "snr" in metrics_df.columns else np.nan,
        "cosmic_spike_count": float(pd.to_numeric(metrics_df.get("cosmic_spike_count"), errors="coerce").mean()) if "cosmic_spike_count" in metrics_df.columns else np.nan,
        "saturation_fraction": saturation_fraction,
        "baseline_fraction": float(pd.to_numeric(metrics_df.get("baseline_fraction"), errors="coerce").mean()) if "baseline_fraction" in metrics_df.columns else np.nan,
        "fraction_corrected": float(pd.to_numeric(metrics_df.get("fraction_corrected"), errors="coerce").mean()) if "fraction_corrected" in metrics_df.columns else np.nan,
    }


def _load_qc_subject_summary(df_index: pd.DataFrame) -> pd.DataFrame:
    artifact_dir = _existing_qc_artifact_dir()
    if artifact_dir is None or not artifact_dir.exists():
        return pd.DataFrame(columns=["subject_id", "gender", "condition", "region_code", *QC_METRICS])

    mapping = _artifact_index(df_index)
    if mapping.empty:
        return pd.DataFrame(columns=["subject_id", "gender", "condition", "region_code", *QC_METRICS])

    records: list[dict[str, object]] = []
    for meta_row in mapping.itertuples(index=False):
        stem = str(meta_row.stem)
        metrics_path = artifact_dir / f"{stem}_qc_metrics.csv"
        flags_path = artifact_dir / f"{stem}_qc_flags.csv"
        if not metrics_path.is_file():
            continue
        records.append(
            _summarize_qc_artifact(
                pd.Series(meta_row._asdict()),
                metrics_path,
                flags_path,
            )
        )

    area_df = pd.DataFrame(records)
    if area_df.empty:
        return pd.DataFrame(columns=["subject_id", "gender", "condition", "region_code", *QC_METRICS])

    subject_df = (
        area_df.groupby(["subject_id", "gender", "condition", "region_code"], dropna=False)[QC_METRICS]
        .mean()
        .reset_index()
    )
    global_df = (
        area_df.groupby(["subject_id", "gender", "condition"], dropna=False)[QC_METRICS]
        .mean()
        .reset_index()
        .assign(region_code="global")
    )
    return pd.concat([subject_df, global_df], ignore_index=True)


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


def _write_qc_data(
    data_qc_root: Path,
    plot_cfg: dict,
    single_range_data: dict[str, dict[str, object]],
    whole_range_data: dict[str, object],
    qc_subject_df: pd.DataFrame,
) -> None:
    """Export CSV data mirroring _write_qc_plots."""
    if not bool(plot_cfg.get("qc", {}).get("enabled", True)):
        return

    # Cohort mean spectra grid
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
        frames = []
        for region, wave in region_axes.items():
            cols: dict[str, object] = {"wavenumber": wave, "region": region}
            for cohort, segs in cohort_segments.items():
                cond, gen = cohort
                label = f"{cond}_{gen}_mean"
                cols[label] = segs.get(region, np.full(len(wave), np.nan))
            frames.append(pd.DataFrame(cols))
        _save_csv(pd.concat(frames, ignore_index=True), data_qc_root / "cohort_mean_spectra.csv")

    # Per-region mean spectrum
    for region, payload in single_range_data.items():
        pooled = _pool_condition_groups(payload["groups"])
        all_spectra = stack_matrices([pooled.get("control"), pooled.get("disease")])
        if all_spectra.size == 0:
            continue
        wave = np.asarray(payload["wavenumbers"], dtype=float)
        df = pd.DataFrame({
            "wavenumber": wave,
            "mean": np.nanmean(all_spectra, axis=0),
            "std": np.nanstd(all_spectra, axis=0),
            "n": all_spectra.shape[0],
        })
        _save_csv(df, data_qc_root / region / "mean_spectrum.csv")

    # QC metric distributions
    if not qc_subject_df.empty:
        global_qc = qc_subject_df[qc_subject_df["region_code"].eq("global")].copy()
        for metric in plot_cfg.get("qc", {}).get("metrics", QC_METRICS):
            if metric not in global_qc.columns:
                continue
            df = global_qc[["subject_id", "gender", "condition", metric]].copy()
            _save_csv(df, data_qc_root / f"qc_{metric}.csv")

    # Whole-spectra per cohort
    whole_wave = np.asarray(whole_range_data.get("wavenumbers", np.array([])), dtype=float)
    whole_groups = whole_range_data.get("groups", {})
    if whole_wave.size == 0 or not whole_groups:
        return
    split_index = _infer_split_index(whole_wave)
    if split_index is None:
        return

    for cohort in GROUP_FILTERS:
        matrix = whole_groups.get(cohort)
        if matrix is None or np.asarray(matrix).size == 0:
            continue
        region_axes_whole, region_segments = split_whole_spectra(whole_wave, matrix, split_index)
        frames = []
        for reg, wave in region_axes_whole.items():
            mean_vals = np.nanmean(region_segments[reg], axis=0)
            std_vals = np.nanstd(region_segments[reg], axis=0)
            frames.append(pd.DataFrame({
                "wavenumber": wave, "region": reg,
                "mean": mean_vals, "std": std_vals,
                "n": region_segments[reg].shape[0],
            }))
        condition, gender = cohort
        _save_csv(
            pd.concat(frames, ignore_index=True),
            data_qc_root / "whole_spectra" / f"{condition.capitalize()}_{gender}.csv",
        )


def _write_spectral_data(
    data_plot_root: Path,
    plot_cfg: dict,
    single_range_data: dict[str, dict[str, object]],
    whole_range_data: dict[str, object],
) -> None:
    """Export CSV data mirroring _write_spectral_plots."""
    # Condition plots (mean ± std per condition per region)
    if bool(plot_cfg.get("modes", {}).get("condition", {}).get("enabled", True)):
        for region, payload in single_range_data.items():
            pooled = _pool_condition_groups(payload["groups"])
            if pooled["control"].size == 0 or pooled["disease"].size == 0:
                continue
            wave = np.asarray(payload["wavenumbers"], dtype=float)
            df = pd.DataFrame({
                "wavenumber": wave,
                "control_mean": np.nanmean(pooled["control"], axis=0),
                "control_std": np.nanstd(pooled["control"], axis=0),
                "control_n": pooled["control"].shape[0],
                "disease_mean": np.nanmean(pooled["disease"], axis=0),
                "disease_std": np.nanstd(pooled["disease"], axis=0),
                "disease_n": pooled["disease"].shape[0],
            })
            _save_csv(df, data_plot_root / "condition" / f"{region}.csv")

    # Condition by gender
    genders = plot_cfg.get("modes", {}).get("condition_by_gender", {}).get("genders", ["female", "male"])
    if bool(plot_cfg.get("modes", {}).get("condition_by_gender", {}).get("enabled", True)):
        for gender in genders:
            for region, payload in single_range_data.items():
                control = payload["groups"].get(("control", gender))
                disease = payload["groups"].get(("disease", gender))
                if control is None or disease is None or np.asarray(control).size == 0 or np.asarray(disease).size == 0:
                    continue
                wave = np.asarray(payload["wavenumbers"], dtype=float)
                df = pd.DataFrame({
                    "wavenumber": wave,
                    "control_mean": np.nanmean(control, axis=0),
                    "control_std": np.nanstd(control, axis=0),
                    "control_n": control.shape[0],
                    "disease_mean": np.nanmean(disease, axis=0),
                    "disease_std": np.nanstd(disease, axis=0),
                    "disease_n": disease.shape[0],
                })
                _save_csv(df, data_plot_root / "condition_by_gender" / f"{gender}_{region}.csv")

    # Individual spectra
    if bool(plot_cfg.get("modes", {}).get("individual", {}).get("enabled", True)):
        for condition in ("control", "disease"):
            for region, payload in single_range_data.items():
                spectra = _pool_condition_groups(payload["groups"])[condition]
                if spectra.size == 0:
                    continue
                wave = np.asarray(payload["wavenumbers"], dtype=float)
                cols: dict[str, object] = {
                    "wavenumber": wave,
                    "mean": np.nanmean(spectra, axis=0),
                    "std": np.nanstd(spectra, axis=0),
                }
                for i in range(spectra.shape[0]):
                    cols[f"subject_{i + 1}"] = spectra[i]
                _save_csv(pd.DataFrame(cols), data_plot_root / "individual" / f"{condition.capitalize()}_{region}.csv")

        # By-gender individual
        for region, payload in single_range_data.items():
            if not payload["groups"]:
                continue
            wave = np.asarray(payload["wavenumbers"], dtype=float)
            cols_bg: dict[str, object] = {"wavenumber": wave}
            for cohort in GROUP_FILTERS:
                matrix = payload["groups"].get(cohort)
                if matrix is None or np.asarray(matrix).size == 0:
                    continue
                cond, gen = cohort
                prefix = f"{cond}_{gen}"
                cols_bg[f"{prefix}_mean"] = np.nanmean(matrix, axis=0)
                cols_bg[f"{prefix}_std"] = np.nanstd(matrix, axis=0)
                cols_bg[f"{prefix}_n"] = matrix.shape[0]
            _save_csv(pd.DataFrame(cols_bg), data_plot_root / "individual" / f"by_gender_{region}.csv")

    # Global (whole-spectrum) plots
    if not bool(plot_cfg.get("global", True)):
        return

    whole_wave = np.asarray(whole_range_data.get("wavenumbers", np.array([])), dtype=float)
    whole_groups = whole_range_data.get("groups", {})
    if whole_wave.size == 0 or not whole_groups:
        return
    split_index = _infer_split_index(whole_wave)
    if split_index is None:
        return

    whole_pooled = _pool_condition_groups(whole_groups)
    whole_axes, control_segments = split_whole_spectra(whole_wave, whole_pooled["control"], split_index)
    _, disease_segments = split_whole_spectra(whole_wave, whole_pooled["disease"], split_index)

    # Global condition
    if whole_pooled["control"].size and whole_pooled["disease"].size and bool(plot_cfg.get("modes", {}).get("condition", {}).get("enabled", True)):
        frames = []
        for reg, wave in whole_axes.items():
            frames.append(pd.DataFrame({
                "wavenumber": wave, "region": reg,
                "control_mean": np.nanmean(control_segments[reg], axis=0),
                "control_std": np.nanstd(control_segments[reg], axis=0),
                "control_n": whole_pooled["control"].shape[0],
                "disease_mean": np.nanmean(disease_segments[reg], axis=0),
                "disease_std": np.nanstd(disease_segments[reg], axis=0),
                "disease_n": whole_pooled["disease"].shape[0],
            }))
        _save_csv(pd.concat(frames, ignore_index=True), data_plot_root / "condition" / "global.csv")

    # Global condition by gender
    if bool(plot_cfg.get("modes", {}).get("condition_by_gender", {}).get("enabled", True)):
        for gender in genders:
            control = whole_groups.get(("control", gender))
            disease = whole_groups.get(("disease", gender))
            if control is None or disease is None or np.asarray(control).size == 0 or np.asarray(disease).size == 0:
                continue
            _, ctrl_split = split_whole_spectra(whole_wave, control, split_index)
            _, dis_split = split_whole_spectra(whole_wave, disease, split_index)
            frames = []
            for reg, wave in whole_axes.items():
                frames.append(pd.DataFrame({
                    "wavenumber": wave, "region": reg,
                    "control_mean": np.nanmean(ctrl_split[reg], axis=0),
                    "control_std": np.nanstd(ctrl_split[reg], axis=0),
                    "control_n": control.shape[0],
                    "disease_mean": np.nanmean(dis_split[reg], axis=0),
                    "disease_std": np.nanstd(dis_split[reg], axis=0),
                    "disease_n": disease.shape[0],
                }))
            _save_csv(pd.concat(frames, ignore_index=True), data_plot_root / "condition_by_gender" / f"{gender}_global.csv")

    # Global individual
    if bool(plot_cfg.get("modes", {}).get("individual", {}).get("enabled", True)):
        for condition in ("control", "disease"):
            matrix = whole_pooled[condition]
            if matrix.size == 0:
                continue
            _, split_matrix = split_whole_spectra(whole_wave, matrix, split_index)
            frames = []
            for reg, wave in whole_axes.items():
                seg = split_matrix[reg]
                cols_g: dict[str, object] = {
                    "wavenumber": wave, "region": reg,
                    "mean": np.nanmean(seg, axis=0),
                    "std": np.nanstd(seg, axis=0),
                }
                for i in range(seg.shape[0]):
                    cols_g[f"subject_{i + 1}"] = seg[i]
                frames.append(pd.DataFrame(cols_g))
            _save_csv(pd.concat(frames, ignore_index=True), data_plot_root / "individual" / f"{condition.capitalize()}_global.csv")


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
                [*series[:-1], series[-1]],
                title=f"Preprocessed spectra: {condition.capitalize()} individual spectra - Global",
                plot_cfg=plot_cfg,
            )
            _save_fig(fig, plot_root / "individual" / f"{condition.capitalize()}_global.png")


def run_spectral_plots(
    *,
    outdir: str | None = None,
    max_subjects: int | None = None,
    generate_plots: bool = True,
    export_data: bool = True,
) -> dict[str, Path]:
    plot_cfg = _plot_cfg()
    base_out = _base_output_dir(outdir)
    qc_root = base_out / "qc"
    plot_root = base_out / "spectral_plots"
    data_base = _data_output_dir(base_out)
    data_qc_root = data_base / "qc"
    data_plot_root = data_base / "spectral_plots"

    df_index = _load_index()
    requested_regions = [str(region) for region in plot_cfg.get("regions", ["1300", "2850"])]
    single_range_data: dict[str, dict[str, object]] = {}
    for region in requested_regions:
        wave, groups, group_ids = _build_single_range_groups(df_index, region, max_subjects)
        single_range_data[region] = {
            "wavenumbers": wave,
            "groups": groups,
            "ids": group_ids,
        }

    whole_wave = np.array([])
    whole_groups: dict[tuple[str, str], np.ndarray] = {}
    whole_ids: dict[tuple[str, str], list[str]] = {}
    if bool(plot_cfg.get("global", True)) or bool(plot_cfg.get("qc", {}).get("enabled", True)):
        whole_wave, whole_groups, whole_ids = _build_whole_groups(df_index, max_subjects)

    qc_subject_df = _load_qc_subject_summary(df_index) if bool(plot_cfg.get("qc", {}).get("enabled", True)) else pd.DataFrame()
    whole_range_data = {"wavenumbers": whole_wave, "groups": whole_groups, "ids": whole_ids}

    if generate_plots:
        _write_qc_plots(qc_root, plot_cfg, single_range_data, whole_range_data, qc_subject_df)
        _write_spectral_plots(plot_root, plot_cfg, single_range_data, whole_range_data)

    if export_data:
        _write_qc_data(data_qc_root, plot_cfg, single_range_data, whole_range_data, qc_subject_df)
        _write_spectral_data(data_plot_root, plot_cfg, single_range_data, whole_range_data)

    manifest = {
        "base_output_dir": str(base_out),
        "qc_dir": str(qc_root),
        "spectral_plot_dir": str(plot_root),
        "data_output_dir": str(data_base),
        "regions": requested_regions,
        "global": bool(plot_cfg.get("global", True)),
        "plots_generated": generate_plots,
        "data_exported": export_data,
    }
    manifest_path = base_out / "spectral_plot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "base_output_dir": base_out,
        "qc_dir": qc_root,
        "spectral_plot_dir": plot_root,
        "data_output_dir": data_base,
        "manifest": manifest_path,
    }


def _main() -> None:
    args = _parse_args()
    if args.config:
        config.clear()
        config.update(load_config(args.config))
    run_spectral_plots(
        outdir=args.outdir,
        max_subjects=args.max_subjects,
        generate_plots=args.plots,
        export_data=args.export_data,
    )


if __name__ == "__main__":
    _main()
