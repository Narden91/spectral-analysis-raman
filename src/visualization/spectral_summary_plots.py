"""Plotting helpers for spectral comparison and QC summary figures."""

from __future__ import annotations

import math
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator
from scipy.signal import find_peaks

from src.visualization.theme import set_gallery_theme

set_gallery_theme()

CONDITION_COLORS = {
    "control": "#0072B2",
    "disease": "#D55E00",
}

COHORT_COLORS = {
    ("control", "female"): "#0072B2",
    ("control", "male"): "#009E73",
    ("disease", "female"): "#D55E00",
    ("disease", "male"): "#CC79A7",
}

REGION_COLORS = {
    "1300": "#0072B2",
    "2850": "#E69F00",
}

COHORT_ORDER = [
    ("control", "female"),
    ("control", "male"),
    ("disease", "female"),
    ("disease", "male"),
]

REGION_TITLES = {
    "1300": "Fingerprint (1300 cm$^{-1}$)",
    "2850": "CH-stretch (2850 cm$^{-1}$)",
}


def _region_key(region: str) -> str:
    key = str(region).strip().lower()
    if key in {"1300", "fingerprint"}:
        return "1300"
    if key in {"2850", "ch_stretch", "ch"}:
        return "2850"
    return str(region)


def _region_title(region: str) -> str:
    return REGION_TITLES.get(_region_key(region), str(region))


def _cohort_name(condition: str, gender: str) -> str:
    return f"{condition.capitalize()} {gender.capitalize()}"


def _condition_name(condition: str) -> str:
    return str(condition).strip().capitalize()


def _mean_signal(matrix: np.ndarray) -> np.ndarray:
    return np.nanmean(np.asarray(matrix, dtype=float), axis=0)


def _peak_indices(signal: np.ndarray, peak_cfg: dict) -> np.ndarray:
    if not bool(peak_cfg.get("enabled", True)):
        return np.array([], dtype=int)

    y = np.asarray(signal, dtype=float)
    if y.ndim != 1 or y.size == 0:
        return np.array([], dtype=int)

    finite = np.isfinite(y)
    if not np.any(finite):
        return np.array([], dtype=int)

    cleaned = np.where(finite, y, np.nanmedian(y[finite]))
    max_abs = float(np.nanmax(np.abs(cleaned)))
    if max_abs <= 0:
        return np.array([], dtype=int)

    prominence = float(peak_cfg.get("prominence", 0.05)) * max_abs
    distance = max(int(peak_cfg.get("distance", 10)), 1)
    max_peaks = max(int(peak_cfg.get("max_peaks", 12)), 0)
    peaks, props = find_peaks(cleaned, prominence=prominence, distance=distance)
    if peaks.size == 0 or max_peaks == 0:
        return np.array([], dtype=int)

    prominences = props.get("prominences", np.zeros(peaks.size, dtype=float))
    ranked = np.argsort(prominences)[::-1][:max_peaks]
    return np.sort(peaks[ranked])


def _build_layout(
    region_axes: dict[str, np.ndarray],
    gap_cm: float,
) -> tuple[dict[str, np.ndarray], list[tuple[float, float]], dict[str, float]]:
    ordered = [(_region_key(region), np.asarray(axis, dtype=float)) for region, axis in region_axes.items()]
    display_axes: dict[str, np.ndarray] = {}
    gap_spans: list[tuple[float, float]] = []
    label_positions: dict[str, float] = {}
    prev_end: float | None = None

    for idx, (region, axis) in enumerate(ordered):
        if axis.ndim != 1 or axis.size == 0:
            raise ValueError(f"Region '{region}' requires a non-empty 1D axis")
        if idx == 0:
            display = axis.copy()
        else:
            shift = float(prev_end + gap_cm - axis[0])
            display = axis + shift
            gap_spans.append((prev_end, float(display[0])))
        display_axes[region] = display
        label_positions[region] = float((display[0] + display[-1]) / 2.0)
        prev_end = float(display[-1])

    return display_axes, gap_spans, label_positions


def _tick_values(start: float, end: float, step: float) -> list[float]:
    if not np.isfinite(step) or step <= 0:
        return []
    tick = math.ceil(start / step) * step
    values: list[float] = []
    while tick <= end + 1e-9:
        values.append(float(tick))
        tick += step
    return values


def _set_spectrum_limits(
    ax: plt.Axes,
    x_values: Iterable[np.ndarray],
    y_values: Iterable[np.ndarray],
    *,
    top_pad: float = 0.14,
    bottom_pad: float = 0.06,
) -> None:
    x_arrays = [np.asarray(values, dtype=float) for values in x_values if np.asarray(values).size > 0]
    y_arrays = [np.asarray(values, dtype=float) for values in y_values if np.asarray(values).size > 0]
    if not x_arrays or not y_arrays:
        return

    finite_x = [values[np.isfinite(values)] for values in x_arrays if np.any(np.isfinite(values))]
    finite_y = [values[np.isfinite(values)] for values in y_arrays if np.any(np.isfinite(values))]
    if not finite_x or not finite_y:
        return

    x_concat = np.concatenate(finite_x)
    y_concat = np.concatenate(finite_y)
    if x_concat.size == 0 or y_concat.size == 0:
        return

    x_min = float(np.nanmin(x_concat))
    x_max = float(np.nanmax(x_concat))
    y_min = float(np.nanmin(y_concat))
    y_max = float(np.nanmax(y_concat))
    y_span = y_max - y_min
    if not np.isfinite(y_span) or y_span <= 0:
        y_span = max(abs(y_max), 1.0)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min - bottom_pad * y_span, y_max + top_pad * y_span)
    ax.margins(x=0)


def _style_wavenumber_axis(
    ax: plt.Axes,
    region_axes: dict[str, np.ndarray],
    display_axes: dict[str, np.ndarray],
    axis_cfg: dict,
) -> None:
    major_step = float(axis_cfg.get("major_tick", 200.0))
    minor_step = float(axis_cfg.get("minor_tick", 50.0))
    major_positions: list[float] = []
    major_labels: list[str] = []
    minor_positions: list[float] = []

    for region in region_axes:
        axis = np.asarray(region_axes[region], dtype=float)
        display = np.asarray(display_axes[region], dtype=float)
        shift = float(display[0] - axis[0])
        for tick in _tick_values(float(axis.min()), float(axis.max()), major_step):
            major_positions.append(tick + shift)
            major_labels.append(str(int(round(tick))))
        for tick in _tick_values(float(axis.min()), float(axis.max()), minor_step):
            minor_positions.append(tick + shift)

    ax.set_xticks(major_positions)
    ax.set_xticklabels(major_labels)
    ax.xaxis.set_minor_locator(FixedLocator(minor_positions))
    ax.tick_params(axis="x", which="major", length=6)
    ax.tick_params(axis="x", which="minor", length=3)


def _decorate_regions(
    ax: plt.Axes,
    gap_spans: list[tuple[float, float]],
    label_positions: dict[str, float],
) -> None:
    for left, right in gap_spans:
        ax.axvspan(left, right, color="#E5E7EB", alpha=0.6, zorder=0)
        ax.axvline(left, color="#94A3B8", linestyle="--", linewidth=0.8)
        ax.axvline(right, color="#94A3B8", linestyle="--", linewidth=0.8)

    for region, xpos in label_positions.items():
        ax.text(
            xpos,
            1.01,
            _region_title(region),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=10,
        )


def _annotate_peaks(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    actual_wavenumbers: np.ndarray,
    color: str,
    peak_cfg: dict,
) -> None:
    peak_idx = _peak_indices(y, peak_cfg)
    if peak_idx.size == 0:
        return

    y_min = float(np.nanmin(y))
    y_max = float(np.nanmax(y))
    y_span = y_max - y_min
    if not np.isfinite(y_span) or y_span == 0:
        y_span = max(abs(y_max), 1.0)
    y_top = float(ax.get_ylim()[1])
    cap = y_top - 0.03 * y_span

    for order, idx in enumerate(peak_idx):
        offset = (0.02 + 0.012 * (order % 3)) * y_span
        ax.axvline(float(x[idx]), color=color, linewidth=0.7, linestyle=":", alpha=0.2)
        ax.text(
            float(x[idx]),
            float(min(y[idx] + offset, cap)),
            f"{int(round(actual_wavenumbers[idx]))}",
            rotation=90,
            ha="center",
            va="bottom",
            fontsize=8,
            color=color,
            alpha=0.9,
            clip_on=True,
        )


def _finalize_axis(
    ax: plt.Axes,
    title: str,
    ylabel: str = "Intensity (a.u.)",
    *,
    title_pad: float = 16.0,
) -> None:
    ax.set_title(title, fontsize=14, pad=title_pad)
    ax.set_xlabel("Raman Shift (cm$^{-1}$)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.35)


def _place_figure_legend(
    fig: plt.Figure,
    handles: list[object],
    labels: list[str],
    *,
    placement: str = "bottom",
    ncol: int | None = None,
) -> None:
    if not handles or not labels:
        return

    legend_columns = ncol if ncol is not None else max(len(labels), 1)
    y_anchor = -0.01 if placement == "bottom" else 0.985
    location = "lower center" if placement == "bottom" else "upper center"
    fig.legend(
        handles,
        labels,
        loc=location,
        bbox_to_anchor=(0.5, y_anchor),
        ncol=legend_columns,
        frameon=True,
        columnspacing=1.4,
        handletextpad=0.6,
    )


def plot_condition_region(
    wavenumbers: np.ndarray,
    control: np.ndarray,
    disease: np.ndarray,
    region: str,
    plot_cfg: dict,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(12, 5))
    mean_control = _mean_signal(control)
    mean_disease = _mean_signal(disease)
    ax.plot(
        wavenumbers,
        mean_control,
        color=CONDITION_COLORS["control"],
        linewidth=2.0,
        label=f"Control (n={control.shape[0]})",
    )
    ax.plot(
        wavenumbers,
        mean_disease,
        color=CONDITION_COLORS["disease"],
        linewidth=2.0,
        label=f"Disease (n={disease.shape[0]})",
    )
    _set_spectrum_limits(ax, [wavenumbers], [mean_control, mean_disease], top_pad=0.18)
    peak_cfg = plot_cfg.get("peak_detection", {})
    _annotate_peaks(ax, wavenumbers, mean_control, wavenumbers, CONDITION_COLORS["control"], peak_cfg)
    _annotate_peaks(ax, wavenumbers, mean_disease, wavenumbers, CONDITION_COLORS["disease"], peak_cfg)
    _finalize_axis(
        ax,
        title=f"Preprocessed spectra: Disease vs Control - {_region_title(region)}",
    )
    handles, labels = ax.get_legend_handles_labels()
    _place_figure_legend(fig, handles, labels)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig, ax


def plot_gender_condition_region(
    wavenumbers: np.ndarray,
    control: np.ndarray,
    disease: np.ndarray,
    gender: str,
    region: str,
    plot_cfg: dict,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(12, 5))
    mean_control = _mean_signal(control)
    mean_disease = _mean_signal(disease)
    control_color = COHORT_COLORS[("control", gender)]
    disease_color = COHORT_COLORS[("disease", gender)]
    ax.plot(
        wavenumbers,
        mean_control,
        color=control_color,
        linewidth=2.0,
        label=f"Control (n={control.shape[0]})",
    )
    ax.plot(
        wavenumbers,
        mean_disease,
        color=disease_color,
        linewidth=2.0,
        label=f"Disease (n={disease.shape[0]})",
    )
    _set_spectrum_limits(ax, [wavenumbers], [mean_control, mean_disease], top_pad=0.18)
    peak_cfg = plot_cfg.get("peak_detection", {})
    _annotate_peaks(ax, wavenumbers, mean_control, wavenumbers, control_color, peak_cfg)
    _annotate_peaks(ax, wavenumbers, mean_disease, wavenumbers, disease_color, peak_cfg)
    _finalize_axis(
        ax,
        title=f"Preprocessed spectra: {gender.capitalize()} Control vs Disease - {_region_title(region)}",
    )
    handles, labels = ax.get_legend_handles_labels()
    _place_figure_legend(fig, handles, labels)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig, ax


def plot_individual_region(
    wavenumbers: np.ndarray,
    spectra: np.ndarray,
    condition: str,
    region: str,
    plot_cfg: dict,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(12, 5))
    color = CONDITION_COLORS[condition]
    line_alpha = float(plot_cfg.get("modes", {}).get("individual", {}).get("line_alpha", 0.25))
    mean_linewidth = float(plot_cfg.get("modes", {}).get("individual", {}).get("mean_linewidth", 2.5))
    for row in np.asarray(spectra, dtype=float):
        ax.plot(wavenumbers, row, color=color, linewidth=0.6, alpha=line_alpha)
    mean_spectrum = _mean_signal(spectra)
    ax.plot(
        wavenumbers,
        mean_spectrum,
        color=color,
        linewidth=mean_linewidth,
        label=f"{_condition_name(condition)} mean (n={spectra.shape[0]})",
    )
    _set_spectrum_limits(
        ax,
        [wavenumbers],
        [np.asarray(spectra, dtype=float), mean_spectrum],
        top_pad=0.18,
    )
    _annotate_peaks(ax, wavenumbers, mean_spectrum, wavenumbers, color, plot_cfg.get("peak_detection", {}))
    _finalize_axis(
        ax,
        title=f"Preprocessed spectra: {_condition_name(condition)} individual spectra - {_region_title(region)} (n={spectra.shape[0]})",
    )
    handles, labels = ax.get_legend_handles_labels()
    _place_figure_legend(fig, handles, labels)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig, ax


def plot_global_comparison(
    region_axes: dict[str, np.ndarray],
    series: list[dict[str, object]],
    title: str,
    plot_cfg: dict,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(18, 5.5))
    display_axes, gap_spans, label_positions = _build_layout(
        region_axes,
        gap_cm=float(plot_cfg.get("gap_cm", 50.0)),
    )
    peak_cfg = plot_cfg.get("peak_detection", {})
    annotated_segments: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]] = []

    for entry in series:
        label = entry.get("label")
        color = str(entry["color"])
        linewidth = float(entry.get("linewidth", 2.0))
        alpha = float(entry.get("alpha", 1.0))
        linestyle = str(entry.get("linestyle", "-"))
        annotate = bool(entry.get("annotate", True))
        segments = entry["segments"]
        first_region = next(iter(segments.keys()))
        for region, y in segments.items():
            ax.plot(
                display_axes[region],
                y,
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                linestyle=linestyle,
                label=str(label) if (label is not None and region == first_region) else None,
            )
            if annotate:
                annotated_segments.append(
                    (
                        np.asarray(display_axes[region], dtype=float),
                        np.asarray(y, dtype=float),
                        np.asarray(region_axes[region], dtype=float),
                        color,
                    )
                )

    _set_spectrum_limits(
        ax,
        display_axes.values(),
        [np.asarray(entry["segments"][region], dtype=float) for entry in series for region in entry["segments"]],
        top_pad=0.22,
    )
    for x_values, y_values, actual_wavenumbers, color in annotated_segments:
        _annotate_peaks(ax, x_values, y_values, actual_wavenumbers, color, peak_cfg)
    _style_wavenumber_axis(ax, region_axes, display_axes, plot_cfg.get("axis", {}))
    _decorate_regions(ax, gap_spans, label_positions)
    _finalize_axis(ax, title=title, title_pad=24.0)
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        _place_figure_legend(fig, handles, labels)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig, ax


def plot_cohort_mean_spectra_grid(
    region_axes: dict[str, np.ndarray],
    cohort_segments: dict[tuple[str, str], dict[str, np.ndarray]],
    subject_counts: dict[tuple[str, str], int],
    plot_cfg: dict,
) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharey=False)
    display_axes, gap_spans, label_positions = _build_layout(
        region_axes,
        gap_cm=float(plot_cfg.get("gap_cm", 50.0)),
    )
    legend_handles: list[object] = []
    legend_labels: list[str] = []

    for ax, cohort in zip(axes.flat, COHORT_ORDER):
        condition, gender = cohort
        segments = cohort_segments.get(cohort, {})
        annotated_segments: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]] = []
        for region in region_axes:
            y = segments.get(region)
            if y is None:
                continue
            color = REGION_COLORS.get(region, "#475569")
            (line,) = ax.plot(
                display_axes[region],
                y,
                color=color,
                linewidth=1.8,
                label=_region_title(region),
            )
            if _region_title(region) not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(_region_title(region))
            annotated_segments.append(
                (
                    np.asarray(display_axes[region], dtype=float),
                    np.asarray(y, dtype=float),
                    np.asarray(region_axes[region], dtype=float),
                    color,
                )
            )
        _set_spectrum_limits(
            ax,
            display_axes.values(),
            [np.asarray(values, dtype=float) for values in segments.values()],
            top_pad=0.22,
        )
        for x_values, y_values, actual_wavenumbers, color in annotated_segments:
            _annotate_peaks(ax, x_values, y_values, actual_wavenumbers, color, plot_cfg.get("peak_detection", {}))
        _style_wavenumber_axis(ax, region_axes, display_axes, plot_cfg.get("axis", {}))
        _decorate_regions(ax, gap_spans, label_positions)
        _finalize_axis(
            ax,
            title=f"{_cohort_name(condition, gender)} (n={subject_counts.get(cohort, 0)})",
            title_pad=18.0,
        )

    fig.suptitle("Cohort mean spectra", fontsize=16, y=0.995)
    _place_figure_legend(fig, legend_handles, legend_labels, placement="bottom")
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    return fig, axes


def plot_region_mean_spectrum(
    wavenumbers: np.ndarray,
    spectra: np.ndarray,
    region: str,
    plot_cfg: dict,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(12, 5))
    mean_spectrum = _mean_signal(spectra)
    color = REGION_COLORS.get(_region_key(region), "#334155")
    ax.plot(wavenumbers, mean_spectrum, color=color, linewidth=2.0)
    _set_spectrum_limits(ax, [wavenumbers], [mean_spectrum], top_pad=0.18)
    _annotate_peaks(ax, wavenumbers, mean_spectrum, wavenumbers, color, plot_cfg.get("peak_detection", {}))
    _finalize_axis(ax, title=f"Mean spectrum - {_region_title(region)}")
    fig.tight_layout()
    return fig, ax


def plot_qc_metric_distribution(
    qc_df: pd.DataFrame,
    metric: str,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(10, 6))
    data = qc_df.copy()
    metric_label = metric.replace("_", " ").title()
    if {"condition", "gender"}.issubset(data.columns):
        positions = np.arange(1, len(COHORT_ORDER) + 1, dtype=float)
        box_data: list[np.ndarray] = []
        labels: list[str] = []
        colors: list[str] = []
        for idx, cohort in enumerate(COHORT_ORDER):
            condition, gender = cohort
            subset = data[
                data["condition"].astype(str).str.lower().eq(condition)
                & data["gender"].astype(str).str.lower().eq(gender)
            ][metric].dropna().to_numpy(dtype=float)
            box_data.append(subset)
            labels.append(f"{condition.capitalize()}\n{gender.capitalize()}\n n={subset.size}")
            colors.append(COHORT_COLORS[cohort])

        boxplot = ax.boxplot(
            box_data,
            positions=positions,
            patch_artist=True,
            widths=0.55,
            showfliers=False,
        )
        for patch, color in zip(boxplot["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
            patch.set_edgecolor(color)
        for median in boxplot["medians"]:
            median.set_color("#111827")
            median.set_linewidth(1.4)

        rng = np.random.default_rng(42)
        for pos, values, color in zip(positions, box_data, colors):
            if values.size == 0:
                continue
            jitter = rng.uniform(-0.12, 0.12, size=values.size)
            ax.scatter(
                np.full(values.size, pos) + jitter,
                values,
                s=20,
                alpha=0.55,
                color=color,
                edgecolors="white",
                linewidth=0.3,
            )
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
    else:
        series = data[metric].dropna().to_numpy(dtype=float)
        ax.hist(series, bins=32, color="#4C78A8", alpha=0.75, edgecolor="white")
        if series.size:
            ax.axvline(float(np.nanmedian(series)), color="#CC3311", linestyle="--", linewidth=1.5)

    ax.set_title(f"QC metric distribution: {metric_label}", fontsize=14)
    ax.set_xlabel(metric_label, fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    return fig, ax


def plot_whole_spectra_per_cohort(
    region_axes: dict[str, np.ndarray],
    segments: dict[str, np.ndarray],
    cohort: tuple[str, str],
    subject_count: int,
    plot_cfg: dict,
) -> tuple[plt.Figure, plt.Axes]:
    condition, gender = cohort
    color = COHORT_COLORS[cohort]
    return plot_global_comparison(
        region_axes,
        [
            {
                "label": f"{_cohort_name(condition, gender)} (n={subject_count})",
                "color": color,
                "linewidth": 2.3,
                "segments": segments,
            }
        ],
        title=f"{_cohort_name(condition, gender)} whole spectra (n={subject_count})",
        plot_cfg=plot_cfg,
    )


def split_whole_spectra(
    wavenumbers: np.ndarray,
    matrix: np.ndarray,
    split_index: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    return (
        {
            "1300": np.asarray(wavenumbers[:split_index], dtype=float),
            "2850": np.asarray(wavenumbers[split_index:], dtype=float),
        },
        {
            "1300": np.asarray(matrix[:, :split_index], dtype=float),
            "2850": np.asarray(matrix[:, split_index:], dtype=float),
        },
    )


def stack_matrices(matrices: Iterable[np.ndarray]) -> np.ndarray:
    valid = [np.asarray(matrix, dtype=float) for matrix in matrices if matrix is not None and np.asarray(matrix).size > 0]
    if not valid:
        return np.empty((0, 0), dtype=float)
    return np.vstack(valid)
