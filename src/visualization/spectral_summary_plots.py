"""Plotting helpers for spectral comparison and QC summary figures."""

from __future__ import annotations

import math
from collections.abc import Container, Mapping
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator
from scipy.signal import find_peaks

from src.analysis.raman_bands import RAMAN_BANDS, validate_configured_bands
from src.visualization.theme import GROUP_COLORS, set_gallery_theme

set_gallery_theme()

CONDITION_COLORS = {
    "control": "#0072B2",
    "disease": "#D55E00",
}

COHORT_COLORS = {
    ("control", "female"): GROUP_COLORS["female_control"],
    ("control", "male"): GROUP_COLORS["male_control"],
    ("disease", "female"): GROUP_COLORS["female_disease"],
    ("disease", "male"): GROUP_COLORS["male_disease"],
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

BAND_DISPLAY_LABELS = {
    "cytosine_786": "Cytosine",
    "o_p_o_821": "O-P-O",
    "tyrosine_851": "Tyrosine",
    "c_o_c_sugar_925": "C-O-C (sugar)",
    "beta_sheet_980": r"$\beta$-sheet",
    "c_o_c_n_1060": "C-O-C-N",
    "lipid_cc_1089": "Lipid C-C",
    "po2_cn_1112": r"PO$_2^-$/C-N",
    "carotenoid_1156": "Carotenoid",
    "amide_iii_1258": "Amide III",
    "ch2_twist_1301": r"CH$_2$ twist",
    "ch2_ch3_scissor_1440": r"CH$_2$/CH$_3$ scissor",
    "ch2_ch3_bend_1465": r"CH$_2$/CH$_3$ bend",
    "carotenoid_1525": "Carotenoid",
    "tryptophan_1555": "Tryptophan",
    "amide_i_1657": "Amide I",
    "lipid_carbonyl_1721": "Lipid C=O",
    "lipid_ester_1744": "Lipid ester",
    "ch2_sym_stretch_2853": r"CH$_2$ sym. stretch",
    "ch2_asym_stretch_2890": r"CH$_2$ asym. stretch",
    "ch3_asym_stretch_2937": r"CH$_3$ asym. stretch",
}

# Drawn as axvspans at raman_band_style.alpha (0.25), so the printed span is
# far lighter than the swatch itself: these sit a step darker than a
# legend-only palette would, to stay visible once that alpha is applied.
BAND_CATEGORY_COLORS = {
    "nucleic_acids": "#8FC5EC",
    "proteins": "#96CE9E",
    "lipids": "#F0BE83",
    "carotenoids": "#F7E381",
    "mixed": "#BFBFBF",
}
BAND_CATEGORY_LABELS = {
    "nucleic_acids": "Nucleic acids",
    "proteins": "Proteins",
    "lipids": "Lipids",
    "carotenoids": "Carotenoids",
    "mixed": "Mixed",
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


def _plot_bands(plot_cfg: dict) -> Mapping[str, Iterable[float]]:
    bands = plot_cfg.get("bands", {})
    if not isinstance(bands, Mapping):
        return {}
    validate_configured_bands(bands)
    return bands


def _add_raman_band_spans(
    ax: plt.Axes,
    actual_wavenumbers: np.ndarray,
    plot_cfg: dict,
    *,
    display_wavenumbers: np.ndarray | None = None,
) -> None:
    """Draw publication-style, category-coloured pre-specified band windows."""
    style = plot_cfg.get("raman_band_style", {})
    if not bool(style.get("enabled", False)):
        return
    _plot_bands(plot_cfg)  # validates the configured analysis windows before display
    actual = np.asarray(actual_wavenumbers, dtype=float)
    display = actual if display_wavenumbers is None else np.asarray(display_wavenumbers, dtype=float)
    if actual.ndim != 1 or display.shape != actual.shape:
        raise ValueError("Raman-band display axis must match the physical wavenumber axis")
    visible_categories = list(getattr(ax, "_raman_band_categories", ()))
    alpha = float(style.get("alpha", 0.25))
    for band in RAMAN_BANDS.values():
        left, right = band.bounds
        if right < actual[0] or left > actual[-1]:
            continue
        plot_left, plot_right = np.interp((left, right), actual, display)
        ax.axvspan(plot_left, plot_right, color=BAND_CATEGORY_COLORS[band.category], alpha=alpha, linewidth=0, zorder=0)
        if band.category not in visible_categories:
            visible_categories.append(band.category)
    if visible_categories:
        handles = [
            Patch(facecolor=BAND_CATEGORY_COLORS[category], edgecolor="none", alpha=0.8, label=BAND_CATEGORY_LABELS[category])
            for category in visible_categories
        ]
        legend = ax.legend(handles=handles, loc="upper left", ncol=len(handles), fontsize=8, frameon=True, borderpad=0.35)
        legend.set_zorder(3)
    setattr(ax, "_raman_band_categories", visible_categories)


def _mean_signal(matrix: np.ndarray) -> np.ndarray:
    return np.nanmean(np.asarray(matrix, dtype=float), axis=0)


def _upper_envelope(signals: Iterable[np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(signal, dtype=float) for signal in signals]
    if not arrays:
        return np.array([], dtype=float)
    return np.nanmax(np.vstack(arrays), axis=0)


def _snv_rescale(y: np.ndarray) -> np.ndarray:
    """SNV-rescale a 1-D spectrum for display only (zero mean, unit std).

    Applied when regions with different normalization methods (e.g. area norm vs
    SNV) are plotted on a shared y-axis.  Never mutates stored data.
    """
    arr = np.asarray(y, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr
    mu = float(np.mean(finite))
    sd = float(np.std(finite))
    if sd < 1e-12:
        return arr - mu
    return (arr - mu) / sd


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
    return np.sort(peaks[ranked][y[peaks[ranked]] > 0])


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


def _nearest_band_name(
    wavenumber: float,
    bands: Mapping[str, Iterable[float]],
    *,
    tolerance: float = 12.0,
) -> str | None:
    nearest_name: str | None = None
    nearest_distance = tolerance
    for name, bounds in bands.items():
        values = tuple(bounds)
        if len(values) != 2:
            continue
        center = (float(values[0]) + float(values[1])) / 2.0
        distance = abs(wavenumber - center)
        if distance <= nearest_distance:
            nearest_name = name
            nearest_distance = distance
    return nearest_name


def _nearest_band_label(
    wavenumber: float,
    bands: Mapping[str, Iterable[float]],
    *,
    tolerance: float = 12.0,
) -> str | None:
    """Return the curated Raman-band label nearest to a detected peak."""
    band_name = _nearest_band_name(wavenumber, bands, tolerance=tolerance)
    if band_name is None:
        return None
    return BAND_DISPLAY_LABELS.get(band_name, band_name.replace("_", " "))


def _annotate_peaks(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    actual_wavenumbers: np.ndarray,
    peak_cfg: dict,
    *,
    bands: Mapping[str, Iterable[float]] | None = None,
    fontsize: float = 7.5,
    rotation: float = 0.0,
    clip_text: bool = False,
    horizontal_bands: Container[str] | None = None,
) -> None:
    """Annotate positive, curated Raman-band peaks with vertical leader lines.

    ``rotation`` is for narrow axes: horizontal labels of neighbouring bands
    overlap once the axis is only a text column wide, and rotating them is the
    only placement that stays legible without dropping bands.

    ``horizontal_bands`` exempts named bands from that rotation. A rotated
    label is as tall as the label is long, so the tallest peak's label sets the
    headroom the whole panel is scaled to fit; drawing that one label
    horizontally with an arrow to its peak costs a single line of height
    instead, and leaves the traces taller. Use it only for bands whose
    neighbours are far enough away that a full-width label cannot collide.

    ``clip_text`` confines the label and its leader to the axes rectangle.
    ``annotation_clip`` only tests the anchor point, so a tall peak can still
    push its label past the top spine and into a neighbouring panel; this cuts
    it at the frame instead.
    """
    peak_idx = _peak_indices(y, peak_cfg)
    if peak_idx.size == 0 or not bands:
        return

    y_bottom, y_top = map(float, ax.get_ylim())
    y_span = y_top - y_bottom
    if not np.isfinite(y_span) or y_span <= 0:
        return

    annotated_bands = getattr(ax, "_raman_annotated_bands", set())
    label_levels = (0.74, 0.60, 0.88, 0.48)
    y_top = float(ax.get_ylim()[1])
    cap = y_top - 0.03 * y_span
    legend = ax.get_legend()
    if legend is not None:
        renderer = ax.figure.canvas.get_renderer()
        legend_bottom = ax.transData.inverted().transform(legend.get_window_extent(renderer=renderer).get_points())[0, 1]
        cap = min(cap, float(legend_bottom) - 0.035 * y_span)

    for order, idx in enumerate(sorted(peak_idx, key=lambda item: x[item])):
        band_name = _nearest_band_name(float(actual_wavenumbers[idx]), bands)
        if band_name is None or band_name in annotated_bands:
            continue
        label = _nearest_band_label(float(actual_wavenumbers[idx]), bands)
        if label is None:
            continue
        peak_x = float(x[idx])
        peak_y = float(y[idx])
        upright = bool(horizontal_bands) and band_name in horizontal_bands
        if upright:
            # These are the tallest peaks, so the label always clears the trace
            # and the cap is not applied: clamping it would drop the label onto
            # the peak and reverse the arrow it is meant to hang from.
            label_y = peak_y + 0.08 * y_span
        elif rotation:
            # Rotated labels occupy a narrow column, so they cannot collide with
            # their neighbours and do not need the staggered levels that keep
            # horizontal labels apart; parking them just above each peak avoids
            # long leader lines crossing the trace.
            label_y = min(peak_y + 0.04 * y_span, cap)
        else:
            label_y = min(
                max(peak_y + 0.06 * y_span, y_bottom + label_levels[order % len(label_levels)] * y_span),
                cap,
            )
        label_rotation = 0.0 if upright else rotation
        ax.annotate(
            label,
            xy=(peak_x, peak_y),
            xytext=(peak_x, label_y),
            # With rotation_mode="anchor" the alignment is applied in the
            # text's own rotated frame, so a rotated label needs ha="left"
            # (start of the text, which points down after a 90 deg turn) and
            # va="center" to run upward from the peak instead of straddling it.
            ha="left" if label_rotation else "center",
            va="center" if label_rotation else "bottom",
            rotation=label_rotation,
            rotation_mode="anchor",
            fontsize=fontsize,
            color="#111827",
            annotation_clip=True,
            # An upright label sits away from its peak, so it needs a head to
            # say which peak it belongs to; the rotated ones start at the peak
            # and only need a leader.
            arrowprops={
                "arrowstyle": "-|>" if upright else "-",
                "color": "#111827",
                "linewidth": 0.65,
                "shrinkA": 1 if upright else 0,
                "shrinkB": 2,
                **({"mutation_scale": 6.0} if upright else {}),
            },
            zorder=4,
            clip_on=clip_text,
        )
        annotated_bands.add(band_name)
    setattr(ax, "_raman_annotated_bands", annotated_bands)


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
        label="Control",
    )
    ax.fill_between(wavenumbers, mean_control, alpha=0.08, color=CONDITION_COLORS["control"])
    ax.plot(
        wavenumbers,
        mean_disease,
        color=CONDITION_COLORS["disease"],
        linewidth=2.0,
        label="Disease",
    )
    ax.fill_between(wavenumbers, mean_disease, alpha=0.08, color=CONDITION_COLORS["disease"])
    _set_spectrum_limits(ax, [wavenumbers], [mean_control, mean_disease], top_pad=0.30)
    _add_raman_band_spans(ax, wavenumbers, plot_cfg)
    _annotate_peaks(
        ax,
        wavenumbers,
        _upper_envelope((mean_control, mean_disease)),
        wavenumbers,
        plot_cfg.get("peak_detection", {}),
        bands=_plot_bands(plot_cfg),
    )
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
        label="Control",
    )
    ax.fill_between(wavenumbers, mean_control, alpha=0.08, color=control_color)
    ax.plot(
        wavenumbers,
        mean_disease,
        color=disease_color,
        linewidth=2.0,
        label="Disease",
    )
    ax.fill_between(wavenumbers, mean_disease, alpha=0.08, color=disease_color)
    _set_spectrum_limits(ax, [wavenumbers], [mean_control, mean_disease], top_pad=0.30)
    _add_raman_band_spans(ax, wavenumbers, plot_cfg)
    _annotate_peaks(
        ax,
        wavenumbers,
        _upper_envelope((mean_control, mean_disease)),
        wavenumbers,
        plot_cfg.get("peak_detection", {}),
        bands=_plot_bands(plot_cfg),
    )
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
        label=f"{_condition_name(condition)} mean",
    )
    ax.fill_between(wavenumbers, mean_spectrum, alpha=0.08, color=color)
    _set_spectrum_limits(
        ax,
        [wavenumbers],
        [np.asarray(spectra, dtype=float), mean_spectrum],
        top_pad=0.18,
    )
    _annotate_peaks(
        ax, wavenumbers, mean_spectrum, wavenumbers, plot_cfg.get("peak_detection", {}), bands=_plot_bands(plot_cfg)
    )
    _finalize_axis(
        ax,
        title=f"Preprocessed spectra: {_condition_name(condition)} individual spectra - {_region_title(region)} (n={spectra.shape[0]})",
    )
    handles, labels = ax.get_legend_handles_labels()
    _place_figure_legend(fig, handles, labels)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig, ax


def plot_individual_by_gender_region(
    wavenumbers: np.ndarray,
    groups: dict[tuple[str, str], np.ndarray],
    region: str,
    plot_cfg: dict,
) -> tuple[plt.Figure, np.ndarray]:
    """2×2 grid of individual spectra for (control/disease) × (female/male)."""
    order: list[tuple[str, str]] = [
        ("control", "female"),
        ("disease", "female"),
        ("control", "male"),
        ("disease", "male"),
    ]
    line_alpha = float(plot_cfg.get("modes", {}).get("individual", {}).get("line_alpha", 0.25))
    mean_linewidth = float(plot_cfg.get("modes", {}).get("individual", {}).get("mean_linewidth", 2.5))

    fig, axes = plt.subplots(2, 2, figsize=(16, 8), sharex=True, sharey=False)
    axes_flat = axes.flatten()

    for i, cohort in enumerate(order):
        ax = axes_flat[i]
        condition, gender = cohort
        spectra = groups.get(cohort)
        color = COHORT_COLORS.get(cohort, "#444444")
        label = _cohort_name(condition, gender)

        if spectra is None or np.asarray(spectra).size == 0:
            ax.text(0.5, 0.5, f"No data\n{label}", ha="center", va="center", transform=ax.transAxes, fontsize=10)
            ax.set_axis_off()
            continue

        spectra = np.asarray(spectra, dtype=float)
        for row in spectra:
            ax.plot(wavenumbers, row, color=color, linewidth=0.6, alpha=line_alpha)
        mean_spectrum = _mean_signal(spectra)
        ax.plot(
            wavenumbers,
            mean_spectrum,
            color=color,
            linewidth=mean_linewidth,
            label=f"{label} mean",
        )
        ax.fill_between(wavenumbers, mean_spectrum, alpha=0.08, color=color)
        _set_spectrum_limits(ax, [wavenumbers], [spectra, mean_spectrum], top_pad=0.18)
        _annotate_peaks(
            ax, wavenumbers, mean_spectrum, wavenumbers, plot_cfg.get("peak_detection", {}), bands=_plot_bands(plot_cfg)
        )
        _finalize_axis(ax, title=label)
        handles, labels_ax = ax.get_legend_handles_labels()
        ax.legend(handles, labels_ax, loc="upper right", fontsize=8, frameon=False)

    for ax in axes_flat:
        ax.set_xlabel("Raman Shift (cm$^{-1}$)", fontsize=10)

    fig.suptitle(
        f"Preprocessed spectra: individual by group — {_region_title(region)}",
        fontsize=14,
        y=1.01,
    )
    fig.tight_layout()
    return fig, axes_flat


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

    # SNV-rescale every region's series for display so regions with different
    # normalization (area norm vs SNV) share a comparable y-axis. Mirrors the
    # same fix in plot_cohort_mean_spectra_grid. Peak annotation uses the
    # rescaled signal for prominence detection; actual wavenumber positions
    # and stored data are unaffected.
    display_series: list[dict[str, object]] = [
        {**entry, "segments": {region: _snv_rescale(np.asarray(y, dtype=float)) for region, y in entry["segments"].items()}}
        for entry in series
    ]

    annotation_signals: dict[str, list[np.ndarray]] = {}
    for entry in display_series:
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
            if bool(entry.get("shade", True)):
                ax.fill_between(display_axes[region], y, alpha=0.08, color=color)
            if annotate:
                annotation_signals.setdefault(region, []).append(np.asarray(y, dtype=float))

    _set_spectrum_limits(
        ax,
        display_axes.values(),
        [np.asarray(entry["segments"][region], dtype=float) for entry in display_series for region in entry["segments"]],
        top_pad=0.30,
    )
    for region, signals in annotation_signals.items():
        _annotate_peaks(
            ax,
            np.asarray(display_axes[region], dtype=float),
            _upper_envelope(signals),
            np.asarray(region_axes[region], dtype=float),
            plot_cfg.get("peak_detection", {}),
            bands=_plot_bands(plot_cfg),
        )
    for region, axis in region_axes.items():
        _add_raman_band_spans(ax, np.asarray(axis, dtype=float), plot_cfg, display_wavenumbers=display_axes[region])
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
        # SNV-rescale every region's mean spectrum for display so regions with
        # different normalization (area norm vs SNV) share a comparable y-axis.
        # Peak annotation uses the rescaled signal for prominence detection;
        # actual wavenumber positions are unaffected.
        display_segments = {
            region: _snv_rescale(np.asarray(y, dtype=float))
            for region, y in segments.items()
        }
        for region in region_axes:
            y_disp = display_segments.get(region)
            if y_disp is None:
                continue
            color = REGION_COLORS.get(region, "#475569")
            (line,) = ax.plot(
                display_axes[region],
                y_disp,
                color=color,
                linewidth=1.8,
                label=_region_title(region),
            )
            ax.fill_between(display_axes[region], y_disp, alpha=0.08, color=color)
            if _region_title(region) not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(_region_title(region))
        _set_spectrum_limits(
            ax,
            display_axes.values(),
            list(display_segments.values()),
            top_pad=0.30,
        )
        for region, y_disp in display_segments.items():
            _annotate_peaks(
                ax,
                np.asarray(display_axes[region], dtype=float),
                y_disp,
                np.asarray(region_axes[region], dtype=float),
                plot_cfg.get("peak_detection", {}),
                bands=_plot_bands(plot_cfg),
            )
        for region, axis in region_axes.items():
            _add_raman_band_spans(ax, np.asarray(axis, dtype=float), plot_cfg, display_wavenumbers=display_axes[region])
        _style_wavenumber_axis(ax, region_axes, display_axes, plot_cfg.get("axis", {}))
        _decorate_regions(ax, gap_spans, label_positions)
        _finalize_axis(
            ax,
            title=_cohort_name(condition, gender),
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
    ax.fill_between(wavenumbers, mean_spectrum, alpha=0.08, color=color)
    _set_spectrum_limits(ax, [wavenumbers], [mean_spectrum], top_pad=0.30)
    _annotate_peaks(
        ax, wavenumbers, mean_spectrum, wavenumbers, plot_cfg.get("peak_detection", {}), bands=_plot_bands(plot_cfg)
    )
    _add_raman_band_spans(ax, wavenumbers, plot_cfg)
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
    plot_cfg: dict,
) -> tuple[plt.Figure, plt.Axes]:
    condition, gender = cohort
    color = COHORT_COLORS[cohort]
    return plot_global_comparison(
        region_axes,
        [
            {
                "label": _cohort_name(condition, gender),
                "color": color,
                "linewidth": 2.3,
                "segments": segments,
            }
        ],
        title=f"{_cohort_name(condition, gender)} whole spectra",
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
