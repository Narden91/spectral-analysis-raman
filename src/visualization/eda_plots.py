"""Visualization helpers for Raman spectroscopy EDA."""

from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.visualization.theme import (
    POINT_EDGE_COLOR,
    GROUP_COLORS,
    GROUP_LABELS,
    WHOLE_SPECTRA_NEUTRAL,
    WHOLE_SPECTRA_RANGE_COLORS,
    WHOLE_SPECTRA_RANGE_LABELS,
    WHOLE_SPECTRA_TEXT,
    CLUSTER_COLORS,
    SUBJECT_SCATTER_SIZE,
    category_sort_key,
    display_label,
    format_category_label,
    finalize_axis as _finalize_axis,
    get_discrete_palette,
    set_gallery_theme,
)

set_gallery_theme()



def plot_group_mean_spectra(
    wavenumbers: np.ndarray,
    groups: dict[str, np.ndarray],
    title: str = "Group Mean Spectra Comparison",
    show_ci: bool = True,
    ci_alpha: float = 0.2,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot mean spectra for each group with 95% confidence intervals."""
    fig, ax = plt.subplots(figsize=(12, 6))

    diffs = np.diff(wavenumbers)
    has_gap = len(diffs) > 0 and np.max(diffs) > 100
    if has_gap:
        gap_idx = int(np.argmax(diffs))
        gap_start = float(wavenumbers[gap_idx])
        gap_end = float(wavenumbers[gap_idx + 1])
        plot_wavenumbers = np.insert(wavenumbers, gap_idx + 1, (gap_start + gap_end) / 2)
    else:
        plot_wavenumbers = wavenumbers

    for key, spectra in groups.items():
        color = GROUP_COLORS.get(key, "gray")
        label = GROUP_LABELS.get(key, key)
        linestyle = "--" if "control" in key.lower() else "-"
        mean = np.nanmean(spectra, axis=0)
        
        plot_mean = np.insert(mean, gap_idx + 1, np.nan) if has_gap else mean
        ax.plot(
            plot_wavenumbers,
            plot_mean,
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
            label=label,
        )

        if show_ci and spectra.shape[0] > 2:
            ci = 1.96 * _standard_error(spectra)
            plot_mean_lower = np.insert(mean - ci, gap_idx + 1, np.nan) if has_gap else mean - ci
            plot_mean_upper = np.insert(mean + ci, gap_idx + 1, np.nan) if has_gap else mean + ci
            ax.fill_between(
                plot_wavenumbers, plot_mean_lower, plot_mean_upper, color=color, alpha=ci_alpha
            )

    if has_gap:
        ax.axvspan(gap_start, gap_end, color="#D3D3D3", alpha=0.5, zorder=0, linewidth=0)

    _finalize_axis(
        ax,
        title=title,
        xlabel="Raman Shift (cm$^{-1}$)",
        ylabel="Intensity (a.u.)",
        xlim=(float(wavenumbers.min()), float(wavenumbers.max())),
    )

    return fig, ax


def plot_cluster_mean_spectra(
    wavenumbers: np.ndarray,
    groups: dict[str, np.ndarray],
    title: str = "Cluster Mean Spectra Comparison",
    show_ci: bool = True,
    ci_alpha: float = 0.2,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot mean spectra for each cluster with 95% confidence intervals."""
    fig, ax = plt.subplots(figsize=(12, 6))

    diffs = np.diff(wavenumbers)
    has_gap = len(diffs) > 0 and np.max(diffs) > 100
    if has_gap:
        gap_idx = int(np.argmax(diffs))
        gap_start = float(wavenumbers[gap_idx])
        gap_end = float(wavenumbers[gap_idx + 1])
        plot_wavenumbers = np.insert(wavenumbers, gap_idx + 1, (gap_start + gap_end) / 2)
    else:
        plot_wavenumbers = wavenumbers

    line_styles = ["-", "--", "-.", ":"]
    for i, (key, spectra) in enumerate(groups.items()):
        try:
            cluster_id = int(float(key))
            color = CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]
        except (TypeError, ValueError):
            color = "gray"
        
        label = format_category_label("hdbscan_cluster", key)
        linestyle = line_styles[i % len(line_styles)]
        
        mean = np.nanmean(spectra, axis=0)
        plot_mean = np.insert(mean, gap_idx + 1, np.nan) if has_gap else mean
        ax.plot(
            plot_wavenumbers,
            plot_mean,
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
            label=label,
        )

        if show_ci and spectra.shape[0] > 2:
            ci = 1.96 * _standard_error(spectra)
            plot_mean_lower = np.insert(mean - ci, gap_idx + 1, np.nan) if has_gap else mean - ci
            plot_mean_upper = np.insert(mean + ci, gap_idx + 1, np.nan) if has_gap else mean + ci
            ax.fill_between(
                plot_wavenumbers, plot_mean_lower, plot_mean_upper, color=color, alpha=ci_alpha
            )

    if has_gap:
        ax.axvspan(gap_start, gap_end, color="#D3D3D3", alpha=0.5, zorder=0, linewidth=0)

    _finalize_axis(
        ax,
        title=title,
        xlabel="Raman Shift (cm$^{-1}$)",
        ylabel="Intensity (a.u.)",
        xlim=(float(wavenumbers.min()), float(wavenumbers.max())),
    )

    return fig, ax


def plot_cluster_mean_spectra_faceted(
    wavenumbers: np.ndarray,
    groups: dict[str, np.ndarray],
    title: str = "Faceted Cluster Mean Spectra Comparison",
) -> tuple[plt.Figure, np.ndarray]:
    """Plot mean spectra for each cluster faceted in a grid, with context.

    Parameters
    ----------
    wavenumbers : np.ndarray
        1D wavenumber axis.
    groups : dict[str, np.ndarray]
        Dictionary mapping cluster ID string to array of shape (N_subj, N_wn)
        (or (1, N_wn) for cluster-mean spectra).
    title : str
        Global plot title.

    Returns
    -------
    tuple[plt.Figure, np.ndarray]
        Figure and array of Axes.
    """
    n_clusters = len(groups)
    if n_clusters == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        return fig, np.array([ax])

    # Dynamic grid sizing
    if n_clusters <= 4:
        cols = n_clusters
    else:
        cols = 4
    rows = int(np.ceil(n_clusters / cols))

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * 4.5, rows * 3.5),
        sharex=True,
        sharey=True,
    )
    
    # Flatten axes array for simple indexing
    axes_flat = axes.flatten() if n_clusters > 1 else np.array([axes])
    
    # Pre-calculate all mean spectra so we can plot context easily
    mean_spectra = {}
    for key, spectra in groups.items():
        mean_spectra[key] = np.nanmean(spectra, axis=0)

    diffs = np.diff(wavenumbers)
    has_gap = len(diffs) > 0 and np.max(diffs) > 100
    if has_gap:
        gap_idx = int(np.argmax(diffs))
        gap_start = float(wavenumbers[gap_idx])
        gap_end = float(wavenumbers[gap_idx + 1])
        plot_wavenumbers = np.insert(wavenumbers, gap_idx + 1, (gap_start + gap_end) / 2)
    else:
        plot_wavenumbers = wavenumbers

    # Sort keys if they are numeric to make the grid sequential
    def _key_sort_fn(x: str) -> tuple[int, str]:
        try:
            return (0, str(int(float(x))))
        except (TypeError, ValueError):
            return (1, x)

    sorted_keys = sorted(groups.keys(), key=_key_sort_fn)

    for idx, target_key in enumerate(sorted_keys):
        ax = axes_flat[idx]
        
        if has_gap:
            ax.axvspan(gap_start, gap_end, color="#D3D3D3", alpha=0.5, zorder=0, linewidth=0)
        
        # 1. Plot all other clusters in the background as thin light gray lines
        for other_key, mean in mean_spectra.items():
            if other_key != target_key:
                plot_mean = np.insert(mean, gap_idx + 1, np.nan) if has_gap else mean
                ax.plot(
                    plot_wavenumbers,
                    plot_mean,
                    color="#e0e0e0",
                    linewidth=1.0,
                    alpha=0.6,
                    zorder=1,
                )
        
        # 2. Plot the target cluster in the foreground with color and distinct style
        target_mean = mean_spectra[target_key]
        plot_target_mean = np.insert(target_mean, gap_idx + 1, np.nan) if has_gap else target_mean
        
        try:
            cluster_id = int(float(target_key))
            color = CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]
        except (TypeError, ValueError):
            color = "gray"
            
        label = format_category_label("hdbscan_cluster", target_key)
        
        ax.plot(
            plot_wavenumbers,
            plot_target_mean,
            color=color,
            linewidth=2.2,
            zorder=2,
            label=label,
        )
        
        # Set panel title
        ax.set_title(label, fontsize=12, fontweight="semibold")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_xlim(float(wavenumbers.min()), float(wavenumbers.max()))

        # Set axes labels
        # Bottom row subplots get x-label
        if idx >= (rows - 1) * cols or (n_clusters - 1 - idx) < cols:
            ax.set_xlabel("Raman Shift (cm$^{-1}$)", fontsize=10)
        # Leftmost column subplots get y-label
        if idx % cols == 0:
            ax.set_ylabel("Intensity (a.u.)", fontsize=10)

    # Hide unused subplots
    for idx in range(len(sorted_keys), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout()
    
    return fig, axes


def plot_difference_spectra(
    wavenumbers: np.ndarray,
    disease: np.ndarray,
    control: np.ndarray,
    label: str = "Disease - Control",
    title: str = "Difference Spectrum",
    color: str = "#CC3311",
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the mean difference spectrum with a propagated confidence band."""
    fig, ax = plt.subplots(figsize=(12, 5))

    mean_g = np.nanmean(disease, axis=0)
    mean_c = np.nanmean(control, axis=0)
    diff = mean_g - mean_c

    sem_g = _standard_error(disease)
    sem_c = _standard_error(control)
    sem_diff = np.sqrt(sem_g**2 + sem_c**2)
    ci = 1.96 * sem_diff

    ax.plot(wavenumbers, diff, color=color, linewidth=1.5, label=label)
    if np.any(ci > 0):
        ax.fill_between(wavenumbers, diff - ci, diff + ci, color=color, alpha=0.2)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)

    _finalize_axis(
        ax,
        title=title,
        xlabel="Raman Shift (cm$^{-1}$)",
        ylabel="Intensity Difference (a.u.)",
        xlim=(float(wavenumbers.min()), float(wavenumbers.max())),
    )

    return fig, ax


def _standard_error(spectra: np.ndarray) -> np.ndarray:
    if spectra.shape[0] <= 1:
        return np.zeros(spectra.shape[1], dtype=float)
    return np.nanstd(spectra, axis=0, ddof=1) / np.sqrt(spectra.shape[0])


def plot_pca_scores(
    scores: np.ndarray,
    labels: np.ndarray,
    explained_variance: np.ndarray,
    pc_x: int = 0,
    pc_y: int = 1,
    title: str = "PCA Score Plot",
    group_names: Optional[dict] = None,
    loadings: Optional[np.ndarray] = None,
    wavenumbers: Optional[np.ndarray] = None,
    top_n_features: int = 3,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a 2D PCA score view colored by group label. Optional biplot loadings."""
    fig, ax = plt.subplots(figsize=(9, 7))

    unique_labels = np.unique(labels)
    for lab in unique_labels:
        mask = labels == lab
        color = GROUP_COLORS.get(lab, "gray")
        display = (
            GROUP_LABELS.get(lab, lab)
            if group_names is None
            else group_names.get(lab, lab)
        )
        ax.scatter(
            scores[mask, pc_x],
            scores[mask, pc_y],
            c=color,
            label=display,
            s=SUBJECT_SCATTER_SIZE,
            alpha=0.82,
            edgecolors=POINT_EDGE_COLOR,
            linewidth=0.55,
        )

    if loadings is not None and wavenumbers is not None:
        # Get score ranges to scale the loading vectors appropriately
        x_range = scores[:, pc_x].max() - scores[:, pc_x].min()
        y_range = scores[:, pc_y].max() - scores[:, pc_y].min()
        
        loading_x = loadings[pc_x]
        loading_y = loadings[pc_y]
        
        # Determine the most influential features (highest magnitude)
        magnitude = np.sqrt(loading_x**2 + loading_y**2)
        top_indices = np.argsort(magnitude)[-top_n_features:]
        
        # Scale arrows to fit nicely within the plot (about 40% of range)
        max_loading = magnitude.max()
        if max_loading > 0:
            scale_x = (x_range * 0.4) / max_loading
            scale_y = (y_range * 0.4) / max_loading
            
            for idx in top_indices:
                x_val = loading_x[idx] * scale_x
                y_val = loading_y[idx] * scale_y
                wn = wavenumbers[idx]
                
                # Draw the arrow
                ax.arrow(
                    0, 0, x_val, y_val, 
                    color="red", alpha=0.6, width=0.005 * min(x_range, y_range), 
                    head_width=0.03 * min(x_range, y_range)
                )
                
                # Add wavenumber label
                ax.text(
                    x_val * 1.1, y_val * 1.1, 
                    f"{wn:.0f}", 
                    color="darkred", fontsize=9, ha='center', va='center'
                )

    _finalize_axis(
        ax,
        title=title,
        xlabel=f"PC{pc_x + 1} ({explained_variance[pc_x] * 100:.1f}%)",
        ylabel=f"PC{pc_y + 1} ({explained_variance[pc_y] * 100:.1f}%)",
    )

    return fig, ax


def plot_pca_loadings(
    wavenumbers: np.ndarray,
    loadings: np.ndarray,
    explained_variance: np.ndarray,
    n_components: int = 3,
    title: str = "PCA Loadings",
) -> tuple[plt.Figure, list[plt.Axes]]:
    """
    Plot PCA loading vectors for the top N components.
    """
    fig, axes = plt.subplots(
        n_components, 1, figsize=(12, 3 * n_components), sharex=True
    )
    if n_components == 1:
        axes = [axes]

    colors = ["#264653", "#2A9D8F", "#E76F51"]
    for i, ax in enumerate(axes):
        if i >= loadings.shape[0]:
            break
        c = colors[i % len(colors)]
        ax.plot(wavenumbers, loadings[i], color=c, linewidth=1.2)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.5)
        ax.set_ylabel(f"PC{i + 1} ({explained_variance[i] * 100:.1f}%)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.3)

    axes[-1].set_xlabel("Raman Shift (cm$^{-1}$)", fontsize=12)
    fig.suptitle(title, fontsize=14, y=1.01)
    fig.tight_layout()

    return fig, axes


def plot_scree(
    explained_variance: np.ndarray,
    title: str = "Scree Plot",
) -> tuple[plt.Figure, plt.Axes]:
    """
    Scree plot showing individual and cumulative explained variance.
    """
    fig, ax1 = plt.subplots(figsize=(8, 5))

    n = len(explained_variance)
    pcs = np.arange(1, n + 1)
    cumulative = np.cumsum(explained_variance)

    ax1.bar(
        pcs, explained_variance * 100, color="#457B9D", alpha=0.7, label="Individual"
    )
    ax1.set_xlabel("Principal Component", fontsize=12)
    ax1.set_ylabel("Explained Variance (%)", fontsize=12, color="#457B9D")
    ax1.tick_params(axis="y", labelcolor="#457B9D")
    ax1.set_xticks(pcs)

    ax2 = ax1.twinx()
    ax2.plot(
        pcs, cumulative * 100, "o-", color="#CC3311", linewidth=2, label="Cumulative"
    )
    ax2.set_ylabel("Cumulative Variance (%)", fontsize=12, color="#CC3311")
    ax2.tick_params(axis="y", labelcolor="#CC3311")
    ax2.axhline(95, color="gray", linestyle=":", linewidth=1, alpha=0.7)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=10)

    ax1.set_title(title, fontsize=14)
    fig.tight_layout()

    return fig, ax1


def plot_band_boxplots(
    band_df: pd.DataFrame,
    group_labels: np.ndarray,
    title: str = "Raman Band Intensities by Group",
    ylabel: str = "Intensity",
) -> tuple[plt.Figure, list[plt.Axes]]:
    """
    Box plots showing intensity distributions at known Raman bands per group.
    """
    bands = band_df.columns.tolist()
    n_bands = len(bands)
    ncols = min(4, n_bands)
    nrows = (n_bands + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows))
    axes = np.atleast_2d(axes).flatten()

    unique_groups = np.unique(group_labels)
    for i, band in enumerate(bands):
        ax = axes[i]
        data_by_group = []
        labels_by_group = []
        colors = []
        for g in unique_groups:
            vals = band_df.loc[group_labels == g, band].dropna().values
            data_by_group.append(vals)
            labels_by_group.append(GROUP_LABELS.get(g, g))
            colors.append(GROUP_COLORS.get(g, "gray"))

        bp = ax.boxplot(data_by_group, patch_artist=True, widths=0.6)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_xticklabels(labels_by_group, rotation=45, ha="right", fontsize=8)
        ax.set_title(band, fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    # Hide unused axes
    for j in range(n_bands, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()

    return fig, axes.tolist()


def plot_ratio_summary(
    ratio_test_df: pd.DataFrame,
    title: str = "Cross-Region Ratio Summary",
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Summarize ratio results in a presentation-friendly view.

    The left panel shows the pooled disease-vs-control percent shift for each
    ratio. The right panel shows whether the pooled, female-only, and male-only
    contrasts survive FDR correction.
    """
    if ratio_test_df.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No ratio summary available", ha="center", va="center")
        ax.set_axis_off()
        return fig, [ax]

    contrast_order = [
        "disease_vs_control",
        "female_disease_vs_female_control",
        "male_disease_vs_male_control",
    ]
    contrast_labels = {
        "disease_vs_control": "Pooled",
        "female_disease_vs_female_control": "Female",
        "male_disease_vs_male_control": "Male",
    }
    ratio_labels = {
        "amide_i_1655__over__ch2_sym_stretch_2850": "Amide I / CH2\nprotein vs lipid",
        "amide_i_1655__over__ch3_sym_stretch_2930": "Amide I / CH3\nprotein vs CH3",
        "phenylalanine_1004__over__ch3_sym_stretch_2930": "Phenylalanine / CH3",
        "dna_po2_sym_1090__over__ch2_sym_stretch_2850": "DNA / CH2\nDNA vs lipid",
        "amide_iii_1260__over__amide_i_1655": "Amide III / Amide I",
        "ch2_sym_stretch_2850__over__ch3_sym_stretch_2930": "CH2 / CH3\nlipid balance",
    }

    working = ratio_test_df.copy()
    working = working[working["contrast"].isin(contrast_order)].copy()
    if working.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No supported ratio contrasts found", ha="center", va="center")
        ax.set_axis_off()
        return fig, [ax]

    pooled = working[working["contrast"] == "disease_vs_control"].copy()
    ratio_order = pooled["ratio"].tolist() if not pooled.empty else working["ratio"].drop_duplicates().tolist()
    labels = [ratio_labels.get(ratio, ratio.replace("__over__", " / ").replace("_", " ")) for ratio in ratio_order]

    pooled = pooled.set_index("ratio").reindex(ratio_order)
    pooled_shift_pct = 100.0 * ((pooled["mean_ratio_a"] / pooled["mean_ratio_b"]) - 1.0)
    pooled_shift_pct = pooled_shift_pct.fillna(0.0)

    height = max(4.8, 0.95 * len(ratio_order) + 1.5)
    fig, (ax_shift, ax_sig) = plt.subplots(
        1,
        2,
        figsize=(12.5, height),
        gridspec_kw={"width_ratios": [2.6, 1.2]},
    )

    y_pos = np.arange(len(ratio_order))
    bar_colors = ["#CC3311" if value >= 0 else "#0077BB" for value in pooled_shift_pct]
    ax_shift.barh(y_pos, pooled_shift_pct.to_numpy(), color=bar_colors, alpha=0.85, height=0.62)
    ax_shift.axvline(0, color="#444444", linewidth=1.0)
    ax_shift.set_yticks(y_pos)
    ax_shift.set_yticklabels(labels, fontsize=10)
    ax_shift.invert_yaxis()
    ax_shift.set_xlabel("Disease vs control change in ratio (%)", fontsize=11)
    ax_shift.set_title("How the balance shifts in disease", fontsize=13, fontweight="bold")
    ax_shift.grid(True, axis="x", linestyle="--", alpha=0.25)
    ax_shift.text(
        0.02,
        1.02,
        "Lower in disease  <-  ->  Higher in disease",
        transform=ax_shift.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color="#4B5563",
    )

    max_abs = float(np.nanmax(np.abs(pooled_shift_pct.to_numpy()))) if len(pooled_shift_pct) else 0.0
    x_pad = max(0.15, max_abs * 0.16)
    ax_shift.set_xlim(-max_abs - x_pad, max_abs + x_pad)
    for y, value in zip(y_pos, pooled_shift_pct.to_numpy()):
        if value >= 0:
            x_text = value + x_pad * 0.35
            ha = "left"
        else:
            x_text = value - x_pad * 0.35
            ha = "right"
        ax_shift.text(x_text, y, f"{value:+.2f}%", va="center", ha=ha, fontsize=9, color="#1F2937")

    ax_sig.set_title("Which contrasts survive FDR?", fontsize=13, fontweight="bold")
    ax_sig.set_xlim(-0.5, len(contrast_order) - 0.5)
    ax_sig.set_ylim(-0.5, len(ratio_order) - 0.5)
    ax_sig.invert_yaxis()
    ax_sig.set_xticks(np.arange(len(contrast_order)))
    ax_sig.set_xticklabels([contrast_labels[key] for key in contrast_order], fontsize=10)
    ax_sig.set_yticks(y_pos)
    ax_sig.set_yticklabels([])
    ax_sig.grid(True, axis="both", linestyle="--", alpha=0.2)

    for row_index, ratio in enumerate(ratio_order):
        for col_index, contrast in enumerate(contrast_order):
            subset = working[(working["ratio"] == ratio) & (working["contrast"] == contrast)]
            if subset.empty:
                continue
            row = subset.iloc[0]
            direction = float(row["mean_ratio_a"] / row["mean_ratio_b"] - 1.0)
            color = "#CC3311" if direction >= 0 else "#0077BB"
            if bool(row.get("significant", False)):
                ax_sig.scatter(col_index, row_index, s=180, color=color, edgecolor="white", linewidth=1.0, zorder=3)
            else:
                ax_sig.scatter(col_index, row_index, s=110, facecolor="white", edgecolor="#9CA3AF", linewidth=1.4, zorder=3)

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#CC3311", markeredgecolor="white", markersize=9, label="Higher in disease"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#0077BB", markeredgecolor="white", markersize=9, label="Lower in disease"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#9CA3AF", markersize=8, label="Not FDR-significant"),
    ]
    ax_sig.legend(handles=legend_handles, frameon=False, loc="lower left", bbox_to_anchor=(-0.02, -0.18), fontsize=9)

    fig.suptitle(title, fontsize=15, y=0.995)
    fig.tight_layout()
    return fig, [ax_shift, ax_sig]


def plot_variance_decomposition(
    intra_subject: Optional[float],
    inter_subject: float,
    inter_group: float,
    title: str = "Variance Decomposition",
) -> tuple[plt.Figure, plt.Axes]:
    """
    Pie/bar chart showing how variance is partitioned.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    categories = ["Inter-subject\n(within group)", "Inter-group"]
    values = [inter_subject, inter_group]
    colors = ["#0077BB", "#CC3311"]

    _intra_available = intra_subject is not None and np.isfinite(intra_subject)
    if _intra_available:
        categories.insert(0, "Measurement noise")
        values.insert(0, float(intra_subject))
        colors.insert(0, "#88CCEE")

    bars = ax.barh(categories, values, color=colors, edgecolor="white", height=0.5)
    for bar, val in zip(bars, values):
        label = f"{val:.1f}%" if np.isfinite(val) else "N/A"
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            fontsize=11,
        )

    ax.set_xlabel("Fraction of Total Variance (%)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(0, max(values) * 1.2)
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    if not _intra_available:
        ax.text(
            0.98,
            0.02,
            "Measurement noise not estimated",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color="#4B5563",
        )
    fig.tight_layout()

    return fig, ax


def plot_group_distance_heatmap(
    distance_df: pd.DataFrame,
    title: str = "Group Centroid Distances",
) -> tuple[plt.Figure, plt.Axes]:
    """Heatmap of pairwise distances between group centroid spectra."""
    fig, ax = plt.subplots(figsize=(7, 5.5))

    if distance_df.empty:
        ax.text(0.5, 0.5, "No distance data available", ha="center", va="center")
        ax.set_axis_off()
        return fig, ax

    matrix = distance_df.to_numpy(dtype=float)
    pastel_distance_cmap = mcolors.LinearSegmentedColormap.from_list(
        "pastel_distance",
        ["#EAF4FB", "#DDECF7", "#D6EEE6", "#EEE5C9", "#F6E2B0"],
    )
    image = ax.imshow(matrix, cmap=pastel_distance_cmap, aspect="auto")
    labels = [GROUP_LABELS.get(label, label) for label in distance_df.index]
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)

    norm = image.norm
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            rgba = pastel_distance_cmap(norm(value))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            text_color = "#1F2933" if luminance >= 0.6 else "#FFFFFF"
            ax.text(
                col_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
            )

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Euclidean distance", fontsize=10)
    ax.set_title(title, fontsize=14)
    fig.tight_layout()

    return fig, ax


def plot_broken_spectrum(
    wavenumbers: np.ndarray,
    y: np.ndarray,
    split_index: int,
    title: str = "Broken-Axis Spectrum",
    xlabel: str = "Raman Shift (cm$^{-1}$)",
    ylabel: str = "Intensity",
    significant_regions: Optional[pd.DataFrame] = None,
    fill_to_zero: bool = False,
    line_color: str = "#264653",
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Plot a concatenated spectrum with an honest broken x-axis between regions."""
    wavenumbers = np.asarray(wavenumbers, dtype=float)
    y = np.asarray(y, dtype=float)
    split_index = int(split_index)
    if wavenumbers.ndim != 1 or y.ndim != 1 or wavenumbers.size != y.size:
        raise ValueError("wavenumbers and y must be 1D arrays with matching length")
    if split_index <= 0 or split_index >= wavenumbers.size:
        raise ValueError("split_index must fall strictly inside the spectral axis")

    left_x = wavenumbers[:split_index]
    right_x = wavenumbers[split_index:]
    left_y = y[:split_index]
    right_y = y[split_index:]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4.5),
        sharey=True,
        gridspec_kw={"width_ratios": [len(left_x), len(right_x)], "wspace": 0.05},
    )

    for ax, x_values, y_values in zip(axes, [left_x, right_x], [left_y, right_y]):
        ax.plot(x_values, y_values, color=line_color, linewidth=1.2)
        if fill_to_zero:
            ax.fill_between(x_values, 0, y_values, where=y_values >= 0, color="#CC3311", alpha=0.18)
            ax.fill_between(x_values, 0, y_values, where=y_values < 0, color="#457B9D", alpha=0.18)
            ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlim(x_values.min(), x_values.max())
        ax.margins(x=0)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    if significant_regions is not None and not significant_regions.empty:
        left_min, left_max = float(left_x.min()), float(left_x.max())
        right_min, right_max = float(right_x.min()), float(right_x.max())
        for region in significant_regions.itertuples(index=False):
            region_color = "#F4A261" if region.effect_direction == "positive" else "#8ECAE6"
            if region.end_wavenumber >= left_min and region.start_wavenumber <= left_max:
                axes[0].axvspan(
                    max(region.start_wavenumber, left_min),
                    min(region.end_wavenumber, left_max),
                    color=region_color,
                    alpha=0.18,
                    linewidth=0,
                )
            if region.end_wavenumber >= right_min and region.start_wavenumber <= right_max:
                axes[1].axvspan(
                    max(region.start_wavenumber, right_min),
                    min(region.end_wavenumber, right_max),
                    color=region_color,
                    alpha=0.18,
                    linewidth=0,
                )

    axes[0].spines["right"].set_visible(False)
    axes[1].spines["left"].set_visible(False)
    axes[1].tick_params(axis="y", left=False, labelleft=False)

    diagonal = 0.015
    kwargs_left = dict(transform=axes[0].transAxes, color="black", clip_on=False, linewidth=1.0)
    kwargs_right = dict(transform=axes[1].transAxes, color="black", clip_on=False, linewidth=1.0)
    axes[0].plot((1 - diagonal, 1 + diagonal), (-diagonal, +diagonal), **kwargs_left)
    axes[0].plot((1 - diagonal, 1 + diagonal), (1 - diagonal, 1 + diagonal), **kwargs_left)
    axes[1].plot((-diagonal, +diagonal), (-diagonal, +diagonal), **kwargs_right)
    axes[1].plot((-diagonal, +diagonal), (1 - diagonal, 1 + diagonal), **kwargs_right)

    axes[0].set_ylabel(ylabel, fontsize=12)
    fig.supxlabel(xlabel, fontsize=12)
    fig.suptitle(title, fontsize=14)
    fig.subplots_adjust(top=0.84, bottom=0.16, wspace=0.05)
    return fig, axes.tolist()


def plot_correlation_spectrum(
    wavenumbers: np.ndarray,
    rho: np.ndarray,
    significant: np.ndarray,
    title: str = "Correlation Spectrum",
) -> tuple[plt.Figure, plt.Axes]:
    """Plot per-wavenumber Spearman rho with FDR-significant bands shaded."""
    fig, ax = plt.subplots(figsize=(12, 4.5))

    ax.plot(wavenumbers, rho, color="#264653", linewidth=1.2)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)

    sig = np.asarray(significant, dtype=bool)
    if sig.any():
        ax.fill_between(
            wavenumbers, ax.get_ylim()[0], ax.get_ylim()[1],
            where=sig, color="#F4A261", alpha=0.18, linewidth=0,
        )

    ax.set_xlabel("Raman Shift (cm$^{-1}$)", fontsize=12)
    ax.set_ylabel("Spearman ρ", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(wavenumbers.min(), wavenumbers.max())
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()

    return fig, ax


def plot_correlation_matrix(
    correlation_df: pd.DataFrame,
    title: str = "Cross-Region Band Correlation",
) -> tuple[plt.Figure, plt.Axes]:
    """Heatmap for a fingerprint-by-CH band correlation matrix."""
    fig, ax = plt.subplots(figsize=(7, 5.5))

    if correlation_df.empty:
        ax.text(0.5, 0.5, "No correlation data available", ha="center", va="center")
        ax.set_axis_off()
        return fig, ax

    pivot = correlation_df.pivot(
        index="fingerprint_band",
        columns="ch_band",
        values="correlation",
    )
    matrix = pivot.to_numpy(dtype=float)
    image = ax.imshow(np.nan_to_num(matrix, nan=0.0), cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.grid(False, which="both", axis="both")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title(title, fontsize=14)

    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            label = "NA" if np.isnan(value) else f"{value:.2f}"
            ax.text(col_index, row_index, label, ha="center", va="center", fontsize=7)

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.grid(False)
    cbar.set_label("Correlation", fontsize=10)
    fig.tight_layout()
    return fig, ax


def plot_regional_variance_qc(
    variance_df: pd.DataFrame,
    title: str = "Regional SNV Residual Variance",
) -> tuple[plt.Figure, plt.Axes]:
    """Scatter regional residual variance and highlight flagged subjects."""
    fig, ax = plt.subplots(figsize=(7, 5.5))

    if variance_df.empty:
        ax.text(0.5, 0.5, "No QC data available", ha="center", va="center")
        ax.set_axis_off()
        return fig, ax

    colors = [GROUP_COLORS.get(group, "#6B7280") for group in variance_df.get("group", pd.Series([""] * len(variance_df)))]
    ax.scatter(
        variance_df["fingerprint_variance"],
        variance_df["ch_stretch_variance"],
        c=colors,
        alpha=0.8,
        edgecolors="white",
        linewidths=0.5,
        s=SUBJECT_SCATTER_SIZE,
    )

    if "fingerprint_threshold" in variance_df.columns and not variance_df.empty:
        ax.axvline(float(variance_df["fingerprint_threshold"].iloc[0]), color="#0072B2", linestyle="--", linewidth=1.0)
    if "ch_stretch_threshold" in variance_df.columns and not variance_df.empty:
        ax.axhline(float(variance_df["ch_stretch_threshold"].iloc[0]), color="#E69F00", linestyle="--", linewidth=1.0)

    outliers = variance_df[variance_df["is_outlier"]]
    for row in outliers.itertuples(index=False):
        ax.scatter(
            row.fingerprint_variance,
            row.ch_stretch_variance,
            facecolors="none",
            edgecolors="#111827",
            linewidths=1.2,
            s=90,
        )
        ax.text(
            row.fingerprint_variance,
            row.ch_stretch_variance,
            str(row.subject_id),
            fontsize=8,
            ha="left",
            va="bottom",
        )

    ax.set_xlabel("Fingerprint residual variance", fontsize=11)
    ax.set_ylabel("CH-stretch residual variance", fontsize=11)
    ax.set_title(title, fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_mfa_scree(
    explained_variance_ratio: np.ndarray,
    block_contributions: pd.DataFrame,
    title: str = "MFA Scree",
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Stacked scree plot with per-block contribution to each MFA component."""
    explained_variance_ratio = np.asarray(explained_variance_ratio, dtype=float)
    components = np.arange(1, explained_variance_ratio.size + 1)
    fp = (
        explained_variance_ratio
        * block_contributions["fingerprint_contribution"].to_numpy(dtype=float)
        * 100.0
    )
    ch = (
        explained_variance_ratio
        * block_contributions["ch_stretch_contribution"].to_numpy(dtype=float)
        * 100.0
    )
    cumulative = np.cumsum(explained_variance_ratio) * 100.0

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.bar(
        components,
        fp,
        color=WHOLE_SPECTRA_RANGE_COLORS["fingerprint_1300"],
        edgecolor="white",
        width=0.7,
        zorder=3,
    )
    ax.bar(
        components,
        ch,
        bottom=fp,
        color=WHOLE_SPECTRA_RANGE_COLORS["ch_stretch_2850"],
        edgecolor="white",
        width=0.7,
        zorder=3,
    )
    ax.set_xlabel("Component", fontsize=11)
    ax.set_ylabel("Explained Variance (%)", fontsize=11)
    ax.set_xticks(components)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3, zorder=0)

    ax2 = ax.twinx()
    cumulative_line = ax2.plot(
        components,
        cumulative,
        color=WHOLE_SPECTRA_NEUTRAL,
        marker="o",
        linewidth=1.3,
        label="Cumulative variance",
        zorder=4,
    )[0]
    ax2.patch.set_visible(False)
    ax2.grid(False)
    ax2.set_ylabel("Cumulative Variance (%)", fontsize=11)
    ax2.set_ylim(0, max(100.0, float(cumulative.max()) * 1.05))

    ax.legend(
        handles=[
            Patch(
                facecolor=WHOLE_SPECTRA_RANGE_COLORS["fingerprint_1300"],
                edgecolor="white",
                label=WHOLE_SPECTRA_RANGE_LABELS["fingerprint_1300"],
            ),
            Patch(
                facecolor=WHOLE_SPECTRA_RANGE_COLORS["ch_stretch_2850"],
                edgecolor="white",
                label=WHOLE_SPECTRA_RANGE_LABELS["ch_stretch_2850"],
            ),
            cumulative_line,
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        frameon=False,
        fontsize=9,
        ncol=3,
    )
    fig.suptitle(title, fontsize=14, y=0.98)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    return fig, [ax, ax2]


def plot_cca_scatter(
    fingerprint_scores: np.ndarray,
    ch_scores: np.ndarray,
    labels: np.ndarray,
    correlations: np.ndarray,
    title: str = "CCA Scatter",
) -> tuple[plt.Figure, plt.Axes]:
    """Scatter first canonical scores from the two regions."""
    fingerprint_scores = np.asarray(fingerprint_scores, dtype=float)
    ch_scores = np.asarray(ch_scores, dtype=float)
    labels = np.asarray(labels)
    correlations = np.asarray(correlations, dtype=float)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for group_name in np.unique(labels):
        mask = labels == group_name
        ax.scatter(
            fingerprint_scores[mask, 0],
            ch_scores[mask, 0],
            color=GROUP_COLORS.get(group_name, "#6B7280"),
            alpha=0.84,
            edgecolors=POINT_EDGE_COLOR,
            linewidths=0.55,
            s=SUBJECT_SCATTER_SIZE,
            label=GROUP_LABELS.get(group_name, group_name),
        )

    cc1 = correlations[0] if correlations.size else np.nan
    ax.set_xlabel("Fingerprint CC1", fontsize=11)
    ax.set_ylabel("CH-stretch CC1", fontsize=11)
    ax.set_title(f"{title} (CC1 r = {cc1:.2f})", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig, ax


def plot_umap_scatter(
    embedding_df: pd.DataFrame,
    color_col: str,
    title: str = "Pixel-Level UMAP",
    palette: Optional[dict] = None,
    continuous: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Scatter plot of a UMAP embedding coloured by a categorical or continuous column.

    Parameters
    ----------
    embedding_df : pd.DataFrame
        Must have ``UMAP1``, ``UMAP2`` columns and a ``hdbscan_cluster`` column.
    color_col : str
        Column to colour by.
    continuous : bool
        When True, use viridis colormap (5th–95th percentile normalisation).
        NaN rows are drawn as open gray markers.  When False (default), use
        categorical palette.
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    if continuous:
        values = pd.to_numeric(embedding_df[color_col], errors="coerce")
        nan_mask = values.isna()
        valid_vals = values[~nan_mask]

        # Open gray markers for NaN subjects (controls without MoCA)
        n_nan = int(nan_mask.sum())
        if n_nan > 0:
            ax.scatter(
                embedding_df.loc[nan_mask, "UMAP1"],
                embedding_df.loc[nan_mask, "UMAP2"],
                s=SUBJECT_SCATTER_SIZE,
                facecolors="none",
                edgecolors="#888888",
                linewidths=1.0,
                alpha=0.7,
                label=f"No {display_label(color_col)} (n={n_nan})",
                zorder=2,
            )

        if not valid_vals.empty:
            vmin = float(np.nanpercentile(valid_vals, 5))
            vmax = float(np.nanpercentile(valid_vals, 95))
            sc = ax.scatter(
                embedding_df.loc[~nan_mask, "UMAP1"],
                embedding_df.loc[~nan_mask, "UMAP2"],
                c=valid_vals.values,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                s=SUBJECT_SCATTER_SIZE,
                alpha=0.85,
                edgecolors=POINT_EDGE_COLOR,
                linewidths=0.3,
                zorder=3,
            )
            plt.colorbar(sc, ax=ax, label=display_label(color_col), shrink=0.8)

        ax.legend(fontsize=9, frameon=False, loc="best")

    else:
        # Noise first (zorder=1)
        if "hdbscan_cluster" in embedding_df.columns:
            noise = embedding_df[embedding_df["hdbscan_cluster"] == -1]
            if not noise.empty:
                ax.scatter(
                    noise["UMAP1"],
                    noise["UMAP2"],
                    c="#cccccc",
                    s=8,
                    alpha=0.4,
                    label="Noise",
                    zorder=1,
                )
            valid_mask = embedding_df["hdbscan_cluster"] != -1
        else:
            valid_mask = pd.Series(True, index=embedding_df.index)

        categories = pd.unique(embedding_df[color_col].dropna())
        if palette is None:
            palette = get_discrete_palette(color_col, categories)

        for cat in sorted(categories, key=lambda value: category_sort_key(color_col, value)):
            mask = (embedding_df[color_col] == cat) & valid_mask
            if mask.any():
                ax.scatter(
                    embedding_df.loc[mask, "UMAP1"],
                    embedding_df.loc[mask, "UMAP2"],
                    c=palette.get(cat, "#666666"),
                    s=15,
                    alpha=0.82,
                    edgecolors=POINT_EDGE_COLOR,
                    linewidths=0.35,
                    label=format_category_label(color_col, cat),
                    zorder=2,
                )

        ax.legend(fontsize=9, frameon=False, loc="best", title=display_label(color_col))

    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig, ax




def plot_longitudinal_trajectories(
    embedding_df: pd.DataFrame,
    n_paired: int,
    title: str = "Longitudinal MoCA Trajectories",
) -> tuple[plt.Figure, plt.Axes]:
    """Overlay small directional arrows on the UMAP embedding to visualise
    longitudinal MoCA change.

    Because spectra are single-timepoint measurements, each subject occupies a
    single UMAP coordinate.  Arrows are drawn horizontally from that coordinate
    with a length proportional to ΔMoCA; they are purely descriptive and do
    not represent a second spectral measurement.

    Parameters
    ----------
    embedding_df : pd.DataFrame
        Must have ``UMAP1``, ``UMAP2``, ``moca_baseline_total``,
        ``moca_3_months_total``, ``delta_moca``.
    n_paired : int
        Number of subjects with both MoCA timepoints (for caption).
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    delta = pd.to_numeric(embedding_df.get("delta_moca", pd.Series(dtype=float)), errors="coerce")
    vabs = float(np.nanmax(np.abs(delta.dropna()))) if not delta.dropna().empty else 1.0
    norm = plt.Normalize(vmin=-vabs, vmax=vabs)
    cmap = plt.cm.RdBu_r

    # Scale arrows to ~5 % of the UMAP x-range so they are visible but not overwhelming
    x_range = float(embedding_df["UMAP1"].max() - embedding_df["UMAP1"].min())
    scale = (0.05 * x_range) / vabs if vabs > 0 else 1.0

    for _, row in embedding_df.iterrows():
        if pd.isna(row.get("delta_moca")):
            continue
        color = cmap(norm(row["delta_moca"]))
        dx = row["delta_moca"] * scale
        ax.annotate(
            "",
            xy=(row["UMAP1"] + dx, row["UMAP2"]),
            xytext=(row["UMAP1"], row["UMAP2"]),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
        )
        ax.scatter(row["UMAP1"], row["UMAP2"], c=[color], s=SUBJECT_SCATTER_SIZE, zorder=3, edgecolors="white", linewidths=0.4)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="ΔMOCA (3m − baseline)", shrink=0.8)

    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title(
        f"{title}\n(n={n_paired} paired subjects; single-timepoint spectra — descriptive only)",
        fontsize=12,
    )
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_umap_axis_top(
    sub_df: pd.DataFrame,
    axis: str = "UMAP1",
    top_k: int = 20,
    title: Optional[str] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Bar plot of top-K wavenumbers correlated with a UMAP axis (Spearman).

    Parameters
    ----------
    sub_df : pd.DataFrame
        Output of ``umap_axis_correlations`` from cluster_interpretation.
        Required columns: ``wavenumber``, ``rho_UMAP1`` / ``rho_UMAP2``,
        ``fdr_UMAP1`` / ``fdr_UMAP2``.
    axis : str
        ``'UMAP1'`` or ``'UMAP2'``.
    top_k : int
        Number of bars to plot.
    """
    rho_col = f"rho_{axis}"
    fdr_col = f"fdr_{axis}"
    if rho_col not in sub_df.columns:
        raise ValueError(f"Column {rho_col!r} not found in sub_df")

    top = sub_df.nlargest(top_k, rho_col, keep="all").head(top_k)
    top = top.sort_values(rho_col, ascending=True)

    fig, ax = plt.subplots(figsize=(7, max(4, top_k * 0.3 + 1)))
    colors = [
        "#CC3311" if (fdr_col in top.columns and row[fdr_col] < 0.05) else "#457B9D"
        for _, row in top.iterrows()
    ]
    ax.barh(top["wavenumber"].apply(lambda w: f"{w:.0f}"), top[rho_col], color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"Spearman ρ (wavenumber ↔ {axis})", fontsize=11)
    ax.set_ylabel("Wavenumber (cm⁻¹)", fontsize=11)
    t = title or f"Top {top_k} wavenumbers correlated with {axis}"
    ax.set_title(f"{t}\n(red = FDR < 0.05; UMAP axes are nonlinear — exploratory only)", fontsize=10)
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig, ax

