"""
Advanced visualization tools for Raman spectroscopy data.

Includes Raincloud plots for deep distribution analysis and manifold learning visualizations.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.visualization.theme import GROUP_COLORS, GROUP_LABELS

logger = logging.getLogger(__name__)


def plot_raincloud(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    title: str = "Raincloud Plot",
    palette: dict = GROUP_COLORS,
    ax: plt.Axes = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Creates a Raincloud plot (combination of Half-Violin, Box-plot, and Jittered Stripplot).

    This visualization is highly effective for scientific data as it shows the RAW data,
    the summary statistics (box), and the full density distribution (half-violin)
    simultaneously, revealing bimodal distributions or outliers.

    Parameters
    ----------
    df : pd.DataFrame
    group_col : str
        Column name for group labels.
    value_col : str
        Column name for intensities/values.
    title : str
    palette : dict
    ax : plt.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    # Sort labels for consistent ordering
    unique_groups = sorted(df[group_col].unique())
    group_labels = [GROUP_LABELS.get(g, g) for g in unique_groups]

    # 1. Half-Violin (Kernel Density Estimate)
    # Using 'split=True' and 'inner=None' and manual clipping to get "half"
    # Note: Modern Seaborn doesn't have a direct "Half" violin.
    # We use 'violplot' from seaborn but customize it.

    sns.violinplot(
        data=df,
        x=group_col,
        y=value_col,
        hue=group_col,
        palette=palette,
        split=False,
        inner=None,
        alpha=0.3,
        ax=ax,
        order=unique_groups,
        legend=False,
    )

    # 2. Boxplot (Summary Stats)
    sns.boxplot(
        data=df,
        x=group_col,
        y=value_col,
        hue=group_col,
        palette=palette,
        width=0.15,
        zorder=10,
        showcaps=True,
        boxprops={"zorder": 10, "alpha": 0.8},
        showfliers=False,
        whiskerprops={"linewidth": 1, "alpha": 0.8},
        ax=ax,
        order=unique_groups,
        legend=False,
    )

    # 3. Jittered Strip Plot (Raw Data Points)
    sns.stripplot(
        data=df,
        x=group_col,
        y=value_col,
        hue=group_col,
        palette=palette,
        alpha=0.4,
        size=3,
        jitter=0.1,
        dodge=False,
        ax=ax,
        order=unique_groups,
        legend=False,
    )

    ax.set_title(title, fontsize=14)
    ax.set_xticks(range(len(unique_groups)))
    ax.set_xticklabels(group_labels, rotation=45, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel("Intensity (a.u.)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    return fig, ax


def plot_all_bands_raincloud(
    band_df: pd.DataFrame,
    group_labels: np.ndarray,
    title: str = "Raman Band Raincloud Distributions",
) -> tuple[plt.Figure, list[plt.Axes]]:
    """
    Generate a panel of Raincloud plots for all identified Raman bands.
    """
    bands = [c for c in band_df.columns if c != "group"]
    n_bands = len(bands)
    ncols = 2
    nrows = (n_bands + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes_flat = np.atleast_1d(axes).flatten()

    # Prepare tidy dataframe for Seaborn
    tidy_df = band_df.copy()
    tidy_df["group"] = group_labels

    for i, band in enumerate(bands):
        plot_raincloud(
            tidy_df, group_col="group", value_col=band, title=band, ax=axes_flat[i]
        )

    # Clean up empty subplots
    for j in range(n_bands, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(title, fontsize=16, y=1.02)
    fig.tight_layout()

    return fig, axes_flat.tolist()
