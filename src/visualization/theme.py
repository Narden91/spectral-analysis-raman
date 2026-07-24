"""
Centralized visualization theme for EDA Raman.
Applies 'algorithmic-art' inspired aesthetics to Matplotlib plots.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors


def pastelize_color(color: str, retain: float = 0.58) -> str:
    """Blend a base color with white while keeping its original hue family."""
    rgb = np.asarray(mcolors.to_rgb(color), dtype=float)
    blended = retain * rgb + (1.0 - retain)
    return mcolors.to_hex(np.clip(blended, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Core Color Palettes
# ---------------------------------------------------------------------------

POINT_EDGE_COLOR = "#334155"
SUBJECT_SCATTER_SIZE = 45

# Explicit pastel palettes with strong separation between warm/cool families.
# These replace the previous blue/cyan pairing, which was too close in score
# plots. The hues stay colorblind-aware while remaining easy to separate
# on white backgrounds.
GROUP_COLORS = {
    "female_disease": pastelize_color("#B24C7C", retain=0.73),
    "female_control": pastelize_color("#E69F00", retain=0.74),
    "male_disease": pastelize_color("#009E73", retain=0.74),
    "male_control": pastelize_color("#4C78A8", retain=0.70),
}

# Display labels for groups
GROUP_LABELS = {
    "female_disease": "Female Disease",
    "female_control": "Female Control",
    "male_disease": "Male Disease",
    "male_control": "Male Control",
}

WHOLE_SPECTRA_RANGE_LABELS = {
    "fingerprint_1300": "Fingerprint (1300)",
    "ch_stretch_2850": "CH-stretch (2850)",
}

WHOLE_SPECTRA_RANGE_COLORS = {
    "fingerprint_1300": pastelize_color("#0072B2"),
    "ch_stretch_2850": pastelize_color("#E69F00"),
}

WHOLE_SPECTRA_NEUTRAL = "#4B5563"
WHOLE_SPECTRA_TEXT = "#1F2933"


def finalize_axis(
    ax: plt.Axes,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    xlim: tuple[float, float] | None = None,
    legend: bool = True,
    grid_alpha: float = 0.4,
) -> None:
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if legend:
        ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=grid_alpha)
    ax.figure.tight_layout()

# Cluster Color Palette (Up to 8 clusters, algorithmic art inspired)
CLUSTER_COLORS = [
    "#264653",
    "#2a9d8f",
    "#e9c46a",
    "#f4a261",
    "#e76f51",
    "#6d597a",
    "#b56576",
    "#eaac8b",
]


def set_gallery_theme():
    """
    Sets the global matplotlib rcParams to a clean, gallery-quality theme.
    Emphasizes algorithmic art aesthetics:
    - Clean sans-serif fonts
    - Subtle grid lines
    - High-quality layout proportions
    - Removal of heavy borders (spines)
    """
    mpl.rcParams.update(
        {
            # Figure properties
            "figure.facecolor": "#ffffff",
            "figure.edgecolor": "#ffffff",
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
            # Embed TrueType fonts so vector PDFs satisfy journal submission
            # checks rather than shipping Type 3 outlines.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            # Font properties (use system sans-serif like Arial, Helvetica, or DejaVu)
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "legend.title_fontsize": 11,
            # Axes properties
            "axes.facecolor": "#fafafa",  # Very light gray background
            "axes.edgecolor": "#d3d3d3",  # Subtle gray edges
            "axes.linewidth": 1.0,
            "axes.grid": True,
            "axes.axisbelow": True,  # Grid lines behind plots
            "axes.prop_cycle": mpl.cycler(color=CLUSTER_COLORS),
            # Spines (remove top and right borders)
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            # Grid properties
            "grid.color": "#e0e0e0",
            "grid.linestyle": "--",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.7,
            # Tick properties
            "xtick.color": "#555555",
            "ytick.color": "#555555",
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            # Line properties
            "lines.linewidth": 1.5,
            "lines.markersize": 6,
            "lines.antialiased": True,
            # Legend properties
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.fancybox": True,
            "legend.edgecolor": "#e0e0e0",
        }
    )
