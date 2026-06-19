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

_UNKNOWN_COLOR = "#A0AEC0"
POINT_EDGE_COLOR = "#334155"
SUBJECT_SCATTER_SIZE = 45

# Explicit pastel palettes with strong separation between warm/cool families.
# These replace the previous blue/cyan pairing, which was too close in score
# plots. The hues stay colorblind-aware while remaining easy to separate
# on white backgrounds.
CONDITION_COLORS = {
    "disease": pastelize_color("#D55E00", retain=0.74),
    "control": pastelize_color("#009E73", retain=0.72),
    "unknown": _UNKNOWN_COLOR,
}

GENDER_COLORS = {
    "female": pastelize_color("#CC79A7", retain=0.72),
    "male": pastelize_color("#0072B2", retain=0.68),
    "unknown": _UNKNOWN_COLOR,
}

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

# Human-readable display labels for column/axis names
DISPLAY_LABELS: dict[str, str] = {
    "gender": "Gender",
    "condition": "Condition",
    "is_control": "Condition",
    "subject_id": "Subject",
    "moca_baseline_total": "MoCA baseline",
    "moca_3_months_total": "MoCA 3m",
    "delta_moca": "ΔMoCA (3m − baseline)",
    "age_years": "Age (years)",
    "rbans_total_3_months": "RBANS 3m",
    "aspects_scale_value": "ASPECTS",
}


def display_label(col: str) -> str:
    """Return a human-readable display label for a column name."""
    return DISPLAY_LABELS.get(col, col.replace("_", " ").title())


def _normalize_category(value: object) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return "unknown"
    return str(value).strip().lower()


def category_sort_key(color_col: str, category: object) -> tuple[object, ...]:
    col = str(color_col).strip().lower()
    normalized = _normalize_category(category)

    if col in {"condition", "is_control"}:
        order = {"disease": 0, "control": 1, "unknown": 2}
        return (order.get(normalized, 99), normalized)

    if col == "gender":
        order = {"female": 0, "male": 1, "unknown": 2}
        return (order.get(normalized, 99), normalized)

    group_order = {
        "female_disease": 0,
        "female_control": 1,
        "male_disease": 2,
        "male_control": 3,
    }
    if normalized in group_order:
        return (group_order[normalized], normalized)

    return (99, normalized)


def format_category_label(color_col: str, category: object) -> str:
    col = str(color_col).strip().lower()
    normalized = _normalize_category(category)

    if normalized in GROUP_LABELS:
        return GROUP_LABELS[normalized]

    if col in {"condition", "is_control", "gender"}:
        return normalized.capitalize()

    return str(category)


def get_discrete_palette(color_col: str, categories: list[object] | np.ndarray) -> dict[object, str]:
    col = str(color_col).strip().lower()
    categories_list = list(categories)

    if col in {"condition", "is_control"}:
        return {
            category: CONDITION_COLORS.get(_normalize_category(category), _UNKNOWN_COLOR)
            for category in categories_list
        }

    if col == "gender":
        return {
            category: GENDER_COLORS.get(_normalize_category(category), _UNKNOWN_COLOR)
            for category in categories_list
        }

    normalized_categories = {_normalize_category(category) for category in categories_list}
    if normalized_categories and normalized_categories.issubset(GROUP_COLORS.keys()):
        return {
            category: GROUP_COLORS.get(_normalize_category(category), _UNKNOWN_COLOR)
            for category in categories_list
        }

    fallback_cycle = [
        pastelize_color("#CC79A7", retain=0.72),
        pastelize_color("#E69F00", retain=0.74),
        pastelize_color("#009E73", retain=0.74),
        pastelize_color("#4C78A8", retain=0.70),
        pastelize_color("#6D597A", retain=0.72),
        pastelize_color("#D55E00", retain=0.74),
        pastelize_color("#2A9D8F", retain=0.74),
        pastelize_color("#7F5539", retain=0.72),
    ]
    return {
        category: fallback_cycle[i % len(fallback_cycle)]
        for i, category in enumerate(categories_list)
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
