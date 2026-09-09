"""Generate subject-level figures used by the Raman EV manuscript.

The script reads only analysis outputs and writes publication figures plus the
effect-size source table under ``manuscript/paper``.  All resampling is at the
participant level; pixel- and area-level observations are never resampled.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from scipy import stats
from sklearn.metrics import roc_curve

from src.analysis.clinical_correlation_matrix import _plot_correlation_heatmap
from src.analysis.journal_figures import (
    _fig3_difference_spectrum,
    _fig4_nmf_entropy,
    _fig6_moca_association_landscape,
)
from src import load_config
from src.analysis.io_utils import load_csv_or_empty
from src.reporting.candidate_region_report import _classify_candidate
from src.analysis.raman_bands import RAMAN_BANDS
from src.visualization.spectral_summary_plots import (
    BAND_CATEGORY_COLORS,
    BAND_CATEGORY_LABELS,
    _add_raman_band_spans,
    _annotate_peaks,
    _set_spectrum_limits,
    _upper_envelope,
)
from src.visualization.theme import (
    FIGURE_WIDTH_IN,
    MANUSCRIPT_WIDTH_IN,
    panel_width_in,
    set_manuscript_theme,
)
FIGURE_DIR = ROOT / "manuscript" / "paper" / "figures"
FIGURE_DATA_DIR = ROOT / "manuscript" / "paper" / "figure_data"
SUBJECT_SPECTRA = (
    ROOT
    / "Results_control_vs_disease"
    / "04_cross_region_summary"
    / "01_subject_features"
    / "subject_spectra.csv"
)
BAND_SCREEN = (
    ROOT
    / "Results_control_vs_disease"
    / "05_ml_classification"
    / "05_univariate_band_screen"
    / "univariate_band_screen.csv"
)
ML_MODEL_EVAL_DIR = (
    ROOT / "Results_control_vs_disease" / "05_ml_classification" / "02_model_evaluation"
)
ML_MODEL_COMPARISON = ML_MODEL_EVAL_DIR / "model_comparison.csv"
ML_PERMUTATION_NULL = (
    ROOT
    / "Results_control_vs_disease"
    / "05_ml_classification"
    / "04_permutation_tests"
    / "permutation_null_distributions.csv"
)
MOCA_PLS_PERFORMANCE = (
    ROOT
    / "Results_intra_disease"
    / "classification"
    / "pls_regression"
    / "pls_regression_performance.csv"
)
MOCA_PLS_PERMUTATION = (
    ROOT
    / "Results_intra_disease"
    / "classification"
    / "pls_regression"
    / "pls_permutation_results.csv"
)
BIOPCA_WHOLE_REGION_CONTRIBUTIONS = (
    ROOT
    / "Results_control_vs_disease"
    / "03_whole_spectrum_1300_plus_2850"
    / "04_biological_pca"
    / "region_contributions.csv"
)
BIOPCA_WHOLE_EXPLAINED_VARIANCE = (
    ROOT
    / "Results_control_vs_disease"
    / "03_whole_spectrum_1300_plus_2850"
    / "04_biological_pca"
    / "explained_variance.csv"
)
INTEGRATED_CANDIDATE_REGIONS = (
    ROOT
    / "Results_intra_disease"
    / "04_cross_region_summary"
    / "09_integrated_candidates"
    / "integrated_candidate_regions.csv"
)
CLINICAL_CORRELATION_MATRIX = (
    ROOT
    / "Results_intra_disease"
    / "04_cross_region_summary"
    / "03_clinical_correlations"
    / "clinical_correlation_matrix.csv"
)
DIFFERENCE_SPECTRUM = (
    ROOT
    / "Results_intra_disease"
    / "06_spectral_interpretability"
    / "04_discriminant_signatures"
    / "difference_spectrum.csv"
)
PANEL_CONTRAST_TESTS = (
    ROOT
    / "Results_intra_disease"
    / "05_confirmatory_panel"
    / "panel_contrast_tests.csv"
)
NMF_COMPONENTS = (
    ROOT
    / "Results_intra_disease"
    / "06_spectral_interpretability"
    / "02_nmf_decomposition"
    / "nmf_components.csv"
)
SPECTRAL_ENTROPY = (
    ROOT
    / "Results_intra_disease"
    / "06_spectral_interpretability"
    / "03_spectral_entropy"
    / "spectral_entropy.csv"
)
DELTA_MOCA_ASSOCIATIONS = (
    ROOT
    / "Results_intra_disease"
    / "04_cross_region_summary"
    / "04_longitudinal_moca"
    / "delta_moca_associations.csv"
)

CONTROL = "#0072B2"
DISEASE = "#D55E00"
NEUTRAL = "#4D4D4D"
SEED = 20260724
N_BOOTSTRAP = 2_000

CVD_MODEL_LABELS = {
    "pls_da": "PLS-DA (primary)",
    "random_forest": "Random forest",
    "l1_logistic": r"$L_1$ logistic",
}
CVD_MODEL_COLORS = {
    "pls_da": "#0072B2",
    "random_forest": "#009E73",
    "l1_logistic": "#E69F00",
}
MOCA_ENDPOINT_LABELS = {
    "moca_baseline": "Baseline MoCA",
    "moca_3m": "3-month MoCA",
    "delta_moca": r"$\Delta$MoCA",
}

BAND_LABELS = {
    "cytosine_786": "Cytosine (786)",
    "o_p_o_821": "O–P–O (821)",
    "tyrosine_851": "Tyrosine (851)",
    "c_o_c_sugar_925": "C–O–C, sugar (925)",
    "beta_sheet_980": r"$\beta$-sheet (980)",
    "c_o_c_n_1060": "C–O/C–N (1060)",
    "lipid_cc_1089": "Lipid C–C (1089)",
    "po2_cn_1112": r"PO$_2^-$ / C–N (1112)",
    "carotenoid_1156": "Carotenoid (1156)",
    "amide_iii_1258": "Amide III (1258)",
    "ch2_twist_1301": r"CH$_2$ twist (1301)",
    "ch2_ch3_scissor_1440": r"CH$_2$/CH$_3$ scissor (1440)",
    "ch2_ch3_bend_1465": r"CH$_2$/CH$_3$ bend (1465)",
    "carotenoid_1525": "Carotenoid (1525)",
    "tryptophan_1555": "Tryptophan (1555)",
    "amide_i_1657": "Amide I (1657)",
    "lipid_carbonyl_1721": "Lipid carbonyl (1721)",
    "lipid_ester_1744": "Lipid ester (1744)",
    "ch2_sym_stretch_2853": r"CH$_2$ symmetric stretch (2853)",
    "ch2_asym_stretch_2890": r"CH$_2$ asymmetric stretch (2890)",
    "ch3_asym_stretch_2937": r"CH$_3$ asymmetric stretch (2937)",
}

BAND_SHORT_LABELS = {
    "cytosine_786": "Cyt 786",
    "o_p_o_821": "O–P–O 821",
    "tyrosine_851": "Tyr 851",
    "c_o_c_sugar_925": "C–O–C 925",
    "beta_sheet_980": r"$\beta$-sheet 980",
    "c_o_c_n_1060": "C–O/C–N 1060",
    "lipid_cc_1089": "Lip C–C 1089",
    "po2_cn_1112": r"PO$_2^-$ 1112",
    "carotenoid_1156": "Carot 1156",
    "amide_iii_1258": "Am III 1258",
    "ch2_twist_1301": r"CH$_2$ tw 1301",
    "ch2_ch3_scissor_1440": r"CH$_2$/CH$_3$ sc 1440",
    "ch2_ch3_bend_1465": r"CH$_2$/CH$_3$ bd 1465",
    "carotenoid_1525": "Carot 1525",
    "tryptophan_1555": "Trp 1555",
    "amide_i_1657": "Am I 1657",
    "lipid_carbonyl_1721": "Lip C=O 1721",
    "lipid_ester_1744": "Lip ester 1744",
    "ch2_sym_stretch_2853": r"CH$_2$ sym 2853",
    "ch2_asym_stretch_2890": r"CH$_2$ asym 2890",
    "ch3_asym_stretch_2937": r"CH$_3$ asym 2937",
}

BAND_WINDOWS = {
    "cytosine_786": (773, 796),
    "o_p_o_821": (812, 832),
    "tyrosine_851": (836, 874),
    "c_o_c_sugar_925": (900, 938),
    "beta_sheet_980": (965, 995),
    "c_o_c_n_1060": (1044, 1078),
    "lipid_cc_1089": (1082, 1098),
    "po2_cn_1112": (1105, 1120),
    "carotenoid_1156": (1145, 1165),
    "amide_iii_1258": (1240, 1280),
    "ch2_twist_1301": (1290, 1325),
    "ch2_ch3_scissor_1440": (1430, 1445),
    "ch2_ch3_bend_1465": (1450, 1475),
    "carotenoid_1525": (1510, 1535),
    "tryptophan_1555": (1550, 1560),
    "amide_i_1657": (1640, 1675),
    "lipid_carbonyl_1721": (1710, 1730),
    "lipid_ester_1744": (1735, 1755),
    "ch2_sym_stretch_2853": (2845, 2860),
    "ch2_asym_stretch_2890": (2870, 2910),
    "ch3_asym_stretch_2937": (2925, 2945),
}


def bootstrap_mean_interval(
    values: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the mean and participant-bootstrap 95% interval by channel."""
    mean = values.mean(axis=0)
    draws = np.empty((N_BOOTSTRAP, values.shape[1]), dtype=np.float32)
    for draw in range(N_BOOTSTRAP):
        indices = rng.integers(0, values.shape[0], size=values.shape[0])
        draws[draw] = values[indices].mean(axis=0)
    low, high = np.percentile(draws, [2.5, 97.5], axis=0)
    return mean, low, high


def spectral_columns(
    frame: pd.DataFrame, prefix: str
) -> tuple[list[str], np.ndarray]:
    columns = [column for column in frame.columns if column.startswith(prefix)]
    columns.sort(key=lambda column: float(column.split("__", maxsplit=1)[1]))
    wavenumbers = np.asarray(
        [float(column.split("__", maxsplit=1)[1]) for column in columns]
    )
    return columns, wavenumbers


def make_subject_spectra_figure(rng: np.random.Generator) -> None:
    frame = pd.read_csv(SUBJECT_SPECTRA)
    regions = (
        ("1300__", "Fingerprint region", "SNV-normalized intensity"),
        ("2850__", r"C--H stretching region", "Area-normalized intensity"),
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(7.2, 3.35), constrained_layout=True
    )

    legend_handles: list = []
    legend_labels: list[str] = []
    for ax, (prefix, title, ylabel) in zip(axes, regions, strict=True):
        columns, wavenumbers = spectral_columns(frame, prefix)
        for condition, color, label in (
            ("control", CONTROL, "Control"),
            ("disease", DISEASE, "Ischemic stroke"),
        ):
            values = frame.loc[frame["condition"] == condition, columns].to_numpy(
                dtype=float
            )
            mean, low, high = bootstrap_mean_interval(values, rng)
            (line,) = ax.plot(
                wavenumbers,
                mean,
                color=color,
                linewidth=1.25,
            )
            ax.fill_between(
                wavenumbers, low, high, color=color, alpha=0.18, linewidth=0
            )
            if ax is axes[0]:
                legend_handles.append(line)
                legend_labels.append(f"{label} ($n={values.shape[0]}$)")

        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel(r"Raman shift (cm$^{-1}$)")
        ax.set_ylabel(ylabel)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(direction="out", length=3)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=2,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.6,
    )

    for stem in ("subject_level_group_spectra",):
        fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def _manuscript_plot_cfg(max_peaks: int | None = None) -> dict:
    """Band windows and peak-detection settings from the frozen plot profile.

    Read from ``config/plots.yaml`` rather than restated here, so the curated
    21-window registry stays the single source for the figure, the analysis,
    and the band table. ``max_peaks`` narrows the label set for the manuscript
    only: rotated labels are as tall as they are long, so every extra one costs
    vertical space that the printed figure does not have. The full 21-window
    panel is Figure 3 and Table 3; this figure only orients the reader.
    """
    profile = load_config("config/plots.yaml")
    peak_detection = dict(profile["plotting"]["peak_detection"])
    if max_peaks is not None:
        peak_detection["max_peaks"] = max_peaks
    return {
        "bands": {
            name: tuple(bounds)
            for name, bounds in profile["whole_spectra"]["bands"].items()
        },
        "peak_detection": peak_detection,
        "raman_band_style": profile["plotting"]["raman_band_style"],
    }


def _shared_display_scale(traces: list[np.ndarray]) -> list[np.ndarray]:
    """Standardize a block's traces against one shared mean and SD.

    The fingerprint block is SNV-normalized and the C--H block is area-
    normalized, so the two blocks are not on a common intensity scale and the
    display axis has to be standardized per block. Standardizing each group
    separately, as the pipeline's own global plot does, would also rescale the
    control-versus-stroke offset that the figure exists to show; pooling the
    statistics across groups puts the blocks on a comparable scale while
    leaving the between-group difference untouched.
    """
    stacked = np.concatenate([np.asarray(trace, dtype=float) for trace in traces])
    finite = stacked[np.isfinite(stacked)]
    if finite.size == 0:
        return traces
    mean = float(finite.mean())
    sd = float(finite.std())
    if sd < 1e-12:
        return [np.asarray(trace, dtype=float) - mean for trace in traces]
    return [(np.asarray(trace, dtype=float) - mean) / sd for trace in traces]


# Figure 1 is assembled in LaTeX from one PDF per spectral block. The blocks
# carry different normalizations and must not be read across, so they are never
# drawn on a shared axis; keeping them as separate files also lets each be
# placed at the exact width its \includegraphics requests, at scale 1.0.
GLOBAL_PANEL_HEIGHT_IN = 2.1
GLOBAL_MAX_PEAK_LABELS = 6
GLOBAL_PEAK_LABEL_PT = 6.8
GLOBAL_BLOCKS = (
    ("1300__", "global_fingerprint"),
    ("2850__", "global_ch_stretch"),
)


def _fit_annotations(fig, ax, pad_px: float = 4.0, passes: int = 4) -> None:
    """Raise the upper y-limit until no peak label passes the top spine.

    The rotated labels are laid out in display space, so the headroom they need
    depends on the rendered text height and cannot be fixed as a data-space
    padding. Measuring what actually overflows and growing the limit keeps the
    labels intact whichever bands the peak finder selects.
    """
    for _ in range(passes):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        axes_top = ax.get_window_extent(renderer).y1
        overflow = max(
            (text.get_window_extent(renderer).y1 - axes_top for text in ax.texts),
            default=0.0,
        )
        if overflow <= 0.0:
            return
        bottom, top = ax.get_ylim()
        height_px = ax.get_window_extent(renderer).height
        ax.set_ylim(bottom, top + (top - bottom) * (overflow + pad_px) / height_px)


def make_global_spectra_figures() -> None:
    r"""Participant-level group-mean spectra, one standalone PDF per block.

    Per reviewer request the traces are not filled to zero and the intensity
    axis carries no tick labels: the fill is anchored at zero, so its edge
    draws a baseline that invites reading standardized values as absolute
    intensities. Peak labels are clipped to the axes so a tall band cannot push
    its label out of the panel.

    The shared legend rides on the lower (C--H) panel, which is the one placed
    last in the assembled float. Neither panel carries a letter or a block
    name: the assembling float supplies both through ``\subcaption``, so the
    labels stay in the document's type and stay referenceable.
    """
    frame = pd.read_csv(SUBJECT_SPECTRA)
    plot_cfg = _manuscript_plot_cfg(max_peaks=GLOBAL_MAX_PEAK_LABELS)
    conditions = (
        ("control", CONTROL, "Control"),
        ("disease", DISEASE, "Ischemic stroke"),
    )
    categories = list(dict.fromkeys(band.category for band in RAMAN_BANDS.values()))

    for prefix, stem in GLOBAL_BLOCKS:
        carries_legend = stem == GLOBAL_BLOCKS[-1][1]
        height = GLOBAL_PANEL_HEIGHT_IN + (0.75 if carries_legend else 0.0)
        fig, ax = plt.subplots(
            figsize=(FIGURE_WIDTH_IN, height), constrained_layout=True
        )

        columns, wavenumbers = spectral_columns(frame, prefix)
        means = [
            frame.loc[frame["condition"] == condition, columns]
            .to_numpy(dtype=float)
            .mean(axis=0)
            for condition, _, _ in conditions
        ]
        counts = [
            int(frame["condition"].eq(condition).sum())
            for condition, _, _ in conditions
        ]
        display = _shared_display_scale(means)

        trace_handles: list = []
        trace_labels: list[str] = []
        for trace, (_, color, label), count in zip(
            display, conditions, counts, strict=True
        ):
            (line,) = ax.plot(wavenumbers, trace, color=color, linewidth=1.3)
            trace_handles.append(line)
            trace_labels.append(f"{label} ($n={count}$)")

        _set_spectrum_limits(ax, [wavenumbers], display, top_pad=0.9)
        _add_raman_band_spans(ax, wavenumbers, plot_cfg)
        _annotate_peaks(
            ax,
            wavenumbers,
            _upper_envelope(display),
            wavenumbers,
            plot_cfg["peak_detection"],
            bands=plot_cfg["bands"],
            fontsize=GLOBAL_PEAK_LABEL_PT,
            rotation=90.0,
            clip_text=True,
        )
        # _add_raman_band_spans attaches a per-axes legend listing only this
        # block's categories; the assembled figure carries one legend instead.
        if ax.get_legend() is not None:
            ax.get_legend().remove()

        ax.set_xlabel(r"Raman shift (cm$^{-1}$)")
        # "(a.u.)" is redundant here: the caption states the axis is
        # standardized per block and carries no units, and the shorter
        # string fits the panel height without overrunning the canvas.
        ax.set_ylabel("Normalized intensity")
        ax.set_yticks([])
        ax.grid(axis="x", color="#E5E7EB", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="x", direction="out", length=3)

        if carries_legend:
            fig.legend(
                trace_handles
                + [
                    Patch(facecolor=BAND_CATEGORY_COLORS[category], alpha=0.8)
                    for category in categories
                ],
                trace_labels
                + [BAND_CATEGORY_LABELS[category] for category in categories],
                # "outside" is the only placement constrained_layout reserves
                # space for; a negative bbox anchor lands on the x-label.
                loc="outside lower center",
                ncol=3,
                frameon=False,
                handlelength=1.8,
                columnspacing=1.6,
            )

        # Runs last: constrained_layout shrinks the axes when the legend is
        # added, so fitting before that would leave the labels overflowing
        # again.
        _fit_annotations(fig, ax)
        _save_exact(fig, stem)


def bootstrap_rank_biserial(
    disease: np.ndarray, control: np.ndarray, rng: np.random.Generator
) -> tuple[float, float]:
    effects = np.empty(N_BOOTSTRAP, dtype=float)
    for draw in range(N_BOOTSTRAP):
        disease_draw = disease[
            rng.integers(0, disease.size, size=disease.size)
        ]
        control_draw = control[
            rng.integers(0, control.size, size=control.size)
        ]
        statistic = stats.mannwhitneyu(
            disease_draw, control_draw, alternative="two-sided"
        ).statistic
        effects[draw] = 2.0 * statistic / (
            disease_draw.size * control_draw.size
        ) - 1.0
    return tuple(np.percentile(effects, [2.5, 97.5]))


def make_band_effect_figure(rng: np.random.Generator) -> None:
    spectra = pd.read_csv(SUBJECT_SPECTRA)
    screen = pd.read_csv(BAND_SCREEN).set_index("band")
    feature_columns = [
        column
        for column in spectra.columns
        if column.startswith(("1300__", "2850__"))
    ]
    values = pd.DataFrame(index=spectra.index)
    for band, (lower, upper) in BAND_WINDOWS.items():
        block = "1300" if upper < 2000 else "2850"
        columns = [
            column
            for column in feature_columns
            if column.startswith(f"{block}__")
            and lower <= float(column.split("__", maxsplit=1)[1]) <= upper
        ]
        values[band] = spectra[columns].mean(axis=1)
    disease_mask = spectra["condition"].eq("disease")
    rows: list[dict[str, float | str | bool]] = []

    for band, label in BAND_LABELS.items():
        disease = values.loc[disease_mask, band].to_numpy(dtype=float)
        control = values.loc[~disease_mask, band].to_numpy(dtype=float)
        low, high = bootstrap_rank_biserial(disease, control, rng)
        observed = float(screen.loc[band, "effect_size_rank_biserial"])
        direct = 2.0 * stats.mannwhitneyu(
            disease, control, alternative="two-sided"
        ).statistic / (disease.size * control.size) - 1.0
        if not np.isclose(observed, direct):
            raise ValueError(f"Effect orientation mismatch for {band}")
        rows.append(
            {
                "band": band,
                "label": label,
                "rank_biserial": observed,
                "ci_low": low,
                "ci_high": high,
                "fdr_q": float(screen.loc[band, "fdr_q"]),
                "significant_fdr": bool(screen.loc[band, "significant_fdr"]),
                "n_control": int(control.size),
                "n_disease": int(disease.size),
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(
        FIGURE_DATA_DIR / "case_control_band_effects_bootstrap.csv", index=False
    )

    y = np.arange(result.shape[0])
    effects = result["rank_biserial"].to_numpy()
    colors = np.where(effects >= 0, DISEASE, CONTROL)
    fig, ax = plt.subplots(
        figsize=(FIGURE_WIDTH_IN, 6.6), constrained_layout=True
    )
    for index, row in result.iterrows():
        effect = float(row["rank_biserial"])
        ax.errorbar(
            effect,
            index,
            xerr=np.asarray(
                [
                    [effect - float(row["ci_low"])],
                    [float(row["ci_high"]) - effect],
                ]
            ),
            fmt="o",
            color=colors[index],
            ecolor=colors[index],
            markersize=4.0,
            elinewidth=1.1,
            capsize=2.0,
        )
    ax.axvline(0, color=NEUTRAL, linestyle="--", linewidth=0.9)
    ax.set_yticks(y, result["label"])
    ax.invert_yaxis()
    ax.set_xlim(-0.7, 0.7)
    ax.set_xlabel(
        "Rank-biserial correlation\n"
        r"$\leftarrow$ lower in stroke     higher in stroke $\rightarrow$"
    )
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.6)

    for stem in ("case_control_band_effects",):
        fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    make_band_effect_figure_landscape(result)


def make_band_effect_figure_landscape(result: pd.DataFrame) -> None:
    """Landscape variant of the band-effect figure: bands on the x-axis.

    Same data and statistics as ``make_band_effect_figure``; only the axis
    orientation and label rotation change, so the two figures are
    interchangeable evidence, not independent analyses.
    """
    x = np.arange(result.shape[0])
    effects = result["rank_biserial"].to_numpy()
    colors = np.where(effects >= 0, DISEASE, CONTROL)
    fig, ax = plt.subplots(
        figsize=(MANUSCRIPT_WIDTH_IN, 3.3), constrained_layout=True
    )
    for index, row in result.iterrows():
        effect = float(row["rank_biserial"])
        ax.errorbar(
            index,
            effect,
            yerr=np.asarray(
                [
                    [effect - float(row["ci_low"])],
                    [float(row["ci_high"]) - effect],
                ]
            ),
            fmt="o",
            color=colors[index],
            ecolor=colors[index],
            markersize=4.0,
            elinewidth=1.1,
            capsize=2.0,
        )
    ax.axhline(0, color=NEUTRAL, linestyle="--", linewidth=0.9)
    # Full assignments live in Table 3; rotated ticks are as tall as they
    # are long, so abbreviating them is what buys the height reduction.
    ax.set_xticks(
        x,
        [BAND_SHORT_LABELS[band] for band in result["band"]],
        rotation=45,
        ha="right",
        fontsize=7,
    )
    ax.set_xlim(-0.7, result.shape[0] - 0.3)
    ax.set_ylim(-0.7, 0.7)
    # The two-line arrow label is taller than this shorter panel allows;
    # the direction is given in the caption instead.
    ax.set_ylabel("Rank-biserial correlation")
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)

    for stem in ("case_control_band_effects_landscape",):
        fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


PC_COLORS = {"PC1": "#264653", "PC2": "#2A9D8F", "PC3": "#E76F51"}
BLOCK_TAGS = {"1300": "FP", "2850": "CH"}


# Figure 2 is assembled in LaTeX from two separate PDFs so each panel can be
# authored at the exact width its \includegraphics requests. Both are saved at
# BIOPCA_PANEL_HEIGHT_IN with bbox_inches=None, so neither is rescaled on the
# page and the two panels share a type size and a baseline.
# A + gap + B = 0.42 + 0.02 + 0.46 = 0.90 of the text block, matching the
# minipage widths in the manuscript's Figure 2 float.
BIOPCA_PANEL_HEIGHT_IN = 3.0
BIOPCA_SCREE_FRACTION = 0.42
BIOPCA_LOADING_FRACTION = 0.46


def _save_exact(fig, stem: str) -> None:
    """Write a panel at exactly its figure size, with no tight-bbox crop.

    ``savefig(bbox_inches=None)`` falls back to the rcParam, which this project
    sets to "tight"; the crop has to be disabled through the rcParam itself so
    the PDF media box equals the requested width and LaTeX places it at scale
    1.0.
    """
    with matplotlib.rc_context({"savefig.bbox": None}):
        fig.savefig(FIGURE_DIR / f"{stem}.pdf")
        fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=600)
    plt.close(fig)


def make_biopca_scree_figure() -> None:
    """Figure 2A: explained variance of the whole-spectrum joint PCA.

    Reads the archived ``explained_variance.csv``; no PCA refitting. A single
    percentage axis replaces the twin axis of the pipeline's ``plot_scree``:
    both series are percentages, and a twin axis is unreadable in a panel this
    narrow.
    """
    variance = pd.read_csv(BIOPCA_WHOLE_EXPLAINED_VARIANCE).sort_values("PC")
    ratios = variance["explained_variance_ratio"].to_numpy(dtype=float) * 100.0

    fig, ax = plt.subplots(
        figsize=(panel_width_in(BIOPCA_SCREE_FRACTION), BIOPCA_PANEL_HEIGHT_IN),
        constrained_layout=True,
    )
    components = np.arange(1, ratios.size + 1)
    ax.bar(components, ratios, color="#457B9D", width=0.68, label="Individual")
    ax.plot(
        components,
        np.cumsum(ratios),
        "o-",
        color="#CC3311",
        linewidth=1.6,
        markersize=4.5,
        label="Cumulative",
    )
    for component, ratio in zip(components, ratios, strict=True):
        # Inside the bar for PC1/PC2: the cumulative marker sits on the PC1 bar
        # top, so an outside label would collide with it.
        ax.annotate(
            f"{ratio:.1f}",
            (component, ratio),
            textcoords="offset points",
            xytext=(0, -4) if ratio > 12 else (0, 4),
            ha="center",
            va="top" if ratio > 12 else "bottom",
            fontsize=8,
            color="white" if ratio > 12 else NEUTRAL,
        )
    ax.set_xticks(components, [f"PC{index}" for index in components])
    ax.set_ylim(0, 108)
    ax.set_xlabel("Component")
    ax.set_ylabel("Explained variance (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        fontsize=7.5,
        handlelength=1.4,
        borderpad=0.1,
        labelspacing=0.25,
    )
    _save_exact(fig, "biopca_scree")


def make_biopca_loading_figure() -> None:
    """Figure 2B: loading-derived intervals of the whole-spectrum joint PCA.

    Reads the archived ``region_contributions.csv``; no PCA refitting or region
    re-detection. Every detected interval belongs to exactly one component, so
    the chart draws one bar per interval coloured by component rather than a
    PC-grouped matrix whose rows would each be two-thirds empty.
    """
    contributions = pd.read_csv(BIOPCA_WHOLE_REGION_CONTRIBUTIONS)
    contributions["block_tag"] = (
        contributions["spectral_block"].astype(str).map(BLOCK_TAGS)
    )
    bounds = contributions["region_label"].str.extract(
        r"(?P<low>[\d.]+)\D+(?P<high>[\d.]+)\s*cm"
    )
    contributions["compact_label"] = (
        "["
        + contributions["block_tag"]
        + "] "
        + bounds["low"].astype(float).round(0).astype(int).astype(str)
        + "–"
        + bounds["high"].astype(float).round(0).astype(int).astype(str)
    )
    if contributions["compact_label"].duplicated().any():
        raise ValueError("Loading-region labels are not unique across components")
    regions = contributions.sort_values("relative_absolute_contribution")

    fig, ax = plt.subplots(
        figsize=(panel_width_in(BIOPCA_LOADING_FRACTION), BIOPCA_PANEL_HEIGHT_IN),
        constrained_layout=True,
    )
    y = np.arange(regions.shape[0])
    ax.barh(
        y,
        regions["relative_absolute_contribution"].to_numpy(dtype=float),
        height=0.74,
        color=[PC_COLORS[pc] for pc in regions["PC"]],
    )
    ax.set_yticks(y, regions["compact_label"])
    ax.tick_params(axis="y", length=0, labelsize=7, pad=1.5)
    ax.set_ylim(-0.7, regions.shape[0] - 0.3)
    ax.set_xlim(0, None)
    ax.set_xlabel("Relative contribution")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(
        handles=[Patch(facecolor=PC_COLORS[pc], label=pc) for pc in ("PC1", "PC2", "PC3")],
        frameon=False,
        loc="lower right",
        fontsize=7.5,
        handlelength=1.2,
        borderpad=0.1,
        labelspacing=0.25,
    )
    _save_exact(fig, "biopca_loading_regions")


def make_clinical_correlation_heatmap_figure() -> None:
    """Redraw the intra-disease clinical correlation heatmap as a vector PDF.

    Calls the production renderer directly on the already-materialized
    correlation table; matplotlib infers PDF output from the file extension.
    """
    correlations = load_csv_or_empty(CLINICAL_CORRELATION_MATRIX)
    if not correlations.empty:
        # The archived CSV round-trips `spectral_block` as int64; the renderer
        # compares it against the string block labels "1300"/"2850", so it
        # must be cast back to string here (matches the dtype the live
        # pipeline DataFrame carries in-memory before it is ever written out).
        correlations["spectral_block"] = correlations["spectral_block"].astype(str)
    _plot_correlation_heatmap(
        correlations, FIGURE_DIR / "clinical_correlation_heatmap.pdf"
    )


def make_candidate_region_overview_figure() -> None:
    """Redraw the intra-disease candidate-region overview as a vector PDF.

    Reimplements the production ``_plot_overview`` body (which hardcodes a
    PNG output path) against the already-materialized integrated candidate
    table, using the same classification rule, colors, and layout.
    """
    candidates = load_csv_or_empty(INTEGRATED_CANDIDATE_REGIONS)
    if candidates.empty:
        return
    candidates = candidates.copy()
    candidates["report_class"] = candidates.apply(_classify_candidate, axis=1)

    class_order = ("robust", "exploratory", "unstable", "confounded")
    colors = {
        "robust": "#2A9D8F",
        "exploratory": "#E9C46A",
        "unstable": "#F4A261",
        "confounded": "#E76F51",
    }
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=False)
    for axis, block in zip(axes, ("1300", "2850")):
        block_df = candidates.loc[candidates["spectral_block"].astype(str) == block]
        axis.axhline(0.0, color="black", linewidth=0.6)
        for class_name in class_order:
            sub = block_df.loc[block_df["report_class"] == class_name]
            if sub.empty:
                continue
            axis.scatter(
                sub["start_cm-1"].to_numpy(dtype=float),
                sub["max_abs_effect"].to_numpy(dtype=float),
                c=colors[class_name],
                s=60,
                label=class_name,
                edgecolors="white",
                linewidths=0.6,
            )
        axis.set_xlabel("Raman shift (cm⁻¹)")
        axis.set_ylabel("Max |effect|")
        axis.set_title(f"{block} candidate-region overview")
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles, labels, loc="upper right", fontsize=8)
    fig.tight_layout()

    for stem in ("candidate_region_overview",):
        fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def make_difference_spectrum_figure() -> None:
    """Redraw the severity extreme-group difference spectrum as a vector PDF."""
    _fig3_difference_spectrum(
        DIFFERENCE_SPECTRUM,
        FIGURE_DIR / "difference_spectrum_severity_extreme.pdf",
        figsize=(FIGURE_WIDTH_IN, 4.6),
    )


def make_panel_effect_sizes_figure() -> None:
    """Redraw the post-hoc biochemical panel effect-size forest plot as a PDF.

    Reimplements the production ``_plot_effect_sizes`` body (which hardcodes
    a fixed ``output_dir/figures/...png`` path) against the already-
    materialized panel contrast table.
    """
    contrasts = load_csv_or_empty(PANEL_CONTRAST_TESTS)
    ok = contrasts.loc[contrasts.get("status", "") == "ok"].copy()
    if ok.empty or "cliffs_delta" not in ok.columns:
        return
    features = ok["feature_name"].astype(str).str.replace("_", " ", regex=False)
    # Every row shares one contrast, so prefixing each label with it repeats
    # the same string ten times and squeezes out the feature names.
    contrasts_present = ok["contrast_name"].astype(str).nunique()
    ok["label"] = (
        features
        if contrasts_present == 1
        else ok["contrast_name"].astype(str) + " | " + features
    )
    ok = ok.sort_values("cliffs_delta")

    fig, ax = plt.subplots(
        figsize=(FIGURE_WIDTH_IN, max(3.2, 0.30 * len(ok))),
        constrained_layout=True,
    )
    y = np.arange(len(ok))
    xerr = np.vstack(
        [
            ok["cliffs_delta"].to_numpy(dtype=float) - ok["bootstrap_ci_lower"].to_numpy(dtype=float),
            ok["bootstrap_ci_upper"].to_numpy(dtype=float) - ok["cliffs_delta"].to_numpy(dtype=float),
        ]
    )
    ax.errorbar(ok["cliffs_delta"], y, xerr=xerr, fmt="o", color="#0072B2", ecolor="#999999", capsize=3)
    ax.axvline(0.0, color="#333333", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(ok["label"])
    ax.set_xlabel("Cliff's delta (group A - group B)")
    ax.set_title("Fixed post hoc panel effect sizes", loc="left", fontweight="bold")

    for stem in ("panel_effect_sizes",):
        fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def make_nmf_entropy_figure() -> None:
    """Redraw the NMF/spectral-entropy robustness-check figure as a vector PDF."""
    _fig4_nmf_entropy(
        NMF_COMPONENTS, SPECTRAL_ENTROPY, FIGURE_DIR / "nmf_components_entropy.pdf"
    )


def make_moca_landscape_figure() -> None:
    """Redraw the within-stroke MoCA association-landscape figure as a PDF."""
    _fig6_moca_association_landscape(
        DELTA_MOCA_ASSOCIATIONS,
        MOCA_PLS_PERFORMANCE,
        FIGURE_DIR / "exploratory_moca_association_landscape.pdf",
    )


# Figure 5 is assembled in LaTeX from one PDF per panel, at 0.48 + 0.02 +
# 0.48 of the text block per row. No panel carries a letter: the float
# supplies them through \subcaption, so they match the document font and
# stay referenceable.
CLINICAL_PANEL_FRACTION = 0.48
CLINICAL_PANEL_HEIGHT_IN = 2.0

CLINICAL_ENDPOINT_LABELS = {
    "moca_baseline": "MoCA base",
    "moca_3m": "MoCA 3-mo",
    "delta_moca": r"$\Delta$MoCA",
    "rbans_total": "RBANS",
    "aspects_score": "ASPECTS",
    "age": "Age",
}


def _clinical_panel_axes():
    """One axes sized for a quarter of the assembled Figure 5 float."""
    return plt.subplots(
        figsize=(
            panel_width_in(CLINICAL_PANEL_FRACTION),
            CLINICAL_PANEL_HEIGHT_IN,
        ),
        constrained_layout=True,
    )


def _clear_band_for_legend(ax, side: str, fraction: float) -> None:
    """Extend one y-limit so a legend has empty axes space to sit in.

    A legend placed over a scatter or a violin hides data, and a transparent
    one reads as if the markers belong to the plot; growing the axis first
    keeps it opaque and clear of everything drawn.
    """
    bottom, top = ax.get_ylim()
    span = top - bottom
    if side == "top":
        ax.set_ylim(bottom, top + fraction * span)
    else:
        ax.set_ylim(bottom - fraction * span, top)


def make_clinical_null_figures() -> None:
    """The four within-stroke null panels, one standalone PDF each.

    Each panel supports the same negative conclusion, which is why they are
    assembled into a single float rather than published separately; every value
    is read from the same archived table used by the corresponding standalone
    analysis output, and nothing is recomputed here.
    """
    candidates = load_csv_or_empty(INTEGRATED_CANDIDATE_REGIONS)
    correlations = load_csv_or_empty(CLINICAL_CORRELATION_MATRIX)
    performance = pd.read_csv(MOCA_PLS_PERFORMANCE).set_index("target")
    permutation = pd.read_csv(MOCA_PLS_PERMUTATION).set_index("target")
    entropy = load_csv_or_empty(SPECTRAL_ENTROPY)
    if candidates.empty or correlations.empty:
        return

    # Candidate-region screen: every region is classified "unstable".
    fig, ax = _clinical_panel_axes()
    candidates = candidates.assign(
        report_class=candidates.apply(_classify_candidate, axis=1)
    )
    for block, marker in (("1300", "o"), ("2850", "^")):
        block_df = candidates.loc[candidates["spectral_block"].astype(str) == block]
        ax.scatter(
            block_df["start_cm-1"].to_numpy(dtype=float),
            block_df["max_abs_effect"].to_numpy(dtype=float),
            s=12,
            marker=marker,
            color="#F4A261",
            edgecolors="white",
            linewidths=0.3,
            label=f"{block} block",
        )
    ax.set_xlabel(r"Raman shift (cm$^{-1}$)")
    ax.set_ylabel(r"Max $|$effect$|$")
    ax.spines[["top", "right"]].set_visible(False)
    _clear_band_for_legend(ax, "top", 0.30)
    ax.legend(
        loc="upper left",
        fontsize=7,
        frameon=True,
        framealpha=1.0,
        edgecolor="none",
        borderpad=0.3,
        handletextpad=0.4,
    )
    _save_exact(fig, "clinical_null_candidates")

    # Channel-wise correlation surface, fingerprint block only.
    fig, ax = _clinical_panel_axes()
    fingerprint = correlations.loc[correlations["spectral_block"] == 1300]
    surface = fingerprint.pivot_table(index="target", columns="wavenumber", values="rho")
    image = ax.imshow(
        surface.to_numpy(),
        aspect="auto",
        cmap="RdBu_r",
        vmin=-0.4,
        vmax=0.4,
        extent=(
            float(surface.columns.min()),
            float(surface.columns.max()),
            0,
            len(surface),
        ),
    )
    ax.set_yticks(np.arange(len(surface)) + 0.5)
    ax.set_yticklabels(
        [CLINICAL_ENDPOINT_LABELS.get(name, name) for name in surface.index],
        fontsize=7,
    )
    ax.set_xlabel(r"Raman shift (cm$^{-1}$)")
    fig.colorbar(
        image, ax=ax, fraction=0.046, pad=0.02, label=r"Spearman $\rho$"
    )
    _save_exact(fig, "clinical_null_channelwise")

    # Observed CV R^2 against each endpoint's own permutation null.
    fig, ax = _clinical_panel_axes()
    rng = np.random.default_rng(SEED)
    endpoints = list(MOCA_ENDPOINT_LABELS)
    for position, endpoint in enumerate(endpoints):
        draws = rng.normal(
            permutation.loc[endpoint, "null_distribution_mean"],
            permutation.loc[endpoint, "null_distribution_std"],
            size=2000,
        )
        violin = ax.violinplot(draws, positions=[position], widths=0.7, showextrema=False)
        for body in violin["bodies"]:
            body.set_facecolor("#B0B0B0")
            body.set_alpha(0.55)
            body.set_edgecolor("none")
        ax.scatter(
            [position],
            [performance.loc[endpoint, "r2"]],
            color=DISEASE,
            s=26,
            zorder=5,
            label="Observed" if position == 0 else None,
        )
    ax.axhline(0, color=NEUTRAL, linestyle="--", linewidth=0.9)
    ax.set_xticks(range(len(endpoints)))
    ax.set_xticklabels(
        [CLINICAL_ENDPOINT_LABELS[endpoint] for endpoint in endpoints], fontsize=7
    )
    ax.set_ylabel("Cross-validated $R^2$")
    ax.spines[["top", "right"]].set_visible(False)
    _clear_band_for_legend(ax, "bottom", 0.24)
    ax.legend(
        loc="lower left",
        fontsize=7,
        frameon=True,
        framealpha=1.0,
        edgecolor="none",
        borderpad=0.3,
        handletextpad=0.4,
    )
    _save_exact(fig, "clinical_null_moca")

    # Unsupervised descriptor distribution, fingerprint block.
    fig, ax = _clinical_panel_axes()
    values = entropy.loc[entropy["spectral_block"] == 1300, "spectral_entropy"].dropna()
    ax.hist(values, bins=22, color="#4D4D4D", alpha=0.75, edgecolor="white")
    ax.set_xlabel("Spectral entropy (fingerprint)")
    ax.set_ylabel("Participants")
    ax.spines[["top", "right"]].set_visible(False)
    _save_exact(fig, "clinical_null_entropy")


def make_graphical_abstract() -> None:
    """Create a data-backed graphical abstract for journal submission."""
    spectra = pd.read_csv(SUBJECT_SPECTRA)
    columns, wavenumbers = spectral_columns(spectra, "1300__")
    control_mean = spectra.loc[
        spectra["condition"].eq("control"), columns
    ].to_numpy(dtype=float).mean(axis=0)
    disease_mean = spectra.loc[
        spectra["condition"].eq("disease"), columns
    ].to_numpy(dtype=float).mean(axis=0)

    fig = plt.figure(figsize=(13.28, 5.31), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.945,
        "Subject-level Raman analysis separates spectral structure,"
        " group discrimination, and clinical prediction",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color="#222222",
    )

    panels = (
        (0.025, 0.12, 0.29, 0.75, "#EAF3F8"),
        (0.355, 0.12, 0.29, 0.75, "#FDF1E8"),
        (0.685, 0.12, 0.29, 0.75, "#F0F0F0"),
    )
    for x, y, width, height, face in panels:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.012,rounding_size=0.018",
                linewidth=1.2,
                edgecolor="#666666",
                facecolor=face,
            )
        )
    for start, end in ((0.318, 0.352), (0.648, 0.682)):
        ax.add_patch(
            FancyArrowPatch(
                (start, 0.50),
                (end, 0.50),
                arrowstyle="-|>",
                mutation_scale=20,
                linewidth=1.5,
                color="#555555",
            )
        )

    ax.text(
        0.17,
        0.81,
        "Participant is the inferential unit",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
    )
    ax.text(
        0.17,
        0.72,
        "50 controls  |  97 ischemic stroke",
        ha="center",
        va="center",
        fontsize=12,
    )
    spectrum_ax = fig.add_axes([0.06, 0.36, 0.22, 0.27], facecolor="none")
    spectrum_ax.plot(wavenumbers, control_mean, color=CONTROL, linewidth=1.5)
    spectrum_ax.plot(wavenumbers, disease_mean, color=DISEASE, linewidth=1.5)
    spectrum_ax.set_xlabel(r"Raman shift (cm$^{-1}$)", fontsize=9)
    spectrum_ax.set_ylabel("SNV value", fontsize=9)
    spectrum_ax.tick_params(labelsize=8, length=2)
    spectrum_ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.17,
        0.245,
        "Pixels → measurement median\n→ participant median",
        ha="center",
        va="center",
        fontsize=13,
        linespacing=1.4,
    )
    ax.text(
        0.17,
        0.16,
        "No pixel-level pseudoreplication",
        ha="center",
        va="center",
        fontsize=11,
        color="#333333",
    )

    ax.text(
        0.50,
        0.81,
        "What the cohort supports",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
    )
    ax.text(
        0.50,
        0.66,
        "Broad lipid-associated variation",
        ha="center",
        va="center",
        fontsize=14,
        color="#8C4A00",
        fontweight="bold",
    )
    ax.text(
        0.50,
        0.585,
        "PC1: 75.8% fingerprint  |  67.1% C–H",
        ha="center",
        va="center",
        fontsize=12,
    )
    ax.plot([0.395, 0.605], [0.50, 0.50], color="#AAAAAA", linewidth=1)
    ax.text(
        0.50,
        0.425,
        "Modest case–control separation",
        ha="center",
        va="center",
        fontsize=14,
        color="#8C4A00",
        fontweight="bold",
    )
    ax.text(
        0.50,
        0.34,
        "PLS-DA balanced accuracy 0.655\n"
        "95% CI 0.538–0.771",
        ha="center",
        va="center",
        fontsize=12,
        linespacing=1.35,
    )
    ax.text(
        0.50,
        0.19,
        "Exploratory and residually confounded",
        ha="center",
        va="center",
        fontsize=11,
        color="#333333",
    )

    ax.text(
        0.83,
        0.81,
        "Current translational boundary",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
    )
    ax.text(
        0.83,
        0.64,
        "0 / 1802",
        ha="center",
        va="center",
        fontsize=30,
        color="#555555",
        fontweight="bold",
    )
    ax.text(
        0.83,
        0.55,
        "clinical channels FDR-supported",
        ha="center",
        va="center",
        fontsize=12,
    )
    ax.text(
        0.83,
        0.41,
        r"All MoCA models: CV $R^2 < 0$",
        ha="center",
        va="center",
        fontsize=15,
        color="#555555",
        fontweight="bold",
    )
    ax.text(
        0.83,
        0.27,
        "Not a diagnostic or prognostic biomarker",
        ha="center",
        va="center",
        fontsize=12,
    )
    ax.text(
        0.83,
        0.17,
        "Next: balanced external validation\n"
        "and EV-subpopulation-resolved assays",
        ha="center",
        va="center",
        fontsize=11,
        linespacing=1.35,
    )

    output = ROOT / "manuscript" / "paper" / "graphical_abstract"
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_cvd_classifier_figure() -> None:
    """ROC curves (whole-spectrum models) and the primary permutation null.

    Sources only saved repeated-nested-CV out-of-fold predictions and the
    already-computed sex-stratified permutation null; no predictions are
    re-derived.
    """
    comparison = pd.read_csv(ML_MODEL_COMPARISON)
    comparison = comparison[comparison["representation"] == "whole"].set_index("model")

    fig, (ax_roc, ax_null) = plt.subplots(
        1, 2, figsize=(MANUSCRIPT_WIDTH_IN, 3.0), constrained_layout=True
    )

    mean_fpr = np.linspace(0, 1, 200)
    for model in ("pls_da", "random_forest", "l1_logistic"):
        predictions = pd.read_csv(
            ML_MODEL_EVAL_DIR / f"{model}_whole_cv_predictions.csv"
        )
        tprs = []
        for _, repeat_frame in predictions.groupby("repeat"):
            fpr, tpr, _ = roc_curve(
                repeat_frame["y_true"], repeat_frame["y_score"]
            )
            tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs = np.asarray(tprs)
        tprs[:, 0] = 0.0
        mean_tpr = tprs.mean(axis=0)
        mean_tpr[-1] = 1.0
        auc_mean = comparison.loc[model, "roc_auc_mean"]
        auc_low = comparison.loc[model, "bootstrap_roc_auc_ci_low"]
        auc_high = comparison.loc[model, "bootstrap_roc_auc_ci_high"]
        color = CVD_MODEL_COLORS[model]
        ax_roc.plot(
            mean_fpr,
            mean_tpr,
            color=color,
            linewidth=1.4,
            label=CVD_MODEL_LABELS[model],
        )
        if model == "pls_da":
            se_tpr = tprs.std(axis=0, ddof=1) / np.sqrt(tprs.shape[0])
            ax_roc.fill_between(
                mean_fpr,
                np.clip(mean_tpr - 1.96 * se_tpr, 0, 1),
                np.clip(mean_tpr + 1.96 * se_tpr, 0, 1),
                color=color,
                alpha=0.15,
                linewidth=0,
            )

    ax_roc.plot([0, 1], [0, 1], color=NEUTRAL, linestyle="--", linewidth=0.9,
                label="Chance")
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_xlim(-0.02, 1.02)
    ax_roc.set_ylim(-0.02, 1.02)
    ax_roc.spines[["top", "right"]].set_visible(False)
    ax_roc.tick_params(direction="out", length=3)
    ax_roc.legend(frameon=False, loc="lower right", fontsize=7.5, labelspacing=0.4)

    null = pd.read_csv(ML_PERMUTATION_NULL)
    primary_null = null[
        (null["model"] == "pls_da")
        & (null["representation"] == "whole")
        & (null["null_type"] == "sex_stratified")
    ]["null_balanced_accuracy"].to_numpy()
    observed = float(comparison.loc["pls_da", "balanced_accuracy_mean"])
    p_value = 0.005
    ax_null.hist(
        primary_null, bins=22, color="#B0B0B0", edgecolor="white", linewidth=0.4
    )
    ax_null.axvline(
        observed, color=CVD_MODEL_COLORS["pls_da"], linewidth=1.6,
        label=f"Observed ({observed:.3f})",
    )
    ax_null.set_xlabel("Balanced accuracy\n(sex-stratified permuted labels)")
    ax_null.set_ylabel("Permutation draws")
    ax_null.spines[["top", "right"]].set_visible(False)
    ax_null.tick_params(direction="out", length=3)
    ax_null.legend(frameon=False, loc="upper left", fontsize=7.5)

    for stem in ("ml_classification_performance",):
        fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def make_moca_prediction_figure() -> None:
    """Observed CV R^2 against its permutation null for each MoCA endpoint.

    Sources only the saved PLS-regression performance and permutation-null
    summary tables; no fold-level predictions are re-derived because none
    were persisted for these unsupported models.
    """
    performance = pd.read_csv(MOCA_PLS_PERFORMANCE).set_index("target")
    permutation = pd.read_csv(MOCA_PLS_PERMUTATION).set_index("target")

    endpoints = list(MOCA_ENDPOINT_LABELS)
    fig, ax = plt.subplots(figsize=(4.6, 3.15), constrained_layout=True)

    rng = np.random.default_rng(SEED)
    for position, endpoint in enumerate(endpoints):
        null_mean = permutation.loc[endpoint, "null_distribution_mean"]
        null_std = permutation.loc[endpoint, "null_distribution_std"]
        null_draws = rng.normal(null_mean, null_std, size=2000)
        violin = ax.violinplot(
            null_draws, positions=[position], widths=0.7, showextrema=False
        )
        for body in violin["bodies"]:
            body.set_facecolor("#B0B0B0")
            body.set_alpha(0.55)
            body.set_edgecolor("none")
        observed = performance.loc[endpoint, "r2"]
        p_holm = performance.loc[endpoint, "permutation_p_value_holm"]
        ax.scatter(
            [position], [observed], color=DISEASE, s=42, zorder=5,
            label="Observed CV $R^2$" if position == 0 else None,
        )
        ax.annotate(
            f"$p_{{\\mathrm{{Holm}}}}={p_holm:.3f}$",
            (position, observed),
            textcoords="offset points",
            xytext=(0, -13),
            ha="center",
            va="top",
            fontsize=6.8,
        )

    ax.axhline(0, color=NEUTRAL, linestyle="--", linewidth=0.9)
    ax.set_xticks(range(len(endpoints)))
    ax.set_xticklabels([MOCA_ENDPOINT_LABELS[endpoint] for endpoint in endpoints])
    ax.set_ylabel("Cross-validated $R^2$")
    ax.set_title(
        "Within-stroke MoCA prediction vs. permutation null", loc="left",
        fontweight="bold", fontsize=8.5,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=3)
    ax.legend(frameon=False, loc="lower left", fontsize=7.2)

    for stem in ("intra_disease_moca_prediction",):
        fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    set_manuscript_theme()
    rng = np.random.default_rng(SEED)
    make_subject_spectra_figure(rng)
    make_global_spectra_figures()
    make_band_effect_figure(rng)
    make_biopca_scree_figure()
    make_biopca_loading_figure()
    make_clinical_correlation_heatmap_figure()
    make_candidate_region_overview_figure()
    make_difference_spectrum_figure()
    make_panel_effect_sizes_figure()
    make_nmf_entropy_figure()
    make_moca_landscape_figure()
    make_cvd_classifier_figure()
    make_moca_prediction_figure()
    make_clinical_null_figures()
    make_graphical_abstract()


if __name__ == "__main__":
    main()
