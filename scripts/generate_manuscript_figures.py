"""Generate subject-level figures used by the Raman EV manuscript.

The script reads only analysis outputs and writes publication figures plus the
effect-size source table under ``manuscript/paper``.  All resampling is at the
participant level; pixel- and area-level observations are never resampled.
"""

from __future__ import annotations

# ruff: noqa: E402
# Direct execution needs project imports after path and plotting-backend setup.
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

from src import load_config
from src.analysis.io_utils import load_csv_or_empty
from src.analysis.journal_figures import _plot_block_panel
from src.analysis.raman_bands import RAMAN_BANDS
from src.reporting.candidate_region_report import _classify_candidate
from src.visualization.spectral_summary_plots import (
    BAND_CATEGORY_COLORS,
    BAND_CATEGORY_LABELS,
    CONDITION_COLORS,
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
BIOPCA_WHOLE_LOADINGS = (
    ROOT
    / "Results_control_vs_disease"
    / "03_whole_spectrum_1300_plus_2850"
    / "04_biological_pca"
    / "loadings.csv"
)
BIOPCA_WHOLE_LOADING_REGIONS = (
    ROOT
    / "Results_control_vs_disease"
    / "03_whole_spectrum_1300_plus_2850"
    / "04_biological_pca"
    / "loading_regions.csv"
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
SPECTRAL_ENTROPY = (
    ROOT
    / "Results_intra_disease"
    / "06_spectral_interpretability"
    / "03_spectral_entropy"
    / "spectral_entropy.csv"
)

CONTROL = CONDITION_COLORS["control"]
DISEASE = CONDITION_COLORS["disease"]
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

BAND_WINDOWS = {key: band.bounds for key, band in RAMAN_BANDS.items()}


def advance_rng_past_subject_spectra(rng: np.random.Generator) -> None:
    """Replay the draws of the removed subject-spectra bootstrap figure.

    That figure ran before the band-effect bootstrap on the same ``rng``. The
    published ``case_control_band_effects_bootstrap.csv`` depends on the
    stream position, so the removed figure's calls are repeated with the same
    order, bounds, and sizes: 2 regions x (control, disease) x N_BOOTSTRAP.
    The values are discarded; only the generator state matters.
    """
    condition = pd.read_csv(SUBJECT_SPECTRA, usecols=["condition"])["condition"]
    for _region in ("1300", "2850"):
        for group in ("control", "disease"):
            n_subjects = int(condition.eq(group).sum())
            for _ in range(N_BOOTSTRAP):
                rng.integers(0, n_subjects, size=n_subjects)


def spectral_columns(
    frame: pd.DataFrame, prefix: str
) -> tuple[list[str], np.ndarray]:
    columns = [column for column in frame.columns if column.startswith(prefix)]
    columns.sort(key=lambda column: float(column.split("__", maxsplit=1)[1]))
    wavenumbers = np.asarray(
        [float(column.split("__", maxsplit=1)[1]) for column in columns]
    )
    return columns, wavenumbers


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
# A rotated label is as tall as the label is long, so every one of them adds
# headroom that the traces then have to share. Labels are drawn upright with an
# arrow to their peak instead, which costs a single line each. The exception is
# the 925 band: it sits between the 851 and 1060 peaks, close enough that an
# upright label would run into both of their labels, so it stays rotated in its
# own narrow column.
GLOBAL_ROTATED_PEAK_LABELS = frozenset({"c_o_c_sugar_925"})
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

        # Only the one rotated label needs real headroom now, and
        # _fit_annotations grows the axis to whatever the labels actually
        # measure; reserving more up front just pushes the traces down.
        _set_spectrum_limits(ax, [wavenumbers], display, top_pad=0.18)
        _add_raman_band_spans(ax, wavenumbers, plot_cfg)
        # _add_raman_band_spans attaches a per-axes legend listing only this
        # block's categories; the assembled figure carries one legend instead.
        # Removed before annotating, not after: _annotate_peaks lowers its
        # label ceiling to clear a legend, and on a tight axis that ceiling
        # would drag a label down onto the trace.
        if ax.get_legend() is not None:
            ax.get_legend().remove()
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
            horizontal_bands=frozenset(plot_cfg["bands"]) - GLOBAL_ROTATED_PEAK_LABELS,
        )

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

    make_band_effect_figure_landscape(result)


def make_band_effect_figure_landscape(result: pd.DataFrame) -> None:
    """Band-effect figure with bands on the x-axis.

    Plots the bootstrap table that ``make_band_effect_figure`` writes, so the
    figure and the CSV always show the same estimates.
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
BIOPCA_BLOCK_TITLES = {"1300": "Fingerprint", "2850": "C–H stretch"}


# Figure 2 is authored at the exact width its \includegraphics requests and
# saved with bbox_inches=None, so it is not rescaled on the page.
BIOPCA_LOADING_FRACTION = 0.90
BIOPCA_LOADING_HEIGHT_IN = 3.1
BIOPCA_LABEL_MIN_GAP_CM = 60.0


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


def make_biopca_loading_figure() -> None:
    """Figure 2: loading spectra of the whole-spectrum joint PCA.

    Reads the archived ``loadings.csv`` and ``loading_regions.csv``; no PCA
    refitting or region re-detection. One row per component, one column per
    block. Each row shares its y-axis across blocks because MFA weighting puts
    both blocks in one space, which is what makes their magnitudes comparable.
    Shaded spans are the archived high-loading intervals (SI Section S2).
    """
    loadings = pd.read_csv(BIOPCA_WHOLE_LOADINGS)
    loadings["spectral_block"] = loadings["spectral_block"].astype(str)
    regions = pd.read_csv(BIOPCA_WHOLE_LOADING_REGIONS)
    regions["spectral_block"] = regions["spectral_block"].astype(str)
    variance = pd.read_csv(BIOPCA_WHOLE_EXPLAINED_VARIANCE).set_index("PC")
    blocks = list(BIOPCA_BLOCK_TITLES)
    axes_cm = {
        block: loadings.loc[loadings["spectral_block"].eq(block), "raman_shift"].to_numpy(dtype=float)
        for block in blocks
    }
    pcs = ("PC1", "PC2", "PC3")

    fig, axes = plt.subplots(
        len(pcs),
        len(blocks),
        figsize=(panel_width_in(BIOPCA_LOADING_FRACTION), BIOPCA_LOADING_HEIGHT_IN),
        sharex="col",
        sharey="row",
        gridspec_kw={"width_ratios": [np.ptp(axes_cm[block]) for block in blocks]},
        constrained_layout=True,
    )
    for row, pc in enumerate(pcs):
        color = PC_COLORS[pc]
        for col, block in enumerate(blocks):
            ax = axes[row, col]
            values = loadings.loc[loadings["spectral_block"].eq(block), f"{pc}_loading"].to_numpy(dtype=float)
            ax.axhline(0.0, color=NEUTRAL, linewidth=0.5)
            intervals = regions.loc[regions["PC"].eq(pc) & regions["spectral_block"].eq(block)]
            for _, interval in intervals.iterrows():
                ax.axvspan(
                    interval["start_cm-1"], interval["end_cm-1"], color=color, alpha=0.16, linewidth=0
                )
            ax.plot(axes_cm[block], values, color=color, linewidth=0.9)
            # Adjacent intervals (e.g. 1048/1101) would print overlapping labels;
            # keep the stronger one, the weaker interval stays shaded.
            labelled: list[float] = []
            for _, interval in intervals.sort_values("peak_absolute_loading", ascending=False).iterrows():
                peak_cm = float(interval["peak_shift_cm-1"])
                peak = float(interval["peak_loading"])
                if any(abs(peak_cm - other) < BIOPCA_LABEL_MIN_GAP_CM for other in labelled):
                    continue
                labelled.append(peak_cm)
                ax.annotate(
                    f"{peak_cm:.0f}",
                    (peak_cm, peak),
                    textcoords="offset points",
                    xytext=(0, 1.5 if peak >= 0 else -1.5),
                    ha="center",
                    va="bottom" if peak >= 0 else "top",
                    fontsize=GLOBAL_PEAK_LABEL_PT,
                    color=NEUTRAL,
                )
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=7)
            ax.margins(x=0.01, y=0.30)
            if row == 0:
                ax.set_title(BIOPCA_BLOCK_TITLES[block], fontsize=8)
            if col == 0:
                ratio = float(variance.loc[pc, "explained_variance_ratio"]) * 100.0
                ax.set_ylabel(f"{pc} ({ratio:.1f}%)", fontsize=7.5)
            else:
                ax.tick_params(axis="y", left=False, labelleft=False)
                ax.spines["left"].set_visible(False)
    fig.supxlabel(r"Raman shift (cm$^{-1}$)", fontsize=8)
    fig.supylabel("Loading (MFA-weighted)", fontsize=8)
    _save_exact(fig, "biopca_loadings")


# The SI float assembles the severity contrast from one PDF per panel, two per
# row at 0.48 of the text block, so LaTeX supplies the panel letters.
DIFFERENCE_PANEL_FRACTION = 0.48
DIFFERENCE_PANEL_HEIGHT_IN = 1.9
DIFFERENCE_BLOCK_STEMS = {
    "1300": "difference_spectrum_fingerprint",
    "2850": "difference_spectrum_ch_stretch",
}


def make_difference_spectrum_figure() -> None:
    """The severity extreme-group contrast, one standalone PDF per panel.

    Panel content comes from the production ``_plot_block_panel`` so the SI
    panels stay identical to the pipeline figure; it writes into a (mean,
    effect) axes pair, and the pair here simply lives in two figures instead
    of two rows of one.
    """
    diff = load_csv_or_empty(DIFFERENCE_SPECTRUM)
    if diff.empty or "wavenumber" not in diff.columns or "block" not in diff.columns:
        return
    contrast = (
        diff["contrast_name"].iloc[0] if "contrast_name" in diff.columns else "unknown"
    )

    for block in sorted(diff["block"].unique().tolist()):
        stem = DIFFERENCE_BLOCK_STEMS.get(str(block))
        if stem is None:
            continue
        sub = diff[diff["block"] == block]
        if "contrast_name" in sub.columns:
            sub = sub[sub["contrast_name"] == contrast]
        if sub.empty:
            continue

        fig_mean, ax_mean = _difference_panel_axes()
        fig_effect, ax_effect = _difference_panel_axes()
        _plot_block_panel((ax_mean, ax_effect), sub, block, contrast)
        # _plot_block_panel labels the shared x axis on the effect axes only,
        # which leaves the mean panel unlabelled once the two are separate
        # floats; the block title moves to the subcaption for the same reason.
        ax_mean.set_xlabel("Raman shift (cm⁻¹)")
        ax_mean.set_title("")
        # The two blocks are normalized by different procedures, so a printed
        # intensity number is not comparable across the pair and is not read
        # off these traces anyway; the axis keeps its "a.u." label and drops
        # the scale. The effect-size panels keep their ticks: Cohen's d is
        # read as a magnitude.
        ax_mean.set_yticks([])
        for ax in (ax_mean, ax_effect):
            ax.spines[["top", "right"]].set_visible(False)

        _save_exact(fig_mean, f"{stem}_mean")
        _save_exact(fig_effect, f"{stem}_effect")


def _difference_panel_axes():
    """One axes sized for a quarter of the assembled SI severity float."""
    return plt.subplots(
        figsize=(
            panel_width_in(DIFFERENCE_PANEL_FRACTION),
            DIFFERENCE_PANEL_HEIGHT_IN,
        ),
        constrained_layout=True,
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


# Figure 5 is assembled in LaTeX from one PDF per panel, at 0.48 + 0.02 +
# 0.48 of the text block per row. No panel carries a letter: the float
# supplies them through \subcaption, so they match the document font and
# stay referenceable.
CLINICAL_PANEL_FRACTION = 0.48
CLINICAL_PANEL_HEIGHT_IN = 2.0

# The channel-wise surface is a 6 x 1055 image. At the shared panel height its
# data axes prints square, which wastes the wavenumber axis it is read along;
# authoring it shorter, with a slimmer colorbar, gives the image a wide aspect
# while the printed width stays the one \includegraphics requests.
CLINICAL_HEATMAP_HEIGHT_IN = 1.6

# Four panels share one float, so Figure 5 sets type 20% below the
# set_manuscript_theme() body sizes to keep the assembled float legible.
CLINICAL_FONT_SCALE = 0.8
CLINICAL_RC = {
    "font.size": 7.2,
    "axes.labelsize": 7.6,
    "axes.titlesize": 8.0,
    "legend.fontsize": 6.8,
    "legend.title_fontsize": 6.8,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
}
# Panel-local overrides that predate the scale factor, carried down with it.
CLINICAL_SMALL_FS = 7.0 * CLINICAL_FONT_SCALE

CLINICAL_ENDPOINT_LABELS = {
    "moca_baseline": "MoCA base",
    "moca_3m": "MoCA 3-mo",
    "delta_moca": r"$\Delta$MoCA",
    "rbans_total": "RBANS",
    "aspects_score": "ASPECTS",
    "age": "Age",
}


def _clinical_panel_axes(height: float = CLINICAL_PANEL_HEIGHT_IN):
    """One axes sized for a quarter of the assembled Figure 5 float."""
    return plt.subplots(
        figsize=(panel_width_in(CLINICAL_PANEL_FRACTION), height),
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
    """The four within-stroke clinical association panels, drawn at the Figure 5 type scale."""
    with matplotlib.rc_context(CLINICAL_RC):
        _draw_clinical_null_panels()


def _draw_clinical_null_panels() -> None:
    """The four within-stroke clinical association panels, one standalone PDF each.

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
        fontsize=CLINICAL_SMALL_FS,
        frameon=True,
        framealpha=1.0,
        edgecolor="none",
        borderpad=0.3,
        handletextpad=0.4,
    )
    _save_exact(fig, "clinical_null_candidates")

    # Channel-wise correlation surface, fingerprint block only.
    fig, ax = _clinical_panel_axes(height=CLINICAL_HEATMAP_HEIGHT_IN)
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
    # imshow draws the first matrix row at the top of the extent box, so the
    # ticks have to descend from the top for a label to name the row above it.
    # Ascending ticks paired the first endpoint with the bottom row, which
    # reversed the whole axis against the data it labelled.
    ax.set_yticks(len(surface) - np.arange(len(surface)) - 0.5)
    ax.set_yticklabels(
        [CLINICAL_ENDPOINT_LABELS.get(name, name) for name in surface.index],
        fontsize=CLINICAL_SMALL_FS,
    )
    ax.set_xlabel(r"Raman shift (cm$^{-1}$)")
    # The key rides on top rather than beside the image: a vertical colorbar
    # and its label take a sixth of the panel width away from the surface,
    # which is read along the wavenumber axis and wants every inch of it.
    bar = fig.colorbar(
        image,
        ax=ax,
        orientation="horizontal",
        location="top",
        fraction=0.055,
        pad=0.04,
        aspect=45,
    )
    bar.set_ticks([-0.4, 0.0, 0.4])
    bar.set_label(r"Spearman $\rho$", fontsize=CLINICAL_SMALL_FS, labelpad=2)
    bar.ax.tick_params(labelsize=CLINICAL_SMALL_FS, length=2, pad=1.0)
    bar.outline.set_linewidth(0.4)
    _save_exact(fig, "clinical_null_channelwise")

    # Observed CV R^2 against each endpoint's own permutation reference distribution.
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
        [CLINICAL_ENDPOINT_LABELS[endpoint] for endpoint in endpoints],
        fontsize=CLINICAL_SMALL_FS,
    )
    ax.set_ylabel("Cross-validated $R^2$")
    ax.spines[["top", "right"]].set_visible(False)
    _clear_band_for_legend(ax, "bottom", 0.24)
    ax.legend(
        loc="lower left",
        fontsize=CLINICAL_SMALL_FS,
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


# Figure 4 is assembled in LaTeX from one PDF per panel, at 0.48 + 0.02 + 0.48
# of the text block. Each panel is authored at the width its \includegraphics
# requests, so neither is rescaled and the two keep a common type size.
CVD_PANEL_FRACTION = 0.48
CVD_PANEL_HEIGHT_IN = 2.4


def _cvd_panel_axes():
    """One axes sized for half of the assembled Figure 4 float."""
    return plt.subplots(
        figsize=(panel_width_in(CVD_PANEL_FRACTION), CVD_PANEL_HEIGHT_IN),
        constrained_layout=True,
    )


def make_cvd_classifier_figure() -> None:
    """ROC curves (whole-spectrum models) and the primary permutation reference distribution.

    Sources only saved repeated-nested-CV out-of-fold predictions and the
    already-computed sex-stratified permutation reference distribution; no predictions are
    re-derived.
    """
    comparison = pd.read_csv(ML_MODEL_COMPARISON)
    comparison = comparison[comparison["representation"] == "whole"].set_index("model")

    # One standalone PDF per panel: the float assembles them with \subcaption,
    # so the panel letters are typeset by LaTeX and stay referenceable.
    fig_roc, ax_roc = _cvd_panel_axes()
    fig_null, ax_null = _cvd_panel_axes()

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
    counts, _, _ = ax_null.hist(
        primary_null, bins=22, color="#B0B0B0", edgecolor="white", linewidth=0.4
    )
    ax_null.axvline(observed, color=CVD_MODEL_COLORS["pls_da"], linewidth=1.6)
    ax_null.set_xlabel("Balanced accuracy\n(sex-stratified permuted labels)")
    ax_null.set_ylabel("Permutation draws")
    ax_null.spines[["top", "right"]].set_visible(False)
    ax_null.tick_params(direction="out", length=3)
    # Labelled at the line rather than through a legend: a legend box in a
    # corner of a 2.6 in panel sits on the reference bars, and it separates the
    # observed value from the line it names. The p-value belongs here too --
    # it is what the panel is evidence for, and it was only in the caption.
    ax_null.set_ylim(0, float(counts.max()) * 1.2)
    # The observed line lands past the reference distribution's right tail; without a margin it
    # is drawn on the frame and reads as a clipped axis rather than a value.
    ax_null.set_xlim(right=observed + 0.02)
    ax_null.annotate(
        f"Observed {observed:.3f}\n$p$ = {p_value:.3f}",
        xy=(observed, ax_null.get_ylim()[1] * 0.97),
        xytext=(-4, 0),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=7.5,
        color=CVD_MODEL_COLORS["pls_da"],
        linespacing=1.3,
    )

    _save_exact(fig_roc, "ml_classification_roc")
    _save_exact(fig_null, "ml_classification_permutation_null")


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    set_manuscript_theme()
    rng = np.random.default_rng(SEED)
    advance_rng_past_subject_spectra(rng)
    make_global_spectra_figures()
    make_band_effect_figure(rng)
    make_biopca_loading_figure()
    make_difference_spectrum_figure()
    make_panel_effect_sizes_figure()
    make_cvd_classifier_figure()
    make_clinical_null_figures()
    make_graphical_abstract()


if __name__ == "__main__":
    main()
