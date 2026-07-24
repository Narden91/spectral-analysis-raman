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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy import stats
from sklearn.metrics import roc_curve

from src.analysis.clinical_correlation_matrix import _plot_correlation_heatmap
from src.analysis.journal_figures import (
    _fig3_difference_spectrum,
    _fig4_nmf_entropy,
    _fig6_moca_association_landscape,
)
from src.analysis.io_utils import load_csv_or_empty
from src.reporting.candidate_region_report import _classify_candidate
from src.visualization.eda_plots import plot_scree
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
    "o_p_o_821": "O--P--O (821)",
    "tyrosine_851": "Tyrosine (851)",
    "c_o_c_sugar_925": "C--O--C, sugar (925)",
    "beta_sheet_980": r"$\beta$-sheet (980)",
    "c_o_c_n_1060": "C--O/C--N (1060)",
    "lipid_cc_1089": "Lipid C--C (1089)",
    "po2_cn_1112": r"PO$_2^-$ / C--N (1112)",
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


def configure_style() -> None:
    """Apply a compact, journal-friendly Matplotlib style."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


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
    fig, ax = plt.subplots(figsize=(6.8, 7.4), constrained_layout=True)
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
    fig, ax = plt.subplots(figsize=(9.5, 4.2), constrained_layout=True)
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
    ax.set_xticks(x, result["label"], rotation=55, ha="right")
    ax.set_xlim(-0.7, result.shape[0] - 0.3)
    ax.set_ylim(-0.7, 0.7)
    ax.set_ylabel(
        "Rank-biserial correlation\n"
        r"$\leftarrow$ lower in stroke     higher in stroke $\rightarrow$"
    )
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)

    for stem in ("case_control_band_effects_landscape",):
        fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


PC_COLORS = {"PC1": "#264653", "PC2": "#2A9D8F", "PC3": "#E76F51"}
BLOCK_TAGS = {"1300": "FP", "2850": "CH"}


def make_biopca_loading_regions_figure() -> None:
    """Redraw the whole-spectrum loading-region contribution chart.

    Sources the already-materialized ``region_contributions.csv`` from the
    production Biological PCA run; no PCA refitting or region re-detection is
    performed. Only the label formatting and orientation are changed relative
    to the pipeline's own figure, to remove the duplicated block-prefix text
    and shorten the y-axis labels.
    """
    contributions = pd.read_csv(BIOPCA_WHOLE_REGION_CONTRIBUTIONS)
    contributions["block_tag"] = contributions["spectral_block"].astype(str).map(
        BLOCK_TAGS
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

    pcs = ["PC1", "PC2", "PC3"]
    pivot = contributions.pivot_table(
        index="compact_label",
        columns="PC",
        values="relative_absolute_contribution",
        aggfunc="sum",
    ).reindex(columns=pcs, fill_value=0.0)
    order = pivot.sum(axis=1).sort_values(ascending=True).index
    pivot = pivot.loc[order]

    y = np.arange(pivot.shape[0])
    height = 0.24
    offsets = np.linspace(-height, height, len(pcs))

    fig, ax = plt.subplots(
        figsize=(6.8, max(3.6, 0.34 * pivot.shape[0])), constrained_layout=True
    )
    for pc, offset in zip(pcs, offsets, strict=True):
        ax.barh(
            y + offset,
            pivot[pc].to_numpy(),
            height=height * 0.95,
            color=PC_COLORS[pc],
            label=pc,
        )
    ax.set_yticks(y, pivot.index)
    ax.set_xlabel("Relative absolute contribution")
    ax.set_title(
        "PC1–PC3 loading-derived regions, whole-spectrum model",
        loc="left",
        fontweight="bold",
    )
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.6)
    ax.legend(
        title="Component",
        frameon=False,
        loc="lower right",
        fontsize=7.5,
        title_fontsize=7.5,
    )

    fig.text(
        0.01,
        -0.02,
        "FP = fingerprint block (734–1788 cm$^{-1}$); "
        "CH = C–H stretch block (2451–3197 cm$^{-1}$).",
        fontsize=6.8,
        color=NEUTRAL,
        ha="left",
    )

    for stem in ("biopca_loading_regions_pc1_pc3",):
        fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def make_biopca_scree_figure() -> None:
    """Redraw the whole-spectrum biological-PCA scree plot as a vector PDF.

    Sources the already-materialized ``explained_variance.csv``; no PCA
    refitting. Reuses the production ``plot_scree`` renderer so the visual
    design matches the pipeline's own figure exactly.
    """
    variance = pd.read_csv(BIOPCA_WHOLE_EXPLAINED_VARIANCE).sort_values("PC")
    ratios = variance["explained_variance_ratio"].to_numpy(dtype=float)
    fig, _ = plot_scree(ratios, title="Whole-spectrum biological PCA")
    for stem in ("biopca_scree_plot",):
        fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


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
        DIFFERENCE_SPECTRUM, FIGURE_DIR / "difference_spectrum_severity_extreme.pdf"
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
    ok["label"] = ok["contrast_name"].astype(str) + " | " + ok["feature_name"].astype(str)
    ok = ok.sort_values("cliffs_delta")

    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.32 * len(ok))))
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
    ax.set_title("Fixed post hoc panel effect sizes")
    fig.tight_layout()

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


def make_clinical_null_figure() -> None:
    """Combine the four within-stroke null results into one figure.

    Panels a--d redraw, at reduced size, the candidate-region screen, the
    channel-wise correlation surface, the PLS-regression permutation
    comparison, and the unsupervised NMF/entropy check.  They are merged
    because each panel supports the same negative conclusion; showing them as
    four separate figures gave a null result more visual weight than the
    positive case--control result.  Every value is read from the same archived
    table used by the corresponding standalone figure.
    """
    candidates = load_csv_or_empty(INTEGRATED_CANDIDATE_REGIONS)
    correlations = load_csv_or_empty(CLINICAL_CORRELATION_MATRIX)
    performance = pd.read_csv(MOCA_PLS_PERFORMANCE).set_index("target")
    permutation = pd.read_csv(MOCA_PLS_PERMUTATION).set_index("target")
    entropy = load_csv_or_empty(SPECTRAL_ENTROPY)
    if candidates.empty or correlations.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), constrained_layout=True)

    # (a) candidate-region screen: every region is classified "unstable".
    ax = axes[0, 0]
    candidates = candidates.assign(
        report_class=candidates.apply(_classify_candidate, axis=1)
    )
    for block, marker in (("1300", "o"), ("2850", "^")):
        block_df = candidates.loc[candidates["spectral_block"].astype(str) == block]
        ax.scatter(
            block_df["start_cm-1"].to_numpy(dtype=float),
            block_df["max_abs_effect"].to_numpy(dtype=float),
            s=14,
            marker=marker,
            color="#F4A261",
            edgecolors="white",
            linewidths=0.3,
            label=f"{block} block",
        )
    ax.set_xlabel(r"Raman shift (cm$^{-1}$)")
    ax.set_ylabel(r"Max $|$effect$|$")
    ax.set_title(
        f"a  Candidate regions ({len(candidates)}, all unstable)",
        loc="left", fontweight="bold", fontsize=8.5,
    )
    ax.legend(frameon=False, fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)

    # (b) channel-wise correlation surface, fingerprint block only.
    ax = axes[0, 1]
    fingerprint = correlations.loc[correlations["spectral_block"] == 1300]
    surface = fingerprint.pivot_table(
        index="target", columns="wavenumber", values="rho"
    )
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
    endpoint_display = {
        "moca_baseline": "Baseline MoCA",
        "moca_3m": "3-month MoCA",
        "delta_moca": r"$\Delta$MoCA",
        "rbans_total": "RBANS",
        "aspects_score": "ASPECTS",
        "age": "Age",
    }
    ax.set_yticks(np.arange(len(surface)) + 0.5)
    ax.set_yticklabels(
        [endpoint_display.get(name, name) for name in surface.index], fontsize=6.5
    )
    ax.set_xlabel(r"Raman shift (cm$^{-1}$)")
    ax.set_title(
        "b  Channel-wise $\\rho$ (uncorrected)",
        loc="left", fontweight="bold", fontsize=8.5,
    )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02, label=r"Spearman $\rho$")

    # (c) observed CV R^2 against each endpoint's own permutation null.
    ax = axes[1, 0]
    rng = np.random.default_rng(SEED)
    endpoints = list(MOCA_ENDPOINT_LABELS)
    for position, endpoint in enumerate(endpoints):
        draws = rng.normal(
            permutation.loc[endpoint, "null_distribution_mean"],
            permutation.loc[endpoint, "null_distribution_std"],
            size=2000,
        )
        violin = ax.violinplot(
            draws, positions=[position], widths=0.7, showextrema=False
        )
        for body in violin["bodies"]:
            body.set_facecolor("#B0B0B0")
            body.set_alpha(0.55)
            body.set_edgecolor("none")
        ax.scatter(
            [position],
            [performance.loc[endpoint, "r2"]],
            color=DISEASE,
            s=32,
            zorder=5,
            label="Observed" if position == 0 else None,
        )
    ax.axhline(0, color=NEUTRAL, linestyle="--", linewidth=0.9)
    ax.set_xticks(range(len(endpoints)))
    ax.set_xticklabels(
        [MOCA_ENDPOINT_LABELS[endpoint] for endpoint in endpoints], fontsize=7.5
    )
    ax.set_ylabel("Cross-validated $R^2$")
    ax.set_title(
        "c  MoCA prediction vs. null", loc="left", fontweight="bold", fontsize=8.5
    )
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)

    # (d) unsupervised descriptor distribution, fingerprint block.
    ax = axes[1, 1]
    values = entropy.loc[
        entropy["spectral_block"] == 1300, "spectral_entropy"
    ].dropna()
    ax.hist(values, bins=22, color="#4D4D4D", alpha=0.75, edgecolor="white")
    ax.set_xlabel("Spectral entropy (fingerprint)")
    ax.set_ylabel("Participants")
    ax.set_title(
        "d  Unsupervised descriptor", loc="left", fontweight="bold", fontsize=8.5
    )
    ax.spines[["top", "right"]].set_visible(False)

    for stem in ("clinical_null_overview",):
        fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


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
        1, 2, figsize=(7.2, 3.15), constrained_layout=True
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
            label=(
                f"{CVD_MODEL_LABELS[model]}\nAUC {auc_mean:.3f} "
                f"({auc_low:.3f}–{auc_high:.3f})"
            ),
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
    ax_roc.set_title(
        "Whole-spectrum classifiers (repeated nested CV)", loc="left",
        fontweight="bold", fontsize=8.5,
    )
    ax_roc.spines[["top", "right"]].set_visible(False)
    ax_roc.tick_params(direction="out", length=3)
    ax_roc.legend(frameon=False, loc="lower right", fontsize=6.3, labelspacing=0.6)

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
    ax_null.set_title(
        f"PLS-DA null distribution ($n={primary_null.size}$, $p={p_value:.3f}$)",
        loc="left", fontweight="bold", fontsize=8.5,
    )
    ax_null.spines[["top", "right"]].set_visible(False)
    ax_null.tick_params(direction="out", length=3)
    ax_null.legend(frameon=False, loc="upper left", fontsize=7)

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
    configure_style()
    rng = np.random.default_rng(SEED)
    make_subject_spectra_figure(rng)
    make_band_effect_figure(rng)
    make_biopca_loading_regions_figure()
    make_biopca_scree_figure()
    make_clinical_correlation_heatmap_figure()
    make_candidate_region_overview_figure()
    make_difference_spectrum_figure()
    make_panel_effect_sizes_figure()
    make_nmf_entropy_figure()
    make_moca_landscape_figure()
    make_cvd_classifier_figure()
    make_moca_prediction_figure()
    make_clinical_null_figure()
    make_graphical_abstract()


if __name__ == "__main__":
    main()
