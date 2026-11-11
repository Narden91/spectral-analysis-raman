"""Phase U: assembled journal figure set for Spectrochimica Acta A.

Assembles 6 publication-quality figures from Tier-1 and Tier-2 outputs:

  fig1 — Cohort overview + named-index rainclouds by contrast
  fig2 — Effect-size forest + robustness ranking (anchor highlighted)
  fig3 — Difference spectrum + signed-effect map (block-separated)
  fig4 — NMF component spectra + spectral entropy vs ΔMoCA
  fig5 — Feature coupling heatmap + ratio–ratio regimes
  fig6 — ΔMoCA whole-spectrum association landscape + PLS permutation check

All figures are written to ``06_spectral_interpretability/08_figures/``.
Captions use region-overlapping language; no diagnostic or causal claims.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import (
    get_classification_phase_dir,
    get_confirmatory_panel_dir,
    get_cross_region_phase_dir,
    get_spectral_interpretability_dir,
)
from src import resolve_cfg as _cfg
from src.analysis.feature_coupling import REGIME_PALETTE, coupling_df_to_matrix
from src.analysis.io_utils import load_csv_or_empty
from src.analysis.phase_status import STATUS_MISSING_REQUIRED_DATA, STATUS_OK, PhaseStatus

logger = logging.getLogger(__name__)

_INDEX_COLS = [
    "membrane_integrity_index",
    "oxidative_index",
    "protein_remodeling_index",
    "tyrosine_environment_index",
]
_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
_SMALL_EFFECT_THRESHOLD = 0.147

def _save_and_close(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1 — Cohort overview + named-index rainclouds
# ---------------------------------------------------------------------------


def _fig1_cohort_rainclouds(panel_matrix_path: Path, output_path: Path) -> bool:
    panel = load_csv_or_empty(panel_matrix_path)
    index_cols = [c for c in _INDEX_COLS if c in panel.columns]
    if not index_cols or panel.empty:
        return False

    group_cols = [c for c in panel.columns if c.endswith("_group_extreme") or c.endswith("_group")]
    group_col = group_cols[0] if group_cols else None
    if group_col is None:
        logger.info(
            "Phase U fig1: no contrast-group column in panel matrix; "
            "falling back to ungrouped distributions."
        )
    palette = {"severe": "#D55E00", "mild": "#0072B2", "decline": "#CC79A7", "recovery": "#009E73"}

    fig, axes = plt.subplots(1, len(index_cols), figsize=(3.5 * len(index_cols), 4), squeeze=False)
    for ax, col in zip(axes[0], index_cols):
        if group_col:
            for label, grp in panel.groupby(group_col):
                v = grp[col].dropna().to_numpy(dtype=float)
                jitter = np.random.default_rng(0).uniform(-0.08, 0.08, v.size)
                color = palette.get(str(label), "#888888")
                ax.scatter(jitter, v, alpha=0.6, color=color, s=20, label=str(label))
                ax.boxplot(v, positions=[0], widths=0.15, patch_artist=True,
                           boxprops=dict(facecolor=color, alpha=0.3),
                           medianprops=dict(color="black"),
                           whiskerprops=dict(color=color),
                           capprops=dict(color=color),
                           flierprops=dict(marker=""))
        else:
            ax.violinplot(panel[col].dropna().to_numpy(dtype=float), positions=[0])
        ax.set_xticks([])
        ax.set_title(col.replace("_", "\n"), fontsize=8)
        ax.set_ylabel("index value (a.u.)" if ax is axes[0][0] else "")

    if group_col:
        handles, labels_leg = axes[0][0].get_legend_handles_labels()
        if handles:
            fig.legend(handles[:4], labels_leg[:4], loc="lower center",
                       ncol=min(4, len(labels_leg)), fontsize=7, bbox_to_anchor=(0.5, -0.02))

    suptitle = (
        "Named EV biochemical indices by contrast group"
        if group_col
        else "Named EV biochemical indices (no contrast group available)"
    )
    fig.suptitle(suptitle, fontsize=9)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    _save_and_close(fig, output_path)
    return True


# ---------------------------------------------------------------------------
# Figure 2 — Effect-size forest + robustness ranking
# ---------------------------------------------------------------------------


def _rank_lookup(ranking: pd.DataFrame) -> dict[str, bool]:
    if ranking.empty or "feature_name" not in ranking.columns:
        return {}
    return {str(r["feature_name"]): bool(r.get("permutation_flag", False)) for _, r in ranking.iterrows()}


def _feature_color(feature: str, has_nominal_permutation_flag: bool, anchor: str) -> str:
    if feature == anchor:
        return "#D55E00"
    if has_nominal_permutation_flag:
        return "#0072B2"
    return "#AAAAAA"


def _fig2_effect_forest(
    contrasts_path: Path,
    ranking_path: Path,
    anchor_feature: str,
    output_path: Path,
) -> bool:
    contrasts = load_csv_or_empty(contrasts_path)
    if contrasts.empty or "feature_name" not in contrasts.columns:
        return False

    required_numeric = ("cliffs_delta", "bootstrap_ci_lower", "bootstrap_ci_upper")
    if "contrast_name" not in contrasts.columns or any(column not in contrasts.columns for column in required_numeric):
        return False
    eligible = contrasts.copy()
    if "status" in eligible.columns:
        eligible = eligible.loc[eligible["status"].eq("ok")].copy()
    for column in required_numeric:
        eligible[column] = pd.to_numeric(eligible[column], errors="coerce")
    eligible = eligible.loc[np.isfinite(eligible[list(required_numeric)]).all(axis=1)].copy()

    selected_contrast = None
    severity = pd.DataFrame()
    for contrast_name in ("severity_extreme", "trajectory_extreme"):
        candidate = eligible.loc[eligible["contrast_name"].eq(contrast_name)].copy()
        if not candidate.empty:
            severity = candidate
            selected_contrast = contrast_name
            break
    if severity.empty:
        return False

    features = severity["feature_name"].unique().tolist()
    if not features:
        return False

    scores = _rank_lookup(load_csv_or_empty(ranking_path))

    def _first_numeric(series: pd.Series) -> float:
        vals = pd.to_numeric(series, errors="coerce").dropna()
        return float(vals.iloc[0]) if not vals.empty else np.nan

    deltas, ci_lowers, ci_uppers, feat_scores = [], [], [], []
    for feat in features:
        sub = severity[severity["feature_name"] == feat]
        deltas.append(_first_numeric(sub.get("cliffs_delta", pd.Series(dtype=float))))
        ci_lowers.append(_first_numeric(sub.get("bootstrap_ci_lower", pd.Series(dtype=float))))
        ci_uppers.append(_first_numeric(sub.get("bootstrap_ci_upper", pd.Series(dtype=float))))
        feat_scores.append(scores.get(str(feat), 0))

    colors = [_feature_color(f, s, anchor_feature) for f, s in zip(features, feat_scores)]
    y_pos = np.arange(len(features))

    fig, ax = plt.subplots(figsize=(6, max(3, len(features) * 0.5 + 1)))
    ax.scatter(deltas, y_pos, color=colors, zorder=3, s=50)
    for i, (lo, hi) in enumerate(zip(ci_lowers, ci_uppers)):
        if np.isfinite(lo) and np.isfinite(hi):
            ax.hlines(y_pos[i], lo, hi, color=colors[i], linewidth=1.5)
    ax.axvline(0.0, color="black", linewidth=0.8, linestyle="--")
    for x in (_SMALL_EFFECT_THRESHOLD, -_SMALL_EFFECT_THRESHOLD):
        ax.axvline(x, color="#999999", linewidth=0.6, linestyle=":")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f.replace("_", " ") for f in features], fontsize=8)
    ax.set_xlabel("Cliff's δ (95 % bootstrap CI)")
    ax.set_title(
        f"Effect-size forest — {selected_contrast.replace('_', ' ')}\n"
        "(anchor = orange; nominal permutation flag = blue; all checks are exploratory)"
    )
    ax.invert_yaxis()
    fig.tight_layout()
    _save_and_close(fig, output_path)
    return True


# ---------------------------------------------------------------------------
# Figure 3 — Difference spectrum + signed-effect map
# ---------------------------------------------------------------------------


def _plot_block_panel(axes_row: tuple, sub: pd.DataFrame, block: str, contrast: str) -> None:
    wn = sub["wavenumber"].to_numpy(dtype=float)
    sort_idx = np.argsort(wn)
    wn = wn[sort_idx]
    se_vals = sub["signed_effect"].to_numpy(dtype=float)[sort_idx]
    mean_a = sub["mean_group_a"].to_numpy(dtype=float)[sort_idx]
    mean_b = sub["mean_group_b"].to_numpy(dtype=float)[sort_idx]
    label_a = str(sub["group_a"].iloc[0]) if "group_a" in sub.columns else "A"
    label_b = str(sub["group_b"].iloc[0]) if "group_b" in sub.columns else "B"

    ax0, ax1 = axes_row
    ax0.plot(wn, mean_a, color="#0072B2", linewidth=1.2, label=label_a)
    ax0.plot(wn, mean_b, color="#E69F00", linewidth=1.2, label=label_b)
    ax0.fill_between(wn, mean_a, mean_b, where=(mean_a > mean_b), alpha=0.2, color="#0072B2")
    ax0.fill_between(wn, mean_a, mean_b, where=(mean_a <= mean_b), alpha=0.2, color="#E69F00")
    # ``block`` round-trips from the archived CSV as int64, so comparing it
    # against the string "1300" silently labelled both panels "CH-stretch".
    # The contrast is already named in the figure title.
    ax0.set_title(_BLOCK_LABELS.get(str(block), str(block)))
    ax0.legend(fontsize=7)
    ax0.set_ylabel("Norm. intensity (a.u.)")

    bar_width = float(np.diff(wn).mean()) * 0.85 if wn.size > 1 else 1.0
    pos = se_vals >= 0
    ax1.bar(wn[pos], se_vals[pos], width=bar_width, color="#0072B2", alpha=0.7)
    ax1.bar(wn[~pos], se_vals[~pos], width=bar_width, color="#E69F00", alpha=0.7)
    ax1.axhline(0, color="black", linewidth=0.7)
    ax1.set_xlabel("Raman shift (cm⁻¹)")
    ax1.set_ylabel("sign(Δ)×|Cohen's d|")


def _fig3_difference_spectrum(
    difference_spectrum_path: Path,
    output_path: Path,
    figsize: tuple[float, float] | None = None,
) -> bool:
    diff = load_csv_or_empty(difference_spectrum_path)
    if diff.empty or "wavenumber" not in diff.columns or "block" not in diff.columns:
        return False

    blocks = sorted(diff["block"].unique().tolist())
    contrast = diff["contrast_name"].iloc[0] if "contrast_name" in diff.columns else "unknown"

    fig, axes = plt.subplots(
        2,
        len(blocks),
        figsize=figsize or (5.5 * len(blocks), 5),
        squeeze=False,
    )
    plotted_any = False
    for col_idx, block in enumerate(blocks):
        sub = diff[diff["block"] == block]
        if "contrast_name" in sub.columns:
            sub = sub[sub["contrast_name"] == contrast]
        if sub.empty:
            continue
        _plot_block_panel((axes[0][col_idx], axes[1][col_idx]), sub, block, contrast)
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        return False

    fig.suptitle(f"Difference spectrum: {contrast.replace('_', ' ')}", fontsize=9)
    fig.tight_layout()
    _save_and_close(fig, output_path)
    return True


# ---------------------------------------------------------------------------
# Figure 4 — NMF components + spectral entropy vs ΔMoCA
# ---------------------------------------------------------------------------


def _plot_nmf_components(ax: plt.Axes, components: pd.DataFrame) -> bool:
    if "wavenumber" not in components.columns:
        return False

    # Phase N writes long-form loadings: component, wavenumber, loading.
    if "loading" in components.columns and "component" in components.columns:
        comp_ids = sorted(components["component"].dropna().astype(str).unique())
        if not comp_ids:
            return False
        plotted = False
        for i, comp_id in enumerate(comp_ids):
            sub = components.loc[components["component"].astype(str) == comp_id].dropna(
                subset=["wavenumber", "loading"]
            )
            if sub.empty:
                continue
            wn = sub["wavenumber"].to_numpy(dtype=float)
            sort_idx = np.argsort(wn)
            ax.plot(
                wn[sort_idx],
                sub["loading"].to_numpy(dtype=float)[sort_idx],
                color=_COLORS[i % len(_COLORS)],
                linewidth=1.2,
                label=comp_id.replace("_", " "),
            )
            plotted = True
        if not plotted:
            return False
        ax.set_xlabel("Raman shift (cm⁻¹)")
        ax.set_ylabel("Component loading (a.u.)")
        ax.set_title("NMF latent components (fingerprint)")
        ax.legend(fontsize=7)
        return True

    # Wide-form fallback: component_1, component_2, … as numeric columns.
    comp_cols = [c for c in components.columns if c.startswith("component_")]
    if not comp_cols:
        comp_cols = [c for c in components.columns if c not in {"wavenumber", "block", "nearest_band"}][:5]
    if not comp_cols:
        return False
    for i, col in enumerate(comp_cols):
        sub = components.dropna(subset=["wavenumber", col])
        wn = sub["wavenumber"].to_numpy(dtype=float)
        sort_idx = np.argsort(wn)
        ax.plot(wn[sort_idx], pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)[sort_idx],
                color=_COLORS[i % len(_COLORS)], linewidth=1.2, label=f"NMF {i + 1}")
    ax.set_xlabel("Raman shift (cm⁻¹)")
    ax.set_ylabel("Component loading (a.u.)")
    ax.set_title("NMF latent components (fingerprint)")
    ax.legend(fontsize=7)
    return True


def _plot_entropy_scatter(ax: plt.Axes, entropy: pd.DataFrame) -> bool:
    if "spectral_entropy" not in entropy.columns:
        return False
    if "delta_moca" in entropy.columns:
        ent = entropy["spectral_entropy"].to_numpy(dtype=float)
        dm = entropy["delta_moca"].to_numpy(dtype=float)
        valid = np.isfinite(ent) & np.isfinite(dm)
        ax.scatter(ent[valid], dm[valid], alpha=0.7, color="#0072B2", s=40,
                   edgecolors="white", linewidths=0.5)
        if valid.sum() >= 3:
            m, b = np.polyfit(ent[valid], dm[valid], 1)
            xfit = np.linspace(ent[valid].min(), ent[valid].max(), 50)
            ax.plot(xfit, m * xfit + b, color="#D55E00", linewidth=1.0, linestyle="--")
        ax.set_ylabel("ΔMoCA")
    else:
        ax.violinplot(entropy["spectral_entropy"].dropna().to_numpy(dtype=float), positions=[0])
    ax.set_xlabel("Spectral entropy (dispersion descriptor)")
    ax.set_title("Spectral entropy vs ΔMoCA")
    return True


def _fig4_nmf_entropy(
    nmf_components_path: Path,
    entropy_path: Path,
    output_path: Path,
) -> bool:
    components = load_csv_or_empty(nmf_components_path)
    entropy = load_csv_or_empty(entropy_path)
    has_components, has_entropy = not components.empty, not entropy.empty
    if not has_components and not has_entropy:
        return False

    n_plots = int(has_components) + int(has_entropy)
    fig, axes = plt.subplots(1, n_plots, figsize=(5.5 * n_plots, 4), squeeze=False)
    ax_list = list(axes[0])

    if has_components:
        _plot_nmf_components(ax_list.pop(0), components)
    if has_entropy:
        _plot_entropy_scatter(ax_list.pop(0), entropy)

    fig.suptitle("NMF latent components and spectral entropy", fontsize=9)
    fig.tight_layout()
    _save_and_close(fig, output_path)
    return True


# ---------------------------------------------------------------------------
# Figure 5 — Feature coupling heatmap + ratio–ratio regimes
# ---------------------------------------------------------------------------


def _plot_fig5_coupling(ax: plt.Axes, coupling: pd.DataFrame) -> None:
    mat, features = coupling_df_to_matrix(coupling)
    n = len(features)
    im = ax.imshow(mat, vmin=-1.0, vmax=1.0, cmap="RdBu_r", aspect="auto")
    plt.colorbar(im, ax=ax, label="Spearman ρ", shrink=0.8)
    labels = [f.replace("_", "\n") for f in features]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_title("Feature coupling matrix")


def _plot_fig5_regimes(ax: plt.Axes, regimes: pd.DataFrame) -> None:
    ox_col = "oxidative_index" if "oxidative_index" in regimes.columns else None
    mem_col = "membrane_integrity_index" if "membrane_integrity_index" in regimes.columns else None
    if ox_col and mem_col:
        for regime, grp in regimes.groupby("biochemical_regime"):
            ax.scatter(
                grp[mem_col].to_numpy(dtype=float),
                grp[ox_col].to_numpy(dtype=float),
                label=str(regime).replace("_", " "),
                color=REGIME_PALETTE.get(str(regime), "#888888"),
                alpha=0.75, s=45, edgecolors="white", linewidths=0.4,
            )
        ax.set_xlabel("Membrane integrity index")
        ax.set_ylabel("Oxidative index")
        ax.legend(fontsize=7)
    ax.set_title("Ratio–ratio biochemical regimes")


def _fig5_coupling_regimes(coupling_path: Path, regimes_path: Path, output_path: Path) -> bool:
    coupling = load_csv_or_empty(coupling_path)
    regimes = load_csv_or_empty(regimes_path)
    has_coupling = not coupling.empty and "feature_a" in coupling.columns
    has_regimes = not regimes.empty and "biochemical_regime" in regimes.columns
    if not has_coupling and not has_regimes:
        return False

    n_plots = int(has_coupling) + int(has_regimes)
    fig, axes = plt.subplots(1, n_plots, figsize=(5.5 * n_plots, 4.5), squeeze=False)
    ax_list = list(axes[0])

    if has_coupling:
        _plot_fig5_coupling(ax_list.pop(0), coupling)
    if has_regimes:
        _plot_fig5_regimes(ax_list.pop(0), regimes)

    fig.suptitle("Feature coupling and biochemical regimes", fontsize=9)
    fig.tight_layout()
    _save_and_close(fig, output_path)
    return True


# ---------------------------------------------------------------------------
# Figure 6 — ΔMoCA whole-spectrum association landscape + PLS permutation check
# ---------------------------------------------------------------------------

_BLOCK_LABELS = {"1300": "Fingerprint", "2850": "CH-stretch"}
_FDR_SIGNIFICANT_ALPHA = 0.05
_TARGET_LABELS = {
    "moca_baseline": "MoCA\nbaseline",
    "moca_3m": "MoCA\n3 months",
    "delta_moca": "ΔMoCA",
}


def _plot_moca_landscape_block(ax: plt.Axes, sub: pd.DataFrame, *, show_legend_labels: bool) -> tuple[float, float]:
    sub = sub.sort_values("wavenumber")
    wn = sub["wavenumber"].to_numpy(dtype=float)
    rho = sub["rho"].to_numpy(dtype=float)
    fdr_q = pd.to_numeric(sub.get("fdr_q", pd.Series(np.nan, index=sub.index)), errors="coerce").to_numpy(dtype=float)

    is_significant = np.isfinite(fdr_q) & (fdr_q < _FDR_SIGNIFICANT_ALPHA)
    ax.scatter(
        wn[~is_significant], rho[~is_significant],
        color="#AAAAAA", s=8, alpha=0.6, linewidths=0,
        label="FDR q ≥ 0.05" if show_legend_labels else None,
    )
    if is_significant.any():
        ax.scatter(
            wn[is_significant], rho[is_significant],
            color="#D55E00", s=14, alpha=0.9, linewidths=0,
            label="FDR q < 0.05" if show_legend_labels else None,
        )
    return (float(wn.min()), float(wn.max())) if wn.size else (np.nan, np.nan)


def _fig6_moca_landscape_panel(ax: plt.Axes, associations: pd.DataFrame) -> bool:
    required = {"analysis_name", "spectral_block", "wavenumber", "rho"}
    if associations.empty or not required.issubset(associations.columns):
        return False
    delta = associations.loc[associations["analysis_name"].eq("delta_moca_association")].copy()
    if delta.empty:
        return False
    delta["fdr_q"] = pd.to_numeric(delta.get("fdr_q"), errors="coerce")

    blocks = sorted(delta["spectral_block"].astype(str).unique(), key=lambda b: (b != "1300", b))
    n_significant = int((delta["fdr_q"] < _FDR_SIGNIFICANT_ALPHA).sum())
    n_total = int(len(delta))
    min_fdr_q = delta["fdr_q"].min()
    min_cluster_p = pd.to_numeric(delta.get("cluster_p_value"), errors="coerce").min()

    block_ranges: list[tuple[float, float]] = []
    for block_index, block in enumerate(blocks):
        sub = delta.loc[delta["spectral_block"].astype(str) == block]
        wn_min, wn_max = _plot_moca_landscape_block(ax, sub, show_legend_labels=block_index == 0)
        if np.isfinite(wn_min):
            block_ranges.append((wn_min, wn_max))
        if block_index < len(blocks) - 1 and np.isfinite(wn_max):
            ax.axvline(wn_max, color="#DDDDDD", linewidth=0.8, linestyle=":")

    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.set_xlabel("Raman shift (cm⁻¹)")
    ax.set_ylabel("Spearman ρ (feature vs. ΔMoCA)")

    # Place block-name labels just inside the top of the data range (data
    # coordinates) rather than above the axes, so they never collide with
    # the panel title.
    rho_max = float(delta["rho"].max())
    rho_min = float(delta["rho"].min())
    label_y = rho_max + 0.06 * (rho_max - rho_min if rho_max > rho_min else 1.0)
    for block, (wn_min, wn_max) in zip(blocks, block_ranges):
        ax.text(
            0.5 * (wn_min + wn_max), label_y, _BLOCK_LABELS.get(block, block),
            ha="center", va="bottom", fontsize=8, color="#555555",
        )
    ax.set_ylim(top=label_y + 0.08 * (rho_max - rho_min if rho_max > rho_min else 1.0))
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=7, loc="upper right")

    annotation = (
        f"{n_significant} / {n_total} wavenumbers FDR-significant"
        f"\nmin FDR q = {min_fdr_q:.2g}"
        + (f", min cluster p = {min_cluster_p:.2g}" if np.isfinite(min_cluster_p) else "")
    )
    ax.text(
        0.02, 0.04, annotation, transform=ax.transAxes, fontsize=8,
        va="bottom", ha="left",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#CCCCCC", alpha=0.9),
    )
    ax.set_title("Whole-spectrum ΔMoCA association scan (exploratory, uncorrected + FDR)", fontsize=10)
    return True


def _fig6_pls_null_panel(ax: plt.Axes, performance: pd.DataFrame) -> bool:
    required = {"target", "r2", "permutation_p_value_holm", "model_supported"}
    if performance.empty or not required.issubset(performance.columns):
        return False

    perf = performance.loc[performance["target"].isin(_TARGET_LABELS)].copy()
    if perf.empty:
        return False
    perf["target"] = pd.Categorical(perf["target"], categories=list(_TARGET_LABELS), ordered=True)
    perf = perf.sort_values("target")

    y_pos = np.arange(len(perf))
    r2 = pd.to_numeric(perf["r2"], errors="coerce").to_numpy(dtype=float)
    supported = perf["model_supported"].astype(bool).to_numpy()
    colors = np.where(supported, "#D55E00", "#0072B2")

    ax.scatter(r2, y_pos, color=colors, s=60, zorder=3)
    ax.axvline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([_TARGET_LABELS[t] for t in perf["target"]], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Cross-validated R² (observed)")
    ax.set_title("PLS1 regression: permutation comparison\n(R² ≤ 0 = no explanatory power beyond the mean)", fontsize=10)

    for y, row in zip(y_pos, perf.itertuples()):
        p_holm = getattr(row, "permutation_p_value_holm", np.nan)
        p_label = f"Holm p={p_holm:.2g}" if np.isfinite(p_holm) else "Holm p=n/a"
        ax.annotate(
            p_label, (row.r2, y), xytext=(8, 0), textcoords="offset points",
            fontsize=8, va="center",
        )
    return True


def _fig6_moca_association_landscape(
    associations_path: Path,
    pls_performance_path: Path,
    output_path: Path,
) -> bool:
    associations = load_csv_or_empty(associations_path)
    performance = load_csv_or_empty(pls_performance_path)

    fig, (ax_landscape, ax_pls) = plt.subplots(
        1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [2.2, 1]}, constrained_layout=True,
    )
    has_landscape = _fig6_moca_landscape_panel(ax_landscape, associations)
    has_pls = _fig6_pls_null_panel(ax_pls, performance)

    if not has_landscape and not has_pls:
        plt.close(fig)
        return False
    if not has_landscape:
        ax_landscape.set_visible(False)
    if not has_pls:
        ax_pls.set_visible(False)

    fig.suptitle(
        "ΔMoCA association landscape: exploratory whole-spectrum scan (Holm/FDR-corrected) "
        "and PLS1 permutation check — no wavenumber or model exceeds the correction or permutation threshold",
        fontsize=9,
    )
    _save_and_close(fig, output_path)
    return True


# ---------------------------------------------------------------------------
# Orchestrating runner
# ---------------------------------------------------------------------------


def _try_figure(
    key: str,
    path: Path,
    fn: Callable[..., bool],
    *args: Any,
) -> tuple[str, Path] | None:
    """Return (key, path) when written, None for skipped inputs; raise builder errors.

    All figure builders accept output_path as their final positional argument.
    """
    path.unlink(missing_ok=True)
    try:
        if fn(*args, path):
            return key, path
    except Exception as exc:
        raise RuntimeError(f"Phase U: {key} failed") from exc
    return None


def run_journal_figures(
    *,
    panel_matrix_path: str | Path | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, object]:
    """Phase U orchestrator: assemble 6 journal figures from prior phase outputs."""
    cfg = _cfg(cfg)
    phase_cfg = cfg.get("analysis", {}).get("journal_figures", {})
    if not bool(phase_cfg.get("enabled", True)):
        return {}

    panel_dir = get_confirmatory_panel_dir()
    figures_dir = get_spectral_interpretability_dir("figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    anchor_feature = (
        cfg.get("analysis", {}).get("preregistered_panel", {})
        .get("primary_hypothesis", {}).get("feature", "oxidative_index")
    )
    pmat = Path(panel_matrix_path) if panel_matrix_path is not None else panel_dir / "panel_feature_matrix.csv"

    fig_specs = [
        ("fig1", figures_dir / "fig1_cohort_panel_overview.png",
         _fig1_cohort_rainclouds, pmat),
        ("fig2", figures_dir / "fig2_effect_size_forest_robustness.png",
         _fig2_effect_forest,
         panel_dir / "panel_contrast_tests.csv",
         panel_dir / "panel_evidence_flags.csv",
         anchor_feature),
        ("fig3", figures_dir / "fig3_difference_spectrum_signed_effect.png",
         _fig3_difference_spectrum,
         get_spectral_interpretability_dir("discriminant") / "difference_spectrum.csv"),
        ("fig4", figures_dir / "fig4_nmf_components_entropy.png",
         _fig4_nmf_entropy,
         get_spectral_interpretability_dir("nmf") / "nmf_components.csv",
         get_spectral_interpretability_dir("entropy") / "spectral_entropy.csv"),
        ("fig5", figures_dir / "fig5_feature_coupling_regimes.png",
         _fig5_coupling_regimes,
         get_spectral_interpretability_dir("coupling") / "feature_coupling_matrix.csv",
         get_spectral_interpretability_dir("coupling") / "ratio_ratio_regimes.csv"),
        ("fig6", figures_dir / "fig6_moca_association_landscape.png",
         _fig6_moca_association_landscape,
         get_cross_region_phase_dir("longitudinal_moca") / "delta_moca_associations.csv",
         get_classification_phase_dir("pls_regression") / "pls_regression_performance.csv"),
    ]

    outputs: dict[str, object] = {}
    for key, path, fn, *args in fig_specs:
        result = _try_figure(key, path, fn, *args)
        if result:
            outputs[result[0]] = result[1]

    missing = [key for key, *_ in fig_specs if key not in outputs]
    outputs["_status"] = PhaseStatus(
        phase_key="journal_figures",
        status=STATUS_OK if not missing else STATUS_MISSING_REQUIRED_DATA,
        reason="" if not missing else f"missing figure inputs or valid data: {', '.join(missing)}",
        n_artifacts=len(outputs),
    ).as_row()
    logger.info("Phase U: %d/%d figures produced in %s", len(outputs) - 1, len(fig_specs), figures_dir)
    return outputs
