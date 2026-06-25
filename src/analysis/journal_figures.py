"""Phase U: assembled journal figure set for Spectrochimica Acta A.

Assembles 5 publication-quality figures from Tier-1 and Tier-2 outputs:

  fig1 — Cohort overview + named-index rainclouds by contrast
  fig2 — Effect-size forest + robustness ranking (anchor highlighted)
  fig3 — Difference spectrum + signed-effect map (block-separated)
  fig4 — NMF component spectra + spectral entropy vs ΔMoCA
  fig5 — Feature coupling heatmap + ratio–ratio regimes

All figures are written to ``06_spectral_interpretability/figures/``.
Captions use region-overlapping language; no diagnostic or causal claims.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config, get_confirmatory_panel_dir, get_spectral_interpretability_dir
from src.analysis.io_utils import load_csv_or_empty

logger = logging.getLogger(__name__)


def _cfg(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    return config if cfg is None else cfg


# ---------------------------------------------------------------------------
# Figure 1 — Cohort overview + named-index rainclouds
# ---------------------------------------------------------------------------


def _fig1_cohort_rainclouds(
    panel_matrix_path: Path,
    contrast_tests_path: Path,
    output_path: Path,
) -> bool:
    panel = load_csv_or_empty(panel_matrix_path)
    contrasts = load_csv_or_empty(contrast_tests_path)

    index_cols = [c for c in ["membrane_integrity_index", "oxidative_index",
                               "protein_remodeling_index", "tyrosine_environment_index"]
                  if c in panel.columns]
    if not index_cols or panel.empty:
        return False

    n_indices = len(index_cols)
    fig, axes = plt.subplots(1, n_indices, figsize=(3.5 * n_indices, 4), squeeze=False)

    # Attempt to overlay group labels from any available contrast column
    group_cols = [c for c in panel.columns if c.endswith("_group_extreme") or c.endswith("_group")]
    group_col = group_cols[0] if group_cols else None

    palette = {"severe": "#D55E00", "mild": "#0072B2",
                "decline": "#CC79A7", "recovery": "#009E73"}

    for ax, col in zip(axes[0], index_cols):
        vals = panel[col].dropna().to_numpy(dtype=float)
        if group_col and group_col in panel.columns:
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
            ax.violinplot(vals, positions=[0])

        ax.set_xticks([])
        ax.set_title(col.replace("_", "\n"), fontsize=8)
        ax.set_ylabel("index value (a.u.)" if ax is axes[0][0] else "")

    if group_col:
        handles, labels_leg = axes[0][0].get_legend_handles_labels()
        if handles:
            fig.legend(handles[:4], labels_leg[:4], loc="lower center",
                       ncol=min(4, len(labels_leg)), fontsize=7,
                       bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Named EV biochemical indices by contrast group", fontsize=9)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Figure 2 — Effect-size forest + robustness ranking
# ---------------------------------------------------------------------------


def _fig2_effect_forest(
    contrasts_path: Path,
    ranking_path: Path,
    anchor_feature: str,
    output_path: Path,
) -> bool:
    contrasts = load_csv_or_empty(contrasts_path)
    ranking = load_csv_or_empty(ranking_path)

    if contrasts.empty or "feature_name" not in contrasts.columns:
        return False

    # Use first contrast per feature (severity_extreme preferred)
    severity = contrasts[contrasts.get("contrast_name", pd.Series(dtype=str)) == "severity_extreme"] if "contrast_name" in contrasts.columns else contrasts
    if severity.empty:
        severity = contrasts

    features = severity["feature_name"].unique().tolist()
    deltas = []
    ci_lowers = []
    ci_uppers = []
    scores = []

    rank_lookup = {}
    if not ranking.empty and "feature_name" in ranking.columns:
        for _, row in ranking.iterrows():
            rank_lookup[str(row["feature_name"])] = int(row.get("robust_score", 0))

    for feat in features:
        sub = severity[severity["feature_name"] == feat]
        if sub.empty:
            continue
        delta = pd.to_numeric(sub["cliffs_delta"], errors="coerce").dropna()
        ci_lo = pd.to_numeric(sub.get("bootstrap_ci_lower", pd.Series(dtype=float)), errors="coerce").dropna()
        ci_hi = pd.to_numeric(sub.get("bootstrap_ci_upper", pd.Series(dtype=float)), errors="coerce").dropna()
        deltas.append(float(delta.iloc[0]) if not delta.empty else np.nan)
        ci_lowers.append(float(ci_lo.iloc[0]) if not ci_lo.empty else np.nan)
        ci_uppers.append(float(ci_hi.iloc[0]) if not ci_hi.empty else np.nan)
        scores.append(rank_lookup.get(str(feat), 0))

    if not features:
        return False

    fig, ax = plt.subplots(figsize=(6, max(3, len(features) * 0.5 + 1)))
    y_pos = np.arange(len(features))
    colors = ["#D55E00" if f == anchor_feature else
              ("#0072B2" if s >= 3 else ("#009E73" if s == 2 else "#AAAAAA"))
              for f, s in zip(features, scores)]

    ax.scatter(deltas, y_pos, color=colors, zorder=3, s=50)
    for i, (lo, hi) in enumerate(zip(ci_lowers, ci_uppers)):
        if np.isfinite(lo) and np.isfinite(hi):
            ax.hlines(y_pos[i], lo, hi, color=colors[i], linewidth=1.5)

    ax.axvline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(0.147, color="#999999", linewidth=0.6, linestyle=":")
    ax.axvline(-0.147, color="#999999", linewidth=0.6, linestyle=":")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f.replace("_", " ") for f in features], fontsize=8)
    ax.set_xlabel("Cliff's δ (95 % bootstrap CI)")
    ax.set_title("Effect-size forest — robustness ranking\n(anchor = orange; strong trend = blue; moderate = green)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Figure 3 — Difference spectrum + signed-effect map
# ---------------------------------------------------------------------------


def _fig3_difference_spectrum(
    difference_spectrum_path: Path,
    output_path: Path,
) -> bool:
    diff = load_csv_or_empty(difference_spectrum_path)
    if diff.empty or "wavenumber" not in diff.columns:
        return False

    blocks = diff["block"].unique().tolist() if "block" in diff.columns else []
    if not blocks:
        return False

    contrasts = diff["contrast_name"].unique().tolist() if "contrast_name" in diff.columns else ["unknown"]
    contrast = contrasts[0]

    fig, axes = plt.subplots(2, len(blocks), figsize=(5.5 * len(blocks), 5), squeeze=False)

    for col_idx, block in enumerate(sorted(blocks)):
        sub = diff[diff["block"] == block] if "block" in diff.columns else diff
        sub = sub[sub["contrast_name"] == contrast] if "contrast_name" in sub.columns else sub
        if sub.empty:
            continue

        wn = sub["wavenumber"].to_numpy(dtype=float)
        sort_idx = np.argsort(wn)
        wn = wn[sort_idx]
        diff_vals = sub["difference"].to_numpy(dtype=float)[sort_idx]
        se_vals = sub["signed_effect"].to_numpy(dtype=float)[sort_idx]
        mean_a = sub["mean_group_a"].to_numpy(dtype=float)[sort_idx]
        mean_b = sub["mean_group_b"].to_numpy(dtype=float)[sort_idx]

        # Top: mean spectra
        ax0 = axes[0][col_idx]
        label_a = str(sub["group_a"].iloc[0]) if "group_a" in sub.columns else "A"
        label_b = str(sub["group_b"].iloc[0]) if "group_b" in sub.columns else "B"
        ax0.plot(wn, mean_a, color="#0072B2", linewidth=1.2, label=label_a)
        ax0.plot(wn, mean_b, color="#E69F00", linewidth=1.2, label=label_b)
        ax0.fill_between(wn, mean_a, mean_b, where=(mean_a > mean_b), alpha=0.2, color="#0072B2")
        ax0.fill_between(wn, mean_a, mean_b, where=(mean_a <= mean_b), alpha=0.2, color="#E69F00")
        ax0.set_title(f"{'Fingerprint' if block == '1300' else 'CH-stretch'} — {contrast}")
        ax0.legend(fontsize=7)
        ax0.set_ylabel("Norm. intensity (a.u.)")

        # Bottom: signed-effect map
        ax1 = axes[1][col_idx]
        bar_width = float(np.diff(wn).mean()) * 0.85 if wn.size > 1 else 1.0
        pos = se_vals >= 0
        ax1.bar(wn[pos], se_vals[pos], width=bar_width, color="#0072B2", alpha=0.7)
        ax1.bar(wn[~pos], se_vals[~pos], width=bar_width, color="#E69F00", alpha=0.7)
        ax1.axhline(0, color="black", linewidth=0.7)
        ax1.set_xlabel("Raman shift (cm⁻¹)")
        ax1.set_ylabel("sign(Δ)×|Cohen's d|")

    fig.suptitle(f"Difference spectrum: {contrast}", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Figure 4 — NMF components + spectral entropy vs ΔMoCA
# ---------------------------------------------------------------------------


def _fig4_nmf_entropy(
    nmf_components_path: Path,
    entropy_path: Path,
    assoc_entropy_path: Path,
    output_path: Path,
) -> bool:
    components = load_csv_or_empty(nmf_components_path)
    entropy = load_csv_or_empty(entropy_path)

    has_components = not components.empty
    has_entropy = not entropy.empty

    if not has_components and not has_entropy:
        return False

    n_plots = int(has_components) + int(has_entropy)
    fig, axes = plt.subplots(1, n_plots, figsize=(5.5 * n_plots, 4), squeeze=False)
    ax_idx = 0

    if has_components and "wavenumber" in components.columns:
        ax = axes[0][ax_idx]
        comp_cols = [c for c in components.columns if c.startswith("component_")]
        if not comp_cols:
            comp_cols = [c for c in components.columns if c not in {"wavenumber", "block", "nearest_band"}][:5]
        colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
        for i, col in enumerate(comp_cols):
            sub = components.dropna(subset=["wavenumber", col])
            wn = sub["wavenumber"].to_numpy(dtype=float)
            vals = sub[col].to_numpy(dtype=float)
            sort_idx = np.argsort(wn)
            ax.plot(wn[sort_idx], vals[sort_idx], color=colors[i % len(colors)],
                    linewidth=1.2, label=f"NMF {i + 1}")
        ax.set_xlabel("Raman shift (cm⁻¹)")
        ax.set_ylabel("Component loading (a.u.)")
        ax.set_title("NMF latent components (fingerprint)")
        ax.legend(fontsize=7)
        ax_idx += 1

    if has_entropy and "spectral_entropy" in entropy.columns:
        ax = axes[0][ax_idx]
        assoc = load_csv_or_empty(assoc_entropy_path)
        delta_moca_col = "delta_moca"
        if delta_moca_col in entropy.columns:
            ent = entropy["spectral_entropy"].to_numpy(dtype=float)
            dm = entropy[delta_moca_col].to_numpy(dtype=float)
            valid = np.isfinite(ent) & np.isfinite(dm)
            ax.scatter(ent[valid], dm[valid], alpha=0.7, color="#0072B2", s=40, edgecolors="white", linewidths=0.5)
            if valid.sum() >= 3:
                m, b = np.polyfit(ent[valid], dm[valid], 1)
                xfit = np.linspace(ent[valid].min(), ent[valid].max(), 50)
                ax.plot(xfit, m * xfit + b, color="#D55E00", linewidth=1.0, linestyle="--")
        else:
            ent = entropy["spectral_entropy"].dropna().to_numpy(dtype=float)
            ax.violinplot(ent, positions=[0])
        ax.set_xlabel("Spectral entropy (dispersion descriptor)")
        ax.set_ylabel("ΔMoCA")
        ax.set_title("Spectral entropy vs ΔMoCA")
        ax_idx += 1

    fig.suptitle("NMF latent components and spectral entropy", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Figure 5 — Feature coupling heatmap + ratio–ratio regimes
# ---------------------------------------------------------------------------


def _fig5_coupling_regimes(
    coupling_path: Path,
    regimes_path: Path,
    output_path: Path,
) -> bool:
    coupling = load_csv_or_empty(coupling_path)
    regimes = load_csv_or_empty(regimes_path)

    has_coupling = not coupling.empty and "feature_a" in coupling.columns
    has_regimes = not regimes.empty and "biochemical_regime" in regimes.columns

    if not has_coupling and not has_regimes:
        return False

    n_plots = int(has_coupling) + int(has_regimes)
    fig, axes = plt.subplots(1, n_plots, figsize=(5.5 * n_plots, 4.5), squeeze=False)
    ax_idx = 0

    if has_coupling:
        ax = axes[0][ax_idx]
        features = sorted(set(coupling["feature_a"]) | set(coupling["feature_b"]))
        n = len(features)
        idx_map = {f: i for i, f in enumerate(features)}
        mat = np.full((n, n), np.nan)
        for _, row in coupling.iterrows():
            i = idx_map[str(row["feature_a"])]
            j = idx_map[str(row["feature_b"])]
            rho = float(row["spearman_rho"]) if np.isfinite(float(row["spearman_rho"])) else np.nan
            mat[i, j] = rho
            mat[j, i] = rho
        np.fill_diagonal(mat, 1.0)
        im = ax.imshow(mat, vmin=-1.0, vmax=1.0, cmap="RdBu_r", aspect="auto")
        plt.colorbar(im, ax=ax, label="Spearman ρ", shrink=0.8)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        labels = [f.replace("_", "\n") for f in features]
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_title("Feature coupling matrix")
        ax_idx += 1

    if has_regimes:
        ax = axes[0][ax_idx]
        ox_col = "oxidative_index" if "oxidative_index" in regimes.columns else None
        mem_col = "membrane_integrity_index" if "membrane_integrity_index" in regimes.columns else None
        if ox_col and mem_col:
            palette = {
                "oxidative_dominant": "#D55E00",
                "lipid_dominant": "#0072B2",
                "protein_dominant": "#009E73",
                "unknown": "#AAAAAA",
            }
            for regime, grp in regimes.groupby("biochemical_regime"):
                color = palette.get(str(regime), "#888888")
                ax.scatter(
                    grp[mem_col].to_numpy(dtype=float),
                    grp[ox_col].to_numpy(dtype=float),
                    label=str(regime).replace("_", " "),
                    color=color, alpha=0.75, s=45,
                    edgecolors="white", linewidths=0.4,
                )
            ax.set_xlabel("Membrane integrity index")
            ax.set_ylabel("Oxidative index")
            ax.legend(fontsize=7)
        ax.set_title("Ratio–ratio biochemical regimes")
        ax_idx += 1

    fig.suptitle("Feature coupling and biochemical regimes", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Orchestrating runner
# ---------------------------------------------------------------------------


def run_journal_figures(
    *,
    panel_matrix_path: str | Path | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Phase U orchestrator: assemble 5 journal figures from prior phase outputs."""
    cfg = _cfg(cfg)
    phase_cfg = cfg.get("analysis", {}).get("journal_figures", {})
    if not bool(phase_cfg.get("enabled", True)):
        return {}

    panel_dir = get_confirmatory_panel_dir()
    interp_root = _interp_root()
    figures_dir = get_spectral_interpretability_dir("figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    anchor_feature = (
        cfg.get("analysis", {}).get("preregistered_panel", {})
        .get("primary_hypothesis", {}).get("feature", "oxidative_index")
    )

    pmat = (
        Path(panel_matrix_path) if panel_matrix_path is not None
        else panel_dir / "panel_feature_matrix.csv"
    )

    outputs: dict[str, Path] = {}

    # Figure 1
    fig1_path = figures_dir / "fig1_cohort_panel_overview.png"
    try:
        ok = _fig1_cohort_rainclouds(
            pmat,
            panel_dir / "panel_contrast_tests.csv",
            fig1_path,
        )
        if ok:
            outputs["fig1"] = fig1_path
    except Exception:
        logger.exception("Phase U: fig1 failed")

    # Figure 2
    fig2_path = figures_dir / "fig2_effect_size_forest_robustness.png"
    try:
        ok = _fig2_effect_forest(
            panel_dir / "panel_contrast_tests.csv",
            panel_dir / "feature_robustness_ranking.csv",
            anchor_feature,
            fig2_path,
        )
        if ok:
            outputs["fig2"] = fig2_path
    except Exception:
        logger.exception("Phase U: fig2 failed")

    # Figure 3
    fig3_path = figures_dir / "fig3_difference_spectrum_signed_effect.png"
    try:
        ok = _fig3_difference_spectrum(
            interp_root / "discriminant_signatures" / "difference_spectrum.csv",
            fig3_path,
        )
        if ok:
            outputs["fig3"] = fig3_path
    except Exception:
        logger.exception("Phase U: fig3 failed")

    # Figure 4
    fig4_path = figures_dir / "fig4_nmf_components_entropy.png"
    try:
        ok = _fig4_nmf_entropy(
            interp_root / "nmf_decomposition" / "nmf_components.csv",
            interp_root / "spectral_entropy" / "spectral_entropy.csv",
            interp_root / "spectral_entropy" / "spectral_entropy_associations.csv",
            fig4_path,
        )
        if ok:
            outputs["fig4"] = fig4_path
    except Exception:
        logger.exception("Phase U: fig4 failed")

    # Figure 5
    fig5_path = figures_dir / "fig5_feature_coupling_regimes.png"
    try:
        ok = _fig5_coupling_regimes(
            interp_root / "feature_coupling" / "feature_coupling_matrix.csv",
            interp_root / "feature_coupling" / "ratio_ratio_regimes.csv",
            fig5_path,
        )
        if ok:
            outputs["fig5"] = fig5_path
    except Exception:
        logger.exception("Phase U: fig5 failed")

    logger.info("Phase U: %d/%d figures produced in %s", len(outputs), 5, figures_dir)
    return outputs


def _interp_root() -> Path:
    from src import _output_dir
    return _output_dir("eda_subdir") / "06_spectral_interpretability"
