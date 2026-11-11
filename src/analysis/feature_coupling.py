"""Phase Q: cross-feature coupling and descriptive ratio groups.

Quantifies pairwise Spearman correlations between band integrals and ratios,
then describes cohort median splits of the carotenoid-to-carbonyl and
CH-bulk-to-carbonyl ratios without assigning molecular or biological states.

Outputs go to ``06_spectral_interpretability/05_feature_coupling/``.
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
from scipy import stats

from src import config, get_confirmatory_panel_dir, get_spectral_interpretability_dir
from src.analysis.io_utils import load_csv_or_empty, write_csv_atomic
from src.analysis.preregistered_panel import panel_cfg

logger = logging.getLogger(__name__)

_CAROTENOID_COL = "carotenoid_to_carbonyl_ratio"
_CH_BULK_COL = "ch_bulk_to_carbonyl_ratio"
_TRYPTOPHAN_RATIO_COL = "tryptophan_to_sugar_beta_sheet_ratio"

REGIME_PALETTE: dict[str, str] = {
    "upper_carotenoid_ratio": "#D55E00",
    "lower_carotenoid_upper_ch_bulk_ratio": "#0072B2",
    "lower_carotenoid_lower_ch_bulk_ratio": "#009E73",
    "unknown": "#AAAAAA",
}


# ---------------------------------------------------------------------------
# Coupling matrix
# ---------------------------------------------------------------------------


def _spearman_with_bootstrap(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Return (rho, ci_lower, ci_upper) via percentile bootstrap."""
    finite = np.isfinite(x) & np.isfinite(y)
    xf, yf = x[finite], y[finite]
    n = int(xf.size)
    if n < 4:
        return np.nan, np.nan, np.nan
    if np.std(xf) == 0 or np.std(yf) == 0:
        return np.nan, np.nan, np.nan
    rho = float(stats.spearmanr(xf, yf).statistic)
    if n_bootstrap <= 0:
        return rho, np.nan, np.nan
    boot_values: list[float] = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        x_boot, y_boot = xf[indices], yf[indices]
        if np.std(x_boot) == 0 or np.std(y_boot) == 0:
            boot_values.append(np.nan)
        else:
            boot_values.append(float(stats.spearmanr(x_boot, y_boot).statistic))
    boot = np.asarray(boot_values, dtype=float)
    return rho, float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))


def compute_coupling_matrix(
    feature_frame: pd.DataFrame,
    feature_cols: list[str],
    *,
    n_bootstrap: int = 500,
    random_state: int = 42,
) -> pd.DataFrame:
    """Pairwise Spearman correlation with 95 % bootstrap CI for all feature pairs.

    Returns long-form DataFrame: feature_a, feature_b, spearman_rho,
    bootstrap_ci_lower, bootstrap_ci_upper, n_subjects.
    """
    cols = [c for c in feature_cols if c in feature_frame.columns]
    if len(cols) < 2:
        return pd.DataFrame(
            columns=["feature_a", "feature_b", "spearman_rho",
                     "bootstrap_ci_lower", "bootstrap_ci_upper", "n_subjects"]
        )

    rng = np.random.default_rng(random_state)
    arr = feature_frame[cols].to_numpy(dtype=float)
    rows: list[dict[str, object]] = []
    for i, col_a in enumerate(cols):
        for j in range(i + 1, len(cols)):
            col_b = cols[j]
            x, y = arr[:, i], arr[:, j]
            n_valid = int(np.sum(np.isfinite(x) & np.isfinite(y)))
            rho, ci_lo, ci_hi = _spearman_with_bootstrap(x, y, n_bootstrap=n_bootstrap, rng=rng)
            rows.append({
                "feature_a": col_a,
                "feature_b": col_b,
                "spearman_rho": rho,
                "bootstrap_ci_lower": ci_lo,
                "bootstrap_ci_upper": ci_hi,
                "n_subjects": n_valid,
            })

    return pd.DataFrame(rows)


def coupling_df_to_matrix(coupling: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Convert long-form coupling DataFrame to symmetric numpy matrix + feature list.

    Returns (matrix, features) where matrix[i, j] = Spearman ρ for features[i] vs features[j].
    """
    features = sorted(set(coupling["feature_a"]) | set(coupling["feature_b"]))
    n = len(features)
    idx = {f: i for i, f in enumerate(features)}
    mat = np.full((n, n), np.nan)
    for _, row in coupling.iterrows():
        i, j = idx[str(row["feature_a"])], idx[str(row["feature_b"])]
        rho = float(row["spearman_rho"])
        if np.isfinite(rho):
            mat[i, j] = rho
            mat[j, i] = rho
    np.fill_diagonal(mat, 1.0)
    return mat, features


# ---------------------------------------------------------------------------
# Descriptive ratio groups
# ---------------------------------------------------------------------------


def assign_ratio_groups(
    feature_frame: pd.DataFrame,
    *,
    carotenoid_col: str = _CAROTENOID_COL,
    ch_bulk_col: str = _CH_BULK_COL,
) -> pd.DataFrame:
    """Describe cohort median splits of two ratios without assigning biology."""
    result = feature_frame.copy()
    has_ox = carotenoid_col in result.columns
    has_mem = ch_bulk_col in result.columns

    if not has_ox and not has_mem:
        result["ratio_group"] = "unknown"
        return result

    ox = result[carotenoid_col].to_numpy(dtype=float) if has_ox else np.full(len(result), np.nan)
    mem = result[ch_bulk_col].to_numpy(dtype=float) if has_mem else np.full(len(result), np.nan)
    ox_finite = ox[np.isfinite(ox)]
    mem_finite = mem[np.isfinite(mem)]
    ox_med = float(np.median(ox_finite)) if ox_finite.size else np.nan
    mem_med = float(np.median(mem_finite)) if mem_finite.size else np.nan

    def _classify(o_val: float, m_val: float) -> str:
        if not np.isfinite(o_val) or not np.isfinite(m_val):
            return "unknown"
        if o_val >= ox_med:
            return "upper_carotenoid_ratio"
        if np.isfinite(m_val) and m_val >= mem_med:
            return "lower_carotenoid_upper_ch_bulk_ratio"
        return "lower_carotenoid_lower_ch_bulk_ratio"

    result["ratio_group"] = [_classify(o, m) for o, m in zip(ox, mem)]
    return result


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _plot_coupling_heatmap(coupling: pd.DataFrame, *, output_path: Path) -> None:
    """Symmetric coupling heatmap of Spearman rho values."""
    if coupling.empty or "feature_a" not in coupling.columns:
        return
    mat, features = coupling_df_to_matrix(coupling)
    n = len(features)
    if n < 2:
        return

    fig, ax = plt.subplots(figsize=(max(4, n * 0.9), max(3.5, n * 0.85)))
    im = ax.imshow(mat, vmin=-1.0, vmax=1.0, cmap="RdBu_r", aspect="auto")
    plt.colorbar(im, ax=ax, label="Spearman ρ")
    labels = [f.replace("_", "\n") for f in features]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    for i in range(n):
        for j in range(n):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if abs(mat[i, j]) > 0.6 else "black")
    ax.set_title("Feature coupling — Spearman ρ (bootstrapped 95 % CI)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_regime_scatter(
    regimes: pd.DataFrame,
    *,
    carotenoid_col: str,
    ch_bulk_col: str,
    output_path: Path,
) -> None:
    """Scatter plot of carotenoid ratio × CH-bulk ratio plane coloured by descriptive median group."""
    if regimes.empty or "ratio_group" not in regimes.columns:
        return
    if carotenoid_col not in regimes.columns or ch_bulk_col not in regimes.columns:
        return

    fig, ax = plt.subplots(figsize=(5, 4))
    for regime, group in regimes.groupby("ratio_group"):
        ax.scatter(
            group[ch_bulk_col].to_numpy(dtype=float),
            group[carotenoid_col].to_numpy(dtype=float),
            label=str(regime).replace("_", " "),
            color=REGIME_PALETTE.get(str(regime), "#888888"),
            alpha=0.75, edgecolors="white", linewidths=0.5, s=50,
        )
    ax.set_xlabel(ch_bulk_col.replace("_", " ") + " (a.u.)")
    ax.set_ylabel(carotenoid_col.replace("_", " ") + " (a.u.)")
    ax.set_title("Descriptive ratio median groups")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestrating runner
# ---------------------------------------------------------------------------


def run_feature_coupling(
    *,
    panel_matrix_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Phase Q orchestrator: coupling matrix + descriptive ratio groups + figures."""
    if cfg is None:
        cfg = config
    panel = panel_cfg(cfg)
    phase_cfg = cfg.get("analysis", {}).get("feature_coupling", {})
    if not bool(phase_cfg.get("enabled", True)):
        return {}

    target_dir = (
        Path(output_dir) if output_dir is not None
        else get_spectral_interpretability_dir("coupling")
    )
    figures_dir = target_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    resolved_matrix = (
        Path(panel_matrix_path) if panel_matrix_path is not None
        else get_confirmatory_panel_dir() / "panel_feature_matrix.csv"
    )
    frame = load_csv_or_empty(resolved_matrix)
    if frame.empty:
        logger.warning("Phase Q: panel feature matrix not found at %s; skipping", resolved_matrix)
        return {}

    feature_cols = [
        c for c in list(panel.get("indices", {}).keys()) + list(panel.get("integrals", {}).keys())
        if c in frame.columns
    ]
    if len(feature_cols) < 2:
        logger.warning("Phase Q: fewer than 2 feature columns found; skipping")
        return {}

    n_bootstrap = int(phase_cfg.get("n_bootstrap", panel.get("n_bootstrap", 500)))
    random_state = int(phase_cfg.get("random_state", panel.get("random_state", 42)))

    coupling = compute_coupling_matrix(frame, feature_cols, n_bootstrap=n_bootstrap, random_state=random_state)
    coupling_path = target_dir / "feature_coupling_matrix.csv"
    write_csv_atomic(coupling, coupling_path, index=False)

    regime_input_cols = ["subject_id"] + feature_cols if "subject_id" in frame.columns else feature_cols
    regimes = assign_ratio_groups(
        frame[[c for c in regime_input_cols if c in frame.columns]],
        carotenoid_col=_CAROTENOID_COL,
        ch_bulk_col=_CH_BULK_COL,
    )
    regime_out_cols = [c for c in ["subject_id", _CAROTENOID_COL, _CH_BULK_COL, _TRYPTOPHAN_RATIO_COL, "ratio_group"]
                       if c in regimes.columns]
    regimes_path = target_dir / "ratio_ratio_regimes.csv"
    write_csv_atomic(regimes[regime_out_cols], regimes_path, index=False)

    heatmap_path = figures_dir / "feature_coupling_heatmap.png"
    _plot_coupling_heatmap(coupling, output_path=heatmap_path)

    regime_plot_path = figures_dir / "ratio_ratio_regime_plot.png"
    _plot_regime_scatter(regimes, carotenoid_col=_CAROTENOID_COL, ch_bulk_col=_CH_BULK_COL, output_path=regime_plot_path)

    logger.info("Phase Q outputs written to %s", target_dir)
    return {
        "feature_coupling_matrix": coupling_path,
        "ratio_ratio_regimes": regimes_path,
        "feature_coupling_heatmap": heatmap_path,
        "ratio_ratio_regime_plot": regime_plot_path,
    }
