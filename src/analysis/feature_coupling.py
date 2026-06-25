"""Phase Q: cross-feature coupling and biochemical regime assignments.

Quantifies coupling between band integrals and named indices via pairwise
Spearman correlation with bootstrap CIs, then assigns per-subject biochemical
regimes using the oxidative_index × membrane_integrity_index ratio-ratio plane.

Outputs go to ``06_spectral_interpretability/feature_coupling/``.
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

from src import config, get_spectral_interpretability_dir
from src.analysis.io_utils import load_csv_or_empty, write_csv_atomic
from src.analysis.preregistered_panel import panel_cfg

logger = logging.getLogger(__name__)

_OXIDATIVE_COL = "oxidative_index"
_MEMBRANE_COL = "membrane_integrity_index"
_PROTEIN_COL = "protein_remodeling_index"


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
    rho = float(stats.spearmanr(xf, yf).statistic)
    if n_bootstrap <= 0:
        return rho, np.nan, np.nan
    boot = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot[i] = float(stats.spearmanr(xf[idx], yf[idx]).statistic)
    ci_lower = float(np.nanpercentile(boot, 2.5))
    ci_upper = float(np.nanpercentile(boot, 97.5))
    return rho, ci_lower, ci_upper


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
        for j, col_b in enumerate(cols):
            if j <= i:
                continue
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


# ---------------------------------------------------------------------------
# Biochemical regime assignment
# ---------------------------------------------------------------------------


def assign_biochemical_regimes(
    feature_frame: pd.DataFrame,
    *,
    oxidative_col: str = _OXIDATIVE_COL,
    membrane_col: str = _MEMBRANE_COL,
) -> pd.DataFrame:
    """Assign per-subject biochemical regime from oxidative × membrane plane.

    Regime logic (median-split):
    - high oxidative_index → ``oxidative_dominant``
    - low oxidative_index + high membrane_integrity_index → ``lipid_dominant``
    - low oxidative_index + low membrane_integrity_index → ``protein_dominant``

    Returns a copy of the input with an added ``biochemical_regime`` column.
    """
    result = feature_frame.copy()
    has_ox = oxidative_col in result.columns
    has_mem = membrane_col in result.columns

    if not has_ox and not has_mem:
        result["biochemical_regime"] = "unknown"
        return result

    ox = result[oxidative_col].to_numpy(dtype=float) if has_ox else np.full(len(result), np.nan)
    mem = result[membrane_col].to_numpy(dtype=float) if has_mem else np.full(len(result), np.nan)

    ox_med = float(np.nanmedian(ox))
    mem_med = float(np.nanmedian(mem))

    regimes: list[str] = []
    for o_val, m_val in zip(ox, mem):
        if not np.isfinite(o_val) and not np.isfinite(m_val):
            regimes.append("unknown")
        elif not np.isfinite(o_val):
            regimes.append("lipid_dominant" if m_val >= mem_med else "protein_dominant")
        elif o_val >= ox_med:
            regimes.append("oxidative_dominant")
        elif np.isfinite(m_val) and m_val >= mem_med:
            regimes.append("lipid_dominant")
        else:
            regimes.append("protein_dominant")

    result["biochemical_regime"] = regimes
    return result


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _plot_coupling_heatmap(
    coupling: pd.DataFrame,
    *,
    output_path: Path,
) -> None:
    """Symmetric coupling heatmap of Spearman rho values."""
    if coupling.empty or "feature_a" not in coupling.columns:
        return

    features = sorted(set(coupling["feature_a"]) | set(coupling["feature_b"]))
    n = len(features)
    if n < 2:
        return

    idx = {f: i for i, f in enumerate(features)}
    mat = np.full((n, n), np.nan)
    for _, row in coupling.iterrows():
        i = idx[str(row["feature_a"])]
        j = idx[str(row["feature_b"])]
        rho = float(row["spearman_rho"]) if np.isfinite(float(row["spearman_rho"])) else np.nan
        mat[i, j] = rho
        mat[j, i] = rho
    np.fill_diagonal(mat, 1.0)

    fig, ax = plt.subplots(figsize=(max(4, n * 0.9), max(3.5, n * 0.85)))
    im = ax.imshow(mat, vmin=-1.0, vmax=1.0, cmap="RdBu_r", aspect="auto")
    plt.colorbar(im, ax=ax, label="Spearman ρ")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    labels = [f.replace("_", "\n") for f in features]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = mat[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if abs(val) > 0.6 else "black")

    ax.set_title("Feature coupling — Spearman ρ (bootstrapped 95 % CI)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_regime_scatter(
    regimes: pd.DataFrame,
    *,
    oxidative_col: str,
    membrane_col: str,
    output_path: Path,
) -> None:
    """Scatter plot of oxidative × membrane plane coloured by regime."""
    if regimes.empty or "biochemical_regime" not in regimes.columns:
        return
    if oxidative_col not in regimes.columns or membrane_col not in regimes.columns:
        return

    palette = {
        "oxidative_dominant": "#D55E00",
        "lipid_dominant": "#0072B2",
        "protein_dominant": "#009E73",
        "unknown": "#AAAAAA",
    }
    fig, ax = plt.subplots(figsize=(5, 4))
    for regime, group in regimes.groupby("biochemical_regime"):
        color = palette.get(str(regime), "#888888")
        ax.scatter(
            group[membrane_col].to_numpy(dtype=float),
            group[oxidative_col].to_numpy(dtype=float),
            label=str(regime).replace("_", " "),
            color=color,
            alpha=0.75,
            edgecolors="white",
            linewidths=0.5,
            s=50,
        )
    ax.set_xlabel(membrane_col.replace("_", " ") + " (a.u.)")
    ax.set_ylabel(oxidative_col.replace("_", " ") + " (a.u.)")
    ax.set_title("Ratio–ratio biochemical regimes")
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
    """Phase Q orchestrator: coupling matrix + biochemical regimes + figures."""
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
    target_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = target_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load panel feature matrix (integrals + named indices)
    matrix_path = (
        Path(panel_matrix_path)
        if panel_matrix_path is not None
        else _default_panel_path()
    )
    frame = load_csv_or_empty(matrix_path)
    if frame.empty:
        logger.warning("Phase Q: panel feature matrix not found at %s; skipping", matrix_path)
        return {}

    # Feature columns: indices listed first, then integrals
    index_names = list(panel.get("indices", {}).keys())
    integral_names = list(panel.get("integrals", {}).keys())
    # Keep only those actually in the frame (no fallback columns)
    feature_cols = [c for c in index_names + integral_names if c in frame.columns]
    if len(feature_cols) < 2:
        logger.warning("Phase Q: fewer than 2 feature columns found; skipping")
        return {}

    n_bootstrap = int(phase_cfg.get("n_bootstrap", panel.get("n_bootstrap", 500)))
    random_state = int(phase_cfg.get("random_state", panel.get("random_state", 42)))

    # 1. Coupling matrix
    coupling = compute_coupling_matrix(
        frame, feature_cols,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )
    coupling_path = target_dir / "feature_coupling_matrix.csv"
    write_csv_atomic(coupling, coupling_path, index=False)

    # 2. Biochemical regimes
    regimes = assign_biochemical_regimes(
        frame[["subject_id"] + [c for c in feature_cols if c in frame.columns]]
        if "subject_id" in frame.columns
        else frame[feature_cols],
        oxidative_col=_OXIDATIVE_COL,
        membrane_col=_MEMBRANE_COL,
    )
    regime_cols = ["subject_id", _OXIDATIVE_COL, _MEMBRANE_COL, _PROTEIN_COL, "biochemical_regime"]
    regime_out_cols = [c for c in regime_cols if c in regimes.columns]
    regimes_path = target_dir / "ratio_ratio_regimes.csv"
    write_csv_atomic(regimes[regime_out_cols], regimes_path, index=False)

    outputs: dict[str, Path] = {
        "feature_coupling_matrix": coupling_path,
        "ratio_ratio_regimes": regimes_path,
    }

    # 3. Figures
    heatmap_path = figures_dir / "feature_coupling_heatmap.png"
    _plot_coupling_heatmap(coupling, output_path=heatmap_path)
    outputs["feature_coupling_heatmap"] = heatmap_path

    regime_plot_path = figures_dir / "ratio_ratio_regime_plot.png"
    _plot_regime_scatter(
        regimes,
        oxidative_col=_OXIDATIVE_COL,
        membrane_col=_MEMBRANE_COL,
        output_path=regime_plot_path,
    )
    outputs["ratio_ratio_regime_plot"] = regime_plot_path

    logger.info("Phase Q outputs written to %s", target_dir)
    return outputs


def _default_panel_path() -> Path:
    from src import get_confirmatory_panel_dir
    return get_confirmatory_panel_dir() / "panel_feature_matrix.csv"
