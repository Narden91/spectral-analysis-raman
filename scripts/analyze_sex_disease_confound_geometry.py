"""Diagnose whether the disease PLS-DA direction is a sex proxy.

Exploratory, read-only manuscript-support analysis. It answers a question the
primary case-control classifier (`src/analysis/control_vs_disease_ml.py`) does
not: does the whole-spectrum direction that separates stroke from control
overlap with the direction that separates male from female participants?

This script never touches the production `Results_*` trees or the primary
nested-CV classifier; it reads only the already-materialized participant-level
spectral matrix and writes a small CSV plus one figure under
``manuscript/paper``. It is not a replacement for the primary classifier and is
not subject to the same permutation/bootstrap validation; it is a geometric
diagnostic reported alongside the confound checks already in the manuscript.
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
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.visualization.theme import panel_width_in, set_manuscript_theme

SUBJECT_SPECTRA = (
    ROOT
    / "Results_control_vs_disease"
    / "04_cross_region_summary"
    / "01_subject_features"
    / "subject_spectra.csv"
)
FIGURE_DIR = ROOT / "manuscript" / "paper" / "figures"
FIGURE_DATA_DIR = ROOT / "manuscript" / "paper" / "figure_data"

RANDOM_STATE = 42
N_OUTER_SPLITS = 5
COMPONENT_GRID = (1, 2, 3, 4, 5)
CONTROL = "#0072B2"
DISEASE = "#D55E00"
NEUTRAL = "#4D4D4D"


# The SI float assembles the two panels side by side at 0.48 of the text block,
# so each is authored at the width its \includegraphics requests.
PANEL_FRACTION = 0.48
PANEL_HEIGHT_IN = 2.5


def _panel_axes():
    """One axes sized for half of the assembled confound-geometry float."""
    return plt.subplots(
        figsize=(panel_width_in(PANEL_FRACTION), PANEL_HEIGHT_IN),
        constrained_layout=True,
    )


def spectral_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    columns = [c for c in frame.columns if c.startswith(("1300__", "2850__"))]
    return frame[columns].to_numpy(dtype=float), columns


def select_n_components(
    X: np.ndarray, y: np.ndarray, rng_seed: int
) -> int:
    """Pick PLS component count by mean CV ROC-AUC on a fixed inner split."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=rng_seed)
    best_n, best_auc = 1, -np.inf
    max_comp = min(X.shape[0] - 2, X.shape[1])
    for n_comp in [c for c in COMPONENT_GRID if 1 <= c <= max_comp]:
        scores = np.zeros(y.shape[0], dtype=float)
        for tr_idx, te_idx in cv.split(X, y):
            scaler = StandardScaler().fit(X[tr_idx])
            pls = PLSRegression(n_components=n_comp, scale=False)
            pls.fit(scaler.transform(X[tr_idx]), y[tr_idx])
            scores[te_idx] = pls.predict(scaler.transform(X[te_idx])).ravel()
        auc = roc_auc_score(y, scores)
        if auc > best_auc:
            best_auc, best_n = auc, n_comp
    return best_n


def outer_cv_auc(X: np.ndarray, y: np.ndarray, n_comp: int) -> tuple[float, np.ndarray]:
    """Repeated-fold-free single 5-fold outer AUC and out-of-fold scores."""
    cv = StratifiedKFold(n_splits=N_OUTER_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_scores = np.zeros(y.shape[0], dtype=float)
    for tr_idx, te_idx in cv.split(X, y):
        scaler = StandardScaler().fit(X[tr_idx])
        pls = PLSRegression(n_components=n_comp, scale=False)
        pls.fit(scaler.transform(X[tr_idx]), y[tr_idx])
        oof_scores[te_idx] = pls.predict(scaler.transform(X[te_idx])).ravel()
    return roc_auc_score(y, oof_scores), oof_scores


def full_fit_direction(X: np.ndarray, y: np.ndarray, n_comp: int) -> np.ndarray:
    """Full-sample PLS discriminant direction, standardized coefficient vector."""
    scaler = StandardScaler().fit(X)
    pls = PLSRegression(n_components=n_comp, scale=False)
    pls.fit(scaler.transform(X), y)
    coef = np.asarray(pls.coef_).reshape(-1)
    return coef / np.linalg.norm(coef)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    set_manuscript_theme()

    frame = pd.read_csv(SUBJECT_SPECTRA)
    X, _ = spectral_matrix(frame)
    y_disease = frame["condition"].eq("disease").to_numpy(dtype=int)
    y_female = frame["gender"].eq("female").to_numpy(dtype=int)

    n_comp_disease = select_n_components(X, y_disease, RANDOM_STATE)
    n_comp_sex = select_n_components(X, y_female, RANDOM_STATE)

    auc_disease, oof_disease = outer_cv_auc(X, y_disease, n_comp_disease)
    auc_sex, oof_sex = outer_cv_auc(X, y_female, n_comp_sex)

    direction_disease = full_fit_direction(X, y_disease, n_comp_disease)
    direction_sex = full_fit_direction(X, y_female, n_comp_sex)
    cosine_similarity = float(np.dot(direction_disease, direction_sex))

    # Partial correlation of disease score with disease label, controlling
    # for the sex-discriminant projection, as a complementary linear check.
    sex_projection = X @ direction_sex
    disease_projection = X @ direction_disease
    residual_disease_score = disease_projection - np.polyval(
        np.polyfit(sex_projection, disease_projection, 1), sex_projection
    )
    residual_auc = roc_auc_score(y_disease, residual_disease_score)

    summary = pd.DataFrame(
        [
            {
                "quantity": "disease_whole_spectrum_oof_auc",
                "value": auc_disease,
                "n_components": n_comp_disease,
            },
            {
                "quantity": "sex_whole_spectrum_oof_auc",
                "value": auc_sex,
                "n_components": n_comp_sex,
            },
            {
                "quantity": "cosine_similarity_disease_sex_direction",
                "value": cosine_similarity,
                "n_components": np.nan,
            },
            {
                "quantity": "disease_auc_after_removing_sex_direction",
                "value": residual_auc,
                "n_components": np.nan,
            },
        ]
    )
    summary.to_csv(
        FIGURE_DATA_DIR / "sex_disease_confound_geometry.csv", index=False
    )

    # One standalone PDF per panel: the SI float assembles them with
    # \subcaption, so the panel letters are typeset by LaTeX rather than
    # baked into the figure, and stay referenceable.
    fig_auc, ax = _panel_axes()
    labels = ["Disease\n(whole spectrum)", "Sex\n(whole spectrum)"]
    values = [auc_disease, auc_sex]
    colors = [DISEASE, CONTROL]
    ax.bar(labels, values, color=colors, width=0.55)
    ax.axhline(0.5, color=NEUTRAL, linestyle="--", linewidth=0.9)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("Out-of-fold ROC-AUC")
    ax.spines[["top", "right"]].set_visible(False)
    for index, value in enumerate(values):
        ax.annotate(f"{value:.3f}", (index, value), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=8)

    fig_projection, ax = _panel_axes()
    ax.scatter(sex_projection[y_disease == 0], disease_projection[y_disease == 0],
               s=16, color=CONTROL, alpha=0.75, label="Control")
    ax.scatter(sex_projection[y_disease == 1], disease_projection[y_disease == 1],
               s=16, color=DISEASE, alpha=0.75, label="Ischemic stroke")
    ax.set_xlabel("Sex-discriminant projection")
    ax.set_ylabel("Disease-discriminant projection")
    ax.set_title(f"cosine = {cosine_similarity:.2f}", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="best", fontsize=7.5)

    for fig, stem in (
        (fig_auc, "sex_disease_confound_auc"),
        (fig_projection, "sex_disease_confound_projections"),
    ):
        fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=600, bbox_inches="tight")
        plt.close(fig)

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
