"""Compare disease and sex score geometry using nested held-out predictions.

The full-cohort projection plot is descriptive. Predictive AUCs, including the
residual-score AUC, use training-only model selection and residualization.
"""

from __future__ import annotations

# ruff: noqa: E402
# Direct execution needs project imports after path and plotting-backend setup.
import argparse
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

from scripts.source_provenance import write_source_provenance
from src.data_preparation.manifest import hash_file
from src.visualization.theme import panel_width_in, set_manuscript_theme

SUBJECT_SPECTRA = (
    ROOT
    / "Results_control_vs_disease"
    / "04_cross_region_summary"
    / "01_subject_features"
    / "subject_spectra.csv"
)

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


def select_n_components(X: np.ndarray, y: np.ndarray, rng_seed: int) -> int:
    """Select component count using only the supplied training participants."""
    counts = np.bincount(y, minlength=2)
    n_splits = min(5, int(counts.min()))
    if n_splits < 2:
        raise ValueError("Insufficient training participants per class for inner CV")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rng_seed)
    folds = list(cv.split(X, y))
    max_comp = min(X.shape[1], min(len(train) - 1 for train, _ in folds))
    candidates = [c for c in COMPONENT_GRID if 1 <= c <= max_comp]
    if not candidates:
        raise ValueError("No PLS component count is supported by the inner folds")
    best_n, best_auc = candidates[0], -np.inf
    for n_comp in candidates:
        scores = np.zeros(y.shape[0], dtype=float)
        for train, test in folds:
            scores[test] = _projection_scores(X[train], y[train], X[test], n_comp)
        auc = roc_auc_score(y, scores)
        if auc > best_auc:
            best_auc, best_n = auc, n_comp
    return best_n


def _projection_scores(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, n_comp: int) -> np.ndarray:
    scaler = StandardScaler().fit(X_train)
    pls = PLSRegression(n_components=n_comp, scale=False)
    pls.fit(scaler.transform(X_train), y_train)
    return pls.predict(scaler.transform(X_test)).ravel()


def outer_cv_auc(X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Return nested held-out AUC and scores, without whole-cohort tuning."""
    n_splits = min(N_OUTER_SPLITS, int(np.bincount(y, minlength=2).min()))
    if n_splits < 2:
        raise ValueError("Insufficient participants per class for outer CV")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = np.zeros(len(y), dtype=float)
    for train, test in cv.split(X, y):
        n_comp = select_n_components(X[train], y[train], RANDOM_STATE)
        scores[test] = _projection_scores(X[train], y[train], X[test], n_comp)
    return float(roc_auc_score(y, scores)), scores


def nested_confound_predictions(
    X: np.ndarray, y_disease: np.ndarray, y_sex: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Fit both scores and their residualization inside each common outer training fold."""
    if X.shape[0] != len(y_disease) or y_disease.shape != y_sex.shape or not np.isfinite(X).all():
        raise ValueError("Confound geometry requires aligned finite participant-level inputs")
    if set(np.unique(y_disease)) != {0, 1} or set(np.unique(y_sex)) != {0, 1}:
        raise ValueError("Confound geometry requires two observed disease and sex classes")
    joint = y_disease * 2 + y_sex
    _, joint_counts = np.unique(joint, return_counts=True)
    n_splits = min(N_OUTER_SPLITS, int(joint_counts.min()))
    if n_splits < 2:
        raise ValueError("Insufficient participants in a joint disease/sex stratum")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    disease_scores = np.empty(len(y_disease), dtype=float)
    sex_scores = np.empty(len(y_sex), dtype=float)
    residual_scores = np.empty(len(y_disease), dtype=float)
    settings = []
    for fold, (train, test) in enumerate(cv.split(X, joint)):
        nd = select_n_components(X[train], y_disease[train], RANDOM_STATE)
        ns = select_n_components(X[train], y_sex[train], RANDOM_STATE)
        scaler = StandardScaler().fit(X[train])
        train_scaled, test_scaled = scaler.transform(X[train]), scaler.transform(X[test])
        disease_model = PLSRegression(n_components=nd, scale=False).fit(train_scaled, y_disease[train])
        sex_model = PLSRegression(n_components=ns, scale=False).fit(train_scaled, y_sex[train])
        disease_train = disease_model.predict(train_scaled).ravel()
        sex_train = sex_model.predict(train_scaled).ravel()
        adjustment = np.linalg.lstsq(np.column_stack([np.ones(len(train)), sex_train]), disease_train, rcond=None)[0]
        disease_test = disease_model.predict(test_scaled).ravel()
        sex_test = sex_model.predict(test_scaled).ravel()
        disease_scores[test], sex_scores[test] = disease_test, sex_test
        residual_scores[test] = disease_test - np.column_stack([np.ones(len(test)), sex_test]) @ adjustment
        settings.append({"fold": fold, "n_train": len(train), "n_test": len(test),
                         "disease_n_components": nd, "sex_n_components": ns})
    return disease_scores, sex_scores, residual_scores, pd.DataFrame(settings)


def full_fit_direction(X: np.ndarray, y: np.ndarray, n_comp: int) -> np.ndarray:
    """Full-sample PLS discriminant direction, standardized coefficient vector."""
    scaler = StandardScaler().fit(X)
    pls = PLSRegression(n_components=n_comp, scale=False)
    pls.fit(scaler.transform(X), y)
    coef = np.asarray(pls.coef_).reshape(-1)
    norm = np.linalg.norm(coef)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Cannot define a direction for a degenerate fitted model")
    return coef / norm


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=SUBJECT_SPECTRA)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    frame = pd.read_csv(args.input, dtype={"subject_id": "string"})
    if "subject_id" not in frame or frame["subject_id"].isna().any() or frame["subject_id"].duplicated().any():
        raise ValueError("Confound geometry requires one identified row per participant")
    disease = frame["condition"].astype("string").str.lower()
    sex = frame["gender"].astype("string").str.lower()
    if not disease.isin(["control", "disease"]).all() or not sex.isin(["female", "male"]).all():
        raise ValueError("Confound geometry requires recorded control/disease and female/male labels")
    X, columns = spectral_matrix(frame)
    if not columns:
        raise ValueError("No spectral columns were found in the input table")
    y_disease = disease.eq("disease").to_numpy(dtype=int)
    y_female = sex.eq("female").to_numpy(dtype=int)
    disease_scores, sex_scores, residual_scores, settings = nested_confound_predictions(X, y_disease, y_female)
    auc_disease = roc_auc_score(y_disease, disease_scores)
    auc_sex = roc_auc_score(y_female, sex_scores)
    residual_auc = roc_auc_score(y_disease, residual_scores)

    n_comp_disease = select_n_components(X, y_disease, RANDOM_STATE)
    n_comp_sex = select_n_components(X, y_female, RANDOM_STATE)
    direction_disease = full_fit_direction(X, y_disease, n_comp_disease)
    direction_sex = full_fit_direction(X, y_female, n_comp_sex)
    cosine_similarity = float(np.dot(direction_disease, direction_sex))
    descriptive_scaled = StandardScaler().fit_transform(X)
    sex_projection = descriptive_scaled @ direction_sex
    disease_projection = descriptive_scaled @ direction_disease
    figure_dir = args.output_dir / "figures"
    figure_data_dir = args.output_dir / "figure_data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_data_dir.mkdir(parents=True, exist_ok=True)
    set_manuscript_theme()

    summary = pd.DataFrame(
        [
            {
                "quantity": "disease_whole_spectrum_oof_auc",
                "value": auc_disease,
                "estimation_scope": "nested_held_out",
            },
            {
                "quantity": "sex_whole_spectrum_oof_auc",
                "value": auc_sex,
                "estimation_scope": "nested_held_out",
            },
            {
                "quantity": "cosine_similarity_disease_sex_direction",
                "value": cosine_similarity,
                "estimation_scope": "full_cohort_descriptive",
            },
            {
                "quantity": "disease_auc_after_removing_sex_direction",
                "value": residual_auc,
                "estimation_scope": "nested_held_out",
            },
        ]
    )
    summary.to_csv(
        figure_data_dir / "sex_disease_confound_geometry.csv", index=False
    )

    settings.to_csv(figure_data_dir / "sex_disease_confound_fold_settings.csv", index=False)
    write_source_provenance(args.output_dir, source_files=[args.input], rendering_settings={
        "component_grid": list(COMPONENT_GRID), "random_state": RANDOM_STATE,
        "method": "nested_training_only_selection_and_residualization",
        "projection_scope": "full_cohort_descriptive",
        "script_sha256": hash_file(Path(__file__)),
    })

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
    ax.set_xlabel("Sex projection (in-sample descriptive)")
    ax.set_ylabel("Disease projection (in-sample descriptive)")
    ax.set_title(f"cosine = {cosine_similarity:.2f}", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="best", fontsize=7.5)

    for fig, stem in (
        (fig_auc, "sex_disease_confound_auc"),
        (fig_projection, "sex_disease_confound_projections"),
    ):
        fig.savefig(figure_dir / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(figure_dir / f"{stem}.png", dpi=600, bbox_inches="tight")
        plt.close(fig)

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
