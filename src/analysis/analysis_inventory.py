"""Analysis inventory and manuscript recommendation generator.

Inspects existing EDA/ML outputs and produces a machine-readable inventory
plus a markdown recommendation report.  It does not rerun science.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.analysis._reporting import dataframe_to_markdown

logger = logging.getLogger(__name__)


# Canonical analysis surface known to this repo.
_INVENTORY_TEMPLATE: list[dict] = [
    {
        "analysis_name": "group_mean_spectra",
        "track": "EDA",
        "phase": "02_group_comparisons",
        "primary_question": "What is the average spectrum per sex/condition group?",
        "artifact_examples": "group_mean_spectra.png",
        "scientific_value": "High — establishes baseline spectral morphology and group differences.",
        "overlap_with": "condition_region, cohort_mean_spectra",
        "recommended_status": "keep_main",
        "manuscript_recommendation": "main",
        "notes": "Use as primary spectral-overview figure.",
    },
    {
        "analysis_name": "difference_spectra",
        "track": "EDA",
        "phase": "02_group_comparisons",
        "primary_question": "Which wavenumbers differ between disease and control per gender?",
        "artifact_examples": "difference_spectrum_{gender}.png",
        "scientific_value": "Medium — direct contrast but restates group-mean information.",
        "overlap_with": "volcano, condition_region",
        "recommended_status": "move_to_supplement",
        "manuscript_recommendation": "supplement",
        "notes": "Consider demoting when volcano plots and group means are shown.",
    },
    {
        "analysis_name": "band_boxplots",
        "track": "EDA",
        "phase": "04_band_analysis",
        "primary_question": "How do known biochemical band intensities vary across groups?",
        "artifact_examples": "band_boxplots.png",
        "scientific_value": "Medium — interpretable but can be redundant with volcano/summary spectra.",
        "overlap_with": "band_rainclouds, volcano",
        "recommended_status": "move_to_supplement",
        "manuscript_recommendation": "supplement",
        "notes": "Prefer rainclouds if distributions are non-normal.",
    },
    {
        "analysis_name": "band_rainclouds",
        "track": "EDA",
        "phase": "04_band_analysis",
        "primary_question": "What is the full distribution of band intensities per group?",
        "artifact_examples": "band_rainclouds.png",
        "scientific_value": "Medium — detailed distributional information.",
        "overlap_with": "band_boxplots",
        "recommended_status": "demote_optional",
        "manuscript_recommendation": "candidate_removal",
        "notes": "Keep only if distributional shape is a key claim.",
    },
    {
        "analysis_name": "pca",
        "track": "EDA",
        "phase": "03_pca",
        "primary_question": "What is the global linear structure of the cohort?",
        "artifact_examples": "pca_scores_pc1_pc2.png, pca_variance_{range}.csv",
        "scientific_value": "High — subject-level global structure and variance decomposition.",
        "overlap_with": "",
        "recommended_status": "keep_main",
        "manuscript_recommendation": "main",
        "notes": "Include scree plot and loading interpretation table.",
    },

    {
        "analysis_name": "region_influence",
        "track": "EDA",
        "phase": "07b_region_robustness",
        "primary_question": "Do suspicious regions (e.g. 2655-2685 cm^-1) drive latent structure?",
        "artifact_examples": "region_influence_{range}.md, robustness_pca_{range}.csv",
        "scientific_value": "High — sensitivity analysis for the CH-stretch concern.",
        "overlap_with": "pca",
        "recommended_status": "keep_main",
        "manuscript_recommendation": "main",
        "notes": "Report explicitly, especially for CH-stretch range.",
    },
    {
        "analysis_name": "ml_feature_importance",
        "track": "ML",
        "phase": "archive",
        "primary_question": "Which engineered features drive supervised classification?",
        "artifact_examples": "feature_importance.csv, top_features_spectrum.png",
        "scientific_value": "High — connects latent structure to supervised performance.",
        "overlap_with": "",
        "recommended_status": "keep_main",
        "manuscript_recommendation": "main",
        "notes": "Interpret in terms of spectral regions / feature families.",
    },
    {
        "analysis_name": "feature_track_ablation",
        "track": "ML",
        "phase": "archive",
        "primary_question": "Which feature families are necessary for classification?",
        "artifact_examples": "feature_track_ablation.csv",
        "scientific_value": "High — mechanistic insight into feature-family contributions.",
        "overlap_with": "feature_importance",
        "recommended_status": "keep_main",
        "manuscript_recommendation": "supplement",
        "notes": "Use to argue which spectral structures matter.",
    },

]


def build_analysis_inventory(
    eda_base: Path | None = None,
    ml_base: Path | None = None,
) -> pd.DataFrame:
    """Return a DataFrame inventory of analyses and recommendations."""
    rows = [dict(item) for item in _INVENTORY_TEMPLATE]

    # Determine actual presence from filesystem when available
    for row in rows:
        present = False
        if row["track"] == "EDA" and eda_base is not None:
            phase_dir = eda_base / row["phase"]
            if phase_dir.exists() and any(phase_dir.iterdir()):
                present = True
        if row["track"] == "ML" and ml_base is not None:
            artifact = ml_base / row["artifact_examples"].split(",")[0].strip()
            if artifact.exists():
                present = True
        row["artifacts_present"] = present

    return pd.DataFrame(rows)


def write_manuscript_recommendations(
    inventory_df: pd.DataFrame,
    out_path: str | Path,
) -> None:
    """Write a markdown report organizing outputs by manuscript placement."""
    out_path = Path(out_path)

    main_df = inventory_df[inventory_df["manuscript_recommendation"] == "main"]
    supp_df = inventory_df[inventory_df["manuscript_recommendation"] == "supplement"]
    demote_df = inventory_df[inventory_df["manuscript_recommendation"] == "candidate_removal"]

    def _bullet_list(df: pd.DataFrame) -> list[str]:
        lines = []
        for _, row in df.iterrows():
            lines.append(
                f"- **{row['analysis_name']}** ({row['track']}, {row['phase']}): "
                f"{row['primary_question']} — *{row['recommended_status']}*"
            )
        return lines

    lines = [
        "# Manuscript figure and analysis recommendations",
        "",
        "## Main manuscript",
        "",
        *_bullet_list(main_df),
        "",
        "## Supplement",
        "",
        *_bullet_list(supp_df),
        "",
        "## Candidate for demotion/removal",
        "",
        *_bullet_list(demote_df),
        "",
        "## Full inventory",
        "",
        dataframe_to_markdown(inventory_df),
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def run_inventory(
    eda_base: Path | str | None = None,
    ml_base: Path | str | None = None,
    output_dir: Path | str | None = None,
) -> dict[str, Path]:
    """Generate inventory CSV and recommendation markdown.

    Returns
    -------
    dict
        Paths to the written ``analysis_inventory.csv`` and
        ``manuscript_figure_recommendations.md`` files.
    """
    eda_base = Path(eda_base) if eda_base is not None else None
    ml_base = Path(ml_base) if ml_base is not None else None
    output_dir = Path(output_dir) if output_dir is not None else (eda_base or Path("."))
    output_dir.mkdir(parents=True, exist_ok=True)

    inventory_df = build_analysis_inventory(eda_base=eda_base, ml_base=ml_base)
    inventory_path = output_dir / "analysis_inventory.csv"
    inventory_df.to_csv(inventory_path, index=False)

    recommendations_path = output_dir / "manuscript_figure_recommendations.md"
    write_manuscript_recommendations(inventory_df, recommendations_path)

    logger.info("Analysis inventory written to %s", inventory_path)
    logger.info("Manuscript recommendations written to %s", recommendations_path)
    return {
        "inventory_csv": inventory_path,
        "recommendations_md": recommendations_path,
    }
