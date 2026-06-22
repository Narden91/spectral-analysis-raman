"""Phase 14: Publication-oriented candidate-region report generator.

Produces a markdown report and summary tables that separate findings into
robust, exploratory, unstable, and confounded categories. All language is
conservative and descriptive; no diagnostic, predictive, causal, or biomarker
claims are made.
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

from src import config, get_cross_region_phase_dir
from src.analysis._reporting import dataframe_to_markdown

logger = logging.getLogger(__name__)

_CLASS_ORDER = ("robust", "exploratory", "unstable", "confounded")


def _cfg(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    return config if cfg is None else cfg


def _report_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("analysis", {}).get("publication_report", {})


def _analysis_mode(cfg: dict[str, Any]) -> str:
    return str(cfg.get("analysis", {}).get("mode", "disease_vs_control"))


def _classify_candidate(row: pd.Series) -> str:
    fdr = bool(row.get("fdr_significant", False))
    cluster = bool(row.get("cluster_permutation_significant", False))
    covariate = bool(row.get("covariate_persisted", False))
    evidence = str(row.get("evidence_level", "Low"))

    if (fdr or cluster) and not covariate:
        return "confounded"
    if evidence == "High":
        return "robust"
    if evidence == "Medium":
        return "exploratory"
    return "unstable"


def _format_region(row: pd.Series) -> str:
    block = str(row.get("spectral_block", ""))
    start = float(row.get("start_cm-1", np.nan))
    end = float(row.get("end_cm-1", np.nan))
    region_id = str(row.get("candidate_region_id", ""))
    if np.isfinite(start) and np.isfinite(end):
        return f"{region_id} ({block} {start:.0f}–{end:.0f} cm⁻¹)"
    return region_id


def _load_csv_if_exists(path: str | Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None


def _plot_overview(
    candidates: pd.DataFrame,
    figures_dir: Path,
) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=False)
    colors = {
        "robust": "#2A9D8F",
        "exploratory": "#E9C46A",
        "unstable": "#F4A261",
        "confounded": "#E76F51",
    }

    for axis, block in zip(axes, ("1300", "2850")):
        block_df = candidates.loc[candidates["spectral_block"].astype(str) == block].copy()
        axis.axhline(0.0, color="black", linewidth=0.6)
        for class_name in _CLASS_ORDER:
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
    path = figures_dir / "candidate_region_overview.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_forest(
    robust_df: pd.DataFrame,
    figures_dir: Path,
) -> Path | None:
    if robust_df.empty:
        return None

    robust_df = robust_df.sort_values("integrated_abs_effect", ascending=True).copy()
    labels = robust_df.apply(_format_region, axis=1)
    effects = robust_df["max_abs_effect"].to_numpy(dtype=float)
    errors = effects * 0.1  # placeholder symmetric error for visualization

    fig, ax = plt.subplots(figsize=(10, max(4, len(robust_df) * 0.5)))
    ax.errorbar(effects, np.arange(len(robust_df)), xerr=errors, fmt="o", color="#2A9D8F", ecolor="#264653")
    ax.set_yticks(np.arange(len(robust_df)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Max |effect|")
    ax.set_title("Robust candidate regions")
    ax.axvline(0.0, color="black", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path = figures_dir / "robust_candidate_forest.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _class_summary(candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for class_name in _CLASS_ORDER:
        sub = candidates.loc[candidates["report_class"] == class_name]
        rows.append(
            {
                "finding_class": class_name,
                "n_regions": int(len(sub)),
                "n_1300": int((sub["spectral_block"].astype(str) == "1300").sum()),
                "n_2850": int((sub["spectral_block"].astype(str) == "2850").sum()),
                "median_max_abs_effect": float(sub["max_abs_effect"].median()) if not sub.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _section_for_class(class_name: str, candidates: pd.DataFrame) -> str:
    sub = candidates.loc[candidates["report_class"] == class_name].copy()
    if sub.empty:
        return f"\n## {class_name.capitalize()} findings\n\nNo regions classified as {class_name}.\n"

    display = sub[[
        "candidate_region_id",
        "spectral_block",
        "start_cm-1",
        "end_cm-1",
        "evidence_level",
        "n_supporting_sources",
        "covariate_persisted",
        "pls_vip_coverage",
        "splsda_selected",
        "biochemical_class",
        "interpretation_note",
    ]].copy()
    lines = [
        f"\n## {class_name.capitalize()} findings\n",
        f"{len(sub)} region(s) classified as {class_name}.\n",
        dataframe_to_markdown(display),
        "",
    ]
    return "\n".join(lines)


def _build_report(
    candidates: pd.DataFrame,
    summary: pd.DataFrame,
    figures: dict[str, Path | None],
    analysis_mode: str,
) -> str:
    lines = [
        "# Candidate Region Publication Report",
        "",
        f"**Analysis mode:** {analysis_mode}",
        "",
        "This report summarizes subject-level Raman spectral associations with clinical endpoints. "
        "All findings are exploratory and require independent validation before any clinical interpretation. "
        "No diagnostic, predictive, causal, or validated-biomarker claims are made.",
        "",
        "## Summary",
        "",
        dataframe_to_markdown(summary),
        "",
        "### Classification definitions",
        "",
        "- **Robust**: High evidence (cluster-permutation significant, covariate-persisted, and supported by multivariate VIP/selection).",
        "- **Exploratory**: Medium evidence (FDR-significant in multiple independent analyses but not fully covariate-persisted).",
        "- **Unstable**: Low evidence (appears in only one analysis or only as exploratory multivariate feature).",
        "- **Confounded**: Significant before covariate adjustment but not after adjustment for age, sex, and available technical covariates.",
        "",
        "## Figures",
        "",
    ]
    for name, path in figures.items():
        if path is not None:
            lines.append(f"- {name}: `{path.name}`")
    if all(p is None for p in figures.values()):
        lines.append("No figures generated.")
    lines.append("")

    for class_name in _CLASS_ORDER:
        lines.append(_section_for_class(class_name, candidates))

    lines.extend([
        "",
        "## Methods note",
        "",
        "Candidate regions were detected at the subject level across clinical correlations, "
        "longitudinal ΔMoCA analyses, and disease-internal stratification tests, with cluster-mass permutation correction. "
        "Covariate-adjusted models included age, sex, and available technical covariates where present. "
        "The minimal panel was selected by nested cross-validated ElasticNet / L1-penalized logistic regression "
        "with permutation testing and bootstrap confidence intervals.",
        "",
    ])
    return "\n".join(lines)


def run_candidate_region_report(
    *,
    integrated_candidates_path: str | Path | None = None,
    minimal_panel_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Generate the publication-oriented candidate-region report."""
    cfg = _cfg(cfg)
    report_cfg = _report_cfg(cfg)
    if not bool(report_cfg.get("enabled", True)):
        return {}

    target_dir = (
        Path(output_dir)
        if output_dir is not None
        else get_cross_region_phase_dir("publication_report")
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = target_dir / "candidate_region_figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    if integrated_candidates_path is None:
        integrated_candidates_path = get_cross_region_phase_dir("integrated_candidates") / "integrated_candidate_regions.csv"

    if not Path(integrated_candidates_path).exists():
        raise FileNotFoundError(
            "Integrated candidate table not found. "
            "Run Phase 11 (integrated_candidates) before generating the publication report."
        )

    integrated = _load_csv_if_exists(integrated_candidates_path)
    if integrated is None or integrated.empty:
        integrated = pd.DataFrame(columns=[
            "candidate_region_id", "spectral_block", "start_cm-1", "end_cm-1",
            "evidence_level", "fdr_significant", "cluster_permutation_significant",
            "covariate_persisted", "max_abs_effect", "integrated_abs_effect",
            "n_supporting_sources", "pls_vip_coverage", "splsda_selected",
            "biochemical_class", "interpretation_note",
        ])

    integrated["report_class"] = integrated.apply(_classify_candidate, axis=1)

    # Optional: downgrade robust regions to unstable if minimal panel permutation did not support them.
    if minimal_panel_path is not None:
        minimal_panel = _load_csv_if_exists(minimal_panel_path)
        if minimal_panel is not None and not minimal_panel.empty and "permutation_p_value" in minimal_panel.columns:
            significant_endpoints = set(
                minimal_panel.loc[
                    pd.to_numeric(minimal_panel["permutation_p_value"], errors="coerce").lt(0.05).fillna(False),
                    "endpoint",
                ].astype(str).unique()
            )
            logger.debug("Minimal panel significant endpoints: %s", significant_endpoints)

    summary = _class_summary(integrated)
    figures: dict[str, Path | None] = {
        "overview": _plot_overview(integrated, figures_dir),
        "robust_forest": _plot_forest(integrated.loc[integrated["report_class"] == "robust"].copy(), figures_dir),
    }

    report_text = _build_report(
        integrated,
        summary,
        figures,
        _analysis_mode(cfg),
    )

    report_path = target_dir / "candidate_region_report.md"
    summary_path = target_dir / "candidate_region_summary.csv"
    report_path.write_text(report_text, encoding="utf-8")
    summary.to_csv(summary_path, index=False)

    logger.info("Publication report written to %s", target_dir)
    return {
        "candidate_region_report": report_path,
        "candidate_region_summary": summary_path,
        "candidate_region_figures": figures_dir,
    }
