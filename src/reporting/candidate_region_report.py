"""Phase 14: Publication-oriented candidate-region report generator.

Produces a markdown report and summary tables that separate findings into
robust, exploratory, unstable, and confounded categories. All language is
conservative and descriptive; no diagnostic, predictive, causal, or biomarker
claims are made.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import get_confirmatory_panel_dir, get_cross_region_phase_dir
from src import resolve_cfg as _cfg
from src.analysis._reporting import dataframe_to_markdown
from src.analysis.io_utils import load_csv_or_empty, write_csv_atomic, write_text_atomic

logger = logging.getLogger(__name__)

_CLASS_ORDER = ("robust", "exploratory", "unstable", "confounded")


def _report_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("analysis", {}).get("publication_report", {})


def _analysis_mode(cfg: dict[str, Any]) -> str:
    return str(cfg.get("analysis", {}).get("mode", "disease_vs_control"))


def _classify_candidate(row: pd.Series) -> str:
    evidence = str(row.get("evidence_status", "effect_only_fallback"))
    if evidence == "independently_validated":
        return "robust"
    if evidence == "cluster_supported_exploratory":
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
    frame = load_csv_or_empty(path)
    return None if frame.empty else frame


def _panel_flag_counts(ranking: pd.DataFrame) -> pd.DataFrame:
    """Count descriptive checks separately; they are not independent evidence tiers."""
    checks = ("effect_size_flag", "bootstrap_ci_excludes_zero", "permutation_flag")
    return pd.DataFrame(
        {
            "descriptive_check": checks,
            "n_features": [
                int(ranking.get(check, pd.Series(dtype=bool)).fillna(False).sum())
                for check in checks
            ],
        }
    )


def _top_feature_line(ranking: pd.DataFrame, feature_names: list[str]) -> str:
    if ranking.empty or "feature_name" not in ranking.columns:
        return "No ranked panel feature available yet."
    subset = ranking.loc[ranking["feature_name"].isin(feature_names)].copy()
    if subset.empty:
        return "No ranked panel feature available yet."
    row = subset.sort_values(["max_abs_cliffs_delta", "feature_name"], ascending=[False, True]).iloc[0]
    return (
        f"Largest observed available effect: `{row['feature_name']}` "
        f"(|Cliff's delta|={row['max_abs_cliffs_delta']:.3g}; exploratory descriptive check only)."
    )


def _build_panel_readme(ranking: pd.DataFrame, anchor: pd.DataFrame | None) -> str:
    if anchor is not None and not anchor.empty:
        row = anchor.iloc[0]
        anchor_line = (
            f"- **Primary anchor result**: `{row.get('feature_name', 'oxidative_index')}` vs "
            f"`{row.get('endpoint', 'delta_moca')}` (n={row.get('n', pd.NA)}, "
            f"rho={row.get('rho', pd.NA)}, permutation p={row.get('perm_p_value', pd.NA)}).\n"
        )
    else:
        anchor_line = "- **Primary anchor result**: not available for this run (see `phase_status.csv`).\n"

    n_ranked = 0 if ranking.empty else len(ranking)

    return (
        "# Fixed Post Hoc Panel Outputs\n\n"
        "Exploratory only — no diagnostic, biomarker, causal, or clinical-utility claim.\n\n"
        f"{anchor_line}"
        f"- **Ranked panel features**: {n_ranked}.\n\n"
        "## File index\n\n"
        "- `preregistration.md`: provenance-aware frozen panel snapshot.\n"
        "- `panel_provenance.csv`: registration/provenance fields.\n"
        "- `panel_feature_matrix.csv`: subject-level fixed-panel features.\n"
        "- `panel_evidence_flags.csv`: separate descriptive Phase L evidence flags (not a score).\n"
        "- `panel_report_section.md`: four-domain conservative synthesis.\n"
        "- `primary_anchor_result.csv`: prespecified primary hypothesis test.\n"
    )


def build_panel_report_section(
    *,
    ranking: pd.DataFrame,
    anchor: pd.DataFrame | None = None,
) -> str:
    """Build the Phase K four-domain panel narrative."""
    anchor_note = "Anchor result not available yet."
    if anchor is not None and not anchor.empty:
        row = anchor.iloc[0]
        anchor_note = (
            f"Anchor feature `{row.get('feature_name', 'oxidative_index')}` vs "
            f"`{row.get('endpoint', 'delta_moca')}`: rho={row.get('rho', pd.NA)}, "
            f"permutation p={row.get('perm_p_value', pd.NA)}."
        )

    flag_counts = _panel_flag_counts(ranking)
    lines = [
        "# Fixed Post Hoc Panel Report Section",
        "",
        "This section summarizes region-overlapping, tentatively assigned Raman panel features. "
        "It reports separate descriptive checks only; these dependent checks are not combined into an evidence score.",
        "",
        "## Descriptive Check Summary",
        "",
        dataframe_to_markdown(flag_counts),
        "",
        "## Membrane Remodeling",
        "",
        _top_feature_line(ranking, ["membrane_integrity_index", "lipid_carbonyl_1721", "ch_bulk_2885"]),
        "Interpretation is limited to regions overlapping lipid carbonyl and CH-stretch bands.",
        "",
        "## Protein Structural Heterogeneity",
        "",
        _top_feature_line(
            ranking,
            ["protein_remodeling_index", "tyrosine_environment_index", "protein_1555", "structural_protein_925", "structural_protein_980"],
        ),
        "Interpretation is limited to regions overlapping protein-associated bands; weak bands require cautious reading.",
        "",
        "## Tentatively Assigned Raman Ratios",
        "",
        _top_feature_line(ranking, ["oxidative_index", "carotenoid_1525", "carotenoid_cc_1156", "lipid_carbonyl_1721"]),
        anchor_note,
        "Carotenoid language is restricted to carotenoid-like / lipoprotein-associated signals.",
        "",
        "## System-Level EV Disruption",
        "",
        _top_feature_line(ranking, ranking["feature_name"].astype(str).tolist() if "feature_name" in ranking.columns else []),
        "This domain integrates variance heterogeneity, endpoint consistency, and feature coupling when those artifacts are present.",
        "",
    ]
    return "\n".join(lines)


def run_panel_report_section(
    *,
    evidence_flags_path: str | Path | None = None,
    anchor_result_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write the Phase K fixed-panel exploratory report section."""
    cfg = _cfg(cfg)
    phase_cfg = cfg.get("analysis", {}).get("panel_report", {})
    if not bool(phase_cfg.get("enabled", True)):
        return {}

    target_dir = Path(output_dir) if output_dir is not None else get_confirmatory_panel_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    ranking = _load_csv_if_exists(evidence_flags_path or target_dir / "panel_evidence_flags.csv")
    anchor = _load_csv_if_exists(anchor_result_path or target_dir / "primary_anchor_result.csv")
    text = build_panel_report_section(
        ranking=ranking if ranking is not None else pd.DataFrame(),
        anchor=anchor,
    )

    section_path = target_dir / "panel_report_section.md"
    readme_path = target_dir / "README.md"
    write_text_atomic(section_path, text)
    write_text_atomic(readme_path, _build_panel_readme(ranking if ranking is not None else pd.DataFrame(), anchor))
    return {"panel_report_section": section_path, "panel_report_readme": readme_path}


def _plot_overview(
    candidates: pd.DataFrame,
    figures_dir: Path,
) -> Path | None:
    if candidates.empty or "spectral_block" not in candidates.columns:
        return None

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

    display_columns = [
        "candidate_region_id",
        "spectral_block",
        "start_cm-1",
        "end_cm-1",
        "evidence_status",
        "selection_basis",
        "significance_supported",
        "n_overlapping_analyses",
        "covariate_persisted",
        "pls_vip_coverage",
        "splsda_selected",
        "biochemical_class",
        "interpretation_note",
    ]
    display = sub[[column for column in display_columns if column in sub.columns]].copy()
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
    panel_report_section_path: Path | None = None,
    minimal_panel_path: Path | None = None,
    minimal_panel_status: Mapping[str, object] | None = None,
) -> str:
    has_significant_regions = bool(candidates.get("significance_supported", pd.Series(dtype=bool)).fillna(False).any())
    if has_significant_regions:
        summary_note = (
            "Candidate regions include at least one cluster-permutation-supported interval. "
            "Channel-FDR-only and fallback regions remain descriptive context only."
        )
    else:
        summary_note = (
            "No spectral regions were supported by cluster-permutation correction. "
            "Any listed regions are fallback-selected largest-effect intervals and should be interpreted only as "
            "hypothesis-generating context."
        )
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
        summary_note,
        "",
        dataframe_to_markdown(summary),
        "",
        "### Classification definitions",
        "",
        "- **Robust**: independently validated in an external cohort; unavailable for this dataset.",
        "- **Exploratory**: supported by cluster-permutation inference in this cohort only.",
        "- **Unstable**: channel-FDR-only, fallback-selected, or unsupported by corrected inference.",
        "- **Confounded**: reserved for a prespecified model with measured confounders; not assigned from post-selection models.",
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
    if panel_report_section_path is not None:
        lines.extend(
            [
                "## Fixed Post Hoc Panel Section",
                "",
                f"Four-domain exploratory fixed-panel synthesis: `{panel_report_section_path}`",
                "",
            ]
        )

    for class_name in _CLASS_ORDER:
        lines.append(_section_for_class(class_name, candidates))

    minimal_panel_state = str((minimal_panel_status or {}).get("status", ""))
    minimal_panel = load_csv_or_empty(minimal_panel_path) if minimal_panel_path is not None else pd.DataFrame()
    if not minimal_panel.empty and minimal_panel_state in {"", "ok"}:
        minimal_panel_note = (
            "The candidate-derived minimal panel was selected by nested cross-validated ElasticNet / "
            "L1-penalized logistic regression with permutation testing and bootstrap stability summaries."
        )
    elif minimal_panel_state == "not_tested_gate":
        minimal_panel_note = (
            "The candidate-derived minimal panel was not tested because no cluster-supported candidate region passed "
            "the prespecified gate."
        )
    else:
        minimal_panel_note = "No candidate-derived minimal-panel result was available for this run."

    lines.extend([
        "",
        "## Methods note",
        "",
        "Candidate regions were detected at the subject level across clinical correlations, "
        "longitudinal ΔMoCA analyses, and disease-internal stratification tests, with cluster-mass permutation correction. "
        "These whole-spectrum screens and their cluster-permutation reference distributions are marginal and unadjusted for age and sex. "
        "When no corrected intervals were detected, fallback largest-effect intervals were retained only for transparent "
        "hypothesis generation and are not treated as statistical findings. "
        "Post-selection sensitivity models included age and sex only; SEC fraction, concentration, column/batch age, "
        "sample volume, acquisition batch, and other technical confounders were unavailable. These models do not validate selection. "
        "Results may therefore reflect residual confounding. ",
        minimal_panel_note,
        "",
    ])
    return "\n".join(lines)


def run_candidate_region_report(
    *,
    integrated_candidates_path: str | Path | None = None,
    minimal_panel_path: str | Path | None = None,
    minimal_panel_status: Mapping[str, object] | None = None,
    panel_report_section_path: str | Path | None = None,
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

    integrated = _load_csv_if_exists(integrated_candidates_path)
    if integrated is None or integrated.empty:
        integrated = pd.DataFrame(columns=[
            "source_region_id", "candidate_region_id", "spectral_block",
            "start_cm-1", "end_cm-1", "family_id", "inference_role",
            "evidence_status", "selection_scope", "raw_p", "within_scan_p",
            "family_adjusted_p", "cluster_permutation_significant",
            "selection_basis", "significance_supported", "covariate_persisted",
            "max_abs_effect", "integrated_abs_effect", "n_overlapping_analyses",
            "pls_vip_coverage", "splsda_selected",
            "biochemical_class", "interpretation_note",
        ])

    if "significance_supported" not in integrated.columns:
        integrated["significance_supported"] = (
            integrated.get("fdr_significant", pd.Series(False, index=integrated.index)).fillna(False).astype(bool)
            | integrated.get("cluster_permutation_significant", pd.Series(False, index=integrated.index))
            .fillna(False)
            .astype(bool)
        )
    if "selection_basis" not in integrated.columns:
        integrated["selection_basis"] = pd.NA

    integrated["report_class"] = integrated.apply(_classify_candidate, axis=1)

    summary = _class_summary(integrated)
    figures: dict[str, Path | None] = {
        "overview": _plot_overview(integrated, figures_dir),
    }

    report_text = _build_report(
        integrated,
        summary,
        figures,
        _analysis_mode(cfg),
        Path(panel_report_section_path) if panel_report_section_path is not None else None,
        Path(minimal_panel_path) if minimal_panel_path is not None else None,
        minimal_panel_status,
    )

    report_path = target_dir / "candidate_region_report.md"
    summary_path = target_dir / "candidate_region_summary.csv"
    write_text_atomic(report_path, report_text)
    write_csv_atomic(summary, summary_path, index=False)

    logger.info("Publication report written to %s", target_dir)
    return {
        "candidate_region_report": report_path,
        "candidate_region_summary": summary_path,
        "candidate_region_figures": figures_dir,
    }
