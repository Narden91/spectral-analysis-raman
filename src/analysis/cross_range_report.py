"""Cross-range narrative report for Raman EDA (Phase E).

Collects summary artefacts from Phases B–D (per range) plus the `both`
whole-spectra analyses and renders a markdown (+ optional HTML) report
with embedded caveats.

The report does NOT re-run any analysis — it reads previously saved CSVs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

COHORT_GROUP_COUNTS_TEMPLATE = "cohort_{range}_group_counts.csv"
COHORT_CLINICAL_MISSINGNESS_TEMPLATE = "cohort_{range}_clinical_missingness.csv"
PCA_TOP_WAVENUMBERS_TEMPLATE = "pca_top_wavenumbers_per_pc_{range}.csv"
PCA_VARIANCE_TEMPLATE = "pca_variance_{range}.csv"
SUBJECT_UMAP_EMBEDDING_TEMPLATE = "subject_umap_embedding_{range}.csv"
SUBJECT_UMAP_PURITY_TEMPLATE = "subject_umap_purity_{range}.csv"
CLUSTER_PROFILES_TEMPLATE = "cluster_profiles_{range}.csv"
CLUSTER_DRIVERS_TEMPLATE = "cluster_driver_wavenumbers_{range}.csv"
MFA_BLOCK_CONTRIBUTIONS = "mfa_block_contributions.csv"

# ---------------------------------------------------------------------------
# Caveats emitted in every report
# ---------------------------------------------------------------------------

_CAVEATS = """\
> **Scientific caveats (all analyses)**
>
> * Inter-subject variance conflates biological variability with technical
>   confounders: SEC fraction composition, concentration (5×), column batch /
>   age, lipoprotein contamination, and sample volume.
> * Sex imbalance may produce Simpson's paradox — pooled summaries should
>   always be accompanied by sex-stratified comparisons.
> * QC flags indicate measurement reliability concerns; flagged areas are
>   retained unless otherwise noted but should be interpreted cautiously.
> * UMAP axes are nonlinear — Spearman correlations with UMAP coordinates
>   are exploratory visualisation aids, not confirmatory tests.
> * All p-values / FDR q-values are nominal; replication in independent
>   cohorts is required before any clinical interpretation.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        logger.debug("Cross-range report: artefact not found — %s", path)
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return None


def _require(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required cross-range report artifact missing: {path}")
    df = _read_csv(path)
    if df is None:
        raise RuntimeError(f"Required cross-range report artifact could not be read: {path}")
    return df


def _optional(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        logger.warning("Skipping optional cross-range report section; artifact not found: %s", path)
        return None
    return _read_csv(path)


def _overview_section(ranges: list[str], output_base: Path) -> str:
    """Build cohort overview table with disease/control counts, gender breakdown,
    and paired-MoCA availability per range."""
    lines = ["## Cohort Overview\n"]

    for r in ranges:
        descriptive_dir = output_base / _range_label(r) / "01_descriptive_statistics"
        group_counts = _require(descriptive_dir / COHORT_GROUP_COUNTS_TEMPLATE.format(range=r))
        if group_counts is None or group_counts.empty:
            lines.append(f"### Range {r} cm⁻¹\n\n*(cohort summary not found)*\n")
            continue

        # Disease / control totals
        disease = group_counts[group_counts["condition"].str.lower() == "disease"]["n_subjects"].sum()
        control = group_counts[group_counts["condition"].str.lower() == "control"]["n_subjects"].sum()

        # Gender breakdown
        gender_lines = []
        for _, row in group_counts.iterrows():
            gender_lines.append(f"{row['gender']} {row['condition']}: {int(row['n_subjects'])}")

        lines.append(f"### Range {r} cm⁻¹\n")
        lines.append(f"- **Disease**: {int(disease)} | **Control**: {int(control)}")
        lines.append("- **Gender breakdown**: " + "; ".join(gender_lines))

        # Paired MoCA from subject UMAP embedding (most reliable if available)
        embedding_path = (
            output_base
            / _range_label(r)
            / "06_manifold_subject"
            / SUBJECT_UMAP_EMBEDDING_TEMPLATE.format(range=r)
        )
        emb_df = _optional(embedding_path)
        if emb_df is not None:
            required = ["moca_baseline_total", "moca_3_months_total"]
            if all(c in emb_df.columns for c in required):
                paired = emb_df.dropna(subset=required)
                n_paired = len(paired)
                lines.append(f"- **Paired MoCA** (baseline + 3-month): {n_paired} subjects")
            else:
                lines.append("- **Paired MoCA**: columns missing in embedding CSV")
        else:
            # Fallback to clinical missingness summary
            missingness = _optional(descriptive_dir / COHORT_CLINICAL_MISSINGNESS_TEMPLATE.format(range=r))
            if missingness is not None:
                moca_baseline = missingness[missingness["column"] == "moca_baseline_total"]
                moca_3m = missingness[missingness["column"] == "moca_3_months_total"]
                if not moca_baseline.empty and not moca_3m.empty:
                    avail_base = (1.0 - float(moca_baseline.iloc[0]["missing_rate"])) * 100
                    avail_3m = (1.0 - float(moca_3m.iloc[0]["missing_rate"])) * 100
                    lines.append(
                        f"- **MoCA availability**: baseline {avail_base:.0f}%, 3-month {avail_3m:.0f}%"
                    )
                else:
                    lines.append("- **MoCA availability**: not reported")
            else:
                lines.append("- **MoCA availability**: not reported")

    return "\n".join(lines) + "\n"


def _pca_section(phase_dir_fn, spectral_range: str) -> str:
    pca_dir = phase_dir_fn("pca")
    top_df = _require(pca_dir / PCA_TOP_WAVENUMBERS_TEMPLATE.format(range=spectral_range))
    variance_df = _optional(pca_dir / PCA_VARIANCE_TEMPLATE.format(range=spectral_range))

    if top_df is None and variance_df is None:
        return "*(PCA artefacts not found)*\n"

    lines = [f"### PCA — {spectral_range} cm⁻¹\n"]

    if variance_df is not None and not variance_df.empty:
        pc1_var = float(variance_df.iloc[0]["explained_variance_ratio"]) * 100
        pc2_var = float(variance_df.iloc[1]["explained_variance_ratio"]) * 100 if len(variance_df) > 1 else 0.0
        lines.append(f"- **PC1** = {pc1_var:.1f}%, **PC2** = {pc2_var:.1f}%")

    if top_df is not None:
        for pc, grp in top_df.groupby("pc"):
            top_rows = grp.sort_values("rank").head(5)
            waves = ", ".join(f"{w:.1f}" for w in top_rows["wavenumber"])
            lines.append(f"- **{pc}**: top 5 wavenumbers = {waves}")

    return "\n".join(lines) + "\n"


def _subject_umap_section(phase_dir_fn, spectral_range: str) -> str:
    subj_dir = phase_dir_fn("manifold_subject")
    purity_df = _optional(subj_dir / SUBJECT_UMAP_PURITY_TEMPLATE.format(range=spectral_range))
    if purity_df is None:
        return ""

    lines = [f"### Subject UMAP — {spectral_range} cm⁻¹\n"]
    for _, row in purity_df.iterrows():
        target = row["target"]
        ami = float(row["ami"]) if pd.notna(row["ami"]) else np.nan
        pval = float(row["p_value"]) if pd.notna(row["p_value"]) else np.nan
        lines.append(
            f"- **{target}**: AMI = {ami:.3f}, p = {pval:.3f}"
        )
    return "\n".join(lines) + "\n"


def _cluster_section(phase_dir_fn, spectral_range: str) -> str:
    clust_dir = phase_dir_fn("cluster_interpretation")
    profile_df = _optional(clust_dir / CLUSTER_PROFILES_TEMPLATE.format(range=spectral_range))
    driver_df = _optional(clust_dir / CLUSTER_DRIVERS_TEMPLATE.format(range=spectral_range))
    if profile_df is None:
        return ""

    lines = [f"### Cluster interpretation — {spectral_range} cm⁻¹\n"]
    for _, row in profile_df.iterrows():
        lines.append(
            f"- **Cluster {int(row['cluster'])}**: "
            f"n={int(row['n_subjects'])}, "
            f"{int(row['n_disease'])} disease / {int(row['n_control'])} control, "
            f"{row['pct_female']:.0f}% female."
        )

    if driver_df is not None:
        sig_drivers = driver_df[driver_df["significant"] == True]  # noqa: E712
        for cid, grp in sig_drivers.groupby("cluster"):
            top_waves = grp.nsmallest(5, "fdr_q")["wavenumber"]
            waves_str = ", ".join(f"{w:.1f}" for w in top_waves)
            lines.append(
                f"  → Cluster {int(cid)} top significant drivers: {waves_str} cm⁻¹"
            )

    return "\n".join(lines) + "\n"


def _mfa_section(both_dir: Path) -> str:
    mfa_df = _optional(both_dir / "01_global_structure" / MFA_BLOCK_CONTRIBUTIONS)
    if mfa_df is None:
        return ""

    lines = ["### MFA block contributions (whole spectra)\n"]
    if "component" in mfa_df.columns:
        for _, row in mfa_df.head(3).iterrows():
            comp = int(row["component"])
            fp = float(row.get("fingerprint_contribution", np.nan))
            ch = float(row.get("ch_stretch_contribution", np.nan))
            lines.append(
                f"- **PC{comp}**: fingerprint contribution = {fp:.3f}, CH-stretch contribution = {ch:.3f}"
            )
    return "\n".join(lines) + "\n"


def _range_label(spectral_range: str) -> str:
    """Local mirror of ``src._range_label`` to avoid importing ``src.__init__``.

    The canonical helper lives in ``src.__init__`` and reads
    ``config["output"]["range_labels"]`` from the merged config at
    import time. Importing that module here would drag in the YAML
    config loader for callers of the cross-range report, which is a much
    larger side effect than the report itself needs.

    The values below MUST stay in sync with the
    ``output.range_labels`` mapping in the merged config.
    """
    return {"1300": "fingerprint_1300", "2850": "ch_stretch_2850", "both": "whole_spectra"}.get(
        spectral_range, f"range_{spectral_range}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def collect_range_summaries(
    ranges: list[str],
    phase_dir_fn,
    output_base: Path,
) -> dict[str, dict]:
    """Read per-range Phase B/C/D CSV artefacts and return a nested dict.

    Parameters
    ----------
    ranges : list[str]
        e.g. ``['1300', '2850']``
    phase_dir_fn : callable
        Callable(phase_key) → Path, scoped to the current spectral range.
        Should be called as ``phase_dir_fn(range, phase_key)``.
    output_base : Path
        Root EDA output directory (contains ``ch_stretch_2850/`` etc.).
    """
    summary: dict[str, dict] = {}
    for r in ranges:
        summary[r] = {"range": r}
    return summary


def render_narrative_md(
    ranges: list[str],
    phase_dir_fns: dict[str, callable],
    output_base: Path,
) -> str:
    """Build cross-range markdown narrative.

    Parameters
    ----------
    ranges : list[str]
        Spectral ranges analysed, e.g. ``['1300', '2850']``.
    phase_dir_fns : dict[str, callable]
        ``{range_str: phase_dir_fn}`` where each callable maps a phase key
        to its output ``Path``.
    output_base : Path
        Root EDA output dir; ``both/`` sub-dir is checked for MFA outputs.
    """
    sections = [
        "# Cross-Range EDA Narrative Report\n",
        _CAVEATS,
        "---\n",
        _overview_section(ranges, output_base),
    ]

    for r in ranges:
        phase_dir_fn = phase_dir_fns[r]
        sections.append(f"## Range {r} cm⁻¹\n")
        sections.append(_pca_section(phase_dir_fn, r))
        sections.append(_subject_umap_section(phase_dir_fn, r))
        sections.append(_cluster_section(phase_dir_fn, r))

    both_dir = output_base / "whole_spectra"
    if both_dir.exists():
        sections.append("## Whole-spectra (`both`)\n")
        sections.append(_mfa_section(both_dir))

    sections.append("\n---\n")
    sections.append(
        "*Report generated automatically by `src/analysis/cross_range_report.py`.*\n"
    )
    return "\n".join(sections)


def render_narrative_html(md_text: str) -> str:
    """Convert markdown narrative to HTML (requires ``markdown`` package).

    Falls back to wrapping in a ``<pre>`` tag if ``markdown`` is unavailable.
    """
    try:
        import markdown  # type: ignore[import-untyped]

        return markdown.markdown(
            md_text,
            extensions=["tables", "fenced_code"],
        )
    except ImportError:
        logger.warning(
            "Package 'markdown' not installed — emitting plain-text HTML fallback. "
            "Install with: uv add markdown"
        )
        escaped = md_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return f"<html><body><pre>{escaped}</pre></body></html>"


def write_cross_range_report(
    ranges: list[str],
    phase_dir_fns: dict[str, callable],
    output_base: Path,
    out_dir: Path,
) -> None:
    """Generate and save ``cross_range_report.md`` (+ ``.html``).

    Parameters
    ----------
    ranges : list[str]
    phase_dir_fns : dict[str, callable]
    output_base : Path
        Root EDA output directory.
    out_dir : Path
        Directory where the report files are written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    md_text = render_narrative_md(ranges, phase_dir_fns, output_base)
    md_path = out_dir / "cross_range_report.md"
    md_path.write_text(md_text, encoding="utf-8")
    logger.info("Cross-range report written: %s", md_path)

    html_text = render_narrative_html(md_text)
    html_path = out_dir / "cross_range_report.html"
    html_path.write_text(html_text, encoding="utf-8")
    logger.info("Cross-range HTML report written: %s", html_path)
