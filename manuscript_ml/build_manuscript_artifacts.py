"""Deterministic manuscript LaTeX table and diagnostic figure builder.

Reads stage outputs from ``manuscript_ml/`` and emits publication-ready LaTeX
table fragments (``tables/*.tex``) and Matplotlib figures (``figures/*.png``).

Strict safety contract: refuses to build cohort result tables or figures from
non-cohort runs (e.g., ``data_mode == "synthetic"`` or missing provenance)
unless ``--allow-synthetic-preview`` is explicitly passed, in which case every
preview table and figure is prominently watermarked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manuscript_ml import ensure_output_dir, print_stage_outputs, write_json_atomic  # noqa: E402
from src.analysis.raman_bands import RAMAN_BANDS  # noqa: E402

_PREVIEW_BANNER = "SYNTHETIC PREVIEW — NOT COHORT EVIDENCE"
_PREVIEW_BANNER_TEX = "SYNTHETIC PREVIEW --- NOT COHORT EVIDENCE"

_MODEL_LABELS: dict[str, str] = {
    "covariate_ridge": "Covariate-only ridge",
    "fixed_band_ridge": "Fixed 21-band + ridge",
    "pca_ridge": "Block-scaled PCA + ridge",
    "pls_regression": "PLS ($C$ + spectra)",
    "paired_linear_ridge": "Paired linear + ridge",
    "band_paired_ridge": "Band-paired ablation",
    "strongest_comparator": "Inner-selected established comparator",
}

_PALETTE: dict[str, str] = {
    "paired": "#0072B2",
    "fixed_band": "#E69F00",
    "pca": "#009E73",
    "pls": "#CC79A7",
    "strongest": "#D55E00",
    "neutral": "#4D4D4D",
}


def _tex_escape(text: Any) -> str:
    s = str(text)
    replacements = (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_\allowbreak{}"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    )
    for old, new in replacements:
        s = s.replace(old, new)
    return s


def _read_csv_if_nonempty(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _fmt_num(val: Any, decimals: int = 3) -> str:
    try:
        f = float(val)
    except (TypeError, ValueError):
        return "--"
    if not math.isfinite(f):
        return "--"
    return f"{f:.{decimals}f}"


def _fmt_ci(ci: Any, decimals: int = 3) -> str:
    if not isinstance(ci, (list, tuple)) or len(ci) != 2:
        return "--"
    lo, hi = _fmt_num(ci[0], decimals), _fmt_num(ci[1], decimals)
    if lo == "--" or hi == "--":
        return "--"
    return f"[{lo}, {hi}]"


def _write_text_atomic(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def _detect_cohort_data_mode(primary_dir: Path, sensitivity_dir: Path) -> str:
    modes: set[str] = set()
    manifest_path = primary_dir / "eligibility_and_split_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        dm = (manifest.get("source_metadata") or {}).get("data_mode")
        if dm is not None:
            modes.add(str(dm))
    for prov_file in (
        primary_dir / "run_provenance_primary_evaluation.json",
        sensitivity_dir / "run_provenance_sensitivity_and_secondary.json",
    ):
        if prov_file.is_file():
            prov = json.loads(prov_file.read_text(encoding="utf-8"))
            dm = (prov.get("data") or {}).get("data_mode")
            if dm is not None:
                modes.add(str(dm))
    if not modes:
        return "unknown"
    if len(modes) > 1:
        raise ValueError(f"Mixed data_mode values detected across input artifacts: {sorted(modes)}")
    return next(iter(modes))


def _preview_header_lines(n_cols: int, is_preview: bool) -> list[str]:
    if not is_preview:
        return []
    return [
        f"% {_PREVIEW_BANNER_TEX}",
        rf"\multicolumn{{{n_cols}}}{{c}}{{\textbf{{{_PREVIEW_BANNER_TEX}}}}} \\",
        r"\midrule",
    ]


def _build_synthetic_scenarios_table(df: pd.DataFrame) -> str:
    lines = [
        "% Auto-generated from simulation_scenarios_summary.csv",
        r"\noindent{\footnotesize\setlength{\tabcolsep}{3pt}%",
        r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}X r >{\raggedright\arraybackslash}p{0.18\textwidth} r r r r@{}}",
        r"\toprule",
        r"Scenario & $N$ & Preset & $\widehat{\Delta}_{\text{pair}}$ & $\widehat{\Delta}_{\text{comp}}$ & ICC & $R^2$ \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        scen = _tex_escape(row.get("scenario", ""))
        n_val = int(row["n_participants"]) if pd.notna(row.get("n_participants")) else "--"
        preset = _tex_escape(row.get("covariate_preset", ""))
        dp = _fmt_num(row.get("mean_delta_mse_paired"))
        dc = _fmt_num(row.get("mean_delta_vs_strongest_comparator"))
        icc = _fmt_num(row.get("mean_spectral_icc"))
        r2 = _fmt_num(row.get("mean_recovery_r2_ldl_factor"))
        lines.append(f"{scen} & {n_val} & {preset} & {dp} & {dc} & {icc} & {r2} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabularx}}", ""])
    return "\n".join(lines)


def _build_simulation_calibration_table(df: pd.DataFrame) -> str:
    lines = [
        "% Auto-generated from simulation_calibration_report.csv",
        r"\noindent{\footnotesize\setlength{\tabcolsep}{2.5pt}%",
        r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}X >{\raggedright\arraybackslash}p{0.16\textwidth} r r r r r r r@{}}",
        r"\toprule",
        r"Scenario & Method & $R$ & P.\ Cov & W.\ Lo & P.\ W & C.\ Cov & C.\ W & Fail \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        scen = _tex_escape(row.get("scenario", "single_setting"))
        method = _tex_escape(row.get("interval_method", "studentized_t"))
        reps = int(row["n_repetitions"]) if pd.notna(row.get("n_repetitions")) else "--"
        pcov = _fmt_num(row.get("unconditional_coverage"))
        pwlo = _fmt_num(row.get("wilson_95_lower"))
        pw = _fmt_num(row.get("mean_interval_width"))
        ccov = _fmt_num(row.get("unconditional_coverage_vs_comparator"))
        cw = _fmt_num(row.get("mean_interval_width_vs_comparator"))
        fail = _fmt_num(row.get("dataset_failure_rate"))
        lines.append(f"{scen} & {method} & {reps} & {pcov} & {pwlo} & {pw} & {ccov} & {cw} & {fail} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabularx}}", ""])
    return "\n".join(lines)


def _build_mde_curve_table(df: pd.DataFrame) -> str:
    lines = [
        "% Auto-generated from simulation_mde_curve.csv",
        r"\noindent{\footnotesize\setlength{\tabcolsep}{3.5pt}%",
        r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}X r r r r r r@{}}",
        r"\toprule",
        r"Signal & Target $\Delta_{\text{pair}}$ & Mean $\widehat{\Delta}_{\text{pair}}$ & Det.\ rate & Prim.\ Cov & Comp.\ Tgt & Comp.\ Det \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        sig = _fmt_num(row.get("signal_strength"), 2)
        t_p = _fmt_num(row.get("target_delta_mse"))
        m_p = _fmt_num(row.get("mean_delta_mse"))
        det_p = _fmt_num(row.get("detection_rate_delta_mse"))
        cov_p = _fmt_num(row.get("unconditional_coverage"))
        t_c = _fmt_num(row.get("target_delta_vs_comparator"))
        det_c = _fmt_num(row.get("detection_rate_vs_comparator"))
        lines.append(f"{sig} & {t_p} & {m_p} & {det_p} & {cov_p} & {t_c} & {det_c} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabularx}}", ""])
    return "\n".join(lines)


def _build_primary_results_table(
    comp_df: pd.DataFrame,
    manifest: dict[str, Any],
    *,
    is_preview: bool,
) -> str:
    by_comp = {str(r["comparison"]): r for _, r in comp_df.iterrows()}
    ordered_models = [
        "covariate_ridge",
        "fixed_band_ridge",
        "pca_ridge",
        "pls_regression",
        "paired_linear_ridge",
        "band_paired_ridge",
        "strongest_comparator",
    ]
    lines = [
        "% Auto-generated from comparator_errors_and_differences.csv and eligibility_and_split_manifest.json",
        r"\noindent{\footnotesize\setlength{\tabcolsep}{3.5pt}%",
        r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}X r r r r r r@{}}",
        r"\toprule",
        *_preview_header_lines(7, is_preview),
        r"Procedure & MSE & Rel.\ red. & RMSE & MAE & Cal.\ int. & Cal.\ slope \\",
        r"\midrule",
    ]
    for m_name in ordered_models:
        row = by_comp.get(f"{m_name}_vs_covariate_ridge")
        if row is None:
            continue
        label = _MODEL_LABELS.get(m_name, _tex_escape(m_name))
        lines.append(
            f"{label} & {_fmt_num(row.get('mse_model'))} & {_fmt_num(row.get('relative_mse_reduction'))} & "
            f"{_fmt_num(row.get('rmse_model'))} & {_fmt_num(row.get('mae_model'))} & "
            f"{_fmt_num(row.get('calibration_intercept'))} & {_fmt_num(row.get('calibration_slope'))} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}}", "", r"\medskip"])

    boot = manifest.get("multiplicity_bootstrap") or {}
    im_label = _tex_escape(boot.get("interval_method", "not_run"))
    lines.extend(
        [
            r"\noindent{\footnotesize\setlength{\tabcolsep}{3.5pt}%",
            r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}X r c c@{}}",
            r"\toprule",
            *_preview_header_lines(4, is_preview),
            rf"Contrast (method: \texttt{{{im_label}}}) & Estimate & 95\% interval & Pct.\ 95\% \\",
            r"\midrule",
        ]
    )
    row_pair = by_comp.get("paired_linear_ridge_vs_covariate_ridge", {})
    row_comp = by_comp.get("paired_linear_ridge_vs_strongest_comparator", {})
    row_strong = by_comp.get("strongest_comparator_vs_covariate_ridge", {})
    contrasts = [
        (
            "Covariate-only minus paired linear",
            row_pair.get("delta_mse"),
            boot.get("ci_95_delta_mse"),
            boot.get("ci_95_percentile_delta_mse"),
        ),
        (
            "Inner-selected comparator minus paired linear",
            row_comp.get("delta_mse"),
            boot.get("ci_95_vs_comparator"),
            boot.get("ci_95_percentile_vs_comparator"),
        ),
        (
            "Covariate-only minus inner-selected comparator",
            row_strong.get("delta_mse"),
            boot.get("ci_95_strongest_vs_covariate"),
            boot.get("ci_95_percentile_strongest_vs_covariate"),
        ),
    ]
    for label, est, ci_main, ci_pct in contrasts:
        lines.append(f"{label} & {_fmt_num(est)} & {_fmt_ci(ci_main)} & {_fmt_ci(ci_pct)} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabularx}}", ""])
    return "\n".join(lines)


def _build_repeatability_and_angles_table(
    rep_df: pd.DataFrame,
    angles_df: pd.DataFrame,
    *,
    is_preview: bool,
) -> str:
    lines = [
        "% Auto-generated from baseline_repeatability_report.csv and subspace_principal_angles.csv",
        r"\noindent{\footnotesize\setlength{\tabcolsep}{3pt}%",
        r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}X r r r r r r r@{}}",
        r"\toprule",
        *_preview_header_lines(8, is_preview),
        r"Fold & $n_{\text{pair}}$ & Full ICC & Spec.\ ICC & SD($s$) & Strong.\ ICC & Discrim. & Top-1 \\",
        r"\midrule",
    ]
    for _, row in rep_df.iterrows():
        fold = int(row["outer_fold"])
        n_p = int(row["n_paired_test"]) if pd.notna(row.get("n_paired_test")) else 0
        lines.append(
            f"Fold {fold} & {n_p} & "
            f"{_fmt_num(row.get('paired_full_icc_estimate'))} & "
            f"{_fmt_num(row.get('paired_spectral_icc_estimate'))} & "
            f"{_fmt_num(row.get('paired_sd_spectral_contrib'))} & "
            f"{_fmt_num(row.get('strongest_comparator_spectral_icc_estimate'))} & "
            f"{_fmt_num(row.get('paired_discriminability_score'))} & "
            f"{_fmt_num(row.get('paired_discriminability_retrieval_top1'))} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}}", ""])

    if not angles_df.empty:
        lines.extend(
            [
                r"\medskip",
                r"\noindent{\footnotesize\setlength{\tabcolsep}{3.5pt}%",
                r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}X l r r r r@{}}",
                r"\toprule",
                *_preview_header_lines(6, is_preview),
                r"Model & Fold pair & $r_a / r_b$ & $\theta_1$ ($^\circ$) & Mean ($^\circ$) & Max ($^\circ$) \\",
                r"\midrule",
            ]
        )
        for _, row in angles_df.iterrows():
            mdl = _tex_escape(row.get("model", "paired_linear_ridge"))
            pair_lbl = f"{int(row['fold_a'])}--{int(row['fold_b'])}"
            dims = f"{int(row['dim_a'])}/{int(row['dim_b'])}"
            lines.append(
                f"{mdl} & {pair_lbl} & {dims} & "
                f"{_fmt_num(row.get('angle_1_deg'), 1)} & "
                f"{_fmt_num(row.get('mean_principal_angle_deg'), 1)} & "
                f"{_fmt_num(row.get('max_principal_angle_deg'), 1)} \\\\"
            )
        lines.extend([r"\bottomrule", r"\end{tabularx}}", ""])
    return "\n".join(lines)


def _build_sensitivity_and_secondary_table(
    sens_df: pd.DataFrame | None,
    seed_df: pd.DataFrame | None,
    sec_df: pd.DataFrame | None,
    anchor_df: pd.DataFrame | None,
    *,
    is_preview: bool,
) -> str:
    lines = ["% Auto-generated from sensitivity, secondary endpoint, and anchor CSV outputs"]
    if sens_df is not None and not sens_df.empty:
        lines.extend(
            [
                r"\noindent{\footnotesize\setlength{\tabcolsep}{3pt}%",
                r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{0.18\textwidth} >{\raggedright\arraybackslash}X r r r r r@{}}",
                r"\toprule",
                *_preview_header_lines(7, is_preview),
                r"Axis & Setting & $N$ & $\mathrm{MSE}_C$ & $\widehat{\Delta}_{\text{pair}}$ & $\widehat{\Delta}_{\text{comp}}$ & Atten. \\",
                r"\midrule",
            ]
        )
        for _, row in sens_df.iterrows():
            lines.append(
                f"{_tex_escape(row.get('sensitivity_axis', ''))} & "
                f"{_tex_escape(row.get('setting', ''))} & "
                f"{int(row['n_participants'])} & "
                f"{_fmt_num(row.get('mse_covariate'))} & "
                f"{_fmt_num(row.get('delta_mse_paired'))} & "
                f"{_fmt_num(row.get('delta_vs_strongest_comparator'))} & "
                f"{_fmt_num(row.get('attenuation_vs_primary_delta_mse'))} \\\\"
            )
        lines.extend([r"\bottomrule", r"\end{tabularx}}", "", r"\medskip"])

    if seed_df is not None and not seed_df.empty:
        lines.extend(
            [
                r"\noindent{\footnotesize\setlength{\tabcolsep}{3.5pt}%",
                r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}r >{\raggedright\arraybackslash}X r r r r@{}}",
                r"\toprule",
                *_preview_header_lines(6, is_preview),
                r"Seed & Role & $\mathrm{MSE}_C$ & $\widehat{\Delta}_{\text{pair}}$ & $\widehat{\Delta}_{\text{comp}}$ & Atten. \\",
                r"\midrule",
            ]
        )
        for _, row in seed_df.iterrows():
            lines.append(
                f"{int(row['outer_seed'])} & {_tex_escape(row.get('role', ''))} & "
                f"{_fmt_num(row.get('mse_covariate'))} & "
                f"{_fmt_num(row.get('delta_mse_paired'))} & "
                f"{_fmt_num(row.get('delta_vs_strongest_comparator'))} & "
                f"{_fmt_num(row.get('attenuation_vs_primary_delta_mse'))} \\\\"
            )
        lines.extend([r"\bottomrule", r"\end{tabularx}}", "", r"\medskip"])

    if sec_df is not None and not sec_df.empty:
        lines.extend(
            [
                r"\noindent{\footnotesize\setlength{\tabcolsep}{3.5pt}%",
                r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}X r r r r r@{}}",
                r"\toprule",
                *_preview_header_lines(6, is_preview),
                r"Secondary endpoint & $N$ & $\mathrm{MSE}_C$ & $\widehat{\Delta}_{\text{pair}}$ & $\widehat{\Delta}_{\text{comp}}$ & RMSE \\",
                r"\midrule",
            ]
        )
        for _, row in sec_df.iterrows():
            lines.append(
                f"{_tex_escape(row.get('endpoint', ''))} & "
                f"{int(row['n_participants'])} & "
                f"{_fmt_num(row.get('mse_covariate'))} & "
                f"{_fmt_num(row.get('delta_mse_paired'))} & "
                f"{_fmt_num(row.get('delta_vs_strongest_comparator'))} & "
                f"{_fmt_num(row.get('rmse_paired'))} \\\\"
            )
        lines.extend([r"\bottomrule", r"\end{tabularx}}", "", r"\medskip"])

    if anchor_df is not None and not anchor_df.empty:
        lines.extend(
            [
                r"\noindent{\footnotesize\setlength{\tabcolsep}{3pt}%",
                r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{0.20\textwidth} >{\raggedright\arraybackslash}X r r c r@{}}",
                r"\toprule",
                *_preview_header_lines(6, is_preview),
                r"Anchor & Role & $N$ & Partial $r$ & 95\% range & Fold med.\ $r$ \\",
                r"\midrule",
            ]
        )
        for _, row in anchor_df.iterrows():
            ci_str = _fmt_ci([row.get("resample_ci_low"), row.get("resample_ci_high")])
            lines.append(
                f"{_tex_escape(row.get('anchor', ''))} & "
                f"{_tex_escape(row.get('target_role', ''))} & "
                f"{int(row['n_participants'])} & "
                f"{_fmt_num(row.get('partial_pearson_r'))} & "
                f"{ci_str} & "
                f"{_fmt_num(row.get('fold_median_partial_r'))} \\\\"
            )
        lines.extend([r"\bottomrule", r"\end{tabularx}}", ""])
    return "\n".join(lines)


def _apply_preview_watermark(fig: plt.Figure, is_preview: bool) -> None:
    if not is_preview:
        return
    fig.text(
        0.5,
        0.99,
        _PREVIEW_BANNER,
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
        color="#B00020",
        bbox={"facecolor": "#FFF0F0", "edgecolor": "#B00020", "pad": 3.0},
    )
    fig.text(
        0.5,
        0.5,
        _PREVIEW_BANNER,
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
        color="#B00020",
        alpha=0.14,
        rotation=20,
    )


def _plot_repeatability_figure(rep_df: pd.DataFrame, out_path: Path, *, is_preview: bool) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), dpi=300)
    folds = rep_df["outer_fold"].to_numpy(dtype=int)
    models = [
        ("paired", "Paired linear", _PALETTE["paired"]),
        ("fixed_band", "Fixed 21-band", _PALETTE["fixed_band"]),
        ("pca", "Block PCA", _PALETTE["pca"]),
        ("pls", "PLS", _PALETTE["pls"]),
    ]

    ax0, ax1 = axes
    for offset, (prefix, label, color) in enumerate(models):
        full_col = f"{prefix}_full_icc_estimate"
        spec_col = f"{prefix}_spectral_icc_estimate"
        if full_col not in rep_df.columns or spec_col not in rep_df.columns:
            continue
        x = folds + (offset - 1.5) * 0.16
        ax0.scatter(
            x - 0.03,
            rep_df[full_col],
            marker="o",
            s=48,
            facecolors="none",
            edgecolors=color,
            linewidth=1.4,
            label=f"{label} (full)",
        )
        ax0.scatter(
            x + 0.03,
            rep_df[spec_col],
            marker="D",
            s=36,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            label=f"{label} (spectral)",
        )
    ax0.axhline(0.0, color="#666666", linestyle="--", linewidth=0.8)
    ax0.set_xticks(folds)
    ax0.set_xlabel("Outer Fold")
    ax0.set_ylabel("Single-Measure ICC(A,1)")
    ax0.set_title("Full Prediction vs Spectral-Contribution Repeatability")
    if ax0.get_legend_handles_labels()[0]:
        ax0.legend(fontsize=7.2, ncol=2, loc="lower left", frameon=True)
    ax0.grid(axis="y", alpha=0.25)

    for offset, (prefix, label, color) in enumerate(models):
        sd_col = f"{prefix}_sd_spectral_contrib"
        rmsd_col = f"{prefix}_spectral_icc_rmsd"
        if sd_col not in rep_df.columns or rmsd_col not in rep_df.columns:
            continue
        ax1.scatter(
            rep_df[sd_col],
            rep_df[rmsd_col],
            s=48,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            label=label,
        )
    ax1.set_xlabel("Between-Participant Spectral Contribution SD")
    ax1.set_ylabel("Paired A/B Spectral Contribution RMSD")
    ax1.set_title("Spectral Signal Scale vs Technical Disagreement")
    if ax1.get_legend_handles_labels()[0]:
        ax1.legend(fontsize=8.0, frameon=True)
    ax1.grid(alpha=0.25)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93 if is_preview else 1.0))
    _apply_preview_watermark(fig, is_preview)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def _plot_spectral_weights_figure(weights_df: pd.DataFrame, out_path: Path, *, is_preview: bool) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2), dpi=300, sharey=True)
    fold_cols = sorted(
        [c for c in weights_df.columns if c.startswith("paired_weight_fold_")],
        key=lambda c: int(c.rsplit("_", 1)[1]),
    )
    block_specs = [
        ("1300", "Fingerprint Block (734–1788 cm⁻¹)", axes[0]),
        ("2850", "C–H Stretching Block (2451–3197 cm⁻¹)", axes[1]),
    ]
    for blk_name, title, ax in block_specs:
        sub = weights_df[weights_df["block"].astype(str) == blk_name].sort_values("wavenumber_cm1")
        if sub.empty:
            ax.set_title(f"{title} (no channels)")
            continue
        wn = sub["wavenumber_cm1"].to_numpy(dtype=float)
        wn_min, wn_max = float(wn[0]), float(wn[-1])
        for band in RAMAN_BANDS.values():
            lo, hi = float(band.bounds[0]), float(band.bounds[1])
            if hi >= wn_min and lo <= wn_max:
                ax.axvspan(lo, hi, color="#E69F00", alpha=0.14, linewidth=0.0)
        for idx_f, col in enumerate(fold_cols):
            ax.plot(
                wn,
                sub[col].to_numpy(dtype=float),
                color=_PALETTE["paired"],
                alpha=0.35,
                linewidth=0.9,
                label="Outer folds" if idx_f == 0 else None,
            )
        mean_col = "paired_weight_mean" if "paired_weight_mean" in sub.columns else None
        mean_vals = (
            sub[mean_col].to_numpy(dtype=float)
            if mean_col is not None
            else np.mean(sub[fold_cols].to_numpy(dtype=float), axis=1)
        )
        ax.plot(
            wn,
            mean_vals,
            color="#003E6B",
            linewidth=1.8,
            label="Mean paired weight",
        )
        ax.axhline(0.0, color="#666666", linestyle="--", linewidth=0.7)
        ax.set_xlabel("Raman Shift (cm⁻¹)")
        ax.set_title(title)
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("Effective Spectral Weight (LDL-C units / norm. intensity)")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Patch(facecolor="#E69F00", alpha=0.22, edgecolor="none", label="Curated Raman bands (21)"))
    labels.append("Curated Raman bands (21)")
    fig.legend(
        handles=handles,
        labels=labels,
        fontsize=8.2,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        frameon=True,
    )

    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.93 if is_preview else 1.0))
    _apply_preview_watermark(fig, is_preview)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def _plot_sensitivity_forest_figure(
    sens_df: pd.DataFrame,
    seed_df: pd.DataFrame | None,
    out_path: Path,
    *,
    is_preview: bool,
) -> Path:
    records: list[dict[str, Any]] = []
    for _, row in sens_df.iterrows():
        records.append(
            {
                "label": f"{row['sensitivity_axis']}: {row['setting']}",
                "delta": float(row["delta_mse_paired"]),
                "vs_comp": float(row["delta_vs_strongest_comparator"]),
                "lo": float(row["ci_95_delta_mse_low"]) if "ci_95_delta_mse_low" in sens_df.columns else float("nan"),
                "hi": float(row["ci_95_delta_mse_high"]) if "ci_95_delta_mse_high" in sens_df.columns else float("nan"),
                "comp_lo": (
                    float(row["ci_95_vs_comparator_low"])
                    if "ci_95_vs_comparator_low" in sens_df.columns
                    else float("nan")
                ),
                "comp_hi": (
                    float(row["ci_95_vs_comparator_high"])
                    if "ci_95_vs_comparator_high" in sens_df.columns
                    else float("nan")
                ),
            }
        )
    if seed_df is not None and not seed_df.empty:
        for _, row in seed_df.iterrows():
            records.append(
                {
                    "label": f"outer_seed: {int(row['outer_seed'])} ({row['role']})",
                    "delta": float(row["delta_mse_paired"]),
                    "vs_comp": float(row["delta_vs_strongest_comparator"]),
                    "lo": float("nan"),
                    "hi": float("nan"),
                    "comp_lo": float("nan"),
                    "comp_hi": float("nan"),
                }
            )

    n_rows = max(len(records), 1)
    fig, ax = plt.subplots(figsize=(9.5, max(4.0, 0.42 * n_rows + 1.4)), dpi=300)
    y_pos = np.arange(len(records))
    deltas = np.array([r["delta"] for r in records], dtype=float)
    vs_comps = np.array([r["vs_comp"] for r in records], dtype=float)
    los = np.array([r["lo"] for r in records], dtype=float)
    his = np.array([r["hi"] for r in records], dtype=float)
    comp_los = np.array([r["comp_lo"] for r in records], dtype=float)
    comp_his = np.array([r["comp_hi"] for r in records], dtype=float)

    ax.axvline(0.0, color="#444444", linestyle="--", linewidth=1.0)
    if len(records) > 0 and math.isfinite(deltas[0]):
        ax.axvline(deltas[0], color=_PALETTE["paired"], linestyle=":", linewidth=0.9, alpha=0.6)

    has_ci = np.isfinite(los) & np.isfinite(his)
    if np.any(has_ci):
        idx_ci = np.flatnonzero(has_ci)
        ax.errorbar(
            deltas[idx_ci],
            y_pos[idx_ci] - 0.14,
            xerr=np.vstack([np.maximum(deltas[idx_ci] - los[idx_ci], 0.0), np.maximum(his[idx_ci] - deltas[idx_ci], 0.0)]),
            fmt="o",
            color=_PALETTE["paired"],
            capsize=3,
            label="Delta MSE (paired vs covariate) with 95% CI",
        )
    idx_no = np.flatnonzero(~has_ci)
    if idx_no.size > 0:
        ax.scatter(
            deltas[idx_no],
            y_pos[idx_no] - 0.14,
            color=_PALETTE["paired"],
            s=42,
            label="Delta MSE (paired vs covariate)" if not np.any(has_ci) else None,
        )
    has_comp_ci = np.isfinite(comp_los) & np.isfinite(comp_his)
    if np.any(has_comp_ci):
        idx_cc = np.flatnonzero(has_comp_ci)
        ax.errorbar(
            vs_comps[idx_cc],
            y_pos[idx_cc] + 0.16,
            xerr=np.vstack(
                [
                    np.maximum(vs_comps[idx_cc] - comp_los[idx_cc], 0.0),
                    np.maximum(comp_his[idx_cc] - vs_comps[idx_cc], 0.0),
                ]
            ),
            fmt="s",
            color=_PALETTE["strongest"],
            markersize=5.5,
            capsize=2.5,
            label="Delta MSE (paired vs strongest comparator) with 95% CI",
        )
    idx_cc_no = np.flatnonzero(~has_comp_ci)
    if idx_cc_no.size > 0:
        ax.scatter(
            vs_comps[idx_cc_no],
            y_pos[idx_cc_no] + 0.16,
            color=_PALETTE["strongest"],
            marker="s",
            s=36,
            label="Delta MSE (paired vs strongest comparator)" if not np.any(has_comp_ci) else None,
        )
    ax.set_yticks(y_pos)
    ax.set_yticklabels([r["label"] for r in records], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Held-Out Squared Error Reduction (positive favors paired)")
    ax.set_title("Covariate Preset, Subset Mode, and Outer-Partition Sensitivity", pad=26)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(
            fontsize=8.2,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=2,
            frameon=True,
        )
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93 if is_preview else 1.0))
    _apply_preview_watermark(fig, is_preview)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def build_manuscript_artifacts(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    calibration_dir: str | Path | None = None,
    primary_dir: str | Path | None = None,
    sensitivity_dir: str | Path | None = None,
    allow_synthetic_preview: bool = False,
) -> dict[str, Path]:
    """Build deterministic LaTeX table fragments and diagnostic figures from stage outputs."""
    base_in = Path(input_dir)
    cal_in = Path(calibration_dir) if calibration_dir is not None else base_in
    prim_in = Path(primary_dir) if primary_dir is not None else base_in
    sens_in = Path(sensitivity_dir) if sensitivity_dir is not None else base_in

    target_dir = ensure_output_dir(output_dir)
    tables_dir = target_dir / "tables"
    figs_dir = target_dir / "figures"

    cohort_files = [
        prim_in / "comparator_errors_and_differences.csv",
        prim_in / "baseline_repeatability_report.csv",
        prim_in / "subspace_principal_angles.csv",
        prim_in / "fold_spectral_weights.csv",
        prim_in / "fold_selected_settings.csv",
        prim_in / "nested_held_out_predictions.csv",
        sens_in / "covariate_and_subset_sensitivity.csv",
        sens_in / "outer_partition_seed_sensitivity.csv",
        sens_in / "secondary_endpoints_summary.csv",
        sens_in / "anchor_associations_continuous.csv",
    ]
    has_cohort_artifacts = any(p.is_file() for p in cohort_files)
    data_mode = _detect_cohort_data_mode(prim_in, sens_in) if has_cohort_artifacts else "simulation_only"
    is_preview = False
    if has_cohort_artifacts and data_mode != "cohort":
        if not allow_synthetic_preview:
            raise PermissionError(
                f"Refusing to populate manuscript cohort tables/figures from data_mode={data_mode!r}. "
                "Authorize a real cohort run or pass --allow-synthetic-preview for clearly watermarked previews."
            )
        is_preview = True

    outputs: dict[str, Path] = {}
    input_hashes: dict[str, str] = {}

    def _record_input(p: Path) -> None:
        if p.is_file():
            input_hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()

    # Simulation & calibration tables
    sim_scen_path = cal_in / "simulation_scenarios_summary.csv"
    sim_scen_df = _read_csv_if_nonempty(sim_scen_path)
    if sim_scen_df is not None and not sim_scen_df.empty:
        _record_input(sim_scen_path)
        outputs["table_synthetic_scenarios"] = _write_text_atomic(
            tables_dir / "table_synthetic_scenarios.tex",
            _build_synthetic_scenarios_table(sim_scen_df),
        )

    sim_cal_path = cal_in / "simulation_calibration_report.csv"
    sim_cal_df = _read_csv_if_nonempty(sim_cal_path)
    if sim_cal_df is not None and not sim_cal_df.empty:
        _record_input(sim_cal_path)
        outputs["table_simulation_calibration"] = _write_text_atomic(
            tables_dir / "table_simulation_calibration.tex",
            _build_simulation_calibration_table(sim_cal_df),
        )

    sim_mde_path = cal_in / "simulation_mde_curve.csv"
    sim_mde_df = _read_csv_if_nonempty(sim_mde_path)
    if sim_mde_df is not None and not sim_mde_df.empty:
        _record_input(sim_mde_path)
        outputs["table_simulation_mde_curve"] = _write_text_atomic(
            tables_dir / "table_simulation_mde_curve.tex",
            _build_mde_curve_table(sim_mde_df),
        )

    # Primary results table
    comp_path = prim_in / "comparator_errors_and_differences.csv"
    manifest_path = prim_in / "eligibility_and_split_manifest.json"
    comp_df = _read_csv_if_nonempty(comp_path)
    if comp_df is not None and not comp_df.empty:
        _record_input(comp_path)
        manifest_data: dict[str, Any] = {}
        if manifest_path.is_file():
            _record_input(manifest_path)
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        outputs["table_primary_results"] = _write_text_atomic(
            tables_dir / "table_primary_results.tex",
            _build_primary_results_table(comp_df, manifest_data, is_preview=is_preview),
        )

    # Repeatability & principal angles table + figure
    rep_path = prim_in / "baseline_repeatability_report.csv"
    angles_path = prim_in / "subspace_principal_angles.csv"
    rep_df = _read_csv_if_nonempty(rep_path)
    if rep_df is not None and not rep_df.empty:
        _record_input(rep_path)
        angles_df = _read_csv_if_nonempty(angles_path)
        if angles_df is not None:
            _record_input(angles_path)
        else:
            angles_df = pd.DataFrame()
        outputs["table_repeatability_and_angles"] = _write_text_atomic(
            tables_dir / "table_repeatability_and_angles.tex",
            _build_repeatability_and_angles_table(rep_df, angles_df, is_preview=is_preview),
        )
        outputs["fig_repeatability_icc_and_scale"] = _plot_repeatability_figure(
            rep_df, figs_dir / "fig_repeatability_icc_and_scale.png", is_preview=is_preview
        )

    # Canonical spectral weights figure
    weights_path = prim_in / "fold_spectral_weights.csv"
    weights_df = _read_csv_if_nonempty(weights_path)
    if weights_df is not None and not weights_df.empty:
        _record_input(weights_path)
        outputs["fig_spectral_weights_profile"] = _plot_spectral_weights_figure(
            weights_df, figs_dir / "fig_spectral_weights_profile.png", is_preview=is_preview
        )

    # Sensitivity, secondary endpoints, and anchors table + forest plot
    sens_path = sens_in / "covariate_and_subset_sensitivity.csv"
    seed_path = sens_in / "outer_partition_seed_sensitivity.csv"
    sec_path = sens_in / "secondary_endpoints_summary.csv"
    anchor_path = sens_in / "anchor_associations_continuous.csv"
    sens_df = _read_csv_if_nonempty(sens_path)
    seed_df = _read_csv_if_nonempty(seed_path)
    sec_df = _read_csv_if_nonempty(sec_path)
    anchor_df = _read_csv_if_nonempty(anchor_path)
    for p in (sens_path, seed_path, sec_path, anchor_path):
        _record_input(p)

    if any(df is not None and not df.empty for df in (sens_df, seed_df, sec_df, anchor_df)):
        outputs["table_sensitivity_and_secondary"] = _write_text_atomic(
            tables_dir / "table_sensitivity_and_secondary.tex",
            _build_sensitivity_and_secondary_table(
                sens_df, seed_df, sec_df, anchor_df, is_preview=is_preview
            ),
        )
    if sens_df is not None and not sens_df.empty:
        outputs["fig_sensitivity_forest"] = _plot_sensitivity_forest_figure(
            sens_df, seed_df, figs_dir / "fig_sensitivity_forest.png", is_preview=is_preview
        )

    build_manifest = write_json_atomic(
        target_dir / "artifact_build_manifest.json",
        {
            "data_mode": data_mode,
            "synthetic_preview": is_preview,
            "input_dirs": {
                "calibration_dir": str(cal_in),
                "primary_dir": str(prim_in),
                "sensitivity_dir": str(sens_in),
            },
            "input_sha256": input_hashes,
            "generated_artifacts": {k: str(v) for k, v in outputs.items()},
        },
    )
    outputs["artifact_build_manifest"] = build_manifest
    return outputs


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic LaTeX tables and diagnostic figures from manuscript_ml stage outputs."
    )
    parser.add_argument("--input-dir", type=str, required=True, help="Default directory holding stage outputs.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for generated tables/ and figures/.")
    parser.add_argument("--calibration-dir", type=str, default=None, help="Optional override for calibration outputs.")
    parser.add_argument("--primary-dir", type=str, default=None, help="Optional override for primary outputs.")
    parser.add_argument("--sensitivity-dir", type=str, default=None, help="Optional override for sensitivity outputs.")
    parser.add_argument(
        "--allow-synthetic-preview",
        action="store_true",
        help="Permit clearly watermarked preview tables/figures from non-cohort runs.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    outputs = build_manuscript_artifacts(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        calibration_dir=args.calibration_dir,
        primary_dir=args.primary_dir,
        sensitivity_dir=args.sensitivity_dir,
        allow_synthetic_preview=args.allow_synthetic_preview,
    )
    print_stage_outputs("build_manuscript_artifacts", outputs)


if __name__ == "__main__":
    main()
