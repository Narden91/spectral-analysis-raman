"""Full Exploratory Data Analysis pipeline for Raman spectroscopy data.

Supports two public workflows:
1. Single-range EDA for `1300` or `2850`, which keeps the broad descriptive,
   PCA, univariate, band-analysis, and variance-decomposition phases.
2. Whole-spectrum EDA for `both`, which fuses paired 1300/2850 areas and emits
   the reduced four-phase global structure, drivers, interaction, and variance
   context workflow.

Note: Inter-subject variance can include both biological variability and
unmeasured technical variation such as SEC fraction composition, concentration
state, column batch effects, lipoprotein contamination, and sample volume.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.figure

matplotlib.use("Agg")  # non-interactive backend for batch processing
import matplotlib.pyplot as plt

from src import (
    config,
    get_eda_phase_dir,
    _output_dir,
    _range_label,
)
from src.analysis.spectral_loader import (
    build_group_matrix,
    build_whole_spectra_group_matrices,
    load_subject_spectra,
    wave_axes_compatible,
)
from src.analysis.cohort import build_cohort_summary, save_cohort_summary
from src.io.file_crawler import build_file_index, disambiguate_subject_ids
from src.metadata.clinical import load_clinical_data, merge_index_with_clinical
from src.analysis.statistics import (
    run_pca,
    rank_wavenumbers_per_pc,
    build_pca_loading_interpretation,
    extract_band_intensities,
    compare_band_intensities_between_groups,
    test_normality,
)
from src.analysis.robustness import run_region_robustness, write_region_influence_report
from src.analysis.intra_disease import (
    build_disease_subject_mean_matrix,
    compute_intra_disease_correlations,
    summarize_significant_targets,
)
from src.preprocessing.pipeline import preprocess_spectra, apply_region_weights
from src.visualization.eda_plots import (
    GROUP_COLORS,
    plot_pca_scores,
    plot_pca_loadings,
    plot_scree,
    plot_band_boxplots,
    plot_group_mean_spectra,
    plot_difference_spectra,
    plot_variance_decomposition,
    plot_correlation_spectrum,
)
from src.visualization.advanced_plots import plot_all_bands_raincloud

logger = logging.getLogger(__name__)


_VALID_RANGES = {"1300", "2850", "both"}


def _range_title(spectral_range: str) -> str:
    return "whole spectra" if spectral_range == "both" else f"{spectral_range} cm⁻¹"


def _save_fig(fig: matplotlib.figure.Figure, name: str, out_dir: Path):

    path = out_dir / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.debug("Saved: %s", path)


def _log_phase(title: str) -> None:
    logger.info("=" * 60)
    logger.info("%s", title)
    logger.info("=" * 60)


def _figure_flag(key: str, default: bool = True) -> bool:
    return bool(config.get("analysis", {}).get("figures", {}).get(key, default))





def _estimate_variance_decomposition(
    df_index: pd.DataFrame,
    spectral_range: str,
    groups: dict[str, np.ndarray],
) -> tuple[float, float, float]:
    """
    ANOVA-style variance decomposition of *preprocessed* subject-level means.

    The `groups` dict contains preprocessed (baseline-corrected, smoothed,
    normalized) subject-mean spectra.  Mixing in raw pixel variance would
    compare different scales, so we decompose only at the subject-mean level:

        Total = Between-group (SSB/N) + Within-group / Between-subject (SSW/N)

    We still report a third "measurement-noise" component — the fraction of
    within-group variance that can be attributed to finite spatial averaging
    (Var(pixel) / n_pixels).  This is estimated by preprocessing pixel
    spectra from a subsample of subjects.

    **CRITICAL CAVEAT**: Inter-subject variance conflates biological and unmeasured
    technical variance (SEC fraction composition, 5x concentration state, column
    batch/age, lipoprotein contamination, sample volume). Do not interpret this
    as purely biological variability.

    Returns (measurement_noise_pct, inter_subject_pct, inter_group_pct).
    """
    all_mat = np.vstack(list(groups.values()))  # (N, W)
    N = all_mat.shape[0]

    # Total variance (mean over wavenumbers of the per-wavenumber variance)
    total_var = np.nanmean(np.nanvar(all_mat, axis=0, ddof=0))

    if total_var == 0:
        return (np.nan, 0.0, 0.0)

    # Between-group variance  (weighted by group size)
    grand_mean = np.nanmean(all_mat, axis=0)
    inter_group_var = sum(
        v.shape[0] / N * np.nanmean((np.nanmean(v, axis=0) - grand_mean) ** 2)
        for v in groups.values()
    )

    # Within-group / between-subject variance
    inter_subject_var = total_var - inter_group_var

    if spectral_range == "both":
        logger.info(
            "Measurement noise not estimated for whole-spectra: area-level SNV already applied"
        )
        return (
            np.nan,
            float(max(inter_subject_var, 0.0) / total_var * 100),
            float(inter_group_var / total_var * 100),
        )

    # --- Measurement noise estimate (optional third component) ---
    # Preprocess a subsample of subjects' pixel spectra to estimate how
    # much of the inter-subject variance is due to finite spatial sampling.
    noise_var = 0.0
    try:
        rng = np.random.default_rng(42)
        sample_sids = rng.choice(
            df_index["subject_id"].unique(),
            size=min(
                int(config.get("analysis", {}).get("noise_sample_subjects", 8)),
                df_index["subject_id"].nunique(),
            ),
            replace=False,
        )
        pix_vars, pix_counts = [], []
        for sid in sample_sids:
            wave, raw_pix = load_subject_spectra(df_index, sid, spectral_range)
            if raw_pix.size == 0 or raw_pix.shape[0] < 2:
                continue
            proc_pix, _ = preprocess_spectra(raw_pix, wave, region=spectral_range)  # same pipeline as groups
            pix_vars.append(np.nanmean(np.nanvar(proc_pix, axis=0)))
            pix_counts.append(proc_pix.shape[0])
        if pix_vars:
            raw_noise_var = np.mean(pix_vars) / np.mean(pix_counts)
            if raw_noise_var > inter_subject_var:
                logger.warning(
                    "Estimated measurement noise variance (%.2e) exceeds inter-subject variance (%.2e). "
                    "Capping noise to inter-subject variance.",
                    raw_noise_var, inter_subject_var
                )
            noise_var = min(raw_noise_var, inter_subject_var)
    except Exception as exc:
        logger.debug("Noise estimation skipped: %s", exc)

    # Partition inter-subject into "biological" + "noise"
    bio_subject_var = max(inter_subject_var - noise_var, 0.0)

    return (
        float(noise_var / total_var * 100),
        float(bio_subject_var / total_var * 100),
        float(inter_group_var / total_var * 100),
    )


_PHASE_DIRS = {
    "descriptive": "01_descriptive_statistics",
    "group_comp": "02_group_comparisons",
    "pca": "03_pca",
    "band": "04_band_analysis",
    "variance": "05_variance_decomposition",
    "intra_disease": "06_intra_disease",
}

def _run_single_range_eda(
    spectral_range: str = "1300",
    skip_preprocess: bool = False,
    max_subjects: int | None = None,
):
    """Run the full EDA pipeline."""

    def phase_dir(key: str) -> Path:
        return get_eda_phase_dir(spectral_range, _PHASE_DIRS[key])

    df_index = disambiguate_subject_ids(build_file_index(measurement="all"))
    if df_index.empty:
        raise RuntimeError(
            "No measurement files found. Is the data directory correct?"
        )

    logger.info(
        "Loaded dynamic dataset index: %d files, %d subjects",
        len(df_index),
        df_index["subject_id"].nunique(),
    )

    subject_group = config.get("data", {}).get("subject_group")
    if subject_group == "disease_only":
        n_before = df_index["subject_id"].nunique()
        df_index = df_index[~df_index["is_control"]].copy()
        n_after = df_index["subject_id"].nunique()
        logger.info(
            "Disease-only mode: retained %d disease subjects, excluded %d controls",
            n_after,
            n_before - n_after,
        )

    clinical_df = load_clinical_data(data_index=df_index)
    df_merged = merge_index_with_clinical(df_index, clinical_df)
    cohort_summary = build_cohort_summary(
        df_merged,
        expected_files_per_subject=config.get("analysis", {}).get(
            "expected_files_per_subject", 12
        ),
    )
    cohort_dir = phase_dir("descriptive")
    cohort_paths = save_cohort_summary(
        cohort_summary,
        cohort_dir,
        prefix=f"cohort_{spectral_range}",
    )
    logger.info("Saved cohort summaries to %s", cohort_dir)
    if not cohort_summary["group_counts"].empty:
        logger.info(
            "Cohort group counts:\n%s",
            cohort_summary["group_counts"].to_string(index=False),
        )
    if not cohort_summary["incomplete_subjects"].empty:
        logger.warning(
            "Detected %d incomplete subjects (< expected files). See %s",
            len(cohort_summary["incomplete_subjects"]),
            cohort_paths["incomplete_subjects"],
        )

    _log_phase(f"PHASE 0: Loading and aggregating spectra (range={spectral_range} cm-1)")

    if subject_group == "disease_only":
        group_configs = {
            "female_disease": {"gender": "female", "is_control": False},
            "male_disease": {"gender": "male", "is_control": False},
        }
    else:
        group_configs = {
            "female_disease": {"gender": "female", "is_control": False},
            "female_control": {"gender": "female", "is_control": True},
            "male_disease": {"gender": "male", "is_control": False},
            "male_control": {"gender": "male", "is_control": True},
        }

    raw_groups: dict[str, np.ndarray] = {}
    group_ids: dict[str, list[str]] = {}
    wavenumbers = None

    if spectral_range == "both":
        wavenumbers, raw_groups, group_ids = build_whole_spectra_group_matrices(
            df_index,
            group_configs,
            max_subjects=max_subjects,
        )
        if wavenumbers.size == 0:
            wavenumbers = None
        for key in group_configs:
            spectra = raw_groups.get(key)
            if spectra is None or spectra.size == 0:
                logger.warning("No data for group %s, skipping", key)
                continue
            logger.debug(
                "  %s: %d subjects × %d wavenumbers",
                key,
                spectra.shape[0],
                spectra.shape[1],
            )
    else:
        for key, filt in group_configs.items():
            wave, spectra, sids = build_group_matrix(
                df_index,
                spectral_range=spectral_range,
                max_subjects=max_subjects,
                **filt,
            )
            if spectra.size == 0:
                logger.warning("No data for group %s, skipping", key)
                continue
            raw_groups[key] = spectra
            group_ids[key] = sids
            if wavenumbers is None:
                wavenumbers = wave
            elif not wave_axes_compatible(wavenumbers, wave):
                raise ValueError(
                    f"Wavenumber axis mismatch for group '{key}': "
                    f"expected {len(wavenumbers)} wavenumbers, got {len(wave)}."
                )
            logger.debug(
                "  %s: %d subjects × %d wavenumbers",
                key,
                spectra.shape[0],
                spectra.shape[1],
            )

    if not raw_groups:
        raise RuntimeError(
            "No valid groups loaded — check that spectra files are present and "
            "preprocessing did not discard all data."
        )
    assert wavenumbers is not None, "wavenumbers must be set when at least one group is loaded"
    range_title = _range_title(spectral_range)

    _log_phase("PHASE 0: Preprocessing")

    if spectral_range == "both":
        logger.info(
            "Whole-spectra groups are already preprocessed in the shared area pipeline; "
            "skipping preprocess_spectra."
        )
        if skip_preprocess:
            logger.info("--skip-preprocess has no additional effect for range=both")
        groups = raw_groups
    else:
        logger.info(
            "Single-range subject means are now built from pixel-level preprocessing before aggregation; "
            "skipping the legacy subject-mean preprocessing pass."
        )
        if skip_preprocess:
            logger.info("--skip-preprocess has no additional effect for single-range EDA")
        groups = raw_groups

        _log_phase("PHASE 0.5: Preprocessing Consistency Checks (PCS)")
        try:
            from src.analysis.pcs import run_pcs_validation
            from src.analysis.spectral_loader import load_subject_spectra
            from src.preprocessing.pipeline import preprocess_spectra
            
            sample_sid = list(group_ids.values())[0][0] if group_ids and list(group_ids.values())[0] else None
            if sample_sid:
                w_raw, r_pix = load_subject_spectra(df_index, sample_sid, spectral_range)
                if r_pix.size > 0:
                    p_pix, w_proc = preprocess_spectra(r_pix, w_raw, region=spectral_range)
                    pcs_metrics = run_pcs_validation(r_pix, p_pix, w_raw, w_proc)
                    if not pcs_metrics.get("passed", False):
                        logger.error("PCS validation FAILED! Preprocessing distorts chemical signal. Metrics: %s", pcs_metrics)
                    else:
                        logger.info("PCS validation PASSED successfully.")
        except Exception as exc:
            logger.warning("PCS validation could not be completed: %s", exc)

    result_both = None
    if spectral_range == "both":
        from src.analysis._whole_spectra_runner import _run_whole_spectra_eda

        result_both = _run_whole_spectra_eda(
            df_index,
            groups,
            wavenumbers,
            group_ids,
            range_title,
        )

    rw_cfg = config.get("analysis", {}).get("region_weighting", {})
    apply_to = set(rw_cfg.get("apply_to", []) if rw_cfg.get("enabled", False) else [])

    if spectral_range != "both":
        _log_phase("PHASE 1: Descriptive statistics")

        desc_rows = []
        for key, spectra in groups.items():
            desc_rows.append(
                {
                    "group": key,
                    "n_subjects": spectra.shape[0],
                    "mean_intensity": np.nanmean(spectra),
                    "std_intensity": np.nanstd(spectra),
                    "min_intensity": np.nanmin(spectra),
                    "max_intensity": np.nanmax(spectra),
                    "median_intensity": np.nanmedian(spectra),
                }
            )
        desc_df = pd.DataFrame(desc_rows)
        desc_out = phase_dir("descriptive")
        desc_csv = desc_out / f"descriptive_stats_{spectral_range}.csv"
        desc_df.to_csv(desc_csv, index=False)
        logger.debug("Descriptive stats:\n%s", desc_df.to_string(index=False))

        _log_phase("PHASE 2: Visualization — Group comparisons")

        gc_dir = phase_dir("group_comp")

        if _figure_flag("emit_group_mean_spectra", default=True):
            fig, _ = plot_group_mean_spectra(
                wavenumbers, groups, title=f"Group Mean Spectra ({range_title})"
            )
            _save_fig(fig, "group_mean_spectra.png", gc_dir)

        # Difference spectra (Disease - control) per gender
        if _figure_flag("emit_difference_spectra", default=False):
            for gender in ["female", "male"]:
                g_key = f"{gender}_disease"
                c_key = f"{gender}_control"
                if g_key in groups and c_key in groups:
                    fig, _ = plot_difference_spectra(
                        wavenumbers,
                        groups[g_key],
                        groups[c_key],
                        label=f"{gender.capitalize()} (Disease − Control)",
                        title=f"Difference Spectrum — {gender.capitalize()} ({range_title})",
                        color=GROUP_COLORS.get(g_key, "#CC3311"),
                    )
                    _save_fig(fig, f"difference_spectrum_{gender}.png", gc_dir)

        _log_phase("PHASE 3: PCA")

        all_spectra = np.vstack(list(groups.values()))
        all_labels = np.concatenate([np.full(v.shape[0], k) for k, v in groups.items()])

        pca_dir = phase_dir("pca")

        pca_input = (
            apply_region_weights(all_spectra, wavenumbers, rw_cfg)
            if "pca" in apply_to
            else all_spectra
        )

        n_comps = min(
            int(config.get("analysis", {}).get("pca_n_components", 10)),
            pca_input.shape[0] - 1,
        )
        
        if subject_group == "disease_only":
            from src.analysis.intra_disease import run_intra_disease_pca
            pca_result = run_intra_disease_pca(
                pca_input,
                n_components=n_comps,
                random_state=42,
            )
        else:
            pca_result = run_pca(
                pca_input,
                n_components=n_comps,
                scale=False,
            )

        fig, _ = plot_pca_scores(
            pca_result["scores"],
            all_labels,
            pca_result["explained_variance_ratio"],
            title=f"PCA Score Plot ({range_title})",
            loadings=pca_result["loadings"],
            wavenumbers=wavenumbers,
        )
        _save_fig(fig, "pca_scores_pc1_pc2.png", pca_dir)

        fig, _ = plot_pca_scores(
            pca_result["scores"],
            all_labels,
            pca_result["explained_variance_ratio"],
            pc_x=0,
            pc_y=2,
            title=f"PCA Score Plot PC1 vs PC3 ({range_title})",
            loadings=pca_result["loadings"],
            wavenumbers=wavenumbers,
        )
        _save_fig(fig, "pca_scores_pc1_pc3.png", pca_dir)

        fig, _ = plot_scree(
            pca_result["explained_variance_ratio"],
            title=f"Scree Plot ({range_title})",
        )
        _save_fig(fig, "scree_plot.png", pca_dir)

        fig, _ = plot_pca_loadings(
            wavenumbers,
            pca_result["loadings"],
            pca_result["explained_variance_ratio"],
            n_components=3,
            title=f"PCA Loadings ({range_title})",
        )
        _save_fig(fig, "pca_loadings.png", pca_dir)

        # Save PCA scores table
        pca_df = pd.DataFrame(
            pca_result["scores"],
            columns=[f"PC{i+1}" for i in range(pca_result["scores"].shape[1])],
        )
        pca_df["group"] = all_labels
        # Add subject IDs
        all_sids = []
        for k in groups:
            all_sids.extend(
                group_ids.get(k, [f"{k}_{i}" for i in range(groups[k].shape[0])])
            )
        pca_df["subject_id"] = all_sids
        pca_df.to_csv(pca_dir / f"pca_scores_{spectral_range}.csv", index=False)

        # -- PCA loadings export (Phase B) --
        pca_cfg = config.get("analysis", {}).get("pca", {})
        top_k = int(pca_cfg.get("top_k_wavenumbers", 20))
        n_comp_export = int(pca_cfg.get("n_components_exported", 10))

        loadings_matrix = pca_result["loadings"][:n_comp_export]
        loadings_df = pd.DataFrame(
            loadings_matrix.T,
            index=[f"{w:.2f}" for w in wavenumbers],
            columns=[f"PC{i + 1}" for i in range(loadings_matrix.shape[0])],
        )
        loadings_df.index.name = "wavenumber"
        loadings_df.to_csv(pca_dir / f"pca_loadings_{spectral_range}.csv")

        top_df = rank_wavenumbers_per_pc(loadings_matrix, wavenumbers, top_k=top_k)
        top_df.to_csv(pca_dir / f"pca_top_wavenumbers_per_pc_{spectral_range}.csv", index=False)

        # Per-component variance table (sliced to exported components)
        variance_df = pd.DataFrame(
            {
                "pc": [f"PC{i + 1}" for i in range(loadings_matrix.shape[0])],
                "explained_variance_ratio": pca_result["explained_variance_ratio"][:loadings_matrix.shape[0]],
                "cumulative_variance": pca_result["cumulative_variance"][:loadings_matrix.shape[0]],
            }
        )
        variance_df.to_csv(pca_dir / f"pca_variance_{spectral_range}.csv", index=False)

        # Optional loading interpretation table
        suspect_windows = rw_cfg.get("exclude_windows", []) or rw_cfg.get("downweight_windows", []) or []
        suspect_windows = [[w[0], w[1]] for w in suspect_windows]
        interp_df = build_pca_loading_interpretation(
            loadings_matrix,
            wavenumbers,
            pca_result["explained_variance_ratio"],
            suspect_windows=suspect_windows,
            top_k=top_k,
        )
        interp_df.to_csv(pca_dir / f"pca_loading_interpretation_{spectral_range}.csv", index=False)

        logger.debug(
            "Saved PCA loadings CSV (%d wavenumbers × %d PCs)",
            len(wavenumbers),
            loadings_matrix.shape[0],
        )

        _log_phase("PHASE 4: Known Raman band analysis")

        band_dfs = []
        band_labels = []
        for key, spectra in groups.items():
            bdf = extract_band_intensities(spectra, wavenumbers)
            bdf["group"] = key
            band_dfs.append(bdf)
            band_labels.extend([key] * len(bdf))

        band_dir = phase_dir("band")

        if band_dfs:
            all_bands = pd.concat(band_dfs, ignore_index=True)
            band_cols = [c for c in all_bands.columns if c != "group"]
            if band_cols:
                all_bands.to_csv(
                    band_dir / f"band_intensities_{spectral_range}.csv", index=False
                )

                if _figure_flag("emit_band_boxplots", default=False):
                    fig, _ = plot_band_boxplots(
                        all_bands[band_cols],
                        np.array(band_labels),
                        title=f"Raman Band Intensities ({range_title})",
                    )
                    _save_fig(fig, "band_boxplots.png", band_dir)

                if _figure_flag("emit_band_rainclouds", default=True):
                    logger.debug("  Generating Raincloud distributions...")
                    fig_rc, _ = plot_all_bands_raincloud(
                        all_bands[band_cols],
                        np.array(band_labels),
                        title=f"Raman Band Distributions (Raincloud) — {range_title}",
                    )
                    _save_fig(fig_rc, "band_rainclouds.png", band_dir)

                # Per-band statistical tests: sex-stratified + pooled disease vs control
                band_test_rows = []
                for gender in ["female", "male"]:
                    g_key, c_key = f"{gender}_disease", f"{gender}_control"
                    if g_key in groups and c_key in groups:
                        test_df = compare_band_intensities_between_groups(
                            all_bands[band_cols], np.array(band_labels), g_key, c_key,
                        )
                        test_df.insert(0, "stratum", gender)
                        band_test_rows.append(test_df)

                pooled_condition = np.array(
                    ["disease" if lbl.endswith("_disease") else "control" for lbl in band_labels]
                )
                pooled_test_df = compare_band_intensities_between_groups(
                    all_bands[band_cols], pooled_condition, "disease", "control",
                )
                pooled_test_df.insert(0, "stratum", "pooled")
                band_test_rows.append(pooled_test_df)

                band_tests_df = pd.concat(band_test_rows, ignore_index=True)
                band_tests_df.to_csv(
                    band_dir / f"band_tests_{spectral_range}.csv", index=False
                )

        _log_phase("PHASE 5: Variance decomposition")

        intra, inter_subj, inter_grp = _estimate_variance_decomposition(
            df_index,
            spectral_range,
            groups,
        )

        def _fmt_pct(v: float | None) -> str:
            return f"{v:.1f}%" if v is not None and np.isfinite(v) else "N/A"

        logger.info(
            "  Measurement noise: %s  |  Inter-subject: %s  |  Inter-group: %s",
            _fmt_pct(intra),
            _fmt_pct(inter_subj),
            _fmt_pct(inter_grp),
        )

        var_dir = phase_dir("variance")

        fig, _ = plot_variance_decomposition(
            intra,
            inter_subj,
            inter_grp,
            title=f"Variance Decomposition ({range_title})",
        )
        _save_fig(fig, "variance_decomposition.png", var_dir)

        pd.DataFrame(
            [
                {
                    "range": spectral_range,
                    "measurement_noise_pct": intra,
                    "inter_subject_pct": inter_subj,
                    "inter_group_pct": inter_grp,
                }
            ]
        ).to_csv(var_dir / "variance_components.csv", index=False)

        _log_phase("PHASE 5b: Normality assessment")

        for key, spectra in groups.items():
            norm = test_normality(spectra)
            logger.debug(
                "  %s: %.1f%% of wavenumbers pass normality (Shapiro-Wilk, alpha=0.05)",
                key,
                norm["fraction_normal"] * 100,
            )

        intra_disease_cfg = (
            config.get("analysis", {}).get("advanced", {}).get("intra_disease", {})
        )
        if intra_disease_cfg.get("enabled", False):
            _log_phase("PHASE 6: Intra-disease clinical correlations")
            try:
                intra_csv = intra_disease_cfg.get("clinical_csv", "data/clinical_data_clean.csv")
                intra_clinical_df = load_clinical_data(intra_csv, data_index=df_index, write_curated=False)
                wn_id, spectra_id, clin_aligned = build_disease_subject_mean_matrix(
                    df_index, intra_clinical_df, spectral_range
                )
                if spectra_id.size == 0:
                    logger.warning("Intra-disease analysis: no disease subjects with cached spectra, skipping")
                else:
                    intra_dir = phase_dir("intra_disease")
                    corr_df = compute_intra_disease_correlations(spectra_id, wn_id, clin_aligned)
                    for target, grp in corr_df.groupby("target"):
                        grp.to_csv(
                            intra_dir / f"intra_disease_corr_{target}_{spectral_range}.csv", index=False
                        )
                        fig, _ = plot_correlation_spectrum(
                            grp["wavenumber"].to_numpy(dtype=float),
                            grp["rho"].to_numpy(dtype=float),
                            grp["significant"].to_numpy(dtype=bool),
                            title=f"Intra-disease correlation — {target} ({range_title})",
                        )
                        _save_fig(fig, f"intra_disease_corr_{target}_{spectral_range}.png", intra_dir)

                    summary_df = summarize_significant_targets(corr_df)
                    summary_df.to_csv(intra_dir / f"intra_disease_summary_{spectral_range}.csv", index=False)
                    logger.info(
                        "Intra-disease correlations: %d subjects, %d targets tested",
                        spectra_id.shape[0],
                        summary_df.shape[0],
                    )
            except Exception as exc:
                logger.warning("Intra-disease correlation phase failed: %s", exc)



    robust_cfg = config.get("analysis", {}).get("advanced", {}).get("robustness", {})
    if spectral_range != "both" and robust_cfg.get("enabled", False):
        _log_phase("PHASE 7b: Region-influence robustness")
        robust_dir = get_eda_phase_dir(spectral_range, "07b_region_robustness")
        try:
            suspect_windows = robust_cfg.get("suspect_windows", [])
            downweight_factor = float(robust_cfg.get("downweight_factor", 0.0))
            robust_rw = {
                "enabled": True,
                "exclude_windows": suspect_windows if downweight_factor == 0.0 else [],
                "downweight_windows": [
                    [lo, hi, downweight_factor]
                    for lo, hi in suspect_windows
                ]
                if downweight_factor > 0.0
                else [],
            }
            rs = int(config.get("analysis", {}).get("random_state", 42))

            rob_result = run_region_robustness(
                all_spectra,
                wavenumbers,
                all_labels,
                region_weighting=robust_rw,
                random_state=rs,
                n_components=min(3, all_spectra.shape[0] - 1),
            )
            rob_result["pca_comparison"]["summary"].to_csv(
                robust_dir / f"robustness_pca_{spectral_range}.csv", index=False
            )
            rob_result["loading_shift"].to_csv(
                robust_dir / f"pca_loading_shift_{spectral_range}.csv", index=False
            )
            write_region_influence_report(
                rob_result["pca_comparison"]["summary"],
                rob_result["loading_shift"],
                suspect_windows,
                robust_dir / f"region_influence_{spectral_range}.md",
            )
        except Exception as exc:
            logger.warning("Region robustness phase failed: %s", exc, exc_info=True)



    _log_phase("EDA COMPLETE")
    eda_base = _output_dir("eda_subdir")
    label = _range_label(spectral_range)
    logger.info("Outputs saved to: %s", eda_base / label)
    for key, pname in _PHASE_DIRS.items():
        logger.info("  %s/", pname)

    if spectral_range == "both":
        return result_both

    return {
        "groups": groups,
        "wavenumbers": wavenumbers,
        "pca_result": pca_result,
        "group_ids": group_ids,
    }


run_eda = _run_single_range_eda
