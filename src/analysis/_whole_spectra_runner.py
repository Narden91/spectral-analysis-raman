"""Whole-spectra EDA runner."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src import (
    WHOLE_SPECTRA_PHASE_DIRS,
    config,
    get_whole_spectra_phase_dir,
    whole_spectra_split_index,
    _output_dir,
    _range_label,
)
from src.analysis._single_range import (
    _estimate_variance_decomposition,
    _log_phase,
    _save_fig,
)
from src.analysis.joint_dr import (
    cca_between_regions,
    joint_pca_mfa,
    regional_pca_scores,
)
from src.analysis.statistics import (
    run_pca,
    summarize_group_centroid_distances,
)
from src.analysis.whole_spectra import (
    compute_band_intensities,
    compute_cross_region_ratios,
    cross_region_band_correlation,
    load_band_definitions,
    regional_snv_residual_variance,
    test_ratios_between_groups,
)
from src.visualization.eda_plots import (
    plot_band_boxplots,
    plot_cca_scatter,
    plot_correlation_matrix,
    plot_group_distance_heatmap,
    plot_mfa_scree,
    plot_pca_scores,
    plot_ratio_summary,
    plot_regional_variance_qc,
    plot_scree,
    plot_variance_decomposition,
)

logger = logging.getLogger(__name__)


def _partition_whole_spectra_bands(
    bands: dict[str, tuple[float, float]],
    wavenumbers: np.ndarray,
    split_index: int,
) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]]]:
    fingerprint_limit = float(wavenumbers[split_index - 1])
    ch_start = float(wavenumbers[split_index])
    fingerprint_bands: dict[str, tuple[float, float]] = {}
    ch_bands: dict[str, tuple[float, float]] = {}

    for band_name, limits in bands.items():
        lower, upper = limits
        if upper <= fingerprint_limit:
            fingerprint_bands[band_name] = limits
        elif lower >= ch_start:
            ch_bands[band_name] = limits
        else:
            raise ValueError(
                f"Band '{band_name}' crosses the whole-spectrum gap between regions"
            )

    return fingerprint_bands, ch_bands


def _run_whole_spectra_eda(
    df_index: pd.DataFrame,
    groups: dict[str, np.ndarray],
    wavenumbers: np.ndarray,
    group_ids: dict[str, list[str]],
    range_title: str,
) -> dict:
    all_spectra = np.vstack(list(groups.values()))
    all_labels = np.concatenate([np.full(v.shape[0], k) for k, v in groups.items()])
    all_sids = []
    for key in groups:
        all_sids.extend(group_ids.get(key, [f"{key}_{i}" for i in range(groups[key].shape[0])]))

    split_index = whole_spectra_split_index(wavenumbers)
    pca_result = None
    mfa_result = None
    regional_pca_result: dict[str, dict[str, object]] = {}
    cca_result = None
    relational_bands: dict[str, tuple[float, float]] = {}
    fingerprint_bands: dict[str, tuple[float, float]] = {}
    ch_bands: dict[str, tuple[float, float]] = {}
    band_intensity_df = pd.DataFrame()
    ratio_df = pd.DataFrame()
    ratio_test_df = pd.DataFrame()
    cross_region_correlations: dict[str, pd.DataFrame] = {}
    regional_variance_df = pd.DataFrame()

    if split_index is not None:
        relational_bands = load_band_definitions(config)
        if relational_bands:
            fingerprint_bands, ch_bands = _partition_whole_spectra_bands(
                relational_bands,
                wavenumbers,
                split_index,
            )
            band_intensity_df = compute_band_intensities(
                all_spectra,
                wavenumbers,
                relational_bands,
            )
    else:
        logger.warning(
            "Skipping split-dependent whole-spectrum relational analyses: split index could not be inferred"
        )

    _log_phase("PHASE 3: Whole-spectrum global structure")

    structure_dir = get_whole_spectra_phase_dir("global_structure")
    structure_components = min(
        int(config.get("analysis", {}).get("pca_n_components", 10)),
        all_spectra.shape[0] - 1,
    )

    distance_df = summarize_group_centroid_distances(groups)
    if not distance_df.empty:
        distance_df.to_csv(structure_dir / "group_distance_matrix_whole.csv")
        fig, _ = plot_group_distance_heatmap(
            distance_df,
            title=f"Group Centroid Distances ({range_title})",
        )
        _save_fig(fig, "group_distance_heatmap.png", structure_dir)

    pca_result = run_pca(
        all_spectra,
        n_components=structure_components,
        scale=False,
    )
    fig, _ = plot_pca_scores(
        pca_result["scores"],
        all_labels,
        pca_result["explained_variance_ratio"],
        title=f"Whole-Spectrum PCA ({range_title})",
    )
    _save_fig(fig, "pca_scores_pc1_pc2.png", structure_dir)

    fig, _ = plot_scree(
        pca_result["explained_variance_ratio"],
        title=f"Whole-Spectrum PCA Variance ({range_title})",
    )
    _save_fig(fig, "scree_plot.png", structure_dir)

    pca_scores_df = pd.DataFrame(
        pca_result["scores"],
        columns=[f"PC{i + 1}" for i in range(pca_result["scores"].shape[1])],
    )
    pca_scores_df["group"] = all_labels
    pca_scores_df["subject_id"] = all_sids
    pca_scores_df.to_csv(structure_dir / "pca_scores_whole.csv", index=False)

    pca_variance_df = pd.DataFrame(
        {
            "component": np.arange(1, len(pca_result["explained_variance_ratio"]) + 1),
            "explained_variance_ratio": pca_result["explained_variance_ratio"],
            "cumulative_variance": pca_result["cumulative_variance"],
        }
    )
    pca_variance_df.to_csv(structure_dir / "pca_variance_whole.csv", index=False)

    if split_index is not None:
        mfa_result = joint_pca_mfa(all_spectra, split_index, structure_components)
        fig, _ = plot_pca_scores(
            mfa_result["scores"],
            all_labels,
            mfa_result["explained_variance_ratio"],
            title=f"Whole-Spectrum MFA ({range_title})",
        )
        _save_fig(fig, "mfa_scores_pc1_pc2.png", structure_dir)

        fig, _ = plot_mfa_scree(
            mfa_result["explained_variance_ratio"],
            mfa_result["block_contributions"],
            title=f"Whole-Spectrum MFA Scree ({range_title})",
        )
        _save_fig(fig, "mfa_scree.png", structure_dir)

        mfa_scores_df = pd.DataFrame(
            mfa_result["scores"],
            columns=[f"PC{i + 1}" for i in range(mfa_result["scores"].shape[1])],
        )
        mfa_scores_df["group"] = all_labels
        mfa_scores_df["subject_id"] = all_sids
        mfa_scores_df.to_csv(structure_dir / "mfa_scores.csv", index=False)

        pd.DataFrame(
            {
                "component": np.arange(1, len(mfa_result["explained_variance_ratio"]) + 1),
                "explained_variance_ratio": mfa_result["explained_variance_ratio"],
                "cumulative_variance": mfa_result["cumulative_variance"],
            }
        ).to_csv(structure_dir / "mfa_variance.csv", index=False)
        mfa_result["block_contributions"].to_csv(
            structure_dir / "mfa_block_contributions.csv",
            index=False,
        )

        fingerprint_pca, ch_pca = regional_pca_scores(
            all_spectra,
            split_index,
            structure_components,
        )
        regional_pca_result = {
            "fingerprint": fingerprint_pca,
            "ch_stretch": ch_pca,
        }
        for region_name, result in regional_pca_result.items():
            scores_df = pd.DataFrame(
                result["scores"],
                columns=[f"PC{i + 1}" for i in range(result["scores"].shape[1])],
            )
            scores_df["group"] = all_labels
            scores_df["subject_id"] = all_sids
            scores_df.to_csv(
                structure_dir / f"regional_pca_scores_{region_name}.csv",
                index=False,
            )

        cca_result = cca_between_regions(
            fingerprint_pca["scores"],
            ch_pca["scores"],
            n_components=min(3, fingerprint_pca["scores"].shape[1], ch_pca["scores"].shape[1]),
        )
        pd.DataFrame(
            {
                "component": np.arange(1, len(cca_result["correlations"]) + 1),
                "correlation": cca_result["correlations"],
            }
        ).to_csv(structure_dir / "cca_correlations.csv", index=False)
        fig, _ = plot_cca_scatter(
            cca_result["fingerprint_scores"],
            cca_result["ch_scores"],
            all_labels,
            cca_result["correlations"],
            title=f"CCA Between Regions ({range_title})",
        )
        _save_fig(fig, "cca_scatter.png", structure_dir)

    _log_phase("PHASE 4: Whole-spectrum global drivers")

    drivers_dir = get_whole_spectra_phase_dir("global_drivers")

    if not band_intensity_df.empty:
        band_intensity_export = band_intensity_df.copy()
        band_intensity_export.insert(0, "group", all_labels)
        band_intensity_export.insert(0, "subject_id", all_sids)
        band_intensity_export.to_csv(drivers_dir / "band_intensities.csv", index=False)

        ratio_pairs = [
            (str(pair[0]), str(pair[1]))
            for pair in config.get("whole_spectra", {}).get("ratio_pairs", [])
        ]
        if ratio_pairs:
            ratio_df = compute_cross_region_ratios(band_intensity_df, ratio_pairs)
            ratio_export = ratio_df.copy()
            ratio_export.insert(0, "group", all_labels)
            ratio_export.insert(0, "subject_id", all_sids)
            ratio_export.to_csv(drivers_dir / "cross_region_ratios.csv", index=False)

            ratio_test_frames = [
                test_ratios_between_groups(
                    ratio_df,
                    np.array(
                        [
                            "disease" if "disease" in label else "control"
                            for label in all_labels
                        ]
                    ),
                    [("disease", "control")],
                )
            ]
            available_groups = set(all_labels.tolist())
            if {"female_disease", "female_control"}.issubset(available_groups):
                ratio_test_frames.append(
                    test_ratios_between_groups(
                        ratio_df,
                        all_labels,
                        [("female_disease", "female_control")],
                    )
                )
            if {"male_disease", "male_control"}.issubset(available_groups):
                ratio_test_frames.append(
                    test_ratios_between_groups(
                        ratio_df,
                        all_labels,
                        [("male_disease", "male_control")],
                    )
                )
            ratio_test_df = pd.concat(ratio_test_frames, ignore_index=True)
            ratio_test_df.to_csv(drivers_dir / "ratio_group_tests.csv", index=False)

            fig, _ = plot_band_boxplots(
                ratio_df,
                all_labels,
                title=f"Whole-Spectrum Ratios ({range_title})",
                ylabel="Ratio",
            )
            _save_fig(fig, "ratio_boxplots.png", drivers_dir)

            fig, _ = plot_ratio_summary(
                ratio_test_df,
                title=f"Cross-Region Ratio Summary ({range_title})",
            )
            _save_fig(fig, "ratio_summary.png", drivers_dir)

    _log_phase("PHASE 5: Whole-spectrum variance context")

    variance_dir = get_whole_spectra_phase_dir("variance_context")
    intra, inter_subj, inter_grp = _estimate_variance_decomposition(
        df_index,
        "both",
        groups,
    )
    if not np.isfinite(intra):
        logger.info(
            "  Measurement noise: not estimated  |  Inter-subject: %.1f%%  |  Inter-group: %.1f%%",
            inter_subj,
            inter_grp,
        )
        variance_df = pd.DataFrame(
            {
                "component": ["inter_subject", "inter_group"],
                "percent": [inter_subj, inter_grp],
            }
        )
    else:
        logger.info(
            "  Measurement noise: %.1f%%  |  Inter-subject: %.1f%%  |  Inter-group: %.1f%%",
            intra,
            inter_subj,
            inter_grp,
        )
        variance_df = pd.DataFrame(
            {
                "component": ["measurement_noise", "inter_subject", "inter_group"],
                "percent": [intra, inter_subj, inter_grp],
            }
        )
    variance_df.to_csv(variance_dir / "variance_summary_whole.csv", index=False)
    fig, _ = plot_variance_decomposition(
        intra,
        inter_subj,
        inter_grp,
        title=f"Variance Context ({range_title})",
    )
    _save_fig(fig, "variance_decomposition.png", variance_dir)

    if not band_intensity_df.empty and fingerprint_bands and ch_bands:
        correlation_cfg = config.get("whole_spectra", {}).get("correlation", {})
        cross_region_correlations = cross_region_band_correlation(
            band_intensity_df[list(fingerprint_bands.keys())],
            band_intensity_df[list(ch_bands.keys())],
            all_labels,
            method=str(correlation_cfg.get("method", "spearman")),
            n_bootstrap=int(correlation_cfg.get("n_bootstrap", 200)),
        )
        for group_name, frame in cross_region_correlations.items():
            frame.to_csv(
                variance_dir / f"cross_region_correlation_{group_name}.csv",
                index=False,
            )
            fig, _ = plot_correlation_matrix(
                frame,
                title=f"Cross-Region Correlation ({group_name})",
            )
            _save_fig(fig, f"cross_region_correlation_{group_name}.png", variance_dir)

    if split_index is not None:
        subject_spectra = {
            f"{subject_id}_spot{idx}": spectrum
            for idx, (subject_id, spectrum) in enumerate(zip(all_sids, all_spectra))
        }
        regional_variance_df = regional_snv_residual_variance(subject_spectra, split_index)
        regional_variance_df.insert(1, "group", all_labels)
        regional_variance_df.to_csv(
            variance_dir / "regional_snv_residual_variance.csv",
            index=False,
        )
        fig, _ = plot_regional_variance_qc(
            regional_variance_df,
            title=f"Regional SNV Residual Variance ({range_title})",
        )
        _save_fig(fig, "regional_snv_residual_variance.png", variance_dir)

    _log_phase("EDA COMPLETE")
    eda_base = _output_dir("eda_subdir")
    label = _range_label("both")
    logger.info("Outputs saved to: %s", eda_base / label)
    for dirname in WHOLE_SPECTRA_PHASE_DIRS.values():
        logger.info("  %s/", dirname)

    return {
        "groups": groups,
        "wavenumbers": wavenumbers,
        "pca_result": pca_result,
        "mfa_result": mfa_result,
        "regional_pca_result": regional_pca_result,
        "cca_result": cca_result,
        "group_ids": group_ids,
        "variance_summary": variance_df,
        "band_intensities": band_intensity_df,
        "ratio_tests": ratio_test_df,
        "cross_region_correlations": cross_region_correlations,
        "regional_snv_residual_variance": regional_variance_df,
    }
