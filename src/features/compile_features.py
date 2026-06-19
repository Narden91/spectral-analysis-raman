from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from src import config
from src.aggregation.subject_builder import build_subject_hierarchy
from src.constants import (
    CANONICAL_FP_CHANNELS,
    CANONICAL_REGION_BOUNDS,
    CANONICAL_TOTAL_CHANNELS,
)
from src.features.ch_lipid_state_features import extract_ch_lipid_state_features
from src.features.clinical_features import build_clinical_feature_block
from src.features.eda_band_features import (
    _get_canonical_wavenumbers,
    _wn_to_channel_idx,
    extract_band_features,
    extract_band_fraction_features,
    extract_ratio_features,
    extract_sex_covariate_block,
)
from src.features.feature_metadata import assert_no_identifier_leakage
from src.features.heterogeneity_features import extract_heterogeneity_features
from src.features.pca_features import fit_region_pca, transform_region_pca
from src.features.peak_features import (
    detect_training_peaks,
    extract_peak_features,
    extract_relative_peak_features,
)
from src.features.spectral_scaler import (
    RobustSpectralScaler,
    fit_spectral_scaler,
    transform_spectral_scaler,
)
from src.features.statistical_features import extract_statistical_features

logger = logging.getLogger(__name__)

_ACQUISITION_COUNT_MARKERS = (
    "n_areas",
    "n_slides",
    "n_paired",
    "n_measurements",
)



def _apply_ml_spectral_scaler(
    train_vectors: np.ndarray,
    test_vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return train-fold-only spectral-channel scaled copies of subject vectors.

    The scaled copies are used only for channel-space learned transforms such
    as spectral PCA.  Engineered Raman features (peaks, bands, ratios,
    statistics, heterogeneity, CH lipid-state) are derived from physically
    preprocessed, unscaled subject vectors.  The final mixed feature matrix is
    scaled fold-locally by a separate feature-level StandardScaler inside the
    CV/holdout loop.
    """
    scaler_cfg = config.get("ml_spectral_scaler", {})
    scaler_type = str(scaler_cfg.get("type", "robust")).strip().lower()
    if scaler_type == "none":
        return train_vectors.copy(), test_vectors.copy(), {"type": "none", "applied": False}

    epsilon = float(scaler_cfg.get("epsilon", 1.0e-9))
    scaler = fit_spectral_scaler(train_vectors, scaler_type=scaler_type, epsilon=epsilon)
    train_scaled = transform_spectral_scaler(scaler, train_vectors)
    test_scaled = transform_spectral_scaler(scaler, test_vectors)

    info: dict = {
        "type": scaler_type,
        "applied": True,
        "epsilon": epsilon,
    }
    if isinstance(scaler, RobustSpectralScaler):
        info["n_zero_scale_channels"] = int(scaler.n_zero_scale_channels_)
    return train_scaled, test_scaled, info



def _compile_pca_track(
    train_subject, test_subject, train_vectors_scaled, test_vectors_scaled, 
    resolved_split, variance_threshold, random_state
) -> tuple:
    pca_model = fit_region_pca(
        train_vectors_scaled,
        variance_threshold=variance_threshold,
        split_index=resolved_split,
        random_state=random_state,
    )
    pca_train = pca_model["train_scores"]
    pca_test = transform_region_pca(pca_model, test_vectors_scaled)
    pca_feature_names = _pca_feature_names(pca_model)
    track_train = _make_named_block_df(train_subject["subject_id"], pca_train, pca_feature_names)
    track_test = _make_named_block_df(test_subject["subject_id"], pca_test, pca_feature_names)
    return track_train, track_test, pca_model

def _compile_peak_tracks(
    train_subject, test_subject, train_vectors, test_vectors, 
    resolved_split, include_relative_features, 
    relative_train_vectors, relative_test_vectors, relative_wavenumbers, relative_eps, relative_cfg
) -> tuple:
    import numpy as np
    import pandas as pd
    from src import config
    
    peak_cfg = config.get("peaks", {})
    peak_indices = detect_training_peaks(
        np.mean(train_vectors, axis=0),
        prominence_percentile=float(peak_cfg.get("prominence_percentile", 75)),
        min_distance=int(peak_cfg.get("min_distance", 15)),
        canonical_bin_width=int(peak_cfg.get("canonical_bin_width", 5)),
    )
    peak_labels = [f"peak_{int(idx)}" for idx in peak_indices.tolist()]
    
    peak_train = extract_peak_features(
        train_vectors, peak_indices, peak_labels=peak_labels,
        peak_window=int(peak_cfg.get("peak_window", 5)), n_top_for_ratios=int(peak_cfg.get("n_top_for_ratios", 6))
    )
    peak_test = extract_peak_features(
        test_vectors, peak_indices, peak_labels=peak_labels,
        peak_window=int(peak_cfg.get("peak_window", 5)), n_top_for_ratios=int(peak_cfg.get("n_top_for_ratios", 6))
    )
    
    track_b_train = _with_subject_id(train_subject["subject_id"], peak_train, "peak")
    track_b_test = _with_subject_id(test_subject["subject_id"], peak_test, "peak")
    
    rel_train = pd.DataFrame()
    rel_test = pd.DataFrame()
    relative_reference_peak_idx = None
    
    if include_relative_features and relative_train_vectors is not None and relative_test_vectors is not None:
        relative_reference_peak_idx = _resolve_reference_peak_idx(
            peak_indices,
            reference_wavenumber=float(relative_cfg.get("reference_peak_wavenumber", 2850.0)),
            n_channels=train_vectors.shape[1],
            split_index=resolved_split,
            tolerance_channels=int(relative_cfg.get("reference_tolerance_channels", 20)),
        )
        relpeak_train = extract_relative_peak_features(
            relative_train_vectors, peak_indices, peak_labels=peak_labels,
            peak_window=int(peak_cfg.get("peak_window", 5)),
            reference_idx=relative_reference_peak_idx,
            wavenumbers=relative_wavenumbers, eps=relative_eps,
        )
        relpeak_test = extract_relative_peak_features(
            relative_test_vectors, peak_indices, peak_labels=peak_labels,
            peak_window=int(peak_cfg.get("peak_window", 5)),
            reference_idx=relative_reference_peak_idx,
            wavenumbers=relative_wavenumbers, eps=relative_eps,
        )
        rel_train = _with_subject_id(train_subject["subject_id"], relpeak_train, "relpeak")
        rel_test = _with_subject_id(test_subject["subject_id"], relpeak_test, "relpeak")
        
    return track_b_train, track_b_test, rel_train, rel_test, peak_indices, peak_labels, relative_reference_peak_idx

def _compile_stat_track(
    train_subject, test_subject, train_vectors, test_vectors, split_index, static
) -> tuple:
    if "track_c" in static:
        track_c_all = static["track_c"]
        train_ids = set(train_subject["subject_id"].astype(str).tolist())
        test_ids = set(test_subject["subject_id"].astype(str).tolist())
        track_train = track_c_all[track_c_all["subject_id"].isin(train_ids)].copy()
        track_test = track_c_all[track_c_all["subject_id"].isin(test_ids)].copy()
    else:
        stat_train = extract_statistical_features(train_vectors, split_index=split_index)
        stat_test = extract_statistical_features(test_vectors, split_index=split_index)
        track_train = _with_subject_id(train_subject["subject_id"], stat_train, "stat")
        track_test = _with_subject_id(test_subject["subject_id"], stat_test, "stat")
    return track_train, track_test

def _compile_clinical_track(
    train_subject, test_subject, train_area_df, test_area_df, clinical_df, static
) -> tuple:
    if "track_e" in static:
        track_e_all = static["track_e"]
        train_ids = set(train_subject["subject_id"].astype(str).tolist())
        test_ids = set(test_subject["subject_id"].astype(str).tolist())
        clinical_train = track_e_all[track_e_all["subject_id"].isin(train_ids)].copy()
        clinical_test = track_e_all[track_e_all["subject_id"].isin(test_ids)].copy()
    else:
        clinical_train = build_clinical_feature_block(train_area_df, clinical_df=clinical_df)
        clinical_test = build_clinical_feature_block(test_area_df, clinical_df=clinical_df)
    return _prefix_columns(clinical_train, "clin"), _prefix_columns(clinical_test, "clin")

def _compile_eda_bands_track(
    train_subject, test_subject, train_vectors, test_vectors, split_index
) -> tuple:
    import pandas as pd
    band_train = extract_band_features(train_vectors, split_index=split_index)
    band_test = extract_band_features(test_vectors, split_index=split_index)
    ratio_train = extract_ratio_features(train_vectors, split_index=split_index)
    ratio_test = extract_ratio_features(test_vectors, split_index=split_index)
    
    t_band = _with_subject_id(train_subject["subject_id"], band_train, "band") if not band_train.empty else pd.DataFrame()
    te_band = _with_subject_id(test_subject["subject_id"], band_test, "band") if not band_train.empty else pd.DataFrame()
    t_ratio = _with_subject_id(train_subject["subject_id"], ratio_train, "ratio") if not ratio_train.empty else pd.DataFrame()
    te_ratio = _with_subject_id(test_subject["subject_id"], ratio_test, "ratio") if not ratio_train.empty else pd.DataFrame()
    
    return t_band, te_band, t_ratio, te_ratio

def _add_sex_interactions(train_compiled, test_compiled, train_numeric, test_numeric):
    import numpy as np
    sex_col = "cov_sex_binary"
    if sex_col in train_numeric.columns and sex_col in test_numeric.columns:
        sex_train_vals = train_numeric[sex_col].to_numpy(dtype=float)
        sex_test_vals = test_numeric[sex_col].to_numpy(dtype=float)
        feature_cols = [c for c in train_numeric.columns if c != sex_col]
        
        train_inter = train_numeric[feature_cols].to_numpy(dtype=float) * sex_train_vals[:, np.newaxis]
        test_inter = test_numeric[feature_cols].to_numpy(dtype=float) * sex_test_vals[:, np.newaxis]
        inter_names = [f"{c}_x_sex" for c in feature_cols]
        
        train_inter_df = _make_block_df(train_compiled["subject_id"], train_inter, "inter")
        train_inter_df.columns = ["subject_id"] + inter_names
        test_inter_df = _make_block_df(test_compiled["subject_id"], test_inter, "inter")
        test_inter_df.columns = ["subject_id"] + inter_names
        
        train_compiled = train_compiled.merge(train_inter_df, on="subject_id", how="left")
        test_compiled = test_compiled.merge(test_inter_df, on="subject_id", how="left")
    return train_compiled, test_compiled

def compile_feature_matrices(
    train_area_df: pd.DataFrame,
    test_area_df: pd.DataFrame,
    clinical_df: pd.DataFrame | None = None,
    split_index: int | None = None,
    variance_threshold: float | None = None,
    include_heterogeneity: bool = False,
    include_clinical: bool = False,
    require_disjoint_subjects: bool = False,
    random_state: int = 42,
    precomputed_static_tracks: dict[str, pd.DataFrame] | None = None,
    include_eda_bands: bool = False,
    include_ch_lipid_state: bool = False,
    include_sex_covariate: bool = False,
    include_sex_interactions: bool = False,
    include_relative_features: bool = False,
) -> dict:
    import numpy as np
    import pandas as pd
    from src import config

    _validate_area_df(train_area_df, "train_area_df")
    _validate_area_df(test_area_df, "test_area_df")
    if require_disjoint_subjects:
        _assert_disjoint_subject_sets(train_area_df, test_area_df)

    train_subject = aggregate_subject_vectors(train_area_df)
    test_subject = aggregate_subject_vectors(test_area_df)
    train_vectors = np.vstack(list(train_subject["mean_vector"].to_numpy()))
    test_vectors = np.vstack(list(test_subject["mean_vector"].to_numpy()))

    train_vectors_scaled, test_vectors_scaled, scaler_info = _apply_ml_spectral_scaler(train_vectors, test_vectors)
    resolved_split = infer_split_index(split_index=split_index, n_features=train_vectors.shape[1])
    
    relative_cfg = config.get("relative_features", {})
    relative_basis_requested = str(relative_cfg.get("basis", "baseline_corrected"))
    relative_eps = float(relative_cfg.get("eps", 1.0e-9))
    
    relative_train_vectors, relative_test_vectors, relative_basis_used, relative_wavenumbers = None, None, None, None
    if include_relative_features:
        relative_train_vectors, relative_test_vectors, relative_basis_used = _resolve_relative_basis(
            train_subject, test_subject, train_vectors, test_vectors, basis=relative_basis_requested,
        )
        relative_wavenumbers = _resolve_feature_wavenumbers(n_features=train_vectors.shape[1], split_index=resolved_split)

    track_a_train, track_a_test, pca_model = _compile_pca_track(
        train_subject, test_subject, train_vectors_scaled, test_vectors_scaled, resolved_split, variance_threshold, random_state
    )
    
    track_b_train, track_b_test, rel_train, rel_test, peak_indices, peak_labels, relative_reference_peak_idx = _compile_peak_tracks(
        train_subject, test_subject, train_vectors, test_vectors, resolved_split, include_relative_features,
        relative_train_vectors, relative_test_vectors, relative_wavenumbers, relative_eps, relative_cfg
    )

    train_tracks = [track_a_train, track_b_train]
    test_tracks = [track_a_test, track_b_test]
    
    if not rel_train.empty:
        train_tracks.append(rel_train)
        test_tracks.append(rel_test)

    track_c_train, track_c_test = _compile_stat_track(
        train_subject, test_subject, train_vectors, test_vectors, int(pca_model["split_index"]), precomputed_static_tracks or {}
    )
    train_tracks.append(track_c_train)
    test_tracks.append(track_c_test)

    if include_heterogeneity:
        hetero_cfg = config.get("heterogeneity", {})
        hetero_train = extract_heterogeneity_features(train_area_df, peak_indices=peak_indices, n_targeted_peaks=int(hetero_cfg.get("n_targeted_peaks", 5)))
        hetero_test = extract_heterogeneity_features(test_area_df, peak_indices=peak_indices, n_targeted_peaks=int(hetero_cfg.get("n_targeted_peaks", 5)))
        train_tracks.append(_prefix_columns(hetero_train, "hetero"))
        test_tracks.append(_prefix_columns(hetero_test, "hetero"))

    if include_clinical:
        clin_train, clin_test = _compile_clinical_track(
            train_subject, test_subject, train_area_df, test_area_df, clinical_df, precomputed_static_tracks or {}
        )
        train_tracks.append(clin_train)
        test_tracks.append(clin_test)

    if include_eda_bands:
        b_train, b_test, r_train, r_test = _compile_eda_bands_track(
            train_subject, test_subject, train_vectors, test_vectors, int(pca_model["split_index"])
        )
        for tr in [b_train, r_train]:
            if not tr.empty:
                train_tracks.append(tr)
        for tr in [b_test, r_test]:
            if not tr.empty:
                test_tracks.append(tr)

    if include_relative_features and relative_train_vectors is not None and relative_test_vectors is not None:
        bf_train = extract_band_fraction_features(relative_train_vectors, split_index=int(pca_model["split_index"]), eps=relative_eps)
        bf_test = extract_band_fraction_features(relative_test_vectors, split_index=int(pca_model["split_index"]), eps=relative_eps)
        if not bf_train.empty:
            train_tracks.append(_with_subject_id(train_subject["subject_id"], bf_train, "bandfrac"))
            test_tracks.append(_with_subject_id(test_subject["subject_id"], bf_test, "bandfrac"))

    track_g_audit = {}
    if include_ch_lipid_state:
        g_train, g_test, track_g_audit = extract_ch_lipid_state_features(
            pd.concat([train_area_df, test_area_df], ignore_index=True, sort=False),
            train_subject_ids=train_subject["subject_id"].astype(str).tolist(),
            test_subject_ids=test_subject["subject_id"].astype(str).tolist(),
            random_state=random_state,
        )
        if not g_train.empty:
            train_tracks.append(_prefix_columns(g_train, "chlipid"))
            test_tracks.append(_prefix_columns(g_test, "chlipid"))

    if include_sex_covariate:
        train_tracks.append(_prefix_columns(extract_sex_covariate_block(train_area_df), "cov"))
        test_tracks.append(_prefix_columns(extract_sex_covariate_block(test_area_df), "cov"))

    train_compiled = _merge_tracks(train_subject["subject_id"], train_tracks)
    test_compiled = _merge_tracks(test_subject["subject_id"], test_tracks)
    train_compiled, test_compiled = _align_feature_columns(train_compiled, test_compiled)

    train_numeric = train_compiled.drop(columns=["subject_id"])
    test_numeric = test_compiled.drop(columns=["subject_id"])

    _log_nan_features("train", train_numeric)
    _log_nan_features("test", test_numeric)

    if include_sex_interactions and include_sex_covariate:
        train_compiled, test_compiled = _add_sex_interactions(train_compiled, test_compiled, train_numeric, test_numeric)

    x_train = train_compiled.drop(columns=["subject_id"]).to_numpy(dtype=float)
    x_test = test_compiled.drop(columns=["subject_id"]).to_numpy(dtype=float)
    feature_names = train_compiled.drop(columns=["subject_id"]).columns.tolist()
    
    _assert_no_acquisition_count_features(feature_names)
    assert_no_identifier_leakage(feature_names, x_train, x_test)

    return {
        "x_train": x_train,
        "x_test": x_test,
        "train_subject_ids": train_compiled["subject_id"].tolist(),
        "test_subject_ids": test_compiled["subject_id"].tolist(),
        "feature_names": feature_names,
        "train_features_df": train_compiled,
        "test_features_df": test_compiled,
        "artifacts": {
            "split_index": int(pca_model["split_index"]),
            "pca_model": pca_model,
            "peak_indices": np.asarray(peak_indices, dtype=int),
            "peak_labels": peak_labels,
            "include_heterogeneity": bool(include_heterogeneity),
            "include_clinical": bool(include_clinical),
            "include_eda_bands": bool(include_eda_bands),
            "include_ch_lipid_state": bool(include_ch_lipid_state),
            "include_sex_covariate": bool(include_sex_covariate),
            "include_sex_interactions": bool(include_sex_interactions),
            "include_relative_features": bool(include_relative_features),
            "relative_reference_peak_idx": relative_reference_peak_idx,
            "relative_basis_used": relative_basis_used,
            "track_g_audit": track_g_audit,
            "spectral_scaler": scaler_info,
        },
    }

def aggregate_subject_vectors(area_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate area-level spectra to one slide-aware mean vector per subject."""
    _validate_area_df(area_df, "area_df")
    hierarchy = build_subject_hierarchy(area_df)
    subject_table = hierarchy["subject_table"].copy()
    if subject_table.empty:
        return pd.DataFrame(columns=["subject_id", "mean_vector", "mean_vector_bc"])

    out = subject_table.rename(columns={"subject_mean": "mean_vector"})[["subject_id", "mean_vector"]]
    out["mean_vector_bc"] = None

    if _has_complete_vector_column(area_df, "area_mean_bc"):
        area_df_bc = area_df.copy()
        area_df_bc["area_mean"] = area_df_bc["area_mean_bc"]
        hierarchy_bc = build_subject_hierarchy(area_df_bc)
        subject_bc = hierarchy_bc["subject_table"].copy()
        if not subject_bc.empty:
            bc_block = subject_bc.rename(columns={"subject_mean": "mean_vector_bc"})[
                ["subject_id", "mean_vector_bc"]
            ]
            out = out.merge(bc_block, on="subject_id", how="left", suffixes=("", "_drop"))
            if "mean_vector_bc_drop" in out.columns:
                out["mean_vector_bc"] = out["mean_vector_bc_drop"]
                out = out.drop(columns=["mean_vector_bc_drop"])

    return out[["subject_id", "mean_vector", "mean_vector_bc"]]


def precompute_static_tracks(
    area_df: pd.DataFrame,
    clinical_df: pd.DataFrame | None = None,
    split_index: int | None = None,
    include_clinical: bool = False,
) -> dict[str, pd.DataFrame]:
    """Precompute Tracks C and E for the full cohort.

    These tracks are independent of train/test split and can be computed once
    and reused across CV folds to avoid redundant computation.
    """
    subject_df = aggregate_subject_vectors(area_df)
    if subject_df.empty:
        return {}

    vectors = np.vstack(list(subject_df["mean_vector"].to_numpy()))
    resolved_split = infer_split_index(split_index=split_index, n_features=vectors.shape[1])

    stat = extract_statistical_features(vectors, split_index=int(resolved_split))
    tracks: dict[str, pd.DataFrame] = {
        "track_c": _with_subject_id(subject_df["subject_id"], stat, "stat"),
    }

    if include_clinical:
        clinical = build_clinical_feature_block(area_df, clinical_df=clinical_df)
        tracks["track_e"] = clinical

    return tracks


def _validate_area_df(area_df: pd.DataFrame, name: str) -> None:
    required = {"subject_id", "slide_number", "area_mean", "area_std", "n_valid_pixels"}
    missing = required - set(area_df.columns)
    if missing:
        raise ValueError(f"{name} missing required columns: {sorted(missing)}")
    if area_df.empty:
        raise ValueError(f"{name} is empty")


def _assert_disjoint_subject_sets(train_area_df: pd.DataFrame, test_area_df: pd.DataFrame) -> None:
    train_subjects = set(train_area_df["subject_id"].astype(str).tolist())
    test_subjects = set(test_area_df["subject_id"].astype(str).tolist())
    overlap = sorted(train_subjects & test_subjects)
    if overlap:
        raise ValueError(f"Train/test subject overlap: {overlap[:10]}")


def _make_block_df(subject_ids: pd.Series, values: np.ndarray, prefix: str) -> pd.DataFrame:
    cols = [f"{prefix}_{i}" for i in range(values.shape[1])]
    return _make_named_block_df(subject_ids, values, cols)


def _make_named_block_df(subject_ids: pd.Series, values: np.ndarray, columns: list[str]) -> pd.DataFrame:
    if len(columns) != values.shape[1]:
        raise ValueError("Number of feature columns must match values width")
    out = pd.DataFrame(values, columns=columns)
    out.insert(0, "subject_id", subject_ids.astype(str).to_numpy())
    return out


def _pca_feature_names(pca_model: dict) -> list[str]:
    n_fp = int(pca_model.get("region_1300", {}).get("n_components", 0))
    n_ch = int(pca_model.get("region_2850", {}).get("n_components", 0))
    names = [f"pca_fingerprint_pc{i + 1}" for i in range(n_fp)]
    names.extend(f"pca_ch_stretch_pc{i + 1}" for i in range(n_ch))
    return names


def _has_complete_vector_column(area_df: pd.DataFrame, column: str) -> bool:
    if column not in area_df.columns:
        return False
    return bool(area_df[column].map(_is_vector_value).all())


def _is_vector_value(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and np.isnan(value):
        return False
    arr = np.asarray(value)
    return arr.ndim >= 1 and arr.size > 0


def _resolve_relative_basis(
    train_subject: pd.DataFrame,
    test_subject: pd.DataFrame,
    train_vectors: np.ndarray,
    test_vectors: np.ndarray,
    basis: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    basis_name = str(basis).strip().lower()
    if basis_name not in {"baseline_corrected", "snv_shift"}:
        raise ValueError(f"Unknown relative feature basis: {basis}")

    if basis_name == "baseline_corrected":
        if _subject_has_complete_vectors(train_subject, "mean_vector_bc") and _subject_has_complete_vectors(
            test_subject,
            "mean_vector_bc",
        ):
            train_bc = np.vstack(list(train_subject["mean_vector_bc"].to_numpy()))
            test_bc = np.vstack(list(test_subject["mean_vector_bc"].to_numpy()))
            return train_bc, test_bc, "baseline_corrected"
        logger.warning(
            "Relative features requested baseline_corrected basis but area_mean_bc is unavailable; falling back to snv_shift.",
        )

    return _snv_shift_vectors(train_vectors), _snv_shift_vectors(test_vectors), "snv_shift"


def _subject_has_complete_vectors(subject_df: pd.DataFrame, column: str) -> bool:
    if column not in subject_df.columns or subject_df.empty:
        return False
    return bool(subject_df[column].map(_is_vector_value).all())


def _snv_shift_vectors(vectors: np.ndarray) -> np.ndarray:
    arr = np.asarray(vectors, dtype=float)
    row_min = np.nanmin(arr, axis=1, keepdims=True)
    return np.maximum(arr - row_min, 0.0)


def _resolve_feature_wavenumbers(n_features: int, split_index: int) -> np.ndarray:
    canonical = _get_canonical_wavenumbers()
    if canonical is not None and canonical.size == n_features:
        return canonical.astype(float, copy=False)

    fp_start, fp_end = CANONICAL_REGION_BOUNDS["fingerprint"]
    ch_start, ch_end = CANONICAL_REGION_BOUNDS["ch_stretch"]
    fp = np.linspace(fp_start, fp_end, num=split_index, dtype=float)
    ch = np.linspace(ch_start, ch_end, num=n_features - split_index, dtype=float)
    return np.concatenate([fp, ch])


def _resolve_reference_peak_idx(
    peak_indices: np.ndarray,
    reference_wavenumber: float,
    n_channels: int,
    split_index: int,
    tolerance_channels: int,
) -> int | None:
    peaks = np.asarray(peak_indices, dtype=int)
    if peaks.size == 0:
        return None

    target_idx = _wn_to_channel_idx(float(reference_wavenumber), n_channels, split_index)
    distances = np.abs(peaks - target_idx)
    nearest_pos = int(np.argmin(distances))
    nearest_distance = int(distances[nearest_pos])
    if nearest_distance > int(tolerance_channels):
        logger.warning(
            "Relative-feature reference peak near %.1f cm-1 was not detected within %d channels.",
            float(reference_wavenumber),
            int(tolerance_channels),
        )
        return None
    return int(peaks[nearest_pos])


def _with_subject_id(subject_ids: pd.Series, block_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    block = block_df.copy()
    block.columns = [f"{prefix}_{c}" for c in block.columns]
    block.insert(0, "subject_id", subject_ids.astype(str).to_numpy())
    return block


def _prefix_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = df.copy()
    renamed = {}
    for c in out.columns:
        if c == "subject_id":
            continue
        renamed[c] = f"{prefix}_{c}"
    return out.rename(columns=renamed)


def _merge_tracks(subject_ids: pd.Series, tracks: list[pd.DataFrame]) -> pd.DataFrame:
    merged = pd.DataFrame({"subject_id": subject_ids.astype(str).to_numpy()})
    for track in tracks:
        merged = merged.merge(track, on="subject_id", how="left")
    return merged


def _align_feature_columns(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align train/test feature schemas while preserving train feature order."""
    train = train_df.copy()
    test = test_df.copy()
    train_cols = [c for c in train.columns if c != "subject_id"]
    test_cols = [c for c in test.columns if c != "subject_id"]
    ordered_cols = train_cols + [c for c in test_cols if c not in train_cols]

    for col in ordered_cols:
        default = _missing_feature_default(col)
        if col not in train.columns:
            train[col] = default
        if col not in test.columns:
            test[col] = default

    return train[["subject_id", *ordered_cols]], test[["subject_id", *ordered_cols]]


def _assert_no_acquisition_count_features(feature_names: list[object]) -> None:
    offenders = sorted(
        {
            str(name)
            for name in feature_names
            if any(marker in str(name).lower() for marker in _ACQUISITION_COUNT_MARKERS)
        }
    )
    if offenders:
        raise ValueError(
            "Acquisition-count feature column(s) leaked into model matrix: "
            f"{offenders}"
        )


def _missing_feature_default(column: object) -> float:
    name = str(column)
    if name.startswith("chlipid_ch_max_cluster"):
        return -1.0
    if name.startswith("chlipid_"):
        return 0.0
    return np.nan


def infer_split_index(split_index: int | None, n_features: int) -> int:
    if n_features < 2:
        raise ValueError("Need at least 2 channels to split regions")

    if split_index is not None:
        idx = int(split_index)
        if idx <= 0 or idx >= n_features:
            raise ValueError("split_index must be between 1 and n_features-1")
        return idx

    cfg_split = int(config.get("pca", {}).get("split_index", CANONICAL_FP_CHANNELS))
    if n_features == CANONICAL_TOTAL_CHANNELS and 0 < cfg_split < n_features:
        return cfg_split

    midpoint = int(n_features // 2)
    if n_features == CANONICAL_TOTAL_CHANNELS:
        logger.warning(
            "Invalid canonical split_index=%d for n_features=%d; falling back to midpoint=%d",
            cfg_split,
            n_features,
            midpoint,
        )
    else:
        logger.debug(
            "Using midpoint split_index=%d for non-canonical feature width n_features=%d",
            midpoint,
            n_features,
        )

    return midpoint

def _log_nan_features(name: str, feature_df: pd.DataFrame) -> None:
    nan_mask = feature_df.isna()
    if not nan_mask.to_numpy().any():
        return

    n_cells = int(nan_mask.to_numpy().sum())
    nan_cols = [str(col) for col in feature_df.columns[nan_mask.any(axis=0).to_numpy()]]
    non_clinical = [
        col
        for col in nan_cols
        if not col.startswith("clin_") and not (col.startswith("relpeak_") and col.endswith("_relint_ref"))
    ]

    if not non_clinical:
        logger.debug(
            "NaN values in compiled %s features are limited to expected clinical or relative-reference columns: %s",
            name,
            ", ".join(nan_cols),
        )
        return

    logger.warning(
        "Unexpected NaN values in compiled %s features: %d cells across %d column(s): %s",
        name,
        n_cells,
        len(nan_cols),
        ", ".join(non_clinical),
    )
