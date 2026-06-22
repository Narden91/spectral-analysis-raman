"""Phase 11: Integrated candidate-region evidence table.

Builds a single priority-ranked table from the cross-analysis candidate
regions (Phase 6), covariate-adjusted models (Phase 7), PLS VIP coverage
(Phase 8), PLS-L1DA selection (Phase 9), and biological PCA loading overlap.

Evidence levels are conservative and descriptive only; no causal,
diagnostic, predictive, or biomarker claims are made.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src import config, get_biological_pca_canonical_dir, get_cross_region_phase_dir

logger = logging.getLogger(__name__)

_BIOLOGICAL_PCA_MODELS = ("PCA_1330", "PCA_2850", "PCA_whole")

_DEFAULT_COLUMNS = [
    "candidate_region_id",
    "spectral_block",
    "start_cm-1",
    "end_cm-1",
    "width_cm-1",
    "source_analyses",
    "n_supporting_sources",
    "fdr_significant",
    "cluster_permutation_significant",
    "covariate_persisted",
    "pls_vip_coverage",
    "splsda_selected",
    "biological_pca_overlap",
    "overlapping_biological_pcs",
    "biochemical_class",
    "max_abs_effect",
    "mean_abs_effect",
    "integrated_abs_effect",
    "evidence_level",
    "interpretation_note",
]


def _cfg(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    return config if cfg is None else cfg


def _integrated_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("analysis", {}).get("integrated_candidates", {})


def _load_csv_if_exists(path: str | Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        logger.warning("Expected input table not found: %s", path)
        return None
    return pd.read_csv(path)


def _region_feature_names(feature_columns: list[str], region: pd.Series) -> list[str]:
    block = str(region["spectral_block"])
    lower = float(region["start_cm-1"])
    upper = float(region["end_cm-1"])
    out: list[str] = []
    for column in feature_columns:
        if not column.startswith(f"{block}__"):
            continue
        _, raw = column.split("__", 1)
        try:
            wavenumber = float(raw)
        except ValueError:
            continue
        if lower <= wavenumber <= upper:
            out.append(column)
    return out


def _annotate_covariate_persistence(
    candidates: pd.DataFrame,
    covariate_models: pd.DataFrame | None,
) -> pd.Series:
    if covariate_models is None or covariate_models.empty or "status" not in covariate_models.columns:
        return pd.Series(False, index=candidates.index, dtype=bool)

    ok_models = covariate_models.loc[covariate_models["status"].eq("ok")].copy()
    if ok_models.empty or "candidate_region_id" not in ok_models.columns:
        return pd.Series(False, index=candidates.index, dtype=bool)

    persisted = set(ok_models["candidate_region_id"].dropna().astype(str).unique())
    return candidates["candidate_region_id"].astype(str).isin(persisted)


def _annotate_pls_vip_coverage(
    candidates: pd.DataFrame,
    pls_vip: pd.DataFrame | None,
    feature_columns: list[str],
    vip_threshold: float,
) -> pd.Series:
    if pls_vip is None or pls_vip.empty:
        return pd.Series(False, index=candidates.index, dtype=bool)

    required = {"feature_name", "vip_score", "above_threshold"}
    if not required.issubset(pls_vip.columns):
        logger.warning("PLS VIP table missing expected columns; skipping VIP annotation")
        return pd.Series(False, index=candidates.index, dtype=bool)

    vip_features = set(
        pls_vip.loc[pls_vip["above_threshold"].fillna(False), "feature_name"].astype(str).unique()
    )
    out = []
    for _, region in candidates.iterrows():
        region_features = set(_region_feature_names(feature_columns, region))
        out.append(bool(region_features & vip_features))
    return pd.Series(out, index=candidates.index, dtype=bool)


def _annotate_splsda_selection(
    candidates: pd.DataFrame,
    splsda_bands: pd.DataFrame | None,
    feature_columns: list[str],
) -> pd.Series:
    if splsda_bands is None or splsda_bands.empty:
        return pd.Series(False, index=candidates.index, dtype=bool)

    required = {"feature_name", "selected"}
    if not required.issubset(splsda_bands.columns):
        logger.warning("PLS-L1DA bands table missing expected columns; skipping selection annotation")
        return pd.Series(False, index=candidates.index, dtype=bool)

    selected_features = set(
        splsda_bands.loc[splsda_bands["selected"].fillna(False), "feature_name"].astype(str).unique()
    )
    out = []
    for _, region in candidates.iterrows():
        region_features = set(_region_feature_names(feature_columns, region))
        out.append(bool(region_features & selected_features))
    return pd.Series(out, index=candidates.index, dtype=bool)


def _load_biological_pca_loading_regions() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for model_name in _BIOLOGICAL_PCA_MODELS:
        path = get_biological_pca_canonical_dir(model_name) / "loading_regions.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _annotate_biological_pca_overlap(
    candidates: pd.DataFrame,
    loading_regions: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    if loading_regions.empty or "start_cm-1" not in loading_regions.columns:
        return (
            pd.Series(False, index=candidates.index, dtype=bool),
            pd.Series(pd.NA, index=candidates.index, dtype="string"),
        )

    out_overlap = []
    out_pcs = []
    for _, region in candidates.iterrows():
        block = str(region["spectral_block"])
        start = float(region["start_cm-1"])
        end = float(region["end_cm-1"])
        overlaps = loading_regions.loc[
            loading_regions["spectral_block"].astype(str).eq(block)
            & (loading_regions["start_cm-1"].astype(float) <= end)
            & (loading_regions["end_cm-1"].astype(float) >= start)
        ].copy()
        out_overlap.append(not overlaps.empty)
        if overlaps.empty:
            out_pcs.append(pd.NA)
        else:
            pcs = sorted(overlaps["PC"].dropna().astype(str).unique())
            out_pcs.append("; ".join(pcs))
    return (
        pd.Series(out_overlap, index=candidates.index, dtype=bool),
        pd.Series(out_pcs, index=candidates.index, dtype="string"),
    )


def _band_windows(cfg: dict[str, Any]) -> dict[str, tuple[float, float]]:
    bands_cfg = cfg.get("whole_spectra", {}).get("bands", {})
    out: dict[str, tuple[float, float]] = {}
    for band_name, limits in bands_cfg.items():
        if isinstance(limits, (list, tuple)) and len(limits) == 2:
            out[str(band_name)] = (float(limits[0]), float(limits[1]))
    return out


def _band_class_map(cfg: dict[str, Any]) -> dict[str, str]:
    classes_cfg = cfg.get("whole_spectra", {}).get("biochemical_classes", {})
    out: dict[str, str] = {}
    for class_name, band_names in classes_cfg.items():
        for band_name in band_names:
            out[str(band_name)] = str(class_name)
    return out


def _annotate_biochemical_class(
    candidates: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.Series:
    windows = _band_windows(cfg)
    class_map = _band_class_map(cfg)
    out: list[str] = []
    for _, region in candidates.iterrows():
        block = str(region["spectral_block"])
        start = float(region["start_cm-1"])
        end = float(region["end_cm-1"])
        classes: set[str] = set()
        for band_name, (lo, hi) in windows.items():
            band_block = "1300" if hi < 2000 else "2850"
            if band_block != block:
                continue
            if start <= hi and end >= lo:
                classes.add(class_map.get(band_name, "mixed"))
        if not classes:
            out.append("unassigned")
        elif len(classes) == 1:
            out.append(next(iter(classes)))
        else:
            out.append("mixed")
    return pd.Series(out, index=candidates.index, dtype="string")


def _source_analysis_list(candidates: pd.DataFrame) -> pd.Series:
    out: list[str] = []
    for _, region in candidates.iterrows():
        source = str(region.get("source_analysis", ""))
        analysis = str(region.get("analysis_name", ""))
        if source and analysis and source != analysis:
            out.append(f"{source}::{analysis}")
        elif source:
            out.append(source)
        elif analysis:
            out.append(analysis)
        else:
            out.append("unknown")
    return pd.Series(out, index=candidates.index, dtype="string")


def _count_supporting_sources(candidates: pd.DataFrame) -> pd.Series:
    return (
        candidates.groupby("candidate_region_id")["source_analysis"]
        .transform(lambda s: s.astype(str).nunique())
        .astype(int)
    )


def _assign_evidence_level(row: pd.Series) -> str:
    cluster = bool(row.get("cluster_permutation_significant", False))
    covariate = bool(row.get("covariate_persisted", False))
    pls = bool(row.get("pls_vip_coverage", False))
    splsda = bool(row.get("splsda_selected", False))
    fdr = bool(row.get("fdr_significant", False))
    n_sources = int(row.get("n_supporting_sources", 1))

    if cluster and covariate and (pls or splsda):
        return "High"
    if cluster and covariate:
        return "High"
    if fdr and n_sources >= 2:
        return "Medium"
    if cluster or (pls or splsda):
        return "Low"
    return "Low"


def _interpretation_note(row: pd.Series) -> str:
    level = str(row.get("evidence_level", ""))
    parts: list[str] = []
    if bool(row.get("cluster_permutation_significant", False)):
        parts.append("cluster-permutation significant")
    if bool(row.get("covariate_persisted", False)):
        parts.append("covariate-persisted")
    if bool(row.get("pls_vip_coverage", False)):
        parts.append("PLS VIP coverage")
    if bool(row.get("splsda_selected", False)):
        parts.append("PLS-L1DA selected")
    if bool(row.get("biological_pca_overlap", False)):
        parts.append("biological PCA loading overlap")

    if not parts:
        parts.append("exploratory candidate region")

    note = f"{level} evidence: " + ", ".join(parts) + "."
    note += " Interpreted as an exploratory spectral association; not diagnostic, predictive, causal, or validated as a biomarker."
    return note


def _empty_table() -> pd.DataFrame:
    return pd.DataFrame(columns=_DEFAULT_COLUMNS)


def run_integrated_candidates(
    *,
    candidate_regions_path: str | Path | None = None,
    covariate_models_path: str | Path | None = None,
    pls_vip_path: str | Path | None = None,
    splsda_bands_path: str | Path | None = None,
    feature_matrix_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Build the integrated candidate-region evidence table."""
    cfg = _cfg(cfg)
    integrated_cfg = _integrated_cfg(cfg)
    if not bool(integrated_cfg.get("enabled", True)):
        return {}

    target_dir = (
        Path(output_dir)
        if output_dir is not None
        else get_cross_region_phase_dir("integrated_candidates")
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    # Resolve default input paths when not provided explicitly.
    if candidate_regions_path is None:
        candidate_regions_path = get_cross_region_phase_dir("candidate_regions") / "candidate_regions_cluster_corrected.csv"
    if covariate_models_path is None:
        covariate_models_path = get_cross_region_phase_dir("covariate_models") / "covariate_adjusted_region_models.csv"
    if pls_vip_path is None:
        pls_vip_path = get_cross_region_phase_dir("pls_regression") / "pls_vip_bands.csv"
    if splsda_bands_path is None:
        splsda_bands_path = get_cross_region_phase_dir("splsda") / "pls_l1da_selected_bands.csv"

    candidates = _load_csv_if_exists(candidate_regions_path)
    if candidates is None or candidates.empty:
        logger.warning("No candidate regions available; writing empty integrated table")
        empty = _empty_table()
        out_path = target_dir / "integrated_candidate_regions.csv"
        empty.to_csv(out_path, index=False)
        return {"integrated_candidate_regions": out_path}

    covariate_models = _load_csv_if_exists(covariate_models_path)
    pls_vip = _load_csv_if_exists(pls_vip_path)
    splsda_bands = _load_csv_if_exists(splsda_bands_path)

    feature_columns: list[str] = []
    if feature_matrix_path is not None and Path(feature_matrix_path).exists():
        frame = pd.read_parquet(feature_matrix_path)
        feature_columns = [c for c in frame.columns if c.startswith(("1300__", "2850__"))]
    else:
        feature_columns = []

    vip_threshold = float(integrated_cfg.get("vip_threshold", 1.0))

    candidates = candidates.copy()
    candidates["source_analyses"] = _source_analysis_list(candidates)
    candidates["n_supporting_sources"] = _count_supporting_sources(candidates)
    candidates["fdr_significant"] = (
        candidates.get("n_fdr_significant", pd.Series(0, index=candidates.index)).fillna(0).astype(int) > 0
    )
    candidates["cluster_permutation_significant"] = (
        candidates.get("n_cluster_significant", pd.Series(0, index=candidates.index)).fillna(0).astype(int) > 0
    )
    candidates["covariate_persisted"] = _annotate_covariate_persistence(candidates, covariate_models)
    candidates["pls_vip_coverage"] = _annotate_pls_vip_coverage(candidates, pls_vip, feature_columns, vip_threshold)
    candidates["splsda_selected"] = _annotate_splsda_selection(candidates, splsda_bands, feature_columns)

    loading_regions = _load_biological_pca_loading_regions()
    candidates["biological_pca_overlap"], candidates["overlapping_biological_pcs"] = (
        _annotate_biological_pca_overlap(candidates, loading_regions)
    )
    candidates["biochemical_class"] = _annotate_biochemical_class(candidates, cfg)

    candidates["evidence_level"] = candidates.apply(_assign_evidence_level, axis=1)
    candidates["interpretation_note"] = candidates.apply(_interpretation_note, axis=1)

    # Keep one row per candidate region with the strongest supporting source.
    candidates = candidates.sort_values(
        ["candidate_region_id", "evidence_level", "integrated_abs_effect"],
        ascending=[True, False, False],
    ).drop_duplicates(subset=["candidate_region_id"], keep="first")

    ordered = [c for c in _DEFAULT_COLUMNS if c in candidates.columns]
    candidates = candidates.loc[:, ordered].reset_index(drop=True)

    out_path = target_dir / "integrated_candidate_regions.csv"
    candidates.to_csv(out_path, index=False)
    logger.info("Integrated candidate table written to %s", out_path)
    return {"integrated_candidate_regions": out_path}
