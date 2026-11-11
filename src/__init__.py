import logging
import os
from pathlib import Path

import yaml
from rich.logging import RichHandler

PROJECT_ROOT = Path(__file__).resolve().parent.parent

WHOLE_SPECTRA_PHASE_DIRS = {
    "global_structure": "01_global_structure",
    "global_drivers": "02_global_drivers",
    "variance_context": "03_variance_context",
    "exploratory_multiblock": "exploratory_multiblock",
}

CROSS_REGION_PHASE_DIRS = {
    # Flat execution-order sequence. Predictive/classifier phases (PLS
    # regression, sPLS-DA, minimal panel, disease subgroups) live under
    # CLASSIFICATION_PHASE_DIRS / get_classification_phase_dir() instead —
    # this tree holds descriptive, association, and confound-check phases.
    "subject_features": "01_subject_features",
    "biological_pca_summary": "02_biological_pca_summary",
    "clinical_correlations": "03_clinical_correlations",
    "longitudinal_moca": "04_longitudinal_moca",
    "disease_stratification": "05_disease_stratification",
    "candidate_regions": "06_candidate_regions",
    "covariate_models": "07_covariate_models",
    "biochemical_ratios": "08_biochemical_ratios",
    "integrated_candidates": "09_integrated_candidates",
    "publication_report": "10_publication_report",
}

CLASSIFICATION_PHASE_DIRS = {
    "pls_regression": "pls_regression",
    "splsda": "splsda_exploratory_classifier",
    "minimal_panel": "minimal_panel",
    "disease_subgroups": "disease_subgroups",
}

# Shared descriptive single-range EDA phases. Mode-specific phases are appended
# below without changing the established disease-only output contract.
SINGLE_RANGE_BASE_PHASE_SEQUENCE: list[tuple[str, str]] = [
    ("group_comp",     "group_comparisons"),
    ("pca",            "area_level_pca_qc"),
    ("band",           "known_band_context"),
    ("variance",       "variance_context"),
]


def get_single_range_phase_dirs(analysis_mode: str = "disease_only") -> dict[str, str]:
    """Return numbered single-range phase directories for one analysis mode."""
    sequence = list(SINGLE_RANGE_BASE_PHASE_SEQUENCE)
    if analysis_mode == "disease_only":
        sequence.extend(
            [
                ("intra_disease", "intra_disease"),
                ("biological_pca", "biological_pca"),
                ("robustness", "region_robustness"),
            ]
        )
    else:
        sequence.extend(
            [
                ("robustness", "region_robustness"),
                ("biological_pca", "biological_pca"),
            ]
        )
    return {
        key: f"{i + 1:02d}_{suffix}"
        for i, (key, suffix) in enumerate(sequence)
    }


def _bio_pca_range_dir(range_label: str, analysis_mode: str) -> Path:
    dirs = get_single_range_phase_dirs(analysis_mode)
    return Path(range_label) / dirs["biological_pca"]


CONTROL_VS_DISEASE_ML_PHASE_DIRS = {
    "cohort": "01_cohort",
    "model_evaluation": "02_model_evaluation",
    "confounder_checks": "03_confounder_checks",
    "permutation_tests": "04_permutation_tests",
    "band_screen": "05_univariate_band_screen",
    "report": "06_report",
}


def resolve_project_path(path_like: str | Path | None) -> Path | None:
    """Resolve relative paths against the repository root."""
    if path_like is None:
        return None

    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep-merge *override* into *base*, returning a new dict.

    Nested dicts are merged recursively; scalars and lists in *override* win.
    """
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _load_yaml_with_extends(path: Path) -> dict:
    """Load a YAML config file, resolving ``_extends`` via deep-merge.

    If the YAML contains ``_extends: <filename>`` that file is loaded first
    (resolved relative to the current file's directory) and the current file's
    settings are deep-merged on top.  Supports transitive inheritance.
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(
            f"Config file '{path}' is empty or not a valid YAML mapping "
            f"(got {type(cfg).__name__})."
        )

    base_key = cfg.pop("_extends", None)
    if base_key is not None:
        base_path = path.parent / base_key
        if not base_path.exists():
            raise FileNotFoundError(
                f"Config '_extends' base file not found: {base_path} "
                f"(referenced from {path})"
            )
        base_cfg = _load_yaml_with_extends(base_path)
        cfg = _deep_merge(base_cfg, cfg)

    return cfg


def load_config(config_path: str = "config/settings.yaml") -> dict:
    """Loads the main project configuration from a YAML file.

    Resolution order:
    1. ``RAMAN_CONFIG`` environment variable (absolute or relative path).
    2. ``config_path`` argument, searched in CWD then one level up.

    Track-specific config files may use ``_extends: settings.yaml`` to inherit
    all defaults and override only the sections relevant to their track.
    """
    env_override = os.environ.get("RAMAN_CONFIG")
    if env_override:
        path = Path(env_override)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            raise FileNotFoundError(
                f"RAMAN_CONFIG env var points to missing file: {env_override}"
            )
    else:
        path = Path(config_path)
        if not path.exists():
            # Traverse up if run from a subdirectory like notebooks/
            path = Path("..") / path
            if not path.exists():
                raise FileNotFoundError(f"Config not found at {config_path}")

    return _load_yaml_with_extends(path)


config = load_config()


def resolve_cfg(cfg: dict | None = None) -> dict:
    """Return `cfg` if given, else the module-level default `config`.

    Shared by analysis/reporting/data_preparation modules that accept an
    optional config override on every public function.
    """
    return config if cfg is None else cfg


def setup_logging(
    level: int = logging.INFO,
    *,
    console_level: int | None = None,
) -> None:
    """Configure root logger for the pipeline (call once at entry point)."""
    console_level = level if console_level is None else console_level
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)

    handler = RichHandler(
        rich_tracebacks=True,
        show_time=True,
        show_level=True,
        show_path=False,
        markup=False,
    )
    handler.setLevel(console_level)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(handler)

    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def get_output_root(track: str) -> Path:
    """Return the default output root for a pipeline track."""
    out_cfg = config.get("output", {})
    if track == "eda":
        return Path(out_cfg.get("eda_root", "outputs"))
    raise ValueError(f"Unknown output track: {track}")


def _output_dir(subdir_key: str) -> Path:
    """Build an output path relative to ``output.eda_root`` from config."""
    out_cfg = config.get("output", {})
    base_dir = get_output_root("eda")
    subdir = out_cfg.get(subdir_key, "")
    if subdir in {"", ".", None}:
        return base_dir
    return base_dir / str(subdir)


def _range_label(spectral_range: str) -> str:
    """Return the human-readable range label from config."""
    range_labels = config["output"].get("range_labels", {})
    return range_labels.get(str(spectral_range), f"range_{spectral_range}")


def get_eda_phase_dir(spectral_range: str, phase: str) -> Path:
    """
    Return a single output directory for a specific EDA phase.

    Figures and tables for that phase live together, making it easy to
    investigate one analysis step at a time.

    Example: get_eda_phase_dir("1300", "03_pca")
             → <eda_root>/fingerprint_1300/03_pca/
    """
    eda_base = _output_dir("eda_subdir")
    label = _range_label(spectral_range)
    phase_dir = eda_base / label / phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    return phase_dir


def _phase_suffix(phase: str, phase_dirs: dict[str, str]) -> str:
    try:
        return phase_dirs[phase]
    except KeyError as exc:
        raise ValueError(f"Unknown analysis phase: {phase}") from exc


def get_whole_spectra_phase_dir(phase: str) -> Path:
    """Return an output directory for a whole-spectra EDA phase."""
    return get_eda_phase_dir("both", _phase_suffix(phase, WHOLE_SPECTRA_PHASE_DIRS))


def get_run_metadata_dir() -> Path:
    """Return the metadata directory for the active EDA run."""
    path = _output_dir("eda_subdir") / "00_run_metadata"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_cross_region_summary_dir() -> Path:
    """Return the cross-region summary directory for the active EDA run."""
    path = _output_dir("eda_subdir") / "04_cross_region_summary"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_cross_region_phase_dir(phase: str) -> Path:
    """Return a numbered phase directory under the cross-region summary root."""
    suffix = _phase_suffix(phase, CROSS_REGION_PHASE_DIRS)
    path = get_cross_region_summary_dir() / suffix
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_classification_dir() -> Path:
    """Return the top-level classification/predictive-modeling root for disease-only mode."""
    path = _output_dir("eda_subdir") / "classification"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_classification_phase_dir(phase: str) -> Path:
    """Return a phase directory under the classification root (PLS regression, sPLS-DA, minimal panel, disease subgroups)."""
    suffix = _phase_suffix(phase, CLASSIFICATION_PHASE_DIRS)
    path = get_classification_dir() / suffix
    path.mkdir(parents=True, exist_ok=True)
    return path


_SPECTRAL_INTERPRETABILITY_SUBDIRS = {
    "peak_shape": "01_peak_shape",
    "nmf": "02_nmf_decomposition",
    "entropy": "03_spectral_entropy",
    "discriminant": "04_discriminant_signatures",
    "coupling": "05_feature_coupling",
    "consistency": "06_endpoint_consistency",
    "figures": "08_figures",
}


def get_confirmatory_panel_dir() -> Path:
    """Return the output root for fixed-panel exploratory results."""
    path = _output_dir("eda_subdir") / "05_confirmatory_panel"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_spectral_interpretability_dir(phase: str) -> Path:
    """Return a phase subfolder under the Tier-2 spectral interpretability root."""
    suffix = _phase_suffix(phase, _SPECTRAL_INTERPRETABILITY_SUBDIRS)
    path = _output_dir("eda_subdir") / "06_spectral_interpretability" / suffix
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_ml_classification_dir() -> Path:
    """Return the control-vs-disease ML classification output root (disease_vs_control mode)."""
    path = _output_dir("eda_subdir") / "05_ml_classification"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_ml_classification_phase_dir(
    phase: str,
    *,
    root: str | Path | None = None,
    create: bool = True,
) -> Path:
    """Return one numbered control-vs-disease ML artifact directory."""
    suffix = _phase_suffix(phase, CONTROL_VS_DISEASE_ML_PHASE_DIRS)
    base = Path(root) if root is not None else get_ml_classification_dir()
    path = base / suffix
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def get_biological_pca_canonical_dir(model_name: str) -> Path:
    """Return the canonical range-local directory for a Biological PCA model."""
    analysis_mode = str(config.get("analysis", {}).get("mode", "disease_only"))
    relative = {
        "PCA_1330": _bio_pca_range_dir("01_fingerprint_1300", analysis_mode),
        "PCA_2850": _bio_pca_range_dir("02_ch_stretch_2850", analysis_mode),
        "PCA_whole": Path("03_whole_spectrum_1300_plus_2850") / "04_biological_pca",
    }.get(str(model_name))
    if relative is None:
        raise ValueError(f"Unknown Biological PCA model: {model_name}")
    path = _output_dir("eda_subdir") / relative
    path.mkdir(parents=True, exist_ok=True)
    return path


def whole_spectra_split_index(wavenumbers) -> int | None:
    """Return the concatenated fingerprint/CH-stretch split index, if detectable."""
    try:
        from src.analysis.statistics import infer_concatenated_range_split_index

        return infer_concatenated_range_split_index(wavenumbers)
    except ValueError:
        return None
