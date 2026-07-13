import logging
import os
from pathlib import Path

import numpy as np
import yaml

if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid
if not hasattr(np, "in1d") and hasattr(np, "isin"):
    np.in1d = np.isin


__version__ = "0.1.0"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

WHOLE_SPECTRA_PHASE_DIRS = {
    "global_structure": "01_global_structure",
    "global_drivers": "02_global_drivers",
    "variance_context": "03_variance_context",
}

CROSS_REGION_PHASE_DIRS = {
    "subject_features": "01_subject_features",
    "biological_pca_summary": "02_biological_pca_summary",
    "clinical_correlations": "03_clinical_correlations",
    "longitudinal_moca": "04_longitudinal_moca",
    "disease_stratification": "05_disease_stratification",
    "candidate_regions": "06_candidate_regions",
    "covariate_models": "07_covariate_models",
    "pls_regression": "08_pls_regression",
    "splsda": "09_splsda_exploratory_classifier",
    "biochemical_ratios": "10_biochemical_ratios",
    "integrated_candidates": "11_integrated_candidates",
    "disease_subgroups": "12_disease_subgroups",
    "minimal_panel": "13_minimal_panel",
    "publication_report": "14_publication_report",
}

# Ordered sequence of single-range EDA phases.  Numbers are derived from
# position (1-indexed, zero-padded to 2 digits) so inserting or reordering
# here automatically renumbers all downstream folders.
SINGLE_RANGE_PHASE_SEQUENCE: list[tuple[str, str]] = [
    ("group_comp",     "group_comparisons"),
    ("pca",            "area_level_pca_qc"),
    ("band",           "known_band_context"),
    ("variance",       "variance_context"),
    ("intra_disease",  "intra_disease"),
    ("biological_pca", "biological_pca"),
]


def get_single_range_phase_dirs() -> dict[str, str]:
    """Return {key: 'NN_suffix'} for every single-range EDA phase."""
    return {
        key: f"{i + 1:02d}_{suffix}"
        for i, (key, suffix) in enumerate(SINGLE_RANGE_PHASE_SEQUENCE)
    }


def _bio_pca_range_dir(range_label: str) -> Path:
    dirs = get_single_range_phase_dirs()
    return Path(range_label) / dirs["biological_pca"]


BIOLOGICAL_PCA_CANONICAL_DIRS = {
    "PCA_1330": _bio_pca_range_dir("01_fingerprint_1300"),
    "PCA_2850": _bio_pca_range_dir("02_ch_stretch_2850"),
    "PCA_whole": Path("03_whole_spectrum_1300_plus_2850") / "04_biological_pca",
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

    try:
        from rich.logging import RichHandler

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
    except Exception:
        log_fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        handler = logging.StreamHandler()
        handler.setLevel(console_level)
        handler.setFormatter(logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S"))
        root_logger.addHandler(handler)

    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def get_output_root(track: str) -> Path:
    """Return the default output root for a pipeline track."""
    out_cfg = config.get("output", {})
    if track == "eda":
        return Path(out_cfg.get("eda_root", out_cfg.get("project_name", "outputs")))
    raise ValueError(f"Unknown output track: {track}")


def _output_dir(subdir_key: str) -> Path:
    """
    Build an output path from config.

    EDA-related keys resolve relative to ``output.eda_root``. Other callers
    fall back to the legacy shared root when a dedicated track root is not used.
    """
    out_cfg = config.get("output", {})
    eda_keys = {"eda_subdir"}
    base_dir = get_output_root("eda") if subdir_key in eda_keys else Path(out_cfg.get("project_name", "outputs"))
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


def get_whole_spectra_phase_dir(phase: str) -> Path:
    """Return an output directory for a whole-spectra EDA phase."""
    return get_eda_phase_dir("both", WHOLE_SPECTRA_PHASE_DIRS.get(phase, phase))


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
    suffix = CROSS_REGION_PHASE_DIRS.get(phase, phase)
    path = get_cross_region_summary_dir() / suffix
    path.mkdir(parents=True, exist_ok=True)
    return path


_SPECTRAL_INTERPRETABILITY_SUBDIRS = {
    "peak_shape": "peak_shape",
    "nmf": "nmf_decomposition",
    "entropy": "spectral_entropy",
    "discriminant": "discriminant_signatures",
    "coupling": "feature_coupling",
    "index_clustering": "index_clustering",
    "consistency": "endpoint_consistency",
    "figures": "figures",
}


def get_confirmatory_panel_dir() -> Path:
    """Compatibility output root for fixed-panel exploratory results."""
    path = _output_dir("eda_subdir") / "05_confirmatory_panel"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_spectral_interpretability_dir(phase: str) -> Path:
    """Return a phase subfolder under the Tier-2 spectral interpretability root."""
    suffix = _SPECTRAL_INTERPRETABILITY_SUBDIRS.get(phase, phase)
    path = _output_dir("eda_subdir") / "06_spectral_interpretability" / suffix
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_ml_classification_dir() -> Path:
    """Return the control-vs-disease ML classification output root (disease_vs_control mode)."""
    path = _output_dir("eda_subdir") / "07_ml_classification"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_biological_pca_canonical_dir(model_name: str) -> Path:
    """Return the canonical range-local directory for a Biological PCA model."""
    relative = BIOLOGICAL_PCA_CANONICAL_DIRS.get(str(model_name))
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
