from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

from src import config
from src.constants import CANONICAL_FP_CHANNELS, CANONICAL_REGION_BOUNDS


_METADATA_COLUMNS = [
    "feature_id",
    "legacy_feature_id",
    "display_name",
    "short_label",
    "track",
    "region",
    "wavenumber_cm1",
    "window_cm1",
    "biological_assignment",
    "sex_interaction",
    "fold_local",
    "interpretation_note",
]

_ID_LIKE_COLUMNS = {
    "subject_id",
    "subject_id_raw",
    "area_id",
    "slide_number",
}

_BAND_ASSIGNMENTS: dict[str, tuple[str, str, str]] = {
    "phenylalanine_1004": ("Phenylalanine 1004", "fingerprint", "Phenylalanine ring-breathing; protein reference"),
    "dna_po2_sym_1090": ("DNA/PO2 1090", "fingerprint", "PO2 symmetric stretch; nucleic acids/phosphodiesters"),
    "c_c_skeletal_1127": ("C-C skeletal 1127", "fingerprint", "Mixed lipid/protein/carbohydrate skeletal modes"),
    "amide_iii_1260": ("Amide III 1260", "fingerprint", "Protein backbone conformation"),
    "ch2_bend_1300": ("CH2 bend 1300", "fingerprint", "Lipid-chain CH2 twisting/bending"),
    "ch2_scissor_1450": ("CH2 scissor 1450", "fingerprint", "CH2 bending; lipids and membranes"),
    "carboxylate_1580": ("Carboxylate 1580", "fingerprint", "Carboxylate/aromatic mixed contribution"),
    "amide_i_1655": ("Amide I 1655", "fingerprint", "Protein secondary structure"),
    "ch2_sym_stretch_2850": ("CH2 symmetric stretch 2850", "ch_stretch", "Lipid acyl-chain CH2 symmetric stretch"),
    "ch2_asym_stretch_2885": ("CH2 asymmetric stretch 2885", "ch_stretch", "Lipid order/chain packing"),
    "ch3_sym_stretch_2930": ("CH3 symmetric stretch 2930", "ch_stretch", "Mixed protein/lipid CH3 signal"),
    "ch3_asym_stretch_2960": ("CH3 asymmetric stretch 2960", "ch_stretch", "Mixed CH3 asymmetric stretch"),
    "oh_nh_stretch_3060": ("OH/NH stretch 3060", "ch_stretch", "OH/NH/aromatic or unsaturated contribution"),
}

_STAT_LABELS = {
    "mean": "mean intensity",
    "std": "standard deviation",
    "kurtosis": "kurtosis",
    "skewness": "skewness",
    "iqr": "interquartile range",
    "centroid": "spectral centroid",
    "spread": "spectral spread",
    "entropy": "spectral entropy",
    "q1_energy": "Q1 energy",
    "q2_energy": "Q2 energy",
    "q3_energy": "Q3 energy",
    "q4_energy": "Q4 energy",
    "q1_q4_ratio": "Q1/Q4 energy ratio",
    "derivative_std": "derivative variability",
    "zcr": "zero-crossing rate",
}


def build_feature_metadata(
    feature_ids: Iterable[object],
    split_index: int | None = None,
) -> pd.DataFrame:
    """Build a presentation metadata table for model feature IDs."""
    rows = [_metadata_for_feature(str(feature), split_index=split_index) for feature in feature_ids]
    out = pd.DataFrame(rows, columns=_METADATA_COLUMNS)
    if out.empty:
        return pd.DataFrame(columns=_METADATA_COLUMNS)
    return out.drop_duplicates(subset=["feature_id"], keep="first").reset_index(drop=True)


def enrich_feature_table(
    df: pd.DataFrame,
    feature_col: str = "feature",
    metadata: pd.DataFrame | None = None,
    split_index: int | None = None,
) -> pd.DataFrame:
    """Attach feature presentation metadata to a table containing feature IDs."""
    if df.empty or feature_col not in df.columns:
        return df

    out = df.copy()
    feature_ids = out[feature_col].astype(str).dropna().unique().tolist()
    meta = build_feature_metadata(feature_ids, split_index=split_index) if metadata is None else metadata.copy()

    rename = {"feature_id": feature_col}
    meta = meta.rename(columns=rename)
    extra_cols = [c for c in meta.columns if c != feature_col]
    out = out.drop(columns=[c for c in extra_cols if c in out.columns], errors="ignore")
    return out.merge(meta, on=feature_col, how="left")


def assert_no_identifier_leakage(
    feature_names: Iterable[object],
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> None:
    """Fail fast if identifier-like columns enter the classifier design matrix."""
    names = [str(name) for name in feature_names]
    leaked = [name for name in names if name in _ID_LIKE_COLUMNS]
    if leaked:
        raise ValueError(f"Identifier-like feature column(s) leaked into model matrix: {leaked}")

    for label, arr in (("x_train", x_train), ("x_test", x_test)):
        matrix = np.asarray(arr)
        if matrix.ndim != 2:
            raise ValueError(f"{label} must be a 2D numeric matrix")
        if not np.issubdtype(matrix.dtype, np.number):
            raise ValueError(f"{label} must be numeric; got dtype {matrix.dtype}")
        if matrix.shape[1] != len(names):
            raise ValueError(
                f"{label} width ({matrix.shape[1]}) does not match feature_names ({len(names)})"
            )


def feature_short_label(feature_id: object, split_index: int | None = None) -> str:
    return str(_metadata_for_feature(str(feature_id), split_index=split_index)["short_label"])


def _metadata_for_feature(feature_id: str, split_index: int | None = None) -> dict[str, object]:
    sex_interaction = feature_id.endswith("_x_sex")
    base_id = feature_id[:-6] if sex_interaction else feature_id
    row = _base_metadata(base_id, split_index=split_index)
    if sex_interaction:
        row["feature_id"] = feature_id
        row["legacy_feature_id"] = base_id
        row["display_name"] = f"{row['display_name']} x sex interaction"
        row["short_label"] = f"{row['short_label']} x sex"
        row["sex_interaction"] = True
        row["interpretation_note"] = (
            f"Sex interaction term: the base feature contribution is modulated by the binary sex covariate. "
            f"{row['interpretation_note']}"
        )
    return row


def _base_metadata(feature_id: str, split_index: int | None = None) -> dict[str, object]:
    row = _empty_row(feature_id)

    if feature_id.startswith("pca_"):
        return _pca_metadata(feature_id, row)
    if feature_id.startswith("peak_"):
        return _peak_metadata(feature_id, row, split_index=split_index)
    if feature_id.startswith("stat_"):
        return _stat_metadata(feature_id, row)
    if feature_id.startswith("hetero_"):
        return _hetero_metadata(feature_id, row, split_index=split_index)
    if feature_id.startswith("clin_"):
        return _clinical_metadata(feature_id, row)
    if feature_id.startswith("band_band_"):
        return _band_metadata(feature_id, row)
    if feature_id.startswith("ratio_ratio_"):
        return _ratio_metadata(feature_id, row)
    if feature_id.startswith("chlipid_"):
        return _ch_state_metadata(feature_id, row)
    if feature_id == "cov_sex_binary":
        row.update(
            {
                "display_name": "Sex covariate (male=1)",
                "short_label": "Sex",
                "track": "covariate",
                "region": "clinical",
                "biological_assignment": "Subject sex covariate",
                "interpretation_note": "Binary sex covariate: female=0, male=1.",
            }
        )
        return row

    row["display_name"] = feature_id.replace("_", " ")
    row["short_label"] = row["display_name"]
    return row


def _empty_row(feature_id: str) -> dict[str, object]:
    return {
        "feature_id": feature_id,
        "legacy_feature_id": "",
        "display_name": feature_id.replace("_", " "),
        "short_label": feature_id.replace("_", " "),
        "track": "other",
        "region": "",
        "wavenumber_cm1": np.nan,
        "window_cm1": "",
        "biological_assignment": "",
        "sex_interaction": False,
        "fold_local": False,
        "interpretation_note": "",
    }


def _pca_metadata(feature_id: str, row: dict[str, object]) -> dict[str, object]:
    suffix = feature_id.removeprefix("pca_")
    match_named = re.match(r"(fingerprint|fp|ch_stretch|ch|region_1300|region_2850)_pc(\d+)$", suffix)
    match_old = re.match(r"(\d+)$", suffix)
    if match_named:
        region_token, pc = match_named.groups()
        region = _normalize_region(region_token)
        pc_i = int(pc)
    elif match_old:
        region = "multi_region"
        pc_i = int(match_old.group(1)) + 1
    else:
        region = "multi_region"
        pc_i = 0

    prefix = _region_label(region)
    row.update(
        {
            "display_name": f"{prefix} PCA component {pc_i}" if pc_i else f"{prefix} PCA component",
            "short_label": f"{prefix} PCA {pc_i}" if pc_i else f"{prefix} PCA",
            "track": "pca",
            "region": region,
            "biological_assignment": "Multivariate spectral variance pattern",
            "interpretation_note": "PCA score from training-fold spectral decomposition; inspect loadings before assigning a single molecule.",
        }
    )
    return row


def _peak_metadata(feature_id: str, row: dict[str, object], split_index: int | None) -> dict[str, object]:
    body = feature_id.removeprefix("peak_")
    if body.startswith("ratio_"):
        return _peak_ratio_metadata(body, row, split_index=split_index)

    match = re.match(r"peak_(\d+)_(intensity|fwhm|derivative)$", body)
    if not match:
        row.update({"track": "peak", "display_name": body.replace("_", " "), "short_label": body.replace("_", " ")})
        return row

    channel = int(match.group(1))
    metric = match.group(2)
    wn = _channel_to_wavenumber(channel, split_index=split_index)
    region = _region_from_wavenumber(wn)
    assignment = _assignment_for_wavenumber(wn)
    metric_label = {"intensity": "mean intensity", "fwhm": "FWHM", "derivative": "local derivative"}[metric]
    row.update(
        {
            "display_name": f"Peak {metric_label} near {_format_wn(wn, channel)}",
            "short_label": f"Peak {_format_wn(wn, channel)} {metric.upper() if metric == 'fwhm' else metric}",
            "track": "peak",
            "region": region,
            "wavenumber_cm1": wn,
            "biological_assignment": assignment,
            "interpretation_note": "Data-driven peak feature detected inside the training fold; assignment is nearest-window guidance.",
        }
    )
    return row


def _peak_ratio_metadata(body: str, row: dict[str, object], split_index: int | None) -> dict[str, object]:
    match = re.match(r"ratio_peak_(\d+)_peak_(\d+)$", body)
    if not match:
        row.update({"track": "peak", "display_name": body.replace("_", " "), "short_label": body.replace("_", " ")})
        return row
    left = int(match.group(1))
    right = int(match.group(2))
    left_wn = _channel_to_wavenumber(left, split_index=split_index)
    right_wn = _channel_to_wavenumber(right, split_index=split_index)
    row.update(
        {
            "display_name": f"Peak ratio {_format_wn(left_wn, left)} / {_format_wn(right_wn, right)}",
            "short_label": f"Peak {_format_wn(left_wn, left)} / {_format_wn(right_wn, right)}",
            "track": "peak",
            "region": _combine_regions(_region_from_wavenumber(left_wn), _region_from_wavenumber(right_wn)),
            "wavenumber_cm1": left_wn,
            "biological_assignment": "Ratio of data-driven peak intensities",
            "interpretation_note": "Ratio feature reduces broad intensity scaling but remains tied to training-fold peak detection.",
        }
    )
    return row


def _stat_metadata(feature_id: str, row: dict[str, object]) -> dict[str, object]:
    body = feature_id.removeprefix("stat_")
    region_token, metric = _split_region_metric(body)
    region = _normalize_region(region_token)
    label = _STAT_LABELS.get(metric, metric.replace("_", " "))
    row.update(
        {
            "display_name": f"{_region_label(region)} {label}",
            "short_label": f"{_region_label(region)} {label}",
            "track": "stat",
            "region": region,
            "biological_assignment": "Global spectral shape descriptor",
            "interpretation_note": "Statistical descriptor of the subject mean spectrum, not a single Raman band.",
        }
    )
    return row


def _hetero_metadata(feature_id: str, row: dict[str, object], split_index: int | None) -> dict[str, object]:
    body = feature_id.removeprefix("hetero_")
    row.update(
        {
            "track": "hetero",
            "region": "multi_region",
            "biological_assignment": "Within-subject spatial/area heterogeneity",
            "interpretation_note": "Heterogeneity feature computed from area-level summaries.",
        }
    )
    if body.startswith("targeted_peak_std_"):
        idx = body.removeprefix("targeted_peak_std_")
        row["display_name"] = f"Targeted peak {idx} area-to-area variability"
        row["short_label"] = f"Peak {idx} variability"
        return row
    row["display_name"] = body.replace("_", " ")
    row["short_label"] = row["display_name"]
    if "wavenumber_max_inter_area_variance_index" in body:
        row["interpretation_note"] = "Channel index of maximum inter-area variance; convert to Raman shift before biological interpretation."
    return row


def _clinical_metadata(feature_id: str, row: dict[str, object]) -> dict[str, object]:
    body = feature_id.removeprefix("clin_")
    labels = {
        "age": "Age",
        "sex_binary": "Sex (male=1)",
        "n_slides": "Number of slides",
        "n_areas_surviving": "Surviving Raman areas",
    }
    row.update(
        {
            "display_name": labels.get(body, body.replace("_", " ")),
            "short_label": labels.get(body, body.replace("_", " ")),
            "track": "clin",
            "region": "clinical",
            "biological_assignment": "Clinical or acquisition covariate",
            "interpretation_note": "Covariate, not a Raman spectral biomarker.",
        }
    )
    return row


def _band_metadata(feature_id: str, row: dict[str, object]) -> dict[str, object]:
    band = feature_id.removeprefix("band_band_")
    name, region, assignment = _band_info(band)
    row.update(
        {
            "display_name": f"Band integral: {name}",
            "short_label": name,
            "track": "band",
            "region": region,
            "window_cm1": _band_window(band),
            "biological_assignment": assignment,
            "interpretation_note": "Configured Raman band integral; assignment is window-level and may include overlapping molecules.",
        }
    )
    return row


def _ratio_metadata(feature_id: str, row: dict[str, object]) -> dict[str, object]:
    body = feature_id.removeprefix("ratio_ratio_")
    if "__" not in body:
        row.update({"track": "ratio"})
        return row
    num, den = body.split("__", 1)
    num_name, num_region, num_assignment = _band_info(num)
    den_name, den_region, den_assignment = _band_info(den)
    row.update(
        {
            "display_name": f"Band ratio: {num_name} / {den_name}",
            "short_label": f"{num_name} / {den_name}",
            "track": "ratio",
            "region": _combine_regions(num_region, den_region),
            "window_cm1": f"{_band_window(num)} / {_band_window(den)}",
            "biological_assignment": f"{num_assignment} relative to {den_assignment}",
            "interpretation_note": "Configured band-ratio feature; interpret as relative balance, not absolute abundance.",
        }
    )
    return row


def _ch_state_metadata(feature_id: str, row: dict[str, object]) -> dict[str, object]:
    body = feature_id.removeprefix("chlipid_")
    labels = {
        "ch_entropy": "CH lipid-state entropy",
        "ch_max_share": "Dominant CH lipid-state share",
        "ch_max_cluster": "Dominant CH lipid-state cluster",
        "ch_noise_dist": "CH lipid-state distance/noise",
    }
    match = re.match(r"ch_prop_(\d+)$", body)
    label = f"CH lipid-state cluster {match.group(1)} proportion" if match else labels.get(body, body.replace("_", " "))
    row.update(
        {
            "display_name": label,
            "short_label": label,
            "track": "chlipid",
            "region": "ch_stretch",
            "biological_assignment": "Unsupervised CH-stretch lipid-state feature",
            "fold_local": True,
            "interpretation_note": "Fold-local UMAP/KMeans feature; cluster IDs are not globally comparable across folds.",
        }
    )
    return row


def _split_region_metric(body: str) -> tuple[str, str]:
    for prefix in ("full_", "fp_", "ch_"):
        if body.startswith(prefix):
            return prefix[:-1], body.removeprefix(prefix)
    return "", body


def _normalize_region(region: str) -> str:
    aliases = {
        "fp": "fingerprint",
        "region_1300": "fingerprint",
        "1300": "fingerprint",
        "ch": "ch_stretch",
        "region_2850": "ch_stretch",
        "2850": "ch_stretch",
        "full": "full_spectrum",
        "": "",
    }
    return aliases.get(region, region)


def _region_label(region: object) -> str:
    return {
        "fingerprint": "Fingerprint",
        "ch_stretch": "CH-stretch",
        "full_spectrum": "Full spectrum",
        "multi_region": "Whole-spectrum",
        "clinical": "Clinical",
        "": "Spectral",
    }.get(str(region), str(region).replace("_", " ").title())


def _combine_regions(left: str, right: str) -> str:
    if left == right:
        return left
    if not left:
        return right
    if not right:
        return left
    return "cross_region"


def _band_info(band: str) -> tuple[str, str, str]:
    return _BAND_ASSIGNMENTS.get(band, (band.replace("_", " "), _region_from_band_name(band), "Configured Raman band"))


def _band_window(band: str) -> str:
    bands = config.get("whole_spectra", {}).get("bands", {})
    values = bands.get(band)
    if not values or len(values) != 2:
        return ""
    return f"{float(values[0]):.0f}-{float(values[1]):.0f}"


def _region_from_band_name(band: str) -> str:
    match = re.search(r"(\d{3,4})", band)
    if not match:
        return ""
    return "ch_stretch" if int(match.group(1)) >= 2450 else "fingerprint"


def _channel_to_wavenumber(channel: int, split_index: int | None = None) -> float:
    axes = _canonical_wavenumbers()
    if axes is not None and 0 <= channel < len(axes):
        return float(axes[channel])
    split = int(split_index) if split_index is not None else CANONICAL_FP_CHANNELS
    if channel >= split:
        ch_start, _ = CANONICAL_REGION_BOUNDS["ch_stretch"]
        return float(ch_start + (channel - split))
    fp_start, _ = CANONICAL_REGION_BOUNDS["fingerprint"]
    return float(fp_start + channel)


def _canonical_wavenumbers() -> np.ndarray | None:
    try:
        from src.fusion.area_pipeline import _CANONICAL_REGION_AXES

        fp = _CANONICAL_REGION_AXES.get("fingerprint")
        ch = _CANONICAL_REGION_AXES.get("ch_stretch")
        if fp is not None and ch is not None:
            return np.concatenate([fp, ch])
    except Exception:
        return None
    return None


def _region_from_wavenumber(wn: float) -> str:
    if not np.isfinite(wn):
        return ""
    return "ch_stretch" if float(wn) >= 2450 else "fingerprint"


def _assignment_for_wavenumber(wn: float) -> str:
    if not np.isfinite(wn):
        return "Data-driven Raman peak"
    bands = config.get("whole_spectra", {}).get("bands", {})
    for band, window in bands.items():
        if len(window) == 2 and float(window[0]) <= float(wn) <= float(window[1]):
            return _band_info(str(band))[2]
    return "Data-driven Raman peak outside configured canonical bands"


def _format_wn(wn: float, channel: int) -> str:
    if np.isfinite(wn):
        return f"{wn:.0f} cm^-1"
    return f"channel {channel}"
