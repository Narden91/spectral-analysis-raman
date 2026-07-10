"""Fixed Raman panel: provenance snapshot, integrals, and named indices."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src import config, get_confirmatory_panel_dir
from src.analysis.band_features import (
    band_integral,
    band_median_snr,
    build_named_index,
    count_window_channels,
)
from src.analysis.clinical_feature_associations import feature_columns, load_disease_feature_frame
from src.analysis.io_utils import write_csv_atomic, write_text_atomic
from src.data_preparation.manifest import config_hash

logger = logging.getLogger(__name__)

_SNR_GATED_INTEGRALS = frozenset(
    {
        "structural_protein_925",
        "structural_protein_980",
        "tyrosine_851",
        "tyrosine_821",
    }
)


def _cfg(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    return config if cfg is None else cfg


def panel_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("analysis", {}).get("preregistered_panel", {})


def panel_provenance(panel: dict[str, Any]) -> dict[str, str]:
    provenance_type = str(panel.get("provenance_type", "post_hoc_fixed"))
    if provenance_type not in {"post_hoc_fixed", "externally_preregistered"}:
        raise ValueError("panel provenance_type must be 'post_hoc_fixed' or 'externally_preregistered'")
    fields = {
        "provenance_type": provenance_type,
        "registration_id": str(panel.get("registration_id") or ""),
        "registered_at": str(panel.get("registered_at") or ""),
        "registration_url": str(panel.get("registration_url") or ""),
        "analysis_plan_hash": str(panel.get("analysis_plan_hash") or ""),
    }
    if provenance_type == "externally_preregistered":
        required = ("registration_id", "registered_at", "analysis_plan_hash")
        missing = [name for name in required if not fields[name]]
        if missing:
            raise ValueError(
                "externally_preregistered panel is missing provenance field(s): " + ", ".join(missing)
            )
    if not fields["analysis_plan_hash"]:
        fields["analysis_plan_hash"] = config_hash(panel)
    return fields


def _integral_spec(name: str, spec: list[Any]) -> tuple[str, str, float, float]:
    if not isinstance(spec, (list, tuple)) or len(spec) != 3:
        raise ValueError(f"Integral '{name}' must be [block, lo_cm, hi_cm]")
    block = str(spec[0])
    return name, block, float(spec[1]), float(spec[2])


def _primary_hypothesis_lines(primary: dict[str, Any]) -> list[str]:
    feature = primary.get("feature", "oxidative_index")
    endpoint = primary.get("endpoint", "delta_moca")
    direction = str(primary.get("direction", "negative"))
    direction_text = "Lower" if direction == "negative" else "Higher"
    covariates = primary.get("covariates", ["age", "sex"])
    covariate_text = ", ".join(str(item) for item in covariates)
    return [
        "## Primary exploratory anchor",
        "",
        f"H₀: {feature} is not associated with {endpoint} (partial Spearman | {covariate_text}).",
        f"H₁: {direction_text} {feature} is associated with worse {endpoint} "
        f"(partial Spearman ρ {'<' if direction == 'negative' else '>'} 0).",
        "",
        "For this cohort this is a fixed post hoc hypothesis, not an independently confirmed test.",
        "",
    ]


def write_preregistration_snapshot(
    *,
    cfg: dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
    runtime_notes: list[str] | None = None,
) -> Path:
    cfg = _cfg(cfg)
    panel = panel_cfg(cfg)
    target_dir = Path(output_dir) if output_dir is not None else get_confirmatory_panel_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    resolved_hash = config_hash(panel)
    provenance = panel_provenance(panel)
    yaml_block = yaml.safe_dump({"analysis": {"preregistered_panel": panel}}, sort_keys=False)
    title = (
        "Externally pre-registered panel"
        if provenance["provenance_type"] == "externally_preregistered"
        else "Fixed post hoc Raman panel"
    )

    lines = [
        f"# {title}",
        "",
        f"- Generated (UTC): {timestamp}",
        f"- Config hash: {resolved_hash}",
        f"- Provenance type: {provenance['provenance_type']}",
        f"- Registration ID: {provenance['registration_id'] or 'not available'}",
        f"- Registered at: {provenance['registered_at'] or 'not available'}",
        f"- Registration URL: {provenance['registration_url'] or 'not available'}",
        f"- Analysis-plan hash: {provenance['analysis_plan_hash']}",
        "",
        *_primary_hypothesis_lines(panel.get("primary_hypothesis", {})),
        "## Frozen configuration",
        "",
        "```yaml",
        yaml_block.rstrip(),
        "```",
        "",
    ]
    if runtime_notes:
        lines.extend(["## Runtime notes", ""])
        lines.extend(f"- {note}" for note in runtime_notes)
        lines.append("")

    path = target_dir / "preregistration.md"
    write_text_atomic(path, "\n".join(lines))
    logger.info("Wrote pre-registration snapshot to %s", path)
    return path


def _resolve_lipid_carbonyl_window(
    columns: list[str],
    panel: dict[str, Any],
    runtime_notes: list[str],
) -> tuple[str, str, float, float]:
    primary_name = "lipid_carbonyl_1720"
    fallback_name = "lipid_carbonyl_1720_fallback"
    integrals = panel.get("integrals", {})
    if primary_name not in integrals:
        raise ValueError("Pre-registered panel is missing lipid_carbonyl_1720 integral")

    _, block, lower, upper = _integral_spec(primary_name, integrals[primary_name])
    if count_window_channels(columns, block=block, lower=lower, upper=upper) >= 3:
        return primary_name, block, lower, upper

    if fallback_name not in integrals:
        runtime_notes.append(
            f"{primary_name} window [{lower}, {upper}] has <3 channels and no fallback is configured."
        )
        return primary_name, block, lower, upper

    _, fb_block, fb_lower, fb_upper = _integral_spec(fallback_name, integrals[fallback_name])
    runtime_notes.append(
        f"{primary_name} narrow window [{lower}, {upper}] had <3 channels; "
        f"using fallback [{fb_lower}, {fb_upper}]."
    )
    return primary_name, fb_block, fb_lower, fb_upper


def build_panel_matrix(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    cfg: dict[str, Any] | None = None,
    runtime_notes: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    cfg = _cfg(cfg)
    panel = panel_cfg(cfg)
    notes = runtime_notes if runtime_notes is not None else []
    min_snr = float(panel.get("min_snr_for_peak_fit", 5.0))

    integrals_cfg: dict[str, list[Any]] = dict(panel.get("integrals", {}))
    resolved_integrals: dict[str, pd.Series] = {}
    lipid_name, lipid_block, lipid_lo, lipid_hi = _resolve_lipid_carbonyl_window(columns, panel, notes)

    for integral_name, spec in integrals_cfg.items():
        if integral_name == "lipid_carbonyl_1720_fallback":
            continue
        if integral_name == "lipid_carbonyl_1720":
            block, lower, upper = lipid_block, lipid_lo, lipid_hi
        else:
            _, block, lower, upper = _integral_spec(integral_name, spec)

        if integral_name in _SNR_GATED_INTEGRALS:
            snr = band_median_snr(frame, columns, block=block, lower=lower, upper=upper)
            if not np.isfinite(snr) or snr < min_snr:
                notes.append(
                    f"Skipped integral '{integral_name}' (median SNR={snr:.3f}, threshold={min_snr})."
                )
                continue

        series = band_integral(frame, columns, block=block, lower=lower, upper=upper)
        if series.notna().any():
            resolved_integrals[integral_name] = series

    index_specs: dict[str, list[str]] = {
        str(name): [str(item) for item in components]
        for name, components in panel.get("indices", {}).items()
    }
    index_columns: dict[str, pd.Series] = {}
    for index_name, components in index_specs.items():
        remapped = list(components)
        if "lipid_carbonyl_1720" in remapped and lipid_name in resolved_integrals:
            remapped = [lipid_name if item == "lipid_carbonyl_1720" else item for item in remapped]
        index_series = build_named_index(resolved_integrals, remapped)
        if index_series.notna().any():
            index_columns[index_name] = index_series

    ordered_index_names = list(index_specs.keys())
    primary_index_cols = [name for name in ordered_index_names if name in index_columns]
    integral_cols = sorted(resolved_integrals.keys())

    panel_frame = pd.DataFrame(index=frame.index)
    for name in primary_index_cols:
        panel_frame[name] = index_columns[name]
    for name in integral_cols:
        panel_frame[name] = resolved_integrals[name]

    column_order = primary_index_cols + integral_cols
    return panel_frame, column_order


def run_preregistered_panel(
    *,
    feature_matrix_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Path]:
    cfg = _cfg(cfg)
    panel = panel_cfg(cfg)
    if not bool(panel.get("enabled", True)):
        return {}
    provenance = panel_provenance(panel)

    runtime_notes: list[str] = []
    target_dir = Path(output_dir) if output_dir is not None else get_confirmatory_panel_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    prereg_path = write_preregistration_snapshot(
        cfg=cfg,
        output_dir=target_dir,
        runtime_notes=runtime_notes,
    )

    disease_only, _meta = load_disease_feature_frame(feature_matrix_path=feature_matrix_path)
    all_columns = feature_columns(disease_only)
    panel_matrix, column_order = build_panel_matrix(
        disease_only,
        all_columns,
        cfg=cfg,
        runtime_notes=runtime_notes,
    )

    if runtime_notes:
        prereg_path = write_preregistration_snapshot(
            cfg=cfg,
            output_dir=target_dir,
            runtime_notes=runtime_notes,
        )

    export = disease_only[["subject_id"]].copy()
    for column in column_order:
        export[column] = panel_matrix[column]

    matrix_path = target_dir / "panel_feature_matrix.csv"
    write_csv_atomic(export, matrix_path, index=False)

    provenance_path = target_dir / "panel_provenance.csv"
    write_csv_atomic(pd.DataFrame([provenance]), provenance_path, index=False)

    logger.info("Fixed panel matrix written to %s", matrix_path)
    return {
        "preregistration": prereg_path,
        "panel_provenance": provenance_path,
        "panel_feature_matrix": matrix_path,
    }
