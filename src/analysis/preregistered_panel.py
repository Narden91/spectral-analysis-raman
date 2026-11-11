"""Fixed Raman panel: provenance snapshot, integrals, and named indices.

Naming note: "preregistered"/"confirmatory" are historical directory and label
names. Unless `provenance_type == "externally_preregistered"` with a real
`registration_id`, the panel is a fixed POST HOC hypothesis (see
`panel_provenance` and `_primary_hypothesis_lines`) — it is not an
independently pre-registered or confirmed test for this cohort.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src import get_confirmatory_panel_dir
from src import resolve_cfg as _cfg
from src.analysis.band_features import (
    band_integral,
    band_median_snr,
    build_named_index,
)
from src.analysis.clinical_feature_associations import load_disease_feature_frame_with_bc
from src.analysis.io_utils import write_csv_atomic, write_text_atomic
from src.analysis.raman_bands import RAMAN_BANDS
from src.data_preparation.manifest import config_hash

logger = logging.getLogger(__name__)

_SNR_GATED_INTEGRALS = frozenset(
    {
        "c_o_c_sugar_925",
        "beta_sheet_980",
        "tyrosine_851",
        "o_p_o_821",
    }
)


def panel_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("analysis", {}).get("preregistered_panel", {})


def panel_provenance(panel: dict[str, Any]) -> dict[str, str]:
    provenance_type = str(panel.get("provenance_type", "post_hoc_fixed"))
    if provenance_type not in {"post_hoc_fixed", "externally_preregistered"}:
        raise ValueError("panel provenance_type must be 'post_hoc_fixed' or 'externally_preregistered'")
    fields = {
        "provenance_type": provenance_type,
        "spectral_basis": "pre_normalization",
        "band_statistic": "trapezoidal_integral",
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
    feature = primary.get("feature", "carotenoid_to_carbonyl_ratio")
    endpoint = primary.get("endpoint", "delta_moca")
    covariates = primary.get("covariates", ["age", "sex"])
    covariate_text = ", ".join(str(item) for item in covariates)
    return [
        "## Primary exploratory anchor",
        "",
        f"H₀: {feature} is not associated with {endpoint} (partial Spearman | {covariate_text}).",
        f"H₁: {feature} is associated with {endpoint} (two-sided partial Spearman test).",
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
        "Declared external panel registration"
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
        "- External registration details are supplied metadata, not independently verified by this snapshot.",
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
    logger.info("Wrote panel-configuration snapshot to %s", path)
    return path


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

    for integral_name, spec in integrals_cfg.items():
        _, block, lower, upper = _integral_spec(integral_name, spec)
        if integral_name in RAMAN_BANDS:
            expected = RAMAN_BANDS[integral_name].bounds
            if (lower, upper) != expected:
                raise ValueError(f"Panel band '{integral_name}' window differs from the Raman registry")
        elif integral_name != "ch_bulk_2885" or (block, lower, upper) != ("2850", 2850, 2950):
            raise ValueError(f"Unknown panel band or composite window: {integral_name}")
        block = f"{block}_bc"

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

    if not resolved_integrals:
        return pd.DataFrame(index=frame.index), []

    index_specs: dict[str, list[str]] = {
        str(name): [str(item) for item in components]
        for name, components in panel.get("indices", {}).items()
    }
    index_columns: dict[str, pd.Series] = {}
    for index_name, components in index_specs.items():
        index_series = build_named_index(resolved_integrals, components)
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
    bc_feature_matrix_path: str | Path | None = None,
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

    disease_only, _meta, all_columns = load_disease_feature_frame_with_bc(
        feature_matrix_path=feature_matrix_path, bc_feature_matrix_path=bc_feature_matrix_path,
    )
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
