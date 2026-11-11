"""Phase S: cross-endpoint sign consistency for panel features."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import get_confirmatory_panel_dir, get_spectral_interpretability_dir
from src import resolve_cfg as _cfg
from src.analysis.io_utils import load_csv_or_empty, write_csv_atomic
from src.analysis.preregistered_panel import panel_cfg

logger = logging.getLogger(__name__)


def _empty_consistency() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "feature_name",
            "dominant_sign",
            "n_endpoints_evaluated",
            "n_same_sign_endpoints",
            "min_endpoint_agreement",
            "consistent",
            "endpoint_signs",
        ]
    )


def _sign_label(value: float) -> str:
    if not np.isfinite(value) or value == 0:
        return "zero_or_missing"
    return "positive" if value > 0 else "negative"


def compute_endpoint_consistency(
    associations: pd.DataFrame,
    *,
    endpoints: list[str],
    min_endpoint_agreement: int,
) -> pd.DataFrame:
    """Score whether each feature has the same partial-Spearman sign across endpoints."""
    required = {"feature_name", "endpoint", "partial_spearman_rho"}
    if associations.empty or not required.issubset(associations.columns):
        return _empty_consistency()

    rows: list[dict[str, object]] = []
    filtered = associations.loc[associations["endpoint"].isin(endpoints)].copy()
    for feature, group in filtered.groupby("feature_name"):
        signs_by_endpoint: dict[str, str] = {}
        for endpoint in endpoints:
            values = pd.to_numeric(
                group.loc[group["endpoint"] == endpoint, "partial_spearman_rho"],
                errors="coerce",
            ).dropna()
            if values.empty:
                signs_by_endpoint[endpoint] = "zero_or_missing"
            else:
                signs_by_endpoint[endpoint] = _sign_label(float(values.iloc[0]))

        nonzero = [sign for sign in signs_by_endpoint.values() if sign in {"positive", "negative"}]
        positive_count = nonzero.count("positive")
        negative_count = nonzero.count("negative")
        if positive_count >= negative_count:
            dominant = "positive" if positive_count else "zero_or_missing"
            same_sign = positive_count
        else:
            dominant = "negative"
            same_sign = negative_count

        rows.append(
            {
                "feature_name": feature,
                "dominant_sign": dominant,
                "n_endpoints_evaluated": int(len(nonzero)),
                "n_same_sign_endpoints": int(same_sign),
                "min_endpoint_agreement": int(min_endpoint_agreement),
                "consistent": bool(same_sign >= min_endpoint_agreement),
                "endpoint_signs": ";".join(f"{endpoint}:{signs_by_endpoint[endpoint]}" for endpoint in endpoints),
            }
        )

    if not rows:
        return _empty_consistency()
    return pd.DataFrame(rows).sort_values(["consistent", "n_same_sign_endpoints", "feature_name"], ascending=[False, False, True])


def run_endpoint_consistency(
    *,
    associations_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write endpoint-consistency summary for Phase L consumption."""
    cfg = _cfg(cfg)
    panel = panel_cfg(cfg)
    phase_cfg = cfg.get("analysis", {}).get("endpoint_consistency", {})
    if not bool(phase_cfg.get("enabled", True)):
        return {}

    target_dir = Path(output_dir) if output_dir is not None else get_spectral_interpretability_dir("consistency")
    target_dir.mkdir(parents=True, exist_ok=True)
    path = (
        Path(associations_path)
        if associations_path is not None
        else get_confirmatory_panel_dir() / "panel_continuous_associations.csv"
    )
    associations = load_csv_or_empty(path)

    endpoints = [str(item) for item in panel.get("endpoints", {}).get("continuous", ["moca_baseline", "moca_3m", "delta_moca"])]
    min_agreement = int(panel.get("robustness", {}).get("min_endpoint_agreement", 2))
    consistency = compute_endpoint_consistency(
        associations,
        endpoints=endpoints,
        min_endpoint_agreement=min_agreement,
    )
    output_path = target_dir / "endpoint_consistency.csv"
    write_csv_atomic(consistency, output_path, index=False)
    logger.info("Endpoint consistency written to %s", output_path)
    return {"endpoint_consistency": output_path}
