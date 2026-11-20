"""Compute readiness from current identity and cohort counts without fitting models."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


def _parse_count(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("Readiness expectations must contain valid identity and cohort counts")
    if isinstance(value, float) and (not math.isfinite(value) or not value.is_integer()):
        raise ValueError("Readiness expectations must contain valid identity and cohort counts")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Readiness expectations must contain valid identity and cohort counts") from exc


def validate_readiness_expectations(readiness: Mapping[str, Any]) -> tuple[int, dict[str, int]]:
    """Validate and coerce expected identity and cohort-join counts."""
    if not isinstance(readiness, Mapping):
        raise ValueError("Readiness expectations must contain valid identity and cohort counts")
    if "expected_identity_verified" not in readiness or "expected_cohort_join" not in readiness:
        raise ValueError("Readiness expectations must contain valid identity and cohort counts")
    raw_join = readiness["expected_cohort_join"]
    if not isinstance(raw_join, Mapping):
        raise ValueError("Readiness expectations must contain valid identity and cohort counts")
    expected_identity = _parse_count(readiness["expected_identity_verified"])
    expected_join = {str(key): _parse_count(value) for key, value in raw_join.items()}
    if expected_identity <= 0 or not expected_join or any(value < 0 for value in expected_join.values()):
        raise ValueError("Readiness expectations must contain valid identity and cohort counts")
    return expected_identity, expected_join


def evaluate_cohort_readiness(
    clinical_audit: Mapping[str, Any],
    cohort_summary: Mapping[str, Any] | None,
    readiness: Mapping[str, Any],
    *,
    join_status: str | None = None,
    store_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate clinical identity and cohort-join count readiness."""
    if join_status is not None and join_status not in ("not_run", "failed"):
        raise ValueError(f"Unsupported explicit join_status: {join_status!r}")
    expected_identity, expected_join = validate_readiness_expectations(readiness)
    identity_ready = bool(clinical_audit.get("stroke_identity_verified") == expected_identity)

    cohort_join: dict[str, Any] = {"status": "not_run", "expected": expected_join}
    if store_meta:
        for key, value in store_meta.items():
            if key not in ("status", "expected", "observed"):
                cohort_join[key] = value

    if join_status is not None:
        cohort_join["status"] = join_status
        if cohort_summary is not None:
            cohort_join["observed"] = {
                key: int(cohort_summary[key]) for key in expected_join if key in cohort_summary
            }
        join_ready = False
    elif cohort_summary is None:
        cohort_join["status"] = "not_run"
        join_ready = False
    else:
        observed_join = {key: int(cohort_summary[key]) for key in expected_join if key in cohort_summary}
        join_ready = observed_join == expected_join
        cohort_join["status"] = "ok" if join_ready else "count_mismatch"
        cohort_join["observed"] = observed_join

    return {
        "expected_identity_verified": expected_identity,
        "observed_identity_verified": clinical_audit.get("stroke_identity_verified"),
        "identity_verified_matches_expected": identity_ready,
        "cohort_join": cohort_join,
        "ready_for_cohort_evaluation": bool(identity_ready and join_ready),
    }
