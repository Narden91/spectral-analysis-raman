"""Compute readiness from current identity and cohort counts without fitting models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def evaluate_cohort_readiness(
    clinical_audit: Mapping[str, Any], cohort_summary: Mapping[str, Any], readiness: Mapping[str, Any],
) -> dict[str, Any]:
    expected_identity = int(readiness["expected_identity_verified"])
    expected_join = {str(key): int(value) for key, value in readiness["expected_cohort_join"].items()}
    if expected_identity <= 0 or not expected_join or any(value < 0 for value in expected_join.values()):
        raise ValueError("Readiness expectations must contain valid identity and cohort counts")
    observed_join = {key: int(cohort_summary[key]) for key in expected_join if key in cohort_summary}
    identity_ready = clinical_audit.get("stroke_identity_verified") == expected_identity
    join_ready = observed_join == expected_join
    return {
        "expected_identity_verified": expected_identity,
        "observed_identity_verified": clinical_audit.get("stroke_identity_verified"),
        "identity_verified_matches_expected": identity_ready,
        "cohort_join": {
            "status": "ok" if join_ready else "count_mismatch",
            "expected": expected_join, "observed": observed_join,
        },
        "ready_for_cohort_evaluation": bool(identity_ready and join_ready),
    }
