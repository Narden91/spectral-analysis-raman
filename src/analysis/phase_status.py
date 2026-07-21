"""Shared status vocabulary and skip-manifest helper for cross-region phases.

Phases that cannot run at the current sample size (below a configured
``min_subjects`` / ``min_subjects_per_group`` floor) report a status instead
of writing header-only CSVs. ``disease_only_pipeline`` aggregates these into
a single ``00_run_metadata/phase_status.csv``.
"""

from __future__ import annotations

from dataclasses import dataclass

STATUS_OK = "ok"
STATUS_INSUFFICIENT_N = "insufficient_n"
STATUS_NOT_TESTED_GATE = "not_tested_gate"
STATUS_MISSING_REQUIRED_DATA = "missing_required_data"
STATUS_DISABLED_OR_SKIPPED = "disabled_or_skipped"
STATUS_FAILED_EXECUTION = "failed_execution"

EVIDENCE_NOT_APPLICABLE = "not_applicable"
EVIDENCE_NO_SUPPORTED_SIGNAL = "no_supported_evidence"
EVIDENCE_EXPLORATORY_SIGNAL = "exploratory_signal"

_VALID_EVIDENCE_STATUSES = frozenset(
    {EVIDENCE_NOT_APPLICABLE, EVIDENCE_NO_SUPPORTED_SIGNAL, EVIDENCE_EXPLORATORY_SIGNAL}
)

_VALID_STATUSES = frozenset(
    {
        STATUS_OK,
        STATUS_INSUFFICIENT_N,
        STATUS_NOT_TESTED_GATE,
        STATUS_MISSING_REQUIRED_DATA,
        STATUS_DISABLED_OR_SKIPPED,
        STATUS_FAILED_EXECUTION,
    }
)


@dataclass
class PhaseStatus:
    """One phase's outcome, for aggregation into ``phase_status.csv``."""

    phase_key: str
    status: str
    reason: str = ""
    n_subjects: int | None = None
    n_artifacts: int = 0
    evidence_status: str = EVIDENCE_NOT_APPLICABLE

    def __post_init__(self) -> None:
        if self.status not in _VALID_STATUSES:
            raise ValueError(f"Unknown phase status '{self.status}' for phase '{self.phase_key}'")
        if self.evidence_status not in _VALID_EVIDENCE_STATUSES:
            raise ValueError(
                f"Unknown evidence status '{self.evidence_status}' for phase '{self.phase_key}'"
            )

    def as_row(self) -> dict[str, object]:
        return {
            "phase_key": self.phase_key,
            "status": self.status,
            "reason": self.reason,
            "n_subjects": self.n_subjects,
            "n_artifacts": self.n_artifacts,
            "evidence_status": self.evidence_status,
        }
