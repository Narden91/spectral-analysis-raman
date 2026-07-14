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

_VALID_STATUSES = frozenset(
    {
        STATUS_OK,
        STATUS_INSUFFICIENT_N,
        STATUS_NOT_TESTED_GATE,
        STATUS_MISSING_REQUIRED_DATA,
        STATUS_DISABLED_OR_SKIPPED,
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

    def __post_init__(self) -> None:
        if self.status not in _VALID_STATUSES:
            raise ValueError(f"Unknown phase status '{self.status}' for phase '{self.phase_key}'")

    def as_row(self) -> dict[str, object]:
        return {
            "phase_key": self.phase_key,
            "status": self.status,
            "reason": self.reason,
            "n_subjects": self.n_subjects,
            "n_artifacts": self.n_artifacts,
        }
