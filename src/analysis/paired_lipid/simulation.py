"""Backward-compatible facade re-exporting synthetic generation and bootstrap calibration."""

from __future__ import annotations

from src.analysis.paired_lipid.calibration import (
    _SCENARIO_META_KEYS as _SCENARIO_META_KEYS,
)
from src.analysis.paired_lipid.calibration import (
    _SCENARIO_RESERVED_KEYS as _SCENARIO_RESERVED_KEYS,
)
from src.analysis.paired_lipid.calibration import (
    CALIBRATION_SEED_RANGES as CALIBRATION_SEED_RANGES,
)
from src.analysis.paired_lipid.calibration import (
    SeedSet as SeedSet,
)
from src.analysis.paired_lipid.calibration import (
    _eval_single_calibration_rep as _eval_single_calibration_rep,
)
from src.analysis.paired_lipid.calibration import (
    _eval_single_scenario_rep as _eval_single_scenario_rep,
)
from src.analysis.paired_lipid.calibration import (
    _ldl_factor_recovery_r2 as _ldl_factor_recovery_r2,
)
from src.analysis.paired_lipid.calibration import (
    _ScenarioRepResult as _ScenarioRepResult,
)
from src.analysis.paired_lipid.calibration import (
    check_gate_attainable as check_gate_attainable,
)
from src.analysis.paired_lipid.calibration import (
    evaluate_bootstrap_calibration as evaluate_bootstrap_calibration,
)
from src.analysis.paired_lipid.calibration import (
    evaluate_mde_signal_curve as evaluate_mde_signal_curve,
)
from src.analysis.paired_lipid.calibration import (
    evaluate_scenario_calibration_suite as evaluate_scenario_calibration_suite,
)
from src.analysis.paired_lipid.calibration import (
    min_reps_for_gate as min_reps_for_gate,
)
from src.analysis.paired_lipid.calibration import (
    run_synthetic_scenario_suite as run_synthetic_scenario_suite,
)
from src.analysis.paired_lipid.calibration import (
    wilson_score_interval as wilson_score_interval,
)
from src.analysis.paired_lipid.nested_cv import (
    INTERVAL_METHODS as INTERVAL_METHODS,
)
from src.analysis.paired_lipid.nested_cv import (
    IntervalMethod as IntervalMethod,
)
from src.analysis.paired_lipid.synthetic import (
    _MAP_CHUNK_SAMPLES as _MAP_CHUNK_SAMPLES,
)
from src.analysis.paired_lipid.synthetic import (
    DEFAULT_N_SINGLE_SAMPLE as DEFAULT_N_SINGLE_SAMPLE,
)
from src.analysis.paired_lipid.synthetic import (
    LABEL_RANGE as LABEL_RANGE,
)
from src.analysis.paired_lipid.synthetic import (
    generate_synthetic_cohort as generate_synthetic_cohort,
)

__all__ = [
    "CALIBRATION_SEED_RANGES",
    "DEFAULT_N_SINGLE_SAMPLE",
    "INTERVAL_METHODS",
    "IntervalMethod",
    "LABEL_RANGE",
    "SeedSet",
    "_MAP_CHUNK_SAMPLES",
    "_SCENARIO_META_KEYS",
    "_SCENARIO_RESERVED_KEYS",
    "_ScenarioRepResult",
    "_eval_single_calibration_rep",
    "_eval_single_scenario_rep",
    "_ldl_factor_recovery_r2",
    "check_gate_attainable",
    "evaluate_bootstrap_calibration",
    "evaluate_mde_signal_curve",
    "evaluate_scenario_calibration_suite",
    "generate_synthetic_cohort",
    "min_reps_for_gate",
    "run_synthetic_scenario_suite",
    "wilson_score_interval",
]
