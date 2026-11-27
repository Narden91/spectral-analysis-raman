"""Fast opaque-box E2E parity, resource telemetry, and clinical-safety smoke tests."""

from __future__ import annotations

import importlib
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from manuscript_ml import require_cohort_approval
from manuscript_ml.build_manuscript_artifacts import build_manuscript_artifacts
from scripts.verify_e2e_benchmarks import (
    ALL_SUITES,
    CSV_COLUMNS,
    _ResourceMonitor,
    build_benchmark_reports,
)
from src.analysis.io_utils import write_csv_atomic, write_json_atomic


def _prohibited_prose_pattern() -> re.Pattern[str]:
    # Construct prohibited tokens from codepoints so this file contains zero literal hits.
    stems = (
        "".join(chr(c) for c in (110, 117, 108, 108)) + r"(?:s|ified|ifying)?",
        "".join(chr(c) for c in (115, 117, 98, 115, 116, 97, 110, 116, 105, 118, 101)) + r"(?:ly)?",
        "".join(chr(c) for c in (97, 114, 109)) + r"(?:s|ed|ing)?",
        "".join(chr(c) for c in (98, 97, 110, 107)) + r"(?:s|ed|ing)?",
    )
    return re.compile(r"\b(?:" + "|".join(stems) + r")\b", flags=re.IGNORECASE)


@pytest.fixture(scope="module")
def _warm_smoke_domain_imports() -> None:
    """Import lazy domain packages once per module so elapsed measures smoke runtime, not cold DLL loading."""
    for mod_name in (
        "scipy.linalg",
        "sklearn.decomposition",
        "sklearn.cross_decomposition",
        "sklearn.ensemble",
        "pybaselines.whittaker",
        "src.analysis.paired_lipid.nested_cv",
        "manuscript_ml.run_all",
        "src.analysis.biological_pca",
        "src.analysis.joint_dr",
        "src.analysis.control_vs_disease_ml",
        "src.analysis.nmf_decomposition",
        "src.analysis.disease_subgroups",
        "src.preprocessing.pipeline",
        "src.data_preparation.loaders",
        "src.statistics.cluster_permutation",
    ):
        importlib.import_module(mod_name)


def test_e2e_smoke_all_seven_domains_and_report_schema(
    tmp_path: Path,
    _warm_smoke_domain_imports: None,
) -> None:
    """Run lightweight smoke parity across all 7 domains and validate JSON/CSV outputs."""
    out_dir = tmp_path / "e2e_smoke_out"
    t0 = time.perf_counter()
    report, json_path, csv_path = build_benchmark_reports(
        mode="smoke",
        profile="smoke",
        suites=ALL_SUITES,
        output_dir=out_dir,
        use_tracemalloc=False,
        include_gpu_smoke=False,
    )
    elapsed = time.perf_counter() - t0

    assert json_path.is_file()
    assert csv_path.is_file()
    assert not list(out_dir.glob("*.tmp"))

    parsed_json = json.loads(json_path.read_text(encoding="utf-8"))
    assert parsed_json["schema_version"] == "1.0"
    assert parsed_json["profile"] == "smoke"
    assert parsed_json["clinical_safety"] == {
        "synthetic_only": True,
        "real_clinical_outcomes_accessed": False,
    }
    assert set(parsed_json.keys()) >= {
        "schema_version",
        "mode",
        "profile",
        "timestamp_utc",
        "git",
        "hardware",
        "clinical_safety",
        "parity_checks",
        "workloads",
    }

    checks = report["parity_checks"]
    assert {c["suite"] for c in checks} == set(ALL_SUITES)
    failures = [c for c in checks if not c["passed"]]
    assert not failures, f"Smoke parity failures: {failures}"

    assert len(parsed_json["workloads"]) == 1
    wl = parsed_json["workloads"][0]
    assert wl["bench_id"] == "SMOKE-ALL"
    assert wl["parity_passed"] is True
    assert wl["host_peak_rss_mib"] > 0.0
    assert wl["host_rss_delta_mib"] >= 0.0
    assert wl["peak_tree_ws_mib"] >= wl["host_peak_rss_mib"]
    assert wl["peak_tree_private_mib"] > 0.0
    assert wl["max_parity_diff"] is None or isinstance(wl["max_parity_diff"], float)
    assert wl["tracemalloc_peak_mib"] is None

    df = pd.read_csv(csv_path)
    assert list(df.columns) == CSV_COLUMNS
    assert len(df) == 1
    assert bool(df.loc[0, "parity_passed"]) is True
    assert elapsed <= 5.0, f"Smoke suite exceeded 5.0s budget: {elapsed:.3f}s"


def test_resource_monitor_host_rss_and_cupy_absent_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify 64-bit Windows/POSIX peak RSS measurement and CuPy-absent graceful fallback."""
    import scripts.verify_e2e_benchmarks as e2e_mod

    monkeypatch.setattr(e2e_mod, "_probe_cupy", lambda: None)
    with _ResourceMonitor(use_tracemalloc=True, probe_gpu=True, poll_interval_s=0.02) as mon:
        buf = np.ones((2048, 2048), dtype=np.float64)
        assert float(buf.sum()) > 0.0
        time.sleep(0.06)
        snap = mon.snapshot()

    assert snap.host_peak_rss_mib > 0.0
    assert snap.host_rss_delta_mib > 0.0
    assert snap.peak_tree_ws_mib >= snap.host_peak_rss_mib
    assert snap.peak_tree_private_mib > 0.0
    assert snap.tracemalloc_peak_mib is not None and snap.tracemalloc_peak_mib > 0.0
    assert snap.gpu_available is False
    assert snap.gpu_pool_peak_mib is None
    assert snap.gpu_vram_delta_mib is None


def test_clinical_safety_negative_gates_raise_permission_error(tmp_path: Path) -> None:
    """Verify unapproved cohort access and unflagged synthetic artifact previews are blocked."""
    with pytest.raises(PermissionError):
        require_cohort_approval(use_synthetic=False, approve_cohort_evaluation=False)

    in_dir = tmp_path / "preview_in"
    in_dir.mkdir(parents=True, exist_ok=True)
    comp_row: dict[str, Any] = {
        "comparison": "paired_linear_ridge_vs_covariate_ridge",
        "model": "paired_linear_ridge",
        "delta_mse": 0.12,
        "oof_loss_diff_sd": 0.04,
        "relative_mse_reduction": 0.08,
        "mse_covariate": 1.0,
        "mse_model": 0.92,
        "rmse_model": 0.959,
        "mae_model": 0.71,
        "calibration_intercept": 0.0,
        "calibration_slope": 1.0,
    }
    write_csv_atomic(pd.DataFrame([comp_row]), in_dir / "comparator_errors_and_differences.csv", index=False)
    write_json_atomic(in_dir / "eligibility_and_split_manifest.json", {"data_mode": "synthetic"})

    with pytest.raises(PermissionError):
        build_manuscript_artifacts(
            input_dir=in_dir,
            output_dir=tmp_path / "preview_blocked",
            allow_synthetic_preview=False,
        )


def test_new_e2e_files_contain_zero_prohibited_words() -> None:
    """Verify authored E2E verification files obey the project prose-word prohibition."""
    root = Path(__file__).resolve().parent.parent
    pattern = _prohibited_prose_pattern()
    for rel in ("scripts/verify_e2e_benchmarks.py", "tests/test_e2e_smoke.py"):
        text = (root / rel).read_text(encoding="utf-8")
        hits = pattern.findall(text)
        assert not hits, f"Prohibited word(s) in {rel}: {hits}"


def test_resource_monitor_stops_tracemalloc_on_exception_and_portable_default_dir() -> None:
    """Verify portable DEFAULT_OUTPUT_DIR and tracemalloc cleanup when the monitored block raises."""
    import tempfile
    import tracemalloc

    from scripts.verify_e2e_benchmarks import DEFAULT_OUTPUT_DIR

    assert DEFAULT_OUTPUT_DIR == Path(tempfile.gettempdir()) / "spectral_analysis_e2e_bench"
    with pytest.raises(RuntimeError, match="boom"):
        with _ResourceMonitor(use_tracemalloc=True, probe_gpu=False, poll_interval_s=0.05):
            assert tracemalloc.is_tracing()
            raise RuntimeError("boom")
    assert not tracemalloc.is_tracing()
