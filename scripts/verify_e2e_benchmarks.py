#!/usr/bin/env python3
"""Opaque-box E2E parity verification and benchmark harness for spectral-analysis.

Covers all 7 analysis/preprocessing/pipeline domains across 4 verification tiers:
- Tier 1: Feature parity & contract checks (T1-A1..T1-I5)
- Tier 2: Boundary & corner regimes (T2-01..T2-07)
- Tier 3: Cross-feature end-to-end chains (T3-01..T3-04)
- Tier 4: Reproducible timing, host RAM, child-tree RSS, and CuPy VRAM workloads (B-01..B-13)

All checks run exclusively on synthetic cohorts, synthetic spectra, and dummy outcomes.
Real clinical outcomes are never evaluated.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import copy
import ctypes
import gc
import importlib.util
import inspect
import json
import math
import os
import pickle
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import tracemalloc
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
_cli_spec = importlib.util.spec_from_file_location("_raman_cli_util", ROOT / "src" / "cli_util.py")
assert _cli_spec is not None and _cli_spec.loader is not None
_cli_util = importlib.util.module_from_spec(_cli_spec)
_cli_spec.loader.exec_module(_cli_util)
_cli_util.ensure_project_root_on_path(ROOT)

from src.analysis.io_utils import (
    write_csv_atomic,
    write_json_atomic,
    write_npz_atomic,
    write_parquet_atomic,
    write_text_atomic,
)
from src.analysis.paired_lipid._backend import _probe_cupy

ALL_SUITES: tuple[str, ...] = (
    "paired_lipid",
    "manuscript_ml",
    "biological_pca",
    "cluster_permutation",
    "control_vs_disease_ml",
    "nmf_subgroups",
    "preprocessing_store",
)

DEFAULT_OUTPUT_DIR = Path(tempfile.gettempdir()) / "spectral_analysis_e2e_bench"

CSV_COLUMNS: list[str] = [
    "bench_id",
    "profile",
    "feature_ids",
    "median_wall_seconds",
    "min_wall_seconds",
    "iqr_wall_seconds",
    "host_peak_rss_mib",
    "host_rss_delta_mib",
    "peak_tree_ws_mib",
    "tracemalloc_peak_mib",
    "gpu_pool_peak_mib",
    "parity_passed",
    "max_parity_diff",
]


# ---------------------------------------------------------------------------
# Resource & Hardware Telemetry (_ResourceMonitor)
# ---------------------------------------------------------------------------

if sys.platform == "win32":
    from ctypes import wintypes as _wt

    class _PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
        _fields_ = [
            ("cb", _wt.DWORD),
            ("PageFaultCount", _wt.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    class _PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", _wt.DWORD),
            ("cntUsage", _wt.DWORD),
            ("th32ProcessID", _wt.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", _wt.DWORD),
            ("cntThreads", _wt.DWORD),
            ("th32ParentProcessID", _wt.DWORD),
            ("pcPriClassBase", _wt.LONG),
            ("dwFlags", _wt.DWORD),
            ("szExeFile", _wt.WCHAR * 260),
        ]

    class _MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", _wt.DWORD),
            ("dwMemoryLoad", _wt.DWORD),
            ("ullTotalPhys", ctypes.c_uint64),
            ("ullAvailPhys", ctypes.c_uint64),
            ("ullTotalPageFile", ctypes.c_uint64),
            ("ullAvailPageFile", ctypes.c_uint64),
            ("ullTotalVirtual", ctypes.c_uint64),
            ("ullAvailVirtual", ctypes.c_uint64),
            ("ullAvailExtendedVirtual", ctypes.c_uint64),
        ]

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _psapi = ctypes.WinDLL("psapi", use_last_error=True)

    _k32.GetCurrentProcess.restype = _wt.HANDLE
    _k32.GetCurrentProcess.argtypes = []
    _k32.OpenProcess.restype = _wt.HANDLE
    _k32.OpenProcess.argtypes = [_wt.DWORD, _wt.BOOL, _wt.DWORD]
    _k32.CloseHandle.restype = _wt.BOOL
    _k32.CloseHandle.argtypes = [_wt.HANDLE]
    _k32.CreateToolhelp32Snapshot.restype = _wt.HANDLE
    _k32.CreateToolhelp32Snapshot.argtypes = [_wt.DWORD, _wt.DWORD]
    _k32.Process32FirstW.restype = _wt.BOOL
    _k32.Process32FirstW.argtypes = [_wt.HANDLE, ctypes.POINTER(_PROCESSENTRY32W)]
    _k32.Process32NextW.restype = _wt.BOOL
    _k32.Process32NextW.argtypes = [_wt.HANDLE, ctypes.POINTER(_PROCESSENTRY32W)]
    _k32.GlobalMemoryStatusEx.restype = _wt.BOOL
    _k32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(_MEMORYSTATUSEX)]

    _psapi.GetProcessMemoryInfo.restype = _wt.BOOL
    _psapi.GetProcessMemoryInfo.argtypes = [
        _wt.HANDLE,
        ctypes.POINTER(_PROCESS_MEMORY_COUNTERS_EX),
        _wt.DWORD,
    ]

    _INVALID_HANDLE_VALUE = _wt.HANDLE(-1).value
    _PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    _TH32CS_SNAPPROCESS = 0x00000002


def _query_process_memory_mib(pid: int | None = None) -> tuple[float, float, float]:
    """Return (working_set_mib, peak_working_set_mib, private_mib) for pid or current process."""
    if sys.platform == "win32":
        pmc = _PROCESS_MEMORY_COUNTERS_EX()
        pmc.cb = ctypes.sizeof(_PROCESS_MEMORY_COUNTERS_EX)
        close_after = False
        if pid is None or pid == os.getpid():
            handle = _k32.GetCurrentProcess()
        else:
            handle = _k32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
            if not handle or handle == _INVALID_HANDLE_VALUE:
                return 0.0, 0.0, 0.0
            close_after = True
        try:
            ok = _psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb)
            if not ok:
                return 0.0, 0.0, 0.0
            mib = float(1024**2)
            return (
                float(pmc.WorkingSetSize) / mib,
                float(pmc.PeakWorkingSetSize) / mib,
                float(pmc.PrivateUsage) / mib,
            )
        finally:
            if close_after:
                _k32.CloseHandle(handle)

    import resource

    rss_raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    peak_mib = rss_raw / (1024.0**2) if sys.platform == "darwin" else rss_raw / 1024.0
    return peak_mib, peak_mib, peak_mib


def _snapshot_tree_memory_mib(root_pid: int) -> tuple[float, float]:
    """Sum current (working_set_mib, private_mib) across root_pid and its descendants."""
    if sys.platform != "win32":
        ws, _, priv = _query_process_memory_mib(root_pid)
        return ws, priv

    snap = _k32.CreateToolhelp32Snapshot(_TH32CS_SNAPPROCESS, 0)
    if not snap or snap == _INVALID_HANDLE_VALUE:
        ws, _, priv = _query_process_memory_mib(root_pid)
        return ws, priv

    children_by_parent: dict[int, list[int]] = {}
    try:
        entry = _PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(_PROCESSENTRY32W)
        if _k32.Process32FirstW(snap, ctypes.byref(entry)):
            while True:
                ppid = int(entry.th32ParentProcessID)
                cpid = int(entry.th32ProcessID)
                children_by_parent.setdefault(ppid, []).append(cpid)
                if not _k32.Process32NextW(snap, ctypes.byref(entry)):
                    break
    finally:
        _k32.CloseHandle(snap)

    tree_pids: list[int] = []
    queue = [root_pid]
    seen: set[int] = set()
    while queue:
        cur = queue.pop(0)
        if cur in seen:
            continue
        seen.add(cur)
        tree_pids.append(cur)
        for child in children_by_parent.get(cur, ()):
            if child not in seen:
                queue.append(child)

    total_ws = 0.0
    total_priv = 0.0
    for pid in tree_pids:
        ws, _, priv = _query_process_memory_mib(pid)
        total_ws += ws
        total_priv += priv
    return total_ws, total_priv


def _total_system_ram_mib() -> float:
    if sys.platform == "win32":
        stat = _MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
        if _k32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return float(stat.ullTotalPhys) / float(1024**2)
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return float(pages * page_size) / float(1024**2)
    except Exception:
        return 0.0


@dataclass(slots=True)
class ResourceSnapshot:
    host_peak_rss_mib: float
    host_rss_delta_mib: float
    peak_tree_ws_mib: float
    peak_tree_private_mib: float
    tracemalloc_peak_mib: float | None
    gpu_available: bool
    gpu_pool_peak_mib: float | None
    gpu_vram_delta_mib: float | None


class _ResourceMonitor:
    """Measure host RSS, child-process tree RSS (50 ms sampler), CuPy VRAM, and optional tracemalloc."""

    def __init__(self, *, use_tracemalloc: bool = False, probe_gpu: bool = True, poll_interval_s: float = 0.05) -> None:
        self.use_tracemalloc = bool(use_tracemalloc)
        self.probe_gpu = bool(probe_gpu)
        self.poll_interval_s = float(poll_interval_s)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._root_pid = os.getpid()
        self._pre_ws_mib = 0.0
        self._peak_tree_ws_mib = 0.0
        self._peak_tree_priv_mib = 0.0
        self._cp: Any | None = None
        self._free_before_bytes = 0
        self._lock = threading.Lock()

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            ws, priv = _snapshot_tree_memory_mib(self._root_pid)
            with self._lock:
                if ws > self._peak_tree_ws_mib:
                    self._peak_tree_ws_mib = ws
                if priv > self._peak_tree_priv_mib:
                    self._peak_tree_priv_mib = priv
            self._stop_event.wait(self.poll_interval_s)

    def __enter__(self) -> _ResourceMonitor:
        ws_now, _, priv_now = _query_process_memory_mib(self._root_pid)
        self._pre_ws_mib = ws_now
        self._peak_tree_ws_mib = ws_now
        self._peak_tree_priv_mib = priv_now
        if self.probe_gpu:
            self._cp = _probe_cupy()
            if self._cp is not None:
                try:
                    pool = self._cp.get_default_memory_pool()
                    pool.free_all_blocks()
                    self._cp.cuda.get_current_stream().synchronize()
                    free_b, _ = self._cp.cuda.runtime.memGetInfo()
                    self._free_before_bytes = int(free_b)
                except Exception:
                    self._cp = None
        if self.use_tracemalloc:
            if tracemalloc.is_tracing():
                tracemalloc.stop()
            tracemalloc.start()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self.use_tracemalloc and tracemalloc.is_tracing():
            tracemalloc.stop()

    def snapshot(self) -> ResourceSnapshot:
        ws_post, peak_ws, priv_post = _query_process_memory_mib(self._root_pid)
        ws_tree, priv_tree = _snapshot_tree_memory_mib(self._root_pid)
        with self._lock:
            peak_tree_ws = max(self._peak_tree_ws_mib, ws_tree, peak_ws)
            peak_tree_priv = max(self._peak_tree_priv_mib, priv_tree, priv_post)

        tm_peak_mib: float | None = None
        if self.use_tracemalloc and tracemalloc.is_tracing():
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            tm_peak_mib = float(peak_bytes) / float(1024**2)

        gpu_avail = False
        gpu_pool_mib: float | None = None
        gpu_delta_mib: float | None = None
        if self._cp is not None:
            try:
                self._cp.cuda.get_current_stream().synchronize()
                pool = self._cp.get_default_memory_pool()
                pool_total = int(pool.total_bytes())
                pool_used = int(pool.used_bytes())
                free_after, _ = self._cp.cuda.runtime.memGetInfo()
                gpu_avail = True
                gpu_pool_mib = float(max(pool_total, pool_used)) / float(1024**2)
                gpu_delta_mib = float(max(0, self._free_before_bytes - int(free_after))) / float(1024**2)
            except Exception:
                gpu_avail = False
                gpu_pool_mib = None
                gpu_delta_mib = None

        return ResourceSnapshot(
            host_peak_rss_mib=round(peak_ws, 3),
            host_rss_delta_mib=round(max(0.0, ws_post - self._pre_ws_mib), 3),
            peak_tree_ws_mib=round(peak_tree_ws, 3),
            peak_tree_private_mib=round(peak_tree_priv, 3),
            tracemalloc_peak_mib=round(tm_peak_mib, 3) if tm_peak_mib is not None else None,
            gpu_available=gpu_avail,
            gpu_pool_peak_mib=round(gpu_pool_mib, 3) if gpu_pool_mib is not None else None,
            gpu_vram_delta_mib=round(gpu_delta_mib, 3) if gpu_delta_mib is not None else None,
        )


def _collect_hardware_metadata(*, probe_gpu: bool = True) -> dict[str, Any]:
    cp = _probe_cupy() if probe_gpu else None
    gpu_name: str | None = None
    vram_total_mib: float | None = None
    if cp is not None:
        try:
            dev_id = cp.cuda.runtime.getDevice()
            props = cp.cuda.runtime.getDeviceProperties(dev_id)
            raw_name = props.get("name", b"")
            gpu_name = raw_name.decode("utf-8", errors="replace") if isinstance(raw_name, bytes) else str(raw_name)
            _, total_b = cp.cuda.runtime.memGetInfo()
            vram_total_mib = round(float(total_b) / float(1024**2), 2)
        except Exception:
            gpu_name = None
            vram_total_mib = None
    return {
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor() or platform.machine(),
        "logical_cores": int(os.cpu_count() or 1),
        "ram_total_mib": round(_total_system_ram_mib(), 2),
        "gpu_name": gpu_name,
        "vram_total_mib": vram_total_mib,
    }


def _collect_git_metadata(*, check_dirty: bool = True) -> dict[str, Any]:
    commit = "unknown"
    dirty = False
    try:
        head_path = ROOT / ".git" / "HEAD"
        if head_path.is_file():
            head_txt = head_path.read_text(encoding="utf-8").strip()
            if head_txt.startswith("ref: "):
                ref_path = ROOT / ".git" / head_txt[5:].strip()
                if ref_path.is_file():
                    commit = ref_path.read_text(encoding="utf-8").strip()
            else:
                commit = head_txt
        if commit == "unknown":
            c = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=5)
            if c.returncode == 0:
                commit = c.stdout.strip()
        if check_dirty:
            s = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, timeout=5)
            if s.returncode == 0:
                dirty = bool(s.stdout.strip())
    except Exception:
        pass
    return {"commit": commit, "dirty": dirty}


# ---------------------------------------------------------------------------
# Parity Check Recording & Assertion Helpers
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ParityCheckResult:
    check_id: str
    suite: str
    tier: str
    passed: bool
    max_diff: float | None
    duration_seconds: float
    skipped: bool = False
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "suite": self.suite,
            "tier": self.tier,
            "passed": bool(self.passed),
            "max_diff": self.max_diff,
            "duration_seconds": round(float(self.duration_seconds), 4),
            "skipped": bool(self.skipped),
            "detail": self.detail,
        }


def _max_abs_diff(a: Any, b: Any) -> float:
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    if arr_a.shape != arr_b.shape:
        raise AssertionError(f"Shape mismatch: {arr_a.shape} != {arr_b.shape}")
    if arr_a.size == 0:
        return 0.0
    nan_a = np.isnan(arr_a)
    nan_b = np.isnan(arr_b)
    if not np.array_equal(nan_a, nan_b):
        raise AssertionError("NaN pattern mismatch between target and reference")
    finite = ~nan_a
    if not np.any(finite):
        return 0.0
    return float(np.max(np.abs(arr_a[finite] - arr_b[finite])))


def _assert_close(a: Any, b: Any, *, atol: float, rtol: float = 0.0, label: str = "value") -> float:
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    if not np.allclose(arr_a, arr_b, atol=atol, rtol=rtol, equal_nan=True):
        diff = _max_abs_diff(arr_a, arr_b)
        raise AssertionError(f"{label} exceeded tolerance atol={atol}, rtol={rtol}: max_abs_diff={diff:.3e}")
    return _max_abs_diff(arr_a, arr_b)


def _run_check(
    check_id: str,
    suite: str,
    tier: str,
    fn: Callable[[], tuple[float | None, str] | tuple[float | None, str, bool]],
) -> ParityCheckResult:
    t0 = time.perf_counter()
    try:
        res = fn()
        if len(res) == 3:
            max_diff, detail, skipped = res  # type: ignore[misc]
        else:
            max_diff, detail = res  # type: ignore[misc]
            skipped = False
        dt = time.perf_counter() - t0
        return ParityCheckResult(
            check_id=check_id,
            suite=suite,
            tier=tier,
            passed=True,
            max_diff=max_diff,
            duration_seconds=dt,
            skipped=skipped,
            detail=detail,
        )
    except Exception as exc:
        dt = time.perf_counter() - t0
        return ParityCheckResult(
            check_id=check_id,
            suite=suite,
            tier=tier,
            passed=False,
            max_diff=None,
            duration_seconds=dt,
            skipped=False,
            detail=f"{type(exc).__name__}: {exc}",
        )


# ---------------------------------------------------------------------------
# Synthetic Generators & Reference Helpers (No Fast-Path Sharing)
# ---------------------------------------------------------------------------


def _make_eda_cfg(
    *,
    n_pca_components: int = 3,
    n_draws: int = 4,
    n_outer_splits: int = 3,
    n_outer_repeats: int = 1,
    n_inner_splits: int = 2,
    n_permutations: int = 6,
    n_bootstrap: int = 4,
    n_jobs: int = 1,
    nmf_n_init: int = 3,
    max_clusters: int = 3,
    raw_text_dir: Path | None = None,
    parsed_dir: Path | None = None,
    preprocessed_dir: Path | None = None,
    clinical_csv: Path | None = None,
    clinical_dict: Path | None = None,
) -> dict[str, Any]:
    return {
        "data": {
            "source": "preprocessed",
            "auto_build": False,
            "raw_text_dir": str(raw_text_dir or (ROOT / "data" / "raw")),
            "clinical_data_path": str(clinical_csv or (ROOT / "data" / "clinical_master.csv")),
            "clinical_dictionary_path": str(clinical_dict or (ROOT / "data" / "clinical_master_dictionary.csv")),
            "parsed_dir": str(parsed_dir or (ROOT / "data" / "parsed")),
            "preprocessed_dir": str(preprocessed_dir or (ROOT / "data" / "preprocessed")),
            "file_extension": ".txt",
        },
        "whole_spectra": {
            "bands": {
                "lipid_carbonyl_1721": [1710, 1730],
                "carotenoid_1525": [1510, 1535],
                "amide_i_1657": [1640, 1675],
                "ch2_sym_stretch_2853": [2845, 2860],
                "ch2_asym_stretch_2890": [2870, 2910],
                "ch3_asym_stretch_2937": [2925, 2945],
            }
        },
        "preprocessing": {
            "dead_pixel": {"min_spectral_std": 0.05, "saturation_value": 65000.0},
            "outlier_detection": {"enabled": True, "threshold_factor": 3.0},
            "cosmic_ray": {
                "enabled": True,
                "method": "spatial",
                "n_neighbors": 4,
                "z_threshold": 8.0,
                "max_width": 3,
                "max_corrected_fraction": 0.25,
                "window": 5,
            },
            "baseline": {
                "method": "arpls",
                "lam": 10000.0,
                "p": 0.01,
                "snip": {"max_half_window": 20, "decreasing": True, "smooth_half_window": 3},
                "arpls": {"lam": 10000.0, "ratio": 0.001, "max_iter": 50},
            },
            "smoothing": {"enabled": True, "method": "savgol", "window_length": 7, "polyorder": 2},
            "edge_trim": {"trim": 2, "edge_trim_cm": 5.0},
            "interpolation": {
                "target_spacing": 5.0,
                "method": "linear",
                "frozen_grids": {"fingerprint": [800.0, 1750.0], "ch_stretch": [2800.0, 3000.0]},
            },
            "retention": {"min_pixels_per_area": 2, "min_pixels_per_subject": 4},
            "quality_filter": {
                "spectral_quality": {
                    "enabled": True,
                    "snr_threshold": 2.0,
                    "peak_window": 3,
                    "noise_channels": 10,
                    "min_integrated_signal": None,
                    "anchor_windows": {
                        "fingerprint": [[995.0, 1015.0], [1435.0, 1470.0], [1630.0, 1680.0]],
                        "ch_stretch": [[2840.0, 2975.0]],
                    },
                }
            },
            "ranges": {
                "fingerprint": {
                    "baseline": {"method": "arpls", "lam": 10000.0, "arpls": {"lam": 10000.0, "ratio": 0.001, "max_iter": 50}},
                    "smoothing": {"enabled": True, "window_length": 7, "polyorder": 2},
                    "normalization": {"method": "snv"},
                },
                "ch_stretch": {
                    "baseline": {"method": "arpls", "lam": 10000.0, "arpls": {"lam": 10000.0, "ratio": 0.001, "max_iter": 50}},
                    "smoothing": {"enabled": True, "window_length": 7, "polyorder": 2},
                    "normalization": {"method": "area"},
                },
            },
        },
        "biological_pca": {
            "enabled": True,
            "n_components": n_pca_components,
            "aggregation": "median",
            "scale": False,
            "loading_region_quantile": 0.9,
            "loading_region_min_width_cm": 8.0,
            "loading_region_merge_gap_cm": 6.0,
            "measurement_count_sensitivity": {
                "enabled": True,
                "n_draws": n_draws,
                "random_state": 42,
            },
            "models": {
                "PCA_1330": {"spectral_blocks": ["1300"], "window": "full", "bands": ["carotenoid_1525"]},
                "PCA_2850": {"spectral_blocks": ["2850"], "window": "full", "bands": ["ch2_sym_stretch_2853"]},
                "PCA_whole": {
                    "spectral_blocks": ["1300", "2850"],
                    "window": "full",
                    "bands": ["carotenoid_1525", "ch2_sym_stretch_2853"],
                },
            },
            "biochemical_classes": {"lipid": ["ch2_sym_stretch_2853"], "carotenoid": ["carotenoid_1525"]},
        },
        "analysis": {
            "mode": "disease_vs_control",
            "random_state": 42,
            "clinical_endpoints": {
                "subject_id_col": "subject_id",
                "moca_baseline_col": "moca_baseline",
                "moca_3m_col": "moca_3m",
                "rbans_col": "rbans_total_3m",
                "age_col": "age_years",
                "aspects_col": "aspects",
                "sex_col": "sex",
                "diagnosis_col": "diagnosis_text",
                "lesion_location_col": "stroke_location",
            },
            "control_vs_disease_ml": {
                "enabled": True,
                "representations": ["band_panel"],
                "primary_model": "pls_da",
                "primary_representation": "band_panel",
                "n_outer_splits": n_outer_splits,
                "n_outer_repeats": n_outer_repeats,
                "n_inner_splits": n_inner_splits,
                "n_permutations": n_permutations,
                "permutation_min": max(2, min(4, n_permutations)),
                "permutation_check_interval": max(2, min(4, n_permutations)),
                "permutation_confidence": 0.99,
                "permutation_alpha": 0.05,
                "n_bootstrap": n_bootstrap,
                "importance_permutation_repeats": 2,
                "threshold_metric": "balanced_accuracy",
                "stratify_permutation_by": "sex",
                "random_state": 42,
                "n_jobs": n_jobs,
                "pls_components_grid": [1, 2],
                "logistic_c_grid": [0.1, 1.0],
                "rf_n_estimators_grid": [10],
                "rf_max_depth_grid": [2],
            },
            "nmf_decomposition": {"enabled": True},
            "disease_subgroups": {
                "enabled": True,
                "min_subjects": 6,
                "max_clusters": max_clusters,
                "n_bootstrap": n_bootstrap,
                "random_state": 42,
            },
            "preregistered_panel": {
                "enabled": True,
                "random_state": 42,
                "fdr_alpha": 0.05,
                "min_subjects_per_group": 5,
                "endpoints": {"continuous": ["moca_baseline"], "contrasts": []},
                "covariates": ["age", "sex"],
                "nmf": {"n_components": 2, "sparsity_alpha": 0.05, "n_init": nmf_n_init, "apply_to": "fingerprint"},
            },
        },
    }


def _make_synthetic_subject_inputs(
    *,
    n_control: int = 10,
    n_disease: int = 14,
    p_1300: int = 60,
    p_2850: int = 50,
    n_meas_ctrl: int = 3,
    n_meas_dis: int = 6,
    seed: int = 42,
) -> Any:
    from src.analysis.biological_pca import SubjectInputs

    rng = np.random.default_rng(seed)
    wn_1300 = np.linspace(800.0, 1750.0, p_1300)
    wn_2850 = np.linspace(2800.0, 3000.0, p_2850)
    n_total = n_control + n_disease
    subjects = [f"S{idx:03d}" for idx in range(n_total)]
    is_ctrl = [idx < n_control for idx in range(n_total)]

    latent_1300 = rng.normal(size=(n_total, 4)) @ rng.normal(scale=0.35, size=(4, p_1300))
    latent_2850 = rng.normal(size=(n_total, 3)) @ rng.normal(scale=0.35, size=(3, p_2850))
    base_1300 = np.exp(-0.5 * ((wn_1300 - 1525.0) / 30.0) ** 2)[None, :] + latent_1300
    base_2850 = np.exp(-0.5 * ((wn_2850 - 2853.0) / 20.0) ** 2)[None, :] + latent_2850

    measurement_rows: dict[str, dict[str, np.ndarray]] = {}
    rows_1300: list[np.ndarray] = []
    rows_2850: list[np.ndarray] = []
    counts: list[dict[str, object]] = []
    for idx, sid in enumerate(subjects):
        n_m = n_meas_ctrl if is_ctrl[idx] else n_meas_dis
        m_1300 = base_1300[idx][None, :] + rng.normal(scale=0.05, size=(n_m, p_1300))
        m_2850 = base_2850[idx][None, :] + rng.normal(scale=0.05, size=(n_m, p_2850))
        measurement_rows[sid] = {"1300": m_1300, "2850": m_2850}
        rows_1300.append(np.nanmedian(m_1300, axis=0))
        rows_2850.append(np.nanmedian(m_2850, axis=0))
        counts.append(
            {
                "subject_id": sid,
                "is_control": is_ctrl[idx],
                "n_measurements_1300": n_m,
                "n_measurements_2850": n_m,
            }
        )

    mat_1300 = np.vstack(rows_1300)
    mat_2850 = np.vstack(rows_2850)
    n_meas_list = [n_meas_ctrl if is_ctrl[i] else n_meas_dis for i in range(n_total)]
    meta_df = pd.DataFrame(
        {
            "subject_id": subjects,
            "is_control": is_ctrl,
            "condition": ["control" if c else "disease" for c in is_ctrl],
            "group": ["control" if c else "disease" for c in is_ctrl],
            "gender": ["female" if i % 2 == 0 else "male" for i in range(n_total)],
            "sex": ["female" if i % 2 == 0 else "male" for i in range(n_total)],
            "age": rng.uniform(45.0, 75.0, size=n_total),
            "moca_baseline": rng.uniform(16.0, 29.0, size=n_total),
            "n_measurements": n_meas_list,
            "n_areas": n_meas_list,
            "n_total_pixels": [m * 20 for m in n_meas_list],
            "n_retained_pixels": [m * 18 for m in n_meas_list],
        }
    )
    return SubjectInputs(
        subject_spectra=meta_df,
        subject_region_counts=pd.DataFrame(counts),
        input_provenance={"synthetic_only": True},
        axes={"1300": wn_1300, "2850": wn_2850},
        matrices={"1300": mat_1300, "2850": mat_2850},
        subjects=subjects,
        excluded_subjects=pd.DataFrame(columns=["subject_id", "reason"]),
        measurement_rows=measurement_rows,
    )


def _make_synthetic_feature_frame(
    *,
    n_subjects: int = 40,
    p_1300: int = 40,
    p_2850: int = 20,
    seed: int = 42,
    signal: float = 0.9,
    canonical_grid: bool = False,
    two_disease_clusters: bool = False,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if canonical_grid:
        wn_1300 = np.arange(800.0, 1750.5, 5.0)
        wn_2850 = np.arange(2800.0, 3000.5, 5.0)
        p_1300 = int(wn_1300.size)
        p_2850 = int(wn_2850.size)
    else:
        wn_1300 = np.linspace(1480.0, 1750.0, p_1300)
        wn_2850 = np.linspace(2840.0, 2950.0, p_2850)
    is_ctrl = np.array([i < n_subjects // 2 for i in range(n_subjects)], dtype=bool)
    sex = np.array(["female" if i % 2 == 0 else "male" for i in range(n_subjects)], dtype=object)
    y_dis = (~is_ctrl).astype(float)

    x1 = rng.uniform(0.2, 1.2, size=(n_subjects, p_1300))
    x2 = rng.uniform(0.2, 1.2, size=(n_subjects, p_2850))
    if signal > 0:
        x1[:, : min(8, p_1300)] += signal * y_dis[:, None]
        x2[:, : min(5, p_2850)] += signal * y_dis[:, None]

    if two_disease_clusters:
        dis_idx = np.flatnonzero(~is_ctrl)
        b1_a = (wn_1300 >= 1510.0) & (wn_1300 <= 1535.0)
        b1_b = (wn_1300 >= 1710.0) & (wn_1300 <= 1730.0)
        b2_a = (wn_2850 >= 2845.0) & (wn_2850 <= 2860.0)
        b2_b = (wn_2850 >= 2925.0) & (wn_2850 <= 2945.0)
        for rank_i, row_i in enumerate(dis_idx):
            is_cl_a = (rank_i % 2) == 0
            hi = 2.5 + rng.normal(scale=0.02, size=p_1300)
            lo = 0.3 + rng.normal(scale=0.02, size=p_1300)
            x1[row_i, b1_a] = np.clip(hi[b1_a] if is_cl_a else lo[b1_a], 0.1, None)
            x1[row_i, b1_b] = np.clip(lo[b1_b] if is_cl_a else hi[b1_b], 0.1, None)
            hi2 = 2.5 + rng.normal(scale=0.02, size=p_2850)
            lo2 = 0.3 + rng.normal(scale=0.02, size=p_2850)
            x2[row_i, b2_a] = np.clip(hi2[b2_a] if is_cl_a else lo2[b2_a], 0.1, None)
            x2[row_i, b2_b] = np.clip(lo2[b2_b] if is_cl_a else hi2[b2_b], 0.1, None)

    data: dict[str, Any] = {
        "subject_id": [f"SUBJ_{i:03d}" for i in range(n_subjects)],
        "is_control": is_ctrl,
        "sex": sex,
        "age": rng.uniform(45.0, 75.0, size=n_subjects),
        "moca_baseline": 24.0 - 2.0 * y_dis + rng.normal(scale=1.5, size=n_subjects),
        "binary_group": np.where(is_ctrl, "ctrl", "case"),
    }
    for j, wn in enumerate(wn_1300):
        data[f"1300__{wn:.4f}"] = x1[:, j]
    for j, wn in enumerate(wn_2850):
        data[f"2850__{wn:.4f}"] = x2[:, j]
    return pd.DataFrame(data)


def _build_synthetic_parsed_store(
    base_dir: Path,
    *,
    n_subjects: int = 6,
    n_pixels_per_map: int = 20,
    n_ch_1300: int = 80,
    n_ch_2850: int = 50,
    seed: int = 42,
    n_measurements_per_subject: int | tuple[int, int] = 1,
    add_two_disease_clusters: bool = False,
) -> tuple[Path, Path, Path, dict[str, Any]]:
    """Create a minimal valid parsed store (bypassing real clinical files) for preprocessing/store checks."""
    from src.data_preparation.manifest import hash_file, write_manifest
    from src.data_preparation.materialize import _build_manifest, _write_cohorts

    rng = np.random.default_rng(seed)
    parsed_dir = base_dir / "parsed"
    pre_dir = base_dir / "preprocessed"
    raw_dir = base_dir / "raw"
    parsed_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    clinical_csv = base_dir / "synthetic_clinical.csv"
    clinical_dict = base_dir / "synthetic_clinical_dict.csv"

    wn_1300_nat = np.linspace(785.0, 1765.0, n_ch_1300)
    wn_2850_nat = np.linspace(2785.0, 3015.0, n_ch_2850)

    meas_rows: list[dict[str, Any]] = []
    subj_rows: list[dict[str, Any]] = []
    pixel_rows: list[pd.DataFrame] = []
    raw_1300_parts: list[np.ndarray] = []
    raw_2850_parts: list[np.ndarray] = []
    axes_1300: list[np.ndarray] = []
    axes_2850: list[np.ndarray] = []
    starts = {"1300": 0, "2850": 0}

    grid_w = max(2, int(math.ceil(math.sqrt(n_pixels_per_map))))
    xs = np.array([float(i % grid_w) for i in range(n_pixels_per_map)])
    ys = np.array([float(i // grid_w) for i in range(n_pixels_per_map)])
    noise_scale = 0.3 if add_two_disease_clusters else 1.5

    for s_idx in range(n_subjects):
        sid = f"SYN_{s_idx:03d}"
        is_ctrl = s_idx < (n_subjects // 2)
        gender_val = "female" if s_idx % 2 == 0 else "male"
        subj_rows.append(
            {
                "subject_id": sid,
                "subject_id_raw": sid,
                "is_control": is_ctrl,
                "condition": "control" if is_ctrl else "disease",
                "gender": gender_val,
                "sex": gender_val,
                "age": 55.0 + s_idx,
                "age_years": 55.0 + s_idx,
                "moca_baseline": (25.0 if is_ctrl else (22.0 if s_idx % 2 == 0 else 17.0)) + (s_idx % 3) * 0.2,
                "moca_3m": 23.0 + (s_idx % 5),
                "rbans_total_3m": 85.0 + s_idx,
                "aspects": 8.0 - (s_idx % 3),
                "diagnosis_text": "control" if is_ctrl else "stroke",
                "stroke_location": "none" if is_ctrl else "cortex",
            }
        )
        if isinstance(n_measurements_per_subject, tuple):
            n_meas = int(n_measurements_per_subject[0 if is_ctrl else 1])
        else:
            n_meas = int(n_measurements_per_subject)

        for m_idx in range(1, n_meas + 1):
            area_id = f"area_{m_idx}"
            spot_id = f"spot_{m_idx}"
            for code, axis_nat, parts, axes_list in (
                ("1300", wn_1300_nat, raw_1300_parts, axes_1300),
                ("2850", wn_2850_nat, raw_2850_parts, axes_2850),
            ):
                peak_center = 1450.0 if code == "1300" else 2890.0
                peak = 200.0 * np.exp(-0.5 * ((axis_nat - peak_center) / 25.0) ** 2)
                if add_two_disease_clusters and not is_ctrl:
                    is_even_dis = (s_idx % 2) == 0
                    if code == "1300":
                        peak = peak + (
                            (180.0 if is_even_dis else 20.0) * np.exp(-0.5 * ((axis_nat - 1525.0) / 15.0) ** 2)
                            + (20.0 if is_even_dis else 180.0) * np.exp(-0.5 * ((axis_nat - 1721.0) / 15.0) ** 2)
                        )
                    else:
                        peak = peak + (
                            (180.0 if is_even_dis else 20.0) * np.exp(-0.5 * ((axis_nat - 2853.0) / 12.0) ** 2)
                            + (20.0 if is_even_dis else 180.0) * np.exp(-0.5 * ((axis_nat - 2937.0) / 12.0) ** 2)
                        )
                baseline = 50.0 + 0.02 * (axis_nat - axis_nat[0])
                spec = baseline[None, :] + peak[None, :] + rng.normal(scale=noise_scale, size=(n_pixels_per_map, axis_nat.size))
                if code == "1300" and s_idx == 0 and m_idx == 1:
                    spec[0, axis_nat.size // 2] += 1500.0  # single cosmic spike pixel

                m_id = f"{sid}__s1__{code}__std__meas{m_idx}"
                raw_file = raw_dir / f"{m_id}.txt"
                raw_file.write_text("# synthetic\n", encoding="utf-8")
                raw_sig = hash_file(raw_file)

                start = starts[code]
                stop = start + n_pixels_per_map
                starts[code] = stop
                parts.append(spec)
                axes_list.append(axis_nat)
                meas_rows.append(
                    {
                        "measurement_id": m_id,
                        "file_id": f"f_{m_id}",
                        "subject_id": sid,
                        "is_control": is_ctrl,
                        "gender": gender_val,
                        "sample_id": f"sample_{s_idx}",
                        "spot_id": spot_id,
                        "area_id": area_id,
                        "spectral_range": code,
                        "slide_number": "1",
                        "measurement_number": m_idx,
                        "acquisition_mode": "std",
                        "filepath": str(raw_file),
                        "raw_file_signature": raw_sig,
                        "raw_start": start,
                        "raw_stop": stop,
                        "source_axis_min": float(axis_nat[0]),
                        "source_axis_max": float(axis_nat[-1]),
                        "source_axis_channels": int(axis_nat.size),
                        "native_axis_row": len(axes_list) - 1,
                    }
                )
                pixel_rows.append(
                    pd.DataFrame(
                        {
                            "measurement_id": m_id,
                            "subject_id": sid,
                            "area_id": area_id,
                            "spectral_range": code,
                            "pixel_index": np.arange(n_pixels_per_map, dtype=int),
                            "x": xs,
                            "y": ys,
                            "raw_row": np.arange(start, stop, dtype=int),
                        }
                    )
                )

    meas_df = pd.DataFrame(meas_rows)
    subj_df = pd.DataFrame(subj_rows)
    subj_df.to_csv(clinical_csv, index=False)
    pd.DataFrame({"column": list(subj_df.columns)}).to_csv(clinical_dict, index=False)

    np.save(parsed_dir / "raw_spectra_1300.npy", np.vstack(raw_1300_parts))
    np.save(parsed_dir / "raw_spectra_2850.npy", np.vstack(raw_2850_parts))
    np.save(parsed_dir / "native_wavenumbers_1300.npy", np.vstack(axes_1300))
    np.save(parsed_dir / "native_wavenumbers_2850.npy", np.vstack(axes_2850))
    pd.concat(pixel_rows, ignore_index=True).to_parquet(parsed_dir / "pixels.parquet", index=False)
    meas_df.to_parquet(parsed_dir / "measurements.parquet", index=False)
    subj_df.to_parquet(parsed_dir / "subjects.parquet", index=False)
    areas_df = meas_df[
        [
            "measurement_id",
            "file_id",
            "subject_id",
            "sample_id",
            "spot_id",
            "area_id",
            "spectral_range",
            "slide_number",
            "measurement_number",
            "acquisition_mode",
        ]
    ].copy()
    areas_df.to_parquet(parsed_dir / "areas.parquet", index=False)

    cfg = _make_eda_cfg(
        raw_text_dir=raw_dir,
        parsed_dir=parsed_dir,
        preprocessed_dir=pre_dir,
        clinical_csv=clinical_csv,
        clinical_dict=clinical_dict,
    )
    _write_cohorts(parsed_dir, meas_df, subj_df)
    manifest = _build_manifest(
        "parsed",
        parsed_dir,
        {"type": "synthetic", "path": str(raw_dir), "manifest_hash": "synthetic"},
        cfg,
        meas_df,
        subj_df,
        areas_df,
        {"pixels_1300": starts["1300"], "pixels_2850": starts["2850"]},
    )
    write_manifest(parsed_dir / "manifest.json", manifest)
    return parsed_dir, pre_dir, clinical_csv, cfg


def _reference_pca_full(spectra: np.ndarray, *, n_components: int, scale: bool) -> dict[str, np.ndarray]:
    """Independent PCA(svd_solver='full') reference with deterministic sign flip."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    arr = np.asarray(spectra, dtype=float)
    valid_cols = ~np.all(np.isnan(arr), axis=0)
    arr = arr[:, valid_cols]
    if scale:
        X = StandardScaler().fit_transform(arr)
    else:
        X = arr - np.nanmean(arr, axis=0)
    pca = PCA(n_components=n_components, svd_solver="full")
    scores = pca.fit_transform(X)
    loadings = pca.components_.copy()
    for c in range(loadings.shape[0]):
        idx = int(np.argmax(np.abs(loadings[c])))
        sign = 1.0 if loadings[c, idx] >= 0.0 else -1.0
        loadings[c] *= sign
        scores[:, c] *= sign
    return {
        "scores": scores,
        "loadings": loadings,
        "explained_variance_ratio": pca.explained_variance_ratio_.copy(),
    }


def _reference_joint_pca_mfa(spectra: np.ndarray, split_index: int, n_components: int) -> dict[str, Any]:
    """Independent MFA reference using sklearn PCA(svd_solver='full')."""
    from sklearn.decomposition import PCA

    b1 = spectra[:, :split_index] - np.mean(spectra[:, :split_index], axis=0)
    b2 = spectra[:, split_index:] - np.mean(spectra[:, split_index:], axis=0)
    ev1 = float(PCA(n_components=1, svd_solver="full").fit(b1).explained_variance_[0])
    ev2 = float(PCA(n_components=1, svd_solver="full").fit(b2).explained_variance_[0])
    s1 = 1.0 / math.sqrt(ev1)
    s2 = 1.0 / math.sqrt(ev2)
    combined = np.hstack([b1 * s1, b2 * s2])
    out = _reference_pca_full(combined, n_components=n_components, scale=False)
    out["scaling_factors"] = {"fingerprint": s1, "ch_stretch": s2}
    out["block_eigenvalues"] = {"fingerprint": ev1, "ch_stretch": ev2}
    return out


def _reference_spatial_cosmic(
    spectra: np.ndarray,
    pixel_coords: np.ndarray,
    z_threshold: float,
    n_neighbors: int,
    max_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Brute-force per-pixel spatial cosmic detection reference."""
    from src.preprocessing.pipeline import _limit_spike_width

    n_pix = len(spectra)
    replacement = np.empty_like(spectra, dtype=float)
    spike_mask = np.zeros_like(spectra, dtype=bool)
    k = max(1, min(int(n_neighbors), n_pix - 1))
    for i in range(n_pix):
        dists = [math.hypot(pixel_coords[i, 0] - pixel_coords[j, 0], pixel_coords[i, 1] - pixel_coords[j, 1]) for j in range(n_pix)]
        dists[i] = math.inf
        nbr_idx = np.argsort(dists, kind="mergesort")[:k]
        med = np.median(spectra[nbr_idx], axis=0)
        replacement[i] = med
        res = spectra[i] - med
        mad = float(np.median(np.abs(res)))
        if mad == 0.0:
            mad = 1e-10
        local = res > (1.4826 * mad * z_threshold)
        spike_mask[i] = _limit_spike_width(local[np.newaxis, :], max_width)[0]
    return spike_mask, replacement


# ---------------------------------------------------------------------------
# Suite 1: paired_lipid (Areas A, B, and Tier 2 T2-01..T2-03, T2-05, T2-06, T3-02)
# ---------------------------------------------------------------------------


def _run_paired_lipid_checks(
    *,
    mode: str,
    profile: str,
    include_gpu_smoke: bool,
    tmp_dir: Path,
) -> list[ParityCheckResult]:
    from joblib import Parallel, delayed

    from src.analysis.paired_lipid import (
        _backend,
        _batched_models,
        _eigensolvers,
        _gram_cache,
        batched,
        calibration,
        nested_cv,
    )
    from src.analysis.paired_lipid.calibration import evaluate_bootstrap_calibration, evaluate_mde_signal_curve
    from src.analysis.paired_lipid.cohort import COVARIATE_PRESETS, PairedLipidConfig
    from src.analysis.paired_lipid.models import (
        CovariatePreprocessor,
        PairedMoments,
        WeightedPLSResult,
        WeightedSpectralPreprocessor,
    )
    from src.analysis.paired_lipid.nested_cv import (
        PreparedExternalTest,
        evaluate_nested_cohort,
        run_multiplicity_bootstrap,
        score_external_primary_pair,
    )
    from src.analysis.paired_lipid.splits import make_nested_splits
    from src.analysis.paired_lipid.synthetic import _median_maps, generate_synthetic_cohort

    results: list[ParityCheckResult] = []
    compact = profile == "smoke"

    if mode == "smoke":
        def _smoke_paired() -> tuple[float | None, str]:
            ds, _ = generate_synthetic_cohort(
                n_participants=32,
                n_single_sample=2,
                seed=42,
                compact_grid=True,
                n_nuisance_modes=4,
                nuisance_part_scale=0.3,
                nuisance_sample_scale=0.6,
            )
            cfg = PairedLipidConfig(
                pca_dimensions=(3, 4),
                retained_directions=(1, 2, 3),
                lambda_grid=(0.01, 1.0),
                ridge_alpha_grid=(0.1, 10.0),
                pls_component_grid=(1, 2),
                n_outer_splits=3,
                n_inner_splits=2,
            )
            seq = evaluate_nested_cohort(ds, cfg=cfg, include_band_paired=True, compute_diagnostics=False)
            boot = run_multiplicity_bootstrap(ds, cfg=cfg, n_bootstraps=1, bootstrap_seed=77, use_gpu=False)
            d_pt = _assert_close(
                boot["point_estimate_delta_mse"],
                seq["comparisons_vs_covariate_ridge"]["paired_linear_ridge"]["delta_mse"],
                atol=1e-12,
                label="smoke point delta_mse",
            )
            # Also check 1 resampled draw against sequential reference
            rng = np.random.default_rng(77)
            drawn = rng.integers(0, ds.n_participants, size=(1, ds.n_participants))
            mult0 = np.zeros(ds.n_participants, dtype=float)
            np.add.at(mult0, drawn[0], 1.0)
            seq0 = evaluate_nested_cohort(
                ds, cfg=cfg, participant_multiplicities=mult0, precomputed_splits=seq["splits"], include_band_paired=True, compute_diagnostics=False
            )
            d_b0 = _assert_close(
                boot["bootstrap_draws_delta_mse"][0],
                seq0["comparisons_vs_covariate_ridge"]["paired_linear_ridge"]["delta_mse"],
                atol=1e-12,
                label="smoke draw 0",
            )
            # External prepared test check
            ext_ds, _ = generate_synthetic_cohort(n_participants=16, n_single_sample=2, seed=99, compact_grid=True)
            prep = PreparedExternalTest.from_dataset(ext_ds, cfg.primary_endpoint, preset="primary")
            assert prep.raw_cov_frame is None
            s_raw = score_external_primary_pair(ds, ext_ds, cfg=cfg, splits=seq["splits"])
            s_prep = score_external_primary_pair(ds, prep, cfg=cfg, splits=seq["splits"])
            d_ext = _assert_close(s_prep["external_delta_mse"], s_raw["external_delta_mse"], atol=1e-12, label="smoke ext")

            max_d = max(d_pt, d_b0, d_ext)
            gpu_active_smoke = False
            if include_gpu_smoke and _probe_cupy() is not None:
                boot_gpu = run_multiplicity_bootstrap(ds, cfg=cfg, n_bootstraps=1, bootstrap_seed=77, use_gpu=True)
                d_gpu = _assert_close(
                    boot_gpu["bootstrap_draws_delta_mse"], boot["bootstrap_draws_delta_mse"], atol=1e-11, label="smoke gpu"
                )
                max_d = max(max_d, d_gpu)
                gpu_active_smoke = True
            return max_d, f"smoke CPU+ext parity (gpu_smoke={gpu_active_smoke})"

        results.append(_run_check("SMOKE-PL", "paired_lipid", "smoke", _smoke_paired))
        return results

    cp_avail = _probe_cupy() is not None

    # Full Tier 1 & Tier 2 checks for paired_lipid
    n_p = 80 if compact else 92
    n_draws_gpu = 8 if compact else 12
    ds_dec, _ = generate_synthetic_cohort(
        n_participants=n_p,
        n_single_sample=4,
        seed=42,
        compact_grid=compact,
        n_nuisance_modes=16,
        nuisance_part_scale=0.60,
        nuisance_sample_scale=1.20,
        nuisance_decay=0.88,
    )
    ds_undec, _ = generate_synthetic_cohort(
        n_participants=n_p,
        n_single_sample=4,
        seed=43,
        compact_grid=compact,
        n_nuisance_modes=0,
    )
    cfg = PairedLipidConfig()

    def _seq_draws_ref(dataset: Any, boot_seed: int, n_b: int, single_sample_downstream: bool | None = None, allow_missing: bool = False) -> np.ndarray:
        splits = make_nested_splits(dataset, cfg, cfg.primary_seed)
        rng = np.random.default_rng(boot_seed)
        drawn = rng.integers(0, dataset.n_participants, size=(n_b, dataset.n_participants))
        y_raw = np.asarray(dataset.outcomes[cfg.primary_endpoint], dtype=float)
        obs = np.isfinite(y_raw)
        vals = []
        for b in range(n_b):
            m = np.zeros(dataset.n_participants, dtype=float)
            np.add.at(m, drawn[b], 1.0)
            m *= obs
            r = evaluate_nested_cohort(
                dataset,
                cfg=cfg,
                participant_multiplicities=m,
                precomputed_splits=splits,
                include_band_paired=True,
                single_sample_downstream=single_sample_downstream,
                allow_missing_outcome=allow_missing,
                compute_diagnostics=False,
            )
            vals.append(float(r["comparisons_vs_covariate_ridge"]["paired_linear_ridge"]["delta_mse"]))
        return np.asarray(vals, dtype=float)

    def _t1_a1() -> tuple[float | None, str, bool]:
        if not cp_avail:
            return None, "CuPy/CUDA absent; skipped GPU CheFSI check", True
        boot_cpu = run_multiplicity_bootstrap(ds_dec, cfg=cfg, n_bootstraps=n_draws_gpu, bootstrap_seed=101, use_gpu=False)
        boot_gpu = run_multiplicity_bootstrap(ds_dec, cfg=cfg, n_bootstraps=n_draws_gpu, bootstrap_seed=101, use_gpu=True)
        assert boot_gpu["n_fallback_draws"] == 0, f"Unexpected fallback draws: {boot_gpu['n_fallback_draws']}"
        ref = _seq_draws_ref(ds_dec, 101, n_draws_gpu)
        d_cpu = _assert_close(boot_gpu["bootstrap_draws_delta_mse"], boot_cpu["bootstrap_draws_delta_mse"], atol=1e-11, label="T1-A1 GPU vs CPU")
        d_seq = _assert_close(boot_gpu["bootstrap_draws_delta_mse"], ref, atol=1e-10, label="T1-A1 GPU CheFSI vs seq")
        return max(d_cpu, d_seq), f"GPU CheFSI active across {n_draws_gpu} draws", False

    def _t1_a2() -> tuple[float | None, str, bool]:
        if not cp_avail:
            return None, "CuPy/CUDA absent; skipped GPU un-decayed check", True
        boot_cpu = run_multiplicity_bootstrap(ds_undec, cfg=cfg, n_bootstraps=n_draws_gpu, bootstrap_seed=102, use_gpu=False)
        boot_gpu = run_multiplicity_bootstrap(ds_undec, cfg=cfg, n_bootstraps=n_draws_gpu, bootstrap_seed=102, use_gpu=True)
        ref = _seq_draws_ref(ds_undec, 102, n_draws_gpu)
        d1 = _assert_close(boot_gpu["bootstrap_draws_delta_mse"], boot_cpu["bootstrap_draws_delta_mse"], atol=1e-11, label="T1-A2 GPU vs CPU")
        d2 = _assert_close(boot_gpu["bootstrap_draws_delta_mse"], ref, atol=1e-11, label="T1-A2 GPU vs seq")
        return max(d1, d2), f"Un-decayed GPU matches CPU & sequential ({n_draws_gpu} draws)", False

    def _t1_a3() -> tuple[float | None, str]:
        n_b = 36 if compact else 256
        boot_serial = run_multiplicity_bootstrap(
            ds_dec, cfg=cfg, n_bootstraps=n_b, bootstrap_seed=103, use_gpu=False, n_jobs=1
        )
        orig_resolve = batched._resolve_cpu_bootstrap_workers
        if compact:
            batched._resolve_cpu_bootstrap_workers = (  # type: ignore[assignment]
                lambda n_sub, n_j, min_parallel_draws=256: orig_resolve(n_sub, n_j, min_parallel_draws=8)
            )
        try:
            boot_par = run_multiplicity_bootstrap(
                ds_dec, cfg=cfg, n_bootstraps=n_b, bootstrap_seed=103, use_gpu=False, n_jobs=2
            )
        finally:
            batched._resolve_cpu_bootstrap_workers = orig_resolve  # type: ignore[assignment]
        assert int(boot_par["execution"].get("cpu_n_workers", 1)) >= 2, (
            f"Expected >= 2 CPU bootstrap workers, got {boot_par['execution']}"
        )
        d_par = _assert_close(
            boot_par["bootstrap_draws_delta_mse"],
            boot_serial["bootstrap_draws_delta_mse"],
            atol=1e-13,
            label="T1-A3 parallel vs serial CPU",
        )
        d_pt = _assert_close(
            boot_par["point_estimate_delta_mse"],
            boot_serial["point_estimate_delta_mse"],
            atol=1e-13,
            label="T1-A3 parallel vs serial point",
        )
        ref = _seq_draws_ref(ds_dec, 103, min(n_b, 8))
        d_seq = _assert_close(
            boot_serial["bootstrap_draws_delta_mse"][: min(n_b, 8)], ref, atol=1e-10, label="T1-A3 CPU vs seq (decayed)"
        )
        boot_undec = run_multiplicity_bootstrap(
            ds_undec, cfg=cfg, n_bootstraps=min(n_b, 8), bootstrap_seed=103, use_gpu=False, n_jobs=1
        )
        ref_undec = _seq_draws_ref(ds_undec, 103, min(n_b, 8))
        d_seq_u = _assert_close(
            boot_undec["bootstrap_draws_delta_mse"], ref_undec, atol=1e-12, label="T1-A3 CPU vs seq (un-decayed)"
        )
        d_seq = max(d_seq, d_seq_u)
        with _backend._single_thread_blas():
            ctrl = _backend._get_threadpool_controller()
            if ctrl is not None:
                blas_threads = [info.get("num_threads", 1) for info in ctrl.info() if info.get("user_api") == "blas"]
                assert all(int(t) == 1 for t in blas_threads), f"Expected 1 BLAS thread, got {blas_threads}"
        # Nested-pool guards: env sentinel, joblib thread nesting, and child process
        prev_env = os.environ.get("_PAIRED_LIPID_IN_BOOTSTRAP_WORKER")
        os.environ["_PAIRED_LIPID_IN_BOOTSTRAP_WORKER"] = "1"
        try:
            assert batched._resolve_cpu_bootstrap_workers(512, 4, min_parallel_draws=256) == 1
        finally:
            if prev_env is None:
                os.environ.pop("_PAIRED_LIPID_IN_BOOTSTRAP_WORKER", None)
            else:
                os.environ["_PAIRED_LIPID_IN_BOOTSTRAP_WORKER"] = prev_env
        assert Parallel(n_jobs=2, prefer="threads")(
            delayed(batched._resolve_cpu_bootstrap_workers)(512, 4, min_parallel_draws=256) for _ in range(2)
        ) == [1, 1]
        assert Parallel(n_jobs=2)(
            delayed(batched._resolve_cpu_bootstrap_workers)(512, 4, min_parallel_draws=256) for _ in range(2)
        ) == [1, 1]
        return max(d_par, d_pt, d_seq), f"CPU parallel vs serial ({n_b} draws), single_thread_blas, & nested-pool guards verified"

    def _t1_a4() -> tuple[float | None, str]:
        rng = np.random.default_rng(104)
        n_tr, n_val, p_cov, p_s = 30, 10, 3, 20
        X_cov_tr = rng.normal(size=(n_tr, p_cov))
        X_cov_val = rng.normal(size=(n_val, p_cov))
        X1_tr = rng.normal(size=(n_tr, p_s))
        X1_val = rng.normal(size=(n_val, p_s))
        X2_tr = rng.normal(size=(n_tr, p_s))
        X2_val = rng.normal(size=(n_val, p_s))
        y_tr = rng.normal(size=n_tr)
        y_val = rng.normal(size=n_val)
        w_tr = np.full(n_tr, 1.0 / n_tr)
        w_val = np.full(n_val, 1.0 / n_val)
        max_comp = 5
        primal = WeightedPLSResult.fit(X_cov_tr, np.hstack([X1_tr, X2_tr]), y_tr, w_tr, max_comp)
        preds_primal, _ = primal.predict(X_cov_val, np.hstack([X1_val, X2_val]))
        div = {
            "w_tr": w_tr[None, :],
            "w_val": w_val[None, :],
            "y_tr": y_tr,
            "y_val": y_val,
            "X_cov_tr": X_cov_tr[None, :, :],
            "X_cov_val": X_cov_val[None, :, :],
            "scale1_sq": np.array([1.0]),
            "scale2_sq": np.array([1.0]),
            "K_p1_tr": X1_tr @ X1_tr.T,
            "K_p1_vt": X1_val @ X1_tr.T,
            "K_p2_tr": X2_tr @ X2_tr.T,
            "K_p2_vt": X2_val @ X2_tr.T,
        }
        sse, preds_dual = _batched_models.batched_pls_sse_and_preds(np, div, max_comp)
        d = _assert_close(preds_dual[0], preds_primal, atol=1e-10, rtol=1e-10, label="T1-A4 PLS")
        sse_ref = np.sum(w_val[:, None] * (y_val[:, None] - preds_primal) ** 2, axis=0)
        d_sse = _assert_close(sse[0], sse_ref, atol=1e-10, rtol=1e-10, label="T1-A4 SSE")
        return max(d, d_sse), "Dual Gram PLS matches primal WeightedPLSResult"

    def _t1_a5() -> tuple[float | None, str]:
        gcache = _gram_cache._get_dataset_gram_cache(ds_dec, cfg)
        last_cpu = None
        for s_idx, seed in enumerate((cfg.primary_seed, *cfg.sensitivity_seeds)):
            outer_s, inner_s = make_nested_splits(ds_dec, cfg, seed)
            for f_idx, (tr_p, val_p) in enumerate(outer_s):
                last_cpu = _gram_cache._precompute_division_constants(np, gcache, ds_dec, tr_p, val_p, cfg)
                for in_tr, in_val in inner_s[f_idx]:
                    _gram_cache._precompute_division_constants(np, gcache, ds_dec, in_tr, in_val, cfg)
        assert last_cpu is not None
        max_d = 0.0
        if cp_avail:
            cp = _probe_cupy()
            outer_s, _ = make_nested_splits(ds_dec, cfg, cfg.primary_seed)
            tr_p, val_p = outer_s[0]
            dev_const = _gram_cache._precompute_division_constants(cp, gcache, ds_dec, tr_p, val_p, cfg)
            cpu_const = _gram_cache._precompute_division_constants(np, gcache, ds_dec, tr_p, val_p, cfg)
            max_d = _assert_close(cp.asnumpy(dev_const["K1_div"]), cpu_const["K1_div"], atol=1e-14, label="T1-A5 GPU upgrade")
        return max_d, "80-division _gram_cache & CPU->GPU upgrade verified"

    def _t1_a6() -> tuple[float | None, str]:
        orig_fast = nested_cv._select_inner_fast
        orig_batched = nested_cv._run_batched_multiplicity_draws

        def _forbid(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("Sequential reference path must never call batched fast path")

        nested_cv._select_inner_fast = _forbid  # type: ignore[assignment]
        nested_cv._run_batched_multiplicity_draws = _forbid  # type: ignore[assignment]
        try:
            res = evaluate_nested_cohort(ds_dec, cfg=cfg, include_band_paired=True, compute_diagnostics=False)
            assert "band_paired_ridge" in res["comparisons_vs_covariate_ridge"]
            assert math.isfinite(res["comparisons_vs_covariate_ridge"]["paired_linear_ridge"]["delta_mse"])
        finally:
            nested_cv._select_inner_fast = orig_fast  # type: ignore[assignment]
            nested_cv._run_batched_multiplicity_draws = orig_batched  # type: ignore[assignment]
        return 0.0, "Sequential include_band_paired=True strictly isolated from batched paths"

    def _t1_b1() -> tuple[float | None, str]:
        rng = np.random.default_rng(111)
        maps3 = rng.normal(size=(32, 3, 64))
        out3 = np.empty((32, 64), dtype=float)
        drop = np.array([i % 3 == 0 for i in range(32)], dtype=bool)
        _median_maps(maps3, out3, drop)
        ref3 = np.empty_like(out3)
        ref3[~drop] = np.median(maps3[~drop], axis=1)
        ref3[drop] = np.median(maps3[drop, :-1, :], axis=1)
        d1 = _assert_close(out3, ref3, atol=1e-15, label="T1-B1 3-map drop")
        maps5 = rng.normal(size=(16, 5, 64))
        out5 = np.empty((16, 64), dtype=float)
        _median_maps(maps5, out5, None)
        d2 = _assert_close(out5, np.median(maps5, axis=1), atol=1e-15, label="T1-B1 5-map")
        return max(d1, d2), "_median_maps matches np.median within 1e-15"

    def _t1_b2() -> tuple[float | None, str]:
        a, _ = generate_synthetic_cohort(
            n_participants=60, seed=9000, compact_grid=True, n_nuisance_modes=8, map_spatial_scale=0.5, map_dropout_fraction=0.2
        )
        b, _ = generate_synthetic_cohort(
            n_participants=60, seed=9000, compact_grid=True, n_nuisance_modes=8, map_spatial_scale=0.5, map_dropout_fraction=0.2
        )
        d1 = _assert_close(a.sample_1300, b.sample_1300, atol=0.0, label="T1-B2 sample_1300")
        d2 = _assert_close(a.sample_2850, b.sample_2850, atol=0.0, label="T1-B2 sample_2850")
        d3 = _assert_close(a.outcomes["ldl_mmol_l"], b.outcomes["ldl_mmol_l"], atol=0.0, label="T1-B2 outcomes")
        assert a.covariate_frame.equals(b.covariate_frame)
        return max(d1, d2, d3), "Exact RNG stream invariance"

    def _t1_b3() -> tuple[float | None, str]:
        ext_ds, _ = generate_synthetic_cohort(n_participants=120, n_single_sample=4, seed=222, compact_grid=compact)
        splits = make_nested_splits(ds_dec, cfg, cfg.primary_seed)
        max_d = 0.0
        for preset in COVARIATE_PRESETS:
            cfg_p = replace(cfg, covariate_preset=preset)
            prep = PreparedExternalTest.from_dataset(ext_ds, cfg_p.primary_endpoint, preset=preset)
            s_raw = score_external_primary_pair(ds_dec, ext_ds, cfg=cfg_p, splits=splits)
            s_prep = score_external_primary_pair(ds_dec, prep, cfg=cfg_p, splits=splits)
            max_d = max(max_d, _assert_close(s_prep["external_delta_mse"], s_raw["external_delta_mse"], atol=1e-12, label=f"T1-B3 {preset}"))
        return max_d, "PreparedExternalTest matches raw CohortDataset across all 5 presets"

    def _t1_b4() -> tuple[float | None, str]:
        ext_ds, _ = generate_synthetic_cohort(n_participants=80, n_single_sample=2, seed=223, compact_grid=True)
        prep = PreparedExternalTest.from_dataset(ext_ds, "ldl_mmol_l", preset="primary")
        assert prep.raw_cov_frame is None
        assert "raw_cov_matrix__primary" in prep.raw_cache
        mat, _ = prep.raw_cache["raw_cov_matrix__primary"]
        assert mat.dtype == np.float64
        payload = pickle.dumps(prep)
        assert b"pandas.core.frame" not in payload
        res = Parallel(n_jobs=2)(delayed(lambda p: p.n_participants)(prep) for _ in range(2))
        assert res == [80, 80]
        return 0.0, f"PreparedExternalTest IPC clean ({len(payload)} bytes)"

    def _t1_b5() -> tuple[float | None, str]:
        cal1 = evaluate_bootstrap_calibration(
            n_repetitions=2, n_bootstraps=4, n_participants=32, n_test_participants=80, base_seed=420, n_jobs=1, use_gpu=False, compact_grid=True
        )
        cal2 = evaluate_bootstrap_calibration(
            n_repetitions=2, n_bootstraps=4, n_participants=32, n_test_participants=80, base_seed=420, n_jobs=2, use_gpu=False, compact_grid=True
        )
        d1 = _assert_close(cal1["target_delta_mse"], cal2["target_delta_mse"], atol=1e-12, label="T1-B5 target_delta_mse")
        d2 = _assert_close(cal1["mean_point_estimate"], cal2["mean_point_estimate"], atol=1e-12, label="T1-B5 mean_point")
        d3 = _assert_close(cal1["unconditional_coverage"], cal2["unconditional_coverage"], atol=1e-12, label="T1-B5 coverage")
        return max(d1, d2, d3), "Calibration n_jobs=1 vs n_jobs=2 exact parity"

    def _t1_c1() -> tuple[float | None, str]:
        # Spectral MFA-PCA + paired moments are invariant across covariate_preset
        splits = make_nested_splits(ds_dec, cfg, cfg.primary_seed)
        tr_p, val_p = splits[0][0]
        mult = np.ones(ds_dec.n_participants, dtype=float)
        spec_ref = WeightedSpectralPreprocessor.fit(ds_dec, tr_p, mult[tr_p], cfg=cfg)
        coords_ref = spec_ref.transform_pca(ds_dec.sample_1300, ds_dec.sample_2850, k=max(cfg.pca_dimensions))
        pm_ref = PairedMoments.from_sample_coordinates(coords_ref, ds_dec, tr_p, mult[tr_p])
        max_d = 0.0
        base_w: np.ndarray | None = None
        base_t: np.ndarray | None = None
        for preset in COVARIATE_PRESETS:
            cfg_p = replace(cfg, covariate_preset=preset)
            gcache_p = _gram_cache._get_dataset_gram_cache(ds_dec, cfg_p)
            dconst = _gram_cache._precompute_division_constants(np, gcache_p, ds_dec, tr_p, val_p, cfg_p)
            div = _batched_models.batched_division_prep(
                np, gcache_p, dconst, mult[tr_p][None, :], mult[val_p][None, :], ds_dec.outcomes[cfg.primary_endpoint], np.asarray(cfg.ridge_alpha_grid), cfg_p
            )
            # Eigenvector signs are determined up to +/-1 per component between primal SVD and Gram eigh.
            max_d = max(
                max_d,
                _assert_close(np.abs(div["W_max"][0]), np.abs(pm_ref.W_max), atol=1e-12, label=f"T1-C1 |W_max| {preset}"),
                _assert_close(np.abs(div["T_max"][0]), np.abs(pm_ref.T_max), atol=1e-12, label=f"T1-C1 |T_max| {preset}"),
                _assert_close(np.linalg.eigvalsh(div["W_max"][0]), np.linalg.eigvalsh(pm_ref.W_max), atol=1e-12, label=f"T1-C1 eig(W_max) {preset}"),
                _assert_close(np.linalg.eigvalsh(div["T_max"][0]), np.linalg.eigvalsh(pm_ref.T_max), atol=1e-12, label=f"T1-C1 eig(T_max) {preset}"),
            )
            if base_w is None:
                base_w = div["W_max"][0].copy()
                base_t = div["T_max"][0].copy()
            else:
                assert base_t is not None
                max_d = max(
                    max_d,
                    _assert_close(div["W_max"][0], base_w, atol=1e-14, label=f"T1-C1 preset W_max {preset}"),
                    _assert_close(div["T_max"][0], base_t, atol=1e-14, label=f"T1-C1 preset T_max {preset}"),
                )
        return max_d, "Spectral MFA-PCA & paired moments invariant across all 5 presets"

    def _t2_01() -> tuple[float | None, str]:
        def _seq_cfg_ref(dataset: Any, cfg_cur: PairedLipidConfig, boot_seed: int, n_b: int) -> np.ndarray:
            splits_c = make_nested_splits(dataset, cfg_cur, cfg_cur.primary_seed)
            rng_c = np.random.default_rng(boot_seed)
            drawn_c = rng_c.integers(0, dataset.n_participants, size=(n_b, dataset.n_participants))
            vals_c = []
            for b in range(n_b):
                m = np.zeros(dataset.n_participants, dtype=float)
                np.add.at(m, drawn_c[b], 1.0)
                r = evaluate_nested_cohort(
                    dataset, cfg=cfg_cur, participant_multiplicities=m, precomputed_splits=splits_c, include_band_paired=True, compute_diagnostics=False
                )
                vals_c.append(float(r["comparisons_vs_covariate_ridge"]["paired_linear_ridge"]["delta_mse"]))
            return np.asarray(vals_c, dtype=float)

        ds_miss, _ = generate_synthetic_cohort(
            n_participants=40, n_single_sample=2, seed=501, compact_grid=True, covariate_missing_rate=0.15
        )
        cfg_q = replace(cfg, covariate_preset="quadratic_age_and_label")
        boot = run_multiplicity_bootstrap(ds_miss, cfg=cfg_q, n_bootstraps=4, bootstrap_seed=501, use_gpu=False)
        seq_vals = _seq_cfg_ref(ds_miss, cfg_q, 501, 4)
        d1 = _assert_close(boot["bootstrap_draws_delta_mse"], seq_vals, atol=1e-12, label="T2-01 missing covs")

        # np.inf / -np.inf covariates + all-NaN test-fold row
        splits_q = make_nested_splits(ds_miss, cfg_q, cfg_q.primary_seed)
        val_row_idx = int(splits_q[0][0][1][0])
        tr_row_0 = int(splits_q[0][0][0][0])
        tr_row_1 = int(splits_q[0][0][0][1])
        cov_inf = ds_miss.covariate_frame.copy()
        cov_inf.loc[tr_row_0, "age_years"] = np.inf
        cov_inf.loc[tr_row_1, "age_years"] = -np.inf
        cov_inf.loc[val_row_idx, ["age_years", "sex", "statin", "sample_label_trend"]] = np.nan
        ds_inf = replace(ds_miss, covariate_frame=cov_inf)
        boot_inf = run_multiplicity_bootstrap(ds_inf, cfg=cfg_q, n_bootstraps=3, bootstrap_seed=506, use_gpu=False)
        seq_inf = _seq_cfg_ref(ds_inf, cfg_q, 506, 3)
        d2 = _assert_close(boot_inf["bootstrap_draws_delta_mse"], seq_inf, atol=1e-12, label="T2-01 inf + all-NaN test")

        # p_b == 0 with p_m > 0
        n_m = ds_miss.n_participants
        cov_pb0 = pd.DataFrame(
            {
                "age_years": [65.0 if i % 4 != 0 else np.nan for i in range(n_m)],
                "sex": ["female" if i % 5 != 0 else np.nan for i in range(n_m)],
                "statin": [1.0] * n_m,
                "sample_label_trend": [10.0] * n_m,
            }
        )
        ds_pb0 = replace(ds_miss, covariate_frame=cov_pb0)
        gc_pb0 = _gram_cache._get_dataset_gram_cache(ds_pb0, cfg)
        sp_pb0 = make_nested_splits(ds_pb0, cfg, cfg.primary_seed)
        dc_pb0 = _gram_cache._precompute_division_constants(np, gc_pb0, ds_pb0, sp_pb0[0][0][0], sp_pb0[0][0][1], cfg)
        assert dc_pb0["X_raw_tr"].shape[1] == 0 and dc_pb0["X_miss_tr"].shape[1] > 0
        boot_pb0 = run_multiplicity_bootstrap(ds_pb0, cfg=cfg, n_bootstraps=3, bootstrap_seed=507, use_gpu=False)
        seq_pb0 = _seq_cfg_ref(ds_pb0, cfg, 507, 3)
        d3 = _assert_close(boot_pb0["bootstrap_draws_delta_mse"], seq_pb0, atol=1e-12, label="T2-01 p_b=0 p_m>0")
        return max(d1, d2, d3), "Missing covs, inf/-inf, all-NaN test fold, & p_b=0/p_m>0 parity verified"

    def _t2_02() -> tuple[float | None, str]:
        ds_ss, _ = generate_synthetic_cohort(n_participants=48, n_single_sample=4, seed=502, compact_grid=True, subset_mode="single_sample")
        splits_ss = make_nested_splits(ds_ss, cfg, cfg.primary_seed)
        boot_f = run_multiplicity_bootstrap(ds_ss, cfg=cfg, n_bootstraps=4, bootstrap_seed=502, use_gpu=False, single_sample_downstream=False)
        seq_f = evaluate_nested_cohort(ds_ss, cfg=cfg, precomputed_splits=splits_ss, include_band_paired=True, single_sample_downstream=False, compute_diagnostics=False)
        seq_f_draws = _seq_draws_ref(ds_ss, 502, 4, single_sample_downstream=False)
        d_f_pt = _assert_close(
            boot_f["point_estimate_delta_mse"],
            seq_f["comparisons_vs_covariate_ridge"]["paired_linear_ridge"]["delta_mse"],
            atol=1e-10,
            label="T2-02 single_sample_downstream=False point",
        )
        d_f_draws = _assert_close(
            boot_f["bootstrap_draws_delta_mse"],
            seq_f_draws,
            atol=1e-10,
            label="T2-02 single_sample_downstream=False draws",
        )
        boot_auto = run_multiplicity_bootstrap(ds_ss, cfg=cfg, n_bootstraps=4, bootstrap_seed=502, use_gpu=False, single_sample_downstream=None)
        seq_auto = evaluate_nested_cohort(ds_ss, cfg=cfg, precomputed_splits=splits_ss, include_band_paired=True, single_sample_downstream=None, compute_diagnostics=False)
        seq_auto_draws = _seq_draws_ref(ds_ss, 502, 4, single_sample_downstream=None)
        d_a_pt = _assert_close(
            boot_auto["point_estimate_delta_mse"],
            seq_auto["comparisons_vs_covariate_ridge"]["paired_linear_ridge"]["delta_mse"],
            atol=1e-10,
            label="T2-02 single_sample_downstream=None point",
        )
        d_a_draws = _assert_close(
            boot_auto["bootstrap_draws_delta_mse"],
            seq_auto_draws,
            atol=1e-10,
            label="T2-02 single_sample_downstream=None draws",
        )
        ext_ds, _ = generate_synthetic_cohort(n_participants=48, n_single_sample=4, seed=503, compact_grid=True, subset_mode="single_sample")
        prep_false = PreparedExternalTest.from_dataset(ext_ds, cfg.primary_endpoint, single_sample_downstream=False, preset="primary")
        try:
            score_external_primary_pair(ds_ss, prep_false, cfg=cfg, single_sample_downstream=True)
            raise AssertionError("Expected ValueError on mismatched single_sample_downstream")
        except ValueError as exc:
            assert "single_sample_downstream" in str(exc)
        return max(d_f_pt, d_f_draws, d_a_pt, d_a_draws), "single_sample_downstream=False/None resampled draws & PreparedExternalTest mismatch guard verified"

    def _t2_03() -> tuple[float | None, str, bool]:
        if not cp_avail:
            return None, "CuPy/CUDA absent; skipped GPU decayed/un-decayed boundary", True
        boot_u = run_multiplicity_bootstrap(ds_undec, cfg=cfg, n_bootstraps=6, bootstrap_seed=504, use_gpu=True)
        boot_u_cpu = run_multiplicity_bootstrap(ds_undec, cfg=cfg, n_bootstraps=6, bootstrap_seed=504, use_gpu=False)
        d = _assert_close(boot_u["bootstrap_draws_delta_mse"], boot_u_cpu["bootstrap_draws_delta_mse"], atol=1e-11, label="T2-03 undecayed GPU")
        return d, "GPU un-decayed vs decayed boundary verified", False

    def _t2_05() -> tuple[float | None, str]:
        xp, meta = _backend.resolve_array_backend(True, 16, gpu_probe=lambda: None)
        assert xp is np and meta["gpu_fallback"] is True

        # threadpoolctl absent fallback
        boot_cpu_ref = run_multiplicity_bootstrap(ds_dec, cfg=cfg, n_bootstraps=2, bootstrap_seed=505, use_gpu=False)
        orig_tpc = _backend._get_threadpool_controller
        _backend._get_threadpool_controller = lambda: None  # type: ignore[assignment]
        try:
            with _backend._single_thread_blas():
                boot_no_tpc = run_multiplicity_bootstrap(ds_dec, cfg=cfg, n_bootstraps=2, bootstrap_seed=505, use_gpu=False)
        finally:
            _backend._get_threadpool_controller = orig_tpc  # type: ignore[assignment]
        max_d = _assert_close(
            boot_no_tpc["bootstrap_draws_delta_mse"],
            boot_cpu_ref["bootstrap_draws_delta_mse"],
            atol=1e-13,
            label="T2-05 threadpoolctl absent",
        )

        if cp_avail:
            orig_tri_b = _backend._cupyx_supports_batched_triangular
            orig_kern_b = _backend._get_fused_mfa_gram_kernel
            orig_tri_m = _batched_models._cupyx_supports_batched_triangular
            orig_kern_m = _batched_models._get_fused_mfa_gram_kernel
            orig_tri_e = _eigensolvers._cupyx_supports_batched_triangular
            boot_norm = run_multiplicity_bootstrap(ds_dec, cfg=cfg, n_bootstraps=4, bootstrap_seed=505, use_gpu=True)
            _backend._cupyx_supports_batched_triangular = lambda: False  # type: ignore[assignment]
            _backend._get_fused_mfa_gram_kernel = lambda: None  # type: ignore[assignment]
            _batched_models._cupyx_supports_batched_triangular = lambda: False  # type: ignore[assignment]
            _batched_models._get_fused_mfa_gram_kernel = lambda: None  # type: ignore[assignment]
            _eigensolvers._cupyx_supports_batched_triangular = lambda: False  # type: ignore[assignment]
            try:
                boot_fb = run_multiplicity_bootstrap(ds_dec, cfg=cfg, n_bootstraps=4, bootstrap_seed=505, use_gpu=True)
            finally:
                _backend._cupyx_supports_batched_triangular = orig_tri_b  # type: ignore[assignment]
                _backend._get_fused_mfa_gram_kernel = orig_kern_b  # type: ignore[assignment]
                _batched_models._cupyx_supports_batched_triangular = orig_tri_m  # type: ignore[assignment]
                _batched_models._get_fused_mfa_gram_kernel = orig_kern_m  # type: ignore[assignment]
                _eigensolvers._cupyx_supports_batched_triangular = orig_tri_e  # type: ignore[assignment]
            d_gpu = _assert_close(boot_fb["bootstrap_draws_delta_mse"], boot_norm["bootstrap_draws_delta_mse"], atol=1e-11, label="T2-05 GPU fallback")
            max_d = max(max_d, d_gpu)
        return max_d, "Backend/kernel/threadpoolctl fallback parity verified"

    def _t2_06() -> tuple[float | None, str]:
        from scipy import linalg as sp_linalg

        # +10.0 baseline offset & lam=NaN ValueError
        ds_off = replace(ds_dec, sample_1300=ds_dec.sample_1300 + 10.0, sample_2850=ds_dec.sample_2850 + 10.0)
        splits = make_nested_splits(ds_dec, cfg, cfg.primary_seed)
        tr_p, _ = splits[0][0]
        mult = np.ones(ds_dec.n_participants, dtype=float)
        sp_base = WeightedSpectralPreprocessor.fit(ds_dec, tr_p, mult[tr_p], cfg=cfg)
        sp_off = WeightedSpectralPreprocessor.fit(ds_off, tr_p, mult[tr_p], cfg=cfg)
        d_off = _assert_close(np.abs(sp_off.pca_components), np.abs(sp_base.pca_components), atol=1e-12, label="T2-06 +10 offset")
        pm = PairedMoments.from_sample_coordinates(
            sp_base.transform_pca(ds_dec.sample_1300, ds_dec.sample_2850, k=5), ds_dec, tr_p, mult[tr_p]
        )
        try:
            pm.solve_directions(5, float("nan"))
            raise AssertionError("Expected ValueError for lam=nan")
        except ValueError:
            pass

        # 1e-15 scaling in PairedMoments.solve_directions
        W_small = 1e-15 * np.eye(6, dtype=float)
        T_small = 2e-15 * np.eye(6, dtype=float)
        pm_small = PairedMoments(
            W_max=W_small, T_max=T_small, mean_ab_shift=np.zeros(6, dtype=float), n_paired_multiplicity=10.0, n_paired_distinct=10
        )
        ev_ref, _ = sp_linalg.eigh(T_small, W_small + 0.1 * (np.trace(T_small) / 6.0) * np.eye(6, dtype=float))
        _, ev_small, _ = pm_small.solve_directions(6, 0.1, eigenvalue_tolerance=0.0)
        d_pm = _assert_close(ev_small, ev_ref[::-1], atol=1e-12, rtol=1e-12, label="T2-06 1e-15 PairedMoments")

        # Rank-1 scale=1e4 batched_pls_sse_and_preds early stop (B=2 and B=8)
        rng_pls = np.random.default_rng(4405)
        n_tr_p, n_val_p, p_spec = 30, 10, 40
        u_tr = rng_pls.normal(size=(n_tr_p, 1))
        u_val = rng_pls.normal(size=(n_val_p, 1))
        v_dir = rng_pls.normal(size=(1, p_spec))
        X_big_tr = 1e4 * (u_tr @ v_dir)
        X_big_val = 1e4 * (u_val @ v_dir)
        y_noisy_tr = 2.0 * u_tr[:, 0] + 5.0 * rng_pls.normal(size=n_tr_p)
        y_noisy_val = 2.0 * u_val[:, 0] + 5.0 * rng_pls.normal(size=n_val_p)
        Xc0_tr = np.zeros((n_tr_p, 0), dtype=float)
        Xc0_val = np.zeros((n_val_p, 0), dtype=float)
        ref_big_p, _ = WeightedPLSResult.fit(Xc0_tr, X_big_tr, y_noisy_tr, np.ones(n_tr_p), max_components=5).predict(
            Xc0_val, X_big_val
        )
        d_pls = 0.0
        for b_sz in (2, 8):
            div_pls = {
                "w_tr": np.full((b_sz, n_tr_p), 1.0 / float(n_tr_p), dtype=np.float64),
                "w_val": np.ones((b_sz, n_val_p), dtype=np.float64),
                "y_tr": y_noisy_tr,
                "y_val": y_noisy_val,
                "X_cov_tr": np.zeros((b_sz, n_tr_p, 0), dtype=np.float64),
                "X_cov_val": np.zeros((b_sz, n_val_p, 0), dtype=np.float64),
                "scale1_sq": np.ones(b_sz, dtype=np.float64),
                "scale2_sq": np.zeros(b_sz, dtype=np.float64),
                "K_p1_tr": X_big_tr @ X_big_tr.T,
                "K_p1_vt": X_big_val @ X_big_tr.T,
                "K_p2_tr": np.zeros((n_tr_p, n_tr_p), dtype=np.float64),
                "K_p2_vt": np.zeros((n_val_p, n_tr_p), dtype=np.float64),
            }
            _, bp = _batched_models.batched_pls_sse_and_preds(np, div_pls, max_comp=5)
            d_pls = max(d_pls, _assert_close(bp[0], ref_big_p, atol=1e-10, rtol=1e-10, label=f"T2-06 rank-1 PLS B={b_sz}"))
            assert np.allclose(bp[0, :, 1:], bp[0, :, 1:2], atol=1e-12), f"Expected frozen rank-1 PLS preds for B={b_sz}"

        # Subnormal weighted-median ties (cdf > 0.0)
        cov_sub = pd.DataFrame(
            {
                "age_years": [10.0, 20.0, 30.0, np.nan, np.nan],
                "sex": ["female"] * 5,
                "statin": [0.0] * 5,
                "sample_label_trend": [0.0] * 5,
            }
        )
        M_tr_sub = np.array([[0.0, 5e-324, 5e-324, 1.0], [5e-324, 0.0, 5e-324, 1.0]], dtype=np.float64)
        M_val_sub = np.ones((2, 1), dtype=np.float64)
        dconst_sub = {
            "has_missing_div": True,
            "X_raw_tr": np.array([[10.0], [20.0], [30.0], [0.0]], dtype=np.float64),
            "X_raw_val": np.array([[0.0]], dtype=np.float64),
            "X_fin_tr": np.array([[True], [True], [True], [False]], dtype=bool),
            "X_fin_val": np.array([[False]], dtype=bool),
            "cov_sort_ord": np.array([[0, 1, 2, 3]], dtype=np.int64),
            "cov_sort_vals": np.array([[10.0, 20.0, 30.0, 0.0]], dtype=np.float64),
            "cov_sort_fin": np.array([[1.0, 1.0, 1.0, 0.0]], dtype=np.float64),
            "X_miss_tr": np.array([[0.0], [0.0], [0.0], [1.0]], dtype=np.float64),
            "X_miss_val": np.array([[1.0]], dtype=np.float64),
        }
        _, _, _, X_tr_sub, X_val_sub = _batched_models._prep_covariates_batch(np, dconst_sub, M_tr_sub, M_val_sub)
        d_sub = 0.0
        for b_idx in (0, 1):
            cp_ref = CovariatePreprocessor.fit(cov_sub, np.arange(4, dtype=int), M_tr_sub[b_idx], preset="primary")
            d_sub = max(
                d_sub,
                _assert_close(X_tr_sub[b_idx], cp_ref.transform(cov_sub, np.arange(4, dtype=int)), atol=1e-12, label=f"T2-06 subnormal tr b={b_idx}"),
                _assert_close(X_val_sub[b_idx], cp_ref.transform(cov_sub, np.array([4], dtype=int)), atol=1e-12, label=f"T2-06 subnormal val b={b_idx}"),
            )

        # Corrupted Schur slice isolation as NaN
        gcache = _gram_cache._get_dataset_gram_cache(ds_dec, cfg)
        dconst = _gram_cache._precompute_division_constants(np, gcache, ds_dec, tr_p, splits[0][0][1], cfg)
        M_two = np.vstack([mult[None, :], mult[None, :]])
        div_real = _batched_models.batched_division_prep(
            np, gcache, dconst, M_two[:, tr_p], M_two[:, splits[0][0][1]], ds_dec.outcomes[cfg.primary_endpoint], np.asarray(cfg.ridge_alpha_grid), cfg
        )
        div_real["Schur_p"][1] = np.nan
        sse_schur = _batched_models.batched_pca_schur_sse(np, div_real, list(cfg.pca_dimensions), np.asarray(cfg.ridge_alpha_grid))
        assert np.all(np.isfinite(sse_schur[0])) and np.all(np.isnan(sse_schur[1])), "Corrupted Schur slice must isolate as NaN"
        return max(d_off, d_pm, d_pls, d_sub), "+10 offset, 1e-15 PairedMoments, rank-1 PLS stop, subnormal median, & Schur NaN isolation verified"

    def _t3_02() -> tuple[float | None, str]:
        full_cal = evaluate_bootstrap_calibration(
            n_repetitions=4, n_bootstraps=4, n_participants=32, n_test_participants=80, base_seed=710, n_jobs=2, use_gpu=False, compact_grid=True
        )
        ckpt2 = tmp_dir / "cal_ckpt_resume.npz"
        orig_save = calibration._save_calibration_checkpoint
        save_calls = {"n": 0}

        def _interrupt_after_chunk0(path: Path, state: Any) -> None:
            save_calls["n"] += 1
            if save_calls["n"] == 2:
                raise RuntimeError("simulated checkpoint interrupt")
            orig_save(path, state)

        calibration._save_calibration_checkpoint = _interrupt_after_chunk0  # type: ignore[assignment]
        try:
            try:
                evaluate_bootstrap_calibration(
                    n_repetitions=4, n_bootstraps=4, n_participants=32, n_test_participants=80, base_seed=710, n_jobs=2, use_gpu=False, compact_grid=True, checkpoint_path=ckpt2, checkpoint_chunk_size=2
                )
                raise AssertionError("Expected simulated checkpoint interrupt")
            except RuntimeError as exc:
                assert "simulated checkpoint interrupt" in str(exc)
        finally:
            calibration._save_calibration_checkpoint = orig_save  # type: ignore[assignment]

        with np.load(ckpt2, allow_pickle=False) as z_mid:
            assert z_mid["completed"].tolist() == [True, True, False, False]
            assert np.all(np.isnan(z_mid["point_delta_mse"][2:]))

        resumed = evaluate_bootstrap_calibration(
            n_repetitions=4, n_bootstraps=4, n_participants=32, n_test_participants=80, base_seed=710, n_jobs=2, use_gpu=False, compact_grid=True, checkpoint_path=ckpt2, checkpoint_chunk_size=2
        )
        max_d = 0.0
        for key in (
            "target_delta_mse",
            "target_test_cohort_se",
            "target_total_se",
            "mean_point_estimate",
            "unconditional_coverage",
            "target_delta_vs_comparator",
        ):
            max_d = max(max_d, _assert_close(resumed[key], full_cal[key], atol=1e-13, label=f"T3-02 cal {key}"))
        try:
            evaluate_bootstrap_calibration(
                n_repetitions=4, n_bootstraps=6, n_participants=32, n_test_participants=80, base_seed=710, n_jobs=2, use_gpu=False, compact_grid=True, checkpoint_path=ckpt2
            )
            raise AssertionError("Expected ValueError on checkpoint fingerprint mismatch")
        except ValueError:
            pass

        # evaluate_mde_signal_curve interrupt/resume + mismatch ValueError
        mde_ref = evaluate_mde_signal_curve(
            signal_grid=(0.0, 0.5), n_repetitions=4, n_bootstraps=4, n_participants=32, n_test_participants=80, base_seed=720, n_jobs=2, use_gpu=False, compact_grid=True
        )
        mde_dir = tmp_dir / "mde_ckpt_dir"
        mde_calls = {"n": 0}

        def _interrupt_mde_4th(path: Path, state: Any) -> None:
            mde_calls["n"] += 1
            if mde_calls["n"] == 4:
                raise RuntimeError("simulated mde interrupt")
            orig_save(path, state)

        calibration._save_calibration_checkpoint = _interrupt_mde_4th  # type: ignore[assignment]
        try:
            try:
                evaluate_mde_signal_curve(
                    signal_grid=(0.0, 0.5), n_repetitions=4, n_bootstraps=4, n_participants=32, n_test_participants=80, base_seed=720, n_jobs=2, use_gpu=False, compact_grid=True, checkpoint_dir=mde_dir, checkpoint_chunk_size=2
                )
                raise AssertionError("Expected simulated mde interrupt")
            except RuntimeError as exc:
                assert "simulated mde interrupt" in str(exc)
        finally:
            calibration._save_calibration_checkpoint = orig_save  # type: ignore[assignment]

        with np.load(mde_dir / "mde_00_sig_0p0.npz", allow_pickle=False) as z0:
            assert z0["completed"].tolist() == [True, True, True, True]
        with np.load(mde_dir / "mde_01_sig_0p5.npz", allow_pickle=False) as z1:
            assert z1["completed"].tolist() == [True, True, False, False]

        mde_res = evaluate_mde_signal_curve(
            signal_grid=(0.0, 0.5), n_repetitions=4, n_bootstraps=4, n_participants=32, n_test_participants=80, base_seed=720, n_jobs=2, use_gpu=False, compact_grid=True, checkpoint_dir=mde_dir, checkpoint_chunk_size=2
        )
        for col in ("target_delta_mse", "target_total_se", "mean_delta_mse", "unconditional_coverage"):
            max_d = max(max_d, _assert_close(mde_res[col], mde_ref[col], atol=1e-13, label=f"T3-02 mde {col}"))
        try:
            evaluate_mde_signal_curve(
                signal_grid=(0.0, 0.5), n_repetitions=4, n_bootstraps=6, n_participants=32, n_test_participants=80, base_seed=720, n_jobs=2, use_gpu=False, compact_grid=True, checkpoint_dir=mde_dir, checkpoint_chunk_size=2
            )
            raise AssertionError("Expected ValueError on MDE checkpoint fingerprint mismatch")
        except ValueError:
            pass
        return max_d, "Calibration & MDE curve mid-run interrupt/resume + mismatch guards verified"

    results.append(_run_check("T1-A1", "paired_lipid", "tier1", _t1_a1))
    results.append(_run_check("T1-A2", "paired_lipid", "tier1", _t1_a2))
    results.append(_run_check("T1-A3", "paired_lipid", "tier1", _t1_a3))
    results.append(_run_check("T1-A4", "paired_lipid", "tier1", _t1_a4))
    results.append(_run_check("T1-A5", "paired_lipid", "tier1", _t1_a5))
    results.append(_run_check("T1-A6", "paired_lipid", "tier1", _t1_a6))
    results.append(_run_check("T1-B1", "paired_lipid", "tier1", _t1_b1))
    results.append(_run_check("T1-B2", "paired_lipid", "tier1", _t1_b2))
    results.append(_run_check("T1-B3", "paired_lipid", "tier1", _t1_b3))
    results.append(_run_check("T1-B4", "paired_lipid", "tier1", _t1_b4))
    results.append(_run_check("T1-B5", "paired_lipid", "tier1", _t1_b5))
    results.append(_run_check("T1-C1", "paired_lipid", "tier1", _t1_c1))
    results.append(_run_check("T2-01", "paired_lipid", "tier2", _t2_01))
    results.append(_run_check("T2-02", "paired_lipid", "tier2", _t2_02))
    results.append(_run_check("T2-03", "paired_lipid", "tier2", _t2_03))
    results.append(_run_check("T2-05", "paired_lipid", "tier2", _t2_05))
    results.append(_run_check("T2-06", "paired_lipid", "tier2", _t2_06))
    results.append(_run_check("T3-02", "paired_lipid", "tier3", _t3_02))
    return results


# ---------------------------------------------------------------------------
# Suite 2: manuscript_ml (Areas C, I, T3-01, T3-03)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _isolated_manuscript_data_paths(base_dir: Path) -> Iterator[Path]:
    """Isolate clinical_master_path and preprocessed_dir from ROOT/data/ during synthetic runs."""
    import src

    iso_dir = base_dir / "_isolated_absent_data"
    data_cfg = src.config.setdefault("data", {})
    had_key = "preprocessed_dir" in data_cfg
    prev_val = data_cfg.get("preprocessed_dir")
    data_cfg["preprocessed_dir"] = str(iso_dir / "preprocessed")
    try:
        yield iso_dir / "absent_clinical_master.csv"
    finally:
        if had_key:
            data_cfg["preprocessed_dir"] = prev_val
        else:
            data_cfg.pop("preprocessed_dir", None)


def _run_manuscript_ml_checks(*, mode: str, profile: str, tmp_dir: Path) -> list[ParityCheckResult]:
    from manuscript_ml import (
        clear_dataset_cache,
        load_manuscript_config,
        require_cohort_approval,
        resolve_experiment_dataset,
        write_run_provenance,
    )
    from manuscript_ml.audit_readiness import _audit_clinical_dictionary
    from manuscript_ml.build_manuscript_artifacts import build_manuscript_artifacts
    from manuscript_ml.readiness import evaluate_cohort_readiness
    from src.analysis.paired_lipid.cohort import COVARIATE_PRESETS, PairedLipidConfig
    from src.analysis.paired_lipid.nested_cv import evaluate_nested_cohort, run_multiplicity_bootstrap
    from src.analysis.paired_lipid.synthetic import generate_synthetic_cohort

    results: list[ParityCheckResult] = []
    compact = profile == "smoke"

    if mode == "smoke":
        def _smoke_mml() -> tuple[float | None, str]:
            # Safety gate: unapproved cohort access raises PermissionError
            try:
                require_cohort_approval(use_synthetic=False, approve_cohort_evaluation=False)
                raise AssertionError("Expected PermissionError for unapproved cohort access")
            except PermissionError:
                pass
            # Subset derivation parity (T1-C4 / T1-C5)
            clear_dataset_cache()
            ds_p, _ = resolve_experiment_dataset(use_synthetic=True, subset_mode="paired_only", synthetic_participants=32, synthetic_seed=17, compact_grid=True)
            ref_p, _ = generate_synthetic_cohort(n_participants=32, seed=17, compact_grid=True, subset_mode="paired_only")
            d1 = _assert_close(ds_p.sample_1300, ref_p.sample_1300, atol=0.0, label="smoke paired_only")
            # Readiness audit & clinical dictionary (T1-I2)
            m_csv = tmp_dir / "syn_master.csv"
            d_csv = tmp_dir / "syn_dict.csv"
            m_csv.write_text("participant_key,cohort,spectral_subject_id\np1,stroke,7\n", encoding="utf-8")
            d_csv.write_text("column\nparticipant_key\ncohort\nspectral_subject_id\n", encoding="utf-8")
            d_audit = _audit_clinical_dictionary(m_csv, d_csv)
            assert d_audit["columns_match_exact"] is True
            r_ok = evaluate_cohort_readiness(
                {"stroke_identity_verified": 92},
                {"n_eligible_participants": 92, "n_paired_participants": 88, "n_single_sample_participants": 4, "n_eligible_samples": 180},
                {"expected_identity_verified": 92, "expected_cohort_join": {"n_eligible_participants": 92, "n_paired_participants": 88, "n_single_sample_participants": 4, "n_eligible_samples": 180}},
            )
            assert r_ok["ready_for_cohort_evaluation"] is True
            # Preview gate check (T1-I3)
            preview_in = tmp_dir / "preview_in"
            preview_in.mkdir(parents=True, exist_ok=True)
            write_csv_atomic(
                pd.DataFrame(
                    [
                        {
                            "comparison": "paired_linear_ridge_vs_covariate_ridge",
                            "model": "paired_linear_ridge",
                            "delta_mse": 0.1,
                            "oof_loss_diff_sd": 0.05,
                            "relative_mse_reduction": 0.05,
                            "mse_covariate": 1.0,
                            "mse_model": 0.9,
                            "rmse_model": 0.95,
                            "mae_model": 0.7,
                            "calibration_intercept": 0.0,
                            "calibration_slope": 1.0,
                        }
                    ]
                ),
                preview_in / "comparator_errors_and_differences.csv",
                index=False,
            )
            write_json_atomic(preview_in / "eligibility_and_split_manifest.json", {"data_mode": "synthetic"})
            try:
                build_manuscript_artifacts(input_dir=preview_in, output_dir=tmp_dir / "preview_out", allow_synthetic_preview=False)
                raise AssertionError("Expected PermissionError for preview without flag")
            except PermissionError:
                pass
            art_out = build_manuscript_artifacts(input_dir=preview_in, output_dir=tmp_dir / "preview_out", allow_synthetic_preview=True)
            man = json.loads(art_out["artifact_build_manifest"].read_text(encoding="utf-8"))
            assert man["synthetic_preview"] is True
            return d1, "Safety gates, subset parity, readiness audit, and preview watermark verified"

        results.append(_run_check("SMOKE-MML", "manuscript_ml", "smoke", _smoke_mml))
        return results

    from manuscript_ml.run_all import run_all_pipeline
    from manuscript_ml.run_primary_evaluation import run_primary_evaluation
    from manuscript_ml.run_sensitivity_and_secondary import run_sensitivity_and_secondary

    def _t1_c2() -> tuple[float | None, str]:
        sens_dir = tmp_dir / "t1_c2_sens"
        n_boot = 4 if compact else 8
        out = run_sensitivity_and_secondary(
            use_synthetic=True, output_dir=sens_dir, compact_grid=True, n_bootstraps=n_boot, use_gpu=False
        )
        sens_df = pd.read_csv(out["covariate_and_subset_sensitivity"])
        cfg_base, _ = load_manuscript_config()
        ds_all, _ = generate_synthetic_cohort(n_participants=92, seed=42, compact_grid=True, subset_mode="all_eligible", cfg=cfg_base)
        max_d = 0.0
        for preset in COVARIATE_PRESETS:
            cfg_p = replace(cfg_base, covariate_preset=preset)
            ref_eval = evaluate_nested_cohort(ds_all, cfg=cfg_p, compute_diagnostics=False)
            ref_boot = run_multiplicity_bootstrap(ds_all, cfg=cfg_p, n_bootstraps=n_boot, use_gpu=False)
            row = sens_df.loc[(sens_df["sensitivity_axis"] == "covariate_preset") & (sens_df["setting"] == preset)].iloc[0]
            max_d = max(
                max_d,
                _assert_close(row["delta_mse_paired"], ref_eval["comparisons_vs_covariate_ridge"]["paired_linear_ridge"]["delta_mse"], atol=1e-12, label=f"T1-C2 {preset} pt"),
                _assert_close(row["ci_95_delta_mse_low"], ref_boot["ci_95_delta_mse"][0], atol=1e-12, label=f"T1-C2 {preset} lo"),
                _assert_close(row["ci_95_delta_mse_high"], ref_boot["ci_95_delta_mse"][1], atol=1e-12, label=f"T1-C2 {preset} hi"),
            )
            assert bool(row["interval_procedure_succeeded"]) == bool(ref_boot["interval_procedure_succeeded"])
        return max_d, f"Sensitivity preset sharing matches independent per-preset runs (B={n_boot})"

    def _t1_c3() -> tuple[float | None, str]:
        d_all = tmp_dir / "t1_c3_all"
        d_sep = tmp_dir / "t1_c3_sep"
        with _isolated_manuscript_data_paths(d_all) as absent_master:
            out_all = run_all_pipeline(
                use_synthetic=True,
                clinical_master_path=absent_master,
                quick=True,
                output_dir=d_all,
                n_bootstraps=4,
                sensitivity_bootstraps=0,
                n_scenario_reps=1,
                n_calibration_reps=1,
                n_participants=40,
                n_test_participants=80,
                scenario_names=["linear_biological_signal"],
                skip_scenario_suite=True,
                n_jobs=(1 if compact else -1),
                use_gpu=False,
            )
            out_p = run_primary_evaluation(
                use_synthetic=True,
                clinical_master_path=absent_master,
                output_dir=d_sep,
                compact_grid=True,
                n_bootstraps=4,
                use_gpu=False,
            )
            out_s = run_sensitivity_and_secondary(
                use_synthetic=True,
                clinical_master_path=absent_master,
                output_dir=d_sep,
                compact_grid=True,
                n_bootstraps=0,
                use_gpu=False,
            )
        d1 = _assert_close(
            pd.read_csv(out_all["comparator_errors_and_differences"])["delta_mse"],
            pd.read_csv(out_p["comparator_errors_and_differences"])["delta_mse"],
            atol=1e-13,
            label="T1-C3 primary",
        )
        d2 = _assert_close(
            pd.read_csv(out_all["covariate_and_subset_sensitivity"])["delta_mse_paired"],
            pd.read_csv(out_s["covariate_and_subset_sensitivity"])["delta_mse_paired"],
            atol=1e-13,
            label="T1-C3 sensitivity",
        )
        return max(d1, d2), "run_all_pipeline matches standalone primary + sensitivity stages"

    def _t1_c4() -> tuple[float | None, str]:
        clear_dataset_cache()
        ds_p, meta_p = resolve_experiment_dataset(use_synthetic=True, subset_mode="paired_only", synthetic_participants=48, synthetic_seed=17, compact_grid=True)
        ref_p, ref_meta = generate_synthetic_cohort(n_participants=48, seed=17, compact_grid=True, subset_mode="paired_only")
        d1 = _assert_close(ds_p.sample_1300, ref_p.sample_1300, atol=0.0, label="T1-C4 s1300")
        d2 = _assert_close(ds_p.sample_2850, ref_p.sample_2850, atol=0.0, label="T1-C4 s2850")
        assert set(ds_p.outcomes) == set(ref_p.outcomes)
        d3 = max(
            _assert_close(ds_p.outcomes[k], ref_p.outcomes[k], atol=0.0, label=f"T1-C4 {k}")
            for k in ref_p.outcomes
        )
        assert np.array_equal(ds_p.participant_ids, ref_p.participant_ids)
        assert np.array_equal(ds_p.wavenumbers_1300, ref_p.wavenumbers_1300)
        assert np.array_equal(ds_p.wavenumbers_2850, ref_p.wavenumbers_2850)
        assert np.array_equal(ds_p.sample_participant_idx, ref_p.sample_participant_idx)
        assert np.array_equal(ds_p.sample_ids, ref_p.sample_ids)
        assert np.array_equal(ds_p.sample_parsed_labels, ref_p.sample_parsed_labels, equal_nan=True)
        assert np.array_equal(ds_p.participant_sample_counts, ref_p.participant_sample_counts)
        assert np.array_equal(ds_p.paired_participant_indices, ref_p.paired_participant_indices)
        assert np.array_equal(ds_p.pair_sample_a_indices, ref_p.pair_sample_a_indices)
        assert np.array_equal(ds_p.pair_sample_b_indices, ref_p.pair_sample_b_indices)
        assert ds_p.covariate_frame.equals(ref_p.covariate_frame)
        assert ds_p.manifest_summary == ref_p.manifest_summary
        assert ds_p.exclusions == ref_p.exclusions
        assert ds_p.n_participants == ref_p.n_participants
        assert ds_p.n_samples == ref_p.n_samples
        assert meta_p["signal_strength"] == ref_meta["signal_strength"]
        return max(d1, d2, d3), "paired_only derivation exact match"

    def _t1_c5() -> tuple[float | None, str]:
        clear_dataset_cache()
        ds_ss, _ = resolve_experiment_dataset(use_synthetic=True, subset_mode="single_sample", synthetic_participants=48, synthetic_seed=17, compact_grid=True)
        ref_ss, _ = generate_synthetic_cohort(n_participants=48, seed=17, compact_grid=True, subset_mode="single_sample")
        d1 = _assert_close(ds_ss.sample_1300, ref_ss.sample_1300, atol=0.0, label="T1-C5 s1300")
        d2 = _assert_close(ds_ss.sample_2850, ref_ss.sample_2850, atol=0.0, label="T1-C5 s2850")
        assert set(ds_ss.outcomes) == set(ref_ss.outcomes)
        d3 = max(
            _assert_close(ds_ss.outcomes[k], ref_ss.outcomes[k], atol=0.0, label=f"T1-C5 {k}")
            for k in ref_ss.outcomes
        )
        assert np.array_equal(ds_ss.participant_ids, ref_ss.participant_ids)
        assert np.array_equal(ds_ss.sample_participant_idx, ref_ss.sample_participant_idx)
        assert np.array_equal(ds_ss.sample_ids, ref_ss.sample_ids)
        assert np.array_equal(ds_ss.participant_sample_counts, ref_ss.participant_sample_counts)
        assert np.array_equal(ds_ss.paired_participant_indices, ref_ss.paired_participant_indices)
        assert ds_ss.covariate_frame.equals(ref_ss.covariate_frame)
        assert ds_ss.manifest_summary == ref_ss.manifest_summary
        assert ds_ss.exclusions == ref_ss.exclusions
        ds_3m, _ = generate_synthetic_cohort(n_participants=48, seed=17, compact_grid=True, map_dropout_fraction=0.2, subset_mode="three_map_complete", cfg=PairedLipidConfig(min_maps_per_block=3))
        assert isinstance(ds_3m.exclusions, list) and len(ds_3m.exclusions) > 0
        assert ds_3m.manifest_summary["min_maps_per_block"] == 3
        return max(d1, d2, d3), "single_sample derivation & three_map_complete min_maps=3 isolation verified"

    def _t1_i1() -> tuple[float | None, str]:
        try:
            require_cohort_approval(use_synthetic=False, approve_cohort_evaluation=False)
            raise AssertionError("Expected PermissionError")
        except PermissionError:
            pass
        return 0.0, "require_cohort_approval blocks unapproved cohort run"

    def _t1_i2() -> tuple[float | None, str]:
        m_csv = tmp_dir / "t1_i2_master.csv"
        d_csv = tmp_dir / "t1_i2_dict.csv"
        m_csv.write_text("participant_key,cohort,spectral_subject_id\np1,stroke,7\n", encoding="utf-8")
        d_csv.write_text("column\nparticipant_key\ncohort\nspectral_subject_id\n", encoding="utf-8")
        audit_ok = _audit_clinical_dictionary(m_csv, d_csv)
        assert audit_ok["columns_match_exact"] is True
        d_csv.write_text("column\nparticipant_key\ncohort\n", encoding="utf-8")
        audit_bad = _audit_clinical_dictionary(m_csv, d_csv)
        assert audit_bad["columns_match_exact"] is False
        r_bad = evaluate_cohort_readiness(
            {"stroke_identity_verified": 90},
            {"n_eligible_participants": 92},
            {"expected_identity_verified": 92, "expected_cohort_join": {"n_eligible_participants": 92}},
        )
        assert r_bad["ready_for_cohort_evaluation"] is False
        return 0.0, "Readiness & dictionary count/column audit verified"

    def _t1_i3() -> tuple[float | None, str]:
        p_in = tmp_dir / "t1_i3_in"
        p_in.mkdir(parents=True, exist_ok=True)
        write_csv_atomic(
            pd.DataFrame(
                [
                    {
                        "comparison": "paired_linear_ridge_vs_covariate_ridge",
                        "model": "paired_linear_ridge",
                        "delta_mse": 0.1,
                        "oof_loss_diff_sd": 0.05,
                        "relative_mse_reduction": 0.05,
                        "mse_covariate": 1.0,
                        "mse_model": 0.9,
                        "rmse_model": 0.95,
                        "mae_model": 0.7,
                        "calibration_intercept": 0.0,
                        "calibration_slope": 1.0,
                    }
                ]
            ),
            p_in / "comparator_errors_and_differences.csv",
            index=False,
        )
        write_json_atomic(p_in / "eligibility_and_split_manifest.json", {"data_mode": "synthetic"})
        try:
            build_manuscript_artifacts(input_dir=p_in, output_dir=tmp_dir / "t1_i3_out", allow_synthetic_preview=False)
            raise AssertionError("Expected PermissionError")
        except PermissionError:
            pass
        out = build_manuscript_artifacts(input_dir=p_in, output_dir=tmp_dir / "t1_i3_out", allow_synthetic_preview=True)
        man = json.loads(out["artifact_build_manifest"].read_text(encoding="utf-8"))
        assert man["synthetic_preview"] is True
        return 0.0, "Artifact preview watermark gate verified"

    def _t1_i4() -> tuple[float | None, str]:
        io_dir = tmp_dir / "t1_i4_io"
        io_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({"a": [1, 2, 3], "b": [1.5, 2.5, 3.5]})
        arr = np.arange(6, dtype=float)

        def _worker(tid: int) -> None:
            for it in range(10):
                write_csv_atomic(df, io_dir / "shared.csv", index=False)
                write_parquet_atomic(df, io_dir / "shared.parquet", index=False)
                write_json_atomic(io_dir / "shared.json", {"tid": tid, "it": it})
                write_npz_atomic(io_dir / "shared.npz", {"x": arr})
                write_text_atomic(io_dir / "shared.txt", f"tid={tid};it={it}\n")

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futs = [pool.submit(_worker, t) for t in range(4)]
            for fut in futs:
                fut.result()
        assert not list(io_dir.glob("*.tmp"))
        assert len(pd.read_csv(io_dir / "shared.csv")) == 3
        assert len(pd.read_parquet(io_dir / "shared.parquet")) == 3
        json.loads((io_dir / "shared.json").read_text(encoding="utf-8"))
        with np.load(io_dir / "shared.npz") as z:
            assert z["x"].shape == (6,)
        return 0.0, "4-thread concurrent Windows atomic writes verified across 5 formats"

    def _t1_i5() -> tuple[float | None, str]:
        cfg, raw = load_manuscript_config()
        assert cfg.primary_endpoint == "ldl_mmol_l"
        bad_yaml = tmp_dir / "bad_cfg.yaml"
        bad_yaml.write_text("paired_lipid:\n  unknown_setting: 123\n", encoding="utf-8")
        try:
            load_manuscript_config(bad_yaml)
            raise AssertionError("Expected ValueError or TypeError for unknown PairedLipidConfig key")
        except (ValueError, TypeError):
            pass
        bad_val_yaml = tmp_dir / "bad_val_cfg.yaml"
        bad_val_yaml.write_text(
            "paired_lipid:\n  primary_endpoint: ldl_mmol_l\ncalibration:\n  interval_method: invalid_method\n",
            encoding="utf-8",
        )
        try:
            load_manuscript_config(bad_val_yaml)
            raise AssertionError("Expected ValueError for invalid calibration.interval_method")
        except ValueError:
            pass
        bad_seeds_yaml = tmp_dir / "bad_seeds_cfg.yaml"
        bad_seeds_yaml.write_text(
            "paired_lipid:\n  primary_endpoint: ldl_mmol_l\nseeds:\n  bootstrap_seed: true\n",
            encoding="utf-8",
        )
        try:
            load_manuscript_config(bad_seeds_yaml)
            raise AssertionError("Expected ValueError for boolean seed entry")
        except ValueError:
            pass
        prov_path = write_run_provenance(
            tmp_dir / "t1_i5_prov", "test_stage", config_path=None, cfg=cfg, settings={"quick": True}
        )
        prov = json.loads(prov_path.read_text(encoding="utf-8"))
        assert "config_sha256" in prov and "resolved_paired_lipid_config" in prov and "frozen_spectral_grid_bounds" in prov
        _ = raw
        return 0.0, "Config validation & provenance schema verified"

    def _t3_01() -> tuple[float | None, str]:
        run_dir = tmp_dir / "t3_01_run"
        art_dir = tmp_dir / "t3_01_art"
        with _isolated_manuscript_data_paths(run_dir) as absent_master:
            out = run_all_pipeline(
                use_synthetic=True,
                clinical_master_path=absent_master,
                quick=True,
                output_dir=run_dir,
                n_bootstraps=4,
                sensitivity_bootstraps=4,
                n_scenario_reps=1,
                n_calibration_reps=2,
                n_participants=32,
                n_test_participants=80,
                scenario_names=["linear_biological_signal"],
                run_mde_curve=False,
                n_jobs=(1 if compact else -1),
                use_gpu=False,
            )
        man = json.loads(out["eligibility_and_split_manifest"].read_text(encoding="utf-8"))
        assert man["multiplicity_bootstrap"]["parity_within_1e6"] is True
        art = build_manuscript_artifacts(input_dir=run_dir, output_dir=art_dir, allow_synthetic_preview=True)
        b_man = json.loads(art["artifact_build_manifest"].read_text(encoding="utf-8"))
        assert b_man["synthetic_preview"] is True
        assert len(b_man["input_sha256"]) >= 4
        return 0.0, "Full synthetic run_all + build_manuscript_artifacts chain verified"

    def _t3_03() -> tuple[float | None, str]:
        s_dir = tmp_dir / "t3_03_sens"
        out = run_sensitivity_and_secondary(use_synthetic=True, output_dir=s_dir, compact_grid=True, n_bootstraps=4, use_gpu=False)
        for key in ("covariate_and_subset_sensitivity", "outer_partition_seed_sensitivity", "secondary_endpoints_summary", "anchor_associations_continuous"):
            df = pd.read_csv(out[key])
            assert not df.empty, f"Empty output {key}"
        return 0.0, "Multi-axis sensitivity, secondary endpoints, and anchor summary verified"

    results.append(_run_check("T1-C2", "manuscript_ml", "tier1", _t1_c2))
    results.append(_run_check("T1-C3", "manuscript_ml", "tier1", _t1_c3))
    results.append(_run_check("T1-C4", "manuscript_ml", "tier1", _t1_c4))
    results.append(_run_check("T1-C5", "manuscript_ml", "tier1", _t1_c5))
    results.append(_run_check("T1-I1", "manuscript_ml", "tier1", _t1_i1))
    results.append(_run_check("T1-I2", "manuscript_ml", "tier1", _t1_i2))
    results.append(_run_check("T1-I3", "manuscript_ml", "tier1", _t1_i3))
    results.append(_run_check("T1-I4", "manuscript_ml", "tier1", _t1_i4))
    results.append(_run_check("T1-I5", "manuscript_ml", "tier1", _t1_i5))
    results.append(_run_check("T3-01", "manuscript_ml", "tier3", _t3_01))
    results.append(_run_check("T3-03", "manuscript_ml", "tier3", _t3_03))
    return results


# ---------------------------------------------------------------------------
# Suite 3: biological_pca & joint_dr (Area D)
# ---------------------------------------------------------------------------


def _run_biological_pca_checks(*, mode: str, profile: str) -> list[ParityCheckResult]:
    from src.analysis.biological_pca import (
        BLOCKS,
        fit_biological_pca_models,
        run_measurement_count_sensitivity,
        summarize_measurement_count_sensitivity,
    )
    from src.analysis.joint_dr import joint_pca_mfa
    from src.analysis.statistics import run_pca

    results: list[ParityCheckResult] = []
    compact = profile == "smoke"

    if mode == "smoke":
        def _smoke_bpca() -> tuple[float | None, str]:
            inputs = _make_synthetic_subject_inputs(n_control=6, n_disease=8, p_1300=40, p_2850=30, seed=11)
            cfg = _make_eda_cfg(n_pca_components=3, n_draws=3)
            res = fit_biological_pca_models(inputs, cfg=cfg)
            df = run_measurement_count_sensitivity(inputs, res, cfg=cfg)
            summ = summarize_measurement_count_sensitivity(df)
            assert len(summ) == 9
            # Verify draw 0 against independent PCA(svd_solver='full') reference
            rng_ref = np.random.default_rng(42)
            m1300 = np.vstack(
                [
                    np.nanmedian(
                        inputs.measurement_rows[s]["1300"][
                            rng_ref.choice(inputs.measurement_rows[s]["1300"].shape[0], size=3, replace=False)
                        ]
                        if inputs.measurement_rows[s]["1300"].shape[0] > 3
                        else inputs.measurement_rows[s]["1300"],
                        axis=0,
                    )
                    for s in inputs.subjects
                ]
            )
            ref_1330 = _reference_pca_full(m1300, n_components=3, scale=False)
            obs_1330 = df.loc[(df["draw"] == 0) & (df["model"] == "PCA_1330"), "explained_variance_ratio"].to_numpy()
            d = _assert_close(obs_1330, ref_1330["explained_variance_ratio"], atol=1e-12, label="smoke bio_pca")
            return d, "biological_pca & joint_dr smoke parity verified"

        results.append(_run_check("SMOKE-BPCA", "biological_pca", "smoke", _smoke_bpca))
        return results

    inputs = _make_synthetic_subject_inputs(
        n_control=10 if compact else 30,
        n_disease=16 if compact else 62,
        p_1300=60 if compact else 200,
        p_2850=50 if compact else 150,
        seed=42,
    )
    cfg = _make_eda_cfg(n_pca_components=3, n_draws=6 if compact else 20)
    canonical_results = fit_biological_pca_models(inputs, cfg=cfg)

    def _t1_d1() -> tuple[float | None, str]:
        mat = np.hstack([inputs.matrices["1300"], inputs.matrices["2850"]])
        split_idx = inputs.axes["1300"].size
        got = joint_pca_mfa(mat, split_idx, 3)
        ref = _reference_joint_pca_mfa(mat, split_idx, 3)
        d1 = _assert_close(got["explained_variance_ratio"], ref["explained_variance_ratio"], atol=1e-12, label="T1-D1 evr")
        d2 = _assert_close(got["loadings"], ref["loadings"], atol=1e-12, label="T1-D1 loadings")
        d3 = _assert_close(got["scores"], ref["scores"], atol=1e-12, label="T1-D1 scores")
        return max(d1, d2, d3), "joint_pca_mfa matches PCA(svd_solver='full') reference"

    def _t1_d2() -> tuple[float | None, str]:
        max_d = 0.0
        for scale_flag in (False, True):
            got = run_pca(inputs.matrices["1300"], n_components=3, scale=scale_flag, log_summary=False, warn_on_high_pc1=False)
            ref = _reference_pca_full(inputs.matrices["1300"], n_components=3, scale=scale_flag)
            max_d = max(
                max_d,
                _assert_close(got["explained_variance_ratio"], ref["explained_variance_ratio"], atol=1e-12, label="T1-D2 evr"),
                _assert_close(got["loadings"], ref["loadings"], atol=1e-12, label="T1-D2 loadings"),
            )
        return max_d, "run_pca matches PCA(svd_solver='full') for scale=False/True"

    def _t1_d3() -> tuple[float | None, str]:
        draws_df = run_measurement_count_sensitivity(inputs, canonical_results, cfg=cfg)
        n_draws = int(cfg["biological_pca"]["measurement_count_sensitivity"]["n_draws"])
        rng_ref = np.random.default_rng(42)
        target_n = 3
        can_map = {r.name: r for r in canonical_results}
        max_d = 0.0
        for draw in range(n_draws):
            draw_mats: dict[str, np.ndarray] = {}
            for block in BLOCKS:
                rows_b = []
                for sid in inputs.subjects:
                    arr = inputs.measurement_rows[sid][block]
                    if arr.shape[0] <= target_n:
                        sel = arr
                    else:
                        idx = rng_ref.choice(arr.shape[0], size=target_n, replace=False)
                        sel = arr[idx]
                    rows_b.append(np.nanmedian(sel, axis=0))
                draw_mats[block] = np.vstack(rows_b)
            ref_whole = _reference_joint_pca_mfa(np.hstack([draw_mats["1300"], draw_mats["2850"]]), inputs.axes["1300"].size, 3)
            ref_1330 = _reference_pca_full(draw_mats["1300"], n_components=3, scale=False)
            ref_2850 = _reference_pca_full(draw_mats["2850"], n_components=3, scale=False)
            for model_name, ref_m in (("PCA_whole", ref_whole), ("PCA_1330", ref_1330), ("PCA_2850", ref_2850)):
                sub = draws_df.loc[(draws_df["draw"] == draw) & (draws_df["model"] == model_name)]
                max_d = max(max_d, _assert_close(sub["explained_variance_ratio"].to_numpy(), ref_m["explained_variance_ratio"], atol=1e-12, label=f"T1-D3 {model_name} draw {draw}"))
                pc1_can = can_map[model_name].loadings["PC1_loading"].to_numpy(dtype=float)
                cos_ref = float(np.dot(ref_m["loadings"][0], pc1_can) / (np.linalg.norm(ref_m["loadings"][0]) * np.linalg.norm(pc1_can)))
                cos_got = float(sub.loc[sub["PC"] == "PC1", "pc1_loading_cosine_similarity"].iloc[0])
                max_d = max(max_d, _assert_close(cos_got, cos_ref, atol=1e-12, label=f"T1-D3 cos {model_name}"))
        return max_d, f"Measurement-count sensitivity ({n_draws} draws) matches full-SVD reference"

    def _t1_d4() -> tuple[float | None, str]:
        draws_df = run_measurement_count_sensitivity(inputs, canonical_results, cfg=cfg)
        summ = summarize_measurement_count_sensitivity(draws_df)
        expected_cols = [
            "model",
            "PC",
            "n_draws",
            "canonical_explained_variance_ratio",
            "mean_explained_variance_ratio",
            "std_explained_variance_ratio",
            "ci_lower_2.5pct",
            "ci_upper_97.5pct",
            "mean_pc1_loading_cosine_similarity",
        ]
        assert list(summ.columns) == expected_cols
        g0 = draws_df.loc[(draws_df["model"] == "PCA_1330") & (draws_df["PC"] == "PC1"), "explained_variance_ratio"].to_numpy()
        r0 = summ.loc[(summ["model"] == "PCA_1330") & (summ["PC"] == "PC1")].iloc[0]
        d = _assert_close(r0["std_explained_variance_ratio"], np.std(g0, ddof=1), atol=1e-12, label="T1-D4 ddof=1")
        return d, "Summary schema (9 cols) & ddof=1 / quantiles verified"

    def _t1_d5() -> tuple[float | None, str]:
        cfg_off = copy.deepcopy(cfg)
        cfg_off["biological_pca"]["measurement_count_sensitivity"]["enabled"] = False
        assert run_measurement_count_sensitivity(inputs, canonical_results, cfg=cfg_off).empty
        no_rows = replace(inputs, measurement_rows={})
        assert run_measurement_count_sensitivity(no_rows, canonical_results, cfg=cfg).empty
        equal_inputs = _make_synthetic_subject_inputs(n_control=5, n_disease=5, p_1300=20, p_2850=20, n_meas_ctrl=3, n_meas_dis=3)
        eq_res = fit_biological_pca_models(equal_inputs, cfg=cfg)
        assert run_measurement_count_sensitivity(equal_inputs, eq_res, cfg=cfg).empty
        missing_inputs = replace(inputs, measurement_rows={k: v for k, v in inputs.measurement_rows.items() if k != inputs.subjects[0]})
        try:
            run_measurement_count_sensitivity(missing_inputs, canonical_results, cfg=cfg)
            raise AssertionError("Expected ValueError for missing subject IDs")
        except ValueError:
            pass
        return 0.0, "Early returns & missing subject ID ValueError verified"

    results.append(_run_check("T1-D1", "biological_pca", "tier1", _t1_d1))
    results.append(_run_check("T1-D2", "biological_pca", "tier1", _t1_d2))
    results.append(_run_check("T1-D3", "biological_pca", "tier1", _t1_d3))
    results.append(_run_check("T1-D4", "biological_pca", "tier1", _t1_d4))
    results.append(_run_check("T1-D5", "biological_pca", "tier1", _t1_d5))
    return results


# ---------------------------------------------------------------------------
# Suite 4: cluster_permutation (Area E & Tier 2 T2-04)
# ---------------------------------------------------------------------------


def _run_cluster_permutation_checks(*, mode: str, profile: str) -> list[ParityCheckResult]:
    from scipy import stats

    from src.analysis.clinical_feature_associations import (
        compute_binary_group_feature_tests,
        compute_spearman_feature_tests,
        feature_metadata,
    )
    from src.statistics.cluster_permutation import (
        _binary_cluster_maxima,
        _spearman_cluster_maxima,
        apply_binary_cluster_permutation,
        apply_spearman_cluster_permutation,
        find_signed_clusters,
        prepare_ranked_features,
        prepared_spearman,
        rank_biserial_effect,
    )

    results: list[ParityCheckResult] = []
    compact = profile == "smoke"

    def _ref_binary_maxima(values: np.ndarray, labels: np.ndarray, n_perms: int, alpha: float, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        seeds = rng.integers(0, np.iinfo(np.int32).max, size=n_perms, dtype=np.int64)
        n_left = int((labels == 0).sum())
        maxima = np.zeros(n_perms, dtype=float)
        for p_idx, p_seed in enumerate(seeds):
            order = np.random.default_rng(int(p_seed)).permutation(len(labels))
            left = values[order[:n_left]]
            right = values[order[n_left:]]
            res = stats.mannwhitneyu(left, right, axis=0, alternative="two-sided", method="auto")
            effects = rank_biserial_effect(left, right, np.asarray(res.statistic, dtype=float))
            clusters = find_signed_clusters(np.asarray(effects, dtype=float), np.asarray(res.pvalue, dtype=float), alpha=alpha)
            maxima[p_idx] = max((c[2] for c in clusters), default=0.0)
        return maxima

    if mode == "smoke":
        def _smoke_cp() -> tuple[float | None, str]:
            rng = np.random.default_rng(123)
            vals = rng.normal(size=(24, 60))
            labels = np.array([0] * 12 + [1] * 12, dtype=int)
            got = _binary_cluster_maxima(vals, labels, n_permutations=8, cluster_alpha=0.05, rng=np.random.default_rng(9), n_jobs=1)
            ref = _ref_binary_maxima(vals, labels, 8, 0.05, 9)
            d1 = _assert_close(got, ref, atol=1e-12, label="smoke binary cluster maxima")
            target = rng.normal(size=24)
            got_sp = _spearman_cluster_maxima(vals, target, n_permutations=8, cluster_alpha=0.05, rng=np.random.default_rng(9), n_jobs=1)
            prep_cols, cent_f, den_f = prepare_ranked_features(vals)
            r_tgt = stats.rankdata(target, method="average")
            c_tgt = r_tgt - float(np.mean(r_tgt))
            seeds = np.random.default_rng(9).integers(0, np.iinfo(np.int32).max, size=8, dtype=np.int64)
            ref_sp = np.zeros(8, dtype=float)
            for idx_s, sd in enumerate(seeds):
                perm_t = c_tgt[np.random.default_rng(int(sd)).permutation(len(c_tgt))]
                rho, pvals = prepared_spearman(prep_cols, cent_f, den_f, perm_t)
                cl = find_signed_clusters(rho, pvals, alpha=0.05)
                ref_sp[idx_s] = max((c[2] for c in cl), default=0.0)
            d2 = _assert_close(got_sp, ref_sp, atol=1e-12, label="smoke spearman cluster maxima")
            return max(d1, d2), "cluster_permutation binary + Spearman smoke parity verified"

        results.append(_run_check("SMOKE-CP", "cluster_permutation", "smoke", _smoke_cp))
        return results

    def _t1_e1() -> tuple[float | None, str]:
        rng = np.random.default_rng(201)
        n_p = 40 if compact else 92
        p_ch = 120 if compact else 300
        vals = rng.normal(size=(n_p, p_ch))
        labels = np.array([0] * (n_p // 2) + [1] * (n_p - n_p // 2), dtype=int)
        n_perm = 20 if compact else 40
        got1 = _binary_cluster_maxima(vals, labels, n_permutations=n_perm, cluster_alpha=0.05, rng=np.random.default_rng(7), n_jobs=1)
        got2 = _binary_cluster_maxima(vals, labels, n_permutations=n_perm, cluster_alpha=0.05, rng=np.random.default_rng(7), n_jobs=2)
        ref = _ref_binary_maxima(vals, labels, n_perm, 0.05, 7)
        d1 = _assert_close(got1, ref, atol=1e-12, label="T1-E1 n_jobs=1")
        d2 = _assert_close(got2, got1, atol=1e-13, label="T1-E1 n_jobs=2")
        return max(d1, d2), "Binary cluster maxima matches scipy.stats.mannwhitneyu (n_jobs=1 & 2)"

    def _t1_e2() -> tuple[float | None, str]:
        rng = np.random.default_rng(202)
        vals = rng.normal(size=(40, 120))
        target = np.round(rng.uniform(0.0, 10.0, size=40), 1)
        n_perm = 20
        got1 = _spearman_cluster_maxima(vals, target, n_permutations=n_perm, cluster_alpha=0.05, rng=np.random.default_rng(8), n_jobs=1)
        got2 = _spearman_cluster_maxima(vals, target, n_permutations=n_perm, cluster_alpha=0.05, rng=np.random.default_rng(8), n_jobs=2)
        prep_cols, cent_f, den_f = prepare_ranked_features(vals)
        r_tgt = stats.rankdata(target, method="average")
        c_tgt = r_tgt - float(np.mean(r_tgt))
        seeds = np.random.default_rng(8).integers(0, np.iinfo(np.int32).max, size=n_perm, dtype=np.int64)
        ref = np.zeros(n_perm, dtype=float)
        for idx_s, sd in enumerate(seeds):
            perm_t = c_tgt[np.random.default_rng(int(sd)).permutation(len(c_tgt))]
            rho, pvals = prepared_spearman(prep_cols, cent_f, den_f, perm_t)
            cl = find_signed_clusters(rho, pvals, alpha=0.05)
            ref[idx_s] = max((c[2] for c in cl), default=0.0)
        d1 = _assert_close(got1, ref, atol=1e-12, label="T1-E2 spearman n_jobs=1")
        d2 = _assert_close(got2, got1, atol=1e-13, label="T1-E2 spearman n_jobs=2")
        return max(d1, d2), "Spearman cluster maxima matches prepared_spearman reference"

    def _t1_e3() -> tuple[float | None, str]:
        frame = _make_synthetic_feature_frame(n_subjects=32, p_1300=30, p_2850=20, seed=203, signal=1.5)
        fcols = [c for c in frame.columns if c.startswith(("1300__", "2850__"))]
        fmeta = feature_metadata(fcols)
        tests, _ = compute_binary_group_feature_tests(frame, fmeta, group_column="binary_group", ordered_groups=("ctrl", "case"), contrast_name="c", min_subjects_per_group=5)
        out = apply_binary_cluster_permutation(tests, frame, group_column="binary_group", ordered_groups=("ctrl", "case"), n_permutations=20, cluster_alpha=0.05, random_state=42, n_jobs=1)
        for col in ("cluster_id", "cluster_mass", "cluster_p_value"):
            assert col in out.columns
        assert out["cluster_id"].dropna().str.contains("_C").any()
        return 0.0, "apply_binary_cluster_permutation DataFrame contract verified"

    def _t1_e4() -> tuple[float | None, str]:
        frame = _make_synthetic_feature_frame(n_subjects=32, p_1300=30, p_2850=20, seed=204, signal=1.5)
        frame.loc[:4, "moca_baseline"] = np.nan
        # Inject clean monotonic signal into both blocks so deterministic clusters appear
        y_sig = frame.loc[5:, "moca_baseline"].to_numpy(dtype=float)
        c1 = [c for c in frame.columns if c.startswith("1300__")][:6]
        c2 = [c for c in frame.columns if c.startswith("2850__")][:5]
        for j, col in enumerate(c1):
            frame.loc[5:, col] = 0.5 + (1.0 + 0.05 * j) * y_sig
        for j, col in enumerate(c2):
            frame.loc[5:, col] = 0.5 - (1.0 + 0.05 * j) * y_sig
        fcols = [c for c in frame.columns if c.startswith(("1300__", "2850__"))]
        fmeta = feature_metadata(fcols)
        tests = compute_spearman_feature_tests(frame, fmeta, target="moca_baseline", min_subjects=5)
        out = apply_spearman_cluster_permutation(
            tests, frame, target="moca_baseline", n_permutations=20, cluster_alpha=0.05, random_state=42, n_jobs=1
        )
        out2 = apply_spearman_cluster_permutation(
            tests, frame, target="moca_baseline", n_permutations=20, cluster_alpha=0.05, random_state=42, n_jobs=2
        )
        for col in ("cluster_id", "cluster_mass", "cluster_p_value"):
            assert col in out.columns
        assert out["cluster_id"].notna().any(), "Expected at least one Spearman cluster"
        assert out["cluster_id"].equals(out2["cluster_id"])

        # Independent NaN-filtered scipy.stats.spearmanr + sequential RNG reference
        valid_mask = pd.to_numeric(frame["moca_baseline"], errors="coerce").notna().to_numpy(dtype=bool)
        y_valid = pd.to_numeric(frame.loc[valid_mask, "moca_baseline"], errors="coerce").to_numpy(dtype=float)
        rng_ref = np.random.default_rng(42)
        ref_masses = np.full(len(out), np.nan, dtype=float)
        ref_pvals = np.full(len(out), np.nan, dtype=float)
        max_d = 0.0
        for _, block_df in out.groupby("spectral_block", sort=False):
            ordered = block_df.sort_values("wavenumber").reset_index()
            vals = frame.loc[valid_mask, ordered["feature_name"]].to_numpy(dtype=float)
            obs_rho = np.empty(vals.shape[1], dtype=float)
            obs_pv = np.empty(vals.shape[1], dtype=float)
            for j in range(vals.shape[1]):
                sp_obs = stats.spearmanr(vals[:, j], y_valid)
                obs_rho[j] = float(sp_obs.statistic)
                obs_pv[j] = float(sp_obs.pvalue)
            max_d = max(
                max_d,
                _assert_close(ordered["rho"].to_numpy(dtype=float), obs_rho, atol=1e-12, label="T1-E4 obs rho"),
                _assert_close(ordered["p_value"].to_numpy(dtype=float), obs_pv, atol=1e-12, label="T1-E4 obs p"),
            )
            clusters = find_signed_clusters(obs_rho, obs_pv, alpha=0.05)
            if not clusters:
                continue
            seeds = rng_ref.integers(0, np.iinfo(np.int32).max, size=20, dtype=np.int64)
            maxima = np.zeros(20, dtype=float)
            for p_i, sd in enumerate(seeds):
                perm_y = y_valid[np.random.default_rng(int(sd)).permutation(len(y_valid))]
                p_rho = np.empty(vals.shape[1], dtype=float)
                p_pv = np.empty(vals.shape[1], dtype=float)
                for j in range(vals.shape[1]):
                    sp_p = stats.spearmanr(vals[:, j], perm_y)
                    p_rho[j] = float(sp_p.statistic)
                    p_pv[j] = float(sp_p.pvalue)
                cl_p = find_signed_clusters(p_rho, p_pv, alpha=0.05)
                maxima[p_i] = max((c[2] for c in cl_p), default=0.0)
            for start, stop, mass in clusters:
                corr_p = float((1 + np.sum(maxima >= mass)) / 21.0)
                for row_idx in ordered.loc[start:stop, "index"]:
                    ref_masses[int(row_idx)] = mass
                    ref_pvals[int(row_idx)] = corr_p
        max_d = max(
            max_d,
            _assert_close(out["cluster_mass"].to_numpy(dtype=float), ref_masses, atol=1e-12, label="T1-E4 cluster_mass"),
            _assert_close(out["cluster_p_value"].to_numpy(dtype=float), ref_pvals, atol=1e-12, label="T1-E4 cluster_p_value"),
            _assert_close(out2["cluster_mass"].to_numpy(dtype=float), out["cluster_mass"].to_numpy(dtype=float), atol=1e-12, label="T1-E4 n_jobs=2 mass"),
            _assert_close(out2["cluster_p_value"].to_numpy(dtype=float), out["cluster_p_value"].to_numpy(dtype=float), atol=1e-12, label="T1-E4 n_jobs=2 p"),
        )
        return max_d, "apply_spearman_cluster_permutation NaN-filtered parity vs scipy.stats.spearmanr verified"

    def _t1_e5() -> tuple[float | None, str]:
        effects = np.array([0.8, 0.7, -0.7, -0.8, 0.1, 0.9], dtype=float)
        pvals = np.array([0.001, 0.001, 0.001, 0.001, 0.5, 0.001], dtype=float)
        cl = find_signed_clusters(effects, pvals, alpha=0.05)
        assert cl == [(0, 1, 1.5), (2, 3, 1.5), (5, 5, 0.9)]
        assert find_signed_clusters(effects, np.full(6, 0.5), alpha=0.05) == []
        return 0.0, "Signed cluster splitting & zero-cluster handling verified"

    def _t2_04() -> tuple[float | None, str]:
        rng = np.random.default_rng(205)
        # Small-N untied (6 + 8 <= 8 exact cutoff)
        vals_sm = rng.normal(size=(14, 30))
        lbl_sm = np.array([0] * 6 + [1] * 8, dtype=int)
        got_sm = _binary_cluster_maxima(vals_sm, lbl_sm, n_permutations=12, cluster_alpha=0.05, rng=np.random.default_rng(5), n_jobs=1)
        ref_sm = _ref_binary_maxima(vals_sm, lbl_sm, 12, 0.05, 5)
        d1 = _assert_close(got_sm, ref_sm, atol=1e-12, label="T2-04 small-N untied")
        # Heavy ties + constant column + NaN column
        vals_tie = np.round(rng.uniform(0, 3, size=(24, 25)), 0)
        vals_tie[:, 5] = 1.0
        vals_tie[:, 10] = np.nan
        lbl_tie = np.array([0] * 12 + [1] * 12, dtype=int)
        got_tie = _binary_cluster_maxima(vals_tie, lbl_tie, n_permutations=10, cluster_alpha=0.05, rng=np.random.default_rng(6), n_jobs=1)
        ref_tie = _ref_binary_maxima(vals_tie, lbl_tie, 10, 0.05, 6)
        d2 = _assert_close(got_tie, ref_tie, atol=1e-12, label="T2-04 heavy ties + NaN/const")
        # < 5 valid Spearman rows
        frame = _make_synthetic_feature_frame(n_subjects=10, p_1300=10, p_2850=10, seed=206)
        fcols = [c for c in frame.columns if c.startswith(("1300__", "2850__"))]
        fmeta = feature_metadata(fcols)
        tests = compute_spearman_feature_tests(frame, fmeta, target="moca_baseline", min_subjects=3)
        frame_few = frame.copy()
        frame_few.loc[4:, "moca_baseline"] = np.nan
        out_few = apply_spearman_cluster_permutation(tests, frame_few, target="moca_baseline", n_permutations=10, cluster_alpha=0.05, random_state=42)
        assert "cluster_id" not in out_few.columns
        assert out_few.equals(tests)
        return max(d1, d2), "Small-N exact, heavy ties, constant/NaN cols, and <5 Spearman rows verified"

    results.append(_run_check("T1-E1", "cluster_permutation", "tier1", _t1_e1))
    results.append(_run_check("T1-E2", "cluster_permutation", "tier1", _t1_e2))
    results.append(_run_check("T1-E3", "cluster_permutation", "tier1", _t1_e3))
    results.append(_run_check("T1-E4", "cluster_permutation", "tier1", _t1_e4))
    results.append(_run_check("T1-E5", "cluster_permutation", "tier1", _t1_e5))
    results.append(_run_check("T2-04", "cluster_permutation", "tier2", _t2_04))
    return results


# ---------------------------------------------------------------------------
# Suite 5: control_vs_disease_ml (Area F)
# ---------------------------------------------------------------------------


def _run_control_vs_disease_ml_checks(*, mode: str, profile: str, tmp_dir: Path) -> list[ParityCheckResult]:
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    from src.analysis.control_vs_disease_ml import (
        _cv_predict,
        _locked_joint_stratified_split_plan,
        _rf_permutation_importance,
        _run_repeated_nested_cv,
        run_control_vs_disease_ml_pipeline,
        run_permutation_test,
    )
    from src.data_preparation.materialize import build_preprocessed_store

    def _ref_tune_threshold_and_ba(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
        cand = np.unique(scores)
        thresholds = np.concatenate([[cand[0] - 1e-6], (cand[:-1] + cand[1:]) / 2.0, [cand[-1] + 1e-6]])
        preds = scores[None, :] >= thresholds[:, None]
        pos = y_true == 1
        neg = ~pos
        sens = preds[:, pos].mean(axis=1)
        spec = (~preds[:, neg]).mean(axis=1)
        ba = (sens + spec) / 2.0
        best_i = int(np.argmax(ba))
        return float(thresholds[best_i]), float(ba[best_i])

    def _ref_uncached_cv_predict(
        model_name: str,
        X_in: np.ndarray,
        y_in: np.ndarray,
        outer_cv: Sequence[tuple[np.ndarray, np.ndarray]],
        *,
        ml_cfg_in: dict[str, Any],
        n_inner_splits: int,
        random_state: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        y_pred = np.zeros_like(y_in, dtype=int)
        y_score = np.zeros_like(y_in, dtype=float)
        for tr_idx, te_idx in outer_cv:
            X_tr, y_tr = X_in[tr_idx], y_in[tr_idx]
            X_te = X_in[te_idx]
            n_sp = min(n_inner_splits, int(np.bincount(y_tr).min()))
            in_folds = list(StratifiedKFold(n_splits=n_sp, shuffle=True, random_state=random_state).split(X_tr, y_tr)) if n_sp >= 2 else None
            best_thresh = 0.5
            best_score = -np.inf
            if model_name == "pls_da":
                comp_grid = list(ml_cfg_in.get("pls_components_grid", [1, 2]))
                best_param: Any = comp_grid[0]
                if in_folds is not None:
                    max_valid = min(min(len(it) for it, _ in in_folds), X_tr.shape[1])
                    valid_grid = [c for c in comp_grid if 1 <= c <= max_valid] or [1]
                    for n_comp in valid_grid:
                        oof = np.zeros(len(y_tr), dtype=float)
                        for it, iv in in_folds:
                            sc = StandardScaler()  # fresh uncached scaler per (param, inner_fold)
                            pls = PLSRegression(n_components=n_comp, scale=False)
                            pls.fit(sc.fit_transform(X_tr[it]), y_tr[it])
                            oof[iv] = pls.predict(sc.transform(X_tr[iv])).ravel()
                        th, b_acc = _ref_tune_threshold_and_ba(y_tr, oof)
                        if b_acc > best_score:
                            best_score, best_param, best_thresh = b_acc, n_comp, th
                final_n = max(1, min(int(best_param), min(X_tr.shape[0] - 2, X_tr.shape[1])))
                f_sc = StandardScaler()
                f_pls = PLSRegression(n_components=final_n, scale=False)
                f_pls.fit(f_sc.fit_transform(X_tr), y_tr)
                scores = f_pls.predict(f_sc.transform(X_te)).ravel()
            elif model_name == "l1_logistic":
                c_grid = list(ml_cfg_in.get("logistic_c_grid", [0.1, 1.0]))
                best_param = c_grid[0]
                if in_folds is not None:
                    for c_val in c_grid:
                        oof = np.zeros(len(y_tr), dtype=float)
                        for it, iv in in_folds:
                            sc = StandardScaler()  # fresh uncached scaler per (param, inner_fold)
                            lr = LogisticRegression(
                                solver="liblinear",
                                l1_ratio=1.0,
                                dual=False,
                                C=c_val,
                                class_weight="balanced",
                                random_state=random_state,
                                max_iter=10_000,
                                tol=1e-4,
                            )
                            lr.fit(sc.fit_transform(X_tr[it]), y_tr[it])
                            oof[iv] = lr.predict_proba(sc.transform(X_tr[iv]))[:, 1]
                        th, b_acc = _ref_tune_threshold_and_ba(y_tr, oof)
                        if b_acc > best_score:
                            best_score, best_param, best_thresh = b_acc, c_val, th
                f_sc = StandardScaler()
                f_lr = LogisticRegression(
                    solver="liblinear",
                    l1_ratio=1.0,
                    dual=False,
                    C=float(best_param),
                    class_weight="balanced",
                    random_state=random_state,
                    max_iter=10_000,
                    tol=1e-4,
                )
                f_lr.fit(f_sc.fit_transform(X_tr), y_tr)
                scores = f_lr.predict_proba(f_sc.transform(X_te))[:, 1]
            else:
                raise ValueError(f"Unsupported reference model: {model_name}")
            y_score[te_idx] = scores
            y_pred[te_idx] = (scores >= best_thresh).astype(int)
        return y_pred, y_score

    def _ref_uncached_perm(
        model_name: str,
        X_in: np.ndarray,
        y_in: np.ndarray,
        *,
        observed_ba: float,
        strata: np.ndarray | None,
        ml_cfg_in: dict[str, Any],
        random_state: int,
        outer_cv: Sequence[tuple[np.ndarray, np.ndarray]],
        n_draws: int,
    ) -> tuple[float, np.ndarray]:
        seeds = np.random.default_rng(random_state).integers(0, np.iinfo(np.int32).max, size=n_draws, dtype=np.int64)
        n_inner = int(ml_cfg_in.get("n_inner_splits", 4))
        cfg_rs = int(ml_cfg_in.get("random_state", 42))
        ref_dist = np.empty(n_draws, dtype=float)
        for idx_s, sd in enumerate(seeds):
            r_rng = np.random.default_rng(int(sd))
            if strata is None:
                y_perm = r_rng.permutation(y_in)
            else:
                y_perm = y_in.copy()
                for st_val in np.unique(strata):
                    mask = strata == st_val
                    y_perm[mask] = r_rng.permutation(y_in[mask])
            yp, _ = _ref_uncached_cv_predict(
                model_name, X_in, y_perm, outer_cv, ml_cfg_in=ml_cfg_in, n_inner_splits=n_inner, random_state=cfg_rs
            )
            ba = 0.5 * (float(np.mean(yp[y_perm == 1] == 1)) + float(np.mean(yp[y_perm == 0] == 0)))
            ref_dist[idx_s] = ba
        p_val = float((1 + np.sum(ref_dist >= observed_ba)) / (len(ref_dist) + 1))
        return p_val, ref_dist

    results: list[ParityCheckResult] = []
    compact = profile == "smoke"

    if mode == "smoke":
        def _smoke_ml() -> tuple[float | None, str]:
            rng = np.random.default_rng(300)
            n = 24
            X = rng.normal(size=(n, 10))
            y = np.array([0] * (n // 2) + [1] * (n // 2), dtype=int)
            X[:, :3] += 1.2 * y[:, None]
            cfg = _make_eda_cfg(n_outer_splits=2, n_outer_repeats=1, n_inner_splits=2, n_permutations=2, n_jobs=1)
            ml_cfg = {**cfg["analysis"]["control_vs_disease_ml"], "rf_n_estimators_grid": [5]}
            cv_splits = list(StratifiedKFold(n_splits=2, shuffle=True, random_state=42).split(X, y))
            p1, s1 = _cv_predict("pls_da", X, y, cv_splits, ml_cfg=ml_cfg, n_inner_splits=2, threshold_metric="balanced_accuracy", random_state=42, n_jobs=1)
            p2, s2 = _cv_predict("random_forest", X, y, cv_splits, ml_cfg=ml_cfg, n_inner_splits=2, threshold_metric="balanced_accuracy", random_state=42, n_jobs=1)
            assert p2.shape == y.shape and np.all(np.isfinite(s2))
            ref_p1, ref_s1 = _ref_uncached_cv_predict("pls_da", X, y, cv_splits, ml_cfg_in=ml_cfg, n_inner_splits=2, random_state=42)
            assert np.array_equal(p1, ref_p1)
            d_cv = _assert_close(s1, ref_s1, atol=1e-12, label="smoke pls_da uncached ref")
            pv, perm_dist = run_permutation_test("pls_da", X, y, observed_balanced_accuracy=0.85, strata=None, ml_cfg=ml_cfg, n_jobs=1, random_state=42, outer_split_plan=[cv_splits])
            ref_pv, ref_dist = _ref_uncached_perm(
                "pls_da", X, y, observed_ba=0.85, strata=None, ml_cfg_in=ml_cfg, random_state=42, outer_cv=cv_splits, n_draws=len(perm_dist)
            )
            d_pv = _assert_close(pv, ref_pv, atol=1e-12, label="smoke perm p_value")
            d_dist = _assert_close(perm_dist, ref_dist, atol=1e-12, label="smoke perm dist")
            return max(d_cv, d_pv, d_dist), "control_vs_disease_ml pls_da + permutation vs uncached reference verified"

        results.append(_run_check("SMOKE-ML", "control_vs_disease_ml", "smoke", _smoke_ml))
        return results

    rng = np.random.default_rng(301)
    n_subj = 48 if compact else 80
    n_feat = 24 if compact else 80
    X = rng.normal(size=(n_subj, n_feat))
    y = np.array([0] * (n_subj // 2) + [1] * (n_subj - n_subj // 2), dtype=int)
    sex = np.array(["female" if i % 2 == 0 else "male" for i in range(n_subj)], dtype=object)
    X[:, :4] += 1.1 * y[:, None]
    cfg = _make_eda_cfg(n_outer_splits=4, n_outer_repeats=1, n_inner_splits=3, n_permutations=8, n_bootstrap=4, n_jobs=1)
    ml_cfg = cfg["analysis"]["control_vs_disease_ml"]
    split_plan = _locked_joint_stratified_split_plan(y, sex, n_outer_splits=4, n_outer_repeats=1, random_state=42)

    def _t1_f1() -> tuple[float | None, str]:
        ml_rf = {**ml_cfg, "rf_n_estimators_grid": [10, 20], "rf_max_depth_grid": [2, None]}
        df1 = _run_repeated_nested_cv("random_forest", X, y, ml_cfg=ml_rf, n_jobs=1, outer_split_plan=split_plan)
        df2 = _run_repeated_nested_cv("random_forest", X, y, ml_cfg=ml_rf, n_jobs=2, outer_split_plan=split_plan)
        assert np.array_equal(df1["y_pred"].to_numpy(), df2["y_pred"].to_numpy())
        d_cv = _assert_close(df1["y_score"], df2["y_score"], atol=1e-12, label="T1-F1 RF y_score")
        tr0, te0 = split_plan[0][0]
        rf_est = RandomForestClassifier(
            n_estimators=20, max_depth=None, class_weight="balanced_subsample", random_state=42, n_jobs=1
        ).fit(X[tr0], y[tr0])
        imp_mean, imp_std = _rf_permutation_importance(rf_est, X[te0], y[te0], n_repeats=5, random_state=42, n_jobs=1)
        ref_imp = permutation_importance(
            rf_est, X[te0], y[te0], n_repeats=5, random_state=42, n_jobs=1, scoring="balanced_accuracy"
        )
        d_imp = max(
            _assert_close(imp_mean, ref_imp.importances_mean, atol=1e-12, label="T1-F1 RF perm imp mean"),
            _assert_close(imp_std, ref_imp.importances_std, atol=1e-12, label="T1-F1 RF perm imp std"),
        )
        return max(d_cv, d_imp), "random_forest multi-grid n_jobs=2==1 & _rf_permutation_importance vs sklearn verified"

    def _t1_f2() -> tuple[float | None, str]:
        ml_f2 = {**ml_cfg, "n_permutations": 8, "permutation_min": 8, "permutation_check_interval": 8}
        max_d = 0.0
        for mname in ("pls_da", "l1_logistic"):
            p1, dist1 = run_permutation_test(
                mname, X, y, observed_balanced_accuracy=0.80, strata=sex, ml_cfg=ml_f2, n_jobs=1, random_state=42, outer_split_plan=split_plan
            )
            p2, dist2 = run_permutation_test(
                mname, X, y, observed_balanced_accuracy=0.80, strata=sex, ml_cfg=ml_f2, n_jobs=2, random_state=42, outer_split_plan=split_plan
            )
            ref_p, ref_dist = _ref_uncached_perm(
                mname, X, y, observed_ba=0.80, strata=sex, ml_cfg_in=ml_f2, random_state=42, outer_cv=split_plan[0], n_draws=len(dist1)
            )
            max_d = max(
                max_d,
                _assert_close(p1, ref_p, atol=1e-12, label=f"T1-F2 {mname} n_jobs=1 p"),
                _assert_close(p2, ref_p, atol=1e-12, label=f"T1-F2 {mname} n_jobs=2 p"),
                _assert_close(dist1, ref_dist, atol=1e-12, label=f"T1-F2 {mname} n_jobs=1 dist"),
                _assert_close(dist2, ref_dist, atol=1e-12, label=f"T1-F2 {mname} n_jobs=2 dist"),
            )
        return max_d, "Permutation test parity (pls_da & l1_logistic, n_jobs=1 & 2) vs uncached reference verified"

    def _t1_f3() -> tuple[float | None, str]:
        # Pure sex-confounded feature signal: strata=sex must yield non-significant p > 0.05
        rng_c = np.random.default_rng(302)
        n_c = 60
        sex_c = np.array(["female"] * 30 + ["male"] * 30, dtype=object)
        # Imbalanced disease prevalence by sex so unstratified detects sex, stratified does not
        y_c = np.array([1] * 24 + [0] * 6 + [1] * 6 + [0] * 24, dtype=int)
        X_c = rng_c.normal(scale=0.3, size=(n_c, 12))
        X_c[:, :4] += np.where(sex_c == "female", 2.0, 0.0)[:, None]
        sp_c = _locked_joint_stratified_split_plan(y_c, sex_c, n_outer_splits=3, n_outer_repeats=1, random_state=42)
        obs_df = _run_repeated_nested_cv("pls_da", X_c, y_c, ml_cfg=ml_cfg, n_jobs=1, outer_split_plan=sp_c)
        obs_ba = float(balanced_accuracy_score(obs_df["y_true"], obs_df["y_pred"]))
        ml_perm = {**ml_cfg, "n_permutations": 12, "permutation_min": 12, "permutation_check_interval": 12}
        pv, _ = run_permutation_test("pls_da", X_c, y_c, observed_balanced_accuracy=obs_ba, strata=sex_c, ml_cfg=ml_perm, n_jobs=1, random_state=42, outer_split_plan=sp_c)
        assert pv > 0.05, f"Expected sex-stratified p_value > 0.05 on sex-only signal, got {pv}"
        return 0.0, f"Sex-stratified permutation confound guard passed (p={pv:.3f} > 0.05)"

    def _t1_f4() -> tuple[float | None, str]:
        ckpt = tmp_dir / "t1_f4_perm.parquet"
        ml_ck = {**ml_cfg, "n_permutations": 12, "permutation_min": 6, "permutation_check_interval": 6}
        p_full, dist_full = run_permutation_test(
            "pls_da", X, y, observed_balanced_accuracy=0.80, strata=sex, ml_cfg=ml_ck, n_jobs=1, random_state=99, outer_split_plan=split_plan
        )
        # Write first 6 rows to checkpoint, then resume
        ml_first = {**ml_cfg, "n_permutations": 6, "permutation_min": 6, "permutation_check_interval": 6}
        run_permutation_test(
            "pls_da", X, y, observed_balanced_accuracy=0.80, strata=sex, ml_cfg=ml_first, n_jobs=1, random_state=99, outer_split_plan=split_plan, checkpoint_path=ckpt, resume=False
        )
        p_res, dist_res = run_permutation_test(
            "pls_da", X, y, observed_balanced_accuracy=0.80, strata=sex, ml_cfg=ml_ck, n_jobs=1, random_state=99, outer_split_plan=split_plan, checkpoint_path=ckpt, resume=True
        )
        d1 = _assert_close(p_res, p_full, atol=1e-12, label="T1-F4 resume p")
        d2 = _assert_close(dist_res, dist_full, atol=1e-12, label="T1-F4 resume dist")
        return max(d1, d2), "Parquet permutation checkpoint resume verified"

    def _t1_f5() -> tuple[float | None, str]:
        store_dir = tmp_dir / "t1_f5_store"
        _, pre_dir_f5, _, store_cfg = _build_synthetic_parsed_store(store_dir, n_subjects=4, n_pixels_per_map=8)
        mat_f5 = build_preprocessed_store(store_cfg, force=True)
        cfg_f5 = copy.deepcopy(store_cfg)
        cfg_f5["analysis"]["control_vs_disease_ml"] = copy.deepcopy(ml_cfg)
        fm_path = tmp_dir / "t1_f5_features.parquet"
        out_dir = tmp_dir / "t1_f5_ml_out"
        frame = _make_synthetic_feature_frame(n_subjects=44, seed=303, canonical_grid=True)
        frame.to_parquet(fm_path, index=False)
        outputs = run_control_vs_disease_ml_pipeline(
            feature_matrix_path=fm_path,
            output_dir=out_dir,
            cfg=cfg_f5,
            data_manifest_hash=mat_f5.manifest.get("manifest_hash"),
        )
        assert len(outputs) > 0
        assert (out_dir / "02_model_evaluation" / "model_comparison.csv").is_file()
        assert "final_performance_summary" in outputs and outputs["final_performance_summary"].is_file()
        run_man = json.loads((out_dir / "00_run_metadata" / "run_manifest.json").read_text(encoding="utf-8"))
        assert run_man["data_source_path"] == str(pre_dir_f5)
        assert str(ROOT / "data") not in run_man["data_source_path"]
        return None, "Isolated run_control_vs_disease_ml_pipeline synthetic store & Parquet contract verified"

    results.append(_run_check("T1-F1", "control_vs_disease_ml", "tier1", _t1_f1))
    results.append(_run_check("T1-F2", "control_vs_disease_ml", "tier1", _t1_f2))
    results.append(_run_check("T1-F3", "control_vs_disease_ml", "tier1", _t1_f3))
    results.append(_run_check("T1-F4", "control_vs_disease_ml", "tier1", _t1_f4))
    results.append(_run_check("T1-F5", "control_vs_disease_ml", "tier1", _t1_f5))
    return results


# ---------------------------------------------------------------------------
# Suite 6: nmf_decomposition & disease_subgroups (Area G)
# ---------------------------------------------------------------------------


def _run_nmf_subgroups_checks(*, mode: str, profile: str, tmp_dir: Path) -> list[ParityCheckResult]:
    import warnings

    from sklearn.cluster import AgglomerativeClustering
    from sklearn.decomposition import NMF
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.metrics import adjusted_rand_score

    from src.analysis.disease_subgroups import (
        _bootstrap_ari,
        _cluster_labels,
        _gap_statistic,
        _select_gap_k,
        _within_cluster_dispersion,
        run_disease_subgroups,
    )
    from src.analysis.nmf_decomposition import _fit_nmf_multi_init, run_nmf_decomposition

    results: list[ParityCheckResult] = []
    compact = profile == "smoke"

    if mode == "smoke":
        def _smoke_nmf() -> tuple[float | None, str]:
            rng = np.random.default_rng(400)
            w_true = rng.uniform(0.2, 1.5, size=(24, 2))
            h_true = rng.uniform(0.2, 1.5, size=(2, 40))
            X_nmf = w_true @ h_true + rng.uniform(0.0, 0.05, size=(24, 40))
            res = _fit_nmf_multi_init(X_nmf, n_components=2, alpha_w=0.05, alpha_h=0.0, n_init=2, random_state=42)
            seeds = np.random.default_rng(42).integers(0, np.iinfo(np.int32).max, size=2, dtype=np.int64).tolist()
            errs = []
            for sd in seeds:
                m = NMF(n_components=2, init="random", l1_ratio=1.0, alpha_W=0.05, alpha_H=0.0, max_iter=1000, random_state=int(sd))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    W = m.fit_transform(X_nmf)
                errs.append(float(np.mean((X_nmf - W @ m.components_) ** 2)))
            d1 = _assert_close(res.best_error, min(errs), atol=1e-12, label="smoke NMF error")
            X_cl = rng.normal(size=(24, 6))
            gap = _gap_statistic(X_cl, k_values=[1, 2, 3], n_bootstrap=3, random_state=42)
            assert _select_gap_k(gap) in (1, 2, 3)
            ari = _bootstrap_ari(X_cl, _cluster_labels(X_cl, 2), n_clusters=2, n_bootstrap=3, random_state=42)
            assert ari["n_bootstrap_used"].iloc[0] == 3
            return d1, "NMF + disease_subgroups smoke parity verified"

        results.append(_run_check("SMOKE-NMF", "nmf_subgroups", "smoke", _smoke_nmf))
        return results

    rng = np.random.default_rng(401)
    X_nmf = rng.uniform(0.1, 2.0, size=(36 if compact else 60, 60 if compact else 120))
    X_sub = rng.normal(size=(36 if compact else 50, 8))

    def _t1_g1() -> tuple[float | None, str]:
        n_init = 4 if compact else 8
        got = _fit_nmf_multi_init(X_nmf, n_components=3, alpha_w=0.05, alpha_h=0.0, n_init=n_init, random_state=42)
        seeds = np.random.default_rng(42).integers(0, np.iinfo(np.int32).max, size=n_init, dtype=np.int64).tolist()
        errs = []
        sols = []
        for sd in seeds:
            m = NMF(n_components=3, init="random", l1_ratio=1.0, alpha_W=0.05, alpha_H=0.0, max_iter=1000, random_state=int(sd))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                W = m.fit_transform(X_nmf)
            errs.append(float(np.mean((X_nmf - W @ m.components_) ** 2)))
            sols.append((W, m.components_))
        best_i = int(np.argmin(errs))
        d1 = _assert_close(got.best_error, errs[best_i], atol=1e-12, label="T1-G1 best_error")
        d2 = _assert_close(got.H, sols[best_i][1], atol=1e-12, label="T1-G1 H")
        return max(d1, d2), f"_fit_nmf_multi_init matches serial sklearn NMF ({n_init} starts)"

    def _t1_g2() -> tuple[float | None, str]:
        n_b = 6
        got = _gap_statistic(X_sub, k_values=[1, 2, 3], n_bootstrap=n_b, random_state=42)
        rng_r = np.random.default_rng(42)
        mins, maxs = X_sub.min(axis=0), X_sub.max(axis=0)
        ref_gaps = []
        for k in (1, 2, 3):
            lbl = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X_sub)
            obs = _within_cluster_dispersion(X_sub, lbl)
            logs = []
            for _ in range(n_b):
                ref_x = rng_r.uniform(mins, maxs, size=X_sub.shape)
                r_lbl = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(ref_x)
                logs.append(float(np.log(_within_cluster_dispersion(ref_x, r_lbl))))
            ref_gaps.append(float(np.mean(logs) - np.log(obs)))
        d = _assert_close(got["gap"], ref_gaps, atol=1e-12, label="T1-G2 gap")
        # Rising gap fallback to k=1
        rising = pd.DataFrame({"n_clusters": [1, 2, 3], "gap": [0.1, 0.5, 1.0], "reference_log_dispersion_se": [0.05, 0.05, 0.05]})
        assert _select_gap_k(rising) == 1
        return d, "_gap_statistic & _select_gap_k verified"

    def _t1_g3() -> tuple[float | None, str]:
        base_lbl = _cluster_labels(X_sub, 2)
        got = _bootstrap_ari(X_sub, base_lbl, n_clusters=2, n_bootstrap=10, random_state=42)
        rng_r = np.random.default_rng(42)
        scores = []
        for _ in range(10):
            s = rng_r.integers(0, X_sub.shape[0], size=X_sub.shape[0])
            u = np.unique(s)
            if len(u) < 3:
                continue
            lbl = AgglomerativeClustering(n_clusters=2, linkage="ward").fit_predict(X_sub[u])
            scores.append(float(adjusted_rand_score(base_lbl[u], lbl)))
        d = _assert_close(got["mean_bootstrap_ari"].iloc[0], np.mean(scores), atol=1e-12, label="T1-G3 ARI")
        return d, "_bootstrap_ari matches serial AgglomerativeClustering reference"

    def _t1_g4() -> tuple[float | None, str]:
        from src.analysis.paired_lipid._backend import _get_threadpool_controller, _single_thread_blas

        with _single_thread_blas():
            ctrl = _get_threadpool_controller()
            if ctrl is not None:
                for info in ctrl.info():
                    if info.get("user_api") == "blas":
                        assert int(info.get("num_threads", 1)) == 1
        return 0.0, "Worker BLAS single-thread guard verified"

    def _t1_g5() -> tuple[float | None, str]:
        fm_path = tmp_dir / "t1_g5_features.parquet"
        cand_path = tmp_dir / "t1_g5_candidates.csv"
        frame = _make_synthetic_feature_frame(n_subjects=32, seed=402, canonical_grid=True, two_disease_clusters=True)
        frame.to_parquet(fm_path, index=False)
        pd.DataFrame(
            [
                {
                    "candidate_region_id": "cand_1525",
                    "spectral_block": "1300",
                    "start_cm-1": 1510.0,
                    "end_cm-1": 1535.0,
                    "evidence_status": "cluster_supported_exploratory",
                },
                {
                    "candidate_region_id": "cand_2853",
                    "spectral_block": "2850",
                    "start_cm-1": 2845.0,
                    "end_cm-1": 2860.0,
                    "evidence_status": "cluster_supported_exploratory",
                },
            ]
        ).to_csv(cand_path, index=False)
        cfg = _make_eda_cfg(nmf_n_init=3, n_bootstrap=4, max_clusters=3)
        nmf_out = run_nmf_decomposition(feature_matrix_path=fm_path, output_dir=tmp_dir / "t1_g5_nmf", cfg=cfg)
        sub_out = run_disease_subgroups(
            feature_matrix_path=fm_path,
            integrated_candidates_path=cand_path,
            output_dir=tmp_dir / "t1_g5_sub",
            cfg=cfg,
        )
        assert "nmf_components" in nmf_out and nmf_out["nmf_components"].is_file()
        assert "disease_subgroup_assignments" in sub_out and sub_out["disease_subgroup_assignments"].is_file()
        assign_df = pd.read_csv(sub_out["disease_subgroup_assignments"])
        n_cl = int(assign_df["n_clusters"].iloc[0])
        assert n_cl >= 2 and assign_df["subgroup_label"].nunique() == n_cl
        # Small-N guard (< 6 subjects)
        small_path = tmp_dir / "t1_g5_small.parquet"
        frame.loc[~frame["is_control"]].iloc[:3].to_parquet(small_path, index=False)
        small_res = run_disease_subgroups(
            feature_matrix_path=small_path,
            integrated_candidates_path=cand_path,
            output_dir=tmp_dir / "t1_g5_small_out",
            cfg=cfg,
        )
        assert "_status" in small_res and small_res["_status"]["status"] == "insufficient_n"
        return None, "run_nmf_decomposition & run_disease_subgroups isolated E2E + small-N guard verified"

    results.append(_run_check("T1-G1", "nmf_subgroups", "tier1", _t1_g1))
    results.append(_run_check("T1-G2", "nmf_subgroups", "tier1", _t1_g2))
    results.append(_run_check("T1-G3", "nmf_subgroups", "tier1", _t1_g3))
    results.append(_run_check("T1-G4", "nmf_subgroups", "tier1", _t1_g4))
    results.append(_run_check("T1-G5", "nmf_subgroups", "tier1", _t1_g5))
    return results


# ---------------------------------------------------------------------------
# Suite 7: preprocessing & data_preparation (Area H, T2-07, T3-04)
# ---------------------------------------------------------------------------


def _run_preprocessing_store_checks(*, mode: str, profile: str, tmp_dir: Path) -> list[ParityCheckResult]:
    from pybaselines.smooth import snip
    from pybaselines.whittaker import arpls, asls

    from src.data_preparation.dataset import IngestedRamanDataset
    from src.data_preparation.loaders import _summarize_rows, load_area_summaries
    from src.data_preparation.manifest import config_hash
    from src.data_preparation.materialize import (
        _frozen_target_axis,
        _process_measurements,
        _range_key,
        build_preprocessed_store,
    )
    from src.data_preparation.store import DataSource
    from src.preprocessing.pipeline import (
        _local_cosmic_detection,
        _spatial_cosmic_detection,
        correct_baseline,
        run_preprocessing,
    )

    results: list[ParityCheckResult] = []
    compact = profile == "smoke"

    if mode == "smoke":
        def _smoke_prep() -> tuple[float | None, str]:
            rng = np.random.default_rng(500)
            wn = np.linspace(800.0, 1750.0, 80)
            spec = 20.0 + 0.01 * wn[None, :] + 10.0 * np.exp(-0.5 * ((wn - 1450.0) / 15.0) ** 2)[None, :] + rng.normal(scale=0.1, size=(12, 80))
            corr, bl = correct_baseline(spec, method="arpls", arpls_lam=1e4, arpls_ratio=1e-3, arpls_max_iter=50, return_baseline=True)
            ref_bl = np.vstack([arpls(r, lam=1e4, diff_order=2, max_iter=50, tol=1e-3)[0] for r in spec])
            d1 = _assert_close(bl, ref_bl, atol=1e-10, rtol=1e-10, label="smoke arpls")
            # Mmap _summarize_rows check
            mm_path = tmp_dir / "smoke_mm.dat"
            mm = np.memmap(mm_path, dtype=np.float64, mode="w+", shape=(20, 10))
            mm[:] = rng.normal(size=(20, 10))
            mm.flush()
            mm_ro = np.memmap(mm_path, dtype=np.float64, mode="r", shape=(20, 10))
            rows = pd.DataFrame({"measurement_id": ["m2", "m1", "m2", "m3"], "row_col": [5, 1, 2, -1]})
            mids, sums = _summarize_rows(rows, mm_ro, "row_col")
            assert list(mids) == ["m2", "m1"]
            ref_sums = np.vstack([np.nanmedian(mm_ro[[5, 2]], axis=0), np.nanmedian(mm_ro[[1]], axis=0)])
            d2 = _assert_close(sums, ref_sums, atol=1e-14, label="smoke _summarize_rows")
            del mm_ro, mm
            return max(d1, d2), "arpls vs pybaselines & memmap _summarize_rows smoke verified"

        results.append(_run_check("SMOKE-PREP", "preprocessing_store", "smoke", _smoke_prep))
        return results

    def _t1_h1() -> tuple[float | None, str]:
        rng = np.random.default_rng(501)
        spec = 10.0 + rng.normal(scale=0.2, size=(24, 100))
        coords = np.column_stack([np.repeat(np.arange(6), 4), np.tile(np.arange(4), 6)]).astype(float)
        spec[3, 40] += 50.0  # width-1 spike
        spec[7, 60:62] += 50.0  # width-2 spike
        spec[11, 20:25] += 50.0  # width-5 spike (> max_width=3)
        mask, repl = _spatial_cosmic_detection(spec, coords, z_threshold=8.0, n_neighbors=4, max_width=3)
        ref_mask, ref_repl = _reference_spatial_cosmic(spec, coords, 8.0, 4, 3)
        assert np.array_equal(mask, ref_mask)
        assert not mask[11, 20:25].any(), "Width-5 spike should be excluded by max_width=3"
        d = _assert_close(repl, ref_repl, atol=1e-13, label="T1-H1 replacement")
        return d, "_spatial_cosmic_detection exact mask & replacement verified"

    def _t1_h2() -> tuple[float | None, str]:
        rng = np.random.default_rng(502)
        n_rows = 16 if compact else 48
        n_ch = 120 if compact else 300
        spec = 30.0 + rng.normal(scale=0.5, size=(n_rows, n_ch))
        max_d = 0.0
        _, bl_ar = correct_baseline(spec, method="arpls", arpls_lam=1e5, arpls_ratio=1e-3, arpls_max_iter=80, return_baseline=True)
        ref_ar = np.vstack([arpls(r, lam=1e5, diff_order=2, max_iter=80, tol=1e-3)[0] for r in spec])
        max_d = max(max_d, _assert_close(bl_ar, ref_ar, atol=1e-10, rtol=1e-10, label="T1-H2 arpls"))
        _, bl_als = correct_baseline(spec, method="als", lam=1e5, p=0.01, return_baseline=True)
        ref_als = np.vstack([asls(r, lam=1e5, p=0.01)[0] for r in spec])
        max_d = max(max_d, _assert_close(bl_als, ref_als, atol=1e-10, rtol=1e-10, label="T1-H2 als"))
        _, bl_snip = correct_baseline(spec, method="snip", return_baseline=True)
        ref_snip = np.vstack([snip(r, max_half_window=min(50, (n_ch - 1) // 2), decreasing=True, smooth_half_window=3)[0] for r in spec])
        max_d = max(max_d, _assert_close(bl_snip, ref_snip, atol=1e-10, rtol=1e-10, label="T1-H2 snip"))
        return max_d, "correct_baseline (arpls, als, snip) matches pybaselines within 1e-10"

    def _t1_h3() -> tuple[float | None, str]:
        rng = np.random.default_rng(503)
        wn = np.linspace(785.0, 1765.0, 100)
        peak = 250.0 * np.exp(-0.5 * ((wn - 1450.0) / 20.0) ** 2)
        spec = 40.0 + peak[None, :] + rng.normal(scale=1.0, size=(20, 100))
        coords = np.column_stack([np.repeat(np.arange(5), 4), np.tile(np.arange(4), 5)]).astype(float)
        spec[0, :] = 40.0  # dead pixel
        spec[1, 50] += 2000.0  # cosmic spike
        cfg = _make_eda_cfg()
        res = run_preprocessing(
            spec,
            wn,
            region="fingerprint",
            pixel_coords=coords,
            interpolate=True,
            target_wavenumbers=np.arange(800.0, 1751.0, 10.0),
            return_pre_normalization=True,
            cfg=cfg,
        )
        assert not res.retained_mask[0]
        assert res.retained_mask[1]
        assert np.all(np.isfinite(res.spectra))
        assert "pre_normalization" in res.stage_arrays
        return 0.0, "run_preprocessing dead-pixel, cosmic, baseline, interp, norm invariants verified"

    def _t1_h4() -> tuple[float | None, str]:
        store_base = tmp_dir / "t1_h4_store"
        parsed_dir, pre_dir, _, cfg = _build_synthetic_parsed_store(store_base, n_subjects=4, n_pixels_per_map=12)
        mat = build_preprocessed_store(cfg, force=True)
        assert mat.manifest["preprocessing_config_hash"] == config_hash(cfg["preprocessing"])
        src = DataSource(source_name="preprocessed", source_path=pre_dir, manifest=mat.manifest, manifest_hash=mat.manifest.get("manifest_hash"), was_built=True)
        summ_1300 = load_area_summaries(src, "1300")
        assert summ_1300["area_summary"].shape[0] == 4

        # Independent serial per-measurement run_preprocessing reference
        ingested = IngestedRamanDataset(
            DataSource(source_name="parsed", source_path=parsed_dir, manifest=None, manifest_hash=None, was_built=False)
        )
        meas_df = pd.read_parquet(parsed_dir / "measurements.parquet")
        rows_by_id = meas_df.set_index("measurement_id", drop=False)
        serial_items: list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray]] = []
        for view in ingested.iter_measurements():
            code = view.spectral_range
            row = rows_by_id.loc[view.measurement_id]
            res = run_preprocessing(
                view.raw_spectra,
                view.native_wavenumbers,
                region=_range_key(code),
                pixel_coords=view.coordinates,
                screen_dead=True,
                suppress_cosmic=None,
                correct_bl=True,
                smooth=None,
                interpolate=True,
                target_wavenumbers=_frozen_target_axis(cfg, code),
                filter_quality=None,
                normalize=True,
                norm_for="eda",
                return_pre_normalization=True,
                cfg=cfg,
            )
            serial_items.append(
                (
                    str(row["subject_id"]),
                    code,
                    np.asarray(res.spectra, dtype=float),
                    np.asarray(res.stage_arrays["pre_normalization"], dtype=float),
                    np.asarray(res.retained_mask, dtype=bool),
                )
            )
        min_area = int(cfg["preprocessing"]["retention"]["min_pixels_per_area"])
        min_subj = int(cfg["preprocessing"]["retention"]["min_pixels_per_subject"])
        subj_counts: dict[tuple[str, str], int] = {}
        for sid, code, _, _, rmask in serial_items:
            if int(rmask.sum()) >= min_area:
                subj_counts[(sid, code)] = subj_counts.get((sid, code), 0) + int(rmask.sum())
        ref_norm: dict[str, list[np.ndarray]] = {"1300": [], "2850": []}
        ref_pre: dict[str, list[np.ndarray]] = {"1300": [], "2850": []}
        for sid, code, spec_arr, pre_arr, rmask in serial_items:
            if int(rmask.sum()) >= min_area and subj_counts.get((sid, code), 0) >= min_subj:
                ref_norm[code].append(spec_arr)
                ref_pre[code].append(pre_arr)
        max_d = 0.0
        for code in ("1300", "2850"):
            max_d = max(
                max_d,
                _assert_close(np.load(pre_dir / f"normalized_{code}.npy"), np.vstack(ref_norm[code]), atol=1e-10, label=f"T1-H4 norm_{code}"),
                _assert_close(np.load(pre_dir / f"pre_normalization_{code}.npy"), np.vstack(ref_pre[code]), atol=1e-10, label=f"T1-H4 pre_{code}"),
            )
        if "n_jobs" in inspect.signature(_process_measurements).parameters:
            arr1, _, _ = _process_measurements(parsed_dir, meas_df, cfg, n_jobs=1)  # type: ignore[call-arg]
            arr2, _, _ = _process_measurements(parsed_dir, meas_df, cfg, n_jobs=2)  # type: ignore[call-arg]
            for code in ("1300", "2850"):
                max_d = max(
                    max_d,
                    _assert_close(arr2[f"normalized_{code}"], arr1[f"normalized_{code}"], atol=1e-10, label=f"T1-H4 n_jobs norm_{code}"),
                )
        return max_d, "build_preprocessed_store & serial per-map run_preprocessing parity verified"

    def _t1_h5() -> tuple[float | None, str]:
        rng = np.random.default_rng(504)
        mm_path = tmp_dir / "t1_h5_mm.dat"
        mm = np.memmap(mm_path, dtype=np.float64, mode="w+", shape=(200, 50))
        arr = rng.normal(size=(200, 50))
        arr[10, 5] = np.nan
        mm[:] = arr
        mm.flush()
        mm_ro = np.memmap(mm_path, dtype=np.float64, mode="r", shape=(200, 50))
        order_ids = rng.permutation([f"M_{i // 10:02d}" for i in range(200)])
        row_idx = np.arange(200, dtype=int)
        row_idx[::17] = -1
        rows = pd.DataFrame({"measurement_id": order_ids, "row_col": row_idx})
        mids, sums = _summarize_rows(rows, mm_ro, "row_col")
        usable = rows.loc[rows["row_col"] >= 0]
        groups = list(usable.groupby("measurement_id", sort=False))
        ref_ids = np.asarray([str(k) for k, _ in groups], dtype=str)
        ref_sums = np.vstack([np.nanmedian(arr[g["row_col"].to_numpy(dtype=int)], axis=0) for _, g in groups])
        assert np.array_equal(mids, ref_ids)
        d = _assert_close(sums, ref_sums, atol=1e-14, label="T1-H5 _summarize_rows")
        del mm_ro, mm
        return d, "_summarize_rows on memmap matches encounter-order nanmedian reference"

    def _t2_07() -> tuple[float | None, str]:
        wn = np.linspace(800.0, 1750.0, 100)
        cfg = _make_eda_cfg()
        empty_res = run_preprocessing(np.empty((0, 100), dtype=float), wn, region="fingerprint", cfg=cfg)
        assert empty_res.spectra.shape[0] == 0
        # Short 5-channel snip
        short_s = np.ones((2, 5), dtype=float)
        _, short_bl = correct_baseline(short_s, method="snip", return_baseline=True)
        assert short_bl.shape == (2, 5)
        # All -1 rows in _summarize_rows
        rows_none = pd.DataFrame({"measurement_id": ["a", "b"], "row_col": [-1, -1]})
        m_empty, s_empty = _summarize_rows(rows_none, np.zeros((4, 10), dtype=float), "row_col")
        assert m_empty.size == 0 and s_empty.shape == (0, 10)

        # pixel_coords=None and NaN-row pixel_coords local cosmic fallback
        rng_fb = np.random.default_rng(508)
        wn_fb = np.linspace(800.0, 1750.0, 60)
        spec_fb = 50.0 + 100.0 * np.exp(-0.5 * ((wn_fb - 1450.0) / 15.0) ** 2)[None, :] + rng_fb.normal(scale=0.5, size=(6, 60))
        spec_fb[1, 30] += 800.0
        cfg_fb = copy.deepcopy(cfg)
        cfg_fb["preprocessing"]["cosmic_ray"]["max_corrected_fraction"] = 0.5
        cfg_fb["preprocessing"]["smoothing"]["enabled"] = False
        res_none = run_preprocessing(
            spec_fb, wn_fb, region="fingerprint", pixel_coords=None, screen_dead=False, suppress_cosmic=True, correct_bl=False, smooth=False, interpolate=False, filter_quality=False, normalize=False, trim=0, trim_cm=0.0, cfg=cfg_fb
        )
        coords_nan = np.column_stack([np.arange(6, dtype=float), np.zeros(6, dtype=float)])
        coords_nan[0, 0] = np.nan
        res_nan = run_preprocessing(
            spec_fb, wn_fb, region="fingerprint", pixel_coords=coords_nan, screen_dead=False, suppress_cosmic=True, correct_bl=False, smooth=False, interpolate=False, filter_quality=False, normalize=False, trim=0, trim_cm=0.0, cfg=cfg_fb
        )
        cr_cfg = cfg_fb["preprocessing"]["cosmic_ray"]
        ref_mask, ref_repl = _local_cosmic_detection(
            spec_fb, z_threshold=float(cr_cfg["z_threshold"]), window=int(cr_cfg["window"]), max_width=int(cr_cfg["max_width"])
        )
        ref_corr = np.where(ref_mask, ref_repl, spec_fb)
        d1 = _assert_close(res_none.spectra, ref_corr, atol=1e-13, label="T2-07 pixel_coords=None")
        d2 = _assert_close(res_nan.spectra, ref_corr, atol=1e-13, label="T2-07 pixel_coords NaN")

        # > 2% excessive cosmic rejection (3 spikes / 60 ch = 0.05 > 0.02)
        spec_exc = spec_fb.copy()
        spec_exc[0, [10, 25, 45]] += 900.0
        cfg_exc = copy.deepcopy(cfg_fb)
        cfg_exc["preprocessing"]["cosmic_ray"]["max_corrected_fraction"] = 0.02
        res_exc = run_preprocessing(
            spec_exc, wn_fb, region="fingerprint", pixel_coords=None, screen_dead=False, suppress_cosmic=True, correct_bl=False, smooth=False, interpolate=False, filter_quality=False, normalize=False, trim=0, trim_cm=0.0, cfg=cfg_exc
        )
        assert not res_exc.retained_mask[0]
        assert "excessive_cosmic_correction" in str(res_exc.qc_flags.loc[0, "qc_reason"])
        return max(d1, d2), "Empty map, short snip, all -1 _summarize_rows, local cosmic fallback, & >2% rejection verified"

    def _t3_04() -> tuple[float | None, str]:
        from src.analysis.biological_pca import (
            build_subject_inputs,
            fit_biological_pca_models,
            run_measurement_count_sensitivity,
        )
        from src.analysis.clinical_feature_associations import (
            compute_binary_group_feature_tests,
            compute_spearman_feature_tests,
            feature_metadata,
        )
        from src.analysis.control_vs_disease_ml import run_control_vs_disease_ml_pipeline
        from src.analysis.disease_subgroups import run_disease_subgroups
        from src.analysis.nmf_decomposition import run_nmf_decomposition
        from src.analysis.subject_features import write_subject_feature_exports
        from src.statistics.cluster_permutation import (
            apply_binary_cluster_permutation,
            apply_spearman_cluster_permutation,
        )

        chain_dir = tmp_dir / "t3_04_chain"
        _, pre_dir, _, cfg = _build_synthetic_parsed_store(
            chain_dir,
            n_subjects=28,
            n_pixels_per_map=12,
            n_measurements_per_subject=(1, 2),
            add_two_disease_clusters=True,
        )
        mat = build_preprocessed_store(cfg, force=True)
        src = DataSource(source_name="preprocessed", source_path=pre_dir, manifest=mat.manifest, manifest_hash=mat.manifest.get("manifest_hash"), was_built=True)
        a_1300 = load_area_summaries(src, "1300")
        a_2850 = load_area_summaries(src, "2850")
        assert a_1300["area_summary"].shape[0] > 0 and a_2850["area_summary"].shape[0] > 0

        inputs = build_subject_inputs(cfg=cfg, data_source=src)
        pca_res = fit_biological_pca_models(inputs, cfg=cfg)
        assert len(pca_res) == 3
        sens = run_measurement_count_sensitivity(inputs, pca_res, cfg=cfg)
        assert not sens.empty

        exported = write_subject_feature_exports(inputs, output_dir=chain_dir / "subject_features", cfg=cfg, data_source=src)
        fm_path = exported["subject_feature_matrix_whole"]
        fm_df = pd.read_parquet(fm_path)
        fcols = [c for c in fm_df.columns if c.startswith(("1300__", "2850__"))]
        fmeta = feature_metadata(fcols)

        bin_tests, _ = compute_binary_group_feature_tests(
            fm_df, fmeta, group_column="condition", ordered_groups=("control", "disease"), contrast_name="ctrl_vs_dis", min_subjects_per_group=5
        )
        bin_perm = apply_binary_cluster_permutation(
            bin_tests, fm_df, group_column="condition", ordered_groups=("control", "disease"), n_permutations=10, cluster_alpha=0.05, random_state=42, n_jobs=1
        )
        assert "cluster_p_value" in bin_perm.columns

        sp_tests = compute_spearman_feature_tests(fm_df, fmeta, target="moca_baseline", min_subjects=5)
        sp_perm = apply_spearman_cluster_permutation(
            sp_tests, fm_df, target="moca_baseline", n_permutations=10, cluster_alpha=0.05, random_state=42, n_jobs=1
        )
        assert "cluster_p_value" in sp_perm.columns

        cand_path = chain_dir / "integrated_candidates.csv"
        pd.DataFrame(
            [
                {
                    "candidate_region_id": "cand_1525",
                    "spectral_block": "1300",
                    "start_cm-1": 1510.0,
                    "end_cm-1": 1535.0,
                    "evidence_status": "cluster_supported_exploratory",
                },
                {
                    "candidate_region_id": "cand_2853",
                    "spectral_block": "2850",
                    "start_cm-1": 2845.0,
                    "end_cm-1": 2860.0,
                    "evidence_status": "cluster_supported_exploratory",
                },
            ]
        ).to_csv(cand_path, index=False)

        nmf_out = run_nmf_decomposition(feature_matrix_path=fm_path, output_dir=chain_dir / "nmf", cfg=cfg)
        sub_out = run_disease_subgroups(
            feature_matrix_path=fm_path, integrated_candidates_path=cand_path, output_dir=chain_dir / "sub", cfg=cfg
        )
        assert "nmf_components" in nmf_out and nmf_out["nmf_components"].is_file()
        assert "disease_subgroup_assignments" in sub_out and sub_out["disease_subgroup_assignments"].is_file()

        ml_out = run_control_vs_disease_ml_pipeline(
            feature_matrix_path=fm_path,
            output_dir=chain_dir / "ml",
            cfg=cfg,
            data_manifest_hash=mat.manifest.get("manifest_hash"),
        )
        assert "model_comparison" in ml_out and ml_out["model_comparison"].is_file()
        return None, "Parsed -> Preprocessed Store -> Loaders -> BioPCA -> SubjectExport -> ClusterPerm -> NMF/Subgroups -> ML chain verified"

    results.append(_run_check("T1-H1", "preprocessing_store", "tier1", _t1_h1))
    results.append(_run_check("T1-H2", "preprocessing_store", "tier1", _t1_h2))
    results.append(_run_check("T1-H3", "preprocessing_store", "tier1", _t1_h3))
    results.append(_run_check("T1-H4", "preprocessing_store", "tier1", _t1_h4))
    results.append(_run_check("T1-H5", "preprocessing_store", "tier1", _t1_h5))
    results.append(_run_check("T2-07", "preprocessing_store", "tier2", _t2_07))
    results.append(_run_check("T3-04", "preprocessing_store", "tier3", _t3_04))
    return results


# ---------------------------------------------------------------------------
# Tier 4 Benchmark Workloads (B-01..B-13)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkSpec:
    bench_id: str
    suite: str
    feature_ids: list[int]
    name: str
    warmup_smoke: int
    measured_smoke: int
    warmup_full: int
    measured_full: int


BENCHMARK_SPECS: tuple[BenchmarkSpec, ...] = (
    BenchmarkSpec("B-01", "paired_lipid", [1, 3], "run_multiplicity_bootstrap_gpu_decayed", 0, 1, 1, 3),
    BenchmarkSpec("B-02", "paired_lipid", [1, 3], "run_multiplicity_bootstrap_gpu_undecayed", 0, 1, 1, 3),
    BenchmarkSpec("B-03", "paired_lipid", [2, 3], "run_multiplicity_bootstrap_cpu", 0, 1, 0, 2),
    BenchmarkSpec("B-04", "paired_lipid", [4], "external_cohort_prep_and_score", 0, 1, 1, 3),
    BenchmarkSpec("B-05", "paired_lipid", [4], "bootstrap_calibration_loky", 0, 1, 0, 2),
    BenchmarkSpec("B-06", "manuscript_ml", [5, 6, 7], "run_sensitivity_and_secondary", 0, 1, 0, 2),
    BenchmarkSpec("B-07", "biological_pca", [8], "measurement_count_sensitivity", 0, 1, 1, 3),
    BenchmarkSpec("B-08", "cluster_permutation", [9], "cluster_permutation_binary_and_spearman", 0, 1, 1, 3),
    BenchmarkSpec("B-09", "control_vs_disease_ml", [10], "control_vs_disease_ml_rf_pls_l1", 0, 1, 0, 2),
    BenchmarkSpec("B-10", "nmf_subgroups", [11], "nmf_multi_init_and_disease_subgroups", 0, 1, 1, 3),
    BenchmarkSpec("B-11", "preprocessing_store", [12], "run_preprocessing_map", 0, 1, 1, 3),
    BenchmarkSpec("B-12", "preprocessing_store", [12], "materialize_process_measurements", 0, 1, 0, 2),
    BenchmarkSpec("B-13", "preprocessing_store", [13], "loaders_summarize_rows_memmap", 0, 1, 1, 3),
)


def _execute_single_benchmark(
    spec: BenchmarkSpec,
    *,
    profile: str,
    use_tracemalloc: bool,
    tmp_dir: Path,
) -> dict[str, Any]:
    from manuscript_ml.run_sensitivity_and_secondary import run_sensitivity_and_secondary
    from src.analysis.biological_pca import fit_biological_pca_models, run_measurement_count_sensitivity
    from src.analysis.control_vs_disease_ml import (
        _cv_predict,
        _locked_joint_stratified_split_plan,
        run_permutation_test,
    )
    from src.analysis.disease_subgroups import _bootstrap_ari, _cluster_labels, _gap_statistic
    from src.analysis.nmf_decomposition import _fit_nmf_multi_init
    from src.analysis.paired_lipid.calibration import evaluate_bootstrap_calibration
    from src.analysis.paired_lipid.cohort import PairedLipidConfig
    from src.analysis.paired_lipid.nested_cv import (
        PreparedExternalTest,
        evaluate_nested_cohort,
        run_multiplicity_bootstrap,
        score_external_primary_pair,
    )
    from src.analysis.paired_lipid.splits import make_nested_splits
    from src.analysis.paired_lipid.synthetic import generate_synthetic_cohort
    from src.data_preparation.loaders import _summarize_rows
    from src.data_preparation.materialize import _process_measurements
    from src.preprocessing.pipeline import run_preprocessing
    from src.statistics.cluster_permutation import _binary_cluster_maxima, _spearman_cluster_maxima

    is_smoke = profile == "smoke"
    warmup = spec.warmup_smoke if is_smoke else spec.warmup_full
    measured = spec.measured_smoke if is_smoke else spec.measured_full
    cfg = PairedLipidConfig()
    params: dict[str, Any] = {}
    parity_passed = True
    max_parity_diff: float | None = None
    notes = ""

    # Prepare workload closure
    if spec.bench_id in ("B-01", "B-02", "B-03"):
        cp_avail = _probe_cupy() is not None
        want_gpu = spec.bench_id in ("B-01", "B-02")
        if want_gpu and not cp_avail:
            return {
                "bench_id": spec.bench_id,
                "feature_ids": spec.feature_ids,
                "name": spec.name,
                "params": {"profile": profile, "skipped": True},
                "warmup_trials": 0,
                "measured_trials": 0,
                "trial_wall_seconds": [],
                "median_wall_seconds": 0.0,
                "min_wall_seconds": 0.0,
                "iqr_wall_seconds": 0.0,
                "host_peak_rss_mib": 0.0,
                "host_rss_delta_mib": 0.0,
                "peak_tree_ws_mib": 0.0,
                "peak_tree_private_mib": 0.0,
                "tracemalloc_peak_mib": None,
                "gpu_pool_peak_mib": None,
                "gpu_vram_delta_mib": None,
                "parity_passed": True,
                "max_parity_diff": None,
                "notes": "CuPy/CUDA absent; GPU workload skipped",
            }
        n_p = 32 if is_smoke else 92
        nuis = 0 if spec.bench_id == "B-02" else 16
        n_boot = 8 if is_smoke else (512 if spec.bench_id == "B-02" else 2000)
        bs = 256 if is_smoke else (512 if spec.bench_id == "B-02" else 2048)
        ds, _ = generate_synthetic_cohort(
            n_participants=n_p,
            n_single_sample=4 if not is_smoke else 2,
            seed=42,
            compact_grid=is_smoke,
            n_nuisance_modes=nuis,
            nuisance_part_scale=0.60 if nuis else 0.0,
            nuisance_sample_scale=1.50 if nuis else 0.0,
        )
        params = {"n": n_p, "B": n_boot, "compact_grid": is_smoke, "nuis": nuis, "gpu": want_gpu, "batch_size": bs}

        def _run_b123() -> Any:
            return run_multiplicity_bootstrap(
                ds, cfg=cfg, n_bootstraps=n_boot, bootstrap_seed=20261106, use_gpu=want_gpu, batch_size=bs
            )

        def _check_b123(res: Any) -> float:
            if want_gpu:
                assert int(res["n_fallback_draws"]) == 0, (
                    f"{spec.bench_id}: unexpected GPU fallback draws: {res['n_fallback_draws']}"
                )
            assert len(res["bootstrap_draws_delta_mse"]) == n_boot, (
                f"{spec.bench_id}: expected {n_boot} bootstrap draws, got {len(res['bootstrap_draws_delta_mse'])}"
            )
            splits = make_nested_splits(ds, cfg, cfg.primary_seed)
            seq = evaluate_nested_cohort(
                ds, cfg=cfg, precomputed_splits=splits, include_band_paired=True, compute_diagnostics=False
            )
            max_d = _assert_close(
                res["point_estimate_delta_mse"],
                seq["comparisons_vs_covariate_ridge"]["paired_linear_ridge"]["delta_mse"],
                atol=1e-11,
                label=f"{spec.bench_id} point",
            )
            rng_ref = np.random.default_rng(20261106)
            drawn_mat = rng_ref.integers(0, ds.n_participants, size=(n_boot, ds.n_participants))
            check_indices = sorted({0, n_boot - 1}) if n_boot >= 2 else ([0] if n_boot == 1 else [])
            for b_idx in check_indices:
                mult = np.zeros(ds.n_participants, dtype=float)
                np.add.at(mult, drawn_mat[b_idx], 1.0)
                seq_b = evaluate_nested_cohort(
                    ds,
                    cfg=cfg,
                    participant_multiplicities=mult,
                    precomputed_splits=splits,
                    include_band_paired=True,
                    compute_diagnostics=False,
                )
                max_d = max(
                    max_d,
                    _assert_close(
                        res["bootstrap_draws_delta_mse"][b_idx],
                        seq_b["comparisons_vs_covariate_ridge"]["paired_linear_ridge"]["delta_mse"],
                        atol=1e-10,
                        label=f"{spec.bench_id} draw {b_idx}",
                    ),
                )
            return max_d

        run_fn, check_fn = _run_b123, _check_b123

    elif spec.bench_id == "B-04":
        n_tr = 32 if is_smoke else 92
        n_te = 100 if is_smoke else 25000
        ds_tr, _ = generate_synthetic_cohort(n_participants=n_tr, n_single_sample=2, seed=42, compact_grid=is_smoke, n_nuisance_modes=16, nuisance_part_scale=0.6, nuisance_sample_scale=1.5)
        splits = make_nested_splits(ds_tr, cfg, cfg.primary_seed)
        params = {"n_tr": n_tr, "n_te": n_te, "compact_grid": is_smoke, "nuis": 16}

        def _run_b4() -> Any:
            ds_te, _ = generate_synthetic_cohort(n_participants=n_te, n_single_sample=0, seed=9001, compact_grid=is_smoke, n_nuisance_modes=16, nuisance_part_scale=0.6, nuisance_sample_scale=1.5)
            prep = PreparedExternalTest.from_dataset(ds_te, cfg.primary_endpoint, preset=cfg.covariate_preset)
            assert prep.raw_cov_frame is None
            return score_external_primary_pair(ds_tr, prep, cfg=cfg, splits=splits)

        def _check_b4(res: Any) -> float | None:
            assert math.isfinite(res["external_delta_mse"])
            return None

        run_fn, check_fn = _run_b4, _check_b4

    elif spec.bench_id == "B-05":
        r_reps = 2 if is_smoke else 8
        b_draws = 4 if is_smoke else 64
        n_p = 32 if is_smoke else 92
        n_te = 100 if is_smoke else 25000
        n_j = 2 if is_smoke else -1
        params = {"R": r_reps, "B": b_draws, "n": n_p, "n_te": n_te, "n_jobs": n_j, "compact_grid": is_smoke}

        def _run_b5() -> Any:
            return evaluate_bootstrap_calibration(
                n_repetitions=r_reps, n_bootstraps=b_draws, n_participants=n_p, n_test_participants=n_te, n_jobs=n_j, use_gpu=False, compact_grid=is_smoke
            )

        def _check_b5(res: Any) -> float | None:
            assert math.isfinite(res["target_delta_mse"]) and math.isfinite(res["mean_point_estimate"])
            return None

        run_fn, check_fn = _run_b5, _check_b5

    elif spec.bench_id == "B-06":
        n_b = 2 if is_smoke else 64
        params = {"synthetic": True, "compact_grid": is_smoke, "n_boot": n_b}

        def _run_b6() -> Any:
            return run_sensitivity_and_secondary(
                use_synthetic=True, output_dir=tmp_dir / "b06_out", compact_grid=is_smoke, n_bootstraps=n_b, use_gpu=False
            )

        def _check_b6(res: Any) -> float | None:
            sens_path = res["covariate_and_subset_sensitivity"]
            assert sens_path.is_file()
            sens_df = pd.read_csv(sens_path)
            assert "delta_mse_paired" in sens_df.columns and np.all(np.isfinite(sens_df["delta_mse_paired"]))
            return None

        run_fn, check_fn = _run_b6, _check_b6

    elif spec.bench_id == "B-07":
        n_c = 8 if is_smoke else 30
        n_d = 16 if is_smoke else 62
        p1 = 60 if is_smoke else 1055
        p2 = 50 if is_smoke else 747
        n_draws = 4 if is_smoke else 100
        inputs = _make_synthetic_subject_inputs(n_control=n_c, n_disease=n_d, p_1300=p1, p_2850=p2, seed=42)
        b_cfg = _make_eda_cfg(n_pca_components=3, n_draws=n_draws)
        can = fit_biological_pca_models(inputs, cfg=b_cfg)
        params = {"n": n_c + n_d, "p": [p1, p2], "draws": n_draws}

        def _run_b7() -> Any:
            return run_measurement_count_sensitivity(inputs, can, cfg=b_cfg)

        def _check_b7(res: Any) -> float | None:
            assert len(res) == n_draws * 9
            assert np.all(np.isfinite(res["explained_variance_ratio"]))
            return None

        run_fn, check_fn = _run_b7, _check_b7

    elif spec.bench_id == "B-08":
        n_p = 30 if is_smoke else 92
        p_ch = 80 if is_smoke else 1055
        n_perm = 10 if is_smoke else 200
        rng = np.random.default_rng(808)
        vals = rng.normal(size=(n_p, p_ch))
        labels = np.array([0] * (n_p // 2) + [1] * (n_p - n_p // 2), dtype=int)
        tgt = rng.normal(size=n_p)
        params = {"shape": [n_p, p_ch], "perms": n_perm, "n_jobs": 1}

        def _run_b8() -> Any:
            b_m = _binary_cluster_maxima(vals, labels, n_permutations=n_perm, cluster_alpha=0.05, rng=np.random.default_rng(42), n_jobs=1)
            s_m = _spearman_cluster_maxima(vals, tgt, n_permutations=n_perm, cluster_alpha=0.05, rng=np.random.default_rng(42), n_jobs=1)
            return b_m, s_m

        def _check_b8(res: Any) -> float | None:
            assert len(res[0]) == n_perm and len(res[1]) == n_perm
            assert np.all(np.isfinite(res[0])) and np.all(np.isfinite(res[1]))
            return None

        run_fn, check_fn = _run_b8, _check_b8

    elif spec.bench_id == "B-09":
        n_p = 40 if is_smoke else 140
        p_ch = 60 if is_smoke else 1802
        n_out = 3 if is_smoke else 5
        n_in = 2 if is_smoke else 4
        n_perm = 4 if is_smoke else 50
        rng = np.random.default_rng(809)
        X = rng.normal(size=(n_p, p_ch))
        y = np.array([0] * (n_p // 2) + [1] * (n_p - n_p // 2), dtype=int)
        sex = np.array(["female" if i % 2 == 0 else "male" for i in range(n_p)], dtype=object)
        X[:, :6] += 1.2 * y[:, None]
        ml_cfg = _make_eda_cfg(n_outer_splits=n_out, n_outer_repeats=1, n_inner_splits=n_in, n_permutations=n_perm, n_jobs=1)["analysis"]["control_vs_disease_ml"]
        if not is_smoke:
            ml_cfg["rf_n_estimators_grid"] = [200, 500]
            ml_cfg["rf_max_depth_grid"] = [None, 4, 8]
            ml_cfg["pls_components_grid"] = [1, 2, 3, 4, 5, 6, 8, 10]
            ml_cfg["logistic_c_grid"] = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
        sp = _locked_joint_stratified_split_plan(y, sex, n_outer_splits=n_out, n_outer_repeats=1, random_state=42)
        params = {"shape": [n_p, p_ch], "outer": n_out, "inner": n_in, "perm": n_perm}

        def _run_b9() -> Any:
            _cv_predict("random_forest", X, y, sp[0], ml_cfg=ml_cfg, n_inner_splits=n_in, threshold_metric="balanced_accuracy", random_state=42, n_jobs=-1 if not is_smoke else 1)
            return run_permutation_test("pls_da", X, y, observed_balanced_accuracy=0.85, strata=sex, ml_cfg=ml_cfg, n_jobs=-1 if not is_smoke else 1, random_state=42, outer_split_plan=sp)

        def _check_b9(res: Any) -> float | None:
            assert math.isfinite(res[0]) and np.all(np.isfinite(res[1]))
            return None

        run_fn, check_fn = _run_b9, _check_b9

    elif spec.bench_id == "B-10":
        n_p = 30 if is_smoke else 92
        p_ch = 80 if is_smoke else 1055
        n_init = 2 if is_smoke else 20
        n_boot = 2 if is_smoke else 50
        rng = np.random.default_rng(810)
        X_nmf = rng.uniform(0.1, 2.0, size=(n_p, p_ch))
        X_sub = rng.normal(size=(n_p, 20))
        lbl = _cluster_labels(X_sub, 3)
        params = {"shape": [n_p, p_ch], "n_init": n_init, "boot": n_boot}

        def _run_b10() -> Any:
            nmf_r = _fit_nmf_multi_init(X_nmf, n_components=3, alpha_w=0.1, alpha_h=0.0, n_init=n_init, random_state=42)
            gap_r = _gap_statistic(X_sub, k_values=[1, 2, 3], n_bootstrap=n_boot, random_state=42)
            ari_r = _bootstrap_ari(X_sub, lbl, n_clusters=3, n_bootstrap=n_boot, random_state=42)
            return nmf_r, gap_r, ari_r

        def _check_b10(res: Any) -> float | None:
            assert math.isfinite(res[0].best_error) and not res[1].empty and not res[2].empty
            assert np.all(np.isfinite(res[1]["gap"])) and np.all(np.isfinite(res[2]["mean_bootstrap_ari"]))
            return None

        run_fn, check_fn = _run_b10, _check_b10

    elif spec.bench_id == "B-11":
        n_pix = 24 if is_smoke else 400
        n_ch = 120 if is_smoke else 1055
        rng = np.random.default_rng(811)
        wn = np.linspace(785.0, 1765.0, n_ch)
        spec_arr = 30.0 + 150.0 * np.exp(-0.5 * ((wn - 1450.0) / 20.0) ** 2)[None, :] + rng.normal(scale=1.0, size=(n_pix, n_ch))
        coords = np.column_stack([np.repeat(np.arange(int(math.sqrt(n_pix)) + 1), int(math.sqrt(n_pix)) + 1)[:n_pix], np.tile(np.arange(int(math.sqrt(n_pix)) + 1), int(math.sqrt(n_pix)) + 1)[:n_pix]]).astype(float)
        p_cfg = _make_eda_cfg()
        params = {"shape": [n_pix, n_ch], "method": "arpls"}

        def _run_b11() -> Any:
            return run_preprocessing(spec_arr, wn, region="fingerprint", pixel_coords=coords, interpolate=True, target_wavenumbers=np.arange(800.0, 1751.0, 5.0 if is_smoke else 1.0), return_pre_normalization=True, cfg=p_cfg)

        def _check_b11(res: Any) -> float | None:
            assert np.any(res.retained_mask) and np.all(np.isfinite(res.spectra))
            return None

        run_fn, check_fn = _run_b11, _check_b11

    elif spec.bench_id == "B-12":
        n_subj = 2 if is_smoke else 16
        n_pix = 20 if is_smoke else 400
        n_ch1 = 100 if is_smoke else 1055
        n_ch2 = 60 if is_smoke else 747
        store_dir = tmp_dir / "b12_store"
        parsed_dir, _, _, m_cfg = _build_synthetic_parsed_store(store_dir, n_subjects=n_subj, n_pixels_per_map=n_pix, n_ch_1300=n_ch1, n_ch_2850=n_ch2)
        if not is_smoke:
            m_cfg["preprocessing"]["interpolation"]["target_spacing"] = 1.0
        meas_df = pd.read_parquet(parsed_dir / "measurements.parquet")
        params = {"n_maps": int(len(meas_df)), "n_pix": n_pix}

        def _run_b12() -> Any:
            return _process_measurements(parsed_dir, meas_df, m_cfg)

        def _check_b12(res: Any) -> float | None:
            assert res[0]["normalized_1300"].shape[0] > 0
            assert np.all(np.isfinite(res[0]["normalized_1300"])) and np.all(np.isfinite(res[0]["normalized_2850"]))
            return None

        run_fn, check_fn = _run_b12, _check_b12

    elif spec.bench_id == "B-13":
        n_rows = 120 if is_smoke else 100000
        n_ch = 80 if is_smoke else 1055
        n_grp = 8 if is_smoke else 360
        tmp_dir.mkdir(parents=True, exist_ok=True)
        mm_path = tmp_dir / f"b13_{profile}.dat"
        rng = np.random.default_rng(813)
        mm_w = np.memmap(mm_path, dtype=np.float64, mode="w+", shape=(n_rows, n_ch))
        mm_w[:] = rng.normal(size=(n_rows, n_ch))
        mm_w.flush()
        del mm_w
        mm_ro = np.memmap(mm_path, dtype=np.float64, mode="r", shape=(n_rows, n_ch))
        rows_df = pd.DataFrame({"measurement_id": [f"M_{i % n_grp:03d}" for i in range(n_rows)], "normalized_row": np.arange(n_rows, dtype=int)})
        params = {"shape": [n_rows, n_ch], "n_groups": n_grp}

        def _run_b13() -> Any:
            return _summarize_rows(rows_df, mm_ro, "normalized_row")

        def _check_b13(res: Any) -> float | None:
            assert len(res[0]) == n_grp and np.all(np.isfinite(res[1]))
            return None

        run_fn, check_fn = _run_b13, _check_b13
    else:
        raise ValueError(f"Unknown bench_id: {spec.bench_id}")

    trial_times: list[float] = []
    try:
        for _ in range(warmup):
            _ = run_fn()

        last_result: Any = None
        with _ResourceMonitor(use_tracemalloc=use_tracemalloc, probe_gpu=True) as mon:
            for _ in range(measured):
                t0 = time.perf_counter()
                last_result = run_fn()
                trial_times.append(round(time.perf_counter() - t0, 5))
            snap = mon.snapshot()

        try:
            max_parity_diff = check_fn(last_result)
            parity_passed = True
        except Exception as exc:
            parity_passed = False
            max_parity_diff = None
            notes = f"{type(exc).__name__}: {exc}"
    finally:
        if spec.bench_id == "B-13":
            mmap_obj = getattr(mm_ro, "_mmap", None)
            if mmap_obj is not None:
                mmap_obj.close()
            del run_fn
            del mm_ro

    med_t = round(float(statistics.median(trial_times)), 5) if trial_times else 0.0
    min_t = round(float(min(trial_times)), 5) if trial_times else 0.0
    if len(trial_times) >= 2:
        q = np.percentile(trial_times, [25.0, 75.0])
        iqr_t = round(float(q[1] - q[0]), 5)
    else:
        iqr_t = 0.0

    return {
        "bench_id": spec.bench_id,
        "feature_ids": spec.feature_ids,
        "name": spec.name,
        "params": params,
        "warmup_trials": warmup,
        "measured_trials": measured,
        "trial_wall_seconds": trial_times,
        "median_wall_seconds": med_t,
        "min_wall_seconds": min_t,
        "iqr_wall_seconds": iqr_t,
        "host_peak_rss_mib": snap.host_peak_rss_mib,
        "host_rss_delta_mib": snap.host_rss_delta_mib,
        "peak_tree_ws_mib": snap.peak_tree_ws_mib,
        "peak_tree_private_mib": snap.peak_tree_private_mib,
        "tracemalloc_peak_mib": snap.tracemalloc_peak_mib,
        "gpu_pool_peak_mib": snap.gpu_pool_peak_mib,
        "gpu_vram_delta_mib": snap.gpu_vram_delta_mib,
        "parity_passed": bool(parity_passed),
        "max_parity_diff": max_parity_diff,
        "notes": notes,
    }


def _run_benchmark_isolated(
    spec: BenchmarkSpec,
    *,
    profile: str,
    use_tracemalloc: bool,
    output_dir: Path,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--isolated-worker",
        "--workload",
        spec.bench_id,
        "--profile",
        profile,
        "--output-dir",
        str(output_dir),
    ]
    if use_tracemalloc:
        cmd.append("--tracemalloc")
    eff_timeout = timeout_seconds if timeout_seconds is not None else (120.0 if profile == "smoke" else 1800.0)
    worker_json = output_dir / f"worker_{spec.bench_id}.json"
    worker_json.unlink(missing_ok=True)
    try:
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=eff_timeout)
    except subprocess.TimeoutExpired:
        worker_json.unlink(missing_ok=True)
        return {
            "bench_id": spec.bench_id,
            "feature_ids": spec.feature_ids,
            "name": spec.name,
            "params": {"profile": profile},
            "warmup_trials": 0,
            "measured_trials": 0,
            "trial_wall_seconds": [],
            "median_wall_seconds": 0.0,
            "min_wall_seconds": 0.0,
            "iqr_wall_seconds": 0.0,
            "host_peak_rss_mib": 0.0,
            "host_rss_delta_mib": 0.0,
            "peak_tree_ws_mib": 0.0,
            "peak_tree_private_mib": 0.0,
            "tracemalloc_peak_mib": None,
            "gpu_pool_peak_mib": None,
            "gpu_vram_delta_mib": None,
            "parity_passed": False,
            "max_parity_diff": None,
            "notes": f"Worker timed out after {eff_timeout:.1f}s",
        }
    if worker_json.is_file():
        payload = json.loads(worker_json.read_text(encoding="utf-8"))
        worker_json.unlink(missing_ok=True)
        if proc.returncode != 0:
            payload["parity_passed"] = False
            extra_note = f"Worker exited rc={proc.returncode}"
            payload["notes"] = f"{payload.get('notes', '')}; {extra_note}".strip("; ")
        return payload
    return {
        "bench_id": spec.bench_id,
        "feature_ids": spec.feature_ids,
        "name": spec.name,
        "params": {"profile": profile},
        "warmup_trials": 0,
        "measured_trials": 0,
        "trial_wall_seconds": [],
        "median_wall_seconds": 0.0,
        "min_wall_seconds": 0.0,
        "iqr_wall_seconds": 0.0,
        "host_peak_rss_mib": 0.0,
        "host_rss_delta_mib": 0.0,
        "peak_tree_ws_mib": 0.0,
        "peak_tree_private_mib": 0.0,
        "tracemalloc_peak_mib": None,
        "gpu_pool_peak_mib": None,
        "gpu_vram_delta_mib": None,
        "parity_passed": False,
        "max_parity_diff": None,
        "notes": f"Worker exited rc={proc.returncode}: {proc.stderr[-300:]}",
    }


# ---------------------------------------------------------------------------
# Report Building & CLI Orchestration
# ---------------------------------------------------------------------------


def build_benchmark_reports(
    *,
    mode: str = "all",
    profile: str | None = None,
    suites: Sequence[str] = ALL_SUITES,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    use_tracemalloc: bool = False,
    include_gpu_smoke: bool = False,
    isolated_workers: bool | None = None,
) -> tuple[dict[str, Any], Path, Path]:
    """Execute selected parity checks and benchmark workloads, emitting atomic JSON + CSV."""
    resolved_profile = profile or ("smoke" if mode == "smoke" else "full")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_suites = tuple(suites)
    for s in selected_suites:
        if s not in ALL_SUITES:
            raise ValueError(f"Unknown suite {s!r}; expected subset of {ALL_SUITES}.")

    parity_results: list[ParityCheckResult] = []
    workloads: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(dir=out_dir, prefix="e2e_work_") as work_str:
        work_dir = Path(work_str)

        def _subdir(name: str) -> Path:
            p = work_dir / name
            p.mkdir(parents=True, exist_ok=True)
            return p

        try:
            smoke_snap: ResourceSnapshot | None = None
            with contextlib.ExitStack() as smoke_stack:
                smoke_mon = (
                    smoke_stack.enter_context(
                        _ResourceMonitor(use_tracemalloc=use_tracemalloc, probe_gpu=include_gpu_smoke)
                    )
                    if mode == "smoke"
                    else None
                )
                if mode in ("smoke", "parity", "all"):
                    suite_runners: tuple[tuple[str, Callable[[], list[ParityCheckResult]]], ...] = (
                        (
                            "paired_lipid",
                            lambda: _run_paired_lipid_checks(
                                mode=mode, profile=resolved_profile, include_gpu_smoke=include_gpu_smoke, tmp_dir=_subdir("pl")
                            ),
                        ),
                        ("manuscript_ml", lambda: _run_manuscript_ml_checks(mode=mode, profile=resolved_profile, tmp_dir=_subdir("mml"))),
                        ("biological_pca", lambda: _run_biological_pca_checks(mode=mode, profile=resolved_profile)),
                        ("cluster_permutation", lambda: _run_cluster_permutation_checks(mode=mode, profile=resolved_profile)),
                        (
                            "control_vs_disease_ml",
                            lambda: _run_control_vs_disease_ml_checks(mode=mode, profile=resolved_profile, tmp_dir=_subdir("ml")),
                        ),
                        ("nmf_subgroups", lambda: _run_nmf_subgroups_checks(mode=mode, profile=resolved_profile, tmp_dir=_subdir("nmf"))),
                        (
                            "preprocessing_store",
                            lambda: _run_preprocessing_store_checks(mode=mode, profile=resolved_profile, tmp_dir=_subdir("prep")),
                        ),
                    )
                    for suite_name, suite_fn in suite_runners:
                        if suite_name not in selected_suites:
                            continue
                        if resolved_profile == "full":
                            print(f"[e2e] parity suite={suite_name} profile={resolved_profile} start", file=sys.stderr, flush=True)
                        t_suite = time.perf_counter()
                        suite_res = suite_fn()
                        parity_results.extend(suite_res)
                        if resolved_profile == "full":
                            n_ok = sum(1 for item in suite_res if item.passed)
                            dt_suite = time.perf_counter() - t_suite
                            print(
                                f"[e2e] parity suite={suite_name} done passed={n_ok}/{len(suite_res)} elapsed={dt_suite:.2f}s",
                                file=sys.stderr,
                                flush=True,
                            )
                if smoke_mon is not None:
                    smoke_snap = smoke_mon.snapshot()

            if mode in ("benchmark", "all"):
                use_iso = (resolved_profile == "full") if isolated_workers is None else bool(isolated_workers)
                for spec in BENCHMARK_SPECS:
                    if spec.suite not in selected_suites:
                        continue
                    if resolved_profile == "full":
                        mode_tag = "isolated" if use_iso else "inproc"
                        print(
                            f"[e2e] benchmark {spec.bench_id} ({spec.name}) profile={resolved_profile} {mode_tag} start",
                            file=sys.stderr,
                            flush=True,
                        )
                    t_wl = time.perf_counter()
                    if use_iso:
                        wl = _run_benchmark_isolated(spec, profile=resolved_profile, use_tracemalloc=use_tracemalloc, output_dir=out_dir)
                    else:
                        wl = _execute_single_benchmark(spec, profile=resolved_profile, use_tracemalloc=use_tracemalloc, tmp_dir=_subdir(spec.bench_id))
                    if resolved_profile == "full":
                        dt_wl = time.perf_counter() - t_wl
                        print(
                            f"[e2e] benchmark {spec.bench_id} done parity_passed={wl['parity_passed']} "
                            f"median={wl['median_wall_seconds']:.3f}s wall={dt_wl:.2f}s",
                            file=sys.stderr,
                            flush=True,
                        )
                    workloads.append(wl)
            elif mode == "smoke":
                assert smoke_snap is not None
                total_dt = round(sum(c.duration_seconds for c in parity_results), 5)
                workloads.append(
                    {
                        "bench_id": "SMOKE-ALL",
                        "feature_ids": list(range(1, 15)),
                        "name": "smoke_all_domains",
                        "params": {"profile": "smoke", "suites": list(selected_suites)},
                        "warmup_trials": 0,
                        "measured_trials": 1,
                        "trial_wall_seconds": [total_dt],
                        "median_wall_seconds": total_dt,
                        "min_wall_seconds": total_dt,
                        "iqr_wall_seconds": 0.0,
                        "host_peak_rss_mib": smoke_snap.host_peak_rss_mib,
                        "host_rss_delta_mib": smoke_snap.host_rss_delta_mib,
                        "peak_tree_ws_mib": smoke_snap.peak_tree_ws_mib,
                        "peak_tree_private_mib": smoke_snap.peak_tree_private_mib,
                        "tracemalloc_peak_mib": smoke_snap.tracemalloc_peak_mib,
                        "gpu_pool_peak_mib": smoke_snap.gpu_pool_peak_mib,
                        "gpu_vram_delta_mib": smoke_snap.gpu_vram_delta_mib,
                        "parity_passed": all(c.passed for c in parity_results),
                        "max_parity_diff": max((c.max_diff for c in parity_results if c.max_diff is not None), default=None),
                        "notes": f"{len(parity_results)} smoke check(s) completed",
                    }
                )
        finally:
            from src.data_preparation.loaders import clear_loader_caches
            from src.data_preparation.store import set_active_data_source

            clear_loader_caches()
            set_active_data_source(None)  # type: ignore[arg-type]
            gc.collect()

    is_smoke_mode = mode == "smoke"
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "mode": mode,
        "profile": resolved_profile,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": _collect_git_metadata(check_dirty=not is_smoke_mode),
        "hardware": _collect_hardware_metadata(probe_gpu=(not is_smoke_mode) or include_gpu_smoke),
        "clinical_safety": {
            "synthetic_only": True,
            "real_clinical_outcomes_accessed": False,
        },
        "parity_checks": [c.to_dict() for c in parity_results],
        "workloads": workloads,
    }

    json_path = write_json_atomic(out_dir / "benchmark_e2e_results.json", report)
    csv_rows = [
        {
            "bench_id": w["bench_id"],
            "profile": resolved_profile,
            "feature_ids": ",".join(str(fid) for fid in w["feature_ids"]),
            "median_wall_seconds": w["median_wall_seconds"],
            "min_wall_seconds": w["min_wall_seconds"],
            "iqr_wall_seconds": w["iqr_wall_seconds"],
            "host_peak_rss_mib": w["host_peak_rss_mib"],
            "host_rss_delta_mib": w["host_rss_delta_mib"],
            "peak_tree_ws_mib": w["peak_tree_ws_mib"],
            "tracemalloc_peak_mib": w["tracemalloc_peak_mib"],
            "gpu_pool_peak_mib": w["gpu_pool_peak_mib"],
            "parity_passed": w["parity_passed"],
            "max_parity_diff": w["max_parity_diff"],
        }
        for w in workloads
    ]
    csv_df = pd.DataFrame(csv_rows, columns=CSV_COLUMNS)
    csv_path = write_csv_atomic(csv_df, out_dir / "benchmark_e2e_summary.csv", index=False)
    return report, json_path, csv_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Opaque-box E2E parity and benchmark suite for spectral-analysis.")
    parser.add_argument("--mode", choices=("smoke", "parity", "benchmark", "all"), default="all")
    parser.add_argument("--profile", choices=("smoke", "full"), default=None)
    parser.add_argument("--quick", action="store_true", help="Alias for --profile smoke.")
    parser.add_argument(
        "--suites",
        type=str,
        default=",".join(ALL_SUITES),
        help=f"Comma-separated suite list from {ALL_SUITES}.",
    )
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--tracemalloc", action="store_true", help="Enable Python heap peak telemetry (adds overhead).")
    parser.add_argument("--include-gpu-smoke", action="store_true", help="Include tiny GPU parity probe in --mode smoke.")
    parser.add_argument("--isolated-worker", action="store_true", help="Internal: run single --workload in clean subprocess.")
    parser.add_argument("--workload", type=str, default=None, help="Workload bench_id (e.g. B-01) for --isolated-worker.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    profile = "smoke" if args.quick else args.profile
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.isolated_worker:
        if not args.workload:
            raise ValueError("--workload is required with --isolated-worker")
        spec_map = {s.bench_id: s for s in BENCHMARK_SPECS}
        if args.workload not in spec_map:
            raise ValueError(f"Unknown workload {args.workload!r}; expected one of {sorted(spec_map)}")
        eff_profile = profile or "full"
        work_tmp = Path(tempfile.mkdtemp(dir=out_dir, prefix=f"worker_{args.workload}_"))
        try:
            res = _execute_single_benchmark(
                spec_map[args.workload], profile=eff_profile, use_tracemalloc=args.tracemalloc, tmp_dir=work_tmp
            )
            write_json_atomic(out_dir / f"worker_{args.workload}.json", res)
            return 0 if res.get("parity_passed", False) else 1
        finally:
            shutil.rmtree(work_tmp, ignore_errors=True)

    suites = [tok.strip() for tok in args.suites.split(",") if tok.strip()]
    report, json_path, csv_path = build_benchmark_reports(
        mode=args.mode,
        profile=profile,
        suites=suites,
        output_dir=out_dir,
        use_tracemalloc=args.tracemalloc,
        include_gpu_smoke=args.include_gpu_smoke,
    )
    parity_ok = all(item["passed"] for item in report["parity_checks"])
    bench_ok = all(item["parity_passed"] for item in report["workloads"])
    n_checks = len(report["parity_checks"])
    n_passed = sum(1 for item in report["parity_checks"] if item["passed"])
    print(f"E2E parity checks: {n_passed}/{n_checks} passed; workloads: {len(report['workloads'])}")
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    if not parity_ok or not bench_ok:
        for item in report["parity_checks"]:
            if not item["passed"]:
                print(f"FAILED {item['check_id']}: {item['detail']}", file=sys.stderr)
        for wl in report["workloads"]:
            if not wl["parity_passed"]:
                print(f"FAILED {wl['bench_id']}: {wl['notes']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
