"""Optional GPU readiness check for exploratory acceleration work.

This script does not change the analysis pipeline. It verifies whether CuPy can
see a CUDA device and runs a small CPU-vs-GPU numerical equivalence check that
future GPU-backed code paths can use as a minimum acceptance gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _cupy_status() -> tuple[object | None, dict[str, object]]:
    try:
        import cupy as cp  # type: ignore[import-not-found]
    except Exception as exc:
        return None, {"available": False, "reason": f"import_failed:{type(exc).__name__}"}

    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:
        return cp, {"available": False, "reason": f"device_query_failed:{type(exc).__name__}"}

    if device_count < 1:
        return cp, {"available": False, "reason": "no_cuda_device"}

    device = cp.cuda.Device(0)
    props = cp.cuda.runtime.getDeviceProperties(device.id)
    name = props.get("name", b"unknown")
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace")
    return cp, {"available": True, "device_count": device_count, "device_0": name}


def _equivalence_check(cp: object, *, seed: int, tolerance: float) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    x_cpu = rng.normal(size=(256, 64)).astype(np.float64)
    weights_cpu = rng.normal(size=64).astype(np.float64)

    x_gpu = cp.asarray(x_cpu)
    weights_gpu = cp.asarray(weights_cpu)
    centered_gpu = x_gpu - cp.mean(x_gpu, axis=0)
    projected_gpu = centered_gpu @ weights_gpu

    centered_cpu = x_cpu - np.mean(x_cpu, axis=0)
    projected_cpu = centered_cpu @ weights_cpu
    projected_from_gpu = cp.asnumpy(projected_gpu)

    max_abs_diff = float(np.max(np.abs(projected_cpu - projected_from_gpu)))
    return {
        "operation": "center_and_project_float64",
        "tolerance": tolerance,
        "max_abs_diff": max_abs_diff,
        "passed": bool(max_abs_diff <= tolerance),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    cp, status = _cupy_status()
    result: dict[str, object] = {"cupy": status}
    if cp is not None and status.get("available"):
        result["equivalence_check"] = _equivalence_check(cp, seed=args.seed, tolerance=args.tolerance)
    else:
        result["equivalence_check"] = {"passed": False, "reason": "cupy_unavailable"}

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")

    check = result["equivalence_check"]
    return 0 if isinstance(check, dict) and check.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
