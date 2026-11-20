"""Array backend selection (NumPy CPU / CuPy GPU) and VRAM-aware batch sizing."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import Any

import numpy as np


@lru_cache(maxsize=1)
def _probe_cupy() -> Any | None:
    """Import CuPy and verify at least one CUDA device once per process."""
    try:
        import cupy as cp  # type: ignore[import-not-found]

        if cp.cuda.runtime.getDeviceCount() > 0:
            return cp
    except Exception:
        pass
    return None


def _get_gpu_xp() -> Any | None:
    """Return the CuPy module when an NVIDIA CUDA device is available, else None."""
    return _probe_cupy()


def resolve_array_backend(
    use_gpu: bool | None,
    n_draws: int,
    *,
    min_auto_draws: int = 256,
    gpu_probe: Callable[[], Any | None] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Select CuPy or NumPy backend and build standardized execution metadata."""
    probe = _get_gpu_xp if gpu_probe is None else gpu_probe
    cp_mod = probe() if use_gpu is not False else None
    gpu_requested = use_gpu is True or (use_gpu is None and n_draws >= min_auto_draws)
    xp = cp_mod if (cp_mod is not None and gpu_requested) else np
    gpu_fallback = bool(gpu_requested and cp_mod is None)
    meta: dict[str, Any] = {
        "device": "gpu" if xp is not np else "cpu",
        "gpu_requested": use_gpu,
        "gpu_active": xp is not np,
        "gpu_fallback": gpu_fallback,
    }
    if gpu_fallback:
        meta["gpu_fallback_reason"] = "CuPy/CUDA device unavailable"
    return xp, meta


def _to_np(xp: Any, arr: Any) -> np.ndarray:
    """Convert a NumPy or CuPy array to a host NumPy float64 array."""
    if xp is not np and hasattr(xp, "asnumpy"):
        return np.asarray(xp.asnumpy(arr), dtype=np.float64)
    return np.asarray(arr, dtype=np.float64)


def suggest_batch_size(xp: Any, n_draws: int, n_samples_outer: int) -> int:
    """Choose a host- or VRAM-safe batch size for batched bootstrap evaluation."""
    draws = max(1, int(n_draws))
    if xp is np:
        return min(128, draws)

    n_s = max(1, int(n_samples_outer))
    try:
        free_bytes_raw, _ = xp.cuda.runtime.memGetInfo()
        free_bytes = int(free_bytes_raw)
        try:
            free_bytes += int(xp.get_default_memory_pool().free_bytes())
        except Exception:
            pass
    except Exception:
        return min(1024, max(128, draws))

    # Budget ~38% of available VRAM across ~6 live (B, n_s, n_s) float64 buffers.
    bytes_per_draw = max(6 * n_s * n_s * 8, 1)
    raw_max = int(0.38 * free_bytes) // bytes_per_draw
    upper = 2048 if (free_bytes >= 12 * (1024**3) and n_s <= 160 and draws > 1024) else 1024
    step = 256 if raw_max >= 512 else 128
    rounded = (raw_max // step) * step
    return max(128, min(upper, rounded))
