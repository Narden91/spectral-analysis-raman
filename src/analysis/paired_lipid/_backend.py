"""Array backend selection (NumPy CPU / CuPy GPU) and VRAM-aware batch sizing."""

from __future__ import annotations

import contextlib
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


@lru_cache(maxsize=1)
def _get_cupyx_linalg() -> Any | None:
    """Return cupyx.scipy.linalg once per process when CuPy is available."""
    if _probe_cupy() is None:
        return None
    try:
        import cupyx.scipy.linalg as cpx_linalg  # type: ignore[import-not-found]

        return cpx_linalg
    except Exception:
        return None


@lru_cache(maxsize=1)
def _cupyx_supports_batched_triangular() -> bool:
    """Probe once per process whether cupyx.scipy.linalg.solve_triangular handles 3D batches."""
    cp = _probe_cupy()
    cpx_linalg = _get_cupyx_linalg()
    if cp is None or cpx_linalg is None:
        return False
    try:
        eye_b = cp.broadcast_to(cp.eye(2, dtype=cp.float64), (2, 2, 2)).copy()
        rhs_b = cp.ones((2, 2, 1), dtype=cp.float64)
        out = cpx_linalg.solve_triangular(eye_b, rhs_b, lower=True, check_finite=False)
        return tuple(out.shape) == (2, 2, 1)
    except Exception:
        return False


@lru_cache(maxsize=1)
def _get_fused_mfa_gram_kernel() -> Any | None:
    """Compile and memoize a single-pass CuPy kernel for fused MFA combined Gram assembly."""
    cp = _probe_cupy()
    if cp is None:
        return None
    try:
        return cp.ElementwiseKernel(
            "raw float64 K1, raw float64 K2, raw float64 s1, raw float64 s2, "
            "raw float64 k_mu, raw float64 mu_sq, raw float64 sw, int32 n_s",
            "float64 out",
            """
            long long n_ll = (long long)n_s;
            long long n2 = n_ll * n_ll;
            long long idx = (long long)i;
            long long b = idx / n2;
            long long rem = idx - b * n2;
            long long r = rem / n_ll;
            long long c = rem - r * n_ll;
            long long rc = r * n_ll + c;
            double val = (
                s1[b] * K1[rc]
                + s2[b] * K2[rc]
                - (k_mu[b * n_ll + r] + k_mu[b * n_ll + c])
                + mu_sq[b]
            );
            out = val * (sw[b * n_ll + r] * sw[b * n_ll + c]);
            """,
            "fused_mfa_gram_f64",
        )
    except Exception:
        return None


@lru_cache(maxsize=1)
def _get_threadpool_controller() -> Any | None:
    """Create and memoize one ThreadpoolController per process so DLL enumeration runs once."""
    try:
        # Ensure SciPy's OpenBLAS DLL is loaded before enumeration so both NumPy and SciPy pools are controlled.
        import scipy.linalg  # noqa: F401
        from threadpoolctl import ThreadpoolController

        return ThreadpoolController()
    except Exception:
        return None


def _single_thread_blas() -> contextlib.AbstractContextManager[Any]:
    """Restrict BLAS thread pools to 1 thread for small-matrix CPU loops."""
    controller = _get_threadpool_controller()
    if controller is None:
        return contextlib.nullcontext()
    return controller.limit(limits=1, user_api="blas")


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
