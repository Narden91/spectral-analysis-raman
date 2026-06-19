from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def summarize_area_pixels(concatenated_pixels: np.ndarray) -> dict[str, np.ndarray | int]:
    """Compute mean/std summary for one area from concatenated pixel spectra."""
    if concatenated_pixels.ndim != 2:
        raise ValueError("concatenated_pixels must be a 2D array")

    n_valid = int(concatenated_pixels.shape[0])
    if n_valid == 0:
        n_channels = concatenated_pixels.shape[1]
        return {
            "area_mean": np.full(n_channels, np.nan),
            "area_std": np.full(n_channels, np.nan),
            "n_valid_pixels": 0,
        }

    return {
        "area_mean": np.nanmedian(concatenated_pixels, axis=0),
        "area_std": np.nanstd(concatenated_pixels, axis=0),
        "n_valid_pixels": n_valid,
    }


def area_store_from_process_results(results: list[dict]) -> "AreaStore":
    """Build an AreaStore from the list of dicts returned by process_subject_areas()."""
    store = AreaStore()
    for r in results:
        store.add(
            subject_id=r["subject_id"],
            slide_number=r["slide_number"],
            area_id=r["area_id"],
            area_mean=r["area_mean"],
            area_mean_bc=r.get("area_mean_bc"),
            area_std=r["area_std"],
            n_valid_pixels=r["n_valid_pixels"],
            ch_pixels=np.asarray(r.get("ch_pixels", np.empty((0, 0))), dtype=float),
        )

    return store


@dataclass
class AreaStore:
    """In-memory store for area-level summaries with NPZ persistence.

    Each row represents one area (one spatial measurement) for one subject,
    indexed by (subject_id, slide_number, area_id).  This is the input to the
    Phase 6 CV loop (area outlier screening → aggregation → feature extraction).
    """

    _rows: list[dict] = field(default_factory=list)
    wavenumbers: np.ndarray | None = None
    _ch_pixels: list[np.ndarray] = field(default_factory=list)

    def set_wavenumbers(self, wavenumbers: np.ndarray) -> None:
        self.wavenumbers = np.asarray(wavenumbers, dtype=float).copy()

    @property
    def ch_pixels(self) -> list[np.ndarray]:
        """List of CH-stretch pixel arrays (one per area)."""
        return self._ch_pixels


    def add(
        self,
        subject_id: str,
        slide_number: str,
        area_id: str,
        area_mean: np.ndarray,
        area_std: np.ndarray,
        n_valid_pixels: int,
        area_mean_bc: np.ndarray | None = None,
        ch_pixels: np.ndarray | None = None,
    ) -> None:
        self._rows.append(
            {
                "subject_id": str(subject_id),
                "slide_number": str(slide_number),
                "area_id": str(area_id),
                "area_mean": np.asarray(area_mean, dtype=float),
                "area_mean_bc": None if area_mean_bc is None else np.asarray(area_mean_bc, dtype=float),
                "area_std": np.asarray(area_std, dtype=float),
                "n_valid_pixels": int(n_valid_pixels),
            }
        )

        self._ch_pixels.append(
            np.asarray(ch_pixels, dtype=float) if ch_pixels is not None and ch_pixels.size > 0
            else np.empty((0, 0), dtype=float)
        )

    def __len__(self) -> int:
        return len(self._rows)


    def to_dataframe(self) -> pd.DataFrame:
        """Return a DataFrame with one row per area.

        ``area_mean``, ``area_mean_bc``, ``area_std``, and ``ch_pixels`` columns contain numpy arrays (object dtype).
        """
        df = pd.DataFrame(self._rows)
        df["ch_pixels"] = self._ch_pixels
        return df

    # ------------------------------------------------------------------
    # Persistence (NPZ)
    # ------------------------------------------------------------------

    def save_npz(self, path: str | Path) -> None:
        """Persist the store to a compressed NumPy archive.

        The archive contains parallel arrays — one element per area — that can
        be loaded back with :meth:`AreaStore.load_npz`.

        Parameters
        ----------
        path : str or Path
            Destination file.  The ``.npz`` extension is appended if absent.
        """
        if not self._rows:
            raise ValueError("AreaStore is empty — nothing to save")

        path = Path(path)

        subject_ids = np.array([r["subject_id"] for r in self._rows], dtype=object)
        slide_numbers = np.array([r["slide_number"] for r in self._rows], dtype=object)
        area_ids = np.array([r["area_id"] for r in self._rows], dtype=object)
        n_valid = np.array([r["n_valid_pixels"] for r in self._rows], dtype=np.int64)

        # Stack spectra arrays — all must have the same number of channels
        area_means = np.vstack([r["area_mean"] for r in self._rows])   # (n_areas, n_ch)
        area_stds = np.vstack([r["area_std"] for r in self._rows])     # (n_areas, n_ch)

        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "subject_ids": subject_ids,
            "slide_numbers": slide_numbers,
            "area_ids": area_ids,
            "n_valid_pixels": n_valid,
            "area_means": area_means,
            "area_stds": area_stds,
        }

        area_mean_bc_rows = [r.get("area_mean_bc") for r in self._rows]
        if area_mean_bc_rows and all(row is not None for row in area_mean_bc_rows):
            payload["area_means_bc"] = np.vstack(area_mean_bc_rows)
        if self.wavenumbers is not None:
            payload["wavenumbers"] = np.asarray(self.wavenumbers, dtype=float)

        if self._ch_pixels and any(a.size > 0 for a in self._ch_pixels):
            offsets = np.zeros(len(self._ch_pixels) + 1, dtype=np.int64)
            for i, arr in enumerate(self._ch_pixels):
                offsets[i + 1] = offsets[i] + max(arr.shape[0], 0)
            flat = np.vstack([a for a in self._ch_pixels if a.size > 0]) if offsets[-1] > 0 else np.empty((0, 0))
            payload["ch_pixels_flat"] = flat
            payload["ch_pixels_offsets"] = offsets
            if flat.shape[1] > 0:
                payload["ch_pixels_n_channels"] = np.array([flat.shape[1]], dtype=np.int64)

        np.savez_compressed(path, **payload)

    @classmethod
    def load_npz(cls, path: str | Path) -> "AreaStore":
        """Load an AreaStore previously saved with :meth:`save_npz`.

        Parameters
        ----------
        path : str or Path
            Path to the ``.npz`` archive.

        Returns
        -------
        AreaStore
        """
        data = np.load(path, allow_pickle=True)
        store = cls()
        if "wavenumbers" in data:
            store.wavenumbers = np.asarray(data["wavenumbers"], dtype=float)
        n = len(data["subject_ids"])
        area_means_bc = None
        if "area_means_bc" in data:
            area_means_bc = np.asarray(data["area_means_bc"], dtype=float)
        else:
            logger.warning("Legacy area store loaded without area_mean_bc; relative-feature basis will fall back until caches are rebuilt.")
        if "instrument_modes" in data:
            logger.info("Legacy area store contains mode metadata; ignoring it.")
        ch_offsets = None

        ch_flat = None
        ch_nch = 0
        if "ch_pixels_offsets" in data and "ch_pixels_flat" in data:
            ch_offsets = np.asarray(data["ch_pixels_offsets"], dtype=np.int64)
            ch_flat = np.asarray(data["ch_pixels_flat"], dtype=float)
            if "ch_pixels_n_channels" in data:
                ch_nch = int(np.asarray(data["ch_pixels_n_channels"], dtype=np.int64).item())
        for i in range(n):
            ch_arr = np.empty((0, 0), dtype=float)
            if ch_offsets is not None and ch_flat is not None and i < len(ch_offsets) - 1:
                start = int(ch_offsets[i])
                end = int(ch_offsets[i + 1])
                if start < end and ch_flat.size > 0:
                    ch_arr = ch_flat[start:end].reshape(-1, max(ch_nch, 0)) if ch_nch > 0 else np.empty((0, 0))
            store.add(
                subject_id=str(data["subject_ids"][i]),
                slide_number=str(data["slide_numbers"][i]),
                area_id=str(data["area_ids"][i]),
                area_mean=data["area_means"][i],
                area_mean_bc=None if area_means_bc is None else area_means_bc[i],
                area_std=data["area_stds"][i],
                n_valid_pixels=int(data["n_valid_pixels"][i]),
                ch_pixels=ch_arr,
            )

        return store
