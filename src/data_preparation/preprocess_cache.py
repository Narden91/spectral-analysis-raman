"""CLI: build the deterministic data_preprocessed/ cache up front.

Usage:
    uv run -m src.data_preparation.preprocess_cache [--range 1300|2850|all] [--force]
"""

import argparse
import logging

from src import setup_logging
from src.io.file_crawler import build_file_index
from src.preprocessing.cache import build_cache_for_index

logger = logging.getLogger(__name__)

_RANGE_TO_REGION = {"1300": "fingerprint", "2850": "ch_stretch", "all": "all"}


def main(range_arg: str = "all", force: bool = False) -> None:
    region = _RANGE_TO_REGION[range_arg]
    df_index = build_file_index(measurement="all")
    if df_index.empty:
        logger.warning("No files found to build preprocessing cache")
        return
    manifest = build_cache_for_index(df_index, region=region, force=force)
    n_dropped = int((manifest["dropped_reason"].notna()).sum()) if not manifest.empty else 0
    logger.info("Preprocessing cache build complete: %d areas, %d dropped by retention guard", len(manifest), n_dropped)


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description="Build the deterministic per-area preprocessing cache.")
    parser.add_argument("--range", choices=["1300", "2850", "all"], default="all", help="Spectral range to cache.")
    parser.add_argument("--force", action="store_true", help="Rebuild cache even if fresh artifacts exist.")
    args = parser.parse_args()
    main(range_arg=args.range, force=args.force)
