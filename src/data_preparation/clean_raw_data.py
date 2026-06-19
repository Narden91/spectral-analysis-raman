import argparse
import logging
import shutil
from pathlib import Path

from src import setup_logging, config
from src.metadata.parser import clean_subject_id

logger = logging.getLogger(__name__)

def clean_raw_data(force: bool = False) -> None:
    """
    Copy Renishaw .txt measurement files from data_raw/ to data_cleaned/,
    preserving the Subject directory structure but flattening Redone measurements.
    
    Cleaning steps:
    - Uses build_raw_file_index to filter valid files (applying Redone rules).
    - Uses only the first 3 parts of the relative path (Cohort/Sex/SubjectFolder).
    - Removes "''" and ".wdf" strings from the filename.
    - Does NOT alter the subject_id or sex.

    Parameters
    ----------
    force : bool
        Must be True to allow deleting an existing data_cleaned/ directory.
        Prevents accidental data loss when run without intent.
    """
    logger.info("Starting raw data cleaning...")

    raw_dir = Path(config["data"].get("raw_dir", "data_raw"))
    cleaned_dir = Path("data_cleaned")

    if not raw_dir.exists():
        logger.error("Raw directory %s does not exist. Aborting.", raw_dir)
        return

    # Delete data_cleaned if it exists to ensure a fresh start
    if cleaned_dir.exists():
        if not force:
            raise RuntimeError(
                f"'{cleaned_dir}' already exists. Pass force=True (or --force on the CLI) "
                "to overwrite it."
            )
        shutil.rmtree(cleaned_dir)

    # Use the crawler to get only the valid measurement files (incl. Redone logic)
    # The crawler lives in src.io.file_crawler
    from src.io.file_crawler import build_raw_file_index
    df_index = build_raw_file_index(data_dir=raw_dir)
    if df_index.empty:
        logger.warning("No files to clean.")
        return

    copied = 0

    for filepath_str in df_index["filepath"]:
        source = Path(filepath_str)

        # Create relative path to maintain folder hierarchy, up to the Subject folder
        rel_path = source.relative_to(raw_dir)
        # We want to keep exactly the first 3 parts: e.g. Disease/Female/955;51,55
        # If it's shorter, keep all of them.
        parts = rel_path.parts[:3]
        target_dir = cleaned_dir.joinpath(*parts)
        
        # Clean the filename
        stem = source.stem
        # If the file is .wdf.txt, its stem is ".wdf".
        stem = stem.replace(".wdf", "")
        stem = stem.replace("''", "")
        stem = stem.replace("\u2019", "")
        stem = stem.replace("\u2018", "")
        stem = stem.replace("'", "")
        
        # Enforce folder ID into the filename (token 3)
        folder_name = target_dir.name
        raw_folder_id = folder_name.replace(";", ",").split(",")[0]
        expected_id = clean_subject_id(raw_folder_id)
        tokens = stem.split("_")
        if len(tokens) >= 4:
            if tokens[3] != expected_id:
                logger.debug(
                    "Correcting subject token in '%s': '%s' → '%s'",
                    stem,
                    tokens[3],
                    expected_id,
                )
            tokens[3] = expected_id
            stem = "_".join(tokens)
        
        # Ensure it ends with .txt
        new_filename = f"{stem}{source.suffix}"
        
        target_path = target_dir / new_filename
        target_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(source, target_path)
        copied += 1
        logger.debug("Cleaned: %s -> %s", source.name, new_filename)

    logger.info("Data cleaning complete. Copied: %d", copied)

if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description="Clean and copy raw Raman data files.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting an existing data_cleaned/ directory.",
    )
    args = parser.parse_args()
    clean_raw_data(force=args.force)
