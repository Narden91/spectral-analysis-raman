import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Regex to strip non-numeric trailing letters (e.g. "K" in "10K")
_TRAILING_LETTERS_RE = re.compile(r"[A-Za-z]+$")


def clean_subject_id(raw_id: str) -> str:
    """Normalise a raw subject ID to a plain numeric string.

    Strips:
    - Unicode right/left single quotation marks (U+2019/U+2018) — second-slide markers
    - ASCII apostrophes — same purpose
    - Trailing letters (e.g. ``K`` in ``10K`` for Control subjects)
    """
    cleaned = (
        raw_id
        .replace("\u2019", "")
        .replace("\u2018", "")
        .replace("''", "")
        .replace("'", "")
    )
    return _TRAILING_LETTERS_RE.sub("", cleaned)


# Matches the measurement number at the end of a reorganized stem:
# e.g. "109_66_1300_range_meas2"  →  group(1) = "2"
_MEAS_SUFFIX_RE = re.compile(r"_meas(\d+)$")

# Matches "meas.N" token in a raw Renishaw stem
_RAW_MEAS_RE = re.compile(r"meas\.(\d+)")

# Matches "RANGEcm-1" token and common typo variants:
#   - "2850cm-1"   (correct)
#   - "2850-cm-1"  (extra hyphen)
#   - "2850m-1"    (missing 'c')
_RAW_RANGE_RE = re.compile(r"(\d+)-?c?m-1")

# Schema-aware regex for raw Renishaw stems.
# Format: DATE_MODE_LASER_ID_SLIDE_POS_SEX_meas.N_RANGEcm-1
# Handles:
#   - ID with apostrophes (109'' second-slide notation)
#   - FP and SS modes in the filename (matched but discarded from metadata)
#   - range token variants: 1300cm-1, 2850-cm-1, 2850m-1
#   - optional -1 typo suffix
_RAW_STEM_RE = re.compile(
    r"^(?P<date>\d{6,8})"
    r"_(?P<mode>SS|FP)"
    r"_(?P<laser>\d+nm)"
    r"_(?P<raw_id>[^_]+)"
    r"_(?P<slide>\d+)"
    r"_(?P<pos>[^_]+)"
    r"_(?P<sex_token>male|female)"
    r"_meas\.(?P<meas>\d+)"
    r"_(?P<range_token>\d+)-?c?m-1"
    r"(?:-1)?$",
    re.IGNORECASE,
)

# Schema-aware regex for reorganized stems.
# Format: {Subject}_{Slide}_{Range}_range_meas{M}
# Legacy mode-tagged filenames ({Subject}_{Slide}_{Range}_range_SS_meas{M}) still
# parse for backward compatibility, but the mode token is discarded.
_REORG_STEM_RE = re.compile(
    r"^(?P<subject_id>[^_]+)"
    r"_(?P<slide>[^_]+)"
    r"_(?P<range>\d+(?:\.\d+)?)"
    r"_range(?:_(?:SS|FP))?"
    r"_meas(?P<meas>\d+)$",
    re.IGNORECASE,
)


def parse_raw_filename(filepath: str | Path) -> dict[str, Any]:
    """
    Parse a Renishaw raw measurement filename.

    Expected format:
        DATE_{SS|FP}_785nm_ID_SLIDE_POS_SEX_meas.N_RANGEcm-1.txt

    The SS/FP acquisition-mode token is matched for source-filename compatibility
    but is intentionally not returned in the metadata dictionary.

    Handles:
    - FP variant (same parse logic)
    - .wdf.txt double extension → strip to get real stem
    - ID'' second-slide notation (e.g. "109''") → cleaned to bare ID
    - meas.N_RANGEcm-1-1 typo suffix → treated as skip-worthy duplicate
    - Redone path detection via path components
    """
    path = Path(filepath)
    parts_lower = [p.lower() for p in path.parts]

    # Strip double extension (.wdf.txt) if present
    stem = path.stem
    if path.suffix.lower() == ".txt" and stem.endswith(".wdf"):
        stem = stem[: -len(".wdf")]

    metadata: dict[str, Any] = {
        "filename": path.name,
        "filepath": str(filepath),
        "extension": path.suffix.lower(),
        "is_redone": "redone measurements" in parts_lower,
        "is_control": "control" in parts_lower,
    }

    # Sex detection from path components
    metadata["gender"] = "unknown"
    if "female" in parts_lower or "female control" in parts_lower:
        metadata["gender"] = "female"
    elif "male" in parts_lower or "male control" in parts_lower:
        metadata["gender"] = "male"

    match = _RAW_STEM_RE.match(stem)
    if match is None:
        logger.warning("Raw filename does not match expected schema: %s", path.name)
        return metadata

    metadata["date"] = match.group("date")
    metadata["laser_wavelength"] = match.group("laser")
    metadata["subject_id"] = clean_subject_id(match.group("raw_id"))
    metadata["slide_number"] = match.group("slide")
    metadata["position"] = match.group("pos")
    try:
        metadata["measurement_number"] = int(match.group("meas"))
    except (ValueError, AttributeError) as exc:
        logger.warning("Could not parse measurement_number in '%s': %s", path.name, exc)
    try:
        metadata["spectral_range"] = float(match.group("range_token"))
    except (ValueError, AttributeError) as exc:
        logger.warning("Could not parse spectral_range in '%s': %s", path.name, exc)

    return metadata


def parse_filename(filepath: str | Path) -> dict[str, Any]:
    """
    Parse a reorganized measurement filename.

    Expected format:
        {Subject}_{Slide}_{Range}_range_meas{M}.txt

    Legacy mode-tagged filenames ({Subject}_{Slide}_{Range}_range_SS_meas{M}.txt)
    still parse for backward compatibility, but the mode token is discarded.

    Example:
        109_66_1300_range_meas1.txt        (mode-free)
        109_66_1300_range_SS_meas1.txt     (legacy mode-tagged)
        10K_12_2850_range_FP_meas3.txt     (legacy mode-tagged)
    """
    path = Path(filepath)
    stem = path.stem
    parts_lower = [p.lower() for p in path.parts]

    metadata: dict[str, Any] = {
        "filename": path.name,
        "filepath": str(filepath),
        "extension": path.suffix.lower(),
        "is_redone": False,
        "is_control": "control" in parts_lower,
    }

    metadata["gender"] = "unknown"
    if "female" in parts_lower or "female control" in parts_lower:
        metadata["gender"] = "female"
    elif "male" in parts_lower or "male control" in parts_lower:
        metadata["gender"] = "male"

    match = _REORG_STEM_RE.match(stem)
    if match is None:
        logger.warning("Reorganized filename does not match expected schema: %s", path.name)
        return metadata

    try:
        metadata["subject_id"] = clean_subject_id(match.group("subject_id"))
        metadata["slide_number"] = match.group("slide")
        metadata["spectral_range"] = float(match.group("range"))
        metadata["measurement_number"] = int(match.group("meas"))
    except (ValueError, AttributeError) as exc:
        logger.warning("Failed to parse reorganized filename '%s': %s", path.name, exc)

    return metadata
