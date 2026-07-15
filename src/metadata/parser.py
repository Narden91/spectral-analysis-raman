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
    result = _TRAILING_LETTERS_RE.sub("", cleaned)
    if result != raw_id:
        logger.debug("subject_id rewrite: %r -> %r", raw_id, result)
    return result


# Matches the measurement number at the end of a reorganized stem:
# e.g. "109_66_1300_range_meas2"  →  group(1) = "2"
_MEAS_SUFFIX_RE = re.compile(r"_meas(\d+)$")

# Schema-aware regex for reorganized stems.
# Format: {Subject}_{Slide}_{Range}_range_meas{M}
# Legacy mode-tagged filenames ({Subject}_{Slide}_{Range}_range_SS_meas{M}) still
# parse for backward compatibility, but the mode token is discarded.
_REORG_STEM_RE = re.compile(
    r"^(?P<subject_id>[^_]+)"
    r"_(?P<slide>[^_]+)"
    r"_(?P<range>\d+(?:\.\d+)?)"
    r"_range(?:_(?P<acquisition_mode>SS|FP))?"
    r"_meas(?P<meas>\d+)$",
    re.IGNORECASE,
)


def _subject_id_from_folder(path: Path) -> str | None:
    """Return the subject directory immediately above a ``sample_*`` folder.

    The raw directory hierarchy is the authoritative participant identity.  A
    legacy filename can contain an obsolete participant prefix, so it is kept
    as provenance but must never override the enclosing subject folder.
    """
    for parent in path.parents:
        if parent.name.lower().startswith("sample_"):
            subject_dir = parent.parent.name.strip()
            return clean_subject_id(subject_dir) if subject_dir else None
    return None

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
        filename_subject_id = clean_subject_id(match.group("subject_id"))
        folder_subject_id = _subject_id_from_folder(path)
        metadata["filename_subject_id"] = filename_subject_id
        metadata["subject_id"] = folder_subject_id or filename_subject_id
        metadata["slide_number"] = match.group("slide")
        metadata["spectral_range"] = float(match.group("range"))
        metadata["measurement_number"] = int(match.group("meas"))
        mode = match.group("acquisition_mode")
        metadata["acquisition_mode"] = mode.upper() if mode else None
    except (ValueError, AttributeError) as exc:
        logger.warning("Failed to parse reorganized filename '%s': %s", path.name, exc)

    return metadata
