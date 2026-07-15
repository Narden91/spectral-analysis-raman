from __future__ import annotations

from pathlib import Path

from src.metadata.parser import parse_filename


def test_folder_subject_identity_takes_precedence_over_filename_prefix(tmp_path: Path) -> None:
    path = tmp_path / "Control" / "Female" / "81" / "sample_a" / "72_7_1300_range_SS_meas1.txt"
    path.parent.mkdir(parents=True)
    path.touch()

    metadata = parse_filename(path)

    assert metadata["subject_id"] == "81"
    assert metadata["filename_subject_id"] == "72"


def test_filename_subject_identity_is_used_only_when_sample_hierarchy_is_absent(tmp_path: Path) -> None:
    path = tmp_path / "72_7_1300_range_meas1.txt"
    path.touch()

    metadata = parse_filename(path)

    assert metadata["subject_id"] == "72"
    assert metadata["filename_subject_id"] == "72"
