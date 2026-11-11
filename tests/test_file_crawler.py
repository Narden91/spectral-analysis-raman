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


def test_crawler_rejects_malformed_measurement_filename(tmp_path: Path) -> None:
    import pytest

    from src.io.file_crawler import build_file_index

    path = tmp_path / "Control" / "Female" / "81" / "sample_a" / "misnamed.txt"
    path.parent.mkdir(parents=True)
    path.touch()

    with pytest.raises(ValueError, match="Invalid measurement filename"):
        build_file_index(tmp_path)


def test_crawler_keeps_hidden_and_test_exports_excluded(tmp_path: Path) -> None:
    from src.io.file_crawler import build_file_index

    measurement = tmp_path / "81_7_1300_range_SS_meas1.txt"
    measurement.touch()
    (tmp_path / ".notes.txt").touch()
    excluded = tmp_path / "test measurements" / "misnamed.txt"
    excluded.parent.mkdir()
    excluded.touch()

    frame = build_file_index(tmp_path)

    assert frame["filename"].tolist() == [measurement.name]
