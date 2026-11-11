from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.reporting.publication_bundle import export_publication_bundle


def test_export_publication_bundle_copies_only_allowlisted_aggregate_files(tmp_path: Path) -> None:
    source = tmp_path / "private_results"
    source.mkdir()
    pd.DataFrame({"phase_key": ["pls_regression"], "status": ["ok"]}).to_csv(
        source / "phase_status.csv", index=False
    )
    pd.DataFrame({"subject_id": ["S01"], "age": [60]}).to_csv(
        source / "clinical_subject_table.csv", index=False
    )
    (source / "raw_spectra.parquet").write_bytes(b"private")

    output = tmp_path / "publication_bundle"
    exported = export_publication_bundle(source, output)

    assert [path.name for path in exported] == ["phase_status.csv"]
    assert (output / "phase_status.csv").exists()
    assert not (output / "clinical_subject_table.csv").exists()
    manifest = json.loads((output / "publication_bundle_manifest.json").read_text(encoding="utf-8"))
    assert manifest["bundle_type"] == "aggregate_only_publication_bundle"


def test_export_publication_bundle_rejects_allowlisted_file_with_identifier_column(tmp_path: Path) -> None:
    source = tmp_path / "private_results"
    source.mkdir()
    pd.DataFrame({"subject_id": ["S01"], "status": ["ok"]}).to_csv(source / "phase_status.csv", index=False)

    pd.DataFrame({"status": ["ok"]}).to_csv(source / "candidate_region_summary.csv", index=False)
    output = tmp_path / "publication_bundle"
    with pytest.raises(ValueError, match="direct identifier"):
        export_publication_bundle(source, output)
    assert not output.exists()


@pytest.mark.parametrize("failure_step", ["copy2", "write_json_atomic", "write_text_atomic"])
def test_export_publication_bundle_failure_leaves_no_destination(tmp_path: Path, monkeypatch, failure_step: str) -> None:
    import src.reporting.publication_bundle as bundle

    source = tmp_path / "private_results"
    source.mkdir()
    pd.DataFrame({"status": ["ok"]}).to_csv(source / "phase_status.csv", index=False)
    output = tmp_path / "publication_bundle"

    def fail(*args, **kwargs):
        raise OSError("simulated export failure")

    target = bundle.shutil if failure_step == "copy2" else bundle
    monkeypatch.setattr(target, failure_step, fail)
    with pytest.raises(OSError, match="simulated export failure"):
        export_publication_bundle(source, output)
    assert not output.exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["private_results"]
