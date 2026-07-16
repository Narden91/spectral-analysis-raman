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

    with pytest.raises(ValueError, match="direct identifier"):
        export_publication_bundle(source, tmp_path / "publication_bundle")
