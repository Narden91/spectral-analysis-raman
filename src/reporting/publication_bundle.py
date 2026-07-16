"""Create an aggregate-only publication bundle from a private analysis run."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pandas as pd

from src.analysis.io_utils import write_json_atomic, write_text_atomic

# The exporter is deliberately opt-in: files not named here never cross the
# private-run/public-bundle boundary, even if they look aggregate after a scan.
_PUBLIC_FILENAMES = frozenset(
    {
        "candidate_region_report.md",
        "candidate_region_summary.csv",
        "cross_phase_multiplicity_summary.csv",
        "disease_subgroup_gap_statistic.csv",
        "endpoint_consistency.csv",
        "panel_evidence_flags.csv",
        "integrated_candidate_regions.csv",
        "panel_continuous_associations.csv",
        "panel_contrast_tests.csv",
        "panel_report_section.md",
        "phase_status.csv",
        "pls_permutation_results.csv",
        "pls_regression_performance.csv",
        "primary_anchor_result.csv",
    }
)
_PUBLIC_FIGURE_NAMES = frozenset(
    {
        "candidate_region_overview.png",
        "fig1_cohort_panel_overview.png",
        "fig2_effect_size_forest_robustness.png",
    }
)
_DIRECT_IDENTIFIER_COLUMNS = frozenset(
    {
        "subject_id",
        "patient_id",
        "participant_id",
        "sample_id",
        "record_id",
        "age",
        "sex",
        "diagnosis",
        "lesion_location",
        "rbans_total",
        "aspects_score",
    }
)


def _is_public_candidate(path: Path) -> bool:
    return path.name in _PUBLIC_FILENAMES or path.name in _PUBLIC_FIGURE_NAMES


def _assert_aggregate_csv(path: Path) -> None:
    columns = {str(column).strip().lower() for column in pd.read_csv(path, nrows=0).columns}
    forbidden = sorted(columns & _DIRECT_IDENTIFIER_COLUMNS)
    if forbidden:
        raise ValueError(f"Refusing to export {path.name}: direct identifier/clinical columns {forbidden}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_publication_bundle(source_dir: str | Path, output_dir: str | Path) -> list[Path]:
    """Copy an allowlisted, identifier-free aggregate bundle for publication.

    The source run remains private. Existing destination files are rejected to
    prevent accidental mixing of bundles from different runs.
    """
    source = Path(source_dir).resolve()
    output = Path(output_dir).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Private analysis directory not found: {source}")
    if output.exists():
        raise FileExistsError(f"Publication bundle destination already exists: {output}")

    selected = [path for path in sorted(source.rglob("*")) if path.is_file() and _is_public_candidate(path)]
    if not selected:
        raise ValueError("No allowlisted aggregate publication artifacts were found")

    exported: list[Path] = []
    for path in selected:
        if path.suffix.lower() == ".csv":
            _assert_aggregate_csv(path)
        relative = path.relative_to(source)
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        exported.append(destination)

    manifest = {
        "bundle_type": "aggregate_only_publication_bundle",
        "n_files": len(exported),
        "files": [
            {"path": str(path.relative_to(output)).replace("\\", "/"), "sha256": _sha256(path)}
            for path in exported
        ],
    }
    write_json_atomic(output / "publication_bundle_manifest.json", manifest)
    write_text_atomic(
        output / "README.md",
        "# Aggregate-only publication bundle\n\n"
        "This bundle intentionally excludes raw spectra, subject-level feature matrices, "
        "direct identifiers, clinical subject tables, run logs, and source-data paths. "
        "It is derived from a private analysis run and contains allowlisted aggregate tables and figures only.\n",
    )
    return exported
