from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from manuscript_ml.readiness import evaluate_cohort_readiness
from src.clinical.endpoints import build_clinical_endpoints
from src.metadata.clinical import _build_master_subject_table, _load_raw_clinical_data
from src.metadata.clinical_master import load_master_for_linkage


def _master() -> pd.DataFrame:
    return pd.DataFrame({
        "participant_key": ["C7", "D7", "D8"], "cohort": ["control", "stroke", "stroke"],
        "spectral_subject_id": ["7", "7", "8"], "has_spectra": [1, 1, 1],
        "spectral_link_status": ["matched"] * 3,
        "qc_flags": ["none", "none", "SEX_FOLDER_CONFLICT"],
        "sex": ["female", "male", "female"], "age_years": [40, 70, 80],
        "moca_baseline": [None, 18, 20], "moca_3m": [None, 23, 22],
        "rbans_total_3m": [None, 90, 80], "aspects": [None, 8, 7],
        "diagnosis_text": ["control", "ischemic stroke", "ischemic stroke"],
        "stroke_location": [None, "left hemisphere", "right hemisphere"],
    })


def _inventory() -> pd.DataFrame:
    return pd.DataFrame({
        "subject_id": ["control__7", "disease__7", "disease__8"],
        "subject_id_raw": ["7", "7", "8"], "is_control": [True, False, False],
        "gender": ["female", "male", "male"], "condition": ["Control", "Disease", "Disease"],
    })


def test_master_join_uses_cohort_and_preserves_unverified_spectra(tmp_path):
    path = tmp_path / "master.csv"
    _master().to_csv(path, index=False)
    result = _build_master_subject_table(_inventory(), load_master_for_linkage(path)).set_index("subject_id")
    assert result.loc["control__7", "age_years"] == 40
    assert result.loc["disease__7", "age_years"] == 70
    assert result.loc["disease__7", "delta_moca"] == 5
    assert not result.loc["disease__8", "identity_verified"]
    assert pd.isna(result.loc["disease__8", "age_years"])
    assert pd.isna(result.loc["disease__8", "moca_baseline_total"])
    assert result.loc["disease__8", "participant_key"] == "D8"
    assert result.loc["disease__8", "gender"] == "male"


def test_master_endpoints_use_canonical_store_ids(tmp_path, monkeypatch):
    import src.data_preparation.loaders as loaders

    path = tmp_path / "master.csv"
    _master().to_csv(path, index=False)
    monkeypatch.setattr(loaders, "load_clinical_subjects", lambda: _inventory())
    result = build_clinical_endpoints(cfg={"analysis": {}}, clinical_csv=path, output_dir=tmp_path)
    table = result.clinical_subject_table.set_index("subject_id")
    assert table.loc["disease__7", "delta_moca"] == 5
    assert pd.isna(table.loc["disease__8", "delta_moca"])
    assert table.loc["disease__7", "participant_key"] == "D7"


def test_clinical_loader_rejects_an_unidentified_first_column(tmp_path):
    path = tmp_path / "clinical.csv"
    pd.DataFrame({"arbitrary": ["7"], "age": [60]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="explicit subject_id"):
        _load_raw_clinical_data(path)


@pytest.mark.parametrize("identity_count,joined,ready", [(3, 3, True), (2, 3, False), (3, 2, False)])
def test_computed_readiness_requires_identity_and_join_counts(identity_count, joined, ready):
    expected = {"expected_identity_verified": 3, "expected_cohort_join": {"n_eligible_participants": 3}}
    result = evaluate_cohort_readiness({"stroke_identity_verified": identity_count}, {"n_eligible_participants": joined}, expected)
    assert result["ready_for_cohort_evaluation"] is ready


def test_real_resolver_cannot_bypass_computed_count_gate(tmp_path, monkeypatch):
    import manuscript_ml as runner
    import src.data_preparation.store as store

    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    master_path = tmp_path / "master.csv"
    master_path.write_text("fixture", encoding="utf-8")
    source = SimpleNamespace(source_path=tmp_path, manifest_hash="fixture")
    monkeypatch.setattr(store, "get_active_data_source", lambda: source)
    dataset = SimpleNamespace(manifest_summary={"n_eligible_participants": 2})
    monkeypatch.setattr(runner, "load_paired_cohort_from_store", lambda *args, **kwargs: (dataset, {"stroke_identity_verified": 3}))
    runner.clear_dataset_cache()
    with pytest.raises(RuntimeError, match="readiness expectations"):
        runner.resolve_experiment_dataset(
            use_synthetic=False, approve_cohort_evaluation=True, clinical_master_path=master_path,
            readiness={"expected_identity_verified": 3, "expected_cohort_join": {"n_eligible_participants": 3}},
        )
    runner.clear_dataset_cache()
