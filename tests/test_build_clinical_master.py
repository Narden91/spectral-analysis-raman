"""Synthetic checks: no real participant records are used in these tests."""

import copy
import json

import pandas as pd
import pytest
from openpyxl import Workbook

from scripts import build_clinical_master as clinical


@pytest.fixture
def files(tmp_path):
    return {
        "clinical_csv": tmp_path / clinical.CLINICAL_CSV.name,
        "patient_sheet": tmp_path / clinical.SHEETS_XLSX.name,
        "control_sheet": tmp_path / clinical.SHEETS_XLSX.name,
        "registry": tmp_path / clinical.REGISTRY_XLSX.name,
    }


@pytest.fixture
def sources():
    registry = {
        "_source_row": 3, "number": 99, clinical.RKEY: 1,
        "age": 60, "gender 1=females": 1, "ApoE": "e3/e3",
        "e2": 0, "e3": 2, "e4": 0, "TOAST": "Undetermined",
        clinical.LOC_HEADER: "L", "IVT: no - 0; yes - 1": 0,
        "mechanical thrombectomy: 0 - no; 1 - yes": 0, "causative tratment": 0,
        "kreatynine": "x", "SAH": None, "initials": None,
    }
    registry.update({name: int(name.startswith("e3/e4=")) for name in clinical.APOE_ONEHOT})
    return {
        "clinical_csv": {1: {
            "_source_row": 2, "participant_id": 1, "age_years": 60, "gender": "female", "is_female": 1,
            "diagnosis_category": "right_hemisphere", "diagnosis_text": "synthetic stroke record",
            "death": 0,
        }},
        "patient_sheet": {1: {
            "_source_row": 2, "Sample No": 1, "Identification code of each participant": "KW_K1",
            "REDCap Record ID": "record-A", "Age": 60, "Sex": "Female", "Unnamed: 3": None,
        }},
        "registry": {1: registry, 999: {clinical.RKEY: 999}},
        "control_sheet": {1: {
            "_source_row": 3, "sample_no": "K1", "sex": "female", "age": 55,
            "nicotinism": 1, "smoking_past": 1, "other_medication": 0,
        }},
    }


def spectrum_entry(number, sex="female", prefix=None):
    return {
        "sex": {sex}, "paths": {f"data/raw/Control/{sex.title()}/{number}"},
        "samples": {f"Control/{sex.title()}/{number}/sample_1"},
        "prefixes": {str(number if prefix is None else prefix)}, "regions": {"SS"},
        "files": 1, "folder_ids": {str(number)},
    }


@pytest.fixture
def spectra():
    return {("stroke", 1): spectrum_entry(1, sex="male"), ("control", 1): spectrum_entry(1)}


def test_union_preserves_cohort_identity_and_excludes_unrelated_registry(sources, spectra, files):
    master = clinical.build_master(sources, spectra, files).set_index("participant_key")
    assert set(master.index) == {"stroke_1", "control_1"}
    assert master.loc["stroke_1", "redcap_record_id"] == "record-A"
    assert master.loc["stroke_1", "sex"] == "female"
    assert master.loc["stroke_1", "sex_spectral_folder"] == "male"
    assert master.loc["stroke_1", "spectral_link_status"] == "identity_review_required"
    assert master.loc["control_1", "spectral_link_status"] == "matched"
    assert json.loads(master.loc["stroke_1", "source_records"])["registry"]["record_number"] == 99


def test_conflicting_encodings_and_invalid_tokens_are_recoverable(sources, spectra, files):
    master = clinical.build_master(sources, spectra, files)
    stroke = master[master.cohort == "stroke"].iloc[0]
    evidence = json.loads(stroke.source_evidence)["registry"]
    assert stroke.apoe_genotype == "e3/e3"
    assert "APOE_FIELDS_DISAGREE" in stroke.qc_flags
    assert evidence["e3/e3=3"] == 0
    assert evidence["e3/e4=5"] == 1
    assert evidence[clinical.LOC_HEADER] == "L"
    assert evidence["kreatynine"] == "x"
    assert stroke.creatinine_umol_l == "unknown"
    assert stroke.stroke_location == "right"
    assert "apoe_genotype" in json.loads(stroke.inferred_fields)
    clinical.validate_preservation(sources, master)


def test_unknowns_are_not_negative_findings_and_cohort_fields_are_inapplicable(sources, spectra, files):
    master = clinical.build_master(sources, spectra, files).set_index("cohort")
    assert master.loc["stroke", "sah"] == "unknown"
    assert master.loc["stroke", "blood_collection_date"] == "unknown"
    assert master.loc["stroke", "death_followup_days"] == "unknown"
    assert master.loc["stroke", "death"] == 0
    assert master.loc["control", "sah"] == "not_applicable"
    assert master.loc["control", "creatinine_umol_l"] == "not_applicable"
    assert master.loc["control", "smoking_status"] == "current"
    assert master.loc["control", "control_nicotinism"] == 1
    assert master.loc["control", "control_past_smoking_over_6m"] == 1
    assert not master.isna().any().any()
    assert not master.map(lambda value: isinstance(value, str) and not value.strip()).any().any()


def test_filename_mismatch_does_not_automatically_merge_control_records(sources, files):
    raw = {("control", 2): spectrum_entry(2, prefix=1)}
    master = clinical.build_master(sources, raw, files).set_index("participant_key")
    assert set(master.index) == {"stroke_1", "control_1", "control_2"}
    assert master.loc["control_1", "has_spectra"] == 0
    assert master.loc["control_1", "spectral_subject_id"] == "not_available"
    assert master.loc["control_2", "has_clinical_record"] == 0
    assert master.loc["control_2", "sex_source"] == "spectral_folder"
    assert master.loc["control_2", "age_years"] == "unknown"
    assert "POSSIBLE_SPECTRA_IN_ANOTHER_FOLDER" in master.loc["control_1", "qc_flags"]
    assert "SPECTRAL_FILENAME_FOLDER_CONFLICT" in master.loc["control_2", "qc_flags"]


def test_cross_source_fill_retains_the_original_missing_cell(sources, spectra, files):
    sources["clinical_csv"][1]["age_years"] = None
    sources["clinical_csv"][1]["diagnosis_category"] = "ischemic_stroke_unspecified"
    master = clinical.build_master(sources, spectra, files)
    stroke = master[master.cohort == "stroke"].iloc[0]
    assert stroke.age_years == 60
    assert stroke.stroke_location == "left"
    assert json.loads(stroke.source_evidence)["clinical_csv"]["age_years"] is None
    assert json.loads(stroke.inferred_fields)["age_years"]["source"] == "patient_sheet"
    assert "LOCATION_FILLED_FROM_REGISTRY" in stroke.qc_flags


@pytest.mark.parametrize(("nicotinism", "past", "expected"), [
    (None, None, None), (0, None, None), (None, 0, None), (None, 1, None),
    (1, None, "current"), (0, 0, "never"), (0, 1, "former"), (1, 1, "current"),
])
def test_partial_smoking_flags_do_not_silently_become_never(nicotinism, past, expected):
    assert clinical.control_smoking(nicotinism, past) == expected


def test_validation_detects_lost_source_evidence(sources, spectra, files):
    master = clinical.build_master(sources, spectra, files)
    master.loc[master.cohort == "stroke", "source_evidence"] = "{}"
    with pytest.raises(ValueError, match="Source information was lost"):
        clinical.validate_preservation(sources, master)


def test_builds_have_no_mutable_global_flag_leak(sources, spectra, files):
    first = clinical.build_master(sources, spectra, files)
    corrected = copy.deepcopy(sources)
    corrected["registry"][1].update({"e3/e3=3": 1, "e3/e4=5": 0})
    second = clinical.build_master(corrected, spectra, files)
    assert first.qc_flags.str.contains("APOE_FIELDS_DISAGREE").any()
    assert not second.qc_flags.str.contains("APOE_FIELDS_DISAGREE").any()


def test_serialized_zero_therapy_is_not_truthy_and_prior_output_survives_failure(sources, spectra, files, tmp_path):
    master = clinical.build_master(sources, spectra, files)
    clinical.write_outputs(master, clinical.build_dictionary(), "synthetic notes\n", tmp_path, sources)
    path = tmp_path / "clinical_master.csv"
    original = path.read_bytes()
    loaded = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False, dtype=str)
    clinical.verify_serialized(loaded, sources)
    damaged = master.copy()
    damaged.loc[damaged.cohort == "stroke", "source_evidence"] = "{}"
    with pytest.raises(ValueError, match="Source information was lost"):
        clinical.write_outputs(damaged, clinical.build_dictionary(), "damaged\n", tmp_path, sources)
    assert path.read_bytes() == original


def test_raw_inventory_counts_files_and_records_disagreement(tmp_path):
    for name in ("2_1_1300_range_SS_meas1.txt", "1_1_1300_range_SS_meas2.txt"):
        path = tmp_path / "Control" / "Female" / "2" / "sample_1" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic", encoding="utf-8")
    inventory = clinical.scan_spectra(tmp_path)
    assert inventory[("control", 2)]["files"] == 2
    assert inventory[("control", 2)]["prefixes"] == {"1", "2"}
    assert len(inventory[("control", 2)]["samples"]) == 1


def test_empty_raw_directory_is_not_treated_as_a_cohort_without_spectra(tmp_path):
    with pytest.raises(ValueError, match="No original spectra"):
        clinical.scan_spectra(tmp_path)


def write_synthetic_inputs(directory, duplicate_patient=False):
    pd.DataFrame([{
        "participant_id": 1, "age_years": 60, "gender": "female", "is_female": 1,
        "diagnosis_text": "synthetic", "diagnosis_category": "left_hemisphere",
        "rbans_total_3m": 90, "moca_total_baseline": 20, "moca_total_3m": 22, "aspects_score": 9, "death": False,
    }]).to_csv(directory / clinical.CLINICAL_CSV.name, index=False)
    book = Workbook()
    patient = book.active
    patient.title = "Stroke Patients"
    patient.append(["Sample No", "Identification code of each participant", "REDCap Record ID", None, "Age", "Sex"])
    patient.append([1, "KW_K1", "record-A", None, 60, "female"])
    if duplicate_patient:
        patient.append([1, "KW_K1", "record-A", None, 60, "female"])
    controls = book.create_sheet("Control Group")
    controls.append(clinical.CT_HEADER[:4] + ["Risk Factors"] + [None] * 6 + ["Medications Taken"])
    controls.append([None] * 4 + clinical.CT_HEADER[4:])
    controls.append(["K1", "Female", 55, pd.Timestamp("2020-06-15").to_pydatetime()] + [0] * 11)
    book.save(directory / clinical.SHEETS_XLSX.name)
    registry = Workbook()
    sheet = registry.active
    sheet.title = clinical.REGISTRY_SHEET
    headers = sorted(set(clinical.REGISTRY_MAP) | set(clinical.CODED_REGISTRY) | set(clinical.APOE_ONEHOT)
                     | {"e2", "e3", "e4", "number", clinical.RKEY, "initials", "causative tratment"})
    sheet.append(headers)
    values = {clinical.RKEY: 1, "number": 99, "age": 60, "gender 1=females": 1}
    sheet.append([values.get(header) for header in headers])
    registry.save(directory / clinical.REGISTRY_XLSX.name)


def test_excel_headers_dates_and_registry_join_key_are_loaded_without_imputation(tmp_path):
    write_synthetic_inputs(tmp_path)
    sources, files = clinical.load_sources(tmp_path)
    assert set(sources["registry"]) == {1}
    assert sources["registry"][1]["number"] == 99
    assert clinical.date_text(sources["control_sheet"][1]["blood_date"]) == "2020-06-15"
    assert sources["patient_sheet"][1]["_source_row"] == 2
    assert sources["control_sheet"][1]["_source_row"] == 3
    master = clinical.build_master(sources, {("control", 1): spectrum_entry(1)}, files)
    clinical.write_outputs(master, clinical.build_dictionary(), "synthetic\n", tmp_path, sources)


def test_duplicate_source_ids_stop_an_ambiguous_join(tmp_path):
    write_synthetic_inputs(tmp_path, duplicate_patient=True)
    with pytest.raises(ValueError, match="Duplicate participant identifiers"):
        clinical.load_sources(tmp_path)


def test_new_source_columns_are_not_silently_ignored(tmp_path):
    write_synthetic_inputs(tmp_path)
    path = tmp_path / clinical.CLINICAL_CSV.name
    frame = pd.read_csv(path)
    frame["unexpected_measurement"] = 7
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Unexpected source columns"):
        clinical.load_sources(tmp_path)
