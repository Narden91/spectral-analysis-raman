"""Build an auditable study-level CSV from the original clinical tables and raw folders.

Run with the project environment: uv run python scripts/build_clinical_master.py
All observations are preserved, either as a canonical value, an equivalent encoding,
or sparse source_evidence. unknown and not_applicable are explicit missing markers.
No statistical imputation or unverified sample relinking is performed.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from uuid import uuid4

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CLINICAL_CSV = DATA / "clinical_data.csv"
SHEETS_XLSX = DATA / "Finlandia  IBIOSTROKE Patients and Control.xlsx"
REGISTRY_XLSX = DATA / "Finlandia Ibiostroke 2026 (without initials).xlsx"
REGISTRY_SHEET = "Baza 1000 -import-czyszczenieV3"
OUT_CSV = DATA / "clinical_master.csv"
OUT_DICT = DATA / "clinical_master_dictionary.csv"

# The registry holds the participant number under this header. Its `number`
# column is only a row index; joining on it matches by chance.
RKEY = "Number in the primary large data base"
LOC_HEADER = "stroke location: PRAWA (right) /LEWA (left) /TYL (posterior)"  # double space in the file collapsed
CREATININE_OUTLIER = 300  # umol/L; rare outside renal failure, so flag for review


CT_HEADER = [
    "Sample No", "Sex", "Age", "The date on which the blood was collected",
    "Hypertension", "Diabetes", "Atrial Fibrillation", "Coronary Heart Disease",
    "Nicotinism", "Smoking in the past (more than 6 months ago)",
    "Hyperlipidemia or the use of cholesterol-lowering drugs",
    "Antiplatelet Medication", "Anticoagulant Medication", "Statins", "Other",
]
CT_COLS = [
    "sample_no", "sex", "age", "blood_date", "hypertension", "diabetes",
    "atrial_fibrillation", "coronary_artery_disease", "nicotinism", "smoking_past",
    "hyperlipidemia", "antiplatelet", "anticoagulant", "statin", "other_medication",
]
LOC_CSV = {
    "right_hemisphere": "right", "left_hemisphere": "left",
    "posterior_circulation": "posterior", "brainstem": "posterior",
    "left_cerebellum": "posterior",
}  # "ischemic_stroke_unspecified" is deliberately absent: fill from the registry
LOC_REG = {"P": "right", "L": "left", "T": "posterior"}  # prawa / lewa / tyl
TOAST_TEXT = {  # diagnosis_text keyword -> TOAST class it implies
    "small vessels disease": "Small Vessel Occlusion",
    "large vessels disease": "Large Artery Atherosclerosis",
    "dissection": "Other etiology",
}
APOE_ONEHOT = ["e2/e2=1", "e2/e3=2", "e3/e3=3", "e2/e4=4", "e3/e4=5", "e4/e4=6"]
SMOKING_HEADER = "smokin: 0 - NO, 1 - current, 2- previously)"

SPECS: list[dict] = []  # one entry per output column: order and dictionary


def S(column, applies, description, source="", sheet="", source_column="", *,
      reg=None, kind=None, unit="", unit_status="", coding=""):
    """Register an output column. `reg` marks a column read from the registry."""
    if reg:  # explicit source arguments win, for columns that also exist in the control sheet
        source, sheet, source_column = source or REGISTRY_XLSX.name, sheet or REGISTRY_SHEET, source_column or reg
    SPECS.append(dict(
        column=column, applies_to=applies, description=description, unit=unit,
        unit_status=unit_status, coding=coding, source_file=source,
        source_sheet=sheet, source_column=source_column, reg=reg, kind=kind,
    ))


CSV, SHEETS, STORE = CLINICAL_CSV.name, SHEETS_XLSX.name, "data/raw"


def shared(reg_header: str, control_header: str) -> dict:
    """Provenance for variables that patients take from the registry and controls from the sheet."""
    return dict(
        source=f"{REGISTRY_XLSX.name} (stroke); {SHEETS} (control)",
        sheet=f"{REGISTRY_SHEET} (stroke); Control Group (control)",
        source_column=f"{reg_header} (stroke); {control_header} (control)",
    )


B, ST, CT = "both", "stroke", "control"
# unit_status: stated = unit is in the source header; assumed = inferred from the
# value range (confirm with the data owner); unknown = cannot be inferred.
S("participant_key", B, "Unique row key, cohort_number. Numbers repeat across cohorts (patient 11, control K11): never join on the number alone.")
S("cohort", B, "Participant group.", coding="stroke | control")
S("spectral_subject_id", B, "Participant folder ID under data/raw. Use cohort as well as this ID when joining spectra.", STORE, "", "subject_id")
S("has_spectra", B, "Spectra exist under the original participant folder. This does not establish sample identity when spectral_link_status requires review.", STORE, "", "subject_id", coding="1 yes, 0 no")
S("has_clinical_record", B, "A row for this participant exists in the clinical sources. 0 only for controls with spectra but no row in the Control Group sheet.", coding="1 yes, 0 no")
S("spectral_link_status", B, "Identity status for joining spectra to clinical data. Folder/file disagreements are not automatically relinked.", coding="matched | identity_review_required | no_clinical_record | no_spectra")
S("study_code", B, "Study code as written in the source (patients KW_K<n>, controls K<n>).", SHEETS, "Stroke Patients / Control Group", "Identification code of each participant / Sample No")
S("redcap_record_id", ST, "REDCap record ID.", SHEETS, "Stroke Patients", "REDCap Record ID")
S("dna_sample_number", ST, "DNA sample number in the registry.", reg="DNA sample nimber", kind="text")
S("blood_collection_date", B, "Blood collection date. No patient dates are supplied by the sources.", SHEETS, "Control Group", "The date on which the blood was collected", unit="ISO date")
S("age_years", B, "Age as recorded; reference date is not specified.", f"{CSV} / {SHEETS}", "Control Group", "age_years / Age", unit="years", unit_status="stated")
S("sex", B, "Sex from the clinical record. Spectral-folder sex only when sex_source is spectral_folder.", f"{CSV} / {SHEETS}", "Control Group", "gender / Sex", coding="female | male")
S("sex_source", B, "Where sex comes from.", coding="clinical_csv | patient_sheet | control_sheet | registry | spectral_folder")
S("sex_spectral_folder", B, "Sex implied by the original Female/Male folder. Folder metadata cannot confirm biological sex or sample identity.", STORE, "", "gender", coding="female | male")
S("weight_kg", ST, "Body weight.", reg="weight", kind="float", unit="kg", unit_status="assumed")
S("height_cm", ST, "Body height.", reg="hight", kind="float", unit="cm", unit_status="assumed")
S("years_of_schooling", ST, "Years of schooling.", reg="years of schooling", kind="float", unit="years", unit_status="stated")
S("hypertension", B, "Hypertension indicator. Assessment reference time is not supplied.", reg="hypertension", kind="int", coding="1 yes, 0 no", **shared("hypertension", "Hypertension"))
S("diabetes", B, "Diabetes mellitus. Assessment reference time is not supplied.", reg="diabetes mellitus", kind="int", coding="1 yes, 0 no", **shared("diabetes mellitus", "Diabetes"))
S("atrial_fibrillation", B, "Atrial fibrillation. Assessment reference time is not supplied.", reg="atrial fibrillation", kind="int", coding="1 yes, 0 no", **shared("atrial fibrillation", "Atrial Fibrillation"))
S("coronary_artery_disease", B, "Coronary artery disease. Assessment reference time is not supplied.", reg="cronary artery disease", kind="int", coding="1 yes, 0 no", **shared("cronary artery disease", "Coronary Heart Disease"))
S("smoking_status", B, "Smoking. Controls: current if Nicotinism = 1 (also when the 'smoked in the past' flag = 1), former if Nicotinism = 0 and the past flag = 1; never only when both flags are 0. Partial missing flags can leave status unknown. Registry coding for patients. The control interpretation is provisional; original flags are preserved.", reg=SMOKING_HEADER, kind="custom", coding="never | former | current", **shared(SMOKING_HEADER, "Nicotinism + Smoking in the past (more than 6 months ago)"))
S("control_nicotinism", CT, "Original Nicotinism flag; current-smoking meaning is unconfirmed.", SHEETS, "Control Group", "Nicotinism", coding="1 yes, 0 no")
S("control_past_smoking_over_6m", CT, "Original smoking-in-the-past (more than six months ago) flag; may coexist with Nicotinism.", SHEETS, "Control Group", "Smoking in the past (more than 6 months ago)", coding="1 yes, 0 no")
S("hyperlipidemia_or_lipid_lowering_drug", CT, "Hyperlipidemia or use of cholesterol-lowering drugs. Controls only; no patient equivalent.", SHEETS, "Control Group", "Hyperlipidemia or the use of cholesterol-lowering drugs", coding="1 yes, 0 no")
S("antiplatelet", B, "Chronic antiplatelet medication. Patients: before stroke, as stated in the header. Control assessment reference time is not supplied.", reg="chronic antiplatelets before stroke", kind="int", coding="1 yes, 0 no", **shared("chronic antiplatelets before stroke", "Antiplatelet Medication"))
S("anticoagulant", B, "Chronic anticoagulant medication. Same timing as antiplatelet.", reg="chronic anticoagulants before stroke", kind="int", coding="1 yes, 0 no", **shared("chronic anticoagulants before stroke", "Anticoagulant Medication"))
S("statin", B, "Statin use. Patients: before stroke, as stated in the header. Control assessment reference time is not supplied.", reg="chronic statin before stroke", kind="int", coding="1 yes, 0 no", **shared("chronic statin before stroke", "Statins"))
S("other_medication", CT, "Other medication taken. Controls only; meaning of 'Other' is not defined in the source.", SHEETS, "Control Group", "Other", coding="1 yes, 0 no")
S("diagnosis_text", ST, "Free-text diagnosis from the curated clinical CSV.", CSV, "", "diagnosis_text")
S("stroke_location", ST, "Lesion location. Curated CSV diagnosis where specified, else registry.", f"{CSV} / {REGISTRY_XLSX.name}", "", "diagnosis_category / stroke location", coding="right | left | posterior (brainstem and cerebellum counted as posterior)")
S("stroke_location_source", ST, "Source of stroke_location.", coding="clinical_csv | registry")
S("diagnosis_category", ST, "Original diagnosis category, preserving brainstem/cerebellar detail independently of the harmonized side.", CSV, "", "diagnosis_category")
S("toast", ST, "TOAST etiology class. Spelling of 'Occlusion' corrected.", reg="TOAST", kind="custom")
S("mrs_prestroke", ST, "Modified Rankin Scale before stroke.", reg="mRS before stroke", kind="int", coding="0-6")
S("onset_to_admission_min", ST, "Time from stroke onset to hospital admission.", reg="time from stroke onset to hospital admission/minutes", kind="int", unit="min", unit_status="stated")
S("nihss_admission", ST, "NIHSS on admission.", reg="NIHSS on admission", kind="int", coding="0-42")
S("nihss_24h", ST, "NIHSS 24 h after admission.", reg="NIHSS 24h", kind="int", coding="0-42")
S("ivt", ST, "Intravenous thrombolysis.", reg="IVT: no - 0; yes - 1", kind="int", coding="1 yes, 0 no")
S("mechanical_thrombectomy", ST, "Mechanical thrombectomy.", reg="mechanical thrombectomy: 0 - no; 1 - yes", kind="int", coding="1 yes, 0 no")
S("care_pathway", ST, "Admission route. Source blanks remain unknown. The redundant causative-treatment flag is checked against IVT and thrombectomy.", reg="drip and ship=2; mothership=1", kind="custom", coding="mothership | drip_and_ship")
S("aspects", ST, "ASPECTS score.", CSV, "", "aspects_score", coding="0-10")
S("ct_perfusion_cbf", ST, "CBF from CT perfusion post-processing. Header gives no definition.", reg="(CBF) in post processing", kind="float", unit_status="unknown")
S("ct_perfusion_mismatch_volume_ml", ST, "Perfusion mismatch volume from post-processing.", reg="(Mismatch volume) in post processing", kind="float", unit="mL", unit_status="assumed")
S("global_cortical_atrophy", ST, "Global cortical atrophy as recorded. The scale is not defined in the source.", reg="global cortical atrophy", kind="float", unit_status="unknown")
S("hemorrhagic_transformation", ST, "Hemorrhagic transformation.", reg="hemorrhagic transformation 0 - NO, 1 no symptomatic , 2 symptomatic", kind="custom", coding="none | asymptomatic | symptomatic")
S("hemorrhage_type", ST, "Hemorrhage type.", reg="Hi1=1; Hi2=2, PH1=3, PH2-4, , 5 hemorrhage in different location", kind="custom", coding="none | HI1 | HI2 | PH1 | PH2 | other_location")
S("sah", ST, "Subarachnoid hemorrhage. Source blanks remain unknown; no negative status is inferred.", reg="SAH", kind="int", coding="1 yes")
S("body_temperature_c", ST, "Body temperature on admission.", reg="body temperature on admission", kind="float", unit="degC", unit_status="assumed")
S("sbp_mmhg", ST, "Systolic blood pressure on admission.", reg="SBP on admission", kind="int", unit="mmHg", unit_status="assumed")
S("dbp_mmhg", ST, "Diastolic blood pressure on admission.", reg="DBP on admission", kind="int", unit="mmHg", unit_status="assumed")
S("glucose_mg_dl", ST, "Glucose on admission.", reg="glukoza mg%", kind="float", unit="mg/dL", unit_status="stated")
S("wbc_10e9_l", ST, "White blood cell count.", reg="WBC", kind="float", unit="10^9/L", unit_status="assumed")
S("plt_10e9_l", ST, "Platelet count.", reg="PLT", kind="float", unit="10^9/L", unit_status="assumed")
S("inr", ST, "INR.", reg="INR", kind="float")
S("hgb_g_dl", ST, "Hemoglobin.", reg="HGB", kind="float", unit="g/dL", unit_status="assumed")
S("crp_mg_l", ST, "C-reactive protein.", reg="CRP", kind="float", unit="mg/L", unit_status="assumed")
S("fibrinogen_g_l", ST, "Fibrinogen.", reg="Fibrynogen", kind="float", unit="g/L", unit_status="assumed")
S("tsh_uiu_ml", ST, "Thyroid-stimulating hormone.", reg="TSH (ulU/mL)", kind="float", unit="uIU/mL", unit_status="stated")
S("total_cholesterol_mmol_l", ST, "Total cholesterol.", reg="total cholesterol", kind="float", unit="mmol/L", unit_status="assumed")
S("hdl_mmol_l", ST, "HDL cholesterol.", reg="HDL", kind="float", unit="mmol/L", unit_status="assumed")
S("ldl_mmol_l", ST, "LDL cholesterol.", reg="LDL", kind="float", unit="mmol/L", unit_status="assumed")
S("triglycerides_mmol_l", ST, "Triglycerides.", reg="triglycerides", kind="float", unit="mmol/L", unit_status="assumed")
S("creatinine_umol_l", ST, "Creatinine. Non-numeric values are marked unknown and retained in source_evidence. Values above 300 in the assumed unit are kept and flagged.", reg="kreatynine", kind="float", unit="umol/L", unit_status="assumed")
S("apoe_genotype", ST, "ApoE genotype, alleles sorted (registry writes e4/e3 and e3/e2). Other encodings are checked; any disagreement is retained in source_evidence.", reg="ApoE", kind="custom", coding="e2/e3 | e3/e3 | e3/e4")
S("mrs_discharge", ST, "Modified Rankin Scale at discharge.", reg="mRS discharge", kind="int", coding="0-6")
S("mrs_3m", ST, "Modified Rankin Scale 3 months after stroke.", reg="mRS 3month after stroke", kind="int", coding="0-6")
S("moca_baseline", ST, "MoCA at baseline.", CSV, "", "moca_total_baseline", coding="0-30")
S("moca_3m", ST, "MoCA 3 months after stroke.", CSV, "", "moca_total_3m", coding="0-30")
S("rbans_total_3m", ST, "RBANS total score 3 months after stroke.", CSV, "", "rbans_total_3m")
S("death", ST, "Recorded death indicator; the follow-up window is not supplied.", CSV, "", "death", coding="1 yes, 0 no")
S("death_followup_days", ST, "Unknown in the supplied sources; recorded death cannot be assigned a follow-up window.", unit="days", unit_status="unknown")
S("qc_flags", B, "Semicolon-separated explicit data-quality codes. none means no flagged discrepancy, not complete clinical information.")
S("spectral_file_count", B, "Number of original txt spectra under this participant folder.", STORE, coding="nonnegative integer")
S("spectral_sample_count", B, "Number of distinct technical sample directories; not a count of biological participants.", STORE)
S("spectral_regions", B, "JSON list of Raman region codes observed in file names.", STORE)
S("spectral_relative_paths", B, "JSON list of participant directories, relative to the repository root.", STORE)
S("spectral_filename_subject_ids", B, "JSON list of participant IDs encoded in spectrum filename prefixes. Compare with spectral_subject_id.", STORE)
S("inferred_fields", B, "JSON mapping of inferred or provisionally selected fields to their source and reason. Unlisted fields were copied or decoded without a flagged selection.")
S("unconfirmed_fields", B, "JSON list of populated fields whose units or operational definitions remain unconfirmed. unknown markers separately identify unavailable observations.")
S("source_records", B, "JSON mapping of source aliases to file, sheet and 1-based original data row; registry record_number is preserved here.")
S("source_evidence", B, "Sparse JSON source values that differ from the selected value or its reconstructible encoding, including source blanks and invalid tokens. Alias definitions are in source_records.")
OUT_COLUMNS = [s["column"] for s in SPECS]

MISSING = {"unknown", "not_applicable", "not_available"}
CSV_MAP = {
    "participant_id": "participant_key", "age_years": "age_years", "gender": "sex",
    "diagnosis_text": "diagnosis_text", "diagnosis_category": "diagnosis_category",
    "rbans_total_3m": "rbans_total_3m", "moca_total_baseline": "moca_baseline",
    "moca_total_3m": "moca_3m", "aspects_score": "aspects", "death": "death",
}
PATIENT_MAP = {
    "Sample No": "participant_key", "Identification code of each participant": "study_code",
    "REDCap Record ID": "redcap_record_id", "Age": "age_years", "Sex": "sex",
}
CONTROL_MAP = dict(zip(CT_COLS, [
    "study_code", "sex", "age_years", "blood_collection_date", "hypertension", "diabetes",
    "atrial_fibrillation", "coronary_artery_disease", "control_nicotinism",
    "control_past_smoking_over_6m", "hyperlipidemia_or_lipid_lowering_drug",
    "antiplatelet", "anticoagulant", "statin", "other_medication",
], strict=True))
REGISTRY_MAP = {s["reg"]: s["column"] for s in SPECS if s["reg"] and s["kind"] != "custom"}
REGISTRY_MAP.update({"age": "age_years", "gender 1=females": "sex", "TOAST": "toast", "ApoE": "apoe_genotype"})
CODED_REGISTRY = {
    SMOKING_HEADER: ("smoking_status", {0: "never", 1: "current", 2: "former"}),
    LOC_HEADER: ("stroke_location", LOC_REG),
    "drip and ship=2; mothership=1": ("care_pathway", {1: "mothership", 2: "drip_and_ship"}),
    "hemorrhagic transformation 0 - NO, 1 no symptomatic , 2 symptomatic": (
        "hemorrhagic_transformation", {0: "none", 1: "asymptomatic", 2: "symptomatic"}),
    "Hi1=1; Hi2=2, PH1=3, PH2-4, , 5 hemorrhage in different location": (
        "hemorrhage_type", {0: "none", 1: "HI1", 2: "HI2", 3: "PH1", 4: "PH2", 5: "other_location"}),
}
FLAG_MEANINGS = {
    "SEX_FOLDER_CONFLICT": "Clinical sex differs from the original spectrum folder; sample identity needs review.",
    "SEX_CLINICAL_CONFLICT": "Clinical sources disagree on sex; the selected source is provisional.",
    "SEX_CSV_ENCODING_CONFLICT": "The CSV gender and is_female encodings disagree.",
    "AGE_CLINICAL_CONFLICT": "Clinical ages disagree; CSV then patient sheet then registry is the selection order.",
    "LOCATION_CSV_REGISTRY_CONFLICT": "CSV and registry side disagree; CSV is provisionally selected.",
    "LOCATION_FILLED_FROM_REGISTRY": "The CSV has no specified side; registry provides the side.",
    "TOAST_CONFLICTS_WITH_DIAGNOSIS_TEXT": "Registry TOAST differs from etiology inferred from the diagnosis text; TOAST is retained provisionally.",
    "APOE_FIELDS_DISAGREE": "Genotype text, allele counts or one-hot indicators disagree; text then counts then one-hot is the selection order.",
    "APOE_ENCODING_INVALID": "An ApoE encoding is incomplete or invalid; original cells are preserved.",
    "CARE_PATHWAY_MISSING_AFTER_IVT_ONLY": "An IVT-only record lacks an admission route; route remains unknown.",
    "STUDY_CODE_IRREGULAR": "The supplied study code differs from the naming convention; it is preserved.",
    "CREATININE_OUTLIER": "Creatinine exceeds 300 in the assumed unit; no decimal correction is made.",
    "CREATININE_NON_NUMERIC": "The source creatinine token is not numeric; it is retained in source_evidence.",
    "FIBRINOGEN_LOW_UNIT_UNCONFIRMED": "Fibrinogen is below 1 in the assumed unit; value and unit require review.",
    "CONTROL_NO_CLINICAL_RECORD": "A raw control folder has no clinical record.",
    "STROKE_NO_CLINICAL_RECORD": "A raw patient folder has no record in the supplied study sources or registry.",
    "CONTROL_EMPTY_RECORD": "A control record lacks all risk-factor and medication entries.",
    "CONTROL_RECORD_NO_SPECTRA": "A control record has no matching participant folder containing spectra.",
    "STROKE_RECORD_NO_SPECTRA": "A patient record has no matching participant folder containing spectra.",
    "SPECTRAL_FILENAME_FOLDER_CONFLICT": "Spectrum filename participant prefixes disagree with the enclosing participant folder.",
    "POSSIBLE_SPECTRA_IN_ANOTHER_FOLDER": "Another participant folder has this record ID in a filename; no relinking is performed.",
    "MULTIPLE_SEX_FOLDERS": "The same cohort and participant ID occur under multiple sex folders.",
    "SMOKING_BOTH_FLAGS_CURRENT_ASSIGNED": "Both original control smoking flags are positive; current is a provisional interpretation of Nicotinism.",
    "REPERFUSION_ENCODING_CONFLICT": "The causative-treatment flag disagrees with IVT or thrombectomy.",
    "DNA_NUMBER_DUPLICATED": "A nonmissing DNA sample number is shared by more than one study patient.",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def missing(value):
    return value is None or (isinstance(value, str) and (not value.strip() or value in MISSING)) or bool(pd.isna(value))


def scalar(value):
    if missing(value):
        return None
    return value.item() if hasattr(value, "item") else value


def dump(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def numeric(value, integer=False):
    if missing(value):
        return None
    try:
        result = float(value)
    except (ValueError, TypeError):
        raise ValueError("A required numeric field contains a non-numeric token.") from None
    require(math.isfinite(result), "A numeric source value is not finite.")
    if integer:
        require(result.is_integer(), "An integer-coded source field is fractional.")
        return int(result)
    return result


def identifier(value, control=False):
    if control:
        require(isinstance(value, str) and re.fullmatch(r"K\d+", value.strip()) is not None,
                "A control sample identifier has an unsupported format.")
        value = value.strip()[1:]
    result = numeric(value, integer=True)
    require(result is not None and result > 0, "A participant identifier is missing or nonpositive.")
    return result


def date_text(value):
    if missing(value):
        return None
    require(not isinstance(value, (int, float)), "An Excel date lacks date formatting; no date is guessed.")
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def genotype(value):
    if missing(value):
        return None
    parts = str(value).strip().lower().split("/")
    require(len(parts) == 2 and all(p in {"e2", "e3", "e4"} for p in parts), "An ApoE genotype is invalid.")
    return "/".join(sorted(parts))


def decode(value, mapping):
    if missing(value):
        return None
    require(value in mapping, "An unrecognized categorical source code was found.")
    return mapping[value]


def clean_frame(frame):
    frame = frame.copy()
    frame.columns = [" ".join(str(c).split()) for c in frame.columns]
    require(frame.columns.is_unique, "Source column names collide after whitespace normalization.")
    return frame.map(lambda v: None if missing(v) else v.strip() if isinstance(v, str) else v)


def load_sources(data_dir):
    files = {
        "clinical_csv": data_dir / CLINICAL_CSV.name,
        "patient_sheet": data_dir / SHEETS_XLSX.name,
        "control_sheet": data_dir / SHEETS_XLSX.name,
        "registry": data_dir / REGISTRY_XLSX.name,
    }
    csv = clean_frame(pd.read_csv(files["clinical_csv"], encoding="utf-8-sig", keep_default_na=False))
    patient = clean_frame(pd.read_excel(files["patient_sheet"], sheet_name="Stroke Patients",
                                       engine="openpyxl", keep_default_na=False))
    header = pd.read_excel(files["control_sheet"], sheet_name="Control Group", header=None,
                           nrows=2, engine="openpyxl", keep_default_na=False)
    actual = [str(h).strip() for h in list(header.iloc[0, :4]) + list(header.iloc[1, 4:])]
    require(actual == CT_HEADER, "The control sheet layout differs from the supported two-row header.")
    control = pd.read_excel(files["control_sheet"], sheet_name="Control Group", header=None,
                            skiprows=2, engine="openpyxl", keep_default_na=False)
    require(len(control.columns) == len(CT_COLS), "The control sheet has unexpected columns.")
    control.columns = CT_COLS
    control = clean_frame(control)
    registry = clean_frame(pd.read_excel(files["registry"], sheet_name=REGISTRY_SHEET,
                                        engine="openpyxl", keep_default_na=False))
    frames = {"clinical_csv": csv, "patient_sheet": patient, "control_sheet": control, "registry": registry}
    expected = {
        "clinical_csv": set(CSV_MAP) | {"is_female"},
        "patient_sheet": set(PATIENT_MAP) | {"Unnamed: 3"},
        "control_sheet": set(CT_COLS),
        "registry": set(REGISTRY_MAP) | set(CODED_REGISTRY) | set(APOE_ONEHOT)
                    | {"e2", "e3", "e4", "number", RKEY, "initials", "causative tratment"},
    }
    keys = {"clinical_csv": "participant_id", "patient_sheet": "Sample No", "control_sheet": "sample_no", "registry": RKEY}
    result = {}
    for alias, frame in frames.items():
        require(set(frame.columns) == expected[alias], f"Unexpected source columns in {alias}; update the explicit source mapping.")
        frame["_source_row"] = frame.index + (3 if alias == "control_sheet" else 2)
        empty_rows = frame.drop(columns="_source_row").isna().all(axis=1)
        frame = frame.loc[~empty_rows].copy()
        require(frame[keys[alias]].notna().all(), f"A nonempty {alias} row has no participant identifier.")
        frame["_id"] = frame[keys[alias]].map(lambda v: identifier(v, control=alias == "control_sheet"))
        require(frame["_id"].is_unique, f"Duplicate participant identifiers in {alias}; merging would be ambiguous.")
        result[alias] = frame.set_index("_id").to_dict("index")
    return result, files


def scan_spectra(raw_dir):
    require(raw_dir.is_dir(), "The original data/raw directory is unavailable.")
    subjects = {}
    pattern = re.compile(r"(\d+)_(\d+)_(\d+)_range_(SS|FP)_meas(\d+)\.txt", re.IGNORECASE)

    def walk_error(error):
        raise OSError("The raw spectrum inventory is incomplete because a directory cannot be read.") from error

    for base, dirs, files in os.walk(raw_dir, onerror=walk_error):
        dirs.sort()
        for name in sorted(files):
            path = Path(base) / name
            if path.suffix.lower() != ".txt":
                continue
            parts = path.relative_to(raw_dir).parts
            require(len(parts) == 5, "A spectrum has an unsupported directory depth.")
            condition, sex, subject, sample, _ = parts
            require(condition in {"Disease", "Control"} and sex in {"Female", "Male"}, "Unknown cohort or sex folder.")
            require(re.fullmatch(r"sample_\d+", sample) is not None, "An unsupported technical-sample folder was found.")
            match = pattern.fullmatch(name)
            require(match is not None, "A spectrum filename has an unsupported format.")
            key = ("stroke" if condition == "Disease" else "control", identifier(subject))
            entry = subjects.setdefault(key, {"sex": set(), "paths": set(), "samples": set(),
                                              "prefixes": set(), "regions": set(), "files": 0, "folder_ids": set()})
            entry["sex"].add(sex.lower())
            entry["paths"].add((Path("data") / "raw" / condition / sex / subject).as_posix())
            entry["samples"].add(path.parent.relative_to(raw_dir).as_posix())
            entry["prefixes"].add(str(identifier(match.group(1))))
            entry["regions"].add(match.group(4).upper())
            entry["folder_ids"].add(subject)
            entry["files"] += 1
    require(bool(subjects), "No original spectra were found; refusing to claim that every subject lacks spectra.")
    return subjects


def choose(field, candidates, row, flags, inferred, conflict_code):
    available = [(alias, value) for alias, value in candidates if not missing(value)]
    if not available:
        row[field] = None
        return None
    alias, value = available[0]
    row[field] = value
    if any(other != value for _, other in available[1:]):
        flags.add(conflict_code)
        inferred[field] = {"source": alias, "method": "provisional_source_precedence", "reason": "sources_disagree"}
    elif alias != candidates[0][0]:
        inferred[field] = {"source": alias, "method": "cross_source_fill", "reason": "preferred_source_missing"}
    return alias


def sex_text(value):
    if missing(value):
        return None
    result = str(value).strip().lower()
    require(result in {"female", "male"}, "An unsupported clinical sex value was found.")
    return result


def build_stroke(rows, row, flags, inferred):
    cl, pt, reg = (rows.get(alias, {}) for alias in ("clinical_csv", "patient_sheet", "registry"))
    choose("age_years", [("clinical_csv", numeric(cl.get("age_years"))), ("patient_sheet", numeric(pt.get("Age"))),
                         ("registry", numeric(reg.get("age")))], row, flags, inferred, "AGE_CLINICAL_CONFLICT")
    row["sex_source"] = choose("sex", [("clinical_csv", sex_text(cl.get("gender"))), ("patient_sheet", sex_text(pt.get("Sex"))),
                                      ("registry", decode(reg.get("gender 1=females"), {1: "female", 0: "male"}))],
                               row, flags, inferred, "SEX_CLINICAL_CONFLICT")
    if not missing(cl.get("is_female")) and not missing(cl.get("gender")):
        if numeric(cl["is_female"], integer=True) != int(sex_text(cl["gender"]) == "female"):
            flags.add("SEX_CSV_ENCODING_CONFLICT")
    row["study_code"] = scalar(pt.get("Identification code of each participant"))
    row["redcap_record_id"] = scalar(pt.get("REDCap Record ID"))
    for source, target in CSV_MAP.items():
        if source not in {"participant_id", "age_years", "gender"}:
            row[target] = numeric(cl.get(source), integer=True) if source in {
                "rbans_total_3m", "moca_total_baseline", "moca_total_3m", "aspects_score", "death"} else scalar(cl.get(source))
    for spec in SPECS:
        if not spec["reg"] or spec["kind"] == "custom":
            continue
        value = reg.get(spec["reg"])
        if spec["kind"] == "text":
            row[spec["column"]] = str(value) if not missing(value) else None
        elif spec["column"] == "creatinine_umol_l":
            try:
                row[spec["column"]] = numeric(value)
            except ValueError:
                row[spec["column"]] = None
                flags.add("CREATININE_NON_NUMERIC")
        else:
            row[spec["column"]] = numeric(value, integer=spec["kind"] == "int")
    for header, (target, codes) in CODED_REGISTRY.items():
        if target != "stroke_location":
            row[target] = decode(reg.get(header), codes)
    category = cl.get("diagnosis_category")
    require(missing(category) or category in set(LOC_CSV) | {"ischemic_stroke_unspecified"}, "An unmapped CSV diagnosis category was found.")
    row["stroke_location_source"] = choose("stroke_location", [
        ("clinical_csv", LOC_CSV.get(category)), ("registry", decode(reg.get(LOC_HEADER), LOC_REG))],
        row, flags, inferred, "LOCATION_CSV_REGISTRY_CONFLICT")
    if row["stroke_location_source"] == "registry":
        flags.add("LOCATION_FILLED_FROM_REGISTRY")
    row["toast"] = str(reg["TOAST"]).replace("Oclussion", "Occlusion") if not missing(reg.get("TOAST")) else None
    for word, expected in TOAST_TEXT.items():
        if word in str(row.get("diagnosis_text", "")).lower() and not missing(row["toast"]) and row["toast"] != expected:
            flags.add("TOAST_CONFLICTS_WITH_DIAGNOSIS_TEXT")
            inferred["toast"] = {"source": "registry", "method": "provisional_structured_field", "reason": "diagnosis_text_suggests_another_etiology"}
    counts = [numeric(reg.get(a), integer=True) for a in ("e2", "e3", "e4")]
    count_genotype = None
    if all(v is not None for v in counts) and all(0 <= v <= 2 for v in counts) and sum(counts) == 2:
        count_genotype = "/".join(a for a, n in zip(("e2", "e3", "e4"), counts, strict=True) for _ in range(n))
    elif any(v is not None for v in counts):
        flags.add("APOE_ENCODING_INVALID")
    onehot = [numeric(reg.get(c), integer=True) for c in APOE_ONEHOT]
    hot_genotype = None
    if all(v in {0, 1} for v in onehot) and sum(onehot) == 1:
        hot_genotype = APOE_ONEHOT[onehot.index(1)].split("=")[0]
    elif any(v is not None for v in onehot):
        flags.add("APOE_ENCODING_INVALID")
    choose("apoe_genotype", [("registry_genotype_text", genotype(reg.get("ApoE"))),
                             ("registry_allele_counts", count_genotype), ("registry_onehot", hot_genotype)],
           row, flags, inferred, "APOE_FIELDS_DISAGREE")
    creat = row.get("creatinine_umol_l")
    if creat is not None and creat > CREATININE_OUTLIER:
        flags.add("CREATININE_OUTLIER")
    fib = row.get("fibrinogen_g_l")
    if fib is not None and fib < 1:
        flags.add("FIBRINOGEN_LOW_UNIT_UNCONFIRMED")
    ivt, mt = row.get("ivt"), row.get("mechanical_thrombectomy")
    if ivt == 1 and mt == 0 and missing(row.get("care_pathway")):
        flags.add("CARE_PATHWAY_MISSING_AFTER_IVT_ONLY")
    if ivt is not None and mt is not None and not missing(reg.get("causative tratment")):
        if numeric(reg["causative tratment"], integer=True) != int(bool(ivt) or bool(mt)):
            flags.add("REPERFUSION_ENCODING_CONFLICT")
    if row["study_code"] is not None and row["study_code"] != "KW_K" + row["participant_key"].split("_")[1]:
        flags.add("STUDY_CODE_IRREGULAR")


def control_smoking(nicotinism, past):
    for value in (nicotinism, past):
        require(missing(value) or value in {0, 1}, "An unsupported control smoking flag was found.")
    if nicotinism == 1:
        return "current"
    if nicotinism == 0 and past == 1:
        return "former"
    if nicotinism == 0 and past == 0:
        return "never"
    return None


def build_control(rows, row, flags, inferred):
    ct = rows.get("control_sheet", {})
    for source, target in CONTROL_MAP.items():
        value = ct.get(source)
        if source == "sex":
            value = sex_text(value)
        elif source == "blood_date":
            value = date_text(value)
        elif source not in {"sample_no"}:
            value = numeric(value, integer=True)
            if source != "age":
                require(value is None or value in {0, 1}, "A control indicator is not binary.")
        row[target] = scalar(value)
    row["sex_source"] = "control_sheet" if row["sex"] is not None else None
    nic, past = row["control_nicotinism"], row["control_past_smoking_over_6m"]
    row["smoking_status"] = control_smoking(nic, past)
    if row["smoking_status"] is not None:
        inferred["smoking_status"] = {"source": "control_sheet", "method": "provisional_definition", "reason": "nicotinism_current_smoking_definition_unconfirmed"}
    if nic == 1 and past == 1:
        flags.add("SMOKING_BOTH_FLAGS_CURRENT_ASSIGNED")
    if ct and all(missing(ct.get(c)) for c in CT_COLS[4:]):
        flags.add("CONTROL_EMPTY_RECORD")


def reconstruct_source_value(alias, column, row):
    """Recover an equivalent source encoding; discrepancies override this in source_evidence."""
    def get(col):
        return scalar(row.get(col))

    subject_id = int(row["participant_key"].split("_")[1])
    if alias == "clinical_csv":
        if column == "participant_id":
            return subject_id
        if column == "is_female":
            return None if get("sex") is None else int(get("sex") == "female")
        return get(CSV_MAP[column])
    if alias == "patient_sheet":
        if column == "Sample No":
            return subject_id
        if column == "Unnamed: 3":
            return None
        return get(PATIENT_MAP[column])
    if alias == "control_sheet":
        return get(CONTROL_MAP[column])
    if column == RKEY:
        return subject_id
    if column == "number":
        sources = row["source_records"]
        if isinstance(sources, str):
            sources = json.loads(sources)
        return sources["registry"]["record_number"]
    if column == "initials":
        return None
    if column == "gender 1=females":
        return None if get("sex") is None else int(get("sex") == "female")
    if column in REGISTRY_MAP:
        return get(REGISTRY_MAP[column])
    if column in CODED_REGISTRY:
        target, codes = CODED_REGISTRY[column]
        return {v: k for k, v in codes.items()}.get(get(target))
    if column == "causative tratment":
        ivt, mt = numeric(get("ivt")), numeric(get("mechanical_thrombectomy"))
        return None if ivt is None or mt is None else int(bool(ivt) or bool(mt))
    if column in APOE_ONEHOT:
        return None if get("apoe_genotype") is None else int(get("apoe_genotype") == column.split("=")[0])
    if column in {"e2", "e3", "e4"}:
        return None if get("apoe_genotype") is None else get("apoe_genotype").split("/").count(column)
    raise ValueError("An original source column has no preservation rule.")


def normalize_source_value(alias, column, value):
    value = scalar(value)
    if value is None:
        return None
    if (alias, column) in {("clinical_csv", "gender"), ("patient_sheet", "Sex"), ("control_sheet", "sex")}:
        return sex_text(value)
    if alias == "control_sheet" and column == "blood_date":
        return date_text(value)
    if alias == "registry" and column == "ApoE":
        return genotype(value)
    if alias == "registry" and column == "TOAST":
        return str(value).replace("Oclussion", "Occlusion")
    # Numeric cells and equivalent numeric strings carry the same information.
    try:
        return numeric(value)
    except ValueError:
        return str(value).strip()


def source_evidence(rows, row):
    evidence = {}
    for alias, original in rows.items():
        for column, value in original.items():
            if column.startswith("_"):
                continue
            equivalent = reconstruct_source_value(alias, column, row)
            if normalize_source_value(alias, column, value) != normalize_source_value(alias, column, equivalent):
                evidence.setdefault(alias, {})[column] = date_text(value) if alias == "control_sheet" and column == "blood_date" else scalar(value)
    return evidence


def validate_preservation(sources, master):
    for row in master.to_dict("records"):
        cohort, number = row["participant_key"].split("_")
        number = int(number)
        aliases = ("clinical_csv", "patient_sheet", "registry") if cohort == "stroke" else ("control_sheet",)
        evidence = json.loads(row["source_evidence"]) if isinstance(row["source_evidence"], str) else row["source_evidence"]
        for alias in aliases:
            original = sources[alias].get(number, {})
            for column, value in original.items():
                if column.startswith("_"):
                    continue
                recovered = evidence.get(alias, {}).get(column) if column in evidence.get(alias, {}) else reconstruct_source_value(alias, column, row)
                require(normalize_source_value(alias, column, value) == normalize_source_value(alias, column, recovered),
                        f"Source information was lost in {alias}, column {column}.")


def build_master(sources, spectra, files):
    study_patients = set(sources["clinical_csv"]) | set(sources["patient_sheet"]) | {n for c, n in spectra if c == "stroke"}
    study_controls = set(sources["control_sheet"]) | {n for c, n in spectra if c == "control"}
    records = []
    sheets = {"clinical_csv": None, "patient_sheet": "Stroke Patients", "control_sheet": "Control Group", "registry": REGISTRY_SHEET}
    for cohort, ids in (("stroke", study_patients), ("control", study_controls)):
        for number in sorted(ids):
            aliases = ("clinical_csv", "patient_sheet", "registry") if cohort == "stroke" else ("control_sheet",)
            rows = {a: sources[a][number] for a in aliases if number in sources[a]}
            flags, inferred = set(), {}
            row = dict.fromkeys(OUT_COLUMNS)
            row.update(participant_key=f"{cohort}_{number}", cohort=cohort, has_clinical_record=int(bool(rows)))
            if cohort == "stroke":
                build_stroke(rows, row, flags, inferred)
            else:
                build_control(rows, row, flags, inferred)
            entry = spectra.get((cohort, number))
            row.update(has_spectra=int(entry is not None), spectral_file_count=entry["files"] if entry else 0,
                       spectral_sample_count=len(entry["samples"]) if entry else 0,
                       spectral_subject_id=str(number) if entry else "not_available")
            for field, source in (("spectral_regions", "regions"), ("spectral_relative_paths", "paths"),
                                  ("spectral_filename_subject_ids", "prefixes")):
                row[field] = dump(sorted(entry[source])) if entry else "[]"
            if entry:
                row["sex_spectral_folder"] = next(iter(entry["sex"])) if len(entry["sex"]) == 1 else "unknown"
                if len(entry["sex"]) > 1 or len(entry["folder_ids"]) > 1:
                    flags.add("MULTIPLE_SEX_FOLDERS")
                if entry["prefixes"] != {str(number)}:
                    flags.add("SPECTRAL_FILENAME_FOLDER_CONFLICT")
                if missing(row["sex"]) and not missing(row["sex_spectral_folder"]):
                    row["sex"], row["sex_source"] = row["sex_spectral_folder"], "spectral_folder"
                    inferred["sex"] = {"source": "spectral_folder", "method": "provisional_folder_metadata", "reason": "clinical_sex_unavailable"}
                elif not missing(row["sex_spectral_folder"]) and row["sex"] != row["sex_spectral_folder"]:
                    flags.add("SEX_FOLDER_CONFLICT")
                if not rows:
                    flags.add(f"{cohort.upper()}_NO_CLINICAL_RECORD")
            else:
                flags.add(f"{cohort.upper()}_RECORD_NO_SPECTRA")
            row["source_records"] = {a: {"file": files[a].name, "sheet": sheets[a], "row": int(r["_source_row"])}
                                     for a, r in rows.items()}
            if "registry" in rows:
                row["source_records"]["registry"]["record_number"] = scalar(rows["registry"].get("number"))
            row["source_evidence"] = source_evidence(rows, row)
            row["inferred_fields"] = inferred
            row["_flags"] = flags
            records.append(row)
    by_key = {r["participant_key"]: r for r in records}
    for (cohort, number), entry in spectra.items():
        for prefix in entry["prefixes"] - {str(number)}:
            candidate = by_key.get(f"{cohort}_{int(prefix)}")
            if candidate is not None:
                candidate["_flags"].add("POSSIBLE_SPECTRA_IN_ANOTHER_FOLDER")
                candidate["inferred_fields"]["possible_spectral_link"] = {
                    "source": "spectrum_filename", "method": "unresolved_candidate_only",
                    "candidate_folder": f"{cohort}_{number}", "reason": "filename_prefix_matches_record_but_folder_id_differs"}
    dna_counts = Counter(r["dna_sample_number"] for r in records if not missing(r["dna_sample_number"]))
    for row in records:
        if not missing(row["dna_sample_number"]) and dna_counts[row["dna_sample_number"]] > 1:
            row["_flags"].add("DNA_NUMBER_DUPLICATED")
        flags = row.pop("_flags")
        require(flags <= set(FLAG_MEANINGS), "An undocumented quality flag was generated.")
        row["qc_flags"] = ";".join(sorted(flags)) or "none"
        if not row["has_spectra"]:
            row["spectral_link_status"] = "no_spectra"
        elif not row["has_clinical_record"]:
            row["spectral_link_status"] = "no_clinical_record"
        elif flags & {"SEX_FOLDER_CONFLICT", "SPECTRAL_FILENAME_FOLDER_CONFLICT", "MULTIPLE_SEX_FOLDERS",
                      "SEX_CLINICAL_CONFLICT", "SEX_CSV_ENCODING_CONFLICT"}:
            row["spectral_link_status"] = "identity_review_required"
        else:
            row["spectral_link_status"] = "matched"
        uncertain = {s["column"] for s in SPECS if s["unit_status"] in {"assumed", "unknown"} and not missing(row[s["column"]])}
        uncertain |= {c for c in ("age_years", "hypertension", "diabetes", "atrial_fibrillation", "coronary_artery_disease")
                      if not missing(row[c])}
        if row["cohort"] == "control":
            uncertain |= {c for c in ("control_nicotinism", "smoking_status", "other_medication", "antiplatelet", "anticoagulant", "statin",
                                     "hyperlipidemia_or_lipid_lowering_drug") if not missing(row[c])}
        if not missing(row["death"]):
            uncertain.add("death")
        row["unconfirmed_fields"] = dump(sorted(uncertain))
        for field in ("source_records", "source_evidence", "inferred_fields"):
            row[field] = dump(row[field])
        for spec in SPECS:
            col = spec["column"]
            if not (isinstance(row[col], str) and row[col] in MISSING) and missing(row[col]):
                row[col] = "not_applicable" if spec["applies_to"] not in {"both", row["cohort"]} else "unknown"
    master = pd.DataFrame(records, columns=OUT_COLUMNS)
    expected = {f"stroke_{n}" for n in study_patients} | {f"control_{n}" for n in study_controls}
    require(set(master["participant_key"]) == expected and master["participant_key"].is_unique, "The participant union was not preserved.")
    require(int(master["spectral_file_count"].sum()) == sum(e["files"] for e in spectra.values()), "The raw spectrum inventory was not preserved.")
    require({(r.cohort, int(r.participant_key.split("_")[1])) for r in master.itertuples() if r.has_spectra} == set(spectra),
            "Raw participant folders and the master do not have the same subjects.")
    validate_preservation(sources, master)
    return master


def build_dictionary():
    keep = ["column", "applies_to", "description", "unit", "unit_status", "coding", "source_file", "source_sheet", "source_column"]
    dictionary = pd.DataFrame(SPECS)[keep].copy()
    dictionary["missing_value_policy"] = "unknown: unavailable or invalid observation; not_applicable: field belongs to other cohort; not_available: no spectra; []/{}: empty metadata collection"
    dictionary["unit_status"] = dictionary["unit_status"].replace("", "not_applicable")
    dictionary.loc[dictionary["column"] == "qc_flags", "coding"] = dump(FLAG_MEANINGS)
    return dictionary.fillna("")


def verify_serialized(master, sources):
    require(master.columns.tolist() == OUT_COLUMNS, "The serialized CSV schema differs from the dictionary.")
    require(master["participant_key"].is_unique, "The CSV has duplicate participant keys.")
    require(not master.isna().any().any(), "The CSV contains unexplained reference distribution cells.")
    require(not master.map(lambda v: isinstance(v, str) and not v.strip()).any().any(), "The CSV contains blank cells.")
    for field in ("source_records", "source_evidence", "inferred_fields", "unconfirmed_fields", "spectral_regions",
                  "spectral_relative_paths", "spectral_filename_subject_ids"):
        for value in master[field]:
            json.loads(value)
    validate_preservation(sources, master)


def build_notes(master, sources, files, spectra):
    counts = Counter(code for flags in master["qc_flags"] if flags != "none" for code in flags.split(";"))
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in set(files.values())}
    lines = [
        "# Clinical master: source mapping and limitations", "",
        "Rebuild: `uv run python scripts/build_clinical_master.py`. No input table, spectrum, pipeline configuration or processed store is modified.", "",
        f"The table has {len(master)} unique study participants: {int((master.cohort == 'stroke').sum())} patients and {int((master.cohort == 'control').sum())} controls.",
        f"The original folders contain {len(spectra)} participants and {sum(e['files'] for e in spectra.values())} txt spectra; every folder is represented exactly once by cohort and participant ID.",
        f"The registry contains {len(sources['registry'])} records; {len(set(sources['registry']) - {int(k.split('_')[1]) for k in master.participant_key if k.startswith('stroke_')})} are outside the study and excluded.", "",
        "## Reading the CSV", "",
        "There are no empty cells. `unknown` means a clinical observation is unavailable, unrecorded, or invalid; `not_applicable` marks a variable belonging to the other cohort; `not_available` marks a missing spectrum folder ID. Zero and FALSE are never substituted for missing observations. Empty JSON lists/objects mean no entries, not missing clinical observations.",
        "For numeric analysis, read with `pd.read_csv(path, na_values=['unknown', 'not_applicable', 'not_available'], encoding='utf-8-sig')`. Filter by cohort and select clinical predictors explicitly; identity/QC/provenance metadata are not model features. `qc_flags = none` does not certify complete data.", "",
        "## Source-based selections", "",
        "The participant key is cohort plus the numeric Sample No / participant_id / registry primary-database number / original participant folder ID. Patient and control numbers can overlap. Registry `number` is not a join key; it is preserved as record_number in source_records. Irregular study-code strings are preserved rather than silently repaired.",
        "Age and sex use the clinical CSV, then patient sheet, then registry. Controls use the control sheet. Sex is inferred from a raw folder only when no clinical sex is supplied, and is marked in inferred_fields. Agreement among clinical tables does not establish that a mismatched spectrum belongs to that person.",
        "Specified CSV stroke side takes precedence over the registry. An unspecified side is filled from the registry. A disagreement is a provisional selection with both values recoverable. The original diagnosis category preserves finer location detail. Registry TOAST is provisionally retained when diagnosis text suggests another etiology.",
        "ApoE selection uses genotype text, then a valid pair of allele counts, then a valid one-hot encoding. Dependent encodings are not independent votes. Conflicting encodings are preserved in source_evidence, and the selected genotype remains provisional.",
        "Control Nicotinism is provisionally interpreted as current smoking. Both original smoking flags are retained; a positive past-smoking flag can coexist with a positive Nicotinism flag. Partial missing flags are never classified as never smoking by default.",
        "No control filename/folder mismatch is automatically relinked. Filename prefixes, folder paths, and candidate links are visible in the CSV. Clinical-only participants remain in the table; spectra-only folders have explicit missing clinical status.", "",
        "## Units, missingness and unavailable definitions", "",
        "The dictionary marks units as stated, assumed, unknown, or not_applicable. Unit suffixes for assumed units are conventions, not confirmation or conversion. unconfirmed_fields lists populated fields with unconfirmed units, reference times, or definitions. No numeric clinical value is statistically imputed: a cohort median would fabricate an individual observation and could bias subsequent analysis.",
        "Creatinine values are not divided by ten. Non-numeric creatinine tokens are retained in source_evidence and the numeric field is unknown. Fibrinogen below 1 is retained and flagged without deciding whether it is physiological, a decimal error, or a unit issue. These checks use the previously assumed units and are review triggers, not diagnoses.",
        "Blank SAH is unknown. Missing care pathway is unknown. No patient blood-collection date or death-follow-up window exists in the supplied sources. CBF post-processing and global cortical atrophy scales are undefined. The meaning of Other medication, current-smoking interpretation of Nicotinism, age reference date, and control risk-factor/medication reference times are not confirmed by the headers.", "",
        "## Preservation and validation", "",
        "Each selected source cell is recoverable as the canonical value, an equivalent redundant encoding, or a sparse original value in source_evidence. This includes source blanks when a different source supplies the selected field. source_records supplies file, worksheet and original row. Blank initials/unused columns contain no additional observations; any future nonblank entry is preserved in source_evidence. ApoE one-hot/count fields and causative-treatment flags are omitted as standalone predictors only because their encodings can be reconstructed or their exceptions are retained.",
        "The build validates schemas, unique keys, source joins, every selected source cell, all raw participant folders and file totals, dictionary order, JSON metadata, and the serialized CSV before replacing any prior output. Errors stop the build without deleting prior outputs. No raw source values or patient IDs are printed to the terminal.",
        "The existing processed stores were not rebuilt. Their pipeline may continue to use the original clinical CSV and sex-folder labels; this standalone table does not change previously generated results.", "",
        "## Quality flags", "", "| Flag | Participants | Meaning |", "|---|---:|---|",
    ]
    lines += [f"| {code} | {counts.get(code, 0)} | {meaning} |" for code, meaning in sorted(FLAG_MEANINGS.items())]
    lines += ["", "## Input fingerprints", "", "SHA-256 hashes of the three original clinical files:", ""]
    lines += [f"- `{name}`: `{digest}`" for name, digest in sorted(hashes.items())]
    return "\n".join(lines) + "\n"


def write_outputs(master, dictionary, notes, data_dir, sources):
    # Stage and validate all outputs before replacing previous deliverables.
    # Use inherited directory permissions: TemporaryDirectory's private Windows
    # ACL would make moved deliverables inaccessible to the workspace owner.
    stage = data_dir / f".clinical_master_{uuid4().hex}"
    stage.mkdir()
    names = (OUT_CSV.name, OUT_DICT.name, "clinical_master_notes.md")
    try:
        master.to_csv(stage / OUT_CSV.name, index=False, encoding="utf-8-sig", lineterminator="\r\n")
        dictionary.to_csv(stage / OUT_DICT.name, index=False, encoding="utf-8-sig", lineterminator="\r\n")
        (stage / "clinical_master_notes.md").write_bytes(notes.replace("\n", "\r\n").encode("utf-8"))
        back = pd.read_csv(stage / OUT_CSV.name, encoding="utf-8-sig", keep_default_na=False, dtype=str)
        verify_serialized(back, sources)
        require(back.equals(master.astype(str)), "CSV round-trip changed a value or table dimensions.")
        require(dictionary["column"].tolist() == back.columns.tolist(), "Dictionary columns do not match the CSV order.")
        for name in names:
            os.replace(stage / name, data_dir / name)
    finally:
        for name in names:
            (stage / name).unlink(missing_ok=True)
        stage.rmdir()


def main():
    sources, files = load_sources(DATA)
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in set(files.values())}
    spectra = scan_spectra(DATA / "raw")
    master = build_master(sources, spectra, files)
    dictionary = build_dictionary()
    notes = build_notes(master, sources, files, spectra)
    require(all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in hashes.items()),
            "An original clinical input changed during the build.")
    write_outputs(master, dictionary, notes, DATA, sources)
    print(f"Validated {len(master)} participants, {len(master.columns)} columns, {len(spectra)} raw subjects, "
          f"{sum(e['files'] for e in spectra.values())} spectra; no blank cells or lost source observations.")
    print("Identity status counts: " + dump(dict(Counter(master["spectral_link_status"]))))
    print("Outputs: clinical_master.csv, clinical_master_dictionary.csv, clinical_master_notes.md")


if __name__ == "__main__":
    main()
