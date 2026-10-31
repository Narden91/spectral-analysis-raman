# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=3.0.1", "openpyxl", "pyarrow"]
# ///
"""Merge the clinical sources into one participant-level master table.

Sources: data/clinical_data.csv, the two-sheet "Patients and Control" workbook
and the 2026 stroke registry. Spectra availability comes from the
preprocessed store (subjects.parquet); data/raw is not scanned.

Writes data/clinical_master.csv and data/clinical_master_dictionary.csv, then
re-reads them and checks them against the sources. On a failed check the
outputs are deleted and the script exits with an error.
data/clinical_data.csv is never modified, so the store manifest stays valid.

Run: uv run scripts/build_clinical_master.py
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CLINICAL_CSV = DATA / "clinical_data.csv"
SHEETS_XLSX = DATA / "Finlandia  IBIOSTROKE Patients and Control.xlsx"
REGISTRY_XLSX = DATA / "Finlandia Ibiostroke 2026 (without initials).xlsx"
REGISTRY_SHEET = "Baza 1000 -import-czyszczenieV3"
STORE_SUBJECTS = DATA / "preprocessed" / "subjects.parquet"
OUT_CSV = DATA / "clinical_master.csv"
OUT_DICT = DATA / "clinical_master_dictionary.csv"

# The registry holds the participant number under this header. Its `number`
# column is only a row index; joining on it matches by chance.
RKEY = "Number in the primary large data base"
LOC_HEADER = "stroke location: PRAWA (right) /LEWA (left) /TYL (posterior)"  # double space in the file collapsed
CREATININE_OUTLIER = 300  # umol/L; rare outside renal failure, so flag for review

# Expected result of the checks; a change means a source or the store changed.
N_STROKE, N_CONTROL_SHEET, N_CONTROL_EXTRA = 102, 51, 2
N_SPECTRA = {"stroke": 97, "control": 50}
EXPECTED_SEX_CONFLICTS = {f"stroke_{n}" for n in (226, 245, 268, 278, 636)}
EXPECTED_LOCATION_CONFLICTS = {"stroke_665"}

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

FLAGS: defaultdict[str, set[str]] = defaultdict(set)  # qc code -> participant keys
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


CSV, SHEETS, STORE = CLINICAL_CSV.name, SHEETS_XLSX.name, STORE_SUBJECTS.name


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
S("spectral_subject_id", B, "Folder name under data/raw, equal to subject_id in the preprocessed store. Empty when the participant has no spectra.", STORE, "", "subject_id")
S("has_spectra", B, "Spectra for this participant are in the preprocessed store.", STORE, "", "subject_id", coding="1 yes, 0 no")
S("has_clinical_record", B, "A row for this participant exists in the clinical sources. 0 only for controls with spectra but no row in the Control Group sheet.", coding="1 yes, 0 no")
S("study_code", B, "Study code as written in the source (patients KW_K<n>, controls K<n>).", SHEETS, "Stroke Patients / Control Group", "Identification code of each participant / Sample No")
S("redcap_record_id", ST, "REDCap record ID.", SHEETS, "Stroke Patients", "REDCap Record ID")
S("dna_sample_number", ST, "DNA sample number in the registry.", reg="DNA sample nimber", kind="text")
S("blood_collection_date", CT, "Blood collection date. Not available for patients.", SHEETS, "Control Group", "The date on which the blood was collected", unit="ISO date")
S("age_years", B, "Age. Patient age agrees across CSV, sheet and registry for 102/102.", f"{CSV} / {SHEETS}", "Control Group", "age_years / Age", unit="years", unit_status="stated")
S("sex", B, "Sex from the clinical record. Spectral-folder sex only when sex_source is spectral_folder.", f"{CSV} / {SHEETS}", "Control Group", "gender / Sex", coding="female | male")
S("sex_source", B, "Where sex comes from.", coding="clinical_record | spectral_folder")
S("sex_spectral_folder", B, "Sex implied by the data/raw folder (Female/Male), from the store. Differs from sex for 5 patients (see qc_flags).", STORE, "", "gender", coding="female | male")
S("weight_kg", ST, "Body weight.", reg="weight", kind="float", unit="kg", unit_status="assumed")
S("height_cm", ST, "Body height.", reg="hight", kind="float", unit="cm", unit_status="assumed")
S("years_of_schooling", ST, "Years of schooling.", reg="years of schooling", kind="float", unit="years", unit_status="assumed")
S("hypertension", B, "Hypertension. Controls: at blood draw; patients: before or at admission.", reg="hypertension", kind="int", coding="1 yes, 0 no", **shared("hypertension", "Hypertension"))
S("diabetes", B, "Diabetes mellitus. Same timing as hypertension.", reg="diabetes mellitus", kind="int", coding="1 yes, 0 no", **shared("diabetes mellitus", "Diabetes"))
S("atrial_fibrillation", B, "Atrial fibrillation. Same timing as hypertension.", reg="atrial fibrillation", kind="int", coding="1 yes, 0 no", **shared("atrial fibrillation", "Atrial Fibrillation"))
S("coronary_artery_disease", B, "Coronary artery disease. Same timing as hypertension.", reg="cronary artery disease", kind="int", coding="1 yes, 0 no", **shared("cronary artery disease", "Coronary Heart Disease"))
S("smoking_status", B, "Smoking. Controls: current if Nicotinism = 1 (also when the 'smoked in the past' flag = 1), else former if that flag = 1, else never. Registry coding for patients.", reg=SMOKING_HEADER, kind="custom", coding="never | former | current", **shared(SMOKING_HEADER, "Nicotinism + Smoking in the past (more than 6 months ago)"))
S("hyperlipidemia_or_lipid_lowering_drug", CT, "Hyperlipidemia or use of cholesterol-lowering drugs. Controls only; no patient equivalent.", SHEETS, "Control Group", "Hyperlipidemia or the use of cholesterol-lowering drugs", coding="1 yes, 0 no")
S("antiplatelet", B, "Chronic antiplatelet medication. Controls: at blood draw; patients: before stroke.", reg="chronic antiplatelets before stroke", kind="int", coding="1 yes, 0 no", **shared("chronic antiplatelets before stroke", "Antiplatelet Medication"))
S("anticoagulant", B, "Chronic anticoagulant medication. Same timing as antiplatelet.", reg="chronic anticoagulants before stroke", kind="int", coding="1 yes, 0 no", **shared("chronic anticoagulants before stroke", "Anticoagulant Medication"))
S("statin", B, "Statin use. Controls: at blood draw; patients: before stroke.", reg="chronic statin before stroke", kind="int", coding="1 yes, 0 no", **shared("chronic statin before stroke", "Statins"))
S("other_medication", CT, "Other medication taken. Controls only; meaning of 'Other' is not defined in the source.", SHEETS, "Control Group", "Other", coding="1 yes, 0 no")
S("diagnosis_text", ST, "Free-text diagnosis from the curated clinical CSV.", CSV, "", "diagnosis_text")
S("stroke_location", ST, "Lesion location. Curated CSV diagnosis where specified, else registry.", f"{CSV} / {REGISTRY_XLSX.name}", "", "diagnosis_category / stroke location", coding="right | left | posterior (brainstem and cerebellum counted as posterior)")
S("stroke_location_source", ST, "Source of stroke_location.", coding="clinical_csv | registry")
S("stroke_location_registry", ST, "Registry location, decoded: P (prawa) right, L (lewa) left, T (tyl) posterior.", reg=LOC_HEADER, kind="custom", coding="right | left | posterior")
S("toast", ST, "TOAST etiology class. Spelling of 'Occlusion' corrected.", reg="TOAST", kind="custom")
S("mrs_prestroke", ST, "Modified Rankin Scale before stroke.", reg="mRS before stroke", kind="int", coding="0-6")
S("onset_to_admission_min", ST, "Time from stroke onset to hospital admission.", reg="time from stroke onset to hospital admission/minutes", kind="int", unit="min", unit_status="stated")
S("nihss_admission", ST, "NIHSS on admission.", reg="NIHSS on admission", kind="int", coding="0-42")
S("nihss_24h", ST, "NIHSS 24 h after admission.", reg="NIHSS 24h", kind="int", coding="0-42")
S("ivt", ST, "Intravenous thrombolysis.", reg="IVT: no - 0; yes - 1", kind="int", coding="1 yes, 0 no")
S("mechanical_thrombectomy", ST, "Mechanical thrombectomy.", reg="mechanical thrombectomy: 0 - no; 1 - yes", kind="int", coding="1 yes, 0 no")
S("care_pathway", ST, "Admission route. Empty for the 21 patients with no reperfusion therapy and for 1 IVT-only patient (see qc_flags). 'causative treatment' is dropped: it equals IVT or thrombectomy for 102/102.", reg="drip and ship=2; mothership=1", kind="custom", coding="mothership | drip_and_ship")
S("aspects", ST, "ASPECTS score.", CSV, "", "aspects_score", coding="0-10")
S("ct_perfusion_cbf", ST, "CBF from CT perfusion post-processing. Header gives no definition.", reg="(CBF) in post processing", kind="float", unit_status="unknown")
S("ct_perfusion_mismatch_volume_ml", ST, "Perfusion mismatch volume from post-processing.", reg="(Mismatch volume) in post processing", kind="float", unit="mL", unit_status="assumed")
S("global_cortical_atrophy", ST, "Global cortical atrophy. Range 0-32 suggests a sum of regional scores; the scale is not defined in the source.", reg="global cortical atrophy", kind="float", unit_status="unknown")
S("hemorrhagic_transformation", ST, "Hemorrhagic transformation.", reg="hemorrhagic transformation 0 - NO, 1 no symptomatic , 2 symptomatic", kind="custom", coding="none | asymptomatic | symptomatic")
S("hemorrhage_type", ST, "Hemorrhage type.", reg="Hi1=1; Hi2=2, PH1=3, PH2-4, , 5 hemorrhage in different location", kind="custom", coding="none | HI1 | HI2 | PH1 | PH2 | other_location")
S("sah", ST, "Subarachnoid hemorrhage. Registry has 1 for 2 patients and blank for all others; blank is kept as empty (probably no).", reg="SAH", kind="int", coding="1 yes")
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
S("creatinine_umol_l", ST, "Creatinine. Source value 'x' becomes empty. Values above 300 are kept and flagged.", reg="kreatynine", kind="float", unit="umol/L", unit_status="assumed")
S("apoe_genotype", ST, "ApoE genotype, alleles sorted (registry writes e4/e3 and e3/e2). One-hot and allele-count columns are dropped.", reg="ApoE", kind="custom", coding="e2/e3 | e3/e3 | e3/e4")
S("mrs_discharge", ST, "Modified Rankin Scale at discharge.", reg="mRS discharge", kind="int", coding="0-6")
S("mrs_3m", ST, "Modified Rankin Scale 3 months after stroke.", reg="mRS 3month after stroke", kind="int", coding="0-6")
S("moca_baseline", ST, "MoCA at baseline.", CSV, "", "moca_total_baseline", coding="0-30")
S("moca_3m", ST, "MoCA 3 months after stroke.", CSV, "", "moca_total_3m", coding="0-30")
S("rbans_total_3m", ST, "RBANS total score 3 months after stroke.", CSV, "", "rbans_total_3m")
S("death", ST, "Death. FALSE for all 102 patients in the CSV; follow-up window not stated.", CSV, "", "death", coding="1 yes, 0 no")
S("qc_flags", B, "Semicolon-separated data-quality codes; meaning of each code is in clinical_master_notes.md.")
OUT_COLUMNS = [s["column"] for s in SPECS]


def decode(series: pd.Series, mapping: dict, name: str) -> pd.Series:
    out = series.map(mapping)
    assert out.notna().sum() == series.notna().sum(), f"unmapped codes in {name}"
    return out


def load_registry() -> pd.DataFrame:
    reg = pd.read_excel(REGISTRY_XLSX)
    # Headers carry stray spaces; the stroke-location header also has a double space.
    reg.columns = [" ".join(str(c).split()) for c in reg.columns]
    assert reg[RKEY].is_unique
    return reg


def load_store() -> pd.DataFrame:
    store = pd.read_parquet(STORE_SUBJECTS, columns=["subject_id", "condition", "gender"])
    store["subject_id"] = store["subject_id"].astype(str)
    assert store["subject_id"].is_unique
    return store.set_index("subject_id")


def build_stroke(store: pd.DataFrame) -> pd.DataFrame:
    cl = pd.read_csv(CLINICAL_CSV, encoding="utf-8-sig")
    pt = pd.read_excel(SHEETS_XLSX, sheet_name="Stroke Patients").dropna(subset=["Sample No"])
    assert pt["Unnamed: 3"].isna().all()
    pt["Sample No"] = pt["Sample No"].astype(int)
    assert set(cl["participant_id"]) == set(pt["Sample No"]) and cl["participant_id"].is_unique
    pt = pt.set_index("Sample No").loc[cl["participant_id"]].reset_index(drop=True)
    reg = load_registry()
    assert set(cl["participant_id"]) <= set(reg[RKEY])
    r = reg.set_index(RKEY).loc[cl["participant_id"]].reset_index(drop=True)

    # The three patient sources must describe the same people.
    assert (r["age"].to_numpy() == cl["age_years"].to_numpy()).all()
    assert (pt["Age"].to_numpy() == cl["age_years"].to_numpy()).all()
    assert (r["gender 1=females"].map({1: "female", 0: "male"}).to_numpy() == cl["gender"].to_numpy()).all()
    assert (pt["Sex"].str.lower().to_numpy() == cl["gender"].to_numpy()).all()
    ivt, mt = r["IVT: no - 0; yes - 1"], r["mechanical thrombectomy: 0 - no; 1 - yes"]
    assert (r["causative tratment"] == (ivt.astype(bool) | mt.astype(bool)).astype(int)).all()

    ids = cl["participant_id"].astype(str)
    key = "stroke_" + ids
    store_pat = store[store["condition"] == "Disease"]
    has = ids.isin(store_pat.index)
    s = pd.DataFrame({
        "participant_key": key, "cohort": "stroke",
        "spectral_subject_id": ids.where(has), "has_spectra": has.astype(int),
        "has_clinical_record": 1,
        "study_code": pt["Identification code of each participant"],
        "redcap_record_id": pt["REDCap Record ID"],
        "age_years": cl["age_years"], "sex": cl["gender"], "sex_source": "clinical_record",
        "sex_spectral_folder": ids.map(store_pat["gender"]),
        "diagnosis_text": cl["diagnosis_text"],
        "aspects": pd.to_numeric(cl["aspects_score"]).astype("Int64"),
        "moca_baseline": pd.to_numeric(cl["moca_total_baseline"]).astype("Int64"),
        "moca_3m": pd.to_numeric(cl["moca_total_3m"]).astype("Int64"),
        "rbans_total_3m": pd.to_numeric(cl["rbans_total_3m"]).astype("Int64"),
        "death": cl["death"].astype(int),
    })

    for spec in SPECS:  # registry columns copied without recoding
        if not spec["reg"] or spec["kind"] == "custom":
            continue
        src = r[spec["reg"]]
        if spec["kind"] == "text":
            s[spec["column"]] = src.astype(str)
            assert not s[spec["column"]].str.endswith(".0").any()
            continue
        num = pd.to_numeric(src, errors="coerce")
        if spec["column"] != "creatinine_umol_l":  # 'x' is the only text value we expect
            assert num.isna().equals(src.isna()), f"non-numeric text in {spec['reg']}"
        s[spec["column"]] = num.astype("Int64") if spec["kind"] == "int" else num.astype(float)

    s["smoking_status"] = decode(r[SMOKING_HEADER], {0: "never", 1: "current", 2: "former"}, "smoking")
    s["toast"] = r["TOAST"].str.replace("Oclussion", "Occlusion")
    s["care_pathway"] = decode(r["drip and ship=2; mothership=1"], {1: "mothership", 2: "drip_and_ship"}, "care pathway")
    s["hemorrhagic_transformation"] = decode(
        r["hemorrhagic transformation 0 - NO, 1 no symptomatic , 2 symptomatic"],
        {0: "none", 1: "asymptomatic", 2: "symptomatic"}, "hemorrhagic transformation")
    s["hemorrhage_type"] = decode(
        r["Hi1=1; Hi2=2, PH1=3, PH2-4, , 5 hemorrhage in different location"],
        {0: "none", 1: "HI1", 2: "HI2", 3: "PH1", 4: "PH2", 5: "other_location"}, "hemorrhage type")

    # Location: CSV where specified, else registry; disagreements are flagged.
    assert set(cl["diagnosis_category"]) <= set(LOC_CSV) | {"ischemic_stroke_unspecified"}
    loc_csv = cl["diagnosis_category"].map(LOC_CSV)
    loc_reg = decode(r[LOC_HEADER], LOC_REG, "location")
    s["stroke_location"] = loc_csv.fillna(loc_reg)
    s["stroke_location_source"] = np.where(loc_csv.notna(), "clinical_csv", "registry")
    s["stroke_location_registry"] = loc_reg
    FLAGS["LOCATION_CSV_REGISTRY_CONFLICT"] |= set(key[loc_csv.notna() & (loc_csv != loc_reg)])
    FLAGS["LOCATION_FILLED_FROM_REGISTRY"] |= set(key[loc_csv.isna()])

    for word, expected in TOAST_TEXT.items():
        bad = cl["diagnosis_text"].str.contains(word) & (s["toast"] != expected)
        FLAGS["TOAST_CONFLICTS_WITH_DIAGNOSIS_TEXT"] |= set(key[bad])

    geno = r["ApoE"].map(lambda g: "/".join(sorted(g.split("/"))))
    assert (r[APOE_ONEHOT].sum(axis=1) == 1).all()
    onehot = r[APOE_ONEHOT].idxmax(axis=1).str.split("=").str[0]
    counts_ok = np.ones(len(r), dtype=bool)
    for allele in ("e2", "e3", "e4"):
        counts_ok &= geno.str.count(allele).to_numpy() == r[allele].to_numpy()
    s["apoe_genotype"] = geno
    FLAGS["APOE_FIELDS_DISAGREE"] |= set(key[(onehot != geno) | ~counts_ok])

    creat = pd.to_numeric(r["kreatynine"], errors="coerce")
    FLAGS["CREATININE_NON_NUMERIC"] |= set(key[creat.isna() & r["kreatynine"].notna()])
    FLAGS["CREATININE_OUTLIER"] |= set(key[creat > CREATININE_OUTLIER])

    # 21 patients had no reperfusion therapy, so an empty route is expected for them.
    assert s.loc[mt == 1, "care_pathway"].notna().all()
    assert s.loc[(ivt == 0) & (mt == 0), "care_pathway"].isna().all()
    FLAGS["CARE_PATHWAY_MISSING_AFTER_IVT_ONLY"] |= set(key[(ivt == 1) & (mt == 0) & s["care_pathway"].isna()])
    FLAGS["STUDY_CODE_IRREGULAR"] |= set(key[s["study_code"] != "KW_K" + ids])
    FLAGS["DNA_NUMBER_DUPLICATED"] |= set(key[r["DNA sample nimber"].astype(str).duplicated(keep=False)])
    return s


def build_control(store: pd.DataFrame) -> pd.DataFrame:
    hdr = pd.read_excel(SHEETS_XLSX, sheet_name="Control Group", header=None, nrows=2)
    got = [str(h).strip() for h in list(hdr.iloc[0, :4]) + list(hdr.iloc[1, 4:])]  # 'Nicotinism ' has a trailing space
    assert got == CT_HEADER, "Control Group layout changed"
    ct = pd.read_excel(SHEETS_XLSX, sheet_name="Control Group", header=None, skiprows=2)
    ct.columns = CT_COLS
    num = ct["sample_no"].str.removeprefix("K").astype(int)
    ids = num.astype(str)
    key = "control_" + ids
    store_ctl = store[store["condition"] == "Control"]
    has = ids.isin(store_ctl.index)
    nic, past = ct["nicotinism"], ct["smoking_past"]
    smoking = pd.Series(np.where(nic == 1, "current", np.where(past == 1, "former", "never")))
    c = pd.DataFrame({
        "participant_key": key, "cohort": "control",
        "spectral_subject_id": ids.where(has), "has_spectra": has.astype(int),
        "has_clinical_record": 1, "study_code": ct["sample_no"],
        "blood_collection_date": pd.to_datetime(ct["blood_date"]).dt.strftime("%Y-%m-%d"),
        "age_years": pd.to_numeric(ct["age"]).astype("Int64"),
        "sex": ct["sex"].str.lower(), "sex_source": "clinical_record",
        "sex_spectral_folder": ids.map(store_ctl["gender"]),
        "smoking_status": smoking.mask(nic.isna() & past.isna()),
        "hyperlipidemia_or_lipid_lowering_drug": ct["hyperlipidemia"].astype("Int64"),
        "other_medication": ct["other_medication"].astype("Int64"),
    })
    for col in ("hypertension", "diabetes", "atrial_fibrillation", "coronary_artery_disease",
                "antiplatelet", "anticoagulant", "statin"):
        c[col] = ct[col].astype("Int64")

    empty = ct[CT_COLS[4:]].isna().all(axis=1)
    FLAGS["CONTROL_EMPTY_RECORD"] |= set(key[empty])
    FLAGS["SMOKING_BOTH_FLAGS_CURRENT_ASSIGNED"] |= set(key[(nic == 1) & (past == 1)])
    FLAGS["CONTROL_RECORD_NO_SPECTRA"] |= set(key[~has])
    # Found by reading file names under data/raw/Control/Female/15 (prefix 11_);
    # raw is not rescanned here. The folder name stays authoritative, so no link.
    FLAGS["CONTROL_ID_11_15_UNRESOLVED"] |= {"control_11", "control_15"}

    extra = sorted(set(store_ctl.index) - set(ids), key=int)  # spectra without a sheet row
    x = pd.DataFrame({
        "participant_key": ["control_" + i for i in extra], "cohort": "control",
        "spectral_subject_id": extra, "has_spectra": 1, "has_clinical_record": 0,
        "sex": store_ctl.loc[extra, "gender"].to_numpy(), "sex_source": "spectral_folder",
        "sex_spectral_folder": store_ctl.loc[extra, "gender"].to_numpy(),
    })
    FLAGS["CONTROL_NO_CLINICAL_RECORD"] |= set(x["participant_key"])
    return pd.concat([c, x], ignore_index=True)


def write_outputs(master: pd.DataFrame) -> None:
    master.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    keep = ["column", "applies_to", "description", "unit", "unit_status", "coding",
            "source_file", "source_sheet", "source_column"]
    pd.DataFrame(SPECS)[keep].to_csv(OUT_DICT, index=False, encoding="utf-8-sig")


def verify(store: pd.DataFrame) -> None:
    """Re-read the written files and compare them with the sources."""
    back = pd.read_csv(OUT_CSV, encoding="utf-8-sig", dtype={"spectral_subject_id": str})
    dic = pd.read_csv(OUT_DICT, encoding="utf-8-sig")
    assert len(back) == N_STROKE + N_CONTROL_SHEET + N_CONTROL_EXTRA, len(back)
    assert back["participant_key"].is_unique
    assert list(back.columns) == list(dic["column"]), "dictionary and CSV columns differ"

    for cohort, cond in (("stroke", "Disease"), ("control", "Control")):
        got = set(back.loc[(back["cohort"] == cohort) & (back["has_spectra"] == 1), "spectral_subject_id"])
        want = set(store.index[store["condition"] == cond])
        assert got == want and len(want) == N_SPECTRA[cohort], f"{cohort} spectra mismatch"

    flagged = lambda code: set(back.loc[back["qc_flags"].fillna("").str.contains(code), "participant_key"])
    sex_conflict = set(back.loc[back["sex_spectral_folder"].notna() & (back["sex"] != back["sex_spectral_folder"]), "participant_key"])
    assert sex_conflict == EXPECTED_SEX_CONFLICTS, sex_conflict
    assert flagged("SEX_FOLDER_CONFLICT") == sex_conflict, "qc_flags misses a sex conflict"
    assert flagged("LOCATION_CSV_REGISTRY_CONFLICT") == EXPECTED_LOCATION_CONFLICTS
    assert flagged("LOCATION_FILLED_FROM_REGISTRY") == {f"stroke_{n}" for n in (446, 606, 833)}

    # Copied registry values must equal the source exactly.
    stroke = back[back["cohort"] == "stroke"]
    reg = load_registry().set_index(RKEY).loc[stroke["participant_key"].str.removeprefix("stroke_").astype(int)]
    for spec in SPECS:
        if spec["reg"] and spec["kind"] in ("int", "float"):
            want = pd.to_numeric(reg[spec["reg"]], errors="coerce").to_numpy(dtype=float)
            np.testing.assert_allclose(stroke[spec["column"]].to_numpy(dtype=float), want,
                                       equal_nan=True, err_msg=spec["column"])


def main() -> None:
    store = load_store()
    master = pd.concat([build_stroke(store), build_control(store)], ignore_index=True)
    master["_n"] = master["participant_key"].str.rsplit("_", n=1).str[1].astype(int)
    master = master.sort_values(["cohort", "_n"], ascending=[False, True]).drop(columns="_n")
    folder_differs = master["sex_spectral_folder"].notna() & (master["sex"] != master["sex_spectral_folder"])
    FLAGS["SEX_FOLDER_CONFLICT"] |= set(master.loc[folder_differs, "participant_key"])
    master["qc_flags"] = [";".join(c for c in sorted(FLAGS) if k in FLAGS[c]) for k in master["participant_key"]]
    assert not set(master.columns) - set(OUT_COLUMNS), set(master.columns) - set(OUT_COLUMNS)
    write_outputs(master[OUT_COLUMNS])
    try:
        verify(store)
    except AssertionError:
        OUT_CSV.unlink(missing_ok=True)
        OUT_DICT.unlink(missing_ok=True)
        raise
    print(f"wrote {OUT_CSV.name}: {len(master)} rows x {len(OUT_COLUMNS)} columns; all checks passed")
    for code in sorted(FLAGS):
        print(f"  {code} ({len(FLAGS[code])}): {sorted(FLAGS[code], key=lambda k: (k.split('_')[0], int(k.split('_')[1])))}")


if __name__ == "__main__":
    main()
