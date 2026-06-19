import logging
import re
import shutil
from pathlib import Path

import pandas as pd

from src.io.file_crawler import build_raw_file_index
from src.metadata.parser import clean_subject_id
from src import setup_logging, config, _output_dir

logger = logging.getLogger(__name__)

_H = "\u2500"       # ─
_V = "\u2502"       # │
_TL = "\u250C"      # ┌
_TR = "\u2510"      # ┐
_BL = "\u2514"      # └
_BR = "\u2518"      # ┘
_VR = "\u251C"      # ├
_VL = "\u2524"      # ┤
_HD = "\u252C"      # ┬
_HU = "\u2534"      # ┴
_X = "\u253C"       # ┼


def _clean_subject_id(raw_id: str) -> str:
    """Return the primary part of a compound subject ID (before ';' or ',').

    Delegates quote/letter stripping to ``clean_subject_id`` and then
    splits on ``;`` or ``,`` to take only the first ID component.
    """
    return clean_subject_id(str(raw_id).replace(";", ",").split(",")[0])


def _target_dir(out_root: Path, cohort: str, sex: str, subject_id: str, slide_num: str) -> Path:
    """Build the sample output directory path."""
    cohort_folder = "Control" if cohort else "Disease"
    sex_folder = "Female" if sex.lower() == "female" else "Male"
    return out_root / cohort_folder / sex_folder / subject_id / f"sample_{slide_num}"


def _output_filename(
    subject_id: str,
    slide_num: str,
    spectral_range: float,
    meas_num: int,
) -> str:
    """Canonical mode-free output filename: {Subject}_{Slide}_{Range}_range_meas{M}.txt."""
    return f"{subject_id}_{slide_num}_{int(spectral_range)}_range_meas{meas_num}.txt"


def reorganize_data() -> None:
    """Reorganize raw Renishaw data from data_cleaned/ into data/.

    For each measurement file in data_cleaned/:
    - Applies Redone override (Redone measurements replace originals
      when both exist for the same subject/slide).
    - Copies each file individually -- no merging or averaging.
    - Outputs to: data/{Cohort}/{Sex}/{Subject}/sample_{N}/{filename}

    Also writes summary CSVs to the indexing output directory.
    """
    logger.info("Starting dataset reorganization ...")

    clean_dir = Path("data_cleaned")
    if not clean_dir.exists():
        logger.error("data_cleaned/ directory missing. Run clean_raw_data.py first.")
        return

    df_index = build_raw_file_index(data_dir=clean_dir)
    if df_index.empty:
        logger.error("No raw measurement files found. Aborting.")
        return

    out_root = Path(config["data"]["root_dir"])
    out_csv_dir = out_root / "reports"
    out_csv_dir.mkdir(parents=True, exist_ok=True)

    report_rows: list[dict] = []
    copied = 0
    skipped = 0
    collisions = 0

    for _, row in df_index.iterrows():
        source = Path(row["filepath"])
        if not source.exists():
            logger.error("Source file missing: %s", source)
            skipped += 1
            continue

        subject_id = _clean_subject_id(row["subject_id"])
        slide_num = str(row["slide_number"])
        spectral_range = float(row["spectral_range_cm-1"])
        meas_num = int(row["measurement_number"])
        is_control = bool(row["is_control"])
        gender = str(row.get("gender", "unknown"))

        target_dir = _target_dir(out_root, is_control, gender, subject_id, slide_num)
        target_dir.mkdir(parents=True, exist_ok=True)

        filename = _output_filename(subject_id, slide_num, spectral_range, meas_num)
        target_path = target_dir / filename

        # Collision guard: never silently overwrite.
        if target_path.exists():
            logger.error(
                "Output collision (skipping, NOT overwriting): %s already exists; "
                "source=%s",
                target_path,
                source,
            )
            collisions += 1
            skipped += 1
            continue

        shutil.copy2(source, target_path)
        copied += 1
        logger.debug("Copied -> %s", target_path)

        report_rows.append(
            {
                "subject_id": subject_id,
                "slide_number": slide_num,
                "spectral_range_cm-1": int(spectral_range),
                "measurement_number": meas_num,
                "gender": gender,
                "Condition": "Control" if is_control else "Disease",
                "is_redone": row.get("is_redone", False),
                "date": row.get("date", ""),
                "source_file": str(source),
                "output_file": str(target_path),
            }
        )

    df_report = pd.DataFrame(report_rows)
    if not df_report.empty:
        df_report = df_report.sort_values(
            ["Condition", "subject_id", "slide_number", "spectral_range_cm-1", "measurement_number"]
        )
        df_report.to_csv(out_csv_dir / "dataset_report.csv", index=False)
        _write_subject_summary(df_report, out_csv_dir / "subject_summary.csv")

    invalid_samples = _validate_output(out_root)
    log_text1 = _log_final_summary(
        out_root, copied, skipped, out_csv_dir, invalid_samples, collisions
    )
    log_text2 = ""

    clinical_path = Path(config["data"].get("clinical_csv", ""))
    if not clinical_path.is_file():
        for candidate in [out_root / "clinical_data.csv", Path("data_old/clinical_data.csv")]:
            if candidate.is_file():
                clinical_path = candidate
                break
    if clinical_path.is_file() and not df_report.empty:
        log_text2 = _cross_check_clinical(df_report, clinical_path)
    else:
        logger.info("No clinical CSV found — skipping cross-check.")

    with open(out_root / "reorganization_log.txt", "w", encoding="utf-8") as f:
        f.write(log_text1)
        if log_text2:
            f.write("\n\n" + log_text2)


def _write_subject_summary(df_report: pd.DataFrame, out_path: Path) -> None:
    """Write a per-subject summary CSV with sample and measurement counts."""
    samples = (
        df_report.groupby(["subject_id", "Condition", "gender"])["slide_number"]
        .nunique()
        .reset_index(name="num_samples")
    )

    meas_per_slide = (
        df_report.groupby(["subject_id", "slide_number"])["measurement_number"]
        .nunique()
        .reset_index(name="measurements")
    )
    meas_agg = (
        meas_per_slide.groupby("subject_id")["measurements"]
        .agg(["min", "max"])
        .reset_index()
    )
    meas_agg["measurements_per_sample"] = meas_agg["min"].astype(int).astype(str)
    varying = meas_agg["min"] != meas_agg["max"]
    meas_agg.loc[varying, "measurements_per_sample"] = (
        meas_agg.loc[varying, "min"].astype(int).astype(str)
        + "-"
        + meas_agg.loc[varying, "max"].astype(int).astype(str)
    )

    summary = samples.merge(meas_agg[["subject_id", "measurements_per_sample"]], on="subject_id")
    summary = summary.rename(columns={"Condition": "condition"})
    summary = summary[["subject_id", "condition", "gender", "num_samples", "measurements_per_sample"]]
    summary = summary.sort_values(["condition", "subject_id"]).reset_index(drop=True)
    summary.to_csv(out_path, index=False)


def _validate_output(out_root: Path) -> list[str]:
    """Return sample folders with incomplete file sets.

    A complete sample has 6 files: 2 spectral ranges x 3 measurements.
    """
    issues = []
    for sample_dir in out_root.rglob("sample_*"):
        if not sample_dir.is_dir():
            continue
        txt_files = list(sample_dir.glob("*.txt"))
        if not txt_files:
            issues.append(f"{sample_dir} (0 files)")
            continue
        if len(txt_files) != 6:
            issues.append(f"{sample_dir} ({len(txt_files)} files)")
    return issues



def _log_final_summary(
    out_root: Path,
    copied: int,
    skipped: int,
    csv_dir: Path,
    invalid_samples: list[str],
    collisions: int = 0,
) -> str:
    """Print a clean, box-drawn summary of the reorganization results and return the text."""
    counts: dict[tuple[str, str], int] = {}
    empty_folders: list[str] = []

    for cohort_dir in sorted(out_root.iterdir()):
        if not cohort_dir.is_dir():
            continue
        cohort = cohort_dir.name
        for sex_dir in sorted(cohort_dir.iterdir()):
            if not sex_dir.is_dir():
                continue
            sex = sex_dir.name
            for subject_dir in sorted(sex_dir.iterdir()):
                if not subject_dir.is_dir():
                    continue
                if list(subject_dir.rglob("*.txt")):
                    counts[(cohort, sex)] = counts.get((cohort, sex), 0) + 1
                else:
                    empty_folders.append(
                        f"{cohort}/{sex}/{subject_dir.name}"
                    )

    total = sum(counts.values())
    w = 52  # inner width

    lines: list[str] = []

    lines.append(f"{_TL}{_H * w}{_TR}")
    lines.append(f"{_V}{'REORGANIZATION COMPLETE':^{w}}{_V}")
    lines.append(f"{_VR}{_H * w}{_VL}")

    lines.append(f"{_V}  Files copied  : {copied:<{w - 19}}{_V}")
    lines.append(f"{_V}  Files skipped : {skipped:<{w - 19}}{_V}")
    lines.append(f"{_V}  Collisions    : {collisions:<{w - 19}}{_V}")
    lines.append(f"{_VR}{_H * w}{_VL}")

    lines.append(f"{_V}{'SUBJECT COUNTS':^{w}}{_V}")
    lines.append(f"{_VR}{_H * 20}{_HD}{_H * 10}{_HD}{_H * 10}{_HD}{_H * (w - 43)}{_VL}")
    lines.append(
        f"{_V}  {'Group':<18}{_V}{'Female':^10}{_V}{'Male':^10}{_V}{'Total':^{w - 43}}{_V}"
    )
    lines.append(f"{_VR}{_H * 20}{_X}{_H * 10}{_X}{_H * 10}{_X}{_H * (w - 43)}{_VL}")

    for cohort in ("Disease", "Control"):
        f = counts.get((cohort, "Female"), 0)
        m = counts.get((cohort, "Male"), 0)
        t = f + m
        lines.append(
            f"{_V}  {cohort:<18}{_V}{f:^10}{_V}{m:^10}{_V}{t:^{w - 43}}{_V}"
        )

    total_f = sum(v for (c, s), v in counts.items() if s == "Female")
    total_m = sum(v for (c, s), v in counts.items() if s == "Male")
    lines.append(f"{_VR}{_H * 20}{_X}{_H * 10}{_X}{_H * 10}{_X}{_H * (w - 43)}{_VL}")
    lines.append(
        f"{_V}  {'TOTAL':<18}{_V}{total_f:^10}{_V}{total_m:^10}{_V}{total:^{w - 43}}{_V}"
    )
    lines.append(f"{_VR}{_H * 20}{_HU}{_H * 10}{_HU}{_H * 10}{_HU}{_H * (w - 43)}{_VL}")

    r1 = str(csv_dir / "dataset_report.csv")
    r2 = str(csv_dir / "subject_summary.csv")
    lines.append(f"{_V}  Reports:{'':<{w - 10}}{_V}")
    lines.append(f"{_V}    {r1:<{w - 4}.{w - 4}}{_V}")
    lines.append(f"{_V}    {r2:<{w - 4}.{w - 4}}{_V}")

    if invalid_samples:
        lines.append(f"{_VR}{_H * w}{_VL}")
        hdr = f"!! INVALID SAMPLE FOLDERS ({len(invalid_samples)})"
        lines.append(f"{_V}  {hdr:<{w - 2}}{_V}")
        for issue in invalid_samples:
            lines.append(f"{_V}    {issue:<{w - 4}.{w - 4}}{_V}")

    if empty_folders:
        lines.append(f"{_VR}{_H * w}{_VL}")
        hdr = f"!! EMPTY SUBJECT FOLDERS ({len(empty_folders)})"
        lines.append(f"{_V}  {hdr:<{w - 2}}{_V}")
        for folder in empty_folders:
            lines.append(f"{_V}    {folder:<{w - 4}.{w - 4}}{_V}")

    lines.append(f"{_BL}{_H * w}{_BR}")

    log_str = "\n".join(lines)
    logger.info("\n%s", log_str)
    return log_str


def _subject_sort_key(subject_id: str) -> tuple[int, int | str]:
    return (0, int(subject_id)) if subject_id.isdigit() else (1, subject_id)


def _cross_check_clinical(df_report: pd.DataFrame, clinical_path: Path) -> str:
    """Compare reorganized subjects against a clinical CSV and log detailed mismatches. Returns the text log."""
    clinical = pd.read_csv(clinical_path, encoding="utf-8-sig")
    col_id = clinical.columns[0]
    col_gender = [c for c in clinical.columns if "Gender" in c or "Female" in c][0]
    clinical = clinical.rename(columns={col_id: "subject_id", col_gender: "gender"})
    clinical["subject_id"] = clinical["subject_id"].astype(str).str.strip()
    clinical["gender"] = clinical["gender"].str.strip().str.upper()

    # Create lookups from the reorganized report
    data_disease_ids = set(df_report[df_report["Condition"] == "Disease"]["subject_id"].unique())
    data_control_ids = set(df_report[df_report["Condition"] == "Control"]["subject_id"].unique())
    all_data_ids = data_disease_ids | data_control_ids
    clin_ids = set(clinical["subject_id"])

    missing_from_data = sorted(clin_ids - all_data_ids, key=_subject_sort_key)
    missing_from_csv = sorted(all_data_ids - clin_ids, key=_subject_sort_key)
    in_control_but_csv = sorted(data_control_ids & clin_ids, key=_subject_sort_key)
    common = sorted(all_data_ids & clin_ids, key=_subject_sort_key)
    data_gender = df_report.drop_duplicates("subject_id").set_index("subject_id")["gender"].str.upper()
    clin_gender = clinical.set_index("subject_id")["gender"]

    gender_mismatches = [
        (sid, clin_gender[sid], data_gender[sid])
        for sid in common
        if clin_gender[sid] != data_gender[sid]
    ]

    w = 64
    lines: list[str] = []

    lines.append(f"{_TL}{_H * w}{_TR}")
    lines.append(f"{_V}{'CLINICAL CROSS-CHECK (DETAILED)':^{w}}{_V}")
    lines.append(f"{_VR}{_H * w}{_VL}")
    lines.append(f"{_V}  Clinical CSV : {str(clinical_path):<{w - 18}.{w - 18}}{_V}")
    lines.append(f"{_V}  Total Clinical IDs : {len(clin_ids):<{w - 23}}{_V}")
    lines.append(f"{_V}  Total Data IDs     : {len(all_data_ids):<{w - 23}}{_V}")

    if missing_from_data:
        lines.append(f"{_VR}{_H * w}{_VL}")
        dgt = len(str(len(missing_from_data)))
        lines.append(f"{_V}  [MISSING] IN CSV BUT NOT IN DATA ({len(missing_from_data)}){'':<{w - 38 - dgt}}{_V}")
        for sid in missing_from_data:
            g = clin_gender.get(sid, "?")
            lines.append(f"{_V}    {sid:<8}{g:<10}{'':<{w - 22}}{_V}")

    if missing_from_csv:
        lines.append(f"{_VR}{_H * w}{_VL}")
        dgt = len(str(len(missing_from_csv)))
        lines.append(f"{_V}  [MISSING] IN DATA BUT NOT IN CSV ({len(missing_from_csv)}){'':<{w - 38 - dgt}}{_V}")
        for sid in missing_from_csv:
            g = data_gender.get(sid, "?")
            lines.append(f"{_V}    {sid:<8}{g:<10}{'':<{w - 22}}{_V}")

    if in_control_but_csv:
        lines.append(f"{_VR}{_H * w}{_VL}")
        dgt = len(str(len(in_control_but_csv)))
        lines.append(f"{_V}  [GROUP MISMATCH] ID IS CONTROL BUT FOUND IN CSV ({len(in_control_but_csv)}){'':<{w - 53 - dgt}}{_V}")
        for sid in in_control_but_csv:
            lines.append(f"{_V}    {sid:<8}{'':<{w - 12}}{_V}")

    if gender_mismatches:
        lines.append(f"{_VR}{_H * w}{_VL}")
        dgt = len(str(len(gender_mismatches)))
        lines.append(f"{_V}  [GENDER MISMATCH] DISAGREEMENT BETWEEN CSV & DATA ({len(gender_mismatches)}){'':<{w - 55 - dgt}}{_V}")
        lines.append(f"{_V}    {'ID':<8}{'CSV':<10}{'Data':<10}{'':<{w - 32}}{_V}")
        for sid, g_csv, g_data in gender_mismatches:
            lines.append(f"{_V}    {sid:<8}{g_csv:<10}{g_data:<10}{'':<{w - 32}}{_V}")

    if not missing_from_data and not missing_from_csv and not in_control_but_csv and not gender_mismatches:
        lines.append(f"{_VR}{_H * w}{_VL}")
        lines.append(f"{_V}  All subjects perfectly match in Existence, Group, and Sex.{'':<{w - 60}}{_V}")

    lines.append(f"{_BL}{_H * w}{_BR}")

    log_str = "\n".join(lines)
    logger.info("\n%s", log_str)
    return log_str


if __name__ == "__main__":
    setup_logging()
    reorganize_data()
