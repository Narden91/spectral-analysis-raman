"""Generate the cohort-characteristics and endpoint-missingness source tables.

Reads only archived analysis outputs and writes two CSVs plus a LaTeX fragment
under the explicitly selected private output directory.  The stroke cohort is defined as the intersection
of the clinical subject table with the participants retained by the spectral
QC gate. Endpoint-specific model cohorts can differ because of missing values.

Sex summaries use the spectral subject table, with observed clinical/spectral
record disagreements reported by ``--check``. Missing or unrecognized values
remain missing rather than being inferred as another category.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.source_provenance import write_source_provenance  # noqa: E402
from src.data_preparation.manifest import hash_file  # noqa: E402

CLINICAL_SUBJECTS = (
    ROOT
    / "Results_intra_disease"
    / "00_run_metadata"
    / "clinical_endpoints"
    / "clinical_subject_table.csv"
)
SUBJECT_SPECTRA = (
    ROOT
    / "Results_control_vs_disease"
    / "04_cross_region_summary"
    / "01_subject_features"
    / "subject_spectra.csv"
)

CONTINUOUS = [
    ("age", "Age (years)"),
    ("moca_baseline", "MoCA, baseline"),
    ("moca_3m", "MoCA, 3 months"),
    ("delta_moca", r"$\Delta$MoCA"),
    ("rbans_total", "RBANS total"),
    ("aspects_score", "ASPECTS"),
]
ENDPOINTS = ["moca_baseline", "moca_3m", "delta_moca", "rbans_total", "aspects_score"]
ENDPOINT_LABELS = {
    "moca_baseline": "Baseline MoCA",
    "moca_3m": "3-month MoCA",
    "delta_moca": r"$\Delta$MoCA",
    "rbans_total": "RBANS total",
    "aspects_score": "ASPECTS",
}


def load_cohort() -> pd.DataFrame:
    """Return the stroke participants that have both spectra and clinical data."""
    clinical = pd.read_csv(CLINICAL_SUBJECTS, dtype={"subject_id": "string"})
    spectra = pd.read_csv(
        SUBJECT_SPECTRA, usecols=["subject_id", "condition", "gender"], dtype={"subject_id": "string"}
    )
    disease = spectra.loc[spectra["condition"] == "disease"]
    merged = clinical.merge(disease[["subject_id", "gender"]], on="subject_id", validate="one_to_one")
    return merged


def build_characteristics(cohort: pd.DataFrame) -> pd.DataFrame:
    """Summarize the stroke cohort: counts for categoricals, IQR for continuous."""
    rows: list[dict[str, object]] = []
    n_total = len(cohort)
    rows.append(
        {
            "characteristic": "Participants",
            "summary": str(n_total),
            "n_observed": n_total,
            "n_missing": 0,
        }
    )
    female = int(cohort["gender"].eq("female").sum())
    male = int(cohort["gender"].eq("male").sum())
    rows.append(
        {
            "characteristic": "Female / male",
            "summary": f"{female} / {male}",
            "n_observed": female + male,
            "n_missing": n_total - female - male,
        }
    )
    for column, label in CONTINUOUS:
        values = cohort[column].dropna()
        rows.append(
            {
                "characteristic": label,
                "summary": (
                    f"{values.median():.1f} "
                    f"({values.quantile(0.25):.1f}--{values.quantile(0.75):.1f})"
                ),
                "n_observed": int(values.size),
                "n_missing": int(n_total - values.size),
            }
        )
    lesion = cohort["lesion_location_group"].value_counts()
    for side in ("left", "right", "posterior", "other"):
        if side not in lesion:
            continue
        rows.append(
            {
                "characteristic": f"Lesion side: {side}",
                "summary": str(int(lesion[side])),
                "n_observed": n_total,
                "n_missing": 0,
            }
        )
    return pd.DataFrame(rows)


def build_missingness(cohort: pd.DataFrame) -> pd.DataFrame:
    """Compare available age, sex and lesion-side records by endpoint missingness.

    These descriptive comparisons do not establish why a value is missing.
    """
    lesion = cohort["lesion_location_group"].astype("string")
    is_left = lesion.eq("left").where(lesion.isin(["left", "right", "posterior", "other"]))
    cohort = cohort.assign(is_left=is_left)
    rows: list[dict[str, object]] = []
    for endpoint in ENDPOINTS:
        observed_mask = cohort[endpoint].notna()
        observed = cohort.loc[observed_mask]
        missing = cohort.loc[~observed_mask]
        if missing.empty:
            continue
        observed_age = observed["age"].dropna()
        missing_age = missing["age"].dropna()
        age_p = (
            stats.mannwhitneyu(observed_age, missing_age).pvalue
            if not observed_age.empty and not missing_age.empty else float("nan")
        )
        observed_sex = observed.loc[observed["gender"].isin(["female", "male"]), "gender"]
        missing_sex = missing.loc[missing["gender"].isin(["female", "male"]), "gender"]
        sex_table = pd.crosstab(observed_mask, cohort["gender"])
        sex_p = (
            stats.fisher_exact(sex_table)[1] if sex_table.shape == (2, 2) else float("nan")
        )
        left_table = pd.crosstab(observed_mask, cohort["is_left"])
        left_p = (
            stats.fisher_exact(left_table)[1]
            if left_table.shape == (2, 2)
            else float("nan")
        )
        rows.append(
            {
                "endpoint": ENDPOINT_LABELS[endpoint],
                "n_observed": int(observed_mask.sum()),
                "n_missing": int((~observed_mask).sum()),
                "age_observed": observed["age"].median(),
                "age_missing": missing["age"].median(),
                "age_p": age_p,
                "female_pct_observed": 100 * observed_sex.eq("female").mean() if not observed_sex.empty else float("nan"),
                "female_pct_missing": 100 * missing_sex.eq("female").mean() if not missing_sex.empty else float("nan"),
                "sex_p": sex_p,
                "left_pct_observed": 100 * observed["is_left"].dropna().mean() if observed["is_left"].notna().any() else float("nan"),
                "left_pct_missing": 100 * missing["is_left"].dropna().mean() if missing["is_left"].notna().any() else float("nan"),
                "left_p": left_p,
            }
        )
    return pd.DataFrame(rows)


def format_p(value: float) -> str:
    """Render a p-value for the table, avoiding false precision."""
    if pd.isna(value):
        return "--"
    return "$<$0.001" if value < 0.001 else f"{value:.3f}"


def write_latex(characteristics: pd.DataFrame, missingness: pd.DataFrame, *, n_subjects: int, output_path: Path) -> None:
    """Emit the two-panel Table 1 as a standalone LaTeX fragment."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{Stroke-cohort characteristics (Panel A; $n={n_subjects}$, median and",
        r"interquartile range) and observed-versus-missing comparison per clinical",
        r"endpoint (Panel B).}",
        r"\label{tab:cohort}",
        r"\small",
        r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}X"
        r">{\centering\arraybackslash}p{0.20\textwidth}"
        r">{\centering\arraybackslash}p{0.16\textwidth}@{}}",
        r"\toprule",
        r"\multicolumn{3}{@{}l}{\emph{Panel A: cohort characteristics}} \\",
        r"\midrule",
        r"Characteristic & Value & $n$ missing \\",
        r"\midrule",
    ]
    for row in characteristics.itertuples():
        lines.append(f"{row.characteristic} & {row.summary} & {row.n_missing} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabularx}",
        "",
        r"\vspace{1em}",
        "",
        r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{0.16\textwidth}"
        r">{\centering\arraybackslash}p{0.07\textwidth}"
        r">{\centering\arraybackslash}p{0.07\textwidth}"
        r">{\centering\arraybackslash}X"
        r">{\centering\arraybackslash}X"
        r">{\centering\arraybackslash}X@{}}",
        r"\toprule",
        r"\multicolumn{6}{@{}l}{\emph{Panel B: observed vs.\ missing cases}} \\",
        r"\midrule",
        r"Endpoint & $n$ obs. & $n$ miss. & Age, obs./miss.\ ($p$) &"
        r" \% female, obs./miss.\ ($p$) & \% left lesion, obs./miss.\ ($p$) \\",
        r"\midrule",
    ]
    for row in missingness.itertuples():
        lines.append(
            f"{row.endpoint} & {row.n_observed} & {row.n_missing} & "
            f"{row.age_observed:.0f}/{row.age_missing:.0f} ({format_p(row.age_p)}) & "
            f"{row.female_pct_observed:.0f}/{row.female_pct_missing:.0f} "
            f"({format_p(row.sex_p)}) & "
            f"{row.left_pct_observed:.0f}/{row.left_pct_missing:.0f} "
            f"({format_p(row.left_p)}) \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table}",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def check_sex_consistency(cohort: pd.DataFrame) -> int:
    """Report participants whose clinical and spectral sex records disagree."""
    disagree = cohort.loc[cohort["sex"] != cohort["gender"]]
    print(f"sex disagreements between clinical and spectral tables: {len(disagree)}")
    return len(disagree)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report clinical/spectral sex disagreements and exit",
    )
    parser.add_argument("--output-dir", type=Path, help="Private Results directory for source tables and LaTeX")
    parser.add_argument("--source-run-metadata", action="append", type=Path, default=[],
                        help="Archived source-run metadata directory; may be repeated")
    args = parser.parse_args()
    if args.output_dir is None and not args.check:
        parser.error("--output-dir is required unless --check is selected")

    cohort = load_cohort()
    if args.check:
        check_sex_consistency(cohort)
        return

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    characteristics = build_characteristics(cohort)
    missingness = build_missingness(cohort)
    characteristics.to_csv(output_dir / "cohort_characteristics.csv", index=False)
    missingness.to_csv(output_dir / "cohort_endpoint_missingness.csv", index=False)
    write_latex(characteristics, missingness, n_subjects=len(cohort), output_path=output_dir / "table_cohort.tex")
    write_source_provenance(
        output_dir, source_files=[CLINICAL_SUBJECTS, SUBJECT_SPECTRA], source_metadata_dirs=args.source_run_metadata,
        rendering_settings={"renderer_sha256": hash_file(Path(__file__)), "continuous_columns": CONTINUOUS,
                            "endpoints": ENDPOINTS, "sex_summary_source": "spectral_subject_table",
                            "pandas_version": pd.__version__},
    )
    print(f"stroke participants: {len(cohort)}")
    print(characteristics.to_string(index=False))
    print()
    print(missingness.to_string(index=False))


if __name__ == "__main__":
    main()
