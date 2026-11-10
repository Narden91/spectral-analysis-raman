# ML Paper Authoring Directory (`ml_paper/`)

This directory contains the initial manuscript scaffolding, source-linked cohort and experimental descriptions extracted from [spectra_manuscript/main.tex](../spectra_manuscript/main.tex), and reconciled methodological specifications from [docs/research_design.md](../docs/research_design.md) and [data/clinical_master_notes.md](../data/clinical_master_notes.md).

## Contents

1. [cohort_and_acquisition_description.md](cohort_and_acquisition_description.md):
   - Extracted wet-lab, instrument, preprocessing, and cohort parameters from [spectra_manuscript/main.tex](../spectra_manuscript/main.tex).
   - Explicit reconciliation of discrepancies between the earlier manuscript text and the verified standalone clinical master ([data/clinical_master.csv](../data/clinical_master.csv)).
2. [main.tex](main.tex):
   - Initial LaTeX manuscript draft structured for biomedical data science / computational biology venues (*IEEE Journal of Biomedical and Health Informatics* or *PLOS Computational Biology*).
   - Includes the cohort description, frozen preprocessing and 21-band window tables, paired representation equations, participant-level evaluation, and explicitly pending result tables. The manuscript is a retrospective development draft, not a completed prediction study.
3. [references.bib](references.bib):
   - Consolidated BibTeX database combining the experimental/clinical citations from `spectra_manuscript` with the primary methodological and lipid-prediction citations from `docs/research_design.md`.

## Mapping `manuscript_ml/` Outputs to Manuscript Sections

| Manuscript Element | Source Script in `manuscript_ml/` | Output Artifact |
|---|---|---|
| Section 2.1 (Cohort Eligibility & Linkage) | [audit_readiness.py](../manuscript_ml/audit_readiness.py), [run_primary_evaluation.py](../manuscript_ml/run_primary_evaluation.py) | `readiness_manifest.json`, `eligibility_and_split_manifest.json` |
| Section 3.1 / scenario table (Synthetic Calibration & Failure Modes) | [run_simulation_calibration.py](../manuscript_ml/run_simulation_calibration.py) | `simulation_scenarios_summary.csv`, `simulation_calibration_report.csv` |
| Section 3.2 / prediction table (Primary Held-Out LDL-C Comparison) | [run_primary_evaluation.py](../manuscript_ml/run_primary_evaluation.py) | `comparator_errors_and_differences.csv`, `fold_selected_settings.csv`, `eligibility_and_split_manifest.json` (bootstrap endpoints and failure status) |
| Section 3.3 (Repeatability $\mathrm{ICC}(A,1)$, Discriminability & Subspace Stability) | [run_primary_evaluation.py](../manuscript_ml/run_primary_evaluation.py) | `baseline_repeatability_report.csv`, `subspace_principal_angles.csv` |
| Section 3.4 (Sensitivity Analyses, Secondary Lipid/Platelet Endpoints & Continuous Anchor Associations) | [run_sensitivity_and_secondary.py](../manuscript_ml/run_sensitivity_and_secondary.py) | `covariate_and_subset_sensitivity.csv`, `outer_partition_seed_sensitivity.csv`, `secondary_endpoints_summary.csv`, `anchor_associations_continuous.csv` |
| Section 4 (Discussion and Limitations) | [audit_readiness.py](../manuscript_ml/audit_readiness.py) | `limitations_table.csv` |


## Computational Notes

- Code lives in the [src/analysis/paired_lipid/](../src/analysis/paired_lipid/) package; settings live in [config/manuscript_ml.yaml](../config/manuscript_ml.yaml). Every stage writes `run_provenance_<stage>.json` (config and hash, git commit, package versions, data identity).
- The batched bootstrap engine matches a fully sequential refit to <= 1e-8 on CPU and on GPU with exact fp64 eigensolvers (test `test_batched_bootstrap_draws_match_sequential_refits`). Two engine bugs were fixed on 8 November 2026: the outer refit regularized with tr(W) instead of tr(T), and the default mixed-precision GPU eigensolver changed every bootstrap draw (up to 3e-2). Earlier bootstrap outputs are invalid.
- Calibration coverage is measured against the expected held-out risk difference, estimated by scoring each replicate's fitted outer-fold models on one independent synthetic test cohort (n = 5,000). Earlier calibration reports used the mean of the point estimates as the target (bias identically 0) and are invalid. Only `--seed-set validation` gives a confirmatory gate result.
- Secondary endpoints use the primary population and outer folds; a missing endpoint value gets weight 0 in training and test.
- Experiment datasets are memoized per argument set by `resolve_experiment_dataset` in [manuscript_ml/__init__.py](../manuscript_ml/__init__.py). The cohort path requires `--approve-cohort-evaluation` on every call. Use `clear_dataset_cache()` to force a reload.
- The planned core-scenario calibration campaign has not been completed. The former ~21 h estimate predates the 9 November optimization and is not a current budget. The recorded real-data dummy-outcome bootstrap took 38.5 s for 2,000 draws on GPU with batch 2048 and 3.79 GiB peak pool use; that is not the runtime of a complete calibration replicate or the campaign. Re-time the intended configuration before seeking approval for a long run.
- The current calibration routine checks one signal-strength setting, separately from scenario prediction summaries. Its target standard error excludes uncertainty from the shared finite test cohort. Validation-seed status and a passing provisional gate do not establish adequate calibration across the design's core scenarios. See [the design limitation](../docs/research_design.md#participant-divisions-and-uncertainty).
- The 21-band table reproduces the inclusive software-registry windows. Their earlier cohort use does not establish independent feature provenance. New data-derived bands would be a separate exploratory change.
- The 92/88/4/180 counts and numerical checks are carried forward from [the 9 November evidence record](../AGENT_LEARNINGS.md#2026-11-09-verified-facts-about-the-manuscript-ml-pipeline-do-not-re-test); no lipid outcome analysis was run for the 10 November documentation revision.

## Targeted citation audit, 10 November 2026

Reconciled entries against the existing earlier-paper bibliography and the primary records below. Citation keys remain unchanged. This is a targeted metadata audit, not a systematic literature review or verification of every protocol detail.

| Key | Verified correction and source |
|---|---|
| `silveira2017` | Correct title and author list: [PubMed record](https://pubmed.ncbi.nlm.nih.gov/28271376/). |
| `bonizzi2025` | Correct title, first author Arianna Bonizzi, and volume 278: [author-deposited article](https://air.unimi.it/retrieve/ca92e45a-8f16-488e-a43d-a9c20e5c2c40/Exploring-breast-cancer-related-biochemical-changes-in-c_2025_Biosensors-and.pdf). |
| `deccwiek2025` | Restore the clinical genetics study in Scientific Reports, 15:31136: [publisher article](https://www.nature.com/articles/s41598-025-14789-0.pdf). It supplies parent-cohort context, not prior Raman findings. |
| `liu2025` | Restore the review in Biosensors and Bioelectronics, 270:116970: [publisher record](https://doi.org/10.1016/j.bios.2024.116970). |
| `koster2021` | Restore the title, authors and Nanoscale publication: [publisher record](https://doi.org/10.1039/D1NR03334D). |
| `lucien2023` | Restore MIBlood-EV in Journal of Extracellular Vesicles, 12:12385: [publisher record](https://doi.org/10.1002/jev2.12385). A reporting guideline does not establish retrospective protocol compliance. |
| `bracht2024` | Restore the SEC-column study in Extracellular Vesicles and Circulating Nucleic Acids, 5:497-508: [publisher article](https://www.oaepublish.com/articles/evcna.2024.34). |
| `gualerzi2019` | Restore author names from the earlier-paper bibliography; article identity checked against [publisher text](https://doi.org/10.1080/20013078.2019.1568780). |
| `koo2017erratum` | Correct the title to Erratum and remove attributed authors: [PubMed record lists no authors](https://pubmed.ncbi.nlm.nih.gov/29276468/). |
