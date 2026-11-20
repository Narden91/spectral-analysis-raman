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
| Section 3.1 / scenario & calibration tables (Synthetic Calibration, Stress Regimes & MDE Curve) | [run_simulation_calibration.py](../manuscript_ml/run_simulation_calibration.py) | `simulation_scenarios_summary.csv`, `simulation_calibration_report.csv`, `simulation_mde_curve.csv` |
| Section 3.2 / prediction table (Primary Held-Out LDL-C Comparison) | [run_primary_evaluation.py](../manuscript_ml/run_primary_evaluation.py) | `comparator_errors_and_differences.csv`, `fold_selected_settings.csv`, `eligibility_and_split_manifest.json`, `bootstrap_draws.npz` |
| Section 3.3 (Repeatability $\mathrm{ICC}(A,1)$, Discriminability, Subspace Stability & Spectral Weights) | [run_primary_evaluation.py](../manuscript_ml/run_primary_evaluation.py) | `baseline_repeatability_report.csv`, `subspace_principal_angles.csv`, `fold_spectral_weights.csv` |
| Section 3.4 (Sensitivity Analyses, Secondary Lipid/Platelet Endpoints & Continuous Anchor Associations) | [run_sensitivity_and_secondary.py](../manuscript_ml/run_sensitivity_and_secondary.py) | `covariate_and_subset_sensitivity.csv`, `outer_partition_seed_sensitivity.csv`, `secondary_endpoints_summary.csv`, `anchor_associations_continuous.csv` |
| Section 4 (Discussion and Limitations) | [audit_readiness.py](../manuscript_ml/audit_readiness.py) | `limitations_table.csv` |
| Deterministic LaTeX Tables & Diagnostic Figures | [build_manuscript_artifacts.py](../manuscript_ml/build_manuscript_artifacts.py) | `tables/*.tex`, `figures/*.png`, `artifact_build_manifest.json` |


## Computational Notes

- Code lives in the [src/analysis/paired_lipid/](../src/analysis/paired_lipid/) package; settings live in [config/manuscript_ml.yaml](../config/manuscript_ml.yaml). Every stage writes `run_provenance_<stage>.json` (config and hash, git commit and diff hash when dirty, package versions, locked 21-band registry, frozen spectral grid bounds, and data identity).
- The batched bootstrap engine matches a fully sequential refit to <= 1e-8 on CPU and on GPU with exact fp64 eigensolvers (including per-draw weighted out-of-fold loss-difference SDs used by `studentized_t`). Two engine bugs were fixed on 8 November 2026: the outer refit regularized with tr(W) instead of tr(T), and the default mixed-precision GPU eigensolver changed every bootstrap draw (up to 3e-2). Earlier bootstrap outputs are invalid.
- Calibration coverage is measured against the expected held-out risk difference, estimated by scoring each replicate's fitted outer-fold models on one independent synthetic test cohort (`n_test_participants = 25,000` in `config/manuscript_ml.yaml`). Earlier calibration reports used the mean of the point estimates as the target (bias identically 0) and are invalid. Only `--seed-set validation` gives a confirmatory gate result.
- Secondary endpoints use the primary population and outer folds; a missing endpoint value gets weight 0 in training and test. Continuous anchor summaries use 2,000 participant-level resamples, per-anchor covariate preprocessing, explicit `target_role` tags, and per-fold partial-correlation summaries.
- Experiment datasets are memoized per argument set by `resolve_experiment_dataset` in [manuscript_ml/__init__.py](../manuscript_ml/__init__.py), with read-only NumPy arrays and active-versus-disk store manifest-hash verification. The cohort path requires `--approve-cohort-evaluation` on every call. Use `clear_dataset_cache()` to force a reload.
- The planned core-scenario validation campaign has not been completed. On 20 November 2026 (RTX 5080 Laptop GPU, exact fp64), the updated CheFSI GPU engine ran 2,000 real-spectra dummy-outcome bootstrap draws in 10.19 s (2.83 GiB peak pool, max discrepancy 2.2e-14 vs sequential refits across 21 draws). Multi-replicate synthetic calibration auto-routes to parallel CPU LAPACK `dsyevr` (`8.66 s/rep` on 16 CPU cores for 2,000 draws on `compact_grid=False`) with atomic chunked checkpointing (`--resume`), `--cohort-grid`, `--n-test-participants`, `--interval-method`, `--run-mde-curve`, and `--scenarios` batching.
- On 20 November 2026, development diagnostics across the 10 legacy scenarios at `n=92` (`compact_grid=True`) identified right-tail undercoverage of the raw percentile bootstrap on `nonlinear_proxy_linear_covariates` (`85.4%`, driven by fourth-order skewness and mean--SD coupling) and conservative overcoverage on `ci_95_vs_comparator` (`100%`, ~`2.1x` SD ratio from $N_{\text{eff}} \approx n/2$ estimation-error doubling). The default uncertainty estimator was revised to the centered full-procedure studentized participant-multiplicity bootstrap-$t$ (`calibration.interval_method: studentized_t`, retaining raw percentile intervals alongside it), the synthetic suite was expanded to 17 scenarios (10 core/legacy plus 7 stress regimes) with a pre-specified MDE signal grid (`[0.0, 0.25, 0.5, 0.75, 1.0]`), and `calibration.n_test_participants` was unified at `25,000`. The full 1,000-repetition cohort-grid (`compact_grid: false`) validation campaign remains pending. See [the design limitation](../docs/research_design.md#participant-divisions-and-uncertainty).
- The 21-band table reproduces the inclusive software-registry windows. Their earlier cohort use does not establish independent feature provenance. New data-derived bands would be a separate exploratory change.
- Readiness counts (`92/88/4/180`, exact clinical dictionary parity, Schema v5 store join) are verified by `audit_readiness.py` without evaluating clinical outcomes; `build_manuscript_artifacts.py` refuses to populate cohort tables from non-cohort runs unless `--allow-synthetic-preview` is passed.

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
