# ML Paper Authoring Directory (`ml_paper/`)

This directory contains the initial manuscript scaffolding, verified cohort and experimental descriptions extracted from [spectra_manuscript/main.tex](../spectra_manuscript/main.tex), and reconciled methodological specifications from [docs/research_design.md](../docs/research_design.md) and [data/clinical_master_notes.md](../data/clinical_master_notes.md).

## Contents

1. [cohort_and_acquisition_description.md](cohort_and_acquisition_description.md):
   - Extracted wet-lab, instrument, preprocessing, and cohort parameters from [spectra_manuscript/main.tex](../spectra_manuscript/main.tex).
   - Explicit reconciliation of discrepancies between the earlier manuscript text and the verified standalone clinical master ([data/clinical_master.csv](../data/clinical_master.csv)).
2. [main.tex](main.tex):
   - Initial LaTeX manuscript draft structured for biomedical data science / computational biology venues (*IEEE Journal of Biomedical and Health Informatics* or *PLOS Computational Biology*).
   - Includes full author list, affiliations, ethics declarations, funding acknowledgments, frozen preprocessing table, 21-band registry table, mathematical formulation of the paired signal-to-disagreement generalized eigensolver, nested cross-validation protocol, and result table templates linked to `manuscript_ml/` deliverables.
3. [references.bib](references.bib):
   - Consolidated BibTeX database combining the experimental/clinical citations from `spectra_manuscript` with the primary methodological and lipid-prediction citations from `docs/research_design.md`.

## Mapping `manuscript_ml/` Outputs to Manuscript Sections

| Manuscript Element | Source Script in `manuscript_ml/` | Output Artifact |
|---|---|---|
| Section 2.1 (Cohort Eligibility & Linkage) | [audit_readiness.py](../manuscript_ml/audit_readiness.py), [run_primary_evaluation.py](../manuscript_ml/run_primary_evaluation.py) | `readiness_manifest.json`, `eligibility_and_split_manifest.json` |
| Section 3.1 / Table 3 (Synthetic Calibration & Failure Modes) | [run_simulation_calibration.py](../manuscript_ml/run_simulation_calibration.py) | `simulation_scenarios_summary.csv`, `simulation_calibration_report.csv` |
| Section 3.2 / Table 4 (Primary Held-Out LDL-C Comparison) | [run_primary_evaluation.py](../manuscript_ml/run_primary_evaluation.py) | `comparator_errors_and_differences.csv`, `fold_selected_settings.csv` |
| Section 3.3 / Table 5 (Repeatability $\mathrm{ICC}(A,1)$, Discriminability & Subspace Stability) | [run_primary_evaluation.py](../manuscript_ml/run_primary_evaluation.py) | `baseline_repeatability_report.csv`, `subspace_principal_angles.csv` |
| Section 3.4 (Sensitivity Analyses, Secondary Lipid/Platelet Endpoints & Continuous Anchor Associations) | [run_sensitivity_and_secondary.py](../manuscript_ml/run_sensitivity_and_secondary.py) | `covariate_and_subset_sensitivity.csv`, `outer_partition_seed_sensitivity.csv`, `secondary_endpoints_summary.csv`, `anchor_associations_continuous.csv` |
| Section 4.2 (Limitations) | [audit_readiness.py](../manuscript_ml/audit_readiness.py) | `limitations_table.csv` |


## Computational Notes

- Code lives in the [src/analysis/paired_lipid/](../src/analysis/paired_lipid/) package; settings live in [config/manuscript_ml.yaml](../config/manuscript_ml.yaml). Every stage writes `run_provenance_<stage>.json` (config and hash, git commit, package versions, data identity).
- The batched bootstrap engine matches a fully sequential refit to <= 1e-8 on CPU and on GPU with exact fp64 eigensolvers (test `test_batched_bootstrap_draws_match_sequential_refits`). Two engine bugs were fixed on 8 November 2026: the outer refit regularized with tr(W) instead of tr(T), and the default mixed-precision GPU eigensolver changed every bootstrap draw (up to 3e-2). Earlier bootstrap outputs are invalid.
- Calibration coverage is measured against the expected held-out risk difference, estimated by scoring each replicate's fitted outer-fold models on one independent synthetic test cohort (n = 5,000). Earlier calibration reports used the mean of the point estimates as the target (bias identically 0) and are invalid. Only `--seed-set validation` gives a confirmatory gate result.
- Secondary endpoints use the primary population and outer folds; a missing endpoint value gets weight 0 in training and test.
- Experiment datasets are memoized per argument set by `resolve_experiment_dataset` in [manuscript_ml/__init__.py](../manuscript_ml/__init__.py). The cohort path requires `--approve-cohort-evaluation` on every call. Use `clear_dataset_cache()` to force a reload.
- The full 1,000-replicate calibration campaign has not been run; its ~21 h duration is an estimate from a 200-draw timing.
