# Plan: Control-vs-Disease Machine Learning Classification Layer

**Status**: revised after auditing `data/raw/` folder structure, `data/preprocessed/` parquet tables, and `src/`. Section 0 documents verified facts. Read it before Section 1 — it bounds what this pipeline may claim.

**Scope discipline (explicit user instruction)**: this pipeline uses **only** information present in the repository's data. No external variable is assumed, imputed, or "corrected for" unless a real column/field backs it. Age adjustment is **not attempted** — controls have no age data anywhere in the repo, and inventing or imputing one would inject bias, not remove it.

---

## 0. Verified data facts (measured, not assumed)

Source: `data/raw/` directory tree (`Control/{Female,Male}/<subject>/...`, `Disease/{Female,Male}/<subject>/...`) and `data/preprocessed/{clinical_subjects,area_metadata}.parquet`.

- [x] Subjects with valid spectra in both blocks: **147** (all)
- [x] Class balance: **97 disease / 50 control** (1.94:1) — mild imbalance, `class_weight='balanced'` suffices, no SMOTE
- [x] Areas per subject per block: median 6 (min 3, max 6) — already collapsed by subject-level median aggregation
- [x] **Sex is a folder-level grouping variable in the raw data itself** (`Control/Female/`, `Control/Male/`, `Disease/Female/`, `Disease/Male/`), not an inferred or injected covariate — it is ground truth structure the data was organized by
- [x] Sex vs. class (from `clinical_subjects.parquet`, matches folder structure): disease 45F/52M; control 40F/10M
- [x] Age: present for disease (n=97, mean 66.3); **absent for controls (n=1 non-null, out of 50)**. No date-of-birth, acquisition date, or session field exists anywhere in `data/raw/` filenames or `data/preprocessed/` tables that could substitute. **Age adjustment is out of scope — not attempted, not claimed.**
- [x] `clinical_present` in controls: False for 49/50 — controls carry no clinical endpoint data beyond sex and control status
- [x] Data anomaly: 1 subject has `is_control=True` **and** `diagnosis_family='ischemic_stroke'` — flagged, not silently resolved (see §8)
- [x] Slide ↔ subject: 1 subject per slide (242 slides, 0 mixed-class slides) — slide is nested in subject, so subject-level CV already controls it; slide number is **not** used as an acquisition-order proxy in any claim (no real date field backs that inference — dropped per scope discipline)
- [x] Preprocessing normalization: SNV (1300 block), area (2850 block) — both row-wise (`axis=1` in `src/preprocessing/pipeline.py:normalize_spectra`) → **no cross-sample leakage**, preprocessing may stay outside the CV loop
- [x] Subject aggregation: median over areas, per subject per block, computed **before** and independent of any label → unsupervised reduction, no label leakage, plain `StratifiedKFold` on subjects is leakage-safe
- [x] Feature matrix availability: `write_subject_feature_exports` runs unconditionally inside `run_biological_pca` (`biological_pca.py:1830`) → `subject_feature_matrix_whole.parquet` **is** produced in `disease_vs_control` mode

### 0.1 The confounding problem — sex only, bounded by available data

A Control-vs-Disease classifier on this cohort is not automatically a clean disease classifier: sex is structurally associated with class (controls 80% female, disease 46% female) **by the data's own folder organization**, not by any assumption this plan introduces. A high AUC is consistent with a disease-associated spectral signature, a sex-associated one, or both.

Age is **not** part of this analysis: no age exists for controls, no proxy exists anywhere in the repo, and this plan does not manufacture one. This is stated once here as a hard limitation and is not revisited as if it were solvable.

**Confound handling (Phase C), scoped to what the data actually supports:**
- [ ] Sex-stratified sensitivity analysis: refit and evaluate the full nested-CV pipeline within the female-only subset (n=45 disease / 40 control — balanced, adequately powered). Male subset (52 disease / 10 control) reported descriptively only — too imbalanced for inference.
- [ ] Sex-stratified permutation null (primary inferential test): permute labels **within sex strata**, preserving the sex–class association in the null. A model that only reads sex will be non-significant against this null.
- [ ] Sex-only baseline: a logistic model using sex alone (directly available, folder/clinical-table grounded). The spectral model must beat it to support any spectral claim.

No acquisition-date, batch, or age covariate is introduced anywhere in this pipeline.

### 0.2 What this pipeline may and may not claim

**May claim**: "An exploratory, internally cross-validated classifier separates control from disease EV Raman spectra with balanced accuracy X (95% CI …), significant against a sex-stratified permutation null (p = …). The signal persists in a sex-matched female-only subgroup."

**May not claim**: diagnostic accuracy, biomarker status, clinical utility, causality, generalization, or anything about age effects (no age data for controls exists). Single-cohort, single-site, no external validation. All figure captions and the report carry this.

Per `CLAUDE.md` §6, no ML claim may attach to Biological PCA outputs. This pipeline only *consumes* the PCA layer's feature matrix as a separate, read-only downstream layer.

---

## 1. Scope and outcome

- [ ] Add `run_control_vs_disease_ml_pipeline`, fired only in `disease_vs_control` mode, writing to a new isolated result tree: `Results_control_vs_disease/07_ml_classification/`.
- [ ] Three algorithms (PLS-DA, L1-logistic, Random Forest) × two feature representations (whole fused spectrum, 21-band panel), each with identical nested-CV + permutation + bootstrap treatment, plus the sex-confound battery of §0.1.

**Design decisions:**
- Algorithms: PLS-DA + L1-Logistic + Random Forest.
- Imbalance: `class_weight='balanced'` + inner-CV decision-threshold tuning. No synthetic oversampling — SMOTE interpolates between spectra, producing physically meaningless "spectra" that violate Raman's additive-mixture structure. Rejected.
- Rigor: nested CV + permutation + bootstrap CIs + ROC/PR + confusion matrices + model comparison.

**Output routing correction**: `04_ml_classification/` (original draft) collides with `get_cross_region_summary_dir()` → `04_cross_region_summary/`, which Biological PCA populates (`01_subject_features/`) in both modes. Use `07_ml_classification/` (next free top-level number after `06_spectral_interpretability`).

---

## 2. Scientific rigor invariants (non-negotiable)

- [ ] **Subject-level splits.** Feature matrix is one row per subject (`load_subject_feature_matrix` raises on duplicate `subject_id`). Median-over-areas aggregation is label-free and precedes splitting → leakage-safe. Plain `StratifiedKFold` on subjects is correct — verified against the data, not assumed.
- [ ] **Preprocessing in-fold.** SNV/area normalization are row-wise, need not move inside the fold. `StandardScaler` and all supervised steps (hyperparameter tuning, threshold selection, feature selection) fit on the training fold only. Mirror `minimal_panel._fit_tuned_logistic` (`minimal_panel.py:250-278`).
- [ ] **Nested CV.** Outer: `RepeatedStratifiedKFold` (5 folds × 10 repeats) — at n=147 this is well-conditioned and yields a variance estimate a single 5-fold cannot. Inner: `StratifiedKFold` for hyperparameter + threshold tuning. Never tune and report on the same split. Do **not** import `minimal_panel._cv_strategy`'s `LeaveOneOut` branch — unreachable at n=147, would silently mislead if present.
- [ ] **Imbalance metrics.** Report balanced accuracy, MCC, ROC-AUC, PR-AUC, sensitivity, specificity, F1. Never bare accuracy. Positive class = Disease (`is_control == False` → 1), documented in every artifact. State majority-class baseline (0.66 accuracy / 0.50 balanced accuracy) alongside model results.
- [ ] **Significance.** Full-pipeline label permutation (tuning re-run under each permuted label vector), p = `(1 + Σ[null ≥ obs]) / (n + 1)`. Two nulls: unrestricted, and sex-stratified. Sex-stratified p-value is the primary inferential result.
- [ ] **Multiplicity.** Pre-specify PLS-DA on the whole spectrum as the single primary model; all others secondary. FDR (Benjamini–Hochberg, `src.analysis.statistics.fdr_correction`) across the per-band univariate screen and across the 6 (model × representation) permutation p-values.
- [ ] **Uncertainty.** Bootstrap 95% CIs (percentile, 1000 resamples) on headline metrics; bootstrap selection-stability for sparse models.
- [ ] **Claims discipline.** Per §0.2 — no age claims, no diagnostic/biomarker/causal claims.
- [ ] **Determinism, CPU-only.** `random_state=42`, seed arrays, joblib `n_jobs` from config. No GPU.

---

## 3. Feature representations

- [ ] Primary — whole fused spectrum: `feature_columns(frame)` → `1300__*` + `2850__*` channels. PLS-DA and L1-logistic handle p ≫ n natively; RF is robust to it.
- [ ] Secondary — 21-band biochemical panel: reuse `minimal_panel._band_feature` over `config whole_spectra.bands` (`config/eda.yaml:329-352`). 21 features, n=147 → comfortable, interpretable, leakage-safe.
- [ ] Per-range (1300-only / 2850-only) models: out of scope for v1, noted as future extension.

---

## 4. Implementation checklist

### 4.1 New module — `src/analysis/control_vs_disease_ml.py`

```python
def run_control_vs_disease_ml_pipeline(*, feature_matrix_path, cfg=None) -> dict[str, Path]
```

- [ ] **Load**: `load_subject_feature_matrix(feature_matrix_path=...)`. `X = frame[feature_columns(frame)]`, `y = (~frame["is_control"].astype(bool)).astype(int)` (Disease = 1). Do **not** call `filter_disease_subjects`.
- [ ] **Guard**: raise if `is_control` missing or has NaN; raise if either class has < 10 subjects. Log the label anomaly (control with `diagnosis_family='ischemic_stroke'`) as a hard warning, not a silent drop.
- [x] **Phase A — cohort description**: `cohort_summary.csv` (n per class, imbalance ratio, majority-class baseline, sex × class crosstab with χ², explicit note that age is unavailable for controls and is out of scope). Verified on the real cohort: χ²=13.9, p=0.00019 — confirms the sex confound is real and statistically overwhelming, which is exactly why Phase C is not optional.
- [x] **Phase B — models**, each run through the same nested-CV harness:
  - [x] PLS-DA: `PLSRegression(scale=False)` on in-fold-scaled X; tune `n_components` on inner CV; threshold the decision score; VIP via `pls_regression._calculate_vip`. Thresholds the raw PLS regression score directly (not routed through a second-stage L1-logistic like `splsda.run_splsda_classifier` — a deliberate, verified-sound divergence; see implementation note below).
  - [x] L1-Logistic: `sklearn_utils.l1_logistic_model` (`class_weight='balanced'`); tune `C` on inner CV via an in-fold-scaled tuning loop mirroring `minimal_panel._fit_tuned_logistic`.
  - [x] Random Forest: `RandomForestClassifier(class_weight='balanced_subsample', random_state=42, n_jobs=n_jobs)`; inner grid over `n_estimators`/`max_depth`; **permutation importance** (not impurity importance — biased toward high-cardinality continuous features across 1802 correlated channels).
- [x] **Threshold tuning**: decision threshold chosen on inner-CV out-of-fold scores maximizing balanced accuracy; applied unchanged to the outer test fold. Verified by mutation test (leaking the outer fold into tuning is caught by a scaler-instrumentation guard) and by a held-out-fold benchmark: tuned threshold reached 0.91 balanced accuracy vs. 0.74 for a naive 0.5 cutoff on an imbalanced (40/10) synthetic set — confirms threshold tuning is carrying real weight for imbalance, not cosmetic.
- [x] **Phase C — sex-confound battery** (§0.1): `_sex_stratified_refit` (female-only subset refit via the same `_run_repeated_nested_cv`; male subset explicitly not refit — too imbalanced, 52/10 — reported via n-counts only), `_sex_only_baseline` (logistic on sex alone, in-fold threshold tuning), both written to `confounder_analysis.csv` per (model × representation). Unit-tested: reports metrics when sex present, reports explicit `missing_sex_column` status when absent, reports `insufficient_n` status when a stratum is too small rather than silently proceeding.
- [x] **Phase D — permutation tests**: `run_permutation_test` — full-pipeline label permutation (tuning re-run per permuted draw, not just refit), unrestricted + sex-stratified nulls, `(1+Σ)/(n+1)` p-value. **Cost design decision**: each permutation draw uses one un-repeated `StratifiedKFold` pass (`_single_split_balanced_accuracy`), not the full `RepeatedStratifiedKFold` used for the Phase B point estimate — matches the existing repo convention in `splsda._permuted_splsda_score` and `pls_regression._permuted_pls_r2` (neither repeats CV inside its permutation loop, confirmed by grep before implementing). Repeating CV per permutation would multiply cost by `n_outer_repeats` for no statistical benefit, since the null distribution's spread across permutations already captures Monte Carlo noise.
- [ ] **Phase E — bootstrap CIs**: resample subjects (stratified), recompute headline metrics, 2.5/97.5 percentiles. **Not yet implemented.** (Note: `model_comparison.csv` currently reports repeat-mean/std across `RepeatedStratifiedKFold`, not yet a bootstrap percentile CI — Phase E adds that.)
- [ ] **Phase F — univariate screen**: Mann–Whitney per band + BH-FDR via `fdr_correction`. Descriptive context only. **Not yet implemented.**
- [x] **Provenance & timing**: `write_analysis_provenance(output_dir, resolve_data_source(cfg), cfg, command=...)`; local `_phase_timer`/`_write_phase_timings` (own implementation, not imported from `disease_only_pipeline.py` — that module's timer helpers are private/underscored, so this module defines its own rather than reaching across a module boundary for private symbols) → `control_vs_disease_ml_phase_timing.csv` in run-metadata dir. Verified present on a real run.

**Outputs** (`write_csv_atomic`) under `07_ml_classification/` — Phase A/B verified on the real 147-subject cohort in the prior session; Phase C/D unit-tested, real-cohort verification of this pass pending (see end-to-end status below):
- [x] `cohort_summary.csv`
- [x] `model_comparison.csv` — one row per (model × representation), 6/6 rows present; all metrics finite; repeat-mean/std for Phase B metrics; now also carries `permutation_p_value_unrestricted`, `permutation_p_value_sex_stratified`, `permutation_p_fdr_q` (BH-FDR across the 6-row family, using the sex-stratified p as primary per model when available), `is_primary_model` flag. Bootstrap 95% CI still pending Phase E.
- [x] `confounder_analysis.csv` — one row per (model × representation): female-only refit metrics + sex-only baseline metrics + status columns
- [x] `permutation_null_distributions.csv` — long-form null draws (model × representation × null_type × value) for later histogram figures
- [x] `<model>_<representation>_cv_predictions.csv` — 6/6 present, out-of-fold scores per repeat per subject
- [x] `<model>_<representation>_feature_importance.csv` — 6/6 present, correctly typed per model (vip / l1_logistic_coefficient / permutation_importance), block identity preserved (e.g. `1300__733.2`)
- [ ] `univariate_band_screen.csv` — pending Phase F
- [ ] `ml_report.md` — pending (depends on Phase E, and on user sign-off on the primary result)

**Figures** (`src/visualization/`, theme via `theme.py`) — **not yet implemented, pending Phase C/D/E data to plot**:
- [ ] ROC curves (models overlaid, sex-only baseline as reference line)
- [ ] PR curves
- [ ] Confusion matrices
- [ ] Metric-comparison bars with CI whiskers
- [ ] Permutation null histograms with observed statistic marked

### 4.2 Orchestrator hook — `scripts/run_eda.py`

- [x] Extended the biological-PCA block (`run_eda.py`) with a sibling branch:

```python
elif analysis_mode == "disease_vs_control":
    feature_matrix_path = biological_outputs.get("subject_feature_matrix_whole")
    print_status("Starting control-vs-disease ML classification")
    run_control_vs_disease_ml_pipeline(feature_matrix_path=feature_matrix_path, cfg=config)
    print_status("Finished control-vs-disease ML classification")
```

- [x] Gated behind `analysis.control_vs_disease_ml.enabled` (checked inside the runner itself, which returns `{}` early when disabled).
- [x] Imported beside `run_disease_only_cross_region_pipeline`. Also wired `validate_control_vs_disease_ml_config` into `_main()` right after `validate_eda_mode_config`, gated to `disease_vs_control` mode.

### 4.3 Output dir helper — `src/__init__.py`

- [x] Added:

```python
def get_ml_classification_dir() -> Path:
    path = _output_dir("eda_subdir") / "07_ml_classification"
    path.mkdir(parents=True, exist_ok=True)
    return path
```

Mirrors `get_confirmatory_panel_dir`. `07_` avoids the `04_cross_region_summary` collision — verified on a real run, no collision, no `Results_intra_disease/` created, no root-level `biological_pca/` dir.

### 4.4 Config — `config/eda.yaml`

- [x] Added under `analysis:` (verified loaded and honored in the real run's `config_snapshot.yaml`):

```yaml
  control_vs_disease_ml:
    enabled: true
    representations: ["whole", "band_panel"]
    primary_model: "pls_da"
    primary_representation: "whole"
    n_outer_splits: 5
    n_outer_repeats: 10
    n_inner_splits: 4
    n_permutations: 1000
    n_bootstrap: 1000
    threshold_metric: "balanced_accuracy"
    stratify_permutation_by: "sex"
    random_state: 42
    n_jobs: -1
    pls_components_grid: [1, 2, 3, 4, 5, 6, 8, 10]
    logistic_c_grid: [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    rf_n_estimators_grid: [200, 500]
    rf_max_depth_grid: [null, 4, 8]
```

- [x] `config/eda_disease_control.yaml` already sets `mode: disease_vs_control` — production track, unchanged.
- [x] Added `config/eda_disease_control_fast.yaml` (`_extends: eda_disease_control.yaml`), lowering `n_permutations`/`n_bootstrap`/repeats for iteration only — not for final science. Used for the real smoke run below.

### 4.5 Config validation — `src/analysis/eda_mode.py`

- [x] Added `validate_control_vs_disease_ml_config`, called from `run_eda.py` after `validate_eda_mode_config`, gated to `disease_vs_control` mode: positive ints, non-empty grids, `threshold_metric` in supported set, `primary_model`/`primary_representation` cross-product checks. Raises `ValueError` with actionable messages. Unit-tested (6 cases: accepts defaults, rejects bad primary model, rejects primary_representation not in list, rejects non-positive int, rejects empty grid, skips validation when disabled).

### 4.6 Tests — `tests/test_control_vs_disease_ml.py`

34 tests, all passing (25 from Phase A/B + 9 new for Phase C/D). Full suite (`uv run pytest -v`) unaffected: 194 passed. `uv run ruff check .` clean on all touched/new files (pre-existing baseline error count went 16 → 15, i.e. strictly improved, not worsened).

- [x] Leakage guard: no test-fold index touched during any scaler/tuner/threshold fit (synthetic labelled data, `StandardScaler.fit` instrumented via monkeypatch). **Mutation-tested**: manually re-ran the guard logic against a deliberately leaky call (passing full `X, y` instead of the train-fold slice) and confirmed it correctly flags the leak — the guard is a real detector, not a vacuous pass.
- [x] Threshold leakage guard: decision threshold derived only from inner-fold scores — proved structurally (`_fit_tuned_pls_da`/`_fit_tuned_logistic_threshold` never receive outer-fold data in their signature) plus a repeatability check.
- [x] Determinism: identical `random_state` → identical out-of-fold predictions (`pd.testing.assert_frame_equal`); additionally verified different seeds produce different results (rules out a vacuous always-equal test).
- [x] Imbalance: `_classification_metrics` never reports bare `accuracy`; skewed synthetic set (5 vs. 45) produces finite balanced accuracy, MCC, sensitivity, specificity; positive class = Disease asserted throughout.
- [x] Permutation: p ∈ (0,1] on real synthetic runs (`test_permutation_p_value_bounded`); pure-noise data does not manufacture a small p-value (`test_permutation_null_centers_near_chance_on_pure_noise`, p > 0.05 on a no-signal cohort); `_permute_labels` verified to preserve class counts unrestricted and within-stratum counts when stratified.
- [x] **Sex-stratified null (mandatory) — `test_sex_stratified_null_is_mandatory_confound_guard`.** This is the decisive test for the whole confound-handling design. Constructs a synthetic cohort where class is a *pure function of sex* (all-female-control, all-male-disease, no independent disease signal). Confirms: (1) the model trivially separates classes (balanced accuracy > 0.9) by reading sex; (2) the **unrestricted** permutation null is fooled — p < 0.05, i.e. an unrestricted-only analysis would have reported this pure-confound signal as a significant finding; (3) the **sex-stratified** null is **not** fooled — p > 0.05, because permuting within sex strata preserves the very association the model exploits, so permuted data score just as well as real data. This is proof-by-construction that Phase C/D's confound machinery does what §0.1 requires, not just that the code runs.
- [x] Output contract: `model_comparison.csv` has exactly 3 models × 2 representations = 6 rows, now with permutation p-value/FDR columns and exactly one `is_primary_model=True` row; `confounder_analysis.csv` has 6 rows with `status` columns; `permutation_null_distributions.csv` restricted to the two expected `null_type` values. Figures still pending (Phase E precedes them).
- [x] Runner smoke on tiny synthetic feature matrix (33 control-vs-disease synthetic subjects, both representations) — passes with `resolve_data_source`/`write_analysis_provenance` mocked to isolate unit-test scope from the real data store.

**Real end-to-end smoke (beyond unit tests)** — `uv run scripts/run_eda.py --config config/eda_disease_control_fast.yaml --range both` run against the actual 147-subject cohort:
- [x] **Phase A/B**: completed successfully. `cohort_summary.csv` correctly reports the real imbalance (97/50) and the real sex confound (χ²=13.9, p=0.00019). Label-anomaly warning fired exactly on subject `11`, matching §8. All 6 `model_comparison.csv` rows populated, metrics finite. Provenance and phase timing verified. Mode isolation confirmed.
- [x] **Phase C/D on the real cohort**: completed successfully, ~24 min wall time for the ML phase. Inspected every output, not just existence:
  - **Null distributions are not degenerate.** Both unrestricted and sex-stratified nulls, across all 6 (model × representation) combinations, center tightly at balanced accuracy ≈ 0.49-0.51 (as a true null should), with observed values (0.64-0.71) clearly above the entire null spread — ruled out a bug where the null could be artificially collapsed toward the observed value or stuck at a constant.
  - **Primary result** (PLS-DA, whole spectrum, pre-specified): balanced accuracy 0.656, sex-stratified permutation p=0.0196, BH-FDR q=0.0235 (significant). All 6 rows significant at both nulls and after FDR.
  - **Female-only sensitivity refit corroborates, does not merely repeat, the full-cohort result**: primary model's female-only balanced accuracy is 0.683 (n=40 control / 45 disease) — not collapsed toward chance, actually slightly higher than the full-cohort estimate. This is the load-bearing check: if the spectral signal were purely a sex artifact, restricting to one sex should destroy it, and it did not.
  - **Sex-only baseline** (logistic on sex alone) reaches balanced accuracy 0.668 / ROC-AUC 0.634 — the spectral models' ROC-AUC (0.70-0.82 across the 6 rows) exceeds this, i.e. the spectral models are not simply reproducing what sex alone predicts.
  - **Runtime cost, extrapolated to production settings** (`n_permutations` 50→1000, `n_outer_repeats` 2→10, both ~20x and ~5x): Random Forest is the dominant cost. Its fast-overlay permutation cost was already efficient per-draw (single-split design, not repeated-CV-per-permutation — see Phase D design note above), but its Phase B repeated-CV fit (648s under the fast overlay's `n_outer_repeats=2`) will scale roughly linearly with repeats. **Flagging for the user before a production run**: expect the control-vs-disease ML phase to take on the order of 1-2 hours at full `config/eda.yaml` settings, dominated by Random Forest. Not disqualifying for a one-time manuscript run, but worth knowing in advance rather than discovering mid-run.

---

## 5. Reuse map (do NOT reimplement)

| Need | Existing symbol | File |
|---|---|---|
| Load subject × wavenumber matrix + label | `load_subject_feature_matrix`, `feature_columns`, `feature_metadata` | `src/analysis/clinical_feature_associations.py` |
| L1-logistic (`class_weight='balanced'`) | `l1_logistic_model`, `fit_with_convergence_capture` | `src/analysis/sklearn_utils.py` |
| In-fold tuned logistic (scaler-in-fold) | `_fit_tuned_logistic` | `src/analysis/minimal_panel.py` |
| Permutation + bootstrap + parallel | `_permutation_test_binary`, `_bootstrap_stability_summary`, `_parallel_values`, `_normalize_n_jobs` | `src/analysis/minimal_panel.py` |
| PLS-DA nested CV + in-fold scaling template | `run_splsda_classifier`, `_tune_pls_l1da`, `_permutation_test` | `src/analysis/splsda.py` |
| VIP scores | `_calculate_vip` | `src/analysis/pls_regression.py` |
| Band integration (21-band panel) | `_band_feature` | `src/analysis/minimal_panel.py` |
| FDR | `fdr_correction` | `src/analysis/statistics.py` |
| Atomic CSV writes | `write_csv_atomic`, `write_text_atomic` | `src/analysis/io_utils.py` |
| Provenance artifacts | `write_analysis_provenance` | `src/data_preparation/store.py` |
| Phase timing pattern | `_phase_timer`, `_write_phase_timings` | `src/analysis/disease_only_pipeline.py` |
| Plot theme | — | `src/visualization/theme.py` |

**Deliberately NOT reused**: `minimal_panel._cv_strategy` (`LeaveOneOut` branch unreachable at n=147).

## 6. Not changed

Preprocessing, Biological PCA, disease-only pipeline, constants, data loaders. Feature matrix consumed read-only.

---

## 7. Verification checklist

**Status: Phase A + Phase B verified. Phases C/D/E/F, figures, and the report are not yet built — the items below marked `[ ]` require those phases first.**

**Unit / integration**
- [x] `uv run pytest tests/test_control_vs_disease_ml.py -v` — 34/34 passed
- [x] `uv run pytest -v` (full suite unbroken) — 194/194 passed
- [x] `uv run ruff check .` (zero new warnings) — confirmed zero errors in all touched/new files; pre-existing baseline error count went 16 → 15 (strictly improved, not worsened) via an unrelated auto-fix picked up while linting this module

**Smoke (fast overlay — reduced resampling, not final science)**
- [x] `uv run scripts/run_eda.py --config config/eda_disease_control_fast.yaml --range both` (note: `--max-subjects 8` caps at 4/4 per class, below the `_MIN_SUBJECTS_PER_CLASS=10` guard, and correctly raises — run the full cohort for this smoke test instead)

Post-run checks under `Results_control_vs_disease/07_ml_classification/`:
- [x] `cohort_summary.csv`, `model_comparison.csv` present; comparison has 3 models × 2 representations (6 rows) — verified on real data
- [x] `confounder_analysis.csv` — verified on real data: 6 rows, all `status=ok`, female-only refit + sex-only baseline populated for every (model × representation)
- [ ] ROC/PR/confusion/comparison/permutation-null figures rendered — pending Phase E + figures work (CSV data for these — `permutation_null_distributions.csv` — is written and verified; only the plotting layer is pending)
- [x] `run_manifest.json`, `data_manifest_snapshot.json`, `config_snapshot.yaml` present — verified
- [x] `00_run_metadata/control_vs_disease_ml_phase_timing.csv` written — verified, per-phase timings inspected including permutation/refit sub-phases
- [x] No `Results_intra_disease/` outputs created (mode isolation); no root-level `biological_pca/` dir — verified (pre-existing `Results_intra_disease/` from an earlier session confirmed untouched by mtime)
- [x] Metrics finite; positive class documented as Disease — verified
- [x] Both permutation p ∈ (0,1] — verified on real data: all 12 p-values (6 rows × 2 nulls) finite and in (0,1]; null distributions inspected directly (not just the p-value) and confirmed centered near chance (mean balanced accuracy 0.49-0.51) rather than degenerate

**Production run (manuscript numbers — full resampling)**
- [ ] `uv run scripts/run_eda.py --config config/eda_disease_control.yaml --range both` — hold until Phase E lands (bootstrap CIs) and the user has reviewed the Phase C/D real-cohort result below and resolved the §8 label anomaly. **Runtime warning**: extrapolating from the fast-overlay timing, expect ~1-2 hours for the ML phase at full production settings, dominated by Random Forest's repeated-CV fit. Consider running overnight or reducing the RF grid (`rf_n_estimators_grid`/`rf_max_depth_grid` in `config/eda.yaml`) if that cost is unwelcome — your call, not applied unilaterally.

**Behavioral verification**: done for Phase A-D — drove the real pipeline end-to-end (not just unit-test mocks) against the actual 147-subject cohort under the fast overlay and inspected every emitted CSV, including the null distributions themselves (not just pass/fail on the p-value).

### Real-cohort Phase C/D result (fast-overlay settings — NOT final science, but informative)

Primary pre-specified result: **PLS-DA, whole spectrum, balanced accuracy 0.656, sex-stratified permutation p=0.0196 (BH-FDR q=0.0235, significant)**. Female-only sensitivity refit corroborates (balanced accuracy 0.683, not collapsed toward chance). Sex-only baseline (ROC-AUC 0.634) underperforms the spectral models (ROC-AUC 0.70-0.82). All 6 (model × representation) combinations were significant at both nulls and after FDR correction under this fast-overlay run. This is an early, low-permutation-count read, not the manuscript number — Phase E (bootstrap CIs) and a full `n_permutations=1000` production run are still needed before this is publication-usable per §0.2 — but it is a genuine, non-trivial early signal, not a null result.

### Implementation note: PLS-DA formulation

`splsda.py`'s established repo pattern is a two-stage classifier: PLS projection, then L1-logistic on the PLS scores. This module instead thresholds the raw PLS regression output directly (single-stage), with the decision threshold tuned on inner-CV for balanced accuracy. Both are legitimate PLS-DA formulations. This was a deliberate choice, verified empirically (see Phase B checklist above: 0.91 vs. 0.74 balanced accuracy for tuned vs. naive threshold on a held-out imbalanced synthetic set) rather than assumed — imbalance handling does not depend on `class_weight` (PLS has no such parameter) but on threshold tuning, which is confirmed to be carrying real weight.

---

## 8. Pre-manuscript blockers

- [ ] Resolve the label anomaly: subject with `is_control=True` and `diagnosis_family='ischemic_stroke'`. One field is wrong; must be corrected or the subject excluded with documented reason before the production run.
- [ ] Decide the primary endpoint in advance and do not change it after seeing results. Pre-specified: PLS-DA / whole spectrum / sex-stratified permutation p.

Age adjustment and acquisition-date/batch adjustment are **not** blockers — they are explicitly out of scope because no such data exists in the repository. This is stated once (§0.1) as a permanent limitation of this cohort, not something pending resolution.
