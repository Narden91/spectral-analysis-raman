# Manuscript Overview: Paired Technical-Sample Raman Representations and Circulating Lipids in Ischemic Stroke

**Draft manuscript:** [main.tex](main.tex)  
**Active research design:** [docs/research_design.md](../docs/research_design.md)  
**Experiment runbook:** [manuscript_ml/runbook.md](../manuscript_ml/runbook.md)  
**Target venue candidates:** *IEEE Journal of Biomedical and Health Informatics* / *PLOS Computational Biology*  
**Status (23 November 2026):** Retrospective method-development protocol, software pipeline, readiness linkage checks, development simulation calibration, and numerical/performance verification are complete. Confirmatory validation-seed calibration (`1,000` repetitions on the `1 cm⁻¹` cohort grid) and authorized real-cohort outcome evaluation remain pending.

---

## 1. Goal of the Manuscript

The manuscript tests one primary research question:

> **Can a low-dimensional linear Raman representation learned from paired technical samples improve held-out LDL-C prediction for unseen eligible acute ischemic stroke participants compared with a locked clinical/technical covariate model and established spectral methods?**

### Intended Contribution and Scope

- **Biological target:** Information in the retained, normalized Raman spectral shape of size-exclusion-chromatography (SEC) plasma extracellular-vesicle (EV)-enriched fractions that predicts a recorded circulating lipid measurement (LDL-C).
- **Computational target:** A leak-free, repeat-aware evaluation of predictive improvement and technical-repeat agreement that prevents pseudoreplication and separates spectral repeatability from shared clinical covariates.
- **Methodological boundary:** The paired generalized eigensolver adapts the established signal-to-noise / noise-fraction family. The first contribution is a rigorous repeat-aware evaluation in this preparation and cohort; a separate methodological claim requires demonstrating a reproducible advantage over established spectral comparators selected strictly inside training folds.
- **What the study cannot claim:** Predictions reflect relative normalized spectral shape under this protocol. Even a positive predictive result cannot identify contributing particle classes (EVs vs co-isolated lipoproteins or residual platelet material), quantify absolute lipid or EV concentration, establish brain origin, prove conditional independence failure, or claim clinical diagnostic utility.

---

## 2. Cohort, Data Hierarchy, and Frozen Preprocessing

### Cohort and Verified Linkage

- **Source study:** Prospective IBioStroke acute ischemic stroke cohort (Comprehensive Stroke Center, University Hospital in Krakow; ethical approval no. `1072.6120.118.2020`).
- **Eligibility audit (verified 9 and 20 November 2026):** Of `97` stroke participants with raw spectral folders, `5` are excluded for `SEX_FOLDER_CONFLICT` (never silently relinked), leaving `92` identity-verified stroke participants (`88` with two eligible technical samples and `4` with one = `180` eligible technical samples; two participants lost one sample because pixel QC rejected two of three C–H maps).
- **Controls:** Excluded from the primary model (`50` controls lack recorded clinical lipid panels and have only one technical sample per participant).
- **Repeat meaning:** Investigator confirmation recorded during design identifies paired technical samples (`A` and `B`) as technical replicates from a single blood collection, not longitudinal visits. Exact preparation and acquisition stages repeated independently between `A` and `B` still require documentation.

### Four-Level Measurement Hierarchy and Frozen Spectral Grids

Measurements preserve the hierarchy `participant -> technical sample (A, B) -> map (2–3 maps) -> retained pixel`, aggregated by median across retained pixels within each map and then median across eligible maps per block (requiring at least two retained maps per block; three-map-complete is evaluated as a sensitivity subset):

| Spectral block | Frozen grid (`1 cm⁻¹` step) | Channels | Baseline / smoothing | Block normalization |
|---|---|---:|---|---|
| Fingerprint (`1300`) | `734–1788 cm⁻¹` | `1,055` | arPLS ($\lambda = 10^6$), Savitzky–Golay (order 2, 7 ch) | Standard Normal Variate (SNV) |
| C–H stretching (`2850`) | `2451–3197 cm⁻¹` | `747` | arPLS ($\lambda = 10^6$), Savitzky–Golay (order 2, 9 ch) | Trapezoidal area |

Pixel-level screening enforces minimum SD `1.0`, saturation ceiling `65,000`, spatial-neighbor cosmic-ray repair ($z > 8$), SNR $\ge 5$, and at least `4` retained pixels per map.

---

## 3. How It Has Been Done (Methodology and Software Pipeline)

### Primary Endpoint and Model Family

The primary endpoint is the participant-weighted difference in held-out LDL-C squared error between the locked covariate-only ridge baseline and the paired representation plus covariates:

$$
\widehat{\Delta} = \sum_{i=1}^{n} \tilde{\mu}_i \left[(y_i - \widehat{y}_{C,i})^2 - (y_i - \widehat{y}_{C+Z,i})^2\right]
$$

Positive values favor adding spectra. All models share identical participant folds, equal participant population weights $w_{is} = \mu_i / (N_{\text{eff}} m_i)$, and training-only preprocessing (median imputation, missingness indicators, constant-column screening, standardization, and MFA block scaling by $1/\sqrt{\lambda_1^{(b)}}$):

1. **`covariate_ridge` (primary baseline):** Ridge regression on locked clinical/technical covariates (`age_years`, `sex_female`, `statin`, and participant-mean parsed `sample_label_trend` as a declared nuisance proxy; acquisition modes FP and SS are pooled as equivalent).
2. **`fixed_band_ridge` (comparator):** Covariates plus mean intensities across `21` locked Raman band windows.
3. **`pca_ridge` (comparator):** Covariates plus top $k \in \{5, 10, 15\}$ weighted MFA-PCA coordinates without pair-disagreement weighting.
4. **`pls_regression` (comparator):** Multiplicity-weighted NIPALS PLS regression ($1\text{--}5$ components) directly on standardized covariates and participant-average block-scaled spectra.
5. **`paired_linear_ridge` (proposed primary spectral procedure):** Uses paired training participants to form uncentered within-pair disagreement $W$ (penalizing both random disagreement and systematic $A/B$ shifts) and between-participant covariance $T$, solves $T v_j = \rho_j (W + \lambda \tau I_k) v_j$ ($\tau = \operatorname{tr}(T)/k$, $\lambda \in \{0.001, 0.01, 0.1, 1, 10\}$), retains $r \in \{1, 2, 3\}$ directions, and fits downstream ridge ($\alpha \in \{10^{-3}, \dots, 10^3\}$).
6. **`strongest_comparator`:** Within each outer fold, selects whichever established comparator (`fixed_band_ridge`, `pca_ridge`, or `pls_regression`) achieved the lowest inner validation MSE, refits it on the outer training fold, and compares its outer errors directly against `paired_linear_ridge`.
7. **`band_paired_ridge` (secondary ablation):** Applies the paired eigensolver to standardized 21-band features, reusing the selected $(\lambda, r, \alpha)$ without extra outer selection.

### Validation, Uncertainty, Repeatability, and Sensitivity

- **Nested participant splits:** Locked `5`-fold outer / `3`-fold inner cross-validation stratified by pair availability (`primary_seed = 42`, deterministic tie-breaking favoring smaller $r$, smaller $k$, larger $\lambda$, larger $\alpha$). All samples and maps of a participant stay in the same fold.
- **Uncertainty intervals:** Centered full-procedure studentized participant-multiplicity bootstrap-$t$ (`2,000` complete refits keeping original fold memberships fixed, carrying multiplicity through preprocessing, moments, inner selection, and outer scoring, with raw percentile limits and Maritz–Jarrett endpoint Monte Carlo SEs recorded alongside). If any resample fails to yield the locked contrast or has degenerate dispersion ($\hat{s} \le 10^{-12}$), the dataset interval is marked failed rather than dropping draws.
- **Repeatability and subspace stability:** Computed strictly within each outer fold (never pooled across folds) for both full predictions and isolated spectral contributions $s_{is} = \widehat{\beta}_Z^\top \tilde{z}_{is}$: two-way mixed-effects single-measure absolute-agreement $\mathrm{ICC}(A,1)$, ANOVA mean squares, $\mathrm{SD}(s)$, spectral variance share, paired RMSD, Bridgeford discriminability, bidirectional global and local top-1 retrieval, metric-orthonormal fold principal angles ($\theta_1\text{--}\theta_3$), and 1,802-channel effective spectral weights.
- **Sensitivity and secondary analyses:** Five covariate presets (`primary`, `no_label_trend`, `quadratic_label_trend`, `quadratic_age_and_label`, `pre_event_risk_factors`), four subset modes (`all_eligible`, `paired_only`, `single_sample`, `three_map_complete`), three additional outer partition seeds (`101, 202, 303`), three secondary endpoints (`hdl_mmol_l`, `triglycerides_mmol_l`, `plt_10e9_l`, each re-tuned inside inner CV with zero weight on missing outcomes), and continuous exploratory anchor partial correlations (`2,000` resamples, no categorical source assignments).

### Reproducible Software Implementation

- **Core math package:** [src/analysis/paired_lipid/](../src/analysis/paired_lipid/) implements both an independent sequential primal reference engine and an exact-fp64 vectorized CPU/GPU batched Schur engine (`_backend.py`, `_gram_cache.py`, `_eigensolvers.py`, `_batched_models.py`, `batched.py`, `models.py`, `nested_cv.py`, `metrics.py`, `synthetic.py`, `calibration.py`, `cohort.py`).
- **Locked settings and staged runners:** [config/manuscript_ml.yaml](../config/manuscript_ml.yaml) locks all grids, seeds, readiness expectations, and `17` synthetic scenarios. Staged CLIs in [manuscript_ml/](../manuscript_ml/) run readiness auditing (`audit_readiness.py`), synthetic simulation and calibration (`run_simulation_calibration.py`), primary evaluation (`run_primary_evaluation.py`), sensitivity/secondary/anchor analyses (`run_sensitivity_and_secondary.py`), end-to-end orchestration (`run_all.py`), and deterministic LaTeX table/figure generation (`build_manuscript_artifacts.py`).
- **Governance guards:** Every stage records `run_provenance_<stage>.json` (config hash, git commit/diff hash, package versions, band registry, store manifest). Real-cohort stages fail closed without `--approve-cohort-evaluation` and verified readiness counts; the artifact builder refuses non-cohort outputs unless `--allow-synthetic-preview` is passed (watermarking all preview tables and figures).

---

## 4. Outcomes So Far: Verified vs Pending

### A. Completed and Verified

| Area | Recorded outcome | Date & source |
|---|---|---|
| **Data linkage & schema readiness** | `92` eligible stroke participants (`88` paired, `4` single-sample, `180` samples), exact `82`-column parity between `clinical_master.csv` and `clinical_master_dictionary.csv`, verified Schema v5 preprocessed store join, `0` missing values across all `5` covariate presets and `4` laboratory endpoints among the `92` eligible participants. | 9 & 20 Nov 2026 ([AGENT_LEARNINGS.md](../AGENT_LEARNINGS.md), `audit_readiness.py`) |
| **Calibration diagnostics & estimator revision (development seeds)** | Across `10` initial scenarios (`n = 92`, compact grid, `n_test = 25,000`), raw percentile bootstrap undercovered `nonlinear_proxy_linear_covariates` (`83.3–85.4%`) and overcovered `ci_95_vs_comparator` (`100%`, `1.88–2.65×` SD ratio). Revising the default estimator to centered studentized-$t$ achieved `96.9–97.9%` primary coverage across linear-signal, no-association, $A/B$-shift, and high-dimensional-nuisance regimes (`96.1%` / `246/256`, Wilson lower `0.930` on `nonlinear_proxy_linear_covariates` at `R = 256`), `97.9–100.0%` comparator coverage (`1.20–1.57×` SD ratio), `0.0%` dataset failures across all core and stress regimes (including `covariate_missingness` at `97.9%` primary / `93.8%` comparator), and clear comparator separation (`target_delta_vs_comparator = +0.0732`) on `high_dim_technical_nuisance`. | 20 Nov 2026 ([docs/research_design.md](../docs/research_design.md), [manuscript_ml/runbook.md](../manuscript_ml/runbook.md)) |
| **Numerical parity & engineering verification** | All `577` repository tests pass (`CPU` + real CUDA `GPU`, `0` failures), `ruff check .` passes, `graft` index current, `0` prohibited prose-word hits. Real preprocessed spectra with seeded dummy outcomes (no real clinical outcomes evaluated) match sequential refits to `~1e-14`. | 23 Nov 2026 ([AGENT_LEARNINGS.md](../AGENT_LEARNINGS.md)) |
| **Vectorization, caching, and runtime efficiency** | Real-spectra dummy-outcome GPU bootstrap (`B = 2,000`, exact fp64 + fused MFA Gram kernel + adaptive CheFSI): `9.19 s`, `2.34 GiB` peak VRAM pool, `0` fallbacks. Vectorized batched Schur support closed sequential fallbacks for `covariate_missingness` (`41.37 s -> 3.34 s`, `12.4×`, `0` fallbacks at `B = 100` CPU) and `single_sample` (`41.07 s -> 2.63 s`, `15.6×`, `0` fallbacks); missing-covariate anchor resampling (`5.40 s -> 0.34 s`, `15.9×`); parallel calibration IPC drops the 25,000-row pandas `raw_cov_frame` (`2.18 MB -> 0.00 MB` pickled/task) and caps spawned Windows `loky` worker OpenBLAS threads (`8 -> 1`). | 23 Nov 2026 ([AGENT_LEARNINGS.md](../AGENT_LEARNINGS.md)) |
| **End-to-end artifact pipeline** | `run_all.py --synthetic --quick` + `build_manuscript_artifacts.py --allow-synthetic-preview` verified end to end, emitting all `5` LaTeX table fragments, `3` figures, and build manifest with SHA-256 input hashes. | 23 Nov 2026 ([AGENT_LEARNINGS.md](../AGENT_LEARNINGS.md)) |

### B. Pending Before Final Manuscript Results

1. **Measurement metadata confirmation:** Confirm source lipid units (currently marked assumed `mmol/L`), direct vs calculated LDL-C assay definition, fasting status and draw timing relative to EV isolation, statin reference window, and which wet-lab/acquisition stages were independently repeated for samples `A` and `B`.
2. **Confirmatory validation-seed calibration campaign:** Run `--seed-set validation` on the `1 cm⁻¹` cohort grid (`1,000` repetitions × `2,000` bootstraps, `n_test = 25,000`) across prespecified core scenarios and the MDE signal grid after separate runtime approval (provisional gate: Wilson 95% lower bound $\ge 0.925$ on unconditional coverage and `< 1%` failed dataset intervals).
3. **Authorized real-cohort evaluation:** Execute `run_primary_evaluation.py` and `run_sensitivity_and_secondary.py` with `--cohort --approve-cohort-evaluation`, then populate the pending LaTeX tables and figures in [main.tex](main.tex) via `build_manuscript_artifacts.py`.

---

## 5. Decision Criteria and Key Limitations

### Interpretation Rules Once Cohort Results Are Run

- **Supported primary improvement:** Requires a positive $\widehat{\Delta}$ and a calibrated 95% interval excluding zero. An interval spanning zero is inconclusive, not proof of absence or equivalence.
- **Methodological advantage claim:** Additionally requires a reproducible advantage over `strongest_comparator` selected strictly inside inner validation. If a simpler baseline performs comparably, retain the simpler method and withdraw any new-method claim.
- **No outcome switching:** If LDL-C prediction is unsupported or inconclusive, report the result and limitations as locked without changing endpoints, preprocessing, or cohort definitions to rescue the claim.

### Structural Study Limitations

1. **Retrospective design:** Earlier exploratory analyses preceded this locked protocol; nested CV controls internal model-selection bias but cannot erase prior researcher familiarity with the cohort.
2. **Normalized shape vs mass/concentration:** SNV and area normalization remove absolute intensity scale.
3. **Co-isolation (MISEV2023):** SEC isolates lipoproteins and plasma proteins alongside EVs; clinical platelet count does not measure residual platelet material in the fraction.
4. **Shared artefacts and transportability:** Repeatability measures only independently repeated stages; random participant splits provide internal validation only, not new-session or external-center generalization.
5. **Sample size (`N = 92`) and laboratory metadata:** Modest participant count limits detection of small predictive gains, and unrecorded fasting/assay timing or calculated LDL-C formulas restrict biological attribution.

---

## 6. Key Files Map

| File / directory | Role |
|---|---|
| [docs/research_design.md](../docs/research_design.md) | Locked scientific design, equations, acceptance gates, and literature review |
| [ml_paper/main.tex](main.tex) | Manuscript LaTeX source with locked reporting schemas |
| [ml_paper/cohort_and_acquisition_description.md](cohort_and_acquisition_description.md) | Reconciled wet-lab, instrument, preprocessing, and cohort metadata |
| [config/manuscript_ml.yaml](../config/manuscript_ml.yaml) | Frozen grids, seeds, readiness counts, and 17 synthetic scenarios |
| [manuscript_ml/runbook.md](../manuscript_ml/runbook.md) | Stage-by-stage CLI commands, runtime budgets, and reporting rules |
| [src/analysis/paired_lipid/](../src/analysis/paired_lipid/) | Core statistical, spectral, batched CPU/GPU, and calibration library |
| [AGENT_LEARNINGS.md](../AGENT_LEARNINGS.md) | Dated evidence snapshot, retired-idea history, and verified engineering benchmarks |
