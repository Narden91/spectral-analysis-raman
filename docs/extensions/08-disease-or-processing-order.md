# Disease or processing order? Checking the case-control Raman difference, then measuring individual deviation with conformal methods

Status: research design. Label-free pilot checks ran on 2026-10-31 (results below). No spectrum-versus-outcome test and no bridge check has been run. The hypotheses below are prespecified and untested.

Data constraint: this design uses only the existing preprocessed store and `data/clinical_master.csv`.

Extends: [03](../brainstorming/03-conditional-modality-testing.md) and [04](../brainstorming/04-conditional-testing-research-proposal.md) (conditional test), [06](06-replication-aware-normative-deviation.md) (normative deviation), [02](../brainstorming/02-support-preserving-multimodal-learning.md) (support certificate). Step 4 may use the components from [07](07-replicate-contrastive-attribution.md).

## Why this comes first

Sample numbers split the groups. Controls are numbered 1–43 and 192–199. Stroke patients are numbered 44–191 and 200–250. The pilot in doc 07 shows that spectra of samples with neighbouring numbers are more similar than spectra of random samples, and this holds inside one acquisition mode (FP 31.7% vs 56.3%; SS 18.5% vs 33.6% identification among neighbours vs random distractors).

The case-control result (AUC 0.711) therefore cannot be separated from processing order with the analyses done so far. This is not evidence of an artefact. It is a confound that has not been tested.

The manuscript's sex checks do not test it. The female-only model and the sex-stratified permutation both keep 37 of the 40 female controls, and all 37 sit in the block 1–43.

## Research questions

- **RQ1.** Can the stroke–control spectral difference be separated from processing order, covariates and thrombolysis?
- **RQ2.** Among patients that the controls cover, what share lies outside the covariate-conditional control distribution? Report a lower confidence bound.
- **RQ3.** Does the size of the deviation track NIHSS at admission, perfusion mismatch volume, cardioembolic versus large-artery class, and 3-month mRS?

## Conditions

- If the data owner confirms that controls were isolated as a separate batch, the case-control questions cannot be answered beyond the bridge check. The paper then documents a shortcut signal, and RQ2 and RQ3 are dropped or restricted to stroke.
- If the two slides per patient are two draws, the replication matching in step 4 changes (see doc 07, Limits).

## Existing data used

| Item | Value | Use |
|---|---|---|
| Sample number | Controls 1–43 (43) and 192–199 (7); stroke 44–191 and 200–250 | Processing-order proxy. |
| Bridge controls | 7 controls (numbers 192–199): 3 female, 4 male; mean age 60.6; blood drawn 2022-06 to 2022-11 | Check on processing order. The early controls were drawn 2022-01 to 2023-01. |
| Controls with age | 47 | Normative model. |
| Supported stroke patients | 58 (age range 42–76, no anticoagulant use; doc 06) | Primary population. |
| Stroke without IVT | 37 | Drug-free comparison. |
| Slide replication | Controls: 1 slide × 3 maps. Stroke: 2 slides × 3 maps (95 of 97). | Replication matching. |
| Acquisition mode | FP in 56% of control slides, 43% of stroke slides | Technical covariate. |

Sample number correlates with admission NIHSS within stroke (ρ = 0.25, p = 0.013, one of 14 checks, uncorrected). Include a drift term in every within-stroke test.

## Method

### 1. Nuisance check

Predict group from sample number, acquisition mode and QC metrics alone. Report how much spectral discrimination remains after conditioning on them.

### 2. Bridge check (needs approval before running)

Score the 7 bridge controls on the cross-validated case-control direction, adjusted for sex. Also compare the early controls with the bridge controls directly.

This check is weak, and the result must be reported that way:

- 4 of the 7 bridge controls are male, against 6 of 43 among the early controls. Sex alone could push them toward the stroke side. A female-only version leaves 3 controls.
- AUC 0.711 implies a shift of about 0.78 SD. The mean of 7 controls has a standard error of about 0.38 SD. A one-sided test at α = 0.05 has about 66% power before sex adjustment, and less after.

A null result does not resolve the confound.

### 3. Drift model

Fit a smooth spectral drift against sample number inside stroke (samples 44–250, where disease status is constant). Check it on the bridge controls. Extending the drift to samples 1–43 is an assumption. Report sensitivity bounds.

### 4. Conformal deviation

1. Choose a low-dimensional representation: the band panel, block PCA, or the linear components from doc 07. If using doc 07 components, train them on stroke folds outside the test fold. Never train them on the calibration controls.
2. Fit a normative model on controls: an age spline plus a sex shift (doc 06).
3. Use full conformal prediction (refit with each test point), so all 47 controls stay in the reference.
4. Compute covariate-shift weights from covariates alone. Select patients with weighted conformal p-values (Jin and Candès 2023).
5. Apply the support certificate (58 patients).
6. Match replication: compare one stroke slide, chosen by a rule independent of the spectra, with the single control slide.
7. Use the second stroke slide as a test-retest of the deviation. Do not merge the two slide-level p-values.

### 5. Estimands set by the p-value floor

With 47 controls, no unweighted conformal p-value falls below 1/48 ≈ 0.021.

| Case | p-value floor | Patients at the floor that BH (q = 0.10) needs, of 58 | Of 97 |
|---|---|---|---|
| Unweighted, 47 controls | 0.0208 | 13 (22%) | 21 |
| Effective control count ≈ 23 after weighting, or 2× merge of two slide p-values | 0.0417 | 25 (43%) | 41 |

Per-patient selection has little power at this size. Primary estimands are aggregate:

- a test of whether the p-values are uniform;
- a lower 95% confidence bound on the number of non-conforming patients.

Per-patient selection is secondary.

### 6. Strata and drift

Use the 37 patients without IVT as the drug-free comparison. IVT is a stratifier fixed in advance. Include the drift term in every within-stroke test.

### 7. Calibration

Add simulated group shifts and simulated order drift to real control spectra. Check the error rate of the full procedure.

## Baselines

- Unadjusted PLS-DA, as in the manuscript.
- The doc 06 parametric z-score.
- The doc 04 conditional test.
- Nuisance-only predictor (step 1) as a floor for any spectral claim.

## Limits

- Control blood draws run from 2022-01 to 2023-01. Patient draw dates are missing, so storage time cannot be conditioned on. The conformal null includes pre-analytical differences.
- Only 10 male controls exist. The shared age curve with a sex shift is an assumption (doc 06).
- Anticoagulant users, atrial fibrillation (2 controls) and coronary disease (1 control) fall outside the control support.
- Conformal coverage holds only if controls and patients are exchangeable after weighting. A processing-order difference breaks this, which is the reason for steps 1–3.

## Result scenarios

| Scenario | Interpretation |
|---|---|
| The difference survives the nuisance check, the drift model and the bridge check. | The strongest case-control evidence this cohort can give. Still exploratory. |
| The difference cannot be separated from processing order. | A documented shortcut signal in spectroscopic liquid biopsy. A caution for the field. DeGrave et al. (2021) is a similar case for chest X-ray AI (cited from memory; not checked). |
| The difference holds, and deviation tracks injury. | An individual-level Raman deviation score with calibrated uncertainty. Needs external replication. |
| The difference holds, and deviation is unrelated to injury. | A group difference without individual clinical information. |

## Open items

- What does the sample number encode? Was the isolation batched by group? Were Raman sessions mixed by group, as `main.tex:192` states? If isolation was batched but Raman sessions were mixed, the confound is in isolation and storage, not in the instrument.
- Are the two slides one draw or two? Default: one draw.
- Was the research plasma drawn before or after the alteplase bolus? Default: after, for IVT patients.

## Claims

- Allowed: "Within the supported population and after adjusting for sample number, patients deviate from the age- and sex-conditional control distribution (lower bound X%)."
- Not allowed: causal stroke effect, diagnostic threshold, EV specificity, brain origin. A null bridge check is not proof of no batch effect.

## Target journals

Analytical Chemistry or Patterns if the shortcut-signal result leads. Artificial Intelligence in Medicine or IEEE JBHI if the deviation score leads. Confirm quartile and scope at submission.

## References

- Jin Y, Candès EJ. Model-free selective inference under covariate shift via weighted conformal p-values. arXiv:2307.09291.
- Bates S et al. Testing for outliers with conformal p-values. Ann Stat 2023. (From memory; not checked.)
- Marquand AF et al. Biol Psychiatry 2016; Wolfers T et al. Psychol Med 2020: normative modelling of patient heterogeneity.
- DeGrave AJ et al. AI for radiographic COVID-19 detection selects shortcuts over signal. Nat Mach Intell 2021. (From memory; not checked.)
- Doc 04: `docs/brainstorming/04-conditional-testing-research-proposal.md`; doc 06: `docs/extensions/06-replication-aware-normative-deviation.md`.
