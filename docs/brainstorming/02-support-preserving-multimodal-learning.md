# Support-Preserving Multimodal Learning under Selective Biomarker Measurement

## Research question

How can a model use an expensive modality available only for a clinically restricted outcome region, while guaranteeing that it does not fabricate improvements for outcome classes where that modality was never observed?

The linked Raman cohort illustrates this sharply. The 102 clinical-data participants link to the large workbook, but none has 3-month mRS 5 or 6, whereas these severe outcomes are present in the broader 1,042-row clinical cohort. A conventional multimodal classifier trained only on Raman-observed participants can silently extrapolate its apparent benefit into a part of the outcome space with no Raman support.

## Core idea

Call the framework **SPAR: Support-Preserving Augmentation with Restricted modalities**. It begins with a full-support clinical-only predictor and permits Raman information to update probabilities only inside outcome categories for which Raman training support exists.

This is not simply an imputation method or a better classifier. It is a constrained decision rule with exact invariants: for unsupported outcome categories, the augmented predictor has the same probability as the clinical-only teacher for every individual. The method can therefore make a precise statement about the portion of risk it did not alter.

## Inputs

- Full clinical cohort with outcome `Y`, such as ordinal 3-month mRS 0–6, and routinely available predictors `X`.
- A selective multimodal cohort with `X`, Raman representation `Z`, and the outcome, along with a verified indicator `S` for Raman availability.
- A supported outcome subset `A`, selected only from observed design support. In the current data, a preliminary candidate is `A = {0,1,2,3,4}`, because no Raman-linked participant has mRS 5 or 6. This must be recomputed after confirming the exact Raman linkage.
- Metadata describing how Raman availability was selected. The method must distinguish outcome-support restriction from ordinary missingness.

## Proposed method

1. Train a clinical-only ordinal or multiclass predictor `p0(Y | X)` on the large clinical cohort using cross-fitting. This is the full-support teacher.
2. Restrict the Raman model to the conditional supported problem, learning `q1(Y | X, Z, Y in A)` only among participants with observed Raman spectra and outcomes in `A`.
3. Construct the augmented distribution:

   - for `k in A`, `p1(k | X, Z) = p0(A | X) q1(k | X, Z, Y in A)`;
   - for `k not in A`, `p1(k | X, Z) = p0(k | X)`.

4. Use shrinkage from `q1` toward the clinical conditional distribution `q0(k | X, Y in A) = p0(k | X)/p0(A | X)`. The shrinkage strength should depend on Raman reliability and the effective multimodal sample size.
5. Supply a support certificate with every model: supported categories, count by category, selection assumptions, pointwise invariants, and outcomes for which no Raman-added-value claim is made.

## Why the construction matters

For every person and every unsupported category, `p1 = p0` exactly. If the true outcome is unsupported, the log-score difference is exactly zero:

`log p1(Y | X, Z) - log p0(Y | X) = 0` for `Y not in A`.

Within the supported set, the log-score difference equals `log q1 - log q0`. This cleanly separates “value of Raman inside the observed support” from “risk handled only by clinical information.” The construction does not claim that Raman is safe, causal, or useful for unobserved severe outcomes. It prevents a false universal claim.

Hard classifications may still change because renormalizing probabilities in `A` changes the largest class. Other scores, including Brier score and ranked probability score, do not automatically cancel outside `A`; these must be analysed directly. The invariant is about probability mass for unsupported categories and log score when the realized outcome is unsupported.

## Scientific and methodological novelty

Two-phase and selective-measurement literature already studies expensive covariates, variable importance, and outcome-dependent sampling. The frontier contribution must be stronger than applying a standard two-phase estimator:

- a distribution-level prediction rule that preserves unsupported outcome probabilities pointwise;
- a characterization of which proper-score improvements are identifiable from the observed multimodal support;
- finite-sample shrinkage based on Raman measurement reliability and selection sensitivity;
- a support certificate that makes the generalization boundary machine-checkable;
- experiments showing that unrestricted fusion can create illusory gains in unsupported outcome regions.

The method should be framed as a general solution for any selectively observed modality: pathology, advanced imaging, genomics, wearables, or spectroscopy. Stroke is the motivating case, not the only domain.

## Theory questions worth developing

- Which strictly proper scoring-rule contrasts decompose into a supported conditional term under the SPAR construction?
- Under what selection assumptions is Raman incremental utility within `A` identifiable?
- Can a cross-fitted orthogonal estimator quantify the added value of `Z` within `A` while using the full clinical cohort for `p0`?
- How should the model abstain when `p0(A | X)` is near zero, so conditional Raman updates would be unstable?
- What lower bounds on risk or regret hold under bounded selection bias within `A`?

The theory must state assumptions plainly. Outcome support alone does not make Raman selection missing at random inside `A`.

## Evaluation plan

### Controlled missing-modality benchmarks

Start with public multimodal classification data where every subject has all modalities. Hide one modality according to outcome categories, severity strata, and covariates. Because the withheld modality remains available to the evaluator, measure both apparent and true performance across the full outcome support.

Compare:

- clinical-only/full-modality baseline;
- naïve fusion trained on the selected sample;
- a standard two-phase or imputation method;
- SPAR without shrinkage;
- full SPAR with reliability-aware shrinkage;
- an oracle trained on the unmasked modality.

Metrics should include log score, Brier score, ranked probability score, calibration, category-specific performance, and preservation of unsupported-category probabilities.

### Finnish data analysis

Use the 1,042-row clinical cohort only to train and validate the clinical teacher with participant-level splits. Fit the Raman conditional model only in the verified Raman-linked participants and supported outcomes. Never allow the same participant’s outcome to train `p0` and evaluate the stacked prediction. Report support counts in every split.

The primary real-data result should be a bounded statement: Raman improves or fails to improve discrimination/calibration *within the supported mRS region*. It must not claim prediction of death or severe disability when no Raman-observed examples support that claim.

## Risks and safeguards

- **Selection inside supported outcomes is biased:** run selection-sensitivity analyses; phrase the result as association or conditional predictive utility, not population-wide benefit.
- **Few Raman participants:** favour interpretable, strongly regularized spectral representations and uncertainty intervals; do not use a large deep network as the main result.
- **Outcome timing leakage:** exclude 24-hour NIHSS, discharge mRS, or follow-up measurements from an admission-prognosis model unless the task is explicitly post-acute prediction.
- **Spectral visit ambiguity:** freeze and verify collection timing before defining prediction time zero.
- **Outcome support changes after linkage correction:** derive `A` from the verified linkage, never from the current provisional table.

## Deliverables for a manuscript

- A support map showing modality availability by outcome category.
- A diagram of the clinical teacher, supported conditional Raman model, and probability-preserving assembly.
- A theorem/proposition for the pointwise support invariant and proper-score decomposition.
- Public masked-modality experiments with an oracle comparison.
- An empirical support certificate for the Finnish analysis.
- A transparent statement of unsupported clinical claims.

## Suitable journals

**Information Fusion** is the clearest target if the paper emphasizes the general selective-modality fusion problem and supplies theory plus broad benchmarks. **Artificial Intelligence in Medicine** fits a method with a compelling clinical-support story. **IEEE Journal of Biomedical and Health Informatics** is appropriate for a rigorously evaluated multimodal clinical-prediction method. **Pattern Recognition** is viable if the constrained prediction framework and benchmark evidence are sufficiently general and algorithmically substantial.

## Key literature to read before implementation

- Dai, Carroll, and Chen, two-phase variable-importance inference, *Biometrics* (2025). https://doi.org/10.1093/biomtc/ujaf095
- Nam et al., ordinal outcomes under outcome-dependent sampling (2026 preprint). https://arxiv.org/abs/2605.27330
- Kluger et al., predict-then-debias under nonuniform covariate sampling (2025). https://arxiv.org/html/2501.18577v2
- Choi, Kim, and Kim, partial identification of shared-outcome risk contrasts (2026 preprint). https://arxiv.org/abs/2610.09658

