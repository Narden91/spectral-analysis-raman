# Conditional Modality Testing for Biological Distributions Observed through Unequal Repeated Measurements

## Research question

Do case and control participants have different *underlying distributions* of Raman spectra after conditioning on pre-event clinical covariates, when each participant is observed through an unequal, spatially dependent set of repeated spectral measurements?

This is deliberately different from asking whether a classifier separates cases from controls. A classifier can exploit age, sex, medication, collection artefacts, or the larger number of repeated spectra in one group. The proposed test targets the participant-level biological distribution while respecting what the instrument actually measured.

## Why this direction fits the new data

The new control workbook improves covariate information: 50 controls have complete age and vascular-risk/medication fields, although their exact Raman linkage still needs confirmation. The stroke workbook provides analogous pre-event vascular and chronic-medication fields for the linked cohort. These variables make a conditional case-control question possible.

The usable support is nevertheless narrow. Only 74 of the 102 linked stroke participants fall inside the observed control age range of 42–77, and some exposure strata have no controls, such as chronic anticoagulant use. A defensible method must diagnose and report lack of overlap rather than obtain a global adjusted p-value by extrapolation.

## Target estimand and null hypothesis

Let `D_i` indicate stroke versus control, `C_i` contain pre-event covariates, and `P_i` denote the unobserved participant-level distribution of Raman spectra across valid sampling locations. Each participant contributes an unequal bag of spectra `B_i = {X_ij}` drawn with spatial dependence and measurement noise.

The primary null is:

`H0: P_i independent of D_i given C_i`,

restricted to the common covariate-support population. This is a conditional distributional association test. It is not evidence that stroke causes a spectral change, because pre-event covariates may be incomplete and disease-related collection processes may remain.

## Proposed method

Call the method **BAG-COMET: Bag-Aware Conditional Modality Testing**.

1. Estimate a participant distribution embedding from repeated Raman areas. The embedding must model finite bag size and spatial correlation, rather than treating all pixel-level spectra as independent subjects.
2. Learn nuisance functions for disease status given covariates and for the spectral distribution embedding given covariates, using cross-fitting at the participant level.
3. Apply a residualized conditional-independence statistic, but propagate uncertainty from the finite, unequal repeated-measurement bags. Bag uncertainty can be estimated through spatial block resampling, a hierarchical bootstrap, or a measurement-error kernel.
4. Limit inference to an explicit overlap region in covariate space. Participants outside common support receive an `unsupported` label instead of extreme inverse-probability weights.
5. Calibrate significance with participant-level permutations or a conditional randomization procedure consistent with the nuisance model. Pixels, areas, and repeated visits never cross a participant split.

The output includes a global conditional-association test, effect localization in the spectral embedding, an overlap report, and a sensitivity range across prespecified adjustment sets.

## What is new beyond existing conditional-independence tests

COMET, generalized covariance measures, and doubly robust kernel conditional-independence tests already assess whether an added modality contains information conditional on covariates. Applying one of them to PCA scores would not be novel.

BAG-COMET would make a distinct contribution by treating the modality as a latent distribution observed through finite, unequal, correlated bags. Its validity analysis should address:

- the error from estimating each participant’s spectral distribution;
- unequal number of areas/spectra by disease group;
- spatial dependence within an area or map;
- representation learning that is cross-fitted independently of the test split;
- covariate-overlap refusal rather than unrestricted reweighting.

The goal is a conditional test whose type-I error remains controlled when group differences exist only in measurement count or within-participant sampling variability. This is a broadly useful computer-science/statistics problem in spatial omics, pathology tiles, wearable streams, and multi-instance biomedical data.

## Statistical plan

### Covariate sets

Specify tiers before looking at results:

1. demographic tier: age and sex;
2. pre-event clinical tier: demographic variables plus hypertension, diabetes, atrial fibrillation, coronary disease, smoking, and chronic medication exposure;
3. exploratory acute tier: adds severity, treatment, and acute laboratory variables, explicitly interpreted as a conditional association after disease presentation rather than a pre-event comparison.

The primary analysis should use the pre-event clinical tier. Avoid including downstream stroke severity, acute treatment, or post-event biomarkers as default “confounders,” because their adjustment changes the estimand and may induce collider bias.

### Representation and resampling

Begin with transparent distribution features: mean spectrum, within-participant dispersion, selected quantiles, and a kernel mean embedding. Compare with a learned set encoder only after it is trained inside training folds. Use a nested, hierarchical resampling design:

- split and cross-fit by participant;
- resample participants for population uncertainty;
- resample spatial blocks or areas within participant for measurement uncertainty;
- keep all spectra from one participant in the same split.

### Simulation study

Build simulations with known conditional nulls. Independently vary:

- true disease effect versus pure covariate confounding;
- equal versus unequal repeated-measurement counts;
- iid versus spatially correlated within-bag observations;
- covariate overlap versus positivity failures;
- spectral shift in the mean, dispersion, or multimodal shape.

Assess false-positive rate, power, uncertainty calibration, overlap refusal, and robustness to representation misspecification. Baselines should include participant-mean PCA plus a conventional conditional-independence test, a matched analysis, fixed-budget subsampling, and a naïve pixel-level test.

## Real-data analysis protocol

1. Verify the Raman membership, control linkage, sample collection dates, and visit labels.
2. Create a public-free, de-identified analysis manifest with counts and missingness only.
3. Define common support from age, sex, and pre-event covariates without inspecting spectral outcomes.
4. Run the primary tier, then the two sensitivity tiers.
5. Report the proportion of participants excluded or labelled unsupported, covariate balance, test statistic, participant-level uncertainty, and a correction-aware spectral localization summary.

With an effective cohort near 147 participants before overlap restriction, this should be framed as a careful feasibility and method demonstration. It should not promise subtle subgroup discovery or molecular mechanism validation.

## Claims the study can and cannot make

Supported target claim: “Within a stated common-support population and adjustment set, the observed participant-level spectral distribution is or is not associated with stroke status beyond measured pre-event covariates.”

Do not claim a causal stroke effect, a brain extracellular-vesicle mechanism, or a disease biomarker ready for diagnosis. Do not use a non-significant result as proof of no biological difference; report a sensitivity/power analysis and uncertainty interval.

## Main risks and mitigations

- **Insufficient common support:** provide an overlap-restricted estimand and report its population explicitly. An unsupported region is a result, not a data-cleaning failure.
- **Unequal sampling drives apparent effects:** simulate this failure mode and use bag-aware uncertainty; include fixed-budget sensitivity analyses.
- **Control linkage uncertainty:** make linkage verification a precondition of the real-data test.
- **Preprocessing dependence:** rerun a small, prespecified set of defensible preprocessing variants and report whether the conclusion is stable. Do not search variants until significance appears.
- **Too little power after adjustment:** use the paper’s simulations and confidence intervals to characterize the limit; do not replace the conditional question with an unadjusted classifier.

## Deliverables for a manuscript

- A formal problem statement for conditional testing of a latent distribution observed through unequal repeated measurements.
- A validity or error-bound result for the bag-aware statistic.
- A simulation suite that exposes false positives from unequal sampling.
- A covariate-overlap figure and cohort flow diagram.
- A real-data conditional association analysis with sensitivity tiers.
- Reusable benchmark code that does not release patient-level spectra or identifiers.

## Suitable journals

**Pattern Recognition** is attractive if the multi-instance distributional testing method, theory, and benchmark suite are central. **Information Fusion** fits a method that treats clinical covariates and repeated spectroscopy as heterogeneous information sources. **Artificial Intelligence in Medicine** is appropriate for a strong methodological paper grounded in a clinical case-control application. **IEEE Journal of Biomedical and Health Informatics** is suitable when the biomedical signal-processing and rigorous validation aspects lead the manuscript.

## Key literature to read before implementation

- Kook and Lundborg, COMET: algorithm-agnostic significance testing for multimodal data (2024). https://arxiv.org/abs/2402.14416
- Doubly robust generative neural conditional-independence testing, *Journal of the Royal Statistical Society Series B* (2026). https://academic.oup.com/jrsssb/article/88/1/195/8217135
- Massiani et al., kernel conditional-independence tests and learning bounds, NeurIPS (2025). https://arxiv.org/html/2506.03898
- He and Sutherland, the practical hardness of conditional independence testing (2025). https://arxiv.org/html/2512.14000
- Fröhlich et al., operator-spectral representations for conditional independence, ICML (2026). https://proceedings.mlr.press/v306/frohlich26a.html

