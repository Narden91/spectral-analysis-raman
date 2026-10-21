# Reliable Clinical Attribution of Spectral Representations under Correlated Biological Anchors

## Research question

Can a model identify reproducible Raman spectral factors that are reliably associated with correlated clinical measurements, while explicitly reporting when the data cannot distinguish a lipid-related factor from an inflammation- or severity-related factor?

The goal is not to assign a Raman band to a molecule, extracellular-vesicle subtype, or causal biological pathway. The intended claim is narrower and testable: a participant-level spectral representation has a stable, out-of-sample association with a measured clinical anchor, or the available anchors only support a joint, unresolved biological subspace.

## Why this is now plausible

The new Finnish workbooks materially change this direction. The 102 participants already present in `clinical_data.csv` join exactly to the large clinical workbook through the workbook's `Number in the primary large data base` field. For these linked participants, age, sex, admission and 24-hour NIHSS, pre-stroke/discharge/3-month mRS, vascular risk factors, treatment indicators, lipid measures, CRP, and APOE are available at high completeness. The workbooks also contain 50 controls with complete age and vascular-risk/medication fields, subject to confirming their exact linkage to the Raman cohort.

This creates measured biological anchors that are far stronger than post-hoc Raman-window labels. It also creates the central methodological difficulty: lipids, inflammation, stroke severity, treatment, age, and medication status are correlated. A model which always emits separately named factors would overstate what the data identify.

## Inputs

- Preprocessed fingerprint and CH-stretch Raman spectra, retaining the hierarchy participant -> sample -> area -> pixel/spot where available.
- Repeated-measurement metadata: participant, collection/visit label when verified, sample, area, spatial coordinate, spectral range, and quality-control flags.
- Linked stroke clinical data: age, sex, admission NIHSS, vascular comorbidities, acute treatment, chronic statin/antithrombotic exposure, total cholesterol, HDL, LDL, triglycerides, CRP, fibrinogen, and optional imaging summaries.
- Control covariates: age, sex, hypertension, diabetes, atrial fibrillation, coronary disease, smoking, and medication exposure.
- A separate, protocol-defined list of anchors for discovery and a held-out list for external validation. This split prevents the same variables from selecting and validating an interpretation.

## Proposed method

Call the method **RAISE: Reliability-Aware Interpretation of Spectral Embeddings**.

1. Estimate a spectral embedding at the participant level, while modelling within-participant variation instead of collapsing all areas to one median. A hierarchical encoder can produce both a participant mean representation and a measurement-reliability estimate from the number, dispersion, and spatial dependence of areas.
2. Learn a low-rank association operator between the reliable part of that embedding and a clinical-anchor block. The model should allow a spectral component to be shared by several anchors and should penalize unstable rotations of the latent space.
3. Fit across repeated participant bootstraps, area resamples, clinically defensible covariate adjustments, and alternative anchor sets. Each proposed factor receives a stability distribution rather than one loading vector.
4. Use a confidence-based factor-merging rule. If two candidate factors exchange clinical meaning across resamples or plausible covariate specifications, the output is one joint subspace labelled, for example, “lipid/inflammation-associated”, together with the reason it could not be separated.
5. Evaluate only on held-out participants. Report calibration for held-out clinical anchors, subspace overlap, factor-merge decisions, and uncertainty due to finite repeated measurements.

The key output is a three-level interpretation: **separate factor**, **shared/joint factor**, or **unresolved with the current data**. That abstention option is scientifically useful: it converts ambiguity from narrative discretion into a measurable result.

## Technical formulation to develop

Let `Z_i` denote a latent participant representation inferred from repeated Raman observations, and let `Y_i` contain clinical anchors. Learn an association matrix `B` under a measurement model for repeated areas:

`X_ij = f(Z_i, U_ij) + epsilon_ij`, with `Y_i = g(B Z_i, C_i) + eta_i`.

`C_i` contains prespecified confounders and design variables. A factor is separate only if its spectral direction, clinical loading pattern, and held-out prediction are stable under bootstrap resampling and under an allowed set of adjustment models. Otherwise, a data-driven merge produces an identifiable subspace rather than falsely distinct axes.

The first implementation can use a transparent linear/sparse model. A nonlinear encoder is justified only if simulations and held-out anchors show that it improves recovery without reducing stability.

## What is novel

Guided factor analysis, CCA, MOFA-like models, and concept-bottleneck models already connect high-dimensional observations to side information. The novel contribution must be the reliability and decision layer:

- repeated Raman areas inform the uncertainty of the participant representation;
- factor identity is assessed across rotations, resamples, covariate choices, and anchor choices;
- correlated anchors trigger an automatic, auditable merge rather than post-hoc factor naming;
- the method reports when scientific attribution is unsupported.

This responds directly to a limitation of SOFA: highly collinear guides can mask factors or produce interpretations that depend on whether a guide is included. The paper should demonstrate a lower rate of false biological factor naming under guide collinearity, not merely add guided factors to an existing model.

## Experiments

### Synthetic and semi-synthetic benchmark

Generate hierarchical spectra with known participant factors, uneven areas per participant, spatially correlated noise, correlated anchors, and controlled confounding. Vary the correlation between lipid and inflammation anchors. Evaluate:

- recovery of the true shared subspace;
- false separation of one factor into two named factors;
- false merging of truly separate factors;
- held-out anchor calibration;
- sensitivity to the number and quality of repeated areas.

### Real-data analysis

Use participant-level outer splits. In each training fold, define factor stability and merge thresholds without observing the held-out fold. Keep the spectral visit time explicit; do not frame a follow-up spectrum as an admission prognostic feature. Start with an adjustment ladder:

1. age and sex;
2. age, sex, and pre-event vascular/medication variables;
3. an exploratory severity/treatment model, labelled as post-event association rather than confounding control.

The primary endpoint should be held-out association with a small prespecified clinical-anchor set. A post-hoc heatmap across all available biomarkers is secondary and must use multiplicity control.

### Baselines

Compare against PCA plus regression, PLS/PLS-DA, regularized CCA, MOFA or a comparable multi-view factor method, sparse multi-task regression, and a guided factor baseline such as SOFA where feasible. Compare both predictive quality and the correctness of factor-attribution decisions in simulation.

## Claims the study can and cannot make

Supported target claim: “The method identifies a stable participant-level Raman subspace associated with measured lipid/inflammation/severity anchors, or correctly declares it unresolved.”

Do not claim molecular identification, extracellular-vesicle specificity, a direct biochemical concentration measurement, causal effects of stroke, or a genetic instrument. LDL may be calculated rather than directly measured, and CRP/fibrinogen are imperfect and time-dependent markers. APOE is a covariate or stratifier here, not an automatically valid instrumental variable.

## Main risks and ways to reduce them

- **Insufficient spectral–clinical linkage:** verify participant IDs and visit labels before modelling; do not infer them from row order.
- **No spectral visit time:** position the work as cross-sectional clinical anchoring, not prognosis.
- **Collinearity makes all factors unresolved:** this is a valid result. Use simulations to show that RAISE avoids overclaiming when separation is impossible.
- **Control laboratory values are absent:** make the primary analysis within the linked stroke cohort; use controls for a separate covariate-aware distributional analysis.
- **Small effective sample size:** constrain complexity, use repeated nested participant splits, and provide uncertainty intervals rather than leaderboard-style point estimates.

## Deliverables for a manuscript

- A model diagram showing repeated Raman observations, participant representation, clinical anchors, and the separate/shared/unresolved decision.
- Simulation figures that map guide correlation and repeated-measurement quality to false attribution and merge rates.
- A held-out clinical-anchor calibration figure.
- A factor-stability/merge matrix with uncertainty.
- Reproducible join, missingness, and leakage audits. No patient-level identifiers or values should appear in public artifacts.

## Suitable journals

The strongest target is **Pattern Recognition** if the manuscript provides a general method, simulations, open non-sensitive benchmark code, and a substantive application. **Information Fusion** is appropriate if the multimodal reliability and fusion contribution is central. **Artificial Intelligence in Medicine** is a good fit if methodological novelty is paired with a clinically careful application. **IEEE Journal of Biomedical and Health Informatics** is suitable if the paper focuses on robust processing and integration of biomedical signals with clinical data.

## Key literature to read before implementation

- Capraz et al., SOFA, *Nature Communications* (2026): guided factor analysis and the challenge of collinear guides. https://www.nature.com/articles/s41467-026-74694-6
- Leakage and Interpretability in Concept-Based Models, *JMLR* (2026): concept accuracy is not semantic purity. https://www.jmlr.org/papers/v27/25-1121.html
- Review of causal representation learning from multimodal biomedical observations (2025): conditions needed before causal language is warranted. https://pmc.ncbi.nlm.nih.gov/articles/PMC11952583/
- Serum Raman prediction of clinical analytes (2025): useful precedent, but it does not establish EV-specific molecular attribution. https://pubmed.ncbi.nlm.nih.gov/40095085/

