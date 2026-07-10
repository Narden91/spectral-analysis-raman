# Intra-Disease Raman Analysis: Scientific Review

## Executive Assessment

The corrected intra-disease pipeline is scientifically conservative and now supports a null/feasibility interpretation. It does not support a claim that clinical decline is significantly correlated with specific Raman shifts or molecular groups in this cohort.

The strongest defensible conclusion is that the analysis searched for subject-level Raman spectral associations with baseline MoCA, 3-month MoCA, and delta MoCA, but no spectral intervals survived FDR or cluster-permutation correction. Candidate intervals remain useful as hypothesis-generating regions for future studies, not as validated disease-progression signals, biomarkers, predictors, or mechanistic biochemical evidence.

## Corrected Evidence Summary

The corrected Phase 6 provenance reports 295 candidate intervals, with 0 FDR- or cluster-supported intervals and 295 fallback-selected intervals. Phase 14 reports 0 robust, 0 exploratory, 38 unstable, and 0 confounded integrated candidate regions.

PLS regression did not pass permutation testing for any continuous endpoint. Cross-validated R2 values were negative for baseline MoCA, 3-month MoCA, and delta MoCA, with permutation p-values of 0.999, 0.341, and 0.983. Therefore, multivariate continuous modeling performs worse than a baseline predictor and should not be interpreted as predictive.

The minimal candidate panel also failed to provide reliable models. Continuous endpoint R2 values were negative, and binary balanced accuracy was weak: 0.559 for MoCA severity and 0.442 for delta MoCA group, with permutation p-values of 0.157 and 0.847. These values do not justify classification or prognostic language.

Targeted biochemical ratio testing did not identify any FDR-significant associations. The smallest observed uncorrected signal was the delta MoCA association with amide_iii_1260 over amide_i_1655, but it had FDR q = 0.862 and must be treated as non-significant.

Disease subgrouping selected 4 clusters with mean bootstrap ARI around 0.52. This is moderate exploratory structure, not a validated disease subtype solution.

## Code Defects Corrected

Several implementation issues previously inflated the apparent evidence level:

- Fallback candidate regions were written under a filename that implied correction support. The pipeline now propagates `selection_basis` and `significance_supported`, and writes `candidate_region_provenance.md`.
- Covariate persistence was previously equivalent to model success. It now requires an adjusted spectral coefficient p-value/FDR result.
- The outcome variable is now removed from covariates, preventing invalid models such as age predicted by a region while adjusting for age.
- PLS VIP coverage is now gated on PLS permutation significance, so failed PLS models cannot contribute evidence.
- Phase 9 now writes empty gated-skip artifacts when PLS fails, preventing stale PLS-L1DA selected-band files from contaminating Phase 11.
- The publication report now honors configured robust requirements and explicitly states when all candidate regions are fallback-selected.

## Scientific Rigor

The strongest aspects of the workflow are subject-level aggregation, explicit disease-only routing, cluster-permutation correction, FDR control, permutation testing for multivariate models, and conservative integrated reporting. These choices are appropriate for a high-dimensional Raman dataset with limited subject-level sample size.

The main limitation is statistical power under a high feature-to-subject ratio. The data can generate plausible spectral hypotheses, but the corrected inferential framework does not show reproducible clinical associations. Any biochemical assignment should therefore be framed as "regions overlapping tentative molecular bands" rather than evidence of molecular mechanisms.

## Recommended Paper Framing

Frame the manuscript as a feasibility and negative-results study:

"A subject-level Raman spectroscopy workflow for intra-disease heterogeneity analysis found no corrected associations between extracellular-vesicle Raman spectra and cognitive outcomes, despite exploratory largest-effect regions overlapping protein/lipid-associated windows. These results emphasize the need for larger cohorts, pre-specified spectral panels, and independent validation before molecular or prognostic interpretation."

Avoid claims that:

- specific Raman shifts are correlated with decline;
- specific molecular groups explain cognitive change;
- the pipeline predicts prognosis or detects disease subtypes;
- fallback candidate regions are biomarkers or validated signatures.

## Next Scientific Improvements

The next study should pre-register a small spectral-region/ratio panel, use an independent validation cohort, report power or detectable-effect calculations, and consider mixed-effects modeling if repeated measurements or acquisition batches are available. The current results are valuable because they identify where the evidence is weak, not because they establish a positive molecular story.

## Scientific-rigor contract

- The current fixed panel is `post_hoc_fixed`, not independently pre-registered for this cohort.
- Same-cohort covariate, VIP, sparse-weight, PCA-overlap, and biochemical annotations do not provide independent validation.
- Whole-spectrum cluster inference is primary; channelwise FDR is descriptive and cross-scan cluster p-values receive Holm adjustment.
- `PCA_whole` uses MFA first-eigenvalue block scaling; unscaled concatenation is sensitivity-only.
- Baseline-corrected spectra retain signed residuals. QC SNR is computed after baseline correction and before smoothing; no batch-relative area percentile is used.
- Missing clinical outcomes are handled by complete-case analysis with inclusion comparisons. Results may remain selection-biased.
