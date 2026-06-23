# Draft Manuscript Sections: Intra-Disease Raman Analysis

## Methods

Raman spectral analyses were performed using a subject-level exploratory workflow designed to evaluate intra-disease heterogeneity while limiting overinterpretation of high-dimensional spectral data. Preprocessed extracellular-vesicle Raman spectra were loaded from the canonical preprocessed data store. Analyses were routed through the `disease_only` mode, which writes outputs to `Results_intra_disease/` and excludes control-vs-disease comparisons from the intra-disease inference layer.

Spectra were analyzed in the fingerprint and high-wavenumber ranges and then combined for whole-spectrum subject-level analyses. The maintained Biological PCA layer was restricted to subject-level summaries and generated three models: `PCA_1330`, `PCA_2850`, and `PCA_whole`. Whole-spectrum PCA used unweighted concatenation of range-local median spectra while preserving spectral block identity in feature names and loadings.

Disease-only cross-region analysis was conducted across clinical correlation matrices, longitudinal delta MoCA analyses, and disease-internal stratification tests. Candidate spectral intervals were defined from subject-level effects and evaluated using FDR and cluster-mass permutation correction. When no corrected regions were detected, largest-effect fallback intervals were retained only as transparent exploratory context. The corrected pipeline records this distinction through `selection_basis` and `significance_supported` fields.

Covariate-adjusted models were fitted for candidate regions and biochemical ratios using age and sex when available, excluding any covariate identical to the modeled outcome. Adjusted spectral coefficients were assigned p-values, FDR-adjusted q-values, and covariate persistence status only when the adjusted spectral coefficient remained statistically supported.

Multivariate continuous associations were assessed with PLS1 regression and permutation testing. PLS VIP coverage was allowed to contribute to integrated evidence only if the corresponding PLS target passed the permutation gate. PLS-L1DA exploratory classification was similarly gated on PLS permutation significance; when no PLS model passed, classifier outputs were written as empty skip artifacts to prevent stale selected-band files from entering integrated evidence.

Targeted biochemical ratios were computed from pre-specified band integrals and assessed against clinical endpoints using Spearman association tests with FDR correction. Minimal candidate panels were evaluated with nested cross-validated ElasticNet or L1-penalized logistic models and permutation testing. Disease subgroup structure was assessed by exploratory clustering and bootstrap adjusted Rand index.

## Results

The corrected candidate-region detector generated 295 candidate intervals. None were FDR- or cluster-permutation-supported, and all 295 were fallback-selected largest-effect intervals. Integrated evidence classification therefore identified 0 robust regions, 0 exploratory regions, 38 unstable regions, and 0 confounded regions. The 38 unstable regions included 15 fingerprint-range and 23 high-wavenumber-range regions, with median maximum absolute effect of 0.283.

Continuous PLS regression did not identify permutation-supported clinical associations. Cross-validated R2 values were negative for baseline MoCA (R2 = -0.445, permutation p = 0.999, n = 72), 3-month MoCA (R2 = -0.056, permutation p = 0.341, n = 92), and delta MoCA (R2 = -0.370, permutation p = 0.983, n = 71). Because no PLS model passed the permutation gate, PLS VIP features and PLS-L1DA selected bands were not treated as supporting evidence.

The minimal candidate panel did not yield reliable prediction or classification. Continuous models had negative cross-validated R2 for baseline MoCA (R2 = -0.207, permutation p = 0.983), 3-month MoCA (R2 = -0.050, permutation p = 0.589), and delta MoCA (R2 = -0.409, permutation p = 0.988). Binary models showed weak performance, with balanced accuracy of 0.559 for MoCA severity group (permutation p = 0.157) and 0.442 for delta MoCA group (permutation p = 0.847).

Targeted biochemical ratio analysis did not identify FDR-significant associations. The smallest uncorrected ratio association involved delta MoCA and the amide III / amide I ratio, but this did not survive correction (p = 0.0479, FDR q = 0.862). Therefore, targeted protein/lipid/nucleic-acid band interpretations remain unsupported in this cohort.

Exploratory disease subgrouping selected four clusters with mean bootstrap adjusted Rand index of 0.517 (SD = 0.153). This suggests moderate exploratory structure but does not establish stable disease subtypes.

## Discussion

This intra-disease Raman analysis did not identify statistically robust spectral correlates of cognitive status or decline. Although largest-effect fallback regions were observed across both spectral ranges, none survived FDR or cluster-permutation correction. Multivariate regression, minimal-panel modeling, and targeted biochemical ratio testing also failed permutation or FDR thresholds. The corrected evidence therefore supports a feasibility and negative-results interpretation rather than a discovery claim.

The findings should be interpreted cautiously. Raman spectra may still contain disease-relevant biochemical information, but the present cohort does not resolve a reproducible subject-level association with the available cognitive endpoints. Candidate regions overlapping commonly interpreted protein, lipid, or mixed biochemical windows should be treated as hypotheses for future pre-specified testing, not as evidence that specific molecular groups drive decline.

The main strength of the workflow is its conservative design. It uses subject-level aggregation, explicit disease-only routing, corrected spectral inference, permutation-tested multivariate models, covariate-adjusted coefficient testing, and integrated reporting that separates robust evidence from fallback context. This reduces the risk of converting high-dimensional noise into biochemical narratives.

The main limitation is statistical power in a high-dimensional setting. The number of spectral features greatly exceeds the number of subjects with complete endpoint data, and clinical endpoints may be heterogeneous, noisy, or weakly aligned with extracellular-vesicle Raman variation. The repeated liblinear convergence warnings observed during minimal-panel fitting further support caution for sparse classification claims.

Future work should use a larger independent cohort, pre-specified spectral windows or biochemical ratios, power-aware effect-size targets, and external validation. If longitudinal spectra become available, mixed-effects models could separate within-subject change from between-subject heterogeneity. Under the current evidence, the appropriate conclusion is that this pipeline provides a rigorous framework and a transparent null result, not a validated Raman signature of cognitive decline.
