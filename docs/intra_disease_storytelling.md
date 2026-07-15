# Intra-Disease Raman Analysis: Storytelling Narrative

> Historical interpretation note: this narrative summarizes a specific
> disease-only run. Operational commands and current processing rules are in
> `README.md` and `docs/preprocessing.md`.

## Core Story

The study asked a biologically attractive question: within diseased subjects, do extracellular-vesicle Raman spectra carry a measurable signal of cognitive status or decline?

The corrected answer is sober but useful. The spectra contain visible exploratory structure, and several largest-effect windows repeatedly appear across clinical correlation and disease-internal contrast analyses. However, once the analysis is evaluated at the subject level with FDR, cluster-permutation correction, covariate adjustment, and permutation-tested multivariate models, the apparent signals do not reach robust statistical support.

This makes the manuscript a rigor story rather than a discovery story. The value is not "we found a Raman biomarker of decline." The value is "we built and stress-tested a conservative Raman workflow for intra-disease heterogeneity and found that the current cohort does not support molecular claims after correction."

## Narrative Arc

First, the workflow starts from a high-dimensional spectral problem. Raman spectra offer dense biochemical information, but subject-level clinical questions require careful aggregation, strict source control, and correction for multiple testing.

Second, the pipeline reduces the risk of overinterpretation. It separates exploratory area-level views from subject-level Biological PCA, applies cluster-based and FDR correction to spectral associations, tests multivariate models by permutation, and requires evidence to persist after covariate adjustment.

Third, the initial visual and fallback regions suggest where future hypotheses could focus. Some unstable regions overlap windows often interpreted as protein-, lipid-, nucleic-acid-, or mixed biochemical contributions. These overlaps are not findings; they are a map for future targeted testing.

Fourth, the corrected inferential results are negative. No candidate interval is FDR- or cluster-supported. PLS regression fails permutation testing, minimal panels show poor or non-significant predictive performance, and biochemical ratios do not survive FDR correction.

Finally, the study becomes a cautionary and constructive feasibility analysis. It shows that, in this dataset, intra-disease Raman heterogeneity cannot yet be translated into clinically meaningful molecular statements. It also defines what a stronger next study needs: larger sample size, pre-specified spectral windows, external validation, and power-aware modeling.

## Suggested Results Message

"Although exploratory largest-effect regions were observed across both fingerprint and high-wavenumber spectral ranges, none survived correction for multiple spectral testing. Integrated evidence classification therefore assigned no robust or exploratory candidate regions. Multivariate PLS and minimal-panel models failed permutation testing, and targeted biochemical ratios showed no FDR-significant clinical associations. These results indicate that the current cohort does not provide sufficient evidence for Raman spectral correlates of cognitive decline, but identifies candidate windows for pre-specified validation."

## Suggested Discussion Message

"The absence of corrected associations should not be interpreted as absence of Raman-relevant biology. Rather, it indicates that any disease-internal biochemical signal is smaller, more heterogeneous, or less aligned with the available clinical endpoints than the present cohort can robustly resolve. The analysis therefore argues for caution: candidate Raman shifts should be presented as exploratory spectral regions, not molecular mechanisms."

## Claim Boundaries

Supported:

- A conservative subject-level Raman workflow was implemented.
- No corrected spectral associations with cognitive endpoints were detected.
- Exploratory largest-effect regions can guide future pre-specified hypotheses.
- The present sample is insufficient for validated molecular or predictive claims.

Not supported:

- Cognitive decline is significantly correlated with specific Raman shifts.
- Protein, lipid, or nucleic-acid signals explain decline.
- Raman spectra predict decline in this cohort.
- Disease subgroups identified here are validated biological subtypes.
