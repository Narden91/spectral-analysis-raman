# Agent learnings

## 2026-10-02: Windows test temporary-directory permissions

- Failure: pytest could not access the existing system temporary test directory or its cache; the first run reported setup errors instead of checking those tests.
- Prevention: run with a fresh, uniquely named `--basetemp` inside the repository and `-p no:cacheprovider` when those paths are inaccessible. Do not reuse an existing directory because pytest clears its base temporary directory.

## 2026-10-02: Empty signatures in scripted Python edits

- Failure: adding a parameter with an unconditional comma generated an invalid empty-argument signature. Parsing caught it before any source file was written.
- Prevention: handle empty argument lists separately and parse the complete rewritten Python source before writing it.

## 2026-10-02: Graft CLI wrapper on PowerShell

- Failure: `graft build` selected `graft.ps1`, which the Windows execution policy blocked.
- Prevention: use the installed `graft.cmd` wrapper for CLI builds; keep PowerShell execution policy unchanged.

## 2026-10-02: Test paths and Windows source encoding

- Failure: a guessed test filename stopped a focused pytest run before any tests ran; printing a source file's UTF-8 BOM also failed under the shell's default encoding.
- Prevention: choose test paths from the repository inventory. For Python source inspection, read with `utf-8-sig` and set stdout to UTF-8; preserve the file's original BOM and line endings when editing.

## 2026-10-20: `sed -i` on CRLF files in Git Bash

- Failure: a `sed -i` substitution on `scripts/generate_manuscript_figures.py` rewrote the whole file with LF endings; a pattern anchored on `\r$` also failed to match.
- Prevention: edit CRLF files with the Edit tool or a Python byte-level rewrite that restores `\r\n`; check `file <path>` after any shell edit.

## 2026-11-06: Research evidence and decision history

The active proposal is [Learning reproducible Raman features to test circulating lipid information](docs/research_design.md). The earlier brainstorming and extension files were compacted here before removal. Their names below are historical identifiers, not links to active proposals. Reconsider an idea only when its stated evidence requirement changes.

### Evidence snapshot

| Evidence | Status and implication |
|---|---|
| Spectral membership: 97 stroke participants and 50 controls. | Aggregate counts observed in `data/clinical_master.csv`; these are maximum counts before linkage and spectral eligibility checks. The master also contains clinical-only records. |
| Five stroke records have identity-review status. | Excluding recorded identity concerns gives 92 provisionally linkable stroke participants, including 90 with two sample labels and two with one. A matched status alone does not establish identity: inspect flags as well. |
| Control age is recorded for 47 spectral participants: 37 women and 10 men. | Aggregate counts observed. An age-range/no-anticoagulant screen among provisionally linked stroke participants retains 53, replacing the earlier draft count of 58. This screen does not establish multivariable overlap. |
| LDL-C, HDL-C, TG and platelet values are populated for the 92 provisionally linked stroke participants; control lipid values are absent. | Aggregate completeness observed. Lipid units are marked assumed in the dictionary. Collection timing, fasting status and measured-versus-calculated LDL-C need source confirmation; populated values are not validated measurements of the isolated fraction. |
| Paired samples were recorded as technical replicates of one collection. | The 21 October proposal explicitly recorded investigator confirmation. The manuscript describes longitudinal collections. Preserve the confirmation, correct the documentation conflict, and identify exactly which preparation/acquisition stages were independently repeated. Do not turn this conflict into a default assumption that the pairs are separate visits. |
| The clinical notes state that processed stores were not rebuilt after the standalone master was created. | Existing results may use older clinical joins and sex-folder labels. Do not describe them as reruns with the new clinical information. The builder also has existing uncommitted edits: a saved CSV need not reflect the current builder. |
| Saved whole-spectrum PLS-DA AUC is 0.711, with reported interval 0.550–0.823. | Inspected saved report, not rerun evidence. It establishes neither disease specificity nor freedom from preparation differences. |
| Saved MoCA prediction R² values are approximately -0.335, -0.067 and -0.403. | Inspected saved results for baseline, three months and change; no model passes its stored adjusted permutation gate. Historical candidate counts differ across runs and must not be mixed. |
| A pilot reported 14.2% identification among 95 participants and 25.3% with local candidate sets, compared with stated chance levels 1.1% and 9.1%. | Reported in the former extension 07; the script was absent and results were not reproduced. Candidate-set construction and direction averaging require reconstruction. These percentages are not reliability estimates. |
| Sample labels are associated with cohort blocks; local-candidate identification was reported below random-candidate identification. | Documentary evidence and an unreproduced pilot. The physical meaning of sample number remains unknown. Use “sample-label-associated structure” until processing/session records support a stronger description. |
| The processed spectral directory was inaccessible during review. | Pixel counts, pilot spectral calculations and agreement estimates were not independently checked. No new spectrum-versus-outcome analysis, bridge-control analysis or method experiment was run. |

Local evidence sources: `data/clinical_master.csv`, `data/clinical_master_dictionary.csv`, [clinical source notes](data/clinical_master_notes.md), [saved classification report](Results_control_vs_disease/05_ml_classification/06_report/final_performance_summary.md), [saved cognitive prediction table](Results_intra_disease/classification/pls_regression/pls_regression_performance.csv), and [scientific review](docs/intra_disease_scientific_review.md). These private analysis paths are provenance, not a release bundle.

### Compact history of all earlier ideas

| Earlier proposal | Question and method worth preserving | Decision, limitation and condition for reconsideration |
|---|---|---|
| 01 — Reliable clinical attribution; working name RAISE. | Connect repeated spectral representations to correlated laboratory/clinical anchors. Assess stability across participant resamples, anchor sets and adjustment models; return separate, shared or unresolved associations rather than always naming factors. | Retain uncertainty about attribution in the active design. Stable loadings do not identify a biological source. A standalone method needs simulation evidence of fewer incorrect attribution decisions than regularized regression/CCA and guided factors such as SOFA. Do not describe another same-cohort anchor as external validation. |
| 02 — Support-preserving multimodal learning; working name SPAR. | Train a clinical predictor on a larger cohort and allow Raman to redistribute probability only within outcome categories represented by Raman. For supported set A, use p1(k)=p0(A)q1(k given A) inside A and p1(k)=p0(k) outside A. | Park. The pointwise probability invariant is valid by construction; for an observed outcome outside A, the log-score change is zero. Brier and ranked-probability scores do not inherit that cancellation. Selection within A can still be biased. Revisit with authorized larger-cohort access, verified prediction timing, and public masked-modality benchmarks. Current Raman-linked mRS categories do not support claims about categories 5–6. |
| 03 — Conditional modality testing; working name BAG-COMET. | Test spectral association beyond measured covariates using participant distribution features from unequal, dependent measurement sets, cross-fitted nuisance models and explicit overlap restrictions. | Park as a broad formulation. Existing conditional tests and set-valued methods already cover much of the machinery. Neither a finite feature set nor a few repeats establishes recovery of an entire latent biological distribution. Revisit only with a precise estimand and demonstrated calibration advantage. |
| 04 — Conditional testing with unequal technical replication. | Refine 03 to first-order map features and within-sample distinct-map products, averaged equally across samples and participants. Use a residualized conditional score and participant multipliers; compare exact pair calculations with an efficient kernel form. | Preserve as the strongest alternative methods direction, not a second active project. It has no completed implementation or proved calibration result. A squared mean contains a measurement-count term; distinct-map products remove that term only under the stated independence and selection assumptions. They retain shared preparation effects. Revisit with a justified calibration result, simulations and a useful advantage over matched replication plus established conditional tests. |
| 05 — Reference-free lipid-associated decomposition. | Learn smooth components from spectra, validate their association with held-out stroke lipid values, and examine the case-control contrast before/after removing a lipid-associated subspace. Compare with PCA/ridge, PLS and the curated band panel. | Retain lipid anchors and a possible decomposition comparator. Do not name LDL/HDL/VLDL components or quantify particle contributions without suitable reference/orthogonal evidence. Subspace removal can remove disease-associated variance and is not a causal decomposition. Revisit only if reconstruction, stability and held-out anchor prediction improve over simple baselines; any experimental input basis must be explicit. |
| 06 — Replication-aware normative deviation. | Fit a small age/sex control reference, account for sample/map uncertainty, and relate direction-independent deviation to clinical phenotype. Use matched replication and sensitivity to borrowed slide variance. | Park. Only 47 age-recorded controls and 10 male controls limit calibration and covariate support. Control slide variance cannot be separated from participant variance with one slide without extra assumptions. Conservativeness of first-order scores is model-dependent. Second moments are not automatically between-map heterogeneity. Revisit with a larger suitable reference, technical replication in controls, or a demonstrated robust aggregate estimator. |
| 07 — Replicate-contrastive attribution. | Learn a shared representation from paired samples, compare local versus random candidate identification, then assess agreement and held-out laboratory associations. Linear generalized-eigenvector learning precedes an optional masked-spectrum/contrastive encoder. | Selected as the starting point, narrowed in the active design. Local sample labels are not verified batches. Pairing can preserve shared technical artefacts. A retrieval rate cannot explain absent clinical prediction or replace agreement analysis. Revisit a deep model only after a reproducible linear baseline and enough independent information justify it. |
| 08 — Disease versus processing, followed by conformal deviation. | Audit technical predictors and seven reported late-block controls; examine sample-label drift and common support before constructing control-reference deviations. | Retain the audit as a condition for case-control interpretation. Drop individual conformal selection from the active design. A small bridge comparison cannot establish absence of preparation effects. Weighting cannot repair unmeasured preparation differences; effective sample size does not determine a weighted p-value floor. Revisit only with verified sessions, usable overlap and a separately justified calibration/multiple-testing procedure. |
| Email draft — two-idea pitch. | Summarized 07 and 08 and requested clarification of sample numbering, repeat meaning, a bridge-control check and an optional PyTorch dependency. | Unsent draft, not evidence of answers or approvals. Its research content is preserved above. Neither analysis execution nor a dependency addition follows from that draft. |

### Mathematical details that should not be rediscovered

- With independent, zero-mean errors, E[mean(Y)^2 given theta] = theta² + sigma²/m. The average of distinct-measurement products targets theta² under that model. For theta variance 0.25 and error variance 1, the respective squared-mean expectations are approximately 0.583 at three measurements and 0.417 at six. These are model calculations, not cohort results.
- Form distinct-map products within each technical sample before averaging samples equally. Pooling all six stroke maps as if they were independent repeats of the same level changes the target. Correlated maps and shared slide effects prevent interpreting the products as pure participant biology.
- The conditional score considered in 04 was (D-e(C))(F-f(C)) tensor h(C), targeting weighted conditional mean contrasts of specified features. Its candidate nuisance-error product condition and multiplier calibration were research requirements, not established guarantees. Independent sample division and overlapping cross-fitting require different justification.
- The SPAR probability invariant concerns each unsupported category and log score for an unsupported observed outcome. It does not establish full-population benefit or correct outcome-dependent selection.
- For nonrandomized unweighted conformal ranks with 47 exchangeable reference observations, the smallest p-value is 1/48. This calculation does not transfer directly to weighted ranks, an effective sample size, dependent multiple testing, or a disease interpretation.
- Under classical independent measurement errors, observed Pearson correlation is attenuated by the square root of the product of the two reliabilities. Unknown outcome reliability, correlated errors and rank correlations prevent automatic correction. Retrieval accuracy is not that reliability coefficient.

### Prior work and limits of novelty

| Primary source checked in the review | What it rules out or motivates |
|---|---|
| [Huber et al., 2021](https://pubmed.ncbi.nlm.nih.gov/33686065/) — primary abstract/record. | Person-specific infrared blood fingerprints already exist; this does not establish Raman repeatability or lipid-source identity in this cohort. |
| [Li et al., 2022](https://arxiv.org/abs/2202.12549) — primary abstract/record. | Contrastive Raman matching is established. |
| [Bridgeford et al., 2021](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009279) — primary methods inspected. | Discriminability and its comparison procedures are established comparators, distinct from an ICC. |
| [Bonizzi et al., 2025](https://doi.org/10.1016/j.bios.2025.117287) — primary full text inspected. | Raman extracellular-particle profiling and comparison with fraction biochemical assays already exist. Plasma clinical lipid values are a different measurement target. |
| [Georgiev et al., 2024](https://arxiv.org/abs/2403.04526) — primary record. | Physics-constrained Raman unmixing is established. Constraints alone do not make a new algorithm or identify particle sources. |
| [Capraz et al., 2026, SOFA](https://www.nature.com/articles/s41467-026-74694-6) — primary discussion/record. | Guided factors exist; correlated guides can mask or alter factor interpretation. Test attribution decisions against this problem instead of claiming guided learning itself is new. |
| [Bellot and van der Schaar, 2021](https://proceedings.mlr.press/v161/bellot21a.html), [Kook and Lundborg, 2024](https://arxiv.org/abs/2402.14416) — primary records/methods. | Unequal-set testing and conditional modality testing are established. |
| [MISEV2023, published 2024](https://doi.org/10.1002/jev2.12404). | Co-isolated particles limit attribution to EVs; lipid association does not establish EV specificity or brain origin. |
| [Weighted conformal prediction](https://arxiv.org/abs/1904.06019) and [Jin and Candès](https://arxiv.org/abs/2307.09291) — primary records. | Distributional assumptions and the exact selection procedure matter; an arbitrary weighting/FDR combination has no automatic guarantee. |

Additional adjacent work found: [serum Raman analyte prediction, 2025](https://pubmed.ncbi.nlm.nih.gov/40095085/), [replicate-held-out bacterial Raman classification, 2026 preprint](https://arxiv.org/abs/2606.27096), and [AMD extracellular-particle multi-omics/Raman, 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13525867/). A targeted search is not a systematic priority review. The [UCI Parkinson dataset](https://archive.ics.uci.edu/dataset/470/parkinson%2Bs%2Bdisease%2Bclassification) documents participant-level triplicate voice measurements and is a possible general-method benchmark, not external validation of stroke Raman biology. Musk Version 2 conformations are finite sets, not verified independent physical replicates.

### Prevention rules from the review

- Never infer identity from row order, change an uncertain unit or decimal, repair an inconsistent medication flag, or invent collection timing. Keep uncertain records and source conflicts visible.
- Do not call sample number physical processing order or call adjacent labels the same batch without records. A nuisance-only predictor reveals structure; it does not identify its cause or recover absent overlap.
- Do not reuse fallback candidate regions, failed-model VIPs or same-cohort annotations as independent confirmation. Keep the provenance and inference gates described in the scientific review.
- Match complexity to independent participants. A large pixel count cannot justify a large neural model or narrow participant-level intervals.
- Freeze representation selection and association testing inside participant-level training divisions. Repeated folds are dependent and do not create new cohorts. All copies of a resampled participant remain in the same division.
- Keep controls outside the primary lipid-anchor analysis because they lack laboratory values. A successful within-stroke result cannot resolve the case-control preparation concern.
- Separate discovery from source identification, prognosis and causation. No diagnostic threshold, EV-specific source claim, brain-origin claim or biological concentration estimate follows from these designs.
- A failure to detect an association is not proof of absence and is not automatically a publishable Q1 contribution. The method needs a demonstrated benefit over established baselines and a clear biological question.
- Preserve superseded ideas compactly before deleting drafts. Keep source verification limits attached to the evidence and retain mathematical restrictions with the idea, rather than preserving only its title.

- Vocabulary cleanup must distinguish an individual permuted statistic from its reference distribution and include plural forms. Check mathematical sentences and captions manually; compare non-text syntax trees before and after source-text edits.

## 2026-11-06: Research-design refinement and evidence update

The active question remains whether paired technical-sample Raman representations improve held-out LDL-C prediction in the eligible stroke cohort. The earlier version combined this question with proposed categorical anchor attribution and treated the paired eigensolver as a potential methods contribution. Neither the representation, attribution rule nor uncertainty procedure had been implemented or validated. The revision retains the question and linear starting point, narrows the predictive claim to specified comparators, and makes attribution conditional on separate development evidence. It does not reopen the parked proposals.

### Additional evidence and verification limits

| Evidence checked on 6 November 2026 | Consequence for the active design |
|---|---|
| [Silveira et al., 2017](https://pubmed.ncbi.nlm.nih.gov/28271376/), primary abstract: Raman/PLS modelling of LDL and other serum analytes in 44 samples. | Lipid prediction itself is established. Serum results do not validate the present fraction, repeat hierarchy or incremental comparison. |
| [Green et al., 1988](https://doi.org/10.1109/36.3001), indexed primary text: noise-fraction transformations order spectral components using signal/noise properties. | Treat the paired generalized eigenproblem as an adaptation of an established family. Its particular regularization and evaluation procedure still need validation. |
| [Bonizzi et al., 2025](https://doi.org/10.1016/j.bios.2025.117287), primary full text, compared with [the local preprocessing specification](docs/preprocessing.md). | The cited 698 and 718 cm⁻¹ features lie below the canonical 734 cm⁻¹ fingerprint bound. The study supports plausibility, not direct transfer of those peak measurements; normalized shapes cannot supply absolute fraction yield. |
| [MISEV2023](https://doi.org/10.1002/jev2.12404), primary full text. | Record available preparation/acquisition metadata and stages independently repeated. Clinical platelet count is not a residual-platelet assay, and missing metadata do not establish purity. |
| [Bates et al.](https://arxiv.org/abs/2104.00673), primary abstract, and [Bengio and Grandvalet](https://www.jmlr.org/papers/v5/grandvalet04a.html), primary abstract. | Cross-validation targets and dependent errors need explicit treatment. A full-procedure bootstrap is a candidate to validate, not an automatic calibrated interval. |
| [ICC guidance](https://doi.org/10.1016/j.jcm.2016.02.012), primary record, and [its correction](https://pmc.ncbi.nlm.nih.gov/articles/PMC5731844/), correction text. | State model, measurement type and agreement definition. Shared covariates can make full predictions agree without a reproducible spectral contribution; evaluate that contribution separately. |
| Graft source audit: `src/analysis/biological_pca.py:305`, `joint_dr.py:41`, `minimal_panel.py:139`, `pls_regression.py:80`, and `subject_features.py:62`. | Existing exports collapse sample pairs; MFA is unweighted; feature pools include cohort-derived regions; inner PLS tuning does not group bootstrap copies; existing exports do not implement the proposed master-table join. The design specifies new work rather than claiming these capabilities already exist. |

This was a targeted literature and implementation review, not a systematic novelty search. The earlier aggregate cohort counts were carried forward without recalculation. No participant-level records or spectral store were inspected, and no cohort experiment or calibration simulation was run. Numerical grids, simulation counts and feasibility tolerances in the design are planning choices, not evidence of achieved performance.

### Additional prevention rules

- Describe improved prediction relative to the named covariate procedure. Spectra may proxy nonlinear covariate effects; the comparison does not establish general conditional association or removal of technical confounding.
- Separate full-prediction agreement from agreement of the spectral contribution. Declare single-sample versus averaged-sample reliability and do not pool ordinary ICCs across independently fitted models.
- Interpret the pair-difference second moment as disagreement unless its error assumptions hold; systematic A/B shifts contribute to it. Compare subspaces only after mapping them to common spectral coordinates and a declared metric.
- Audit feature provenance before calling a comparator fixed. Training-only fitting cannot undo earlier feature selection using the complete cohort.
- Keep comparison families, uncertainty targets, failure rules and simulation validation explicit. Attribution categories remain developmental until their statistics and operating characteristics are specified and checked.

## 2026-11-08: Equivalence claims for fast numerical paths

- Failure: a batched GPU bootstrap was reported to match the sequential refit to 6.6e-16, but only the unresampled base draw was compared, and the sequential function itself used the batched engine for inner selection. Resampled draws differed by up to 3e-2 (a tau=tr(W) bug and a mixed-precision eigensolver).
- Prevention: test fast-vs-reference equivalence on several resampled draws, against a reference path that shares no code with the fast path (here include_band_paired=True), on every device the fast path supports.

## 2026-11-09: Cohort join never ran end to end

- Failure: the readiness audit passed on file presence and a clinical count, but the cohort join returned 0 participants. `not_available` made pandas read `spectral_subject_id` as float, so "7.0" never matched store id "7". All tests used synthetic cohorts.
- Prevention: read identifier columns with an explicit string dtype. Before calling a pipeline experiment-ready, run the real-data join as a counts-only dry run and gate on the expected counts.

## 2026-11-10: Dead-code scans miss indirect loading

- Failure: an import-and-name scan marked `src/cli_util.py` as unreachable. Three scripts load it by file path with `importlib.util.spec_from_file_location`, so no `import` line names it. A shell edit through a Bash heredoc also collapsed `\\` to `\` and silently skipped a fix.
- Prevention: before deleting a module or function marked dead, grep for its file name and for `spec_from_file_location`, `importlib`, `getattr`, and string dispatch. Check `git diff` after any scripted byte-level edit.
- Config-key removal is not a free cleanup. `config_hash(cfg_in)` in `control_vs_disease_ml.py` hashes the whole analysis config into the `--resume-ml` checkpoint key, and any `preprocessing:` key change makes the preprocessed store stale. Batch such edits and rebuild once.

## 2026-11-10: Dummy-outcome checks of the EDA pipeline

- Failure: a dummy clinical CSV alone did not hide the real outcomes. `data/preprocessed/subjects.parquet` already holds real moca, delta_moca, aspects and rbans columns, and `subject_features._clinical_table` drops the CSV columns that overlap with the store, so the store wins. Case-control labels also enter at three places (store tables, `_single_range` Phase 3, the Biological PCA feature matrix), not only at `control_vs_disease_ml.load_subject_feature_matrix`.
- Prevention: patch outcomes and labels at `loaders._cached_table` (one patch for all store tables), permute once per subject within sex, and assert in each consumer that it saw the dummy values. Set `PYTHONUTF8=1 PYTHONIOENCODING=utf-8` for redirected runs, because the rich spinner raises `UnicodeEncodeError` otherwise.
- Floating-point noise in the `PCA_whole` outputs is about 1e-14 between identical runs. Compare against a before-versus-before spread, not against exact equality.

## 2026-11-09: Clinical source and verification data

- `data/clinical_data.csv` is deprecated. The clinical sources are `data/clinical_master.csv` and `data/clinical_master_dictionary.csv`. The paired-lipid code never reads `clinical_data.csv`, so a strict store-validation error about it does not block manuscript ML runs.
- The investigator requires code checks on real data only. Synthetic runs are not accepted as verification evidence. To check code without producing a study result, use the real store spectra, covariates, sample labels and folds, and replace the outcomes with seeded dummy values before any model sees them.

## 2026-11-09: Verified facts about the manuscript ML pipeline (do not re-test)

Checked on real data at commit `12caecf` plus the uncommitted 2026-11-09 review changes. Rerun a check only when the store manifest hash (`b1c1016e…69c3`), `clinical_master.csv`, or the named code changes.

| Fact | Evidence |
|---|---|
| Real cohort join: 92 eligible participants, 88 paired, 4 single-sample, 180 samples. | Readiness audit `cohort_join.status=ok`. Exclusions: 5 `no_spectra`, 5 `identity_review_required`, 2 `fewer_than_2_maps` (participant/sample 504/114 and 738/160). |
| QC-rejected maps never reach the area summaries. | 721 (1300) and 717 (2850) summary rows; 0 all-NaN and 0 partially NaN rows. |
| Covariates and outcomes are complete in the 92 eligible participants. | 0 missing in age, sex, statin, sample-label trend, the 7 risk factors, LDL-C, HDL-C, TG and platelets. Coding: sex `female`/`male`, smoking `never`/`former`/`current`, binaries 0/1. Every covariate preset is eligible for the fast path. |
| Fast inner selection equals the sequential reference. | Dummy outcome: identical selected settings and predictions for all 6 models. |
| Batched bootstrap equals sequential refits. | Base draw plus 20 multiplicity draws: max difference 3.6e-14 (CPU) and 3.1e-14 (GPU, exact fp64). Bootstrap base draw versus nested-CV point estimate: 1.5e-15. |
| GPU bootstrap cost, 2000 draws, exact fp64, 15.9 GiB GPU. | Before: 72.1 s, 10.3 GiB peak pool (real data). The `eigh` share before the change (92%) was profiled on a synthetic cohort only; on real data after the change `eigh` is still 85% of 37.3 s, so a faster exact top-k solver is the next speed lever. After the active-sample `eigh` and in-place gram updates: 38.5 s and 3.79 GiB (batch 2048), 40.2 s and 1.95 GiB (1024), 48.7 s and 0.98 GiB (512). The in-place change is bitwise identical; the active-sample change moved draws by at most 8.2e-14. |
| Calibration gate needs at least 48 replicates. | At 100% coverage the Wilson 95% lower bound is n/(n+z^2): 0.9259 at 48 and 0.9244 at 47. With 1000 replicates, 950 covered gives 0.935 and 940 gives 0.924. `check_gate_attainable` refuses a validation run below 48. |
| The full test suite passes. | `uv run --group gpu pytest -q`: 425 passed, 0 skipped (282 s) after the review changes, including the GPU batched-vs-sequential test; plus `test_calibration_gate_minimum_repetitions`. |

- Prevention: before a new equivalence, join or timing test, check this table. Re-test only what a later change touches, on real data, against a reference path that shares no code with the fast path.

## 2026-11-11: Retired panel interpretations and unsupported figure content

The Tier C correction retains the exploratory question of whether recorded
clinical endpoints associate with measured spectral features. The former fixed
panel used `membrane_integrity_index = ch_bulk_2885 / lipid_carbonyl_1721`,
`oxidative_index = carotenoid_1525 / lipid_carbonyl_1721`,
`protein_remodeling_index = protein_1555 / (structural_protein_925 + structural_protein_980)`,
and `tyrosine_environment_index = tyrosine_851 / tyrosine_821`. These were
retrospective spectral ratios, without evidence validating membrane integrity,
oxidation, protein remodeling, or a tyrosine-doublet interpretation.

Their old windows were carbonyl 1710-1735 (fallback 1700-1750), carotenoid 1156
1145-1170, protein 1555 1540-1570, structural 925 915-935, structural 980 970-990,
tyrosine 851 845-858, and the misnamed 821 band 815-828, all in inverse
centimetres. The revised implementation uses the maintained registry for atomic
bands, preserves CH 2850-2950 as an explicit composite, and gives ratios neutral
names. Previous outputs do not become results under the revised definitions;
they require regeneration with the recorded basis and feature provenance.
Reconsider biological index interpretations only with independent measurement
validation of the named quantity and an explicit analysis definition.

The former manuscript graphical abstract and replay-RNG helper were retired.
The graphic embedded fixed cohort sizes, variance shares, performance values,
channel counts and a clinical conclusion. Its source had no verified connection
between those literals and the current run. Advancing a random stream through
unrelated simulated spectra also obscured the actual bootstrap seed definition.
A future graphical abstract must derive every study quantity from identified
saved results; an explanatory schematic must be labelled as such. Recorded
reference mean/SD values are summaries, not empirical draws: figures must not
manufacture samples from them to depict a measured distribution.

Evidence: source review against the pre-change `017e635` tree and the approved
Tier C implementation on 2026-11-11. This record documents code definitions and
reporting defects; it does not assert a new cohort result or assay validation.
