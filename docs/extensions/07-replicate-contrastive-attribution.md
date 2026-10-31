# What does EV-fraction Raman measure reproducibly in a person? Replicate-contrastive learning with same-batch negatives

Status: research design. Two label-free pilot checks ran on 2026-10-31 (results below). No spectrum-versus-outcome test has been run. The hypotheses below are prespecified and untested.

Data constraint: this design uses only the existing preprocessed store and `data/clinical_master.csv`. It needs no new measurement and no new clinical variable.

Extends: [01](../brainstorming/01-reliable-clinical-attribution.md) (stability labels), [05](05-lipoprotein-decomposition.md) (lipid anchors). Uses the replicate structure from [04](../brainstorming/04-conditional-testing-research-proposal.md).

## Research questions

- **RQ1.** How much participant-specific signal remains in the Raman spectra after removing processing order and technical factors?
- **RQ2.** Which measured biology does that reproducible signal track in held-out stroke patients?
- **RQ3.** Does low reliability explain the null clinical associations in `spectra_manuscript/main.tex:430–433`?

## Why this matters

The manuscript found a case-control difference but no within-stroke association. A feature that cannot be reproduced in the same person cannot correlate with that person's outcome. The reliability of the spectral features therefore sets an upper bound on every clinical correlation, and nobody has measured it for EV-fraction Raman.

Huber et al. (Nat Commun 2021) showed that infrared fingerprints of serum are specific to each person. A short search found no Raman or EV study that trains on technical replicates, draws hard negatives from the same batch, and attributes the result to each patient's own laboratory values. Read the full literature before claiming priority.

## What the pilot found

Pilot run: 95 stroke patients with two slides each. Each slide is summarized by its median fingerprint and C–H spectrum. The identification task asks whether slide B picks out slide A of the same patient. The script is not in the repository.

| Check | Result | Reading |
|---|---|---|
| Identification among all 95 patients | 14.2% (chance 1.1%; 99th percentile of the permutation null 3.7%) | Individual signal exists. Secondary figure. |
| Identification among the 10 nearest sample-number neighbours | 25.3% (chance 9.1%) | Primary figure. It cannot come from batch differences, because all candidates are neighbours. |
| Identification among 10 random non-neighbours | 44.8% (5–95%: 40.0–50.5%) | Higher than the neighbour figure. Batch differences make random distractors easy. Do not cite as evidence of identity. |
| Same contrast inside one acquisition mode | FP 31.7% vs 56.3%; SS 18.5% vs 33.6% | The batch effect follows processing order, not acquisition mode. |
| QC metrics alone (22 features) | 5.8% | Technical metrics carry little identity. |
| Spectra after regressing out QC metrics | 4.7% | Ambiguous. QC metrics such as fluorescence also carry biology. |

Sample numbers split the groups: controls are 1–43 and 192–199; stroke patients are 44–191 and 200–250. Within stroke, sample number does not follow participant number (ρ = 0.14, p = 0.16). It correlates with admission NIHSS (ρ = 0.25, p = 0.013, one of 14 checks, uncorrected).

## Existing data used

| Item | Value | Use |
|---|---|---|
| Slide pairs | 95 stroke patients with 2 slides; 2 with 1 slide; 50 controls with 1 slide | Positive pairs for contrastive training. |
| Retained pixel spectra | 17,725 fingerprint; 17,201 C–H | Label-free pretraining. |
| Lipid panel | TG, LDL-C, HDL-C, TC for 97 patients | Primary anchors (TC excluded: ρ = 0.95 with LDL-C). |
| Blood counts and chemistry | Platelets, haemoglobin, CRP, glucose, WBC for 97 patients; fibrinogen for 93 | Secondary anchors. |
| Fibrinogen and thrombolysis | IVT 2.18 g/L, no IVT 3.13 g/L. All 7 values below 1 g/L are IVT patients. | Fibrinogen is analysed only inside IVT strata. |
| Severity and outcome | NIHSS (admission, 24 h), mRS (before stroke, discharge, 3 months; scores 0–4 only), MoCA (72 baseline, 92 at 3 months) | RQ3. |
| ApoE | e2/e3 11, e3/e3 61, e3/e4 25 | Germline check on the lipid component. Low power. |
| Smoking and statin | Smoking current 24/97 stroke, 7/47 controls; statin 32/97 stroke, 12/47 controls | Positive controls. |
| Lesion side | Left 41, right 45, posterior 11 | Negative control. |

## Prespecified hypotheses

| ID | Prediction | Role |
|---|---|---|
| H1 | Within-batch identification of held-out patients exceeds 25% for the learned embedding and exceeds the linear baseline. | Primary. |
| H2 | A reproducible component tracks LDL-C or TG (lipoprotein co-isolation). | Primary, Holm across LDL-C, TG, HDL-C and platelet count. |
| H3 | A reproducible component tracks platelet count. | Primary. Platelet EVs are a hypothesis, not an assumption. |
| H4 | A component with high loadings on both carotenoid windows (1156 and 1525 cm⁻¹) is lower in current smokers, with the same sign in both groups. | Positive control. |
| H5 | Statin use shifts the lipid component with the same sign in stroke and controls. | Positive control. |
| H6 | No component differs between left and right lesions. | Negative control (Lipsitch et al. 2010; cited from memory). |
| H7 | Associations with NIHSS, mRS and MoCA, corrected for unreliability, are no larger than their reliability ceiling allows. | RQ3. Set in advance. |

## Method

### 1. Splits and drift term

Split by participant. Both slides, all maps and both blocks of a participant stay in one fold. All fitting stays inside training folds, as the repository requires. Add sample number as a smooth drift term to every within-stroke test.

### 2. Linear baseline (no new dependency)

Solve a generalized eigenproblem. Maximize between-patient scatter over within-patient scatter plus within-batch scatter, where a batch is a window of neighbouring sample numbers. Choose the number of dimensions by held-out discriminability (Bridgeford et al., PLoS Comput Biol 2021).

### 3. Deep encoder (needs PyTorch)

PyTorch is not in `pyproject.toml`. Adding it needs approval. Build this step only after the linear baseline is done.

1. Pretrain a 1D encoder by masked-spectrum modelling on the pixel spectra, without labels.
2. Pool pixel to map to slide with attention.
3. Train with InfoNCE or VICReg. The two slides of a patient are positives. Sample-number neighbours are hard negatives.
4. Add a gradient-reversal head that penalizes predicting QC metrics and acquisition mode.
5. Augment with baseline drift, shifts of ±1–2 cm⁻¹, intensity scaling, shot noise and cosmic spikes.
6. Keep the model at or below 10⁵ parameters.
7. Keep the deep encoder only if it beats the linear baseline on held-out within-batch identification.

### 4. Attribution

Use cross-fitting inside stroke. Score each anchor on held-out patients. Label each component–anchor link as separate, joint or unresolved by bootstrap stability (doc 01). Expect joint or unresolved.

### 5. Reliability ceiling

Estimate the intraclass correlation (ICC) of each component from the slide pairs. The largest observable correlation with an outcome is √(ICC × reliability of the outcome). Report associations with NIHSS, mRS and MoCA after correction for unreliability, with Holm correction.

## Baselines

- PCA scores plus ridge or PLS for each anchor, cross-fitted.
- The 21-band panel from `src.analysis.raman_bands`.
- The doc 05 non-negative decomposition, for the lipid anchors.
- Ablations: no hard negatives; no gradient reversal; no pretraining; random negatives.

## Limits

- Same-batch negatives remove local batch similarity. They do not remove the large order trend between samples 1–43 and 44–250. The encoder does not fix the case-control confound (see doc 08).
- The two-slide design assumes the slides are technical replicates of one draw (doc 04:21). `main.tex:178,192` describes two draws (within 6 h and 3 months). If that reading is right, the positive pairs would erase the change from acute to 3 months, and the design must switch to change within each person.
- Controls have no laboratory values. All anchors are tested within stroke.
- CRP, fibrinogen and glucose depend on timing. Platelet and lipid values depend on fasting and on the draw time relative to the EV plasma. Neither is documented.

## Result scenarios

| Scenario | Interpretation |
|---|---|
| Within-batch identification rises well above 25% and a component tracks lipids or platelets. | A first map of what EV-fraction Raman measures reproducibly. Not EV specificity. |
| Identification stays near 25%. | Low reliability explains the null clinical results. A result the field needs. |
| Identification is high but no component tracks any anchor. | The reproducible signal is real and its source is unknown. |

## Open items

- What does the sample number encode (SEC isolation order, batch, storage, slide label)? Is there a session log?
- Are the two slides one draw or two? Default: one draw, as doc 04 states.
- Was the research plasma drawn before or after the alteplase bolus? Default: after, for IVT patients.
- Was the draw fasting? Default: unknown; TG is then secondary.

## Claims

- Allowed: "A learned spectral subspace identifies the same patient in X% of within-batch trials and tracks LDL-C in held-out patients (ρ, interval)."
- Not allowed: EV specificity, brain origin, molecular quantitation, diagnostic use, removal of the case-control confound.

## Target journals

Analytical Chemistry or Biosensors & Bioelectronics if the attribution leads. Pattern Recognition if the method and simulations lead. Confirm quartile and scope at submission.

## References

- Huber M et al. Stability of person-specific blood-based infrared molecular fingerprints. Nat Commun 2021;12:1511.
- Bridgeford EW et al. Eliminating accidental deviations to minimize generalization error and maximize replicability. PLoS Comput Biol 2021;17:e1009279.
- Li B et al. Raman spectrum matching with contrastive representation learning. arXiv:2202.12549.
- Georgiev D et al. Physics-constrained autoencoders for Raman unmixing. PNAS 2024; arXiv:2403.04526.
- Lipsitch M et al. Negative controls. Epidemiology 2010. (From memory; not checked.)
