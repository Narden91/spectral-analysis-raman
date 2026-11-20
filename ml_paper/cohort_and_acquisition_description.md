# Cohort, Wet-Lab, Acquisition, and Preprocessing Reference for the ML Manuscript

This document consolidates the experimental, clinical, and spectroscopic facts from [spectra_manuscript/main.tex](../spectra_manuscript/main.tex) and reconciles them with the audited clinical master ([data/clinical_master.csv](../data/clinical_master.csv), [data/clinical_master_notes.md](../data/clinical_master_notes.md)) and the active study design ([docs/research_design.md](../docs/research_design.md)).

---

## 1. Study Origin, Ethics, and Clinical Cohort

- **Parent Study**: IBioStroke (*Identification and Clinical Validation of Biomarkers for Long-Term Outcome After Cerebral Ischemia*), Comprehensive Stroke Center, University Hospital in Krakow, Poland.
- **Enrollment Window**: July 18, 2020 to December 14, 2022.
- **Ethics Approval**: Jagiellonian University Ethical Committee, approval no. `1072.6120.118.2020`. Written informed consent was obtained from all stroke participants in accordance with the Declaration of Helsinki.
- **Inclusion Criteria (Parent Stroke Cohort)**:
  - Age $\ge 18$ years.
  - Acute ischemic stroke meeting the World Health Organization definition.
  - Signed informed consent.
- **Exclusion Criteria (Parent Stroke Cohort)**:
  - Pre-stroke modified Rankin Scale ($\mathrm{mRS}$) score $> 2$.
  - Pre-stroke diagnosis of dementia or depression.
  - Concomitant chronic neurodegenerative disorder, chronic autoimmune disorder, or active malignancy.

---

## 2. Reconciled Cohort Counts & Crucial Documentation Corrections

When drafting the new manuscript, **do not copy outdated statements** from `spectra_manuscript/main.tex` where they conflict with the verified clinical audit in `data/clinical_master_notes.md` and `AGENT_LEARNINGS.md`:

| Topic | Statement in `spectra_manuscript/main.tex` | Verified Status for `ml_paper` (`docs/research_design.md` & `data/clinical_master.csv`) |
|---|---|---|
| **Stroke participant linkage** | Reports 97 stroke participants (45 female, 52 male) using folder-derived sex labels. | **97 stroke participants** have spectral folders, but **5 records have `SEX_FOLDER_CONFLICT`** (`spectral_link_status = identity_review_required`). Excluding unresolved identity conflicts leaves **92 identity-verified stroke participants** (**90 with paired technical sample labels**, **2 with a single sample label**). After map QC (at least two retained maps per spectral block), **88 have two eligible technical samples and 4 have one**: pixel QC rejected two of three C–H maps in one sample of each of two participants. |
| **Meaning of paired stroke samples** | Section 2.3 states blood was collected longitudinally within 6 h and at 3 months post-stroke ($n=100$ samples). | Investigator confirmation recorded on 21 October 2026 states that the **paired stroke samples are technical replicates from a single collection**, not longitudinal visits. Never infer longitudinal change from sample pairs; explicitly document which preparation/acquisition stages were independently repeated. |
| **Control cohort role & metadata** | States age was available for only 1 control; includes 50 controls (40 female, 10 male by folder) in case-control models. | Standalone `data/clinical_master.csv` records clinical age for **47 spectral control participants** (37 women, 10 men), with 2 raw control folders lacking clinical records and 1 control filename/folder conflict. However, **controls have no recorded lipid values** (`ldl_mmol_l`, `hdl_mmol_l`, `triglycerides_mmol_l`) and only 1 technical sample per participant, so controls are excluded from the primary lipid-prediction pipeline. |
| **Laboratory anchor completeness & units** | Focuses on MoCA ($\mathrm{MoCA}_{\text{baseline}}$, $\mathrm{MoCA}_{3\text{m}}$, $\Delta\mathrm{MoCA}$), RBANS, and ASPECTS. | `ldl_mmol_l`, `hdl_mmol_l`, `triglycerides_mmol_l`, and `plt_10e9_l` are populated for all **92 identity-verified stroke participants**. Units (`mmol/L` for lipids, $10^9/\mathrm{L}$ for platelets) are marked `assumed` in `data/clinical_master_dictionary.csv`; fasting status, blood-draw timing relative to EV isolation, and direct assay vs. Friedewald calculation are unconfirmed and must be stated as explicit limitations. |

---

## 3. EV-Enriched Fraction Isolation Protocol (SEC + Ultrafiltration)

- **Starting Material**: Human plasma snap-frozen and stored at $-80\,^\circ\text{C}$.
- **Clarification**: Thawed and centrifuged at $3{,}000 \times g$ for $10\text{ min}$ at $4\,^\circ\text{C}$ to remove residual cells and debris.
- **Size-Exclusion Chromatography (SEC)**:
  - $0.5\text{ mL}$ clarified supernatant loaded onto a `qEVoriginal/35 nm` column (cat. no. `ICO-35-100`, Izon Science) equilibrated with $0.2\,\mu\text{m}$-filtered DPBS (cat. no. `D8537`, Sigma-Aldrich).
  - Void volume: $2.0\text{ mL}$ DPBS discarded.
  - Purified collection: Sequential $0.4\text{ mL}$ fractions 2–4 pooled.
- **Ultrafiltration Concentration**:
  - Pooled fractions 2–4 transferred to an Amicon Ultra-4 centrifugal filter unit ($10\text{ kDa}$ MWCO, cat. no. `UFC8010`, MilliporeSigma).
  - Centrifuged at $4{,}000 \times g$ for $40\text{ min}$ at room temperature.
  - Adjusted to $100\,\mu\text{L}$ final volume with DPBS and stored at $-80\,^\circ\text{C}$.
- **Nanoparticle Tracking Analysis (NTA)**:
  - Instrument: NanoSight NS300 (Malvern Panalytical) with NTA software v3.2.
  - Dilution: $0.2\,\mu\text{m}$-filtered DPBS to 20–60 particles/frame.
  - Acquisition: Three $30\text{ s}$ videos per sample at $25\,^\circ\text{C}$ under constant syringe-pump flow, camera level 14, detection threshold 4.
- **MISEV2023 / MIBlood-EV Nomenclature Constraint**:
  - Because SEC and NTA do not separate extracellular vesicles from co-isolated lipoproteins (LDL, VLDL, HDL, chylomicron remnants) or residual protein complexes, preparations are strictly termed **circulating EV-enriched fractions**.

---

## 4. Raman Microspectroscopy Acquisition Protocol

- **Pre-Measurement Handling**:
  - Thawed at room temperature, vortexed, and briefly centrifuged ($\le 10\text{ s}$).
  - Note: For approximately 10 early samples, initial setting adjustments required 2–3 freeze–thaw cycles.
- **Substrate & Droplet Deposition**:
  - $1.5\,\mu\text{L}$ droplet deposited onto a $\text{CaF}_2$ substrate (LaserOptex Inc., China) pre-rinsed with deionized water and $70\%$ ethanol.
  - Dried for $10\text{--}20\text{ s}$ with a hot-air dryer held $20\text{--}30\text{ cm}$ from the substrate (surface temperature $\le 30\,^\circ\text{C}$; ambient $21.5\,^\circ\text{C}$, $45\%$ relative humidity).
- **Spectrometer & Optics**:
  - Instrument: Renishaw inVia Qontor Raman microscope with CCD detector.
  - Excitation: $785\text{ nm}$ laser at $100\%$ nominal output ($\approx 50\text{ mW}$ source power; incident power at sample not independently metered).
  - Grating & Objective: $1200\text{ lines/mm}$ grating, $50\times$ objective ($\mathrm{NA} = 0.75$, lateral spatial resolution $\approx 1\,\mu\text{m}$).
  - Calibration: Internal silicon reference band at $520.5\text{ cm}^{-1}$.
- **Spectral Regions & Spatial Mapping**:
  - **Fingerprint block** (centered at $1300\text{ cm}^{-1}$): $3\text{ s}$ exposure, 2 accumulations; frozen trimmed analytical grid **$734\text{--}1788\text{ cm}^{-1}$** ($1\text{ cm}^{-1}$ spacing, **1,055 channels**).
  - **C–H stretching block** (centered at $2850\text{ cm}^{-1}$): $4\text{ s}$ exposure, 2 accumulations; frozen trimmed analytical grid **$2451\text{--}3197\text{ cm}^{-1}$** ($1\text{ cm}^{-1}$ spacing, **747 channels**).
  - **Spatial Hierarchy per Technical Sample**: 3 physical areas (maps) within the coffee-ring region of the dried droplet per sample and block; each area is a $5 \times 5$ grid of 25 pixels spanning $5\,\mu\text{m} \times 5\,\mu\text{m}$ ($1\,\mu\text{m}$ step size), yielding 75 raw pixel spectra per sample and block.

---

## 5. Frozen Preprocessing Pipeline & Curated 21-Band Registry

### Preprocessing Sequence (`docs/preprocessing.md` & `spectra_manuscript/main.tex` Table 1)
1. **Dead / saturated pixel screen**: Minimum spectral SD $= 1.0$; saturation threshold $= 65{,}000$.
2. **Positive cosmic-ray correction**: Spatial-neighbor $z$-threshold $= 8$, maximum width $= 3$ channels, 8 neighbors, half-window $= 10$, maximum corrected fraction $= 0.02$.
3. **Edge trimming & frozen-grid linear interpolation**: $\sim 15\text{ cm}^{-1}$ edge trim; linear interpolation onto $734\text{--}1788\text{ cm}^{-1}$ and $2451\text{--}3197\text{ cm}^{-1}$ ($1\text{ cm}^{-1}$ step) with zero extrapolation.
4. **Baseline correction**: arPLS ($\lambda = 10^6$, max 100 iterations, convergence $0.001$); signed residuals preserved without zero-clipping.
5. **Quality control SNR gate**: Minimum $\mathrm{SNR} = 5$ in region-specific anchor windows; minimum 4 retained pixels per map.
6. **Smoothing**: Savitzky–Golay order 2 (7-channel window in fingerprint; 9-channel window in C–H).
7. **Block-specific normalization**:
   - Fingerprint ($734\text{--}1788\text{ cm}^{-1}$): Standard Normal Variate (**SNV**).
   - C–H stretching ($2451\text{--}3197\text{ cm}^{-1}$): **Trapezoidal area** normalization.

### Curated 21 Raman Windows (`src/analysis/raman_bands.py`)
| Feature Identifier | Peak ($\text{cm}^{-1}$) | Window ($\text{cm}^{-1}$) | Category | Tentative Biochemical Context |
|---|---:|---|---|---|
| `cytosine_786` | 786 | 773–796 | Nucleic acids | Cytosine/uracil breathing/stretching modes |
| `o_p_o_821` | 821 | 812–832 | Nucleic acids | O–P–O stretch |
| `tyrosine_851` | 851 | 836–874 | Mixed | Tyrosine doublet / glycan signature |
| `c_o_c_sugar_925` | 925 | 900–938 | Nucleic acids | C–O–C stretch in sugars |
| `beta_sheet_980` | 980 | 965–995 | Proteins | C–C stretch $\beta$-sheet |
| `c_o_c_n_1060` | 1060 | 1044–1078 | Proteins | C–O / C–N stretch |
| `lipid_cc_1089` | 1089 | 1082–1098 | Lipids | C–C skeletal stretch |
| `po2_cn_1112` | 1112 | 1105–1120 | Mixed | $\text{PO}_2^-$ stretch / C–N stretch |
| `carotenoid_1156` | 1156 | 1145–1165 | Carotenoids | C–C stretch carotene |
| `amide_iii_1258` | 1258 | 1240–1280 | Mixed | CH bend / amide III |
| `ch2_twist_1301` | 1301 | 1290–1325 | Lipids | $\text{CH}_2$ twist |
| `ch2_ch3_scissor_1440` | 1440 | 1430–1445 | Lipids | $\text{CH}_2/\text{CH}_3$ scissor deformation |
| `ch2_ch3_bend_1465` | 1465 | 1450–1475 | Lipids | $\text{CH}_2/\text{CH}_3$ bend |
| `carotenoid_1525` | 1525 | 1510–1535 | Carotenoids | C=C stretch carotene |
| `tryptophan_1555` | 1555 | 1550–1560 | Proteins | Tryptophan signature |
| `amide_i_1657` | 1657 | 1640–1675 | Mixed | C=C stretch / amide I |
| `lipid_carbonyl_1721` | 1721 | 1710–1730 | Lipids | C=O stretch (hydrogen-bonded) |
| `lipid_ester_1744` | 1744 | 1735–1755 | Lipids | C=O ester stretch |
| `ch2_sym_stretch_2853` | 2853 | 2845–2860 | Lipids | $\text{CH}_2$ symmetric stretch |
| `ch2_asym_stretch_2890` | 2890 | 2870–2910 | Mixed | $\text{CH}_2$ asymmetric stretch |
| `ch3_asym_stretch_2937` | 2937 | 2925–2945 | Mixed | $\text{CH}_3$ asymmetric stretch |

