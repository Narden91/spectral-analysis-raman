# Biologically-Constrained Intra-Disease Raman Analysis — Step-by-Step Implementation Plan

This document is a phase-by-phase implementation checklist for evolving the
intra-disease (`disease_only`) layer of the Raman spectral-analysis pipeline
from a high-dimensional *statistical discovery* workflow into a
**pre-registered, biologically-constrained inference** workflow.

It operationalizes external reviewer feedback ("block-aware normalization +
pre-registered features + strong clinical contrasts + effect sizes +
robustness") while staying inside the repository's existing architecture,
conventions, and conservative claim boundaries (`CLAUDE.md`,
`.cursor/skills/spectral-analysis-repo`).

The plan is deliberately **non-duplicative**: most reviewer recommendations are
already implemented in the repository, sometimes in a more rigorous form than
the feedback proposes. Each phase therefore states explicitly whether it
*verifies/hardens* existing behavior or *adds* genuinely new capability.

The plan is organized in two tiers:

- **Tier 1 — Statistical core (Phases A–J, S, L, K):** the pre-registered,
  effect-size-first confirmatory panel (block-aware normalization, integrals, named
  indices, robust contrasts, covariate control, heterogeneity, minimal model,
  endpoint consistency, graded robustness ranking, four-domain report).
- **Tier 2 — Structure-aware biochemical interpretability (Phases M–U):** peak shape,
  NMF, spectral entropy, difference spectra, feature coupling, optional index
  clustering, and journal figures. Phases **S** (endpoint consistency) and **L**
  (robustness ranking) are **Tier-1 capstones** written to `05_confirmatory_panel/`
  but documented alongside Tier 1 because they consume Tier-2 shape features (L)
  or feed the ranking score (S).

Tier 2 is **descriptive/secondary by design** (effect sizes, CIs, annotated
figures), anchored to the single mandatory anchor hypothesis frozen in Phase A, so
that adding chemical detail does not silently re-inflate the multiple-testing
family that Tier 1 controls.

---

## 0. Scientific Framing and Honest Scope

This is exploratory extracellular-vesicle (EV) Raman analysis. Per `CLAUDE.md`,
outputs are **not** diagnosis, prediction, biomarkers, or causal/mechanistic
claims. The existing review (`docs/intra_disease_scientific_review.md`) already
concluded that, after correction, **no spectral interval, PLS model, minimal
panel, or biochemical ratio reached robust support** in the current cohort, and
recommended a feasibility / pre-registration framing.

This plan is the constructive next step that those documents call for: a small,
pre-specified, biologically-motivated confirmatory panel with effect sizes and
robustness, run alongside (not replacing) the existing 14-phase exploratory
suite.

### Publishable narrative strategy (*Spectrochimica Acta A*)

The prior cohort review concluded that **intensity-only, high-dimensional discovery**
found no corrected associations. The upgraded plan therefore shifts from
*feature-centric statistics* to **multi-representation EV biochemical
characterization**:

| Representation | What it captures | Primary risk mitigated |
|---|---|---|
| Named indices (Phase C) | interpretable lipid/protein/carotenoid ratios | too many weak generic ratios |
| Spectral shape (Phase M) | peak shift, FWHM, asymmetry | **no intensity differences → shape still detectable** |
| NMF components (Phase N) | latent lipid/protein/carotenoid-like profiles | high-dimensional noise |
| Spectral entropy (Phase O) | per-subject distributional disorder | mean-based analysis misses heterogeneity |
| Difference spectra (Phase P) | visual biochemical signature | unclear isolated effects |
| Feature coupling (Phase Q) | coupled band/index structure | isolated weak features |
| Robustness ranking (Phase L) | graded evidence, not binary p-values | **"nothing significant" → graded trends** |
| Endpoint consistency (Phase S) | stable sign across MoCA endpoints | endpoint-specific noise |

Even if FDR-corrected p-values remain non-significant, the manuscript can report
a **multi-layer EV biochemical characterization** with one mandatory anchor
hypothesis (H₀/H₁ below), effect sizes, CIs, robustness scores, and annotated
spectral figures — provided all language stays conservative (no biomarker /
diagnostic claims).

**Mandatory anchor hypothesis** (frozen in `preregistration.md` before any tests):

```text
H₀: Oxidative index (carotenoid_1525 / lipid_carbonyl_1720) is not associated with ΔMoCA
H₁: Lower oxidative index is associated with worse ΔMoCA (Spearman ρ < 0)
```

Test with: partial Spearman | age, sex; bootstrap CI; permutation p; Cliff's δ
for extreme trajectory groups. This is the **single confirmatory test** (no FDR
across the anchor itself). All other features use the graded robustness framework.

### Final minimal upgrade set (highest impact)

If implementation time is limited, ship these five phases first — they transform
the narrative from "no significant spectral associations" to **multi-layer EV
biochemical characterization**:

1. **Phase M** — spectral shape (peak shift + FWHM + asymmetry)
2. **Phase N** — NMF decomposition (3 latent components)
3. **Phase O** — spectral entropy
4. **Phase P** — difference spectrum + annotated figure
5. **Phase L** — robustness ranking (requires Phases E, F, S)

Phases A–E (pre-registration + named indices + anchor test) remain prerequisites
for any confirmatory claim. Phase K (four-domain report) should follow once the
minimal set is wired.

### Critical assessment of the feedback (read before implementing)

The reviewer feedback is directionally sound but contains two claims that must
**not** be propagated into code or manuscript text:

1. **"Pre-registration removes the multiple-testing problem (your main
   failure)." — Overstated.** Pre-registration shrinks the testing family from
   hundreds of data-driven candidate regions to a small fixed set
   (~4 integrals + ~4 ratios × a few endpoints). It reduces, but does **not
   remove**, multiplicity. FDR and/or permutation control are still required and
   are retained in every phase below.

2. **"Replace continuous MoCA with extreme contrasts (severe <20 vs mild >25;
   decline <0 vs recovery >2)." — Helps effect size, hurts power.** Dropping the
   ambiguous middle increases between-group separation but **reduces n**, and low
   subject-level n was identified as the actual bottleneck. The net effect on
   power is ambiguous and cohort-dependent. Therefore extreme contrasts are added
   as an **additional, explicitly power-annotated** endpoint definition, not a
   replacement, and every contrast result must report achieved n per group and a
   minimum-detectable-effect note.

3. **Second-round "chemical expressiveness" feedback (peak shift/FWHM, NMF/ICA,
   entropy, named indices, difference spectra). — Adopt, but guard three traps.**
   - *NMF/ICA are non-unique and unstable at small n.* They must use fixed small
     component counts, multiple seeded initializations with a reported stability
     metric (e.g., cophenetic correlation / Amari distance), and remain explicitly
     exploratory. Correlating component weights with endpoints re-introduces
     multiplicity and still needs FDR.
   - *Peak-shift claims need a calibration floor.* A reported shift smaller than
     the instrument's wavenumber calibration/repeatability (and the per-fit
     bootstrap CI) is **not** interpretable. Every shift must be reported with a
     CI and compared against an explicit calibration-uncertainty floor; fits below
     a goodness-of-fit threshold are dropped, not interpreted.
   - *Spectral entropy is normalization-dependent.* It is a **dispersion
     descriptor**, not thermodynamic entropy; it must be computed on a fixed,
     documented normalized representation and reported as such.
   - *Every new feature family re-expands the testing family.* This is resolved by
     designating ONE pre-specified primary anchor hypothesis (Phase A/E) and treating
     all other Tier-2 layers as pre-specified secondary/descriptive outputs
     (effect sizes + CIs, annotated figures), not as additional gatekeeping tests.

Everything else in the feedback (block-aware normalization, region integrals,
mechanistic ratios, nonparametric tests + effect sizes, permutation/bootstrap,
covariate control, heterogeneity testing, clustering, a minimal interpretable
model, named EV indices, peak shape, band coupling, spectral decomposition,
slope/entropy descriptors, difference spectra, signed-effect maps, cross-endpoint
consistency) is scientifically appropriate and is mapped to concrete phases below.

---

## 1. Implementation Rules

Follow these strictly.

### General rules

- [ ] Implement one phase at a time; do not start a phase until the previous one passes its tests.
- [ ] Every phase must create at least one concrete deliverable (module, config block, or artifact) and add/update tests.
- [ ] Every phase must leave the repository in a working, runnable state.
- [ ] Prefer **existing repo helpers and config patterns** over new entry points (`CLAUDE.md` §8). The only user-facing EDA entry point remains `scripts/run_eda.py`. Do **not** create `scripts/run_biological_pca.py` or any new top-level runner.
- [ ] Route Tier-1 outputs through `get_confirmatory_panel_dir()` → `Results_intra_disease/05_confirmatory_panel/`; route Tier-2 outputs through `get_spectral_interpretability_dir(phase)` → `Results_intra_disease/06_spectral_interpretability/<phase>/`. Both helpers live in `src/__init__.py` alongside `get_cross_region_summary_dir()`. Do **not** add Tier-1/2 phases into `04_cross_region_summary/` or create a root-level `biological_pca/`.
- [ ] Keep Biological PCA subject-level and conservative: no diagnostic, causal, abundance, predictive, biomarker, or class-discriminative claims.
- [ ] Preserve `PCA_whole` block identity: `(spectral_block, raman_shift)` and block-aware column names such as `1300__1240.0`.
- [ ] Do not normalize the fingerprint and CH ranges jointly; keep block-aware normalization (already the default — see Phase B).
- [ ] Keep generated artifacts out of version control (`.gitignore`).
- [ ] Do not delete unrelated files from result directories; overwrite only deterministic outputs.
- [ ] Comments explain *why*, not *what*. No narration comments.
- [ ] Every results artifact set must remain accompanied by `run_manifest.json`, `data_manifest_snapshot.json`, and `config_snapshot.yaml`.
- [ ] Commit after each completed phase (only when the user asks to commit).

### Recommended validation commands after each phase

```bash
uv run pytest -v
```

Targeted checks used throughout this plan:

```bash
uv run pytest tests/test_biological_pca.py
python -m py_compile src/analysis/biological_pca.py scripts/run_eda.py
uv run scripts/run_eda.py --analysis-mode disease_only --range both --max-subjects 8
```

---

## 2. Current Repository Situation (feedback → code mapping)

Before writing new code, anchor on what already exists. The table maps each
reviewer recommendation to the current implementation and the resulting action.

| Feedback item | Already in repo? | Where | Action in this plan |
|---|---|---|---|
| Block-aware normalization (fingerprint and CH normalized separately, not jointly) | **Yes** | `config/settings.yaml` `preprocessing.ranges.fingerprint.normalization=snv`, `…ch_stretch.normalization=area`; cross-fusion mode `none`; `src/preprocessing/pipeline.py:normalize_spectra` | **Verify + lock with a test** (Phase B) |
| `PCA_whole` = unweighted concatenation preserving block identity | **Yes** | `src/analysis/biological_pca.py`, `config/eda.yaml biological_pca.models.PCA_whole` | No change; guard preserved |
| Region/peak **integrals** (not single points) | **Yes** | `src/analysis/biochemical_ratios.py:_band_integral` (trapezoidal); `whole_spectra.bands` windows | **Refactor to shared helper**, reuse for pre-registered panel (Phase C) |
| Mechanistic **ratios** | **Yes (different pairs)** | `whole_spectra.ratio_pairs`, `biochemical_ratios._build_ratio_table` | **Add pre-registered mechanistic ratios** (Phase C) |
| Severity / trajectory **endpoint groups** | **Yes (adjacent split)** | `src/clinical/endpoints.py` `moca_severity_group` (<20 vs ≥20), `delta_moca_group` (<0 vs ≥0) | **Add extreme-contrast variants** (Phase D) |
| Nonparametric primary test (Mann–Whitney) | **Yes** | `statistics.compare_band_intensities_between_groups` (auto MWU when n<20) | Reuse (Phase E) |
| Effect size: Cohen's d | **Yes** | `statistics.cohens_d` | Reuse (Phase E) |
| Effect size: **Cliff's delta** | **No** | — | **Add** `statistics.cliffs_delta` (Phase E) |
| **Bootstrap CI for effect size** | Partial | bootstrap exists in `minimal_panel`, `whole_spectra.correlation` | **Add** effect-size bootstrap for panel (Phase E) |
| Permutation testing | **Yes** | `src/statistics/cluster_permutation.py`, PLS/minimal-panel permutation | Reuse (Phase F) |
| Covariate control (age, sex) | **Yes (regression-adjusted)** | `src/analysis/covariate_models.py` (OLS / logistic Wald) | Reuse; **add rank-based partial Spearman** variant (Phase G) |
| **Heterogeneity / variance test (Levene, Brown–Forsythe)** | **No** | — | **Add** (Phase H) |
| Clustering for subgroup insight | **Yes (Ward + Gap + bootstrap ARI)** | `src/analysis/disease_subgroups.py` | Reuse; **optional** UMAP+k-means supplement (Phase I) |
| Minimal interpretable model (elastic net / L1 logistic, CV + permutation) | **Yes** | `src/analysis/minimal_panel.py` (nested CV, permutation, bootstrap) | Reuse, **restricted to pre-registered panel** (Phase J) |
| Conservative interpretation / publication report | **Yes** | `src/reporting/candidate_region_report.py` (Phase 14) | **Extend** with membrane/protein/oxidative framing (Phase K) |
| Named EV indices as **primary panel endpoints** | Partial | `whole_spectra.ratio_pairs` | **Move to Phase C** + treat as primary features, not derived extras |
| Mandatory anchor hypothesis (oxidative index vs ΔMoCA) | **No** | — | **Freeze in Phase A**; test in Phase E before panel FDR |
| Peak shift + FWHM + **asymmetry** | **No** | — | **Add** `spectral_shape.py` (Phase M) |
| NMF latent components | **No** | — | **Add** `nmf_decomposition.py` (Phase N) |
| Spectral entropy / heterogeneity fingerprint | **No** | — | **Add** Phase O |
| Difference spectrum + annotated figure | Partial | corr spectrum only | **Add** Phase P |
| Within/cross feature coupling matrix | **No** | — | **Add** Phase Q |
| **Robustness ranking score** (graded evidence) | **No** | — | **Add** Phase L |
| Endpoint consistency across MoCA endpoints | Partial | planned Phase S | **Implement** Phase S |

**Conclusion:** Tier-1 new code is small and surgical: a pre-registration config
block, a shared band-integral helper, a pre-registered panel feature builder,
extreme-contrast endpoint columns, three statistics functions (Cliff's delta,
partial Spearman, variance-homogeneity test), one robust-contrast phase module, a
publication-report extension, and tests. Tier-2 adds a focused set of
interpretability modules (`spectral_shape.py`, `nmf_decomposition.py`,
`spectral_descriptors.py`, plus discriminant/consistency/figure steps), each
exploratory/descriptive and reusing the existing subject feature matrix and
plotting/clustering infrastructure. Everything else is reuse.

### Spectral feasibility of the feedback's proposed bands

Acquired ranges (`src/constants.py`): fingerprint `734–1788 cm⁻¹`,
CH-stretch `2451–3197 cm⁻¹`.

| Proposed feature | Window (cm⁻¹) | In acquired range? | Caveat to record |
|---|---|---|---|
| Lipid carbonyl (ester C=O) | ~1710–1735 | Yes, but near fingerprint upper edge (1788) | Verify channels survive edge-trim/QC; anchor windows currently top out at 1680. Flag if <2 channels remain. |
| Carotenoid / C–C | ~1145–1170 (1156) | Yes | Sharp resonance band; confirm presence above noise before interpreting. |
| Carotenoid / oxidative C=C | ~1510–1535 (1525) | Yes | Sharp resonance band; interpret as carotenoid-associated only, not "oxidative stress" mechanism. |
| Protein | ~1540–1570 (1555) | Yes | "Protein" at 1555 is ambiguous in Raman (amide II weak; tryptophan/overlap). Use neutral region language. |
| CH bulk | 2850–2950 | Yes | Already covered by existing CH bands. |
| Structural protein (Phase C index) | ~915–935 (925), ~970–990 (980) | Yes | Weak C–C / C–N backbone bands; gate on SNR before use in the protein-remodeling index. |
| Tyrosine doublet (Phase C) | ~845–858 (851), ~815–828 (821) | Yes | Classic Fermi doublet (≈850/830). Weak bands; require GoF/SNR gate; ratio sensitive to baseline. |
| Low-vs-high slope ratio (Phase P) | 800–1000 vs 1500–1700 | Yes | Both windows inside fingerprint; depends on per-block normalization (interpret within-block only). |

These windows are **proposals to register**, not assertions of molecular
identity. All interpretation stays as "regions overlapping tentatively assigned
bands" per the existing review. Tier-2 shape features (peak shift, FWHM) require
that the relevant peak is resolved above noise *and* that the measured effect
exceeds the wavenumber calibration floor; otherwise the feature is reported as
non-interpretable rather than dropped silently.

---

## 3. Target Additions

New and modified files (kept minimal and inside existing packages):

```text
config/
└── eda.yaml
    ├── analysis.clinical_endpoints           # + mild_threshold, recovery_threshold (Phase D)
    └── analysis.preregistered_panel          # NEW config block (Phase A)

src/
├── __init__.py                               # + get_confirmatory_panel_dir(),
│                                             #   get_spectral_interpretability_dir(phase)
│                                             #   (parallel to get_cross_region_summary_dir)
├── clinical/
│   └── endpoints.py                          # + extreme-contrast group columns (Phase D)
├── analysis/
│   ├── band_features.py                      # NEW shared band-integral/ratio helper (Phase C)
│   ├── statistics.py                         # + cliffs_delta, partial_spearman, variance_homogeneity_test (Phases E,G,H)
│   ├── preregistered_panel.py                # panel matrix + named indices + anchor test (Phase C)
│   ├── robust_contrasts.py                   # MWU + Cliff's δ + bootstrap CI + robustness inputs (Phases E–F)
│   ├── robustness_ranking.py                 # NEW: graded evidence score (Phase L)
│   ├── spectral_shape.py                     # peak shift + FWHM + asymmetry (Phase M)
│   ├── nmf_decomposition.py                  # NMF with sparsity + stability (Phase N)
│   ├── spectral_descriptors.py               # spectral entropy + slope descriptors (Phase O)
│   ├── feature_coupling.py                   # cross-band / cross-index coupling (Phase Q)
│   ├── discriminant_spectra.py               # difference spectra + signed-effect map (Phase P)
│   ├── endpoint_consistency.py               # cross-endpoint sign agreement (Phase S)
│   ├── biochemical_ratios.py                 # refactor to use band_features.py (Phase C)
│   ├── covariate_models.py                   # refactor to use band_features.py (Phase C)
│   ├── disease_subgroups.py                  # + optional index-space clustering source (Phase R)
│   ├── disease_only_pipeline.py              # + new gated phases (Phases C–U)
│   └── minimal_panel.py                      # + optional panel-restricted feature source (Phase J)
├── visualization/
│   └── spectral_summary_plots.py             # + difference/signed-effect/regime/index-heatmap figures (Phases S, U)
└── reporting/
    └── candidate_region_report.py            # + interpretation framework + figure index (Phases K, U)

tests/
├── test_band_features.py                     # NEW
├── test_preregistered_panel.py               # NEW (panel + indices + anchor)
├── test_robust_contrasts.py                  # NEW
├── test_robustness_ranking.py                # NEW (Phase L)
├── test_statistics_effectsizes.py            # NEW
├── test_spectral_shape.py                    # NEW (Phase M)
├── test_nmf_decomposition.py                 # NEW (Phase N)
├── test_spectral_descriptors.py              # NEW (Phase O)
├── test_feature_coupling.py                  # NEW (Phase Q)
├── test_discriminant_spectra.py              # NEW (Phase P)
├── test_endpoint_consistency.py              # NEW (Phase S)
└── test_clinical_endpoints.py                # update for extreme contrasts
```

New output locations — both are **top-level siblings** of `04_cross_region_summary/`
under the active EDA result root (`Results_intra_disease/` in `disease_only` mode):

```text
Results_intra_disease/
├── 00_run_metadata/                            # existing — unchanged
├── 01_fingerprint_1300/                        # existing — unchanged
├── 02_ch_stretch_2850/                         # existing — unchanged
├── 03_whole_spectrum_1300_plus_2850/           # existing — unchanged
├── 04_cross_region_summary/                    # existing 14-phase exploratory suite — unchanged
│   ├── 01_subject_features/ … 14_publication_report/
│   └── …
│
├── 05_confirmatory_panel/                      # NEW — Tier 1 (Phases A–J, S, L, K) + anchor + ranking
│   ├── preregistration.md                      # frozen windows, indices, H₀/H₁ anchor, endpoints, tests
│   ├── panel_feature_matrix.csv                # subject × {integrals, named indices} — indices are PRIMARY features
│   ├── primary_anchor_result.csv               # mandatory confirmatory test: oxidative_index vs ΔMoCA
│   ├── panel_peak_shape_features.csv           # copy/summary of shape features for panel reporting (from Phase M)
│   ├── panel_contrast_tests.csv                # MWU p, Cliff's δ + bootstrap CI, FDR q, perm p, n per group
│   ├── panel_continuous_associations.csv       # raw Spearman + partial Spearman | age,sex, BH-FDR
│   ├── panel_variance_homogeneity.csv          # Levene + Brown–Forsythe, per feature × contrast
│   ├── feature_robustness_ranking.csv          # graded robust_score per feature (Phase L)
│   ├── panel_minimal_model_performance.csv     # nested-CV balanced accuracy / R², perm p, bootstrap CI
│   ├── panel_minimal_model_coefficients.csv
│   ├── panel_report_section.md                 # 4-domain biochemical narrative (Phase K)
│   ├── figures/
│   │   ├── panel_effect_sizes.png              # Cliff's δ ± bootstrap CI forest plot
│   │   ├── panel_rainclouds.png
│   │   └── panel_correlation_overview.png
│   └── README.md
│
└── 06_spectral_interpretability/               # NEW — Tier 2 (Phases M–U)
    ├── peak_shape/                             # Phase M
    │   ├── peak_shape_features.csv             # centroid, FWHM, asymmetry ± CI, GoF, SNR flag
    │   └── peak_shape_associations.csv
    ├── nmf_decomposition/                      # Phase N
    │   ├── nmf_components.csv                  # block-aware loadings + stability metric
    │   ├── nmf_weights_per_subject.csv
    │   └── nmf_weight_associations.csv         # weights vs ΔMoCA / severity, FDR
    ├── spectral_entropy/                       # Phase O
    │   ├── spectral_entropy.csv                # per-subject entropy + dispersion index
    │   └── spectral_entropy_associations.csv
    ├── discriminant_signatures/                # Phase P
    │   ├── difference_spectrum.csv             # severe−mild, decline−recovery (per block)
    │   ├── signed_effect_map.csv
    │   └── figures/
    │       ├── difference_spectrum_annotated.png
    │       └── signed_effect_spectral_map.png
    ├── feature_coupling/                       # Phase Q
    │   ├── feature_coupling_matrix.csv         # corr(1720,1555), corr(1525,1720), corr(1555,925), …
    │   ├── ratio_ratio_regimes.csv             # lipid/oxidative/protein-dominant regimes
    │   └── figures/
    │       ├── ratio_ratio_regime_plot.png
    │       └── feature_coupling_heatmap.png
    ├── index_clustering/                       # lower-priority supplement
    │   └── index_cluster_assignments.csv
    ├── endpoint_consistency/                   # Phase S
    │   └── endpoint_consistency.csv            # sign agreement across moca_baseline, moca_3m, delta_moca
    ├── figures/                                # Phase U — assembled journal set
    │   ├── fig1_cohort_panel_overview.png
    │   ├── fig2_effect_size_forest_robustness.png
    │   ├── fig3_difference_spectrum_signed_effect.png
    │   ├── fig4_nmf_components_entropy.png
    │   └── fig5_feature_coupling_regimes.png
    └── README.md
```

**Routing convention:** two new helpers in `src/__init__.py` (alongside `get_cross_region_summary_dir`):

```python
def get_confirmatory_panel_dir() -> Path:
    """Top-level output root for Tier-1 pre-registered panel results."""
    path = _output_dir("eda_subdir") / "05_confirmatory_panel"
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_spectral_interpretability_dir(phase: str) -> Path:
    """Return a phase subfolder under the Tier-2 spectral interpretability root."""
    _INTERP_SUBDIRS = {
        "peak_shape":       "peak_shape",
        "nmf":              "nmf_decomposition",
        "entropy":          "spectral_entropy",
        "discriminant":     "discriminant_signatures",
        "coupling":         "feature_coupling",
        "index_clustering": "index_clustering",
        "consistency":      "endpoint_consistency",
        "figures":          "figures",
    }
    path = _output_dir("eda_subdir") / "06_spectral_interpretability" / _INTERP_SUBDIRS.get(phase, phase)
    path.mkdir(parents=True, exist_ok=True)
    return path
```

These helpers are registered in `src/__init__.py` alongside the existing `get_cross_region_summary_dir` and `get_cross_region_phase_dir`. Every new module calls the appropriate helper rather than constructing paths manually.

---

## 4. Target Pipeline (confirmatory track)

The pre-registered confirmatory track runs after Biological PCA exports the
subject-level feature matrix, in parallel with the existing exploratory phases:

```text
subject-level block-aware normalized spectra (already produced)
    ↓
Biological PCA subject feature matrix (1300__*, 2850__* columns; block identity preserved)
    ↓
freeze pre-registration (windows, ratios, endpoints, tests)        [Phase A]
    ↓
verify block-aware normalization invariants                        [Phase B]
    ↓
build pre-registered panel: integrals + **named indices as primary features** [Phase C]
    ↓
derive endpoints: continuous + adjacent groups + extreme contrasts [Phase D]
    ↓
run **mandatory anchor test** (oxidative index vs ΔMoCA)              [Phase E]
    ↓
robust group contrasts: Mann–Whitney + Cliff's δ + bootstrap CI        [Phase E]
    ↓
permutation validation of contrasts                                    [Phase F]
    ↓
covariate control: partial Spearman (feature ~ endpoint | age,sex)   [Phase G]
    ↓
cross-endpoint consistency (MoCA baseline / 3m / ΔMoCA)               [Phase S]
    ↓
heterogeneity: Levene / Brown–Forsythe per feature × group           [Phase H]
    ↓
minimal interpretable model restricted to named indices              [Phase J]
    ↓
── Tier 2: structure-aware biochemical interpretability ───────────────────────
peak shape: shift + FWHM + asymmetry (Gaussian/Voigt, SNR gate)       [Phase M]
    ↓
NMF decomposition: 3–5 latent components (fingerprint; optional whole) [Phase N]
    ↓
spectral entropy / dispersion fingerprint (fingerprint region)         [Phase O]
    ↓
difference spectrum + signed-effect map (publication figures)          [Phase P]
    ↓
cross-feature coupling + ratio–ratio biochemical regimes               [Phase Q]
    ↓
**graded robustness ranking** (effect + CI + perm + consistency)       [Phase L]
    ↓
4-domain biochemical narrative report (consumes Tier 1 + Tier 2)      [Phase K]
    ↓
optional: clustering insight (reuse disease_subgroups; UMAP supplement) [Phase I]
    ↓
index-space clustering (optional supplement)                           [Phase R]
    ↓
journal figure set (*Spectrochimica Acta A*)                           [Phase U]
    ↓
pipeline wiring, tests, manifests, reproducibility                     [Phase V]
```

---

## 5. Phase-by-Phase Plan

Each phase lists: **Objective**, **Why (scientific rationale)**, **Changes**,
**Deliverables**, **Tests**, **Validation**, **Claim boundaries**.

### Phase A — Pre-registration config block

- **Objective:** Encode the fixed hypotheses (windows, ratios, endpoints, tests, correction) in `config/eda.yaml` and emit them as a frozen, human-readable `preregistration.md` before any test runs.
- **Why:** Pre-specification shrinks the testing family and documents intent before seeing results (guards against HARKing / p-hacking). It does *not* remove multiplicity (see §0), so the block also fixes the correction method.
- **Changes:** Add `analysis.preregistered_panel` to `config/eda.yaml`:

```yaml
analysis:
  preregistered_panel:
    enabled: true
    fdr_alpha: 0.05
    n_permutations: 1000
    n_bootstrap: 2000
    random_state: 42
    min_subjects_per_group: 5
    min_snr_for_peak_fit: 5.0              # skip low-quality spectra (Phase M)
    calibration_floor_cm: 1.5              # shifts below this are not_interpretable
    integrals:                               # name: [block, lo_cm, hi_cm]
      lipid_carbonyl_1720: ["1300", 1710, 1735]
      lipid_carbonyl_1720_fallback: ["1300", 1700, 1750]   # used if narrow window <2 channels
      carotenoid_cc_1156:  ["1300", 1145, 1170]
      carotenoid_1525:     ["1300", 1510, 1535]
      protein_1555:        ["1300", 1540, 1570]
      structural_protein_925: ["1300", 915, 935]
      structural_protein_980: ["1300", 970, 990]
      tyrosine_851:        ["1300", 845, 858]
      tyrosine_821:        ["1300", 815, 828]
      ch_bulk_2885:        ["2850", 2850, 2950]
    indices:                                 # PRIMARY panel features (not secondary ratios)
      membrane_integrity_index: [ch_bulk_2885, lipid_carbonyl_1720]
      oxidative_index:        [carotenoid_1525, lipid_carbonyl_1720]
      protein_remodeling_index: [protein_1555, structural_protein_925, structural_protein_980]
      tyrosine_environment_index: [tyrosine_851, tyrosine_821]
    primary_hypothesis:                      # MANDATORY — single confirmatory test
      feature: oxidative_index
      endpoint: delta_moca
      direction: negative                    # H1: lower index → worse ΔMoCA
      test: partial_spearman
      covariates: [age, sex]
    robustness:                              # Phase L scoring thresholds
      effect_size_threshold: 0.147           # |Cliff's δ| > small effect
      perm_p_threshold: 0.10                 # exploratory perm gate
      min_endpoint_agreement: 2              # of 3 MoCA endpoints same sign
    endpoints:
      continuous: [moca_baseline, moca_3m, delta_moca]
      contrasts:
        - {name: severity_extreme, group_variable: moca_severity_group_extreme, ordered_groups: [severe, mild]}
        - {name: trajectory_extreme, group_variable: delta_moca_group_extreme, ordered_groups: [decline, recovery]}
    covariates: [age, sex]
    peak_shape_windows:                      # Phase M targets
      lipid_carbonyl_1720: [1710, 1735]
      protein_1555:        [1540, 1570]
      carotenoid_1525:     [1510, 1535]
    nmf:
      n_components: 3
      sparsity_alpha: 0.1                    # L1 regularization on NMF (Phase N)
      n_init: 20
      apply_to: fingerprint                  # fingerprint required; whole optional
```

- **Deliverables:** config block including `indices`, `primary_hypothesis`, and `robustness` thresholds; `05_confirmatory_panel/preregistration.md` writer (via `get_confirmatory_panel_dir()`) that snapshots the block verbatim — including explicit H₀/H₁ wording for the anchor — with a timestamp and resolved config hash, **written before any statistical tests run**.
- **Tests:** `test_preregistered_panel.py::test_preregistration_snapshot_is_frozen` — the written `preregistration.md` matches the config and contains every window/ratio/endpoint.
- **Validation:** `python -m py_compile src/analysis/preregistered_panel.py`.
- **Claim boundaries:** windows are *registered hypotheses*, not molecular identifications.

### Phase B — Verify and lock block-aware normalization

- **Objective:** Confirm (and lock with a test) that fingerprint and CH are normalized **separately** and that whole-spectrum concatenation does **not** re-normalize across blocks.
- **Why:** This is the reviewer's headline point. The repo already does it (`fingerprint→SNV`, `ch_stretch→area`, cross-fusion `none`), so the correct engineering action is a regression guard, not a rewrite.
- **Changes:** No production change expected. Add an assertion-style test only.
- **Deliverables:** test that:
  1. `normalize_spectra(..., region="fingerprint")` uses SNV; `region="ch_stretch"` uses area;
  2. the cross/whole path applies `none` (no joint rescale), so CH magnitude cannot dominate fingerprint after concatenation;
  3. block-identity column prefixes (`1300__`, `2850__`) are preserved end-to-end.
- **Tests:** `tests/test_preprocessing.py` (extend) or new `test_block_normalization.py`.
- **Validation:** `uv run pytest tests/test_preprocessing.py -v`.
- **Claim boundaries:** none.

### Phase C — Band-integral helper + named-index panel (primary features)

- **Objective:** (1) Factor duplicated trapezoidal band-integral logic into `band_features.py`; (2) build the pre-registered panel where **named biochemical indices are primary endpoints**, not secondary derived ratios.
- **Why:** Reviewer §4 — generic ratios dilute signal; named indices (`membrane_integrity`, `oxidative_index`, `protein_remodeling`, `tyrosine_environment`) are focused, interpretable, and publishable. Moving them into Phase C ensures all downstream statistics (Phases E–L) operate on the same primary feature set.
- **Changes:**
  - Create `band_features.py` with `block_columns()`, `band_integral()`, `ratio_from_integrals()`, and `build_named_index()` (handles multi-denominator sums like protein_remodeling). Port logic from `biochemical_ratios.py` / `covariate_models.py` unchanged in behavior.
  - Refactor `biochemical_ratios.py` and `covariate_models.py` to import from `band_features.py`.
  - In `preregistered_panel.py:build_panel_matrix(...)`: compute all integrals; if `lipid_carbonyl_1720` has `<2` channels, **automatically fall back** to `lipid_carbonyl_1720_fallback` (1700–1750) and log the switch in `preregistration.md`; compute the four named indices; write `05_confirmatory_panel/panel_feature_matrix.csv` with integrals + indices (indices listed first as primary columns).
  - Skip tyrosine/structural integrals if SNR gate fails; log, do not impute.
- **Deliverables:** `band_features.py`; `05_confirmatory_panel/panel_feature_matrix.csv`.
- **Tests:** `test_band_features.py`; `test_preregistered_panel.py::test_named_indices_are_primary_columns`; `::test_1720_fallback_when_narrow_window_too_small`.
- **Validation:** `uv run pytest tests/test_band_features.py tests/test_preregistered_panel.py -v`.
- **Claim boundaries:** indices are intensity ratios in arbitrary units; label as "region-overlapping" biochemical indices, not validated biomarkers.

### Phase D — Extreme-contrast endpoint redesign (additive)

- **Objective:** Add extreme-contrast endpoint columns alongside the existing adjacent-split groups.
- **Why:** Stronger contrasts raise effect size (reviewer §C) but reduce n (see §0). Adding, not replacing, lets the analysis report both and quantify the power tradeoff.
- **Changes:** In `config/eda.yaml analysis.clinical_endpoints` add `mild_threshold: 25` and `recovery_threshold: 2`. In `src/clinical/endpoints.py:build_clinical_endpoints` add:
  - `moca_severity_group_extreme`: `severe` if `moca_baseline < severity_threshold`, `mild` if `moca_baseline > mild_threshold`, else `<NA>` (the 20–25 middle is dropped);
  - `delta_moca_group_extreme`: `decline` if `delta_moca < decline_threshold`, `recovery` if `delta_moca > recovery_threshold`, else `<NA>`.
  - Extend `_group_counts` to include the two new columns so achieved n per group is always reported.
- **Deliverables:** two new group columns; updated `clinical_group_counts.csv`.
- **Tests:** update `tests/test_clinical_endpoints.py` — boundary values (exactly 20, 25, 0, 2) fall in the documented bucket / middle; counts table includes extreme groups.
- **Validation:** `uv run pytest tests/test_clinical_endpoints.py -v`.
- **Claim boundaries:** groups are operational cut-offs for contrast, not clinical diagnoses.

### Phase E — Mandatory anchor test + robust group contrasts

- **Objective:** (1) Run the **single mandatory anchor test** first; (2) for all panel features × extreme contrasts, compute Mann–Whitney U, Cliff's δ, bootstrap CI, and BH-FDR.
- **Why:** The anchor converts "exploratory only" into one falsifiable H₀/H₁. Robust contrasts on named indices provide graded evidence even when FDR fails.
- **Changes:**
  - Add `statistics.cliffs_delta(a, b)`.
  - In `robust_contrasts.py`, **`run_anchor_test()` runs first**: partial Spearman of `oxidative_index` vs `delta_moca` | age, sex; bootstrap CI; permutation p; Cliff's δ for trajectory_extreme contrast. Write `primary_anchor_result.csv` (no FDR — single pre-specified test).
  - Then `run_panel_contrasts(...)`: iterate **named indices + shape summaries** × contrasts; MWU p, Cliff's δ, Cohen's d, bootstrap CI, BH-FDR → `panel_contrast_tests.csv`.
- **Deliverables:** `primary_anchor_result.csv`; `panel_contrast_tests.csv`; `figures/panel_effect_sizes.png`, `panel_rainclouds.png`.
- **Tests:** `test_robust_contrasts.py::test_anchor_runs_before_panel_fdr`; `::test_cliffs_delta_known_cases`.
- **Validation:** `uv run pytest tests/test_statistics_effectsizes.py tests/test_robust_contrasts.py -v`.
- **Claim boundaries:** anchor result is one association test, not proof of oxidative mechanism; panel contrasts use FDR.

### Phase F — Permutation validation of contrasts

- **Objective:** Add a label-permutation p-value for each panel feature × contrast (Cliff's δ as the test statistic).
- **Why:** Permutation gives an assumption-light null and double-checks FDR on a tiny n (reviewer §D.2). Reuse existing permutation conventions (`random_state`, `n_permutations`).
- **Changes:** In `robust_contrasts.py`, add a seeded permutation loop shuffling the contrast labels and recomputing |Cliff's δ|; add `perm_p_value` column to `panel_contrast_tests.csv`. Prefer reusing helpers from `src/statistics/cluster_permutation.py` where structurally applicable.
- **Deliverables:** `perm_p_value` column.
- **Tests:** `test_robust_contrasts.py::test_permutation_pvalue_bounds` (∈ [1/(B+1), 1]; a strongly separated synthetic case yields small p; null case ≈ uniform).
- **Validation:** `uv run pytest tests/test_robust_contrasts.py -v`.
- **Claim boundaries:** permutation p does not establish causality or generalization.

### Phase G — Covariate control via partial Spearman

- **Objective:** For continuous endpoints, compute partial Spearman correlation of each panel feature with the endpoint, controlling for age and sex.
- **Why:** Reviewer §E — rank-based partial correlation removes age/sex dilution without distributional assumptions. This complements (does not replace) the existing regression-based `covariate_models.py`.
- **Changes:** Add `statistics.partial_spearman(x, y, covars) -> (rho, p, n)` (rank-transform x, y, covariates; residualize x and y on covariates via least squares on ranks; correlate residuals; document df adjustment). Use it in `preregistered_panel`/`robust_contrasts` to emit `panel_continuous_associations.csv` with raw Spearman, partial Spearman | age,sex, n, and BH-FDR.
- **Deliverables:** `panel_continuous_associations.csv`.
- **Tests:** `test_statistics_effectsizes.py::test_partial_spearman` (recovers raw Spearman when covariates are constant; collapses a synthetic age-confound when age is controlled).
- **Validation:** `uv run pytest tests/test_statistics_effectsizes.py -v`.
- **Claim boundaries:** adjustment reduces measured confounding only for measured covariates; residual confounding remains possible.

### Phase H — Heterogeneity / variance-homogeneity testing

- **Objective:** Per panel feature × contrast, test equality of variances between groups (Levene and Brown–Forsythe).
- **Why:** Reviewer §F — EV biology is heterogeneous; a difference in *spread* (not just mean) is biologically meaningful and currently untested in the repo.
- **Changes:** Add `statistics.variance_homogeneity_test(a, b, center="mean"|"median")` wrapping `scipy.stats.levene` (mean center = Levene; median center = Brown–Forsythe). Emit `panel_variance_homogeneity.csv` with both statistics, p-values, per-group variances, and BH-FDR.
- **Deliverables:** `panel_variance_homogeneity.csv`.
- **Tests:** `test_statistics_effectsizes.py::test_variance_homogeneity` (equal-variance synthetic → non-significant; unequal-variance synthetic → significant; Brown–Forsythe robust to a single outlier where Levene is not).
- **Validation:** `uv run pytest tests/test_statistics_effectsizes.py -v`.
- **Claim boundaries:** variance differences are descriptive heterogeneity signals, not subtype proof.

### Phase I — Clustering insight (reuse; optional UMAP+k-means)

- **Objective:** Relate panel features to existing exploratory subgroups; optionally add a UMAP+k-means supplementary view.
- **Why:** Reviewer §G. The repo's `disease_subgroups.py` (Ward + Gap statistic + bootstrap ARI) is already more rigorous than bare UMAP+k-means, so reuse it first and treat UMAP+k-means as an optional, clearly-labeled supplement.
- **Changes:** Add a small reporting step that summarizes panel-feature means/medians and endpoint distributions per existing cluster (no new clustering required for the core deliverable). If `analysis.disease_subgroups.umap_supplement.enabled` is set, add a seeded UMAP→k-means figure labeled "exploratory, unvalidated."
- **Deliverables:** cluster × panel-feature summary CSV; optional UMAP figure.
- **Tests:** smoke test that the summary runs on the disease-only smoke subset and produces the expected columns.
- **Validation:** `uv run scripts/run_eda.py --analysis-mode disease_only --range both --max-subjects 8`.
- **Claim boundaries:** clusters are exploratory structure with moderate stability (~ARI 0.5 historically); not validated subtypes.

### Phase J — Minimal interpretable model restricted to the panel

- **Objective:** Run the existing nested-CV ElasticNet / L1-logistic minimal-model machinery using only the pre-registered panel features (3–5 inputs), with permutation and bootstrap CIs.
- **Why:** Reviewer §H is already implemented in `minimal_panel.py`; the only addition is a panel-restricted feature source so the "minimal model" is genuinely the pre-registered panel rather than data-driven candidates.
- **Changes:** Add an optional `feature_source: preregistered_panel` path to `minimal_panel.run_minimal_panel` (or a thin wrapper) that feeds `panel_feature_matrix.csv`; keep nested CV, permutation, bootstrap, and the PLS-gate logic. Write `panel_minimal_model_performance.csv`.
- **Deliverables:** `panel_minimal_model_performance.csv`, coefficients CSV.
- **Tests:** reuse/extend `tests` for minimal panel with a synthetic panel matrix; assert permutation p and bootstrap CIs are emitted and that performance is reported even when null.
- **Validation:** `uv run pytest tests/test_minimal_panel.py -v` (if present) or the disease-only smoke run.
- **Claim boundaries:** report balanced accuracy / R² with permutation p and CI; negative or non-significant performance must be stated plainly (consistent with current review docs).

### Phase S — Cross-endpoint consistency (Tier 1 bridge)

- **Objective:** Test whether each feature shows the **same sign** across moca_baseline, moca_3m, and ΔMoCA.
- **Why:** Reviewer §8 — consistent features = biologically meaningful trends; feeds robustness score (Phase L).
- **Changes:** `endpoint_consistency.py`: per feature, count endpoints with same-sign partial Spearman ρ; flag `consistent` if ≥ `min_endpoint_agreement`. Output to `06_spectral_interpretability/endpoint_consistency/` **and** expose a summary column consumed by Phase L.
- **Deliverables:** `06_spectral_interpretability/endpoint_consistency/endpoint_consistency.csv`.
- **Tests:** `test_endpoint_consistency.py`.
- **Validation:** `uv run pytest tests/test_endpoint_consistency.py -v`.
- **Claim boundaries:** consistency increases credibility; not external replication.

### Phase L — Graded robustness ranking (Tier 1 capstone)

- **Objective:** Replace binary significance with a **graded evidence score** per feature, enabling a non-null narrative when FDR fails.
- **Why:** Reviewer §7 — "nothing significant" becomes "strongest trends ranked by converging evidence."
- **Changes:** Create `robustness_ranking.py:compute_robustness_scores(...)` aggregating, per named index and shape feature:
  1. `|Cliff's δ| > effect_size_threshold` (+1)
  2. bootstrap CI for δ excludes 0 (+1)
  3. `perm_p_value < perm_p_threshold` (+1)
  4. same-sign association across ≥ `min_endpoint_agreement` of {moca_baseline, moca_3m, delta_moca} (+1)
  Emit `feature_robustness_ranking.csv` with `robust_score` (0–4), component flags, and rank. **Runs after Phases E–G and S** (needs consistency input).
- **Deliverables:** `05_confirmatory_panel/feature_robustness_ranking.csv`.
- **Tests:** `test_robustness_ranking.py` (score arithmetic; rank order deterministic; anchor feature scored separately).
- **Validation:** `uv run pytest tests/test_robustness_ranking.py -v`.
- **Claim boundaries:** robust_score is graded exploratory evidence, not a validated finding tier.

---

## Tier 2 — Structure-Aware Biochemical Interpretability (Phases M–U)

These phases add **multi-representation** chemical expressiveness. All read the subject
feature matrix (`1300__*`, `2850__*`) from Biological PCA exports. Outputs go to
`06_spectral_interpretability/<phase>/` via `get_spectral_interpretability_dir()`.
A summary copy of shape features also lands in `05_confirmatory_panel/panel_peak_shape_features.csv`
for the Tier-1 report. Unless noted, associations are exploratory: effect sizes + FDR.

### Phase M — Spectral-shape analysis (shift + FWHM + asymmetry)

- **Objective:** Fit local Gaussian or pseudo-Voigt peaks at 1720, 1555, 1525; extract centroid, FWHM, and **peak asymmetry** per subject.
- **Why:** Reviewer §1 — **directly mitigates weak intensity-only signal**; shape differences (C=O shift, amide width, carotenoid asymmetry) may be detectable when integrals are flat.
- **Changes:** Create `spectral_shape.py`:
  - `fit_local_peak(...) -> (centroid, fwhm, asymmetry, amplitude, r2, snr)` via `scipy.optimize.curve_fit`
  - **SNR gate:** skip subjects below `min_snr_for_peak_fit`; log count
  - **Calibration floor:** flag shifts within `calibration_floor_cm` as `not_interpretable`
  - Bootstrap CIs on centroid/FWHM/asymmetry
  - Write `peak_shape_features.csv`; copy summary to `05_confirmatory_panel/panel_peak_shape_features.csv`; endpoint associations with FDR
- **Deliverables:** `06_spectral_interpretability/peak_shape/peak_shape_features.csv`, `peak_shape_associations.csv`, `panel_peak_shape_features.csv`.
- **Tests:** `test_spectral_shape.py` (synthetic Gaussian recovery; asymmetry computed; SNR skip; sub-floor flag).
- **Validation:** `uv run pytest tests/test_spectral_shape.py -v`.
- **Claim boundaries:** shifts above calibration floor with CI only; asymmetry is descriptive shape, not mechanism.

### Phase N — NMF decomposition (latent biochemical components)

- **Objective:** Decompose fingerprint-region spectra (optionally whole) into 3–5 NMF components; correlate subject weights with ΔMoCA and severity groups.
- **Why:** Reviewer §2 — converts high-dimensional noise into low-dimensional lipid/protein/carotenoid-like profiles.
- **Changes:** Create `nmf_decomposition.py`:
  - `sklearn.decomposition.NMF` with `n_components=3`, `alpha_W/H` sparsity from config, `n_init` multi-seed
  - Stability metric across inits; keep most stable solution
  - Annotate components by band-window overlap; **always visualize component spectra**
  - Label carotenoid-like components as **"carotenoid-like / lipoprotein-associated"**
  - Weight–endpoint Spearman + partial Spearman + FDR
- **Deliverables:** `nmf_components.csv`, `nmf_weights_per_subject.csv`, `nmf_weight_associations.csv`, `figures/nmf_component_spectra.png`.
- **Tests:** `test_nmf_decomposition.py` (non-negativity; stability metric; sparsity enforced; reproducible seed).
- **Validation:** `uv run pytest tests/test_nmf_decomposition.py -v`.
- **Claim boundaries:** components are statistical factors, not pure molecular species.

### Phase O — Spectral entropy / dispersion fingerprint

- **Objective:** Compute per-subject spectral entropy over the fingerprint region and associate with endpoints.
- **Why:** Reviewer §3 — strongest missing EV biology signal; high entropy = disordered EV population; captures distributional change missed by means.
- **Changes:** In `spectral_descriptors.py`:
  - `p_i = I_i / Σ I_i` on fixed SNV-normalized fingerprint intensities (≥0)
  - `entropy = −Σ p_i log(p_i)` (natural log documented)
  - Secondary dispersion metric (Gini or IQR/median) for normalization robustness
  - Partial Spearman vs endpoints + FDR
- **Deliverables:** `spectral_entropy.csv`, `spectral_entropy_associations.csv`, entropy distribution figure.
- **Tests:** `test_spectral_descriptors.py::test_spectral_entropy`.
- **Claim boundaries:** dispersion descriptor, not thermodynamic entropy.

### Phase P — Difference-spectrum signatures (publication requirement)

- **Objective:** Compute Δspectrum = mean(severe) − mean(mild) and mean(decline) − mean(recovery); smooth, annotate peaks, produce journal figure.
- **Why:** Reviewer §5 — required for publication-quality interpretation; visual biochemical signature when tabular stats are weak.
- **Changes:** `discriminant_spectra.py`: block-separated difference spectra; annotate with `get_raman_bands()`; signed-effect map (`sign(Δ)×|effect|` per wavenumber).
- **Deliverables:** `difference_spectrum.csv`, `signed_effect_map.csv`, annotated figures (fingerprint and CH panels separate).
- **Tests:** `test_discriminant_spectra.py`.
- **Validation:** `uv run pytest tests/test_discriminant_spectra.py -v`.
- **Claim boundaries:** descriptive contrasts; annotated peaks are region-overlapping assignments.

### Phase Q — Cross-feature coupling + biochemical regimes

- **Objective:** Quantify coupling between band integrals/indices and assign biochemical regimes per subject.
- **Why:** Reviewer §6 — EV remodeling is coupled, not independent; multivariate structure stronger than isolated features.
- **Changes:** Create `feature_coupling.py`:
  - Cross-subject Spearman matrix: corr(1720,1555), corr(1525,1720), corr(1555,925 region), plus all named indices — bootstrap CIs
  - Ratio–ratio plane: `oxidative_index` vs `membrane_integrity_index` → `lipid_dominant` / `oxidative_dominant` / `protein_dominant` regimes
- **Deliverables:** `feature_coupling_matrix.csv`, `ratio_ratio_regimes.csv`, coupling heatmap + regime plot.
- **Tests:** `test_feature_coupling.py`.
- **Claim boundaries:** coupling describes co-variation, not biochemical pathways.

### Phase R — Index-space clustering (optional supplement)

- **Objective:** Cluster subjects on named indices only; test clusters vs MoCA and lesion type.
- **Why:** Reviewer §7 (original) — interpretable disease states without black-box spectral clustering.
- **Changes:** Reuse `disease_subgroups.py` Ward + Gap + bootstrap ARI on standardized index matrix. Lower priority than Phases M–Q.
- **Deliverables:** `index_cluster_assignments.csv`.
- **Claim boundaries:** exploratory clusters, not validated subtypes.

### Phase K — Four-domain biochemical narrative + report (Tier 1 synthesis)

- **Objective:** Write `panel_report_section.md` organized into **four mandatory narrative domains**, each synthesizing Tier-1 + Tier-2 evidence (effect size, CI, robustness score, shape/NMF/entropy where available):
  1. **Membrane remodeling** — regions overlapping lipid carbonyl (~1720) + CH (2850–2950); include `membrane_integrity_index`, peak shifts at 1720, difference-spectrum lipid panels.
  2. **Protein structural heterogeneity** — regions overlapping protein bands (~1555, 821–980); include `protein_remodeling_index`, FWHM at 1555, `tyrosine_environment_index`.
  3. **Oxidative imbalance** — regions overlapping carotenoid-like / lipoprotein-associated signals (~1525, 1156); include `oxidative_index`, anchor result, NMF carotenoid-like component weights, spectral entropy.
  4. **System-level EV disruption** — heterogeneity (entropy, variance tests), feature coupling matrix, ratio–ratio regimes, endpoint consistency.
- **Why:** Reviewer §10 — a four-domain scaffold produces a publishable biochemical story even when binary significance is absent, while the fixed wording rule prevents overclaiming.
- **Changes:** Write `05_confirmatory_panel/panel_report_section.md` via `get_confirmatory_panel_dir()`, consuming Phases E–L and linking to Tier-2 artifacts in `06_spectral_interpretability/`. Classify each feature using `feature_robustness_ranking.csv` tiers: `strong_trend` (robust_score ≥3), `moderate_trend` (2), `weak/unstable` (≤1) — **not** binary significant/non-significant. Phase-14 `candidate_region_report.md` links to this section; it does not absorb the files. **Runs after Phases M–Q and L** so shape/NMF/entropy/coupling evidence is available.
- **Deliverables:** `panel_report_section.md`; `README.md`; link from Phase-14 report.
- **Tests:** report unit test — four domains present; carotenoid language uses "carotenoid-like / lipoprotein-associated"; string-blocklist rejects biomarker/diagnostic terms.
- **Validation:** `uv run pytest tests/test_candidate_region_report.py -v` (extend).
- **Claim boundaries:** always "region overlapping tentatively assigned … bands"; never causal/diagnostic/predictive language.

### Phase U — Journal figure set (*Spectrochimica Acta A*)

- **Objective:** Assemble 5 publication figures from Tier-1 + Tier-2 outputs.
- **Proposed set:**
  1. Cohort overview + named-index rainclouds by contrast
  2. Effect-size forest + **robustness ranking** (anchor highlighted)
  3. Difference spectrum + signed-effect map (block-separated)
  4. NMF component spectra + spectral entropy vs ΔMoCA
  5. Feature coupling heatmap + ratio–ratio regimes
- **Deliverables:** `06_spectral_interpretability/figures/fig1_`–`fig5_`; indexed in README + Phase K report.
- **Claim boundaries:** captions use region-overlapping language only.

### Phase V — Pipeline wiring, tests, reproducibility

- **Objective:** Wire all new phases into `disease_only_pipeline.py` behind `_phase_enabled` gates, ensure manifests/snapshots are emitted, and run the full suite.
- **Why:** Keep one orchestrator, reproducible artifacts, and green tests (`CLAUDE.md` §3, §8).
- **Changes:** In `src/analysis/disease_only_pipeline.py`, after `minimal_panel`/`publication_report`, call Tier-1 panel phases (C–J, S, L), Tier-2 phases (M–Q, optional R), synthesis (K, U), each gated on its own `enabled` flag (default Tier-2 enabled but skippable). Confirm `run_manifest.json`, `data_manifest_snapshot.json`, `config_snapshot.yaml` still written. Add new artifacts to `.gitignore` patterns if needed.
- **Deliverables:** wired phases; full `05_confirmatory_panel/` and `06_spectral_interpretability/` output sets, each a sibling of `04_cross_region_summary/` under `Results_intra_disease/`.
- **Tests:** `uv run pytest -v` (whole suite green).
- **Validation:**

```bash
uv run pytest -v
python -m py_compile src/analysis/preregistered_panel.py src/analysis/robust_contrasts.py src/analysis/band_features.py \
  src/analysis/robustness_ranking.py src/analysis/spectral_shape.py src/analysis/nmf_decomposition.py \
  src/analysis/spectral_descriptors.py src/analysis/feature_coupling.py src/analysis/discriminant_spectra.py \
  src/analysis/endpoint_consistency.py
uv run scripts/run_eda.py --analysis-mode disease_only --range both --max-subjects 8
```

- **Claim boundaries:** the smoke run with `--max-subjects 8` is a wiring check only; never interpret its numbers.

---

## 6. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Extreme contrasts shrink n below `min_subjects_per_group` | High | Phase E–J skip / unstable | Report achieved n; keep adjacent-split groups as fallback; annotate minimum detectable effect |
| **1720 edge-band instability** (<2 channels after edge-trim) | Medium | Lipid carbonyl integral/shape missing | Phase C **automatic fallback** to 1700–1750 window; log switch in `preregistration.md`; do not silently impute |
| **Peak fitting instability** (low SNR, poor GoF) | High | Spurious shift/FWHM claims | `min_snr_for_peak_fit` gate; skip low-quality spectra; bootstrap CI + calibration floor; sub-floor shifts marked `not_interpretable` (Phase M) |
| **NMF component non-interpretability** | High | Spurious "latent profiles" | Fixed small k (3), L1 sparsity, multi-seed stability metric, **always visualize component spectra**; label exploratory; FDR on weight associations (Phase N) |
| **Carotenoid misinterpretation** | Medium | Overclaiming oxidative mechanism | Always describe as **"carotenoid-like signal / lipoprotein-associated"**; string-blocklist in Phase K report test |
| Reviewer's "multiplicity removed" assumption leaks into text | Medium | Overclaiming | FDR + permutation retained everywhere; explicit §0 statement |
| Carotenoid/"oxidative"/"protein" labels read as mechanism | Medium | Overclaiming | Neutral region language enforced by report string-blocklist test (Phase K) |
| Refactor of band-integral helper changes existing numbers | Low | Regression in Phases 10/7 | Behavior-preserving refactor + existing tests must pass before merge |
| New phase slows the disease-only run | Low | Runtime | Gate behind `enabled`; panel is tiny (≤13 features) |
| Peak shift below wavenumber calibration floor read as real | Medium | Overclaiming structural change | Calibration-floor flag + bootstrap CI + GoF gate; sub-floor shifts marked `not_interpretable` (Phase M) |
| Spectral entropy sensitive to normalization choice | Medium | Non-reproducible heterogeneity claim | Fixed documented normalized representation + a second dispersion metric (Phase O) |
| Tier-2 layers re-inflate multiple testing | High | Erosion of Tier-1 rigor | Single pre-specified anchor frozen in Phase A, tested in Phase E (uncorrected); all other outputs use graded robustness or descriptive effect sizes + FDR |
| Tyrosine/structural-protein bands (851/821/925/980) too weak | Medium | Noisy indices | SNR/width gate; skip+log if unmet; never impute (Phases C, M) |
| Difference/signed-effect maps mixing block normalizations | Medium | Misleading magnitudes | Render fingerprint and CH in separate panels; mark block boundary (Phase P) |

---

## 7. Definition of Done

### Tier 1 (statistical core)

- [ ] Phases A–J, S, L, and K implemented, each with passing tests.
- [x] Block-aware normalization invariants locked by test (Phase B).
- [x] `preregistration.md` frozen before result tables exist (Phase A).
- [ ] Mandatory anchor hypothesis (oxidative index vs ΔMoCA) tested in Phase E before panel FDR.
- [ ] Every panel result reports effect size + CI + FDR q + permutation p + partial-Spearman + variance test + achieved n.

### Tier 2 (structure-aware biochemical interpretability)

- [ ] Phases M–U implemented (M, N, O, P, Q required; R optional), each with passing tests.
- [ ] Tier-2 outputs are descriptive with effect sizes + FDR; only the Phase A/E anchor is uncorrected confirmatory.
- [ ] Peak shifts reported with bootstrap CIs and flagged against the calibration floor (Phase M).
- [ ] NMF reports a stability metric, sparsity, component spectra figures, and is labeled exploratory (Phase N).
- [ ] Spectral entropy uses a fixed documented normalization + a second dispersion metric (Phase O).
- [ ] Difference/signed-effect figures separate fingerprint and CH blocks (Phase P).
- [ ] `feature_robustness_ranking.csv` aggregates effect + CI + perm + endpoint consistency (Phase L).
- [ ] Four-domain narrative report links Tier-1 and Tier-2 artifacts (Phase K).
- [ ] 4–5 journal figures assembled and indexed in the report (Phase U).

### Cross-cutting

- [ ] `uv run pytest -v` green; `uv run pytest tests/test_biological_pca.py` green.
- [ ] Disease-only smoke run completes and writes `Results_intra_disease/05_confirmatory_panel/` and `Results_intra_disease/06_spectral_interpretability/` as sibling folders alongside `04_cross_region_summary/`, plus run manifests.
- [ ] All reports/captions contain no diagnostic / predictive / biomarker / causal language (enforced by test).
- [ ] No new top-level entry point; outputs routed through existing helpers; generated artifacts git-ignored.

---

## 8. Implementation Progress

Track completed phases here. **Next iteration starts at Phase E.**

### Tier 1 — Statistical core

- [x] **Phase A** — Pre-registration config block (`config/eda.yaml` `analysis.preregistered_panel`; `get_confirmatory_panel_dir()` / `get_spectral_interpretability_dir()` in `src/__init__.py`; `src/analysis/preregistered_panel.py::write_preregistration_snapshot`; tests in `tests/test_preregistered_panel.py::test_preregistration_snapshot_is_frozen`)
- [x] **Phase B** — Block-aware normalization regression guard (`tests/test_block_normalization.py`; no production change)
- [x] **Phase C** — Band-integral helper + named-index panel (`src/analysis/band_features.py`; `build_panel_matrix` / `run_preregistered_panel`; refactored `biochemical_ratios.py`, `covariate_models.py`; tests in `tests/test_band_features.py`, `tests/test_preregistered_panel.py`)
- [x] **Phase D** — Extreme-contrast endpoints (`mild_threshold` / `recovery_threshold` in config; `moca_severity_group_extreme`, `delta_moca_group_extreme` in `src/clinical/endpoints.py`; tests in `tests/test_clinical_endpoints.py`)
- [ ] **Phase E** — Mandatory anchor test + robust group contrasts
- [ ] **Phase F** — Permutation validation of contrasts
- [ ] **Phase G** — Covariate control via partial Spearman
- [ ] **Phase H** — Heterogeneity / variance-homogeneity testing
- [ ] **Phase I** — Clustering insight (reuse; optional UMAP supplement)
- [ ] **Phase J** — Minimal interpretable model restricted to the panel
- [ ] **Phase S** — Cross-endpoint consistency
- [ ] **Phase L** — Graded robustness ranking
- [ ] **Phase K** — Four-domain biochemical narrative report

### Tier 2 — Structure-aware interpretability

- [ ] **Phase M** — Spectral-shape analysis (shift + FWHM + asymmetry)
- [ ] **Phase N** — NMF decomposition
- [ ] **Phase O** — Spectral entropy / dispersion fingerprint
- [ ] **Phase P** — Difference-spectrum signatures
- [ ] **Phase Q** — Cross-feature coupling + biochemical regimes
- [ ] **Phase R** — Index-space clustering (optional)
- [ ] **Phase U** — Journal figure set

### Wiring & reproducibility

- [ ] **Phase V** — Pipeline wiring in `disease_only_pipeline.py`, manifests, full smoke run

### Notes (Phases A–D, 2026-06-23)

- `run_preregistered_panel()` writes `05_confirmatory_panel/preregistration.md` and `panel_feature_matrix.csv` but is **not yet wired** into `disease_only_pipeline.py` (Phase V).
- Path helpers registered; Tier-2 subdirs ready via `get_spectral_interpretability_dir()`.
- Phase A–D tests: `uv run pytest tests/test_band_features.py tests/test_preregistered_panel.py tests/test_block_normalization.py tests/test_clinical_endpoints.py -v` (22 tests green with biochemical/covariate regression tests).
