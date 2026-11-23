<!-- graft:start -->
## Graft — repo context graph

This repo is indexed in `graft/`: small linked markdown nodes that explain each
system and carry exact file:line spans, kept in sync with the code through git.

For ANY task here — understanding how something works, finding where code lives,
or scoping a change — get context from the graph before grepping or opening
source files. Re-ask freely (it's cheap) and reuse literal identifiers you
already have (symbol, error string, file name) as the query. New to this repo?
Run `graft map` first — a token-budgeted orientation (dir clusters, hubs,
hotspots), no LLM, no key.

- Run `graft ask "<your question>" --source` → ranked nodes with the relevant
  code spans inlined (each hit's ≤8-line crux by default; `--full` for whole
  definitions when the crux isn't enough). Match the tool to the task shape:
  for understanding or editing, the top node IS the answer — cite its
  `covers:` file:line spans and edit straight from `--source`. For
  exhaustive tasks ("every occurrence / every caller of this pattern"), ranked
  results are top-N, not complete — run `graft grep "<literal>"` instead
  (exhaustive over indexed files, grouped by enclosing symbol), falling back
  to raw `grep -rn` only for unindexed files.
- `graft skeleton <file>` → every definition's signature + span, ~10× cheaper
  than reading the file; use it to skim an API surface.
- `graft callers <symbol>` gives precomputed, exact edges — who calls this.
  Add `--direction out` for what it calls, or `--depth N` to walk
  transitively for the full blast radius. For structural questions, skip
  ranking and use this directly.
- Or browse: `graft/INDEX.md` lists every node; follow the links.
- Monorepos and folders of multiple repos rank fairly across sub-projects —
  hits carry `[scope/]` labels naming which one they're from. Narrow with
  `graft ask "<task>" --in <scope>/` once you know where you're working.

If a returned span is truncated ("+N more lines"), open the file at that exact
range before finalizing. Only open source files when a node genuinely lacks a
needed detail, and then at the exact file:line the node points to — never
re-read whole files.

After big code changes, refresh the graph with `graft build` (deterministic,
no API key, $0).
<!-- graft:end -->

## Research decisions and wording

- Before suggesting a research extension, read the dated idea history in `AGENT_LEARNINGS.md` and the active design in `docs/research_design.md`. Do not re-propose a recorded idea unless new evidence addresses its stated limitation; identify that evidence explicitly.
- Maintain one active research design. When replacing a proposal, first preserve its question, method, evidence status, limitations, and conditions for reconsideration in the idea history. Removing a draft must not erase the research decision.
- The words `null`, `substantive`, `arms`, and `bank` and their ordinary prose inflections are prohibited in authored prose, comments, docstrings, captions, and report labels. Apply the existing plain-language rules to other filler; do not invent an unspecified vocabulary blacklist.
- Use the meaning-specific alternative: `no-association hypothesis`, `reference distribution`, `no detected association`, `specific contribution`, `study groups`, or `data repository`. Do not mechanically replace a word when that would change the scientific meaning.
- Exceptions are the prohibition list itself, exact code identifiers, filenames, paths, data-format tokens, original citation titles, and verbatim quotations. Preserve public interfaces, parsing rules, and stored formats. Generated research outputs and generated Graft cards are not manually rewritten for style; update their maintained text sources instead.
- Check wording and obsolete proposal links before completing documentation work. Record aggregate evidence with its date, source, and verification limits; keep private observations and task transcripts out of the learnings file.

## Manuscript ML engineering and verification

- Before rerunning equivalence, join, or timing benchmarks on `src/analysis/paired_lipid/` or `manuscript_ml/`, check the dated verified-facts tables in `AGENT_LEARNINGS.md` (2026-11-09 and 2026-11-23). Re-test only paths touched by subsequent edits, against a reference path that shares no fast-path code, and never evaluate real clinical outcomes without explicit approval.
- Keep numerical stability guards across Gram-space and batched paths: pre-center sample Gram blocks via `_finite_column_mean` and enforce `mu^2 <= 1e2 * tr(G)` in both `WeightedSpectralPreprocessor.fit` and `_prep_mfa_pca_batch`; guard normal-equation residualization against `kappa^2 * eps * amp` amplification with exact SVD fallback; keep scale-invariant `1e-14 * max_denom` and finite `ValueError` checks in `PairedMoments.solve_directions`; keep the scale-aware `t_sq_floor` early-stop guard in `batched_pls_sse_and_preds`; isolate corrupted Schur Cholesky/LDL^T slices per draw as `NaN`; and require `(cdf > 0.0)` in vectorized weighted-median selection.
- Preserve estimand parity across fast and fallback paths: keep missing covariates and `single_sample` downstream mode on the vectorized batched engine, resolve `single_sample_downstream` via `_effective_single_sample_downstream`, propagate explicit `single_sample_downstream=False` overrides to `base_eval` and all fallback draws, and validate `PreparedExternalTest` downstream mode and cached preset in `_prepare_external_test`.
- Preserve GPU, parallel-worker, and Windows I/O safeguards: gate every 3D `cupyx.scipy.linalg.solve_triangular` call on `_cupyx_supports_batched_triangular()`, keep GPU division index arrays device-resident, drop `raw_cov_frame` in `PreparedExternalTest.from_dataset(..., preset=...)` after caching the float64 array, keep `single_thread_blas=True` inside parallel `joblib` calibration workers, and route all atomic file writes (`.csv`, `.parquet`, `.txt`, `.json`, `.npz`) through `src/analysis/io_utils.py`. Do not re-propose fusing two external `25,000 x 1,802` ridge projections into a 2-column GEMM without new profiling evidence.
