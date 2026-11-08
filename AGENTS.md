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
