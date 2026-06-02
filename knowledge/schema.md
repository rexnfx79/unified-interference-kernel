# Wiki Schema

This project follows the [LLM Wiki](https://github.com/nashsu/llm_wiki) pattern (Karpathy methodology). The wiki lives in `knowledge/` and can be opened as an LLM Wiki desktop project or maintained by Cursor via `AGENTS.md`.

## Page Types

| Type | Directory | Purpose |
|------|-----------|---------|
| entity | `wiki/entities/` | Named objects: people, papers, operators, models |
| concept | `wiki/concepts/` | Theories, mathematical objects, phenomena (**knowledge islands**) |
| source | `wiki/sources/` | Summaries of ingested documents |
| query | `wiki/queries/` | Open questions under active investigation |
| comparison | `wiki/comparisons/` | Side-by-side analysis of competing approaches |
| synthesis | `wiki/synthesis/` | Cross-cutting maps, bridge diagrams, proof sketches |
| overview | `wiki/overview.md` | Auto-updated global summary |

## Naming Conventions

- Files: `kebab-case.md`
- Concepts: descriptive noun phrases (`riemann-zeta-function.md`)
- Sources: `author-year-slug.md` or `repo-artifact-slug.md`
- Queries: question as slug (`does-phase-structure-imply-spectral-operator.md`)

## Frontmatter

All pages must include:

```yaml
---
type: concept | entity | source | query | comparison | synthesis | overview
title: Human-readable title
tags: []
related: []
status: established | conjecture | speculative | refuted | open | philosophical | deprioritized | phenomenological
created: YYYY-MM-DD
updated: YYYY-MM-DD
---
```

Source pages add:

```yaml
sources:
  - raw/sources/filename.md
```

Concept pages that are **knowledge islands** should include:

```yaml
island: true
bridge_tags: [spectral, information, flavor, zeta, qm, primes, p-adic]
approach: arithmetic | spectral | quantum | effective   # which side in multi-sided framework
plausibility: pursue | watch | deprioritize | dead | established   # see plausibility-register
```

## Plausibility Register

Every bridge claim must appear in [[plausibility-register]] with a verdict. Failed tests → **dead** or **refuted**, not silent reinterpretation.

## Multi-Sided Framework

Approach math↔reality from three sides — see [[multi-sided-bridge-framework]]:

- **A:** primes, p-adics, zeta
- **B:** Hilbert spaces, operators
- **C:** QED/QM → information
- **D:** effective readout (interference kernel repo)

Recommended order: **C → B → (maybe) A → explain D**.

## Cross-referencing Rules

- Use `[[wikilink]]` syntax (slug without `.md`)
- Every concept and entity appears in `wiki/index.md`
- Query pages link to all concepts they touch
- Synthesis pages cite contributing sources via `related:` and inline links
- When updating a page, bump `updated:` in frontmatter

## Bridge Tags

Use consistent tags to mark cross-domain connections:

| Tag | Domain |
|-----|--------|
| `spectral` | Operators, eigenvalues, Hilbert–Polya |
| `zeta` | Riemann zeta, L-functions, zeros |
| `information` | It-from-bit, entropy, compression |
| `flavor` | Yukawa, CKM, PMNS, this repo's kernel |
| `chaos` | Quantum chaos, GUE statistics |
| `qm` | QED, quantum mechanics, path integrals |
| `primes` | Prime numbers, arithmetic physics |
| `p-adic` | p-adic analysis, adelic programs |
| `holographic` | AdS/CFT, boundary/bulk information |

## Ingest Workflow

1. Add raw document to `knowledge/raw/sources/`
2. Tell the agent: **"Ingest `<filename>` into the knowledge wiki"**
3. Agent reads source, updates concept/entity pages, source summary, `index.md`, `log.md`, `overview.md`
4. Flag contradictions with existing pages; create query pages if unresolved

## Query Workflow

1. Ask questions against the wiki (not raw sources alone)
2. File valuable answers to `wiki/queries/` or `wiki/synthesis/`
3. Re-ingest saved queries to extract new entities/concepts

## Lint Workflow

Periodically request: **"Lint the knowledge wiki"**

Check for: contradictions, orphan pages (no inbound links), missing concept pages for mentioned terms, stale claims vs. repo test results, sparse bridge clusters.

Run from repo root:

```bash
python3 scripts/lint_wiki_manifest.py   # SHA256 snapshot integrity
python3 scripts/lint_wiki_links.py      # [[wikilink]] target exists
python3 scripts/lint_wiki_links.py --orphans   # optional orphan report
```

Wikilink linter: scans `knowledge/wiki/**/*.md` for `[[slug]]` (kebab-case slugs only), verifies a matching `slug.md` exists anywhere in the wiki tree. Skips fenced/inline code and non-slug targets (e.g. `[[src/foo.py]]`).

## Log Format

`wiki/log.md` — reverse chronological:

```
## YYYY-MM-DD | ingest | Source Title

- Pages created/updated: [[concept-a]], [[concept-b]]
- Open question raised: [[query-slug]]
```

## Link to Parent Repo

Computational tests live outside the wiki:

- Code: `../src/`, `../scripts/`, `../diagnostics/`
- Findings: `../SCIENTIFIC_FINDINGS.md`, `../manuscript.tex`
- Manifest: `manifest.yaml` — run `python3 scripts/lint_wiki_manifest.py` from repo root after canonical file changes

When a wiki conjecture suggests a test, note it in the query page and optionally add a script reference.

## Obsidian / LLM Wiki App

- Open `knowledge/` as an LLM Wiki project (File → Open Project)
- Or open in Obsidian for graph view; `.obsidian/` may be auto-generated by the app
- Local API: enable in LLM Wiki Settings → API Server (`127.0.0.1:19828`)
- Agent skill: `npx skills add https://github.com/nashsu/llm_wiki_skill.git --skill llm_wiki_skill`
