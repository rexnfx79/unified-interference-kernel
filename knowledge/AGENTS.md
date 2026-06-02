# LLM Wiki Agent Instructions

You maintain the **conjecture notebook** in this directory. Read `purpose.md` first, then `schema.md`.

## Core Mandate

Connect **math to reality** from three sides (see `wiki/synthesis/multi-sided-bridge-framework.md`):

- **A — Arithmetic:** primes, p-adics, zeta
- **B — Spectral:** Hilbert spaces, operators
- **C — Quantum / info:** QED, QM, it-from-bit
- **D — Effective:** interference kernel (parent repo) — explain, don't worship

**Always update `wiki/synthesis/plausibility-register.md`** when a bridge succeeds, fails, or is implausible.

## Hub Pages

1. `wiki/synthesis/multi-sided-bridge-framework.md` — strategy + gap-crossing protocol
2. `wiki/synthesis/plausibility-register.md` — dead / pursue / deprioritize verdicts
3. `wiki/synthesis/information-reality-bridge-map.md` — island catalog

## Ingest

1. Read `purpose.md`, `schema.md`, `wiki/index.md`
2. Analyze source; note **failures** and **implausible** claims explicitly
3. Update concept/source pages, `index.md`, `log.md`, `overview.md`, **plausibility-register**
4. Status labels include: `philosophical`, `deprioritized`, `phenomenological`, `refuted`, `dead` (in register)

## Query

Read wiki pages first. When synthesizing cross-domain links, run the **gap-crossing checklist** (framework page). Save adversarial findings to `wiki/queries/`.

## Lint

- Contradictions vs `../SCIENTIFIC_FINDINGS.md` / [[repo-scientific-findings]]
- Stale snapshots: `python3 scripts/lint_wiki_manifest.py` from repo root
- Broken wikilinks: `python3 scripts/lint_wiki_links.py` (optional `--orphans`)
- Bridges marked pursue without falsifier
- Dead bridges still drawn as live in diagrams
- Orphan pages, broken wikilinks

## Honest Constraints

- Kernel parameters **do not transfer** across sectors (refuted universality)
- Zeta→flavor direct: **dead** (`wiki/queries/why-not-zeta-flavor-numerology.md`)
- 3×3 GUE test: **dead**
- Primary pursue path: **QED/QM → information** (`wiki/concepts/qed-qm-information.md`)

## Parent Repo

Code: `../src/`, `../scripts/`, `../diagnostics/`. Link tests to query pages.

## LLM Wiki App

Submodule: `../tools/llm_wiki`. Open this `knowledge/` folder as project.
