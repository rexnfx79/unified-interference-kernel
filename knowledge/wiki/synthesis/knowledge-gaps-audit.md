---
type: synthesis
title: Knowledge Gaps Audit (Adversarial)
tags: [meta]
related:
  - proven-vs-conjecture-ledger
  - plausibility-register
  - multi-sided-bridge-framework
status: open
created: 2026-06-01
updated: 2026-06-01
---

# Knowledge Gaps Audit (Adversarial)

Post-ingest review: what is **solid**, what is **broken**, and what is **missing** for the math↔reality program.

## Grade: B+ (post-implementation, June 2026)

| Area | Grade | Notes |
|------|-------|-------|
| Repo phenomenology ingested | A | Diagnostics, analysis summary, pareto, overlap test |
| Failure honesty | A | Dead bridges marked |
| QED → information path | B− | Bekenstein + Preskill curated; still not primary papers |
| Arithmetic (primes, zeta, p-adics) | B− | Montgomery + Connes curated; explicit formula added |
| Internal consistency | B+ | Status fixes; boundary report ingested |
| Operational integrity | A− | manifest + link linter (`lint_wiki_links.py`) |

## Resolved since initial audit

- [x] `interference-kernel` status → phenomenological
- [x] Boundary report ingested + [[similar-fitted-scales-vs-transfer]]
- [x] `manifest.yaml` + lint script
- [x] Bekenstein / Preskill curated ingests
- [x] ANALYSIS_SUMMARY ingested
- [x] Pareto envelope ingested
- [x] Split-fermion overlap test (r=0.99996)
- [x] [[neutrino-observables-gap]] documented → **resolved** (PMNS in observables.py)
- [x] S(ρ_Y) information measure — `flavor_information.py` + diagnostic
- [x] Split-fermion overlap extended (4 geometries, Yu+Yd)
- [x] Montgomery / Connes curated ingests
- [x] Wiki link linter

## Remaining gaps (open)

| Artifact | Why it matters | Priority |
|----------|----------------|----------|
| `BOUNDARY_ANALYSIS_REPORT.md` | Pareto shapes; **misleading "universal params"** headline | **High** |
| `ANALYSIS_SUMMARY.md` | Survivor rates, Z-scores | High |
| `OPTION_B_COMPLETE.md` | Option B scope / completion claims | Medium |
| `src/alternative_kernels.py` | Clockwork vs Gaussian — only summarized in diagnostics | Medium |
| `scripts/02_pareto_envelope_comparison.py` | Envelope robustness (generalized p) | Medium |
| Neutrino/PMNS extraction code path | **Resolved** — `observables.py` + tests | — |
| `tests/verify_clockwork_*.py` | Clockwork verification details | Low |

---

## Missing External Knowledge (Goal-Critical Gaps)

These are **pertinent to connecting math with reality** but absent as concept pages or ingested sources:

### Approach C (QED / information) — largest gap

| Missing topic | Why needed |
|---------------|------------|
| [[holographic-principle]] | Bekenstein bound cited but not developed; links info ↔ geometry |
| Decoherence / einselection | Operational it-from-bit; only mentioned inline |
| RG as information flow | Query only; no concept page |
| Path-integral phase interference | Direct QED link to kernel phase — not standalone |
| Fisher / Cramér-Rao in flavor | Measurable info in parameter estimation |

**No primary sources ingested:** Wheeler, Bekenstein, Preskill QIT, Nielsen & Chuang.

### Approach A (Arithmetic)

| Missing topic | Why needed |
|---------------|------------|
| [[explicit-formula-primes-zeros]] | Rigorous prime↔zero bridge (not flavor) |
| Selberg trace formula | Geometric analog of explicit formula |
| [[connes-spectral-triple]] | Modern Hilbert–Polya program — **ingested** |
| [[montgomery-pair-correlation]] | Pair correlation scope — **ingested** |

### Approach B (Spectral)

| Missing topic | Why needed |
|---------------|------------|
| [[clockwork-kernel]] | Partial alternative envelope — in code, thin in wiki |
| [[generalized-envelope-kernel]] | Super-Gaussian p; robustness tests |
| Index theorems / chiral anomaly | SM consistency constraints on info |

### Cross-cutting

| Missing topic | Why needed |
|---------------|------------|
| Three generations | Why 3? — mentioned speculatively nowhere central |
| SM Yukawa as empirical input | Honest baseline: QED doesn't predict Yukawa |
| `knowledge/manifest.yaml` | Sync raw snapshots ↔ repo files (SHA256) |

---

## What the Ingest Did Well

1. **Proven vs conjecture ledger** — actionable single reference
2. **Diagnostics honestly kill** Gaussian full quarks, shared-Q bottleneck, clockwork completeness
3. **Split-fermion pursue path** — correct strategic priority (B → D)
4. **Transfer test** wired everywhere that matters
5. **Adversarial zeta query** prevents numerology relapse

---

## What Still Misleads If Uncorrected

1. **`status: established` on interference-kernel** — overclaims physics
2. **Un ingested boundary report** — will resurrect "universal σ" narrative
3. **Empty `raw/sources/` for external papers** — wiki reads like repo README, not math↔reality research base
4. **No neutrino pipeline page** — overstates completeness of observables ingest
5. **Entity page `alexander-seto`** — zero epistemic value

---

## Recommended Next Ingests (Priority Order)

1. Boundary analysis snapshot + [[similar-fitted-scales-vs-transfer]] disambiguation
2. External: Bekenstein (1–2 pages) + Preskill quantum info lecture notes (entropy section)
3. Concept: [[holographic-principle]], [[explicit-formula-primes-zeros]], [[clockwork-kernel]]
4. ~~Implement or document neutrino observables gap explicitly~~ — done
5. `manifest.yaml` for snapshot integrity
6. ~~Wiki link linter~~ — `scripts/lint_wiki_links.py`

---

## Maintenance Rules (from this audit)

- Never use `status: established` for **fits** — use `phenomenological`
- Any "universal" language must link [[similar-fitted-scales-vs-transfer]] or [[repo-scientific-findings]]
- New bridge edges require [[plausibility-register]] row **before** diagram update
- Ingest external source before expanding arithmetic islands
