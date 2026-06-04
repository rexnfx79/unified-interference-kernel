---
type: synthesis
title: Future Work (Post-Phenomenology Tranche)
tags: [meta, strategy, flavor, manuscript]
related:
  - research-strategy
  - manuscript-ledger-alignment
  - survivor-protocol-preregistered
  - proven-vs-conjecture-ledger
  - quark-geometry-conventions
  - diagnostics-summary
status: active
created: 2026-06-02
updated: 2026-06-02r
---

# Future Work (Post-Phenomenology Tranche)

Forward-looking plan after diagnostics **21–31**, geometry closure (**29–30**), and observables phase-fix (**2026-06-02j**).  
**Closed paths:** [[research-strategy]] dead-ends table — do not reopen without a new falsifier.

## Tier 0 — Close the phenomenology paper (**complete** 2026-06-02o)

| Item | Deliverable | Status |
|------|-------------|--------|
| **P0.1** Manuscript final pass | Abstract, Results, Conclusions ↔ [[manuscript-ledger-alignment]] | **Done** — abstract leads 27.8% joint ν; diag 32–33 in Further Study |
| **P0.2** Quark structural negative | CKM–\(m_c\) Pareto + 0/5759 strict (diag 30) | **Done** |
| **P0.3** Neutrino dual headline | 27.8% joint (diag 28) primary; 78.9% PMNS-only (diag 23) sub | **Done** |
| **P0.4** Lepton honesty | 1% strict (diag 22); legacy ~60% labeled | **Done** |
| **P0.5** Artifact bundle | `scripts/reproduce_phenomenology_tranche.sh` + frozen reports | **Done** |

**Non-goals:** Universal kernel parameters; three-regime “validation”; geometry extension as quark fix.

## Tier 1 — Observables & CP (**complete** 2026-06-02o)

| Item | Deliverable | Status |
|------|-------------|--------|
| **P1.1** Post-fix audit | `diagnostics/36_tier1_phase_fix_audit.py` — N=15, seed 21021 | **Done** — 0/15 strict (repaired & legacy); refutation stable |
| **P1.2** CP observables | `delta_CKM`, `J`, `J_abs` in `observables.py`; `tests/test_cp_observables.py` | **Done** |
| **P1.3** PMNS CP | `delta_PMNS`, `J_PMNS` in `compute_neutrino_observables` | **Done** |

Report: `diagnostics/results/36_tier1_phase_fix_audit.txt`. CKM magnitudes can shift under legacy phases (median rel diff ~0.6); **masses unchanged**; Pareto corr ~0.52 on N=15 sample. Do **not** scale full CP optimization sweeps without new falsifiers.

## Tier 2 — New quark ansatz (**complete — falsified**)

**Diagnostic:** `diagnostics/32_quark_tier2_ansatz.py` — N=100, seed **32032**. Report: `diagnostics/results/32_quark_tier2_ansatz.txt`.

| Kernel | Strict | Holdout median | vs Gaussian holdout |
|--------|--------|----------------|---------------------|
| gaussian (baseline) | 0% | 25.5 | — |
| rank2_clockwork_sum | 0% | 53498 | much worse |
| clockwork_dual_phase | 0% | 58087 | much worse |
| fn_texture | 0% | 132 | worse |
| fn_texture_split | 0% | 139 | worse |
| power_law | 0% | 48538 | much worse |

**Verdict:** P2.1 and P2.2 **falsified** at strict protocol; no holdout improvement >20%. Mean effective rank Yu/Yd ≈ 1.04–1.63 (rank-2 sum did not raise rank materially). **Do not** scale Tier-2 quark ansätze further without a new falsifier.

Pre-register in [[survivor-protocol-preregistered]] extension **before** large runs:

| Item | Hypothesis | Falsifier |
|------|------------|-----------|
| **P2.1** Rank-2+ overlap kernel | Effective Yukawa needs >1 singular direction per sector | 0% strict at N≥100 **and** holdout not improved vs diag 09 Level 0 |
| **P2.2** Flavor-symmetry priors | FN / texture zeros shrink search | Strict rate ≤ diag 21 with same N |
| **P2.3** Scheme/RGE anchor | \(m_c\) failure partly scale definition | One documented scheme where strict rate >5% at fixed geometry count |

**Holdout rule:** Same as diag 09 — accept extra params only if holdout improves >20% vs baseline.

**Explicitly closed without new ansatz:** split \(Q_u,Q_d\) (diag 09), legacy geometry grids (diag 30), envelope-only sweeps (diag 21).

## Tier 3 — Theory bridges (**complete — no mechanism hook**)

**Diagnostic:** `diagnostics/33_tier3_theory_bridges.py` — N=50, seed **33033**. Report: `diagnostics/results/33_tier3_theory_bridges.txt`. Shared overlap code: `src/split_fermion_overlap.py`.

| Track | Pre-registered falsifier | Result (diag 33) |
|-------|--------------------------|------------------|
| **Split-fermion → kernel** | Stable \(w/\sigma\) **and** geometry predicts \(w/\sigma\) (R²≥0.5) **and** all \|mag\| r≥0.99 | \(w/\sigma\) rel spread **0.067** at fixed params (stable); geometry→\(w/\sigma\) R² **0.045** (fail); min r **0.985** (fail one fit) → **post-hoc envelope only** |
| **Path D** (zeta / primes) | Operational Yukawa/CKM hook | 3×3 GUE spacing **not testable**; Yukawa vs random spacing indistinguishable; primes **educational only** (diag 14) — [[why-not-zeta-flavor-numerology]] |
| **Path A** (QIT → flavor) | — | **Closed** (diag 12–19); not re-run |

**Verdict:** Do not fund split-fermion derivation or Path D flavor numerology. Remaining bridge work = **Tier 0 publication** (honest negatives + partial neutrino/lepton).

See [[multi-sided-bridge-framework]] for bridge order: QED/info → spectral → arithmetic last.

## Tier 5 — Conjecture ↔ physics (**complete** 2026-06-02r)

See [[conjecture-to-physics-avenues]] for ranked tracks and falsifiers.

| Item | Hypothesis | Falsifier |
|------|------------|-----------|
| **T5.1** | Trace-formula bridge ingested (Selberg, explicit formula, Montgomery, Connes) | **Done** — [[trace-formula-bridge-ladder]] |
| **T5.2** | Explicit formula non-circular audit | **Done** — diag 34 PASS (specificity + truncation); arithmetic only |
| **T5.3** | Kernel bilinear phase = 1D self-adjoint operator phase | **Done — fail** — diag 35: rel residual gen **0.55**, tri **0.71** (bar 0.12) |
| **T5.4** | Large-N geometry loss landscape shows RMT universality | **Done — fail** — diag 37: Hessian frac(s<0.1)=**0.42** (Poisson ~0.10, GOE ~0); not GUE |
| **T5.5** | Primes appear in standard QED sums | **Done — fail** — diag 38: integer sums converge; prime-only rel_err **0.67–0.93**; Euler product ≠ mode index |

**Non-goals:** RH→CKM numerology; 3×3 GUE; revived QIT→flavor mechanism.

## Tier 4 — Repo hygiene (**in progress**)

| Item | Status |
|------|--------|
| Wiki link lint | **Done** — `scripts/lint_wiki_links.py` |
| [[diagnostics-summary]] sync (diag 34–38) | **Done** |
| `reproduce_phenomenology_tranche.sh` (34–38) | **Done** |
| Legacy CSV protocol | **Done** — `data/README.md` |
| Manuscript Tier 5 + CP closure text | **Done** — `manuscript.tex` §Further Study |
| Rebuild `manuscript.pdf` | Run `pdflatex manuscript.tex` after pull |

## Decision log

| Date | Decision |
|------|----------|
| 2026-06-02r | Tier 5.5 — diag 38 FAIL (no prime-index QED sum; watch only) |
| 2026-06-02q | Tier 5.4 — diag 37 FAIL (landscape Hessian spacings Poisson-like, not GUE) |
| 2026-06-02p | Tier 5.2–5.3 — diag 34 PASS (explicit formula); diag 35 FAIL (Jacobi inverse) |
| 2026-06-02o | Tier 1 complete — phase audit diag 36 + CP observables |
| 2026-06-02o | Tier 0 publication package complete — manuscript + reproduce script |
| 2026-06-02o | Tier 5 adversarial review — T5.2 blocked pending redesign |
| 2026-06-02n | Tier 5.1 wiki ingest complete — [[trace-formula-bridge-ladder]] |
| 2026-06-02n | Tier 5 conjecture↔physics program — [[conjecture-to-physics-avenues]] |
| 2026-06-02m | Tier 3 theory bridges **closed** (diag 33); split-fermion→kernel remains post-hoc |
| 2026-06-02k | Phenomenology tranche **complete**; forward work = publication + optional P1/P2 |
| 2026-06-02j | Geometry extension **closed** (0/5759 strict) |
| 2026-06-02h | Survivor protocol **canonical** (Tier A1) |

## See also

- [[research-strategy]] — active priorities & do-nots
- [[manuscript-honest-limitations-draft]] — limitations prose source
- `manuscript.tex` §Further Study (synced 2026-06-02k)
