---
type: synthesis
title: Plausibility Register
tags: [meta]
related:
  - multi-sided-bridge-framework
  - information-reality-bridge-map
  - why-not-zeta-flavor-numerology
status: open
created: 2026-06-01
updated: 2026-06-12
strategy: research-strategy
---

# Plausibility Register

> Full ledger: [[proven-vs-conjecture-ledger]]

Living record of **what failed**, what is **implausible**, and what remains **worth pursuing**. Update after every ingest, test, or lint.

**Verdicts:** `pursue` | `watch` | `deprioritize` | `dead` | `established`

## Dead or Near-Dead Bridges

| Bridge | Verdict | Why it fails |
|--------|---------|--------------|
| Zeta zeros **directly** predict 3×3 Yukawa entries | **dead** | Scale mismatch (\(\infty\) vs 3); no mechanism; GUE shared by random matrices too — [[why-not-zeta-flavor-numerology]] |
| GUE spacing test on 3×3 flavor matrices | **dead** | Only 2 spacings — statistically meaningless |
| "Universal" kernel parameters across sectors | **dead** (refuted) | Transfer test loss 797.5 frozen vs 779.0 free — [[repo-scientific-findings]] |
| Gaussian kernel **full** quark sector | **dead** (refuted) | Structural CKM–\(m_c\) rank trade-off — [[diagnostics-summary]] |
| Shared-\(Q\) constraint as quark bottleneck | **dead** (refuted) | Minimality ladder: all relaxations worsen holdout — [[diagnostics-summary]] |
| Legacy quark **geometry extension** (more discrete coords) | **dead** (refuted) | 0/5759 strict (diag 30); 0% on 86k sample (diag 29) — [[quark-geometry-conventions]] |
| Independent \(Q_u, Q_d\) without new ansatz | **dead** (refuted) | Level 4 holdout +218% (diag 09) |
| Tier-2 ansätze (rank2 sum, FN texture, dual-phase, power-law) | **dead** (refuted) | Diag 32: 0% strict N=100; holdout ≫ Gaussian |
| Three-regime framework validated for quarks | **dead** (refuted) | 0% strict survivors; \(m_c \sim 15\times\) experimental — [[manuscript-key-results]] |
| Clockwork kernel complete quark solution | **dead** (refuted) | Light masses, \(V_{ub}\), Jarlskog fail — [[diagnostics-summary]] |
| Berry–Keating \(H=xp\) as **proven** Hilbert–Polya operator | **dead** (as proof) | Not self-adjoint on standard domain; semiclassical heuristic only |
| Primes **directly** determine electron mass ratios | **dead** | No QED calculation supports this; numerology |
| Primes as **mode indices** in standard QED sums | **dead** (refuted) | Diag 38: integer sums converge; prime-only rel_err **0.67–0.93** — [[can-primes-enter-via-qed-spectral-sums]] |
| p-adic QM as **leading** SM flavor explanation | **deprioritize** | Adelic programs lack confirmed predictive wins vs split fermions / RG |
| Chiral projection → [[interference-kernel]] / [[projection-regimes]] | **dead** | Category error; regime info link refuted (12–16); diag 42 portal CP fail — [[chiral-projection-formalization-program]] |
| Simple quark mirror portal (parity-\(\pi\), Schur add-on) | **dead** (quark CP) | Diag 42: median \(J\) err unchanged; Schur overfits holdout — [[adversarial-review-chiral-projection-thesis]] |

## Low Plausibility (Track, Don't Bet On)

| Bridge | Verdict | Notes |
|--------|---------|-------|
| [[spectral-interpretation-of-flavor]] via Hilbert–Polya | **deprioritize** | Requires unknown \(H\) *and* projection story *and* 3-generation embedding |
| [[interference-kernel]] ← [[riemann-zeta-function]] | **deprioritize** | Only via shared RMT statistics — insufficient |
| Metric regime = information loss → anarchy | **watch** | Tested in `diagnostics/12_regime_entropy_correlation.py` — see [[information-measure-for-projection-regimes]] |
| Split-fermion overlaps → derive [[interference-kernel]] | **dead** (mechanism) | Diag 33 N=50: geometry→\(w/\sigma\) R² **0.045**; magnitude fit only — [[derive-interference-kernel-from-overlaps]] |
| [[it-from-bit]] → concrete mass formula | **watch** | Philosophical north star, not constructive |
| [[chiral-projection-thesis]] mirror sector (Tier A) | **watch** | Standard BSM; not novel — lab portals required |
| [[chiral-projection-thesis]] orientation map \(\Pi_\Omega\) (Tier B) | **philosophical** | Needs formal map + in-sector observable — [[adversarial-review-chiral-projection-thesis]] |
| Holographic two-reconstruction → mirror sector | **deprioritize** | Metaphor; AdS/CFT mismatch for observed universe |

## Medium Plausibility (Active Exploration)

| Bridge | Verdict | Next test |
|--------|---------|-----------|
| QED/QM → information measures → **flavor mechanism** | **deprioritize** | Diagnostics 12–19 all fail; kernel = phenomenology — [[fisher-transfer-universality-test]] |
| Hilbert–Polya operator exists (independent of flavor) | **watch** | Ingest Connes, Sturm–Liouville constructions |
| Primes enter via **spectral** or **trace** formulas | **watch** | Explicit formula links primes ↔ zeta zeros ↔ hypothetical spectrum |
| QED/QM → S(ρ_Y) mechanism inequalities | **dead** | `diagnostics/12_*`, `13_*`, `15_*`, `16_*` — all refuted at pre-registered thresholds |
| Post-hoc ρ_Y QFI / coherence mechanism | **dead** | Diagnostics 15–16 refuted |
| Experimental Fisher → sector-split **mechanism** | **dead** | Diag 17 alignment 0.46; diag 19 transfer **refuted** (loss 805.8, align 0.41) |
| Fisher transfer universality (quark → lepton) | **dead** | [[fisher-transfer-universality-test]] — falsifier B |
| Open-system pooled mixing predictor | **dead** | Diag 18 pooled \|r\| ≈ 0.007 |
| p-adics for **hierarchy** (not gauge) | **watch** | Ultrametric tree ↔ mass hierarchy — compare to envelope suppression |

## High Plausibility / Established

| Claim | Verdict | Notes |
|-------|---------|-------|
| QM states live in Hilbert spaces | **established** | [[hilbert-spaces-qm]] |
| QED is empirically validated EFT | **established** | [[qed-qm-information]] |
| Zeta zeros have GUE-like pair statistics (numerical + limited theorems) | **established** (partial) | Not full GUE; Montgomery under RH for restricted test |
| Gaussian × interference **parameterizes** flavor | **established** (phenomenology) | [[repo-scientific-findings]] — fit yes, predict no |
| Kernel → SVD → CKM/mass pipeline | **established** | [[yukawa-observables-pipeline]], 56/56 QA — [[diagnostics-summary]] |
| Clockwork improves \(m_c\) vs CKM vs Gaussian | **established** (partial) | Pareto combined error ~0.77 vs ~1.84 — not full sector |
| Bekenstein entropy bound | **established** | Information ↔ geometry; not yet primes |

## Failure Log (Chronological)

| Date | Item | Outcome |
|------|------|---------|
| 2026-01 | Kernel parameter transfer quark→lepton | **Failed** — document in [[repo-scientific-findings]] |
| 2026-06 | Zeta→flavor direct bridge | **Marked dead** on adversarial review |
| 2026-06 | 3×3 GUE test proposed | **Marked dead** — insufficient degrees of freedom |
| 2026-06 | Diagnostics ingest (Gaussian, clockwork, minimality) | Gaussian full quarks **dead**; shared-Q **not** bottleneck; QA **pass** |
| 2026-06 | Regime vs S(ρ_Y) mechanism | **Refuted** — diagnostic 12, max \|r\|=0.047, n=10,080 |
| 2026-06 | QFI/coherence on ρ_Y vs mixing (pooled) | **Refuted** — diagnostic 15, max \|r\|=0.017, n=10,080 |
| 2026-06 | Decoherence proxy vs CKM/PMNS | **Refuted** — diagnostic 16, max \|r\|=0.241, n=2,592 |
| 2026-06 | Split-fermion→kernel pursue path | **Deprioritized** — user strategy; overlap r≈0.99 kept as historical |
| 2026-06-02m | Tier 3 split-fermion→kernel (diag 33) | **Mechanism refuted** — stable \(w/\sigma\) at fixed \(\sigma\) but not geometry-predictable; Path D watch-only |
| 2026-06-02o | Tier 5.2 explicit-formula FFT test | **Blocked** — tautology risk; [[adversarial-review-tier5-trace-formula]] |
| 2026-06-02p | Tier 5.2 diag 34 (holdout+null) | **Pass** — arithmetic identity; not HP |
| 2026-06-02p | Tier 5.3 Jacobi inverse diag 35 | **Fail** — kernel not 3-site H proxy |
| 2026-06 | Experimental Fisher cross-sector (`diagnostics/17_*`) | Alignment 0.46 < 0.50 — **mechanism weak**; rank–regime \|r\|=0.72 confounded |
| 2026-06 | Open-system decoherence pooled (`diagnostics/18_*`) | Pooled \|r\|=0.007 — **refuted**; quark-local \|r\|=0.26 exploratory |
| 2026-06 | Fisher transfer test (`diagnostics/19_*`) | **Refuted** — frozen loss 805.8, alignment 0.41; free-fit Δθ ≫ CR |
| 2026-06 | Path A QIT→flavor mechanism | **Deprioritized** — no surviving falsifier-passing route (12–19) |
| 2026-06 | Collider Fisher sketch (`diagnostics/20_*`) | **Too thin** — event likelihood out of scope; mixing-only under-identifies kernel |
| 2026-06-12 | Chiral projection quark portal audit (diag 42) | **Refuted** — simple portals do not improve \(J\); Schur overfits holdout — [[chiral-projection-formalization-program]] |

## How to Use

1. Before adding a bridge edge in [[information-reality-bridge-map]], check this register.
2. If a test fails, add a row to Failure Log — do **not** reinterpret failure as "different projection" without a new falsifiable prediction.
3. Prefer **pursue** items that start from QED/QM or established Hilbert space physics.
