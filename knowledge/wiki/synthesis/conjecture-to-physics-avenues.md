---
type: synthesis
title: Conjecture-to-Physics Avenues (Post-Phenomenology)
tags: [meta, spectral, primes, qm, strategy]
related:
  - multi-sided-bridge-framework
  - plausibility-register
  - proven-vs-conjecture-ledger
  - hilbert-polya-conjecture
  - future-work
  - research-strategy
status: active
created: 2026-06-02
updated: 2026-06-02n
---

# Conjecture-to-Physics Avenues (Post-Phenomenology)

**Goal:** Connect **unproven but widely assumed** mathematical structure (RH, Hilbert–Polya, quantum chaos universality, trace formulas) to **independent empirical anchors** (computable primes, zero tables, QFT where hooked) — without reviving refuted flavor numerology. See [[adversarial-review-tier5-trace-formula]] before coding T5.2.

**Context:** SM flavor phenomenology in this repo is **closed** for mechanism (Tiers 2–3). Arithmetic→CKM direct bridges are **dead**. The viable program is **parallel tracks**: honest flavor paper (Tier 0) + **independent** B↔A and C→B programs that could eventually constrain effective readouts.

## Taxonomy

| Class | Examples | Empirical anchor (not collider unless stated) |
|-------|----------|-----------------|
| **Established** | QM on Hilbert spaces; Bekenstein bound; GUE stats of many quantum systems | Laboratory QFT, black-hole thermodynamics |
| **Established conditional** | Montgomery pair correlation of zeta zeros **if RH** | No SM observable; constrains HP candidate operators |
| **Open conjecture, structural** | RH; Hilbert–Polya operator \(H\) | Primes via explicit formula **if** spectrum identified |
| **Open conjecture, speculative SM** | Kernel phase = spectral phase of \(H_{\text{flavor}}\) | **Unproven** — split-fermion derivation failed (diag 33) |
| **Refuted in-repo** | Zeta→Yukawa; 3×3 GUE; QIT→mixing mechanism; universal kernel params | Do not reopen without new falsifier |

**Rule:** “Assumed true” is not evidence. Each avenue needs a **direction** (math→physics or physics→math), a **scale**, and a **falsifier** ([[multi-sided-bridge-framework]]).

## Ranked avenues (what to pursue)

### 1. Trace formula / explicit formula lane (B ↔ A) — **highest priority**

**Conjecture stack:** RH (zeros on line) + Hilbert–Polya (\(\zeta\) zeros = spectrum of self-adjoint \(H\)).

**Empirical anchor today:** Prime distribution is **computed** from definitions and compared to analytic formulas; explicit formula

\[
\psi(x) = x - \sum_\rho x^\rho/\rho - \cdots
\]

links primes to **zero heights** (imaginary parts \(E_n\)). This is not SM flavor — it is **number-theoretic physics** in the sense of semiclassical limits and spectral oscillations.

| Step | Repo action | Falsifier |
|------|-------------|-----------|
| Ingest | [[trace-formula-bridge-ladder]] — explicit formula, Selberg, Montgomery, Connes | **Done** (2026-06-02n) |
| Numeric | diag 34 **only if** non-circular — see [[adversarial-review-tier5-trace-formula]] | Holdout \(\psi(x)\) beats wrong-frequency null; else **no-go** |
| Theory | Document **Selberg ↔ Riemann** analogy as template for “spectrum ↔ arithmetic” | Claiming Selberg proves flavor |

**Why first:** Direction is clear (A↔B). No 3×3 scale mismatch. Aligns with Path D **watch** without flavor numerology.

### 2. Quantum chaos ↔ RMT (B, large-N) — **watch, not flavor**

**Assumed:** Generic chaotic spectra have GUE level statistics.

**Repo lesson:** Shared GUE with 3×3 Yukawas is **dead** (diag 33, [[why-not-zeta-flavor-numerology]]). **Different question:** Does the **kernel optimization landscape** (joint loss over geometries) show level repulsion / universality at **large sample size**?

| Step | Action | Falsifier |
|------|--------|-----------|
| Diagnostic idea | `34_chaos_landscape_rmt.py`: eigenvalues of Hessian or loss level spacings across N≫100 geometries | Spacings indistinguishable from Poisson / random |
| Scope | Phenomenology of **optimization**, not fundamental law | Marketing as proof of RH |

### 3. Jacobi / Sturm–Liouville inverse problem (B → D) — **medium, falsifiable**

**Question ([[does-phase-structure-imply-spectral-operator]]):** Is bilinear \(\Phi_{ij}=\alpha+k(x_i+x_j)/2+\eta(x_i-x_j)\) the phase of a **1D** self-adjoint operator (split fermion Laplacian + boundary data)?

**Status:** Split-fermion **overlap magnitudes** fit post-hoc (diag 33); **phases already match** when \(k,\eta,\alpha\) are shared — but geometry does not predict parameters.

| Step | Action | Falsifier |
|------|--------|-----------|
| Construct | Tridiagonal \(H\) on 3-node discretization; match \(Y_{ij}\) magnitudes + phases | Best-fit residual ≫ optimized kernel |
| Predict | Sector splits \((k,\eta)_u \neq (k,\eta)_d\) from **same** \(H\) with different BCs | BC story needs >6 free params per sector |

### 4. Primes in QED spectral sums (A ↔ C) — **low until hook**

**Query:** [[can-primes-enter-via-qed-spectral-sums]] — standard QED sums use integer mode indices, not primes.

**Upgrade path:** Exhibit one **standard** observable whose renormalization or vacuum sum equals a **known** prime-sum **without** redefining the spectrum.

**Until then:** **watch** only.

### 5. p-adic / adelic hierarchy (A → D) — **deprioritize**

Ultrametric trees mimic hierarchy; SM gauge structure is not p-adic. Compete against **envelope suppression** in [[interference-kernel]] with the same holdout protocol as diag 09 — unlikely to beat without new physics.

### 6. It-from-bit ontology (C → all) — **philosophical north star**

Useful for motivation; not a near-term proof path unless Bridges 1–3 produce **constructive** dynamics ([[what-proves-information-creates-reality]]).

## What *not* to do

- Compare zeta zero spacings to 3×3 Yukawa spacings (diag 33: not testable / dead).
- Claim HP “explains” CKM without an operator→Yukawa projection story.
- Re-run QIT→flavor mechanism diagnostics 12–19 without new observables.
- Treat “RH is true” as evidence for any **specific** SM parameter relation.

## Proposed Tier 5 (code + wiki)

| ID | Deliverable | Track |
|----|-------------|-------|
| **T5.1** | Wiki ingest: trace formula, Montgomery, Connes (sources + ledger) | **Done** — [[trace-formula-bridge-ladder]] |
| **T5.2** | `diagnostics/34_explicit_formula_spectral_audit.py` — prime oscillations vs zero heights | B↔A |
| **T5.3** | `diagnostics/35_jacobi_inverse_kernel_phase.py` — 3-site \(H\) vs kernel phases | B→D |
| **T5.4** | Optional: optimization-landscape RMT (large-N geometry sample) | B meta |

Pre-register falsifiers in [[survivor-protocol-preregistered]] before large runs.

## Decision criteria for “connection found”

A result counts as **physics-relevant bridge progress** only if:

1. **Prediction precedes fit** — or holdout-style test on independent data (zeros, primes, or geometries).
2. **Scale declared** — flavor 3×3 vs semiclassical \(N\to\infty\).
3. **Register updated** — [[plausibility-register]] verdict changed with failure log entry.

## See also

- [[multi-sided-bridge-framework]] — gap-crossing protocol
- [[future-work]] — Tier 0 publication runs parallel to Tier 5
- [[hilbert-polya-conjecture]] — HP independent of flavor
- [[research-strategy]] — active priorities after 2026-06-02n
