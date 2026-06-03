---
type: synthesis
title: Proven vs Conjecture Ledger
tags: [meta, flavor, information]
related:
  - plausibility-register
  - multi-sided-bridge-framework
  - repo-scientific-findings
  - diagnostics-summary
  - manuscript-key-results
status: open
created: 2026-06-01
updated: 2026-06-02j
---

# Proven vs Conjecture Ledger

Single reference for **know** / **probably true** / **dead**. Status labels match `schema.md`.

## Established (tested or standard physics)

| Claim | Pointer |
|-------|---------|
| QM on Hilbert spaces | [[hilbert-spaces-qm]] |
| QED as validated EFT | [[qed-qm-information]] |
| Kernel formula implemented correctly | [[kernel-implementation]], 56/56 QA |
| SVD → CKM + anchored quark masses | [[yukawa-observables-pipeline]], [[observables-extraction]] |
| Zeta zeros show GUE-like pair statistics (partial theorems + numerics) | [[riemann-zeta-function]], [[random-matrix-theory]] |
| Split-fermion overlap mechanism (literature) | [[split-fermion-localization]] |
| Bekenstein bound (info ↔ geometry) | [[quantum-information]] |

## Phenomenological (fits data; not predictive theory)

| Claim | Pointer |
|-------|---------|
| Gaussian × interference **parameterizes** flavor per sector | [[repo-scientific-findings]], [[interference-kernel]] |
| Three regime **labels** organize sector-specific parameter sets | [[projection-regimes]], [[manuscript-key-results]] |
| Charged lepton / neutrino subset fits (survivor rates) | [[manuscript-key-results]], `ANALYSIS_SUMMARY.md` |
| Clockwork envelope improves \(m_c\) vs CKM trade-off vs Gaussian | [[diagnostics-summary]] |
| Pareto envelope p scan | [[pareto-envelope-comparison]] — p=3 knee only; mc still wrong |
| Pareto CKM–\(m_c\) frontier for quarks | [[manuscript-key-results]], `diagnostics/21_quark_phenomenology_holdout.py` |
| Train/holdout quark protocol (mc,Vus,Vcb train; mu,md,ms,Vub holdout) | `src/observables.py`, diag 21 |

## Conjecture (plausible; needs derivation or test)

| Claim | Pointer |
|-------|---------|
| Interference kernel from 1D overlap integrals | [[derive-interference-kernel-from-overlaps]] — **refuted (mechanism):** diag 33 geometry→\(w/\sigma\) R²≈0.05; magnitude fit only |
| Phase structure from self-adjoint \(H\) | [[does-phase-structure-imply-spectral-operator]] |
| Primes via spectral/trace routes (not direct flavor) | [[can-primes-enter-via-qed-spectral-sums]] |
| Hilbert–Polya operator exists | [[hilbert-polya-conjecture]] |

## Speculative / philosophical

| Claim | Pointer |
|-------|---------|
| It-from-bit as constructive dynamics | [[it-from-bit]] |
| Metric regime = maximal flavor decoherence | [[projection-regimes]] — not supported by ρ_Y measures |
| Information creates reality (umbrella) | [[information-creates-reality]] |
| p-adic hierarchy for SM flavor | [[can-p-adics-encode-flavor-hierarchy]] |

## Refuted / dead

| Claim | Pointer |
|-------|---------|
| Universal kernel **parameters** across sectors | [[repo-scientific-findings]] — 797.5 vs 779.0 loss |
| Zeta zeros → 3×3 Yukawa directly | [[why-not-zeta-flavor-numerology]] |
| 3×3 GUE spacing test on flavor | [[zeta-zero-spacing-yukawa-structure]] |
| Gaussian kernel **full** quark sector | [[diagnostics-summary]] — structural rank trade-off |
| Shared-\(Q\) as primary quark bottleneck | [[diagnostics-summary]] — minimality ladder |
| Three-regime framework **validated** for quarks | [[manuscript-key-results]] — 0% strict survivors, \(m_c\) failure |
| Berry–Keating \(H=xp\) as proven Hilbert–Polya | [[berry-keating-hamiltonian]] |
| Direct primes → mass ratios | [[prime-numbers-and-physics]] |
| Regime / \(S(\rho_Y)\) / QFI-on-\(\rho_Y\) as flavor mechanism | [[information-measure-for-projection-regimes]], `diagnostics/12–16` |
| Experimental Fisher cross-sector alignment as mechanism | Diag 17: alignment 0.46 < 0.50 |
| Fisher transfer quark → lepton (CR + alignment falsifiers) | [[fisher-transfer-universality-test]] — diag 19: loss 805.8, align 0.41 |
| External open-system \(p\) predicts CKM/PMNS mixing (pooled) | Diag 18: pooled \|r\| ≈ 0.007 |
| Path A QIT→flavor operational mechanism (pooled) | Diagnostics 12–19 — no surviving route |
| Legacy quark geometry extension (coords 5–8) | Diag 29–30 — 0/5759 strict on unified protocol |
| Independent \(Q_u, Q_d\) as quark fix | Diag 09 — holdout degrades at all minimality levels |
| Tier-2 texture/rank2 kernels as quark fix | Diag 32 — 0% strict; holdout worse than Gaussian |

## Probably true (working hypotheses — not proven here)

| Hypothesis | Why "probably" | Falsifier |
|------------|----------------|-----------|
| Flavor textures from extra-dim overlaps + phases | Standard EFT story; repo uses effective version | Derive kernel params from one potential |
| Arithmetic enters only downstream of spectrum | No direct zeta→CKM mechanism found | Construct explicit prime→operator map with SM prediction |
| Sector parameter splits reflect different wavefunction sampling | Consistent with transfer failure + regime labels | Predict \(k_e/k_q\) from geometry without fitting |
| Similar **independently fitted** σ, α across sectors | Boundary report clustering; distinct from transfer | Confuse with universality without [[similar-fitted-scales-vs-transfer]] |

## Maintenance

Update this ledger when ingesting new tests. Mirror verdicts in [[plausibility-register]]. See [[knowledge-gaps-audit]] for missing coverage.
