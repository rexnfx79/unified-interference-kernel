---
type: concept
title: Interference Kernel
tags: [flavor]
related:
  - projection-regimes
  - spectral-interpretation-of-flavor
  - repo-scientific-findings
  - interference-kernel-manuscript
  - yukawa-observables-pipeline
  - diagnostics-summary
  - split-fermion-overlaps
  - proven-vs-conjecture-ledger
island: true
bridge_tags: [flavor, information, spectral]
status: phenomenological
created: 2026-06-01
updated: 2026-06-01
approach: effective
plausibility: watch
---

# Interference Kernel

## Proven

- Formula implemented in `src/kernel.py`; generalized \(p\)-envelope in `kernel_generalized.py` (16/16 tests).
- Same functional form can be fitted independently in quark, lepton, neutrino sectors ([[repo-scientific-findings]], [[diagnostics-summary]]).
- Readout to CKM/masses via [[yukawa-observables-pipeline]] is deterministic and QA-verified.

## Conjecture

- Kernel elements equal split-fermion overlap × interference ([[split-fermion-overlaps]], [[derive-interference-kernel-from-overlaps]]).
- Three [[projection-regimes]] reflect distinct information projection — needs [[information-measure-for-projection-regimes]].

## Refuted

- **Parameter universality** across sectors (transfer test loss 797.5 frozen vs 779.0 free).
- **Full quark sector** with Gaussian envelope (structural CKM–\(m_c\) trade-off).

## Definition

Yukawa couplings organized by envelope suppression and phase interference:

\[
Y_{ij} = \exp\left(-\frac{d_{ij}^2}{2\sigma^2}\right) \times \left[1 + \varepsilon \exp(i\Phi_{ij})\right]
\]

with \(d_{ij} = |x_i - x_j|\) in internal flavor coordinates and

\[
\Phi_{ij} = \alpha + k\frac{x_i + x_j}{2} + \eta(x_i - x_j).
\]

## Implementation

Parent repo: `src/kernel.py`, `src/kernel_generalized.py`

## Honest Status ([[repo-scientific-findings]])

| Claim | Supported? |
|-------|------------|
| Same functional form fits quarks, leptons, neutrinos | Yes |
| Parameters transfer across sectors | **No** (transfer test loss 797.5 frozen vs 779.0 free) |
| "Universal" in UV sense | **Not demonstrated** |

Reframe: **effective readout parameterization**, not proven fundamental kernel.

## Speculative Upgrade Path

If [[does-phase-structure-imply-spectral-operator]] succeeds:

\[
\Phi_{ij} \stackrel{?}{=} \arg\langle \psi_i | e^{-iH} | \psi_j \rangle
\]

for fermion wavefunctions localized in extra dimension and self-adjoint \(H\).

## Three Regimes

See [[projection-regimes]] — envelope-dominated (quarks), phase-sensitive (leptons), metric-dominated (neutrinos).
