---
type: concept
title: Split-Fermion Overlaps
tags: [flavor, qm, spectral]
related:
  - split-fermion-localization
  - hilbert-spaces-qm
  - interference-kernel
  - yukawa-observables-pipeline
  - derive-interference-kernel-from-overlaps
island: true
bridge_tags: [spectral, qm, flavor]
approach: spectral
plausibility: pursue
status: conjecture
created: 2026-06-01
updated: 2026-06-01
---

# Split-Fermion Overlaps

## Established (literature)

For fermions \(\psi_{L,R}(y)\) on an extra dimension \(y\),

\[
Y_{ij} \propto \int dy\, \psi_{L,i}^*(y)\, \psi_{R,j}(y)\, H(y)
\]

(with Higgs profile \(H(y)\) optional). This is the **standard** split-fermion mechanism on [[hilbert-spaces-qm]].

## Proven in parent repo (effective limit only)

- `src/kernel.py` evaluates **discrete** \(Y_{ij}\) from coordinate labels and envelope × \((1 + \varepsilon e^{i\Phi})\) — **not** a literal integral.
- [[yukawa-observables-pipeline]]: SVD of these matrices → CKM + anchored masses — methodology established.

## Conjecture

\[
\exp\left(-\frac{d_{ij}^2}{2\sigma^2}\right)\left[1 + \varepsilon e^{i\Phi_{ij}}\right]
\stackrel{?}{=} \text{leading overlap} + \text{interference correction}
\]

from localized Gaussian wavefunctions. **Partial test:** [[split-fermion-overlap-test]] — r=0.99996 magnitude match, w≈0.69σ for one geometry. See [[derive-interference-kernel-from-overlaps]].

## Dead / refuted shortcuts

- Zeta zero spacing → overlap phases: **dead** ([[why-not-zeta-flavor-numerology]]).
- Universal \(\sigma, k\) across sectors without derivation: **refuted** ([[repo-scientific-findings]]).
