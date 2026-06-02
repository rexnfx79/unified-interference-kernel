---
type: source
title: Interference Kernel Manuscript
tags: [flavor]
related:
  - interference-kernel
  - projection-regimes
  - interference-kernel-manuscript
sources:
  - raw/sources/manuscript-summary.md
authors: [Alexander Seto]
year: 2026
status: established
created: 2026-06-01
updated: 2026-06-01
---

# Interference Kernel Manuscript

Summary of `manuscript.tex` — phenomenological flavor organization paper.

## Model

\[
Y_{ij} = \exp(-d_{ij}^2/2\sigma^2) \cdot [1 + \varepsilon \exp(i\Phi_{ij})]
\]

Discrete fermion coordinates in internal flavor space (split-fermion style).

## Three Projection Regimes

1. **Quarks** — envelope-dominated; CKM–\(m_c\) Pareto trade-off
2. **Charged leptons** — phase-sensitive; \(k_e, \eta_e\) shifts
3. **Neutrinos** — metric-dominated; \(g_{\text{env}} \approx 0.60\) compression → anarchy-like PMNS

## Results (Claimed in Manuscript)

- Quark sector: structural CKM–\(m_c\) Pareto frontier
- Leptons: ~60% survivor rate, Z-scores < 1
- Neutrinos: ~45% survivor rate for PMNS

## Caveat

Manuscript uses "universal" for **functional form reuse**, not parameter transfer. See [[repo-scientific-findings]] for transfer test contradicting parameter universality.

## Code

`src/kernel.py`, optimization scripts in `scripts/`.
