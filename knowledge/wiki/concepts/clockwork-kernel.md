---
type: concept
title: Clockwork Alternative Kernel
tags: [flavor]
related:
  - interference-kernel
  - diagnostics-summary
  - derive-interference-kernel-from-overlaps
island: true
bridge_tags: [flavor]
approach: effective
plausibility: watch
status: phenomenological
created: 2026-06-01
updated: 2026-06-01
---

# Clockwork Alternative Kernel

## What it is

Alternative envelope in `src/alternative_kernels.py`: exponential clockwork decay \(q^{-|d|}\) (vs Gaussian \(\exp(-d^2/2\sigma^2)\)).

## Proven (diagnostics)

| Result | Status |
|--------|--------|
| Improves \(m_c\) + \(|V_{us}|\) vs Gaussian (combined error ~0.77 vs ~1.84) | **phenomenological** partial win |
| Implementation verified in QA | **established** |
| Full quark sector at PDG | **refuted** — light quarks ~\(10^3\times\) off, Jarlskog wrong |
| Complete replacement for Gaussian | **dead** |

## Conjecture

Clockwork chain in extra dimension **derives** \(q^{-|d|}\) envelope — natural fit for [[derive-interference-kernel-from-overlaps]] step 1 (potential choice).

## Not a path to arithmetic

No connection to zeta/primes established — do not add to arithmetic bridge map.

## Related

[[generalized-envelope-kernel]] (super-Gaussian p), [[plausibility-register]]
