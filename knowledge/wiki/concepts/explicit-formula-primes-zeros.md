---
type: concept
title: Explicit Formula (Primes and Zeros)
tags: [primes, zeta]
related:
  - riemann-zeta-function
  - prime-numbers-and-physics
  - hilbert-polya-conjecture
island: true
bridge_tags: [primes, zeta, spectral]
approach: arithmetic
plausibility: watch
status: established
created: 2026-06-01
updated: 2026-06-01
---

# Explicit Formula (Primes and Zeros)

## Established

The **explicit formula** (Riemann–von Mangoldt) links prime distribution to zeta zeros:

\[
\psi(x) = x - \sum_\rho \frac{x^\rho}{\rho} - \log(2\pi) - \cdots
\]

where \(\psi\) is Chebyshev function and \(\rho\) runs over non-trivial zeros.

This is the **rigorous** arithmetic↔spectral bridge — not flavor physics.

## What it proves vs does not

| Proves | Does not prove |
|--------|----------------|
| Primes and zeros are dual descriptions of one analytic object | Zeros are eigenvalues of a known physical Hamiltonian |
| Zero density controls prime counting error | SM Yukawa structure |

## Trace formula analogy

Selberg trace formula: closed geodesics ↔ spectral data (geometric **prime** analog). Compare to [[hilbert-polya-conjecture]] — both seek operator interpretations of explicit formulas.

**Gap:** No Selberg page yet; no ingested source.

## Path to physics (watch only)

If Hilbert–Polya provides \(H\), explicit formula becomes **spectral side of a trace formula**. Still no direct line to [[interference-kernel]] without extra structure.

## SM decoupling (Path D)

This page is **independent of Standard Model flavor physics**. No CKM, Yukawa, or three-regime claims belong here. A hypothetical hook to Path A ([[research-strategy]]) would require:

1. A concrete trace formula whose **spectral side** is an operator expectation (not a 3×3 matrix fit)
2. An **information-theoretic** quantity on that spectral side comparable to \(S(\rho_Y)\) in [[information-measure-for-projection-regimes]]
3. A derivation — not analogy — linking the two

Until then: **watch only**. Demo: `diagnostics/14_explicit_formula_numerics.py`.

## Related

[[can-primes-enter-via-qed-spectral-sums]] — QED sums are not this formula unless engineered.

## Sources to ingest

- Edwards, *Riemann's Zeta Function* (explicit formula chapter)
- Selberg trace formula overview
