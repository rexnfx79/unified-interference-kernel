---
type: concept
title: Explicit Formula (Primes and Zeros)
tags: [primes, zeta]
related:
  - riemann-zeta-function
  - prime-numbers-and-physics
  - hilbert-polya-conjecture
  - selberg-trace-formula
  - trace-formula-bridge-ladder
  - connes-spectral-triple
  - montgomery-pair-correlation
island: true
bridge_tags: [primes, zeta, spectral]
approach: arithmetic
plausibility: watch
status: established
created: 2026-06-01
updated: 2026-06-02n
---

# Explicit Formula (Primes and Zeros)

## Established

The **explicit formula** (Riemann, von Mangoldt, modern treatments in Edwards/Davenport) links prime distribution to zeta zeros:

\[
\psi(x) = x - \sum_\rho \frac{x^\rho}{\rho} - \log(2\pi) - \tfrac{1}{2}\log(1 - x^{-2}) + \cdots
\]

\(\psi(x) = \sum_{p^k \le x} \log p\) is the Chebyshev function. Each zero \(\rho = \beta + i\gamma\) contributes an oscillatory term \(x^\rho/\rho\); under RH, \(\beta = 1/2\) and frequencies are \(\gamma\).

**Curated ingest:** `knowledge/raw/sources/explicit-formula-primes-zeros.md`

## What it proves vs does not

| Proves | Does not prove |
|--------|----------------|
| Primes and zeros are dual descriptions of one analytic object | Zeros are eigenvalues of a known physical Hamiltonian |
| Zero heights control prime counting error | SM Yukawa / CKM structure |
| Weil adelic explicit formula as a trace template | [[hilbert-polya-conjecture]] |

## Trace formula ladder

| Template | Geometric / arithmetic side | Spectral side |
|----------|----------------------------|---------------|
| [[selberg-trace-formula]] | Closed geodesics | Laplacian on \(\Gamma\backslash\mathbb{H}\) |
| This page | Prime powers | Zeta zeros |
| [[connes-spectral-triple]] | Adele class space | Dirac spectrum (program) |

See [[trace-formula-bridge-ladder]] for full B↔A map.

## Repo numerics (Path D)

`diagnostics/14_explicit_formula_numerics.py` compares \(\psi(x)\), \(x\), \(\mathrm{Li}(x)\), and a truncated sum over the first 10 known zero heights. **Educational only** — `no_flavor_connection: true` in `diagnostics/results/14_explicit_formula_numerics.txt`.

**Tier 5.2 (planned):** frequency-stability audit — pre-register before scaling zero count.

## SM decoupling

No CKM, Yukawa, or three-regime claims. A hypothetical merge with Path A ([[research-strategy]]) requires a **derived** spectral measure on an operator, not matrix numerology.

## Related

[[can-primes-enter-via-qed-spectral-sums]], [[why-not-zeta-flavor-numerology]], [[conjecture-to-physics-avenues]]
