---
type: concept
title: Selberg Trace Formula
tags: [spectral, primes, chaos]
related:
  - explicit-formula-primes-zeros
  - hilbert-polya-conjecture
  - riemann-zeta-function
  - random-matrix-theory
  - connes-spectral-triple
island: true
bridge_tags: [spectral, primes]
approach: spectral
plausibility: watch
status: established
created: 2026-06-02
updated: 2026-06-02n
---

# Selberg Trace Formula

## One-line role

**Proved** template: closed geodesics (geometric primes) ↔ Laplacian eigenvalues on a hyperbolic surface. The Riemann explicit formula is the **arithmetic** analogue; Hilbert–Polya asks for an operator making that analogy **spectral**.

## Schematic identity

See curated ingest [[selberg-trace-formula]] (raw) for the standard geodesic ↔ spectral sum.

## Bridge ladder (B ↔ A)

| Rung | Object | Status in wiki |
|------|--------|----------------|
| 1 | Selberg: geodesics ↔ Laplacian on \(\Gamma\backslash\mathbb{H}\) | **Established** |
| 2 | Riemann: primes ↔ \(\zeta\) zeros via \(\psi(x)\) | **Established** ([[explicit-formula-primes-zeros]]) |
| 3 | Montgomery: zero spacings ↔ GUE pair correlation | **Conditional on RH** ([[montgomery-pair-correlation]]) |
| 4 | Hilbert–Polya: zeros = eigenvalues of one \(H\) | **Open** |
| 5 | Connes: NC spectral triple for zeros | **Framework** ([[connes-spectral-triple]]) |

## What Selberg does *not* do

- Identify \(\zeta\) zeros with a known Standard Model Hamiltonian.
- Justify 3×3 Yukawa GUE tests ([[zeta-zero-spacing-yukawa-structure]] is **dead**).

## Repo connection

- Educational parallel only — no Selberg numerics script yet.
- Tier 5 **T5.2** extends [[explicit-formula-primes-zeros]] numerics (diag 14), not Selberg geodesics.

## Related

[[multi-sided-bridge-framework]], [[conjecture-to-physics-avenues]]
