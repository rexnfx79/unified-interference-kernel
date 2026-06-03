---
type: source
title: Montgomery Pair Correlation
tags: [zeta, chaos, spectral]
related:
  - random-matrix-theory
  - riemann-zeta-function
  - hilbert-polya-conjecture
  - connes-spectral-triple
  - explicit-formula-primes-zeros
  - trace-formula-bridge-ladder
sources:
  - raw/sources/montgomery-pair-correlation.md
authors: [Hugh Montgomery]
year: 1973
status: established
created: 2026-06-01
updated: 2026-06-02n
---

# Montgomery Pair Correlation

**Tier 5.1 ingest** — pair correlation of zeta zero spacings; scope and guardrails for the wiki.

## Theorem (conditional on RH)

Montgomery (1973): for normalized consecutive spacings

\[
\delta_n = (\gamma_{n+1} - \gamma_n)\,\frac{\log \gamma_n}{2\pi},
\]

the **pair correlation function** of \(\delta_n\) at separated indices agrees with the GUE prediction in a restricted range — **assuming the Riemann Hypothesis**.

## What is established vs not

| Claim | Status |
|-------|--------|
| Pair correlation = GUE pair function (range, **if RH**) | **Established** (Montgomery 1973) |
| Full n-point GUE for zeta zeros | **Not proved** |
| Unconditional pair correlation | **Open** |
| Zeros = eigenvalues of known \(H\) | **Open** ([[hilbert-polya-conjecture]]) |
| GUE → 3×3 Yukawa / CKM | **Dead** — [[zeta-zero-spacing-yukawa-structure]], diag 33 |

## Odlyzko numerics (phenomenology)

High-zero computations show **level repulsion** and spacing distributions consistent with GUE. This supports the operator/RMT picture but is **not** a proof of full universality.

## Connection to explicit formula

- [[explicit-formula-primes-zeros]] — controls **prime oscillations** (frequencies \(\gamma_n\))
- Montgomery — controls **correlations between** those frequencies’ spacings
- Together: rigorous arithmetic + statistical spectral analogy; **still no SM hook**

## Wiki guardrails

- Write “GUE-like **pair** correlation (conditional on RH)” — not “zeros are GUE eigenvalues.”
- Do not extrapolate to flavor matrices ([[plausibility-register]]).
- RH assumed in theorem; RH itself remains **open**.

## Full curated text

`knowledge/raw/sources/montgomery-pair-correlation.md`

## Related programs

[[random-matrix-theory]], [[berry-keating-hamiltonian]], [[connes-spectral-triple]], [[trace-formula-bridge-ladder]]
