---
type: query
title: Can Primes Enter via QED Spectral Sums?
tags: [primes, qed, qm]
related:
  - prime-numbers-and-physics
  - qed-qm-information
  - primes-via-quantum-effects
status: closed
created: 2026-06-01
updated: 2026-06-02
---

# Can Primes Enter via QED Spectral Sums?

## Question

Do standard QED calculations (mode sums, vacuum polarization, Casimir) **naturally** involve prime numbers — or is prime insertion always artificial?

## Analysis Sketch

QED mode sums: \(\sum_n f(E_n)\) over photon/electron modes — indices \(n\) are integers, not **primes specifically**.

Prime appearance requires:
- Spectrum \(E_n\) tied to zero heights (Hilbert–Polya), or
- Arithmetic regularization schemes, or
- Non-standard boundary conditions

**Default QED:** no primes — verdict **watch** only via indirect Hilbert–Polya.

## Falsifier

Show a **standard** QED observable (e.g. \(g-2\) correction term) equals a known prime-sum without redefinition — would upgrade to **pursue**.

## Result (diag 38, 2026-06-02)

**Closed fail.** Integer-index partial sums converge to textbook targets (Schwinger \(\sum 1/(k(k+1))\), \(\zeta(2),\zeta(3),\zeta(4)\)). Replacing \(n\) with primes only in the same summand gives rel_err **0.67–0.93** at \(N=2\times10^5\). Euler product \(\prod_p(1-p^{-s})^{-1}=\sum_n n^{-s}\) matches \(\zeta(s)\) but is **not** a prime-index mode sum in standard QED texts.

**Verdict:** remain **watch** (HP / arithmetic routes only); do not claim primes in SM loop sums.

## Related

[[primes-via-quantum-effects]], [[plausibility-register]]
