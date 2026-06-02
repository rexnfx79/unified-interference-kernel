# Montgomery Pair Correlation (Curated Summary)

> **Primary source:** Montgomery, "The pair correlation of zeros of the zeta function" (1973); Montgomery–Odlyzko numerical work.  
> **Wiki concepts:** [[random-matrix-theory]], [[riemann-zeta-function]], [[hilbert-polya-conjecture]]

## Theorem scope (established — conditional)

Montgomery proved that, **assuming the Riemann Hypothesis**, the **pair correlation function** of normalized zero spacings matches the GUE prediction for distances in a restricted range.

Key limitation: the result is **pair correlation only**, not full n-point GUE statistics, and it is **conditional on RH**.

## What is NOT proven

| Claim | Status |
|-------|--------|
| Full GUE statistics for all n-point correlations | **Not proven** |
| Unconditional pair correlation | **Open** |
| Zeta zeros = eigenvalues of a known operator | **Open** ([[hilbert-polya-conjecture]]) |
| GUE → flavor / Yukawa matrices | **Dead** in this wiki ([[zeta-zero-spacing-yukawa-structure]]) |

## Normalized spacing

For consecutive zeros \(\gamma_n\), define \( \delta_n = (\gamma_{n+1} - \gamma_n) \cdot \frac{\log \gamma_n}{2\pi} \). Montgomery's result concerns correlations of \(\delta_n\) at separated indices — not individual spacing values.

## Odlyzko numerics (established phenomenology)

Massive numerical studies show GUE-like **level repulsion** and spacing distributions for high zeros. This is **empirical support**, not a proof of full GUE universality for zeta.

## Wiki guardrails

- Say "GUE-like **pair** correlation (conditional on RH)" — not "zeros are GUE eigenvalues."
- Do not extrapolate Montgomery to 3×3 Yukawa spacing tests — insufficient statistics ([[plausibility-register]]).
- Connect to [[explicit-formula-primes-zeros]] for rigorous prime↔zero links, not flavor.

## Related programs

- [[random-matrix-theory]] — GUE ensemble definition
- [[berry-keating-hamiltonian]] — heuristic chaotic Hamiltonians (not Montgomery proof)
- [[connes-spectral-triple]] — alternative operator program
