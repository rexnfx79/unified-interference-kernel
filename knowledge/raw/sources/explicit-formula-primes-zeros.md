# Explicit Formula — Primes and Zeta Zeros (Curated)

> **Primary:** Edwards, *Riemann's Zeta Function* (1974), Ch. 12; Davenport, *Multiplicative Number Theory*; von Mangoldt / Riemann explicit formula.  
> **Repo demo:** `diagnostics/14_explicit_formula_numerics.py`  
> **Wiki:** [[explicit-formula-primes-zeros]]

## Chebyshev form (established)

Let \(\psi(x) = \sum_{p^k \le x} \log p\). Then (under standard assumptions on zero placement),

\[
\psi(x) = x - \sum_{\rho} \frac{x^\rho}{\rho} - \log(2\pi) - \tfrac{1}{2}\log(1 - x^{-2}) + \cdots
\]

where \(\rho\) runs over non-trivial zeros of \(\zeta(s)\), and the sum is ordered by \(|\Im\rho|\).

**Meaning:** Prime counting error is a **superposition of oscillations** whose frequencies are zero heights \(\gamma = \Im\rho\).

## Prime counting \(\pi(x)\)

\(\pi(x) \sim \mathrm{Li}(x)\) with corrections expressible via the same zero sum (Riemann–von Mangoldt). The repo diagnostic compares \(\psi(x)\), \(x\), \(\mathrm{Li}(x)\), and a truncated zero sum using tabulated low zeros.

## Weil explicit formula (established mathematics)

Adelic formulation: test function \(h\) on \(\mathbb{A}_\mathbb{Q}\) with

\[
\sum_v \mathrm{local}(h_v) = \sum_\rho \hat{h}(\rho) + \text{archimedean + pole terms}.
\]

Left side encodes **primes** (local factors); right side encodes **zeros**. This is the template for **trace formulas** ([[selberg-trace-formula]], [[connes-spectral-triple]]).

## What is proven vs conjectural

| Statement | Status |
|-----------|--------|
| Formula relating \(\psi(x)\) to zeros (under RH or weak zero-free region) | **Established** |
| RH (all non-trivial zeros on \(\Re s = 1/2\)) | **Open** |
| Zeros = eigenvalues of a known self-adjoint \(H\) | **Open** ([[hilbert-polya-conjecture]]) |
| Zero sum predicts SM Yukawa phases | **Dead** ([[why-not-zeta-flavor-numerology]]) |

## Empirical anchor (not SM collider physics)

- **Primes** are verified by arithmetic computation (\(\psi(x)\), \(\pi(x)\)) — not collider observables.
- **Zeros** are computed to high precision (Odlyzko tables); frequencies enter prime oscillations.
- **No SM hook** unless a QFT operator’s spectral measure equals the Weil spectral side.

## Repo numerics (diag 14)

At \(x = 10^4\): \(\psi(x) \approx x\) with \(O(10^2)\) correction; truncated 10-zero sum captures **part** of oscillatory side (qualitative). Output: `diagnostics/results/14_explicit_formula_numerics.txt`.

## Tier 5.2 (blocked)

See [[adversarial-review-tier5-trace-formula]] — holdout \(\psi(x)\) + wrong-frequency null required; naive frequency matching is tautological.

## Guardrails

- Do not map individual \(\gamma_n\) to CKM angles.
- Pair with [[montgomery-pair-correlation]] for **statistics** of zeros, not with 3×3 matrices.

## Related

[[selberg-trace-formula]], [[hilbert-polya-conjecture]], [[riemann-zeta-function]], [[connes-spectral-triple]]
