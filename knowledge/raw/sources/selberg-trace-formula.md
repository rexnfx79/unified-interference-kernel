# Selberg Trace Formula (Curated)

> **Primary:** Selberg (1956); Hejhal, *The Selberg Trace Formula for PSL(2,Z)*; Iwaniec–Kowalski, *Analytic Number Theory*.  
> **Wiki:** [[selberg-trace-formula]]

## Role in the bridge program

The Selberg trace formula is the **archetype** of “geometric side = primes, spectral side = eigenvalues” on a **hyperbolic surface** (or other locally symmetric space). It motivates the Hilbert–Polya picture for \(\zeta\) without proving it.

## Schematic form (compact surface)

For a cofinite Fuchsian group \(\Gamma \backslash \mathbb{H}\),

\[
\sum_{\text{closed geodesics } \gamma} \frac{\ell_\gamma}{2\sinh(\ell_\gamma/2)} h(\ell_\gamma)
\;=\;
\sum_{j=0}^\infty h(r_j) + \text{parabolic / elliptic terms}
\]

- **Left (geometric):** closed geodesics — “prime-like” lengths \(\ell_\gamma\)
- **Right (spectral):** Laplacian eigenvalues \( \lambda_j = 1/4 + r_j^2 \)

## Parallel to Riemann explicit formula

| Selberg (hyperbolic) | Riemann (rational) |
|----------------------|-------------------|
| Closed geodesics | Prime powers \(p^k\) |
| Laplacian spectrum on \(\Gamma\backslash\mathbb{H}\) | Zeta zeros \(\rho = 1/2 + i\gamma\) |
| Proved for specific \(\Gamma\) | Proved as analytic identity |
| Does **not** identify \(\zeta\) zeros with a known SM operator | Same |

## Established vs open

| Item | Status |
|------|--------|
| Trace formula for cofinite \(\Gamma\) | **Established** |
| Selberg zeta / Ruelle zeta links spectrum to geodesics | **Established** (framework) |
| Deduce classical \(\zeta(s)\) zeros from a **single** known physical Laplacian | **Not done** |
| Predict flavor Yukawas | **Out of scope** |

## Physics contact

- **Quantum chaos:** eigenvalue statistics of chaotic billiards link to RMT (established for many systems).
- **Not SM flavor:** geodesic lengths are not CKM parameters.

## Wiki use

Template for [[explicit-formula-primes-zeros]] and [[connes-spectral-triple]]: always state **which side is primes / geodesics** and **which side is spectrum**.

## Falsifier (conceptual)

Claiming “Selberg proves zeros are physical eigenvalues of the SM Hamiltonian” — **false** without an explicit \(\Gamma\) / operator embedding.

## Related

[[explicit-formula-primes-zeros]], [[hilbert-polya-conjecture]], [[random-matrix-theory]], [[connes-spectral-triple]]
