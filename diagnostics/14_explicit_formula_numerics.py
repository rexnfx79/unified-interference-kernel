#!/usr/bin/env python3
"""
Explicit formula numerics (Path D — educational, SM-decoupled).

Numerical demo linking prime counting to the oscillatory zero-sum side of
the explicit formula. Does NOT connect to flavor physics or Yukawa matrices.

Educational grounding for wiki: explicit-formula-primes-zeros, hilbert-polya-conjecture.

Path D hook (see knowledge/wiki/synthesis/research-strategy.md): if Path A ever
identifies a flavor spectral measure dμ(λ), compare its Stieltjes moments to the
oscillatory prime side here — SM-decoupled; no Yukawa/CKM mapping.
"""

import os
import math


def primes_up_to(n):
    if n < 2:
        return []
    sieve = bytearray(b"\x01") * (n + 1)
    sieve[0:2] = b"\x00\x00"
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            step = i
            start = i * i
            sieve[start : n + 1 : step] = b"\x00" * ((n - start) // step + 1)
    return [i for i in range(2, n + 1) if sieve[i]]


def chebyshev_psi(x):
    """Chebyshev psi(x) = sum_{p^k <= x} log p."""
    if x < 2:
        return 0.0
    n = int(x)
    primes = primes_up_to(n)
    total = 0.0
    for p in primes:
        pk = p
        while pk <= x:
            total += math.log(p)
            pk *= p
    return total


def li(x):
    """Logarithmic integral (principal)."""
    if x <= 1:
        return 0.0
    # Simple Simpson on [2, x] for 1/log t
    a, b = 2.0, float(x)
    steps = 200
    h = (b - a) / steps
    s = 0.0
    for i in range(steps + 1):
        t = a + i * h
        w = 1.0 if i not in (0, steps) else 0.5
        if t > 1:
            s += w / math.log(t)
    return h * s


def zero_sum_contribution(x, zeros):
    """Oscillatory sum -sum x^rho/rho using known low zeros (imag parts)."""
    total = 0.0
    for gamma in zeros:
        rho = 0.5 + 1j * gamma
        try:
            term = (x ** rho) / rho
            total += term
        except (OverflowError, ValueError):
            continue
    return -total.real


# First several Riemann zero imaginary parts (established tables)
KNOWN_ZERO_IMAG = [
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
]


def main():
    print("=" * 70)
    print("EXPLICIT FORMULA NUMERICS (Path D — no flavor connection)")
    print("=" * 70)
    print("Compares psi(x), x, Li(x), and low-zero oscillatory correction.\n")

    xs = [100, 500, 1000, 5000, 10000]
    rows = []

    print(f"{'x':>8}  {'psi(x)':>12}  {'x':>12}  {'Li(x)':>12}  {'psi-x':>10}  {'zero_corr':>10}")
    for x in xs:
        psi_x = chebyshev_psi(x)
        li_x = li(x)
        err = psi_x - x
        zcorr = zero_sum_contribution(x, KNOWN_ZERO_IMAG)
        rows.append((x, psi_x, li_x, err, zcorr))
        print(f"{x:8d}  {psi_x:12.2f}  {x:12.0f}  {li_x:12.2f}  {err:+10.2f}  {zcorr:+10.2f}")

    print("\nInterpretation:")
    print("  psi(x) ~ x + oscillatory corrections from zeta zeros (explicit formula).")
    print("  Li(x) approximates prime counting; gap to psi reflects prime powers.")
    print("  zero_corr uses first 10 known zeros — qualitative, not a proof.")
    print("\nPath D status: WATCH — arithmetic/spectral bridge only; no SM hook without trace operator.")

    out = os.path.join(os.path.dirname(__file__), "results", "14_explicit_formula_numerics.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write("Explicit formula numerics (Path D, educational)\n\n")
        f.write("x  psi(x)  Li(x)  psi_minus_x  zero_correction\n")
        for x, psi_x, li_x, err, zcorr in rows:
            f.write(f"{x}  {psi_x:.6f}  {li_x:.6f}  {err:.6f}  {zcorr:.6f}\n")
        f.write("\nno_flavor_connection: true\n")
        f.write("path_d_status: watch\n")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
