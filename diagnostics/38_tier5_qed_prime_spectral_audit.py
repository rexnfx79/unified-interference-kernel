#!/usr/bin/env python3
"""
Tier 5.5 — Do primes enter standard QED spectral sums without redefinition?

Pre-registered (see can-primes-enter-via-qed-spectral-sums):
  PASS (pursue): some standard observable O matches a prime-only sum with
    rel_err < PRIME_PASS_REL vs textbook O, using the same summand f(n)→f(p)
    with no extra reweighting / spectrum redesign.
  FAIL (expected): integer-index partial sums converge to O; prime-only surrogates do not.

Euler product ∏_p (1-p^{-s})^{-1} = ∑_n n^{-s} is noted but classified as
  analytic reparameterization — NOT a prime-index mode sum in QED textbooks.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import math

from qed_spectral_sums import (
    SCHWINGER_SUM,
    ZETA2,
    ZETA3,
    ZETA4,
    audit_observable,
    casimir_mode_sum_integer,
    casimir_mode_sum_prime,
    euler_product_zeta,
    integer_partial_sum_zeta,
    prime_partial_sum_zeta,
    schwinger_g2_series_integer,
    schwinger_g2_series_prime,
    vacuum_polarization_coefficient_integer,
    vacuum_polarization_coefficient_prime,
)

RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "results", "38_tier5_qed_prime_spectral_audit.txt"
)

N_MAX = 200_000
PRIME_PASS_REL = 0.01


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    n_max = 5000 if args.smoke else N_MAX

    audits = []

    # 1. Schwinger one-loop g-2: (α/2π) ∑_k 1/(k(k+1)) — integer k
    k_max = n_max
    s_int = schwinger_g2_series_integer(k_max)
    s_pri = schwinger_g2_series_prime(n_max)
    audits.append(
        audit_observable(
            "schwinger_g2_sum",
            SCHWINGER_SUM,
            s_int,
            s_pri,
        )
    )

    # 2. ζ(2) — appears in QED coefficients / EH expansion
    z2_int = integer_partial_sum_zeta(2.0, n_max)
    z2_pri = prime_partial_sum_zeta(2.0, n_max)
    z2_eul = euler_product_zeta(2.0, n_max)
    audits.append(
        audit_observable("zeta2_sum", ZETA2, z2_int, z2_pri, z2_eul)
    )

    # 3. ζ(4) — vacuum polarization / EH
    z4_int = integer_partial_sum_zeta(4.0, n_max)
    z4_pri = prime_partial_sum_zeta(4.0, n_max)
    z4_eul = euler_product_zeta(4.0, n_max)
    audits.append(
        audit_observable("zeta4_sum", ZETA4, z4_int, z4_pri, z4_eul)
    )

    # 4. ζ(3) — two-loop QED context
    z3_int = integer_partial_sum_zeta(3.0, n_max)
    z3_pri = prime_partial_sum_zeta(3.0, n_max)
    audits.append(
        audit_observable("zeta3_sum", ZETA3, z3_int, z3_pri)
    )

    # 5. Casimir: textbook ∑_{n=1}^∞ n → ζ(-1)=-1/12 (integer modes, not prime index)
    c_int = casimir_mode_sum_integer(n_max)
    c_pri = casimir_mode_sum_prime(n_max)
    # Partial sums diverge; compare prime/integer raw ratio (→ 0, not ζ(-1))
    ratio_pri_int = c_pri / c_int if c_int > 0 else 0.0
    audits.append(
        audit_observable(
            "casimir_prime_vs_integer_ratio",
            0.0,  # prime-only does not reproduce ζ(-1) limit
            0.0,
            ratio_pri_int,
        )
    )

    # 6. Vacuum pol proxy ∑ n^{-4}
    vp_int = vacuum_polarization_coefficient_integer(n_max)
    vp_pri = vacuum_polarization_coefficient_prime(n_max)
    audits.append(
        audit_observable("vacuum_pol_zeta4", ZETA4, vp_int, vp_pri)
    )

    prime_pass = any(a["rel_err_prime_only"] < PRIME_PASS_REL for a in audits)
    conv_names = {"schwinger_g2_sum", "zeta2_sum", "zeta4_sum", "zeta3_sum", "vacuum_pol_zeta4"}
    integer_ok = all(
        a["rel_err_integer"] < PRIME_PASS_REL
        for a in audits
        if a["name"] in conv_names
    )

    lines = [
        "=" * 72,
        "TIER 5.5 QED PRIME SPECTRAL AUDIT (diagnostic 38)",
        "=" * 72,
        "Question: do standard QED sums use prime indices without redefinition?",
        f"N_max={n_max}; pass threshold rel_err < {PRIME_PASS_REL} (prime-only vs target)",
        "",
        "--- Observables (integer index = textbook; prime-only = falsifier test) ---",
    ]

    for a in audits:
        lines.append(f"  {a['name']}:")
        lines.append(f"    target={a['target']:.8g}")
        lines.append(
            f"    integer partial: {a['integer']:.8g}  rel_err={a['rel_err_integer']:.4e}"
        )
        lines.append(
            f"    prime-only:      {a['prime_only']:.8g}  rel_err={a['rel_err_prime_only']:.4e}"
        )
        if "euler_product" in a:
            lines.append(
                f"    euler product:   {a['euler_product']:.8g}  rel_err={a['rel_err_euler']:.4e}"
            )

    lines.extend(
        [
            "",
            "--- Classification ---",
            "  euler_product: analytic identity for ζ(s); not prime-index mode sum in QED.",
            "  prime-only: replaces n with primes in same summand — NOT standard unless rel_err passes.",
            "",
            f"  integer partials converge: {integer_ok}",
            f"  any prime-only pass: {prime_pass}",
            "",
            "--- VERDICT ---",
        ]
    )

    if prime_pass:
        verdict = "pursue"
        lines.append(
            "  UNEXPECTED PASS — prime-only sum matches a standard QED observable."
        )
        lines.append("  Upgrade [[can-primes-enter-via-qed-spectral-sums]] to pursue.")
    else:
        verdict = "fail_no_prime_index"
        lines.append(
            "  FAIL — no standard observable equals a prime-only sum without redefinition."
        )
        lines.append(
            "  Primes enter only via Euler product / voluntary reindexing, not QED mode sums."
        )
        lines.append(
            "  Keep A↔C bridge at watch; no HP→flavor or zeta→CKM claim."
        )

    report = "\n".join(lines)
    print(report)

    if not args.smoke:
        os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
        with open(RESULTS_PATH, "w") as f:
            f.write(report + "\n")
            f.write(f"verdict: {verdict}\n")
            f.write("flavor_connection: false\n")
        print(f"\nSaved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
