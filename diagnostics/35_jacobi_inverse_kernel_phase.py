#!/usr/bin/env python3
"""
Tier 5.3 — Jacobi / tridiagonal inverse problem for kernel Yukawa (B -> D).

Given optimized kernel Yukawa Y, fit:
  (A) general 3x3 Hermitian H_gen
  (B) real symmetric tridiagonal H_tri

Falsifiers (pre-registered):
  - Relative Frobenius residual < 0.12 for BOTH -> would support simple operator story
  - If only H_gen fits -> bilinear phase not a minimal Laplacian+BC model
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from scipy.optimize import differential_evolution

from kernel import compute_yukawa_matrix, compute_quark_yukawas
from observables import compute_quark_observables, compute_training_loss

RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "results", "35_jacobi_inverse_kernel_phase.txt"
)

Q, U, D = (0, 1, 0), (0, 3, 6), (0, 3, 7)
DEFAULT = dict(sigma=4.0, k=1.4, alpha=2.5, eta=2.0, eps_u=0.15, eps_d=0.15)
REL_RESID_MAX = 0.12  # pass bar for "simple H" story


def frob_rel(Y, Yhat):
    return float(np.linalg.norm(Y - Yhat, "fro") / (np.linalg.norm(Y, "fro") + 1e-15))


def hermitian_from_params(p):
    """3x3 Hermitian: diag real + three upper-triangle complex (8 real)."""
    e0, e1, e2, re01, im01, re02, im02, re12, im12 = p
    H = np.array(
        [
            [e0, re01 + 1j * im01, re02 + 1j * im02],
            [re01 - 1j * im01, e1, re12 + 1j * im12],
            [re02 - 1j * im02, re12 - 1j * im12, e2],
        ],
        dtype=complex,
    )
    return H


def tridiagonal_from_params(p):
    e0, e1, e2, t01, t12 = p
    return np.array(
        [[e0, t01, 0.0], [t01, e1, t12], [0.0, t12, e2]],
        dtype=float,
    )


def yukawa_from_hermitian(H, scale, phase_alpha):
    """Rank-1-style: Y_ij = scale * exp(i*phase_alpha) * |H_ij| — crude overlap proxy."""
    mag = np.abs(H)
    phi = np.angle(H + 1e-15) + phase_alpha
    return scale * mag * np.exp(1j * phi)


def optimize_yukawa_target():
    bounds = [
        (0.5, 6.0),
        (0.1, 2.5),
        (0.0, 2 * np.pi),
        (1.0, 5.0),
        (0.01, 0.5),
        (0.01, 0.5),
    ]

    def objective(theta):
        Yu, Yd = compute_quark_yukawas(Q, U, D, *theta)
        return compute_training_loss(compute_quark_observables(Yu, Yd))

    res = differential_evolution(
        objective, bounds, seed=35035, maxiter=80, popsize=10, polish=True
    )
    Yu, Yd = compute_quark_yukawas(Q, U, D, *res.x)
    return Yu, res.x


def fit_hermitian(Y_target):
    n = 9

    def loss(p):
        H = hermitian_from_params(p[:9])
        scale, alpha = p[9], p[10]
        Yhat = yukawa_from_hermitian(H, scale, alpha)
        return float(np.linalg.norm(Y_target - Yhat) ** 2)

    bounds = [(-3, 3)] * 9 + [(1e-6, 10.0), (-np.pi, np.pi)]
    r = differential_evolution(loss, bounds, seed=35036, maxiter=120, popsize=12)
    H = hermitian_from_params(r.x[:9])
    Yhat = yukawa_from_hermitian(H, r.x[9], r.x[10])
    return frob_rel(Y_target, Yhat), Yhat


def fit_tridiagonal(Y_target):
    def loss(p):
        H = tridiagonal_from_params(p[:5])
        scale, alpha = p[5], p[6]
        Yhat = yukawa_from_hermitian(H.astype(complex), scale, alpha)
        return float(np.linalg.norm(Y_target - Yhat) ** 2)

    bounds = [(-3, 3)] * 5 + [(1e-6, 10.0), (-np.pi, np.pi)]
    r = differential_evolution(loss, bounds, seed=35037, maxiter=120, popsize=12)
    H = tridiagonal_from_params(r.x[:5])
    Yhat = yukawa_from_hermitian(H.astype(complex), r.x[5], r.x[6])
    return frob_rel(Y_target, Yhat), Yhat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    print("Tier 5.3: Jacobi inverse audit...")
    Yu, _theta = optimize_yukawa_target()

    rel_gen, _ = fit_hermitian(Yu)
    rel_tri, _ = fit_tridiagonal(Yu)

    lines = [
        "=" * 72,
        "TIER 5.3 JACOBI INVERSE KERNEL PHASE (diagnostic 35)",
        "=" * 72,
        f"Geometry Q={Q}, U={U}, D={D}",
        f"Optimized Yu train loss context: theta from DE",
        "",
        f"Target ||Yu||_F = {np.linalg.norm(Yu):.4f}",
        f"Relative residual H_gen fit:  {rel_gen:.4f}",
        f"Relative residual H_tri fit:  {rel_tri:.4f}",
        f"Pass bar (both < {REL_RESID_MAX}): gen={rel_gen < REL_RESID_MAX} tri={rel_tri < REL_RESID_MAX}",
        "",
        "--- VERDICT ---",
    ]

    if rel_tri < REL_RESID_MAX:
        lines.append(
            "  Tridiagonal fit passes bar — 3-site operator *may* approximate Yukawa texture."
        )
        lines.append("  Still post-hoc; does not predict kernel params from geometry.")
        verdict = "partial"
    elif rel_gen < REL_RESID_MAX:
        lines.append(
            "  General Hermitian fits; tridiagonal fails — not a minimal Laplacian story."
        )
        verdict = "fail_tri"
    else:
        lines.append(
            "  FAIL — neither Hermitian ansatz matches kernel Yu at pre-registered tolerance."
        )
        lines.append("  Bilinear kernel phase is not reducible to this 3-parameter H proxy.")
        verdict = "fail"

    report = "\n".join(lines)
    print(report)

    if not args.smoke:
        os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
        with open(RESULTS_PATH, "w") as f:
            f.write(report + "\n")
            f.write(f"verdict: {verdict}\n")
        print(f"\nSaved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
