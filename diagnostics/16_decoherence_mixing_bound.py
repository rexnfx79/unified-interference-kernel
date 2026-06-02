#!/usr/bin/env python3
"""
Decoherence proxy vs mixing bound (Path A, SM only).

Tests whether off-diagonal coherence measures upper-bound or correlate with
CKM / PMNS magnitudes from compute_quark_observables / compute_neutrino_observables.

Pre-registered: if max |r| between decoherence proxies and |V_us|, |V_cb|, |V_ub|
or PMNS angles < 0.25 (pooled quark+neutrino mixing targets), report refuted.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import itertools
import numpy as np
from scipy import stats

from kernel import compute_quark_yukawas, compute_yukawa_matrix
from qed_information import (
    coherence_l1_norm,
    off_diagonal_to_diagonal_ratio,
    yukawa_density_matrix,
)
from observables import compute_quark_observables, compute_neutrino_observables

CORRELATION_THRESHOLD = 0.25
MIN_SAMPLES = 20

QUARK_GEOM = {"Q": (0, 1, 0), "U": (0, 3, 6), "D": (0, 3, 7)}
NEUTRINO_GEOM = {"L": (0, 1, 0), "N": (0, 3, 6)}

SIGMA_VALS = [2.5, 4.0, 5.5]
K_VALS = [1.0, 1.4, 1.75, 2.0]
ETA_VALS = [2.0, 3.0, 3.7, 4.5]
EPS_VALS = [0.10, 0.25, 0.41]
G_ENV_VALS = [0.45, 0.55, 0.65, 0.75]
ALPHA = 2.5


def pearson_safe(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return np.nan, np.nan
    if np.std(x[mask]) < 1e-15 or np.std(y[mask]) < 1e-15:
        return np.nan, np.nan
    r, p = stats.pearsonr(x[mask], y[mask])
    return float(r), float(p)


def upper_bound_fraction(deco, mix, tol=1e-12):
    """Fraction of samples where deco >= mix (loose upper-bound test)."""
    deco = np.asarray(deco, dtype=float)
    mix = np.asarray(mix, dtype=float)
    mask = np.isfinite(deco) & np.isfinite(mix)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(deco[mask] >= mix[mask] - tol))


def collect_quark_rows():
    rows = []
    Q, U, D = QUARK_GEOM["Q"], QUARK_GEOM["U"], QUARK_GEOM["D"]
    for sigma, k, eta, eps_u, eps_d in itertools.product(
        SIGMA_VALS, K_VALS, ETA_VALS, EPS_VALS, EPS_VALS
    ):
        Yu, Yd = compute_quark_yukawas(Q, U, D, sigma, k, ALPHA, eta, eps_u, eps_d)
        obs = compute_quark_observables(Yu, Yd)
        for label, Y in [("up", Yu), ("down", Yd)]:
            rho = yukawa_density_matrix(Y)
            rows.append(
                dict(
                    sector="quark",
                    label=label,
                    coherence_l1=coherence_l1_norm(rho),
                    off_ratio=off_diagonal_to_diagonal_ratio(rho),
                    Vus=obs["Vus"],
                    Vcb=obs["Vcb"],
                    Vub=obs["Vub"],
                    mix_sum=obs["Vus"] + obs["Vcb"] + obs["Vub"],
                )
            )
    return rows


def collect_neutrino_rows():
    rows = []
    L, N = NEUTRINO_GEOM["L"], NEUTRINO_GEOM["N"]
    for sigma, k, eta, eps_nu, eps_e, g_env in itertools.product(
        SIGMA_VALS, K_VALS, ETA_VALS, EPS_VALS, EPS_VALS, G_ENV_VALS
    ):
        Ynu = compute_yukawa_matrix(L, N, sigma * g_env, k, ALPHA, eta, eps_nu)
        Ye = compute_yukawa_matrix(L, N, sigma, k, ALPHA, eta, eps_e)
        obs = compute_neutrino_observables(Ynu, Ye)
        rho = yukawa_density_matrix(Ynu)
        rows.append(
            dict(
                sector="neutrino",
                label="nu",
                coherence_l1=coherence_l1_norm(rho),
                off_ratio=off_diagonal_to_diagonal_ratio(rho),
                theta12=obs["theta12"],
                theta23=obs["theta23"],
                theta13=obs["theta13"],
                mix_sum=obs["theta12"] + obs["theta23"] + obs["theta13"],
            )
        )
    return rows


def main():
    print("=" * 70)
    print("DECOHERENCE PROXY vs MIXING (SM only)")
    print("=" * 70)

    quark = collect_quark_rows()
    neutrino = collect_neutrino_rows()
    rows = quark + neutrino
    n = len(rows)
    print(f"Samples: quark={len(quark)}, neutrino={len(neutrino)}, total={n}")

    deco_keys = ["coherence_l1", "off_ratio"]
    mix_keys_quark = ["Vus", "Vcb", "Vub", "mix_sum"]
    mix_keys_nu = ["theta12", "theta23", "theta13", "mix_sum"]

    correlations = {}
    print("\n--- Quark: decoherence vs CKM ---")
    for dk in deco_keys:
        for mk in mix_keys_quark:
            xs = [r[dk] for r in quark]
            ys = [r[mk] for r in quark]
            r, p = pearson_safe(xs, ys)
            key = f"quark_{dk}_vs_{mk}"
            correlations[key] = r
            if np.isfinite(r):
                print(f"  {dk} vs {mk:8s}  r = {r:+.4f}  p = {p:.2e}")

    print("\n--- Neutrino: decoherence vs PMNS ---")
    for dk in deco_keys:
        for mk in mix_keys_nu:
            xs = [r[dk] for r in neutrino]
            ys = [r[mk] for r in neutrino]
            r, p = pearson_safe(xs, ys)
            key = f"neutrino_{dk}_vs_{mk}"
            correlations[key] = r
            if np.isfinite(r):
                print(f"  {dk} vs {mk:8s}  r = {r:+.4f}  p = {p:.2e}")

    # Pooled mixing sectors (CKM + PMNS magnitudes)
    pooled_mix = [r["mix_sum"] for r in rows]
    print("\n--- Pooled (quark + neutrino) ---")
    for dk in deco_keys:
        xs = [r[dk] for r in rows]
        r, p = pearson_safe(xs, pooled_mix)
        correlations[f"pooled_{dk}_vs_mix_sum"] = r
        frac = upper_bound_fraction(xs, pooled_mix)
        correlations[f"pooled_{dk}_upper_bound_frac"] = frac
        if np.isfinite(r):
            print(f"  {dk} vs mix_sum  r = {r:+.4f}  upper_bound_frac = {frac:.3f}")

    mix_corr_keys = [k for k in correlations if "_vs_" in k and "upper_bound" not in k]
    finite_rs = [abs(correlations[k]) for k in mix_corr_keys if np.isfinite(correlations[k])]
    max_abs_r = max(finite_rs) if finite_rs else 0.0

    passed = n >= MIN_SAMPLES and max_abs_r >= CORRELATION_THRESHOLD

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"  Max |r| (decoherence vs mixing targets): {max_abs_r:.4f}")
    if passed:
        print("  NOT REFUTED at threshold — exploratory correlation only")
        verdict = "not_refuted"
    else:
        print("  REFUTED — decoherence proxy does not correlate with mixing at |r|>=0.25")
        verdict = "refuted"

    out = os.path.join(os.path.dirname(__file__), "results", "16_decoherence_mixing_bound.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write("Decoherence vs mixing bound diagnostic\n")
        f.write(f"threshold: {CORRELATION_THRESHOLD}\n")
        f.write(f"n_samples: {n}\n")
        f.write(f"verdict: {verdict}\n")
        f.write(f"max_abs_r: {max_abs_r}\n\n")
        for k, v in sorted(correlations.items()):
            f.write(f"{k}: {v}\n")
    print(f"\nWrote {out}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
