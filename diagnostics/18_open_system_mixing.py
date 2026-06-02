#!/usr/bin/env python3
"""
Open-system decoherence vs mixing (Path A, SM only).

Sweep g_env (neutrinos) and kernel params; decoherence rate p from **external**
parameters only; compare open-system mixing proxy vs actual PMNS/CKM.

Pre-registered: max |r| >= 0.25 pooled or REFUTED.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import itertools
import numpy as np
from scipy import stats

from kernel import compute_quark_yukawas, compute_yukawa_matrix
from observables import compute_quark_observables, compute_neutrino_observables
from open_system_decoherence import (
    compute_open_system_row,
    external_decoherence_parameter,
    open_system_mixing_proxy,
)

CORRELATION_THRESHOLD = 0.25
MIN_SAMPLES = 20
ALPHA = 2.5

QUARK_GEOMETRIES = [
    {"name": "standard", "Q": (0, 1, 0), "U": (0, 3, 6), "D": (0, 3, 7)},
    {"name": "csv_compact", "Q": (0, 1, 0), "U": (0, 1, 2), "D": (0, 1, 3)},
    {"name": "spread_020", "Q": (0, 2, 0), "U": (0, 2, 4), "D": (0, 2, 5)},
]

NEUTRINO_GEOMETRIES = [
    {"name": "standard", "L": (0, 1, 0), "N": (0, 3, 6)},
    {"name": "compact", "L": (0, 1, 0), "N": (0, 1, 2)},
    {"name": "spread", "L": (0, 2, 0), "N": (0, 2, 4)},
]

SIGMA_VALS = [2.5, 4.0, 5.5]
K_VALS = [1.0, 1.4, 1.75, 2.0]
ETA_VALS = [2.0, 3.0, 3.7, 4.5]
EPS_VALS = [0.10, 0.25, 0.41]
G_ENV_VALS = [0.45, 0.55, 0.65, 0.75]


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


def sweep_quarks():
    rows = []
    for geom in QUARK_GEOMETRIES:
        Q, U, D = geom["Q"], geom["U"], geom["D"]
        for sigma, k, eta, eps_u, eps_d in itertools.product(
            SIGMA_VALS, K_VALS, ETA_VALS, EPS_VALS, EPS_VALS
        ):
            Yu, Yd = compute_quark_yukawas(Q, U, D, sigma, k, ALPHA, eta, eps_u, eps_d)
            obs = compute_quark_observables(Yu, Yd)
            actual = obs["Vus"] + obs["Vcb"] + obs["Vub"]
            p = external_decoherence_parameter("quark", eps_u=eps_u, eps_d=eps_d)
            for label, Y in [("up", Yu), ("down", Yd)]:
                row = compute_open_system_row("quark", Y, p, actual)
                row["geometry"] = geom["name"]
                row["label"] = label
                row["p_from_eps"] = p
                rows.append(row)
    return rows


def sweep_neutrinos():
    rows = []
    for geom in NEUTRINO_GEOMETRIES:
        L, N = geom["L"], geom["N"]
        for sigma, k, eta, eps_nu, eps_e, g_env in itertools.product(
            SIGMA_VALS, K_VALS, ETA_VALS, EPS_VALS, EPS_VALS, G_ENV_VALS
        ):
            Ynu = compute_yukawa_matrix(L, N, sigma * g_env, k, ALPHA, eta, eps_nu)
            Ye = compute_yukawa_matrix(L, N, sigma, k, ALPHA, eta, eps_e)
            obs = compute_neutrino_observables(Ynu, Ye)
            actual = obs["theta12"] + obs["theta23"] + obs["theta13"]
            p = external_decoherence_parameter("neutrino", g_env=g_env)
            row = compute_open_system_row("neutrino", Ynu, p, actual)
            row["geometry"] = geom["name"]
            row["g_env"] = g_env
            rows.append(row)
    return rows


def main():
    print("=" * 70)
    print("OPEN-SYSTEM DECOHERENCE vs MIXING (SM only)")
    print("=" * 70)
    print(f"Falsifier: max |r| >= {CORRELATION_THRESHOLD} (pooled)\n")

    quark = sweep_quarks()
    neutrino = sweep_neutrinos()
    rows = quark + neutrino
    n = len(rows)
    print(f"Samples: quark={len(quark)}, neutrino={len(neutrino)}, total={n}")

    correlations = {}

    print("\n--- External p vs actual mixing (pooled) ---")
    p_vals = [r["p_external"] for r in rows]
    mix_actual = [r["actual_mixing"] for r in rows]
    r_p, pval = pearson_safe(p_vals, mix_actual)
    correlations["p_external_vs_actual_mixing"] = r_p
    print(f"  p_external vs actual_mixing  r = {r_p:+.4f}  p = {pval:.2e}")

    print("\n--- Open-system proxy vs actual mixing (pooled) ---")
    proxy = [r["mixing_proxy_open"] for r in rows]
    r_proxy, pval2 = pearson_safe(proxy, mix_actual)
    correlations["mixing_proxy_open_vs_actual"] = r_proxy
    print(f"  mixing_proxy_open vs actual  r = {r_proxy:+.4f}  p = {pval2:.2e}")

    print("\n--- By sector ---")
    for sector in ("quark", "neutrino"):
        sub = [r for r in rows if r["sector"] == sector]
        xs = [r["mixing_proxy_open"] for r in sub]
        ys = [r["actual_mixing"] for r in sub]
        r_s, _ = pearson_safe(xs, ys)
        correlations[f"{sector}_proxy_vs_actual"] = r_s
        rp, _ = pearson_safe([r["p_external"] for r in sub], ys)
        correlations[f"{sector}_p_vs_actual"] = rp
        print(f"  [{sector}] proxy vs actual r = {r_s:+.4f}; p vs actual r = {rp:+.4f}")

    mix_keys = [k for k in correlations if "_vs_" in k]
    finite_rs = [abs(correlations[k]) for k in mix_keys if np.isfinite(correlations[k])]
    max_abs_r = max(finite_rs) if finite_rs else 0.0

    passed = n >= MIN_SAMPLES and max_abs_r >= CORRELATION_THRESHOLD

    print("\n" + "=" * 70)
    print("PRE-REGISTERED FALSIFIER VERDICT")
    print("=" * 70)
    print(f"  Max |r|: {max_abs_r:.4f}")
    if passed:
        print("  VERDICT: NOT REFUTED at threshold")
        verdict = "not_refuted"
    else:
        print("  VERDICT: REFUTED — open-system proxy does not correlate with mixing")
        verdict = "refuted"

    out = os.path.join(os.path.dirname(__file__), "results", "18_open_system_mixing.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write("Open-system decoherence vs mixing diagnostic\n")
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
