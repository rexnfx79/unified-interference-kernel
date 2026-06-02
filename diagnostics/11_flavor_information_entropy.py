#!/usr/bin/env python3
"""
Flavor information entropy diagnostic (wiki: information-measure-for-projection-regimes).

Computes S(rho_Y), effective rank, and off-diagonal entropy per fermion sector
from representative Yukawa matrices (quark kernel + lepton transfer geometry).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from kernel import compute_quark_yukawas, compute_yukawa_matrix
from flavor_information import compute_yukawa_information


# Representative geometries / params from repo optimization archives
QUARK_GEOMETRY = ((0, 1, 0), (0, 3, 6), (0, 3, 7))
QUARK_PARAMS = dict(sigma=4.0, k=1.4, alpha=2.5, eta=2.0, eps_u=0.15, eps_d=0.15)

LEPTON_GEOMETRY = ((0, 1, 0), (0, 3, 6))
LEPTON_PARAMS = dict(sigma=4.0, k=1.75, alpha=2.5, eta=3.7, eps=0.41)

NEUTRINO_GEOMETRY = ((0, 1, 0), (0, 3, 6))
NEUTRINO_PARAMS = dict(
    sigma=4.0, k=1.4, alpha=2.5, eta=2.0, eps_nu=0.15, eps_e=0.41, g_env=0.55
)


def build_neutrino_yukawas():
    """Construct Y_nu and Y_e with metric-dominated envelope compression."""
    L, N = NEUTRINO_GEOMETRY
    p = NEUTRINO_PARAMS
    g = p["g_env"]
    Ynu = compute_yukawa_matrix(L, N, p["sigma"] * g, p["k"], p["alpha"], p["eta"], p["eps_nu"])
    Ye = compute_yukawa_matrix(L, N, p["sigma"], p["k"], p["alpha"], p["eta"], p["eps_e"])
    return Ynu, Ye


def main():
    Yu, Yd = compute_quark_yukawas(*QUARK_GEOMETRY, **QUARK_PARAMS)
    Ye = compute_yukawa_matrix(*LEPTON_GEOMETRY, **LEPTON_PARAMS)
    Ynu, Ye_nu = build_neutrino_yukawas()

    sectors = [
        ("quark_up", Yu),
        ("quark_down", Yd),
        ("charged_lepton", Ye),
        ("neutrino", Ynu),
        ("charged_lepton_pmns", Ye_nu),
    ]

    print("=" * 60)
    print("FLAVOR INFORMATION ENTROPY BY SECTOR")
    print("=" * 60)
    print(f"Quark geometry: Q={QUARK_GEOMETRY[0]}, U={QUARK_GEOMETRY[1]}, D={QUARK_GEOMETRY[2]}")
    print(f"Lepton geometry: L={LEPTON_GEOMETRY[0]}, E={LEPTON_GEOMETRY[1]}")
    print(f"Neutrino g_env={NEUTRINO_PARAMS['g_env']}\n")

    results = {}
    for name, Y in sectors:
        info = compute_yukawa_information(Y)
        results[name] = info
        print(f"{name}:")
        print(f"  S(rho_Y)           = {info['entropy']:.6f} nats")
        print(f"  effective rank     = {info['effective_rank']:.6f}")
        print(f"  off-diag entropy   = {info['off_diagonal_entropy']:.6f} nats")
        print(f"  Tr(Y Y†)           = {info['trace_yydag']:.6e}")
        print()

    S_quark = (results["quark_up"]["entropy"] + results["quark_down"]["entropy"]) / 2
    S_lep = results["charged_lepton"]["entropy"]
    S_nu = results["neutrino"]["entropy"]
    print("Summary (hypothesis check):")
    print(f"  mean quark S       = {S_quark:.6f}")
    print(f"  lepton S           = {S_lep:.6f}")
    print(f"  neutrino S         = {S_nu:.6f}")
    if S_nu > S_quark:
        print("  → Neutrino entropy > quark (consistent with metric-dominated regime hypothesis)")
    else:
        print("  → Neutrino entropy ≤ quark (hypothesis not confirmed at this geometry)")

    out = os.path.join(os.path.dirname(__file__), "results", "11_flavor_information_entropy.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write("Flavor information entropy by sector\n")
        for name, info in results.items():
            f.write(f"\n[{name}]\n")
            for k, v in info.items():
                f.write(f"{k}: {v}\n")
        f.write(f"\nmean_quark_entropy: {S_quark}\n")
        f.write(f"lepton_entropy: {S_lep}\n")
        f.write(f"neutrino_entropy: {S_nu}\n")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
