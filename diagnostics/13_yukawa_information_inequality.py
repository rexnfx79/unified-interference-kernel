#!/usr/bin/env python3
"""
Yukawa information inequality explorer (Path A mechanism candidates).

Tests whether information measures satisfy cross-sector inequalities relating to:
  - S(rho_Y) vs mixing angles (CKM / PMNS off-diagonals)
  - effective rank vs mass hierarchy (SVD singular value ratios)
  - off-diagonal entropy vs CKM/PMNS off-diagonal magnitudes

Reports statistical summary; no claim of proof without cross-sector consistency.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import itertools
import numpy as np
from scipy import stats

from kernel import compute_quark_yukawas, compute_yukawa_matrix
from flavor_information import compute_yukawa_information
from observables import compute_quark_observables, compute_neutrino_observables

SIGMA_VALS = [2.5, 4.0, 5.5]
K_VALS = [1.0, 1.4, 1.75, 2.0]
ETA_VALS = [2.0, 3.0, 3.7, 4.5]
EPS_VALS = [0.10, 0.25, 0.41]
G_ENV_VALS = [0.45, 0.55, 0.65, 0.75]
ALPHA = 2.5

QUARK_GEOM = {"Q": (0, 1, 0), "U": (0, 3, 6), "D": (0, 3, 7)}
LEPTON_GEOM = {"L": (0, 1, 0), "E": (0, 3, 6)}
NEUTRINO_GEOM = {"L": (0, 1, 0), "N": (0, 3, 6)}

CROSS_SECTOR_THRESHOLD = 0.25  # |r| must exceed in >= 2 sector types to count


def svd_hierarchy(Y):
    _, S, _ = np.linalg.svd(Y, full_matrices=False)
    S = np.clip(S, 1e-15, None)
    return float(S[0] / S[2]), float(S[0] / S[1])


def pearson_safe(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return np.nan, np.nan
    if np.std(x[mask]) < 1e-15 or np.std(y[mask]) < 1e-15:
        return np.nan, np.nan
    return stats.pearsonr(x[mask], y[mask])


def collect_quark_samples():
    rows = []
    Q, U, D = QUARK_GEOM["Q"], QUARK_GEOM["U"], QUARK_GEOM["D"]
    for sigma, k, eta, eps_u, eps_d in itertools.product(
        SIGMA_VALS, K_VALS, ETA_VALS, EPS_VALS, EPS_VALS
    ):
        Yu, Yd = compute_quark_yukawas(Q, U, D, sigma, k, ALPHA, eta, eps_u, eps_d)
        obs = compute_quark_observables(Yu, Yd)
        for label, Y in [("quark_up", Yu), ("quark_down", Yd)]:
            info = compute_yukawa_information(Y)
            h01, h12 = svd_hierarchy(Y)
            mix_off = obs["Vus"] + obs["Vcb"] + obs["Vub"]
            rows.append(
                dict(
                    sector_type="quark",
                    label=label,
                    entropy=info["entropy"],
                    effective_rank=info["effective_rank"],
                    off_diagonal_entropy=info["off_diagonal_entropy"],
                    mix_off=mix_off,
                    hierarchy=h01,
                    hierarchy_01=h12,
                )
            )
    return rows


def collect_lepton_samples():
    rows = []
    L, E = LEPTON_GEOM["L"], LEPTON_GEOM["E"]
    for sigma, k, eta, eps in itertools.product(SIGMA_VALS, K_VALS, ETA_VALS, EPS_VALS):
        Ye = compute_yukawa_matrix(L, E, sigma, k, ALPHA, eta, eps)
        info = compute_yukawa_information(Ye)
        h01, h12 = svd_hierarchy(Ye)
        # Charged leptons: no CKM; use off-diag Yukawa magnitude as mixing proxy
        off_yuk = float(np.sum(np.abs(Ye - np.diag(np.diag(Ye)))))
        rows.append(
            dict(
                sector_type="lepton",
                label="charged_lepton",
                entropy=info["entropy"],
                effective_rank=info["effective_rank"],
                off_diagonal_entropy=info["off_diagonal_entropy"],
                mix_off=off_yuk,
                hierarchy=h01,
                hierarchy_01=h12,
            )
        )
    return rows


def collect_neutrino_samples():
    rows = []
    L, N = NEUTRINO_GEOM["L"], NEUTRINO_GEOM["N"]
    for sigma, k, eta, eps_nu, eps_e, g_env in itertools.product(
        SIGMA_VALS, K_VALS, ETA_VALS, EPS_VALS, EPS_VALS, G_ENV_VALS
    ):
        Ynu = compute_yukawa_matrix(L, N, sigma * g_env, k, ALPHA, eta, eps_nu)
        Ye = compute_yukawa_matrix(L, N, sigma, k, ALPHA, eta, eps_e)
        obs = compute_neutrino_observables(Ynu, Ye)
        info = compute_yukawa_information(Ynu)
        h01, h12 = svd_hierarchy(Ynu)
        mix_off = obs["theta12"] + obs["theta23"] + obs["theta13"]
        rows.append(
            dict(
                sector_type="neutrino",
                label="neutrino",
                entropy=info["entropy"],
                effective_rank=info["effective_rank"],
                off_diagonal_entropy=info["off_diagonal_entropy"],
                mix_off=mix_off,
                hierarchy=h01,
                hierarchy_01=h12,
            )
        )
    return rows


def test_candidate(name, rows_by_type, x_key, y_key):
    """Return per sector-type correlation and cross-sector count."""
    results = {}
    for stype, rows in rows_by_type.items():
        xs = [r[x_key] for r in rows]
        ys = [r[y_key] for r in rows]
        r, p = pearson_safe(xs, ys)
        results[stype] = {"r": r, "p": p, "n": len(rows)}
    strong = sum(
        1 for v in results.values()
        if np.isfinite(v["r"]) and abs(v["r"]) >= CROSS_SECTOR_THRESHOLD
    )
    return name, results, strong


def main():
    print("=" * 70)
    print("YUKAWA INFORMATION INEQUALITY EXPLORER (Path A)")
    print("=" * 70)

    quark = collect_quark_samples()
    lepton = collect_lepton_samples()
    neutrino = collect_neutrino_samples()
    by_type = {"quark": quark, "lepton": lepton, "neutrino": neutrino}
    all_rows = quark + lepton + neutrino
    print(f"Samples: quark={len(quark)}, lepton={len(lepton)}, neutrino={len(neutrino)}")

    candidates = [
        ("S vs mixing (off-diag proxy)", "entropy", "mix_off"),
        ("effective_rank vs hierarchy S0/S2", "effective_rank", "hierarchy"),
        ("off_diag_entropy vs mixing", "off_diagonal_entropy", "mix_off"),
        ("S vs hierarchy S0/S2", "entropy", "hierarchy"),
        ("effective_rank vs mix_off", "effective_rank", "mix_off"),
    ]

    print("\n--- Candidate relations (Pearson r by sector type) ---")
    best = None
    best_score = -1
    report_lines = []

    for name, xk, yk in candidates:
        cname, res, strong = test_candidate(name, by_type, xk, yk)
        print(f"\n{cname}:")
        line = f"\n[{cname}]\n"
        for stype, v in res.items():
            r, p, n = v["r"], v["p"], v["n"]
            s = f"  {stype:10s}  r={r:+.4f}  p={p:.2e}  n={n}" if np.isfinite(r) else f"  {stype:10s}  r=nan  n={n}"
            print(s)
            line += s + "\n"
        line += f"  sectors_with_|r|>={CROSS_SECTOR_THRESHOLD}: {strong}/3\n"
        print(f"  sectors with |r|>={CROSS_SECTOR_THRESHOLD}: {strong}/3")
        report_lines.append(line)
        if strong > best_score:
            best_score = strong
            best = (cname, res, strong)

    # Pooled cross-sector (sector-type as categorical — exploratory only)
    print("\n--- Pooled (all sectors, exploratory) ---")
    pooled = {}
    for name, xk, yk in candidates:
        xs = [r[xk] for r in all_rows]
        ys = [r[yk] for r in all_rows]
        r, p = pearson_safe(xs, ys)
        pooled[name] = r
        if np.isfinite(r):
            print(f"  {name:40s}  r={r:+.4f}")

    print("\n" + "=" * 70)
    print("MECHANISM CANDIDATE SUMMARY")
    print("=" * 70)
    if best and best[2] >= 2:
        verdict = "weak_candidate"
        print(f"  Best: {best[0]} — holds in {best[2]}/3 sector types (exploratory)")
    else:
        verdict = "refuted"
        print("  No candidate relation holds in >= 2 sector types at |r| >= 0.25")
        print("  VERDICT: mechanism inequality candidates REFUTED at this sweep resolution")

    out = os.path.join(os.path.dirname(__file__), "results", "13_yukawa_information_inequality.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write("Yukawa information inequality explorer\n")
        f.write(f"verdict: {verdict}\n")
        f.write(f"cross_sector_threshold: {CROSS_SECTOR_THRESHOLD}\n\n")
        for line in report_lines:
            f.write(line)
        f.write("\npooled_correlations:\n")
        for k, v in pooled.items():
            f.write(f"  {k}: {v}\n")
        if best:
            f.write(f"\nbest_candidate: {best[0]}\n")
            f.write(f"best_strong_sectors: {best[2]}\n")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
