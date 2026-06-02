#!/usr/bin/env python3
"""
Quark geometry follow-up (diagnostic 30) — hardened extension study.

Addresses adversarial review of diagnostic 29:
  A) Re-optimize exhaustive max_coord=5 grid (1000) with unified DE protocol
  B) Exhaust first extension shell (5000 geoms using coordinate value 5)
  C) Store theta* + winning seed; reproducibility check on reported bests
  D) Wilson 95% CI on strict survivor rate

Unified protocol matches diagnostic 29:
  DE maxiter=120, popsize=12, 4 seeds, joint loss L_mass + 5*L_CKM + penalties.

Outputs:
  data/quark_geometry_followup_baseline.csv
  data/quark_geometry_followup_shell5.csv
  diagnostics/results/30_quark_geometry_followup.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

# Reuse enumeration + loss from diagnostic 29
sys.path.insert(0, os.path.dirname(__file__))
from importlib import import_module

_g29 = import_module("29_quark_geometry_extension")  # noqa: E402

generate_geometries = _g29.generate_geometries
extension_geometries = _g29.extension_geometries
compute_joint_quark_loss = _g29.compute_joint_quark_loss
check_strict_survivor = _g29.check_strict_survivor
BOUNDS = _g29.BOUNDS
OPTIMIZER_SETTINGS = _g29.OPTIMIZER_SETTINGS
N_SEEDS = _g29.N_SEEDS
STRICT_TOLERANCES = _g29.STRICT_TOLERANCES
QUARK_TARGETS = _g29.QUARK_TARGETS

from kernel import compute_quark_yukawas
from observables import compute_quark_observables, compute_ckm_loss, compute_mass_loss
from scipy.optimize import differential_evolution

RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "results", "30_quark_geometry_followup.txt"
)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
BASELINE_CSV = os.path.join(DATA_DIR, "quark_geometry_followup_baseline.csv")
SHELL5_CSV = os.path.join(DATA_DIR, "quark_geometry_followup_shell5.csv")
BESTS_JSON = os.path.join(DATA_DIR, "quark_geometry_followup_bests.json")

CSV_FIELDS = [
    "Q1", "Q2", "U1", "U2", "U3", "D1", "D2", "D3",
    "sigma", "k", "alpha", "eta", "eps_u", "eps_d",
    "winning_seed", "joint", "ckm", "mass", "strict",
    "Vus", "Vcb", "Vub", "mu", "mc", "md", "ms",
]


def first_shell_geometries() -> List[Tuple]:
    """Geometries in max_coord=6 enumeration that use coordinate 5 (not in 0..4 grid)."""
    all6 = generate_geometries(6)
    base = { _g29.geometry_key(*g) for g in generate_geometries(5) }
    return [g for g in all6 if _g29.geometry_key(*g) not in base]


def optimize_geometry_full(Q, U, D) -> Optional[Dict]:
    best = None
    for seed in range(N_SEEDS):
        def objective(theta):
            try:
                Yu, Yd = compute_quark_yukawas(Q, U, D, *theta)
                obs = compute_quark_observables(Yu, Yd)
                if obs["mc"] < 0.01 or obs["mc"] > 500:
                    return 1000.0
                return compute_joint_quark_loss(obs)
            except Exception:
                return 1000.0

        try:
            result = differential_evolution(
                objective, BOUNDS, seed=seed, **OPTIMIZER_SETTINGS
            )
        except Exception:
            continue
        if abs(result.fun - 1000.0) < 1e-3:
            continue
        sigma, k, alpha, eta, eps_u, eps_d = result.x
        Yu, Yd = compute_quark_yukawas(Q, U, D, sigma, k, alpha, eta, eps_u, eps_d)
        obs = compute_quark_observables(Yu, Yd)
        rec = {
            "Q": Q, "U": U, "D": D,
            "sigma": float(sigma), "k": float(k), "alpha": float(alpha),
            "eta": float(eta), "eps_u": float(eps_u), "eps_d": float(eps_d),
            "winning_seed": seed,
            "joint": compute_joint_quark_loss(obs),
            "ckm": compute_ckm_loss(obs),
            "mass": compute_mass_loss(obs),
            "strict": check_strict_survivor(obs),
            **{k: obs[k] for k in STRICT_TOLERANCES},
        }
        if best is None or rec["joint"] < best["joint"]:
            best = rec
    return best


def rec_to_row(rec: Dict) -> Dict:
    Q, U, D = rec["Q"], rec["U"], rec["D"]
    return {
        "Q1": Q[0], "Q2": Q[1],
        "U1": U[0], "U2": U[1], "U3": U[2],
        "D1": D[0], "D2": D[1], "D3": D[2],
        "sigma": rec["sigma"], "k": rec["k"], "alpha": rec["alpha"],
        "eta": rec["eta"], "eps_u": rec["eps_u"], "eps_d": rec["eps_d"],
        "winning_seed": rec["winning_seed"],
        "joint": rec["joint"], "ckm": rec["ckm"], "mass": rec["mass"],
        "strict": int(rec["strict"]),
        "Vus": rec["Vus"], "Vcb": rec["Vcb"], "Vub": rec["Vub"],
        "mu": rec["mu"], "mc": rec["mc"], "md": rec["md"], "ms": rec["ms"],
    }


def run_batch(
    geometries: List[Tuple],
    out_csv: str,
    checkpoint_every: int = 50,
    resume: bool = True,
) -> List[Dict]:
    done_keys = set()
    rows: List[Dict] = []
    if resume and os.path.exists(out_csv):
        with open(out_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
                key = tuple(int(row[c]) for c in [
                    "Q1", "Q2", "U1", "U2", "U3", "D1", "D2", "D3"
                ])
                done_keys.add(key)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pending = []
    for g in geometries:
        Q, U, D = g
        key = (Q[0], Q[1], U[0], U[1], U[2], D[0], D[1], D[2])
        if key not in done_keys:
            pending.append(g)

    print(f"  {out_csv}: {len(rows)} done, {len(pending)} pending")
    t0 = time.time()
    for i, (Q, U, D) in enumerate(pending):
        rec = optimize_geometry_full(Q, U, D)
        if rec is not None:
            rows.append(rec_to_row(rec))
        elif (i + 1) % checkpoint_every == 0:
            pass  # still checkpoint below
        if (i + 1) % checkpoint_every == 0 or i == len(pending) - 1:
            with open(out_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"    checkpoint {len(rows)} rows, {i+1}/{len(pending)} "
                  f"({rate:.2f} geom/s)")

    return rows


def wilson_upper_bound(successes: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return float("nan")
    p = successes / n
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    return float((centre + margin) / denom)


def reproduce_record(label: str, rec: Dict) -> Dict:
    Q = (int(rec["Q1"]), int(rec["Q2"]), 0)
    U = (int(rec["U1"]), int(rec["U2"]), int(rec["U3"]))
    D = (int(rec["D1"]), int(rec["D2"]), int(rec["D3"]))
    theta = [
        float(rec["sigma"]), float(rec["k"]), float(rec["alpha"]),
        float(rec["eta"]), float(rec["eps_u"]), float(rec["eps_d"]),
    ]
    Yu, Yd = compute_quark_yukawas(Q, U, D, *theta)
    obs = compute_quark_observables(Yu, Yd)
    joint = compute_joint_quark_loss(obs)
    return {
        "label": label,
        "stored_joint": float(rec["joint"]),
        "repro_joint": joint,
        "abs_diff": abs(joint - float(rec["joint"])),
        "match": abs(joint - float(rec["joint"])) < 1e-4,
    }


def summarize_rows(rows: List[Dict], label: str) -> Dict:
    solved = [r for r in rows if r.get("joint")]
    if not solved:
        return {"label": label, "n": 0}
    joints = [float(r["joint"]) for r in solved]
    strict_n = sum(int(r.get("strict", 0)) for r in solved)
    n = len(solved)
    best = min(solved, key=lambda r: float(r["joint"]))
    return {
        "label": label,
        "n": n,
        "strict_n": strict_n,
        "strict_rate_pct": 100.0 * strict_n / n,
        "wilson_upper_95pct": wilson_upper_bound(strict_n, n),
        "joint_min": min(joints),
        "joint_median": float(np.median(joints)),
        "ckm_min": min(float(r["ckm"]) for r in solved),
        "best_row": best,
    }


def format_report(
    baseline_sum: Dict,
    shell_sum: Dict,
    legacy_csv_row: Optional[Dict],
    repro: List[Dict],
) -> str:
    lines = [
        "=" * 78,
        "QUARK GEOMETRY FOLLOW-UP (diagnostic 30)",
        "=" * 78,
        "",
        "Protocol: DE maxiter=120 popsize=12, 4 seeds, joint 7-obs loss (diag 29)",
        "Geometry convention: legacy (q1,q2,0) + ordered U,D — NOT diag 21 triples",
        "",
    ]

    def block(s: Dict):
        if s.get("n", 0) == 0:
            lines.append(f"  [{s['label']}] no data")
            return
        b = s["best_row"]
        lines.append(f"--- {s['label'].upper()} (n={s['n']}) ---")
        lines.append(
            f"  Strict survivors: {s['strict_n']} ({s['strict_rate_pct']:.2f}%)"
        )
        lines.append(
            f"  Wilson 95% upper bound on strict rate: {s['wilson_upper_95pct']:.4f}"
        )
        lines.append(
            f"  Joint min/median: {s['joint_min']:.4f} / {s['joint_median']:.4f}"
        )
        lines.append(f"  CKM loss min: {s['ckm_min']:.4f}")
        lines.append(
            f"  Best: Q=({b['Q1']},{b['Q2']},0) U=({b['U1']},{b['U2']},{b['U3']}) "
            f"D=({b['D1']},{b['D2']},{b['D3']})"
        )
        lines.append(
            f"    joint={float(b['joint']):.4f} mc={float(b['mc']):.3f} "
            f"seed={b['winning_seed']} strict={b['strict']}"
        )

    block(baseline_sum)
    lines.append("")
    block(shell_sum)

    if legacy_csv_row:
        lines.extend([
            "",
            "--- LEGACY CSV (scripts/01, 5 seeds x 100 iter) best loss_total ---",
            f"  joint={legacy_csv_row['joint']:.4f} (not comparable optimizer)",
        ])

    if baseline_sum.get("n") and shell_sum.get("n"):
        bj = baseline_sum["joint_min"]
        sj = shell_sum["joint_min"]
        lines.extend([
            "",
            "--- CROSS-GRID COMPARISON (unified protocol) ---",
            f"  Baseline 1k joint min: {bj:.4f}",
            f"  Shell-5 joint min:     {sj:.4f}",
            f"  Shell beats baseline:  {sj < bj}",
        ])

    lines.extend(["", "--- REPRODUCIBILITY (stored theta) ---"])
    for r in repro:
        lines.append(
            f"  {r['label']}: stored={r['stored_joint']:.6f} "
            f"repro={r['repro_joint']:.6f} diff={r['abs_diff']:.2e} "
            f"ok={r['match']}"
        )

    lines.extend([
        "",
        "--- VERDICT ---",
    ])
    total_strict = baseline_sum.get("strict_n", 0) + shell_sum.get("strict_n", 0)
    total_n = baseline_sum.get("n", 0) + shell_sum.get("n", 0)
    if total_strict == 0 and total_n > 0:
        ub = wilson_upper_bound(0, total_n)
        lines.append(
            f"  0/{total_n} strict survivors; Wilson 95%% rate upper bound ~{ub:.4f}."
        )
        if shell_sum.get("joint_min", 1e9) < baseline_sum.get("joint_min", 1e9):
            lines.append(
                "  First shell improves joint loss vs re-baselined 1k grid; "
                "still no strict simultaneous PDG match."
            )
        else:
            lines.append(
                "  First shell does not beat re-baselined 1k minimum joint loss."
            )
    elif total_strict > 0:
        lines.append("  Strict survivors found — geometry coverage was limiting.")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=["baseline", "shell5", "all", "report"],
        default="all",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="10 geoms per phase")
    args = parser.parse_args()

    resume = not args.no_resume
    if args.phase in ("baseline", "all"):
        geoms = generate_geometries(5)
        if args.smoke:
            geoms = geoms[:10]
        print(f"Phase A: baseline grid ({len(geoms)} geometries)")
        run_batch(geoms, BASELINE_CSV, resume=resume)

    if args.phase in ("shell5", "all"):
        geoms = first_shell_geometries()
        if args.smoke:
            geoms = geoms[:10]
        print(f"Phase B: first shell coord=5 ({len(geoms)} geometries)")
        run_batch(geoms, SHELL5_CSV, resume=resume)

    if args.phase in ("report", "all") or args.smoke:
        baseline_rows = []
        shell_rows = []
        if os.path.exists(BASELINE_CSV):
            with open(BASELINE_CSV) as f:
                baseline_rows = list(csv.DictReader(f))
        if os.path.exists(SHELL5_CSV):
            with open(SHELL5_CSV) as f:
                shell_rows = list(csv.DictReader(f))

        baseline_sum = summarize_rows(baseline_rows, "baseline_1k")
        shell_sum = summarize_rows(shell_rows, "shell5_5k")

        legacy = None
        legacy_path = os.path.join(DATA_DIR, "quark_results.csv")
        if os.path.exists(legacy_path):
            import pandas as pd
            df = pd.read_csv(legacy_path)
            row = df.loc[df["loss_total"].idxmin()]
            legacy = {"joint": float(row.loss_total)}

        repro = []
        for label, s in [("baseline_best", baseline_sum), ("shell5_best", shell_sum)]:
            if s.get("best_row"):
                repro.append(reproduce_record(label, s["best_row"]))

        bests = {}
        if baseline_sum.get("best_row"):
            bests["baseline_best"] = baseline_sum["best_row"]
        if shell_sum.get("best_row"):
            bests["shell5_best"] = shell_sum["best_row"]
        with open(BESTS_JSON, "w") as f:
            json.dump(bests, f, indent=2)

        report = format_report(baseline_sum, shell_sum, legacy, repro)
        print(report)
        os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
        with open(RESULTS_PATH, "w") as f:
            f.write(report)
        print(f"Saved: {RESULTS_PATH}")
        print(f"Saved: {BESTS_JSON}")


if __name__ == "__main__":
    main()
