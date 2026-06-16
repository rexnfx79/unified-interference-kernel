#!/usr/bin/env python3
"""
Diagnostic 49 - Neutrino CP-weight Pareto scan.

Purpose:
  Pre-register a small, fixed CP-weight grid after N7 showed a near miss:
  w_CP=1 fixes signed J_PMNS but drops joint strict to 20/100; w_CP=0.25
  improves strict to 21/100 but still misses the 22/100 no-degradation bar.

Fixed grid:
  w_CP in {0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.50, 1.00}

Pass bar (full N=100):
  - at least one weight has joint strict >= 22/100 attempted
  - median signed-J relative error < 0.50
  - CP sign-match rate >= 0.60
  - weight selected by this scan only; no adaptive weight additions

Usage:
  python diagnostics/49_neutrino_cp_weight_pareto_scan.py --smoke
  python diagnostics/49_neutrino_cp_weight_pareto_scan.py
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))


WEIGHTS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.50, 1.00]
N_GEOM = 100
SMOKE_N_GEOM = 5
DIAG28_STRICT_BAR = 22
CP_REL_ERR_BAR = 0.50
SIGN_MATCH_BAR = 0.60

RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "results", "49_neutrino_cp_weight_pareto_scan.txt"
)


def load_diag48():
    path = os.path.join(os.path.dirname(__file__), "48_neutrino_cp_weighted_objective.py")
    spec = importlib.util.spec_from_file_location("diag48_cp_weighted", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load diag48 module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_scan(n_geom: int) -> List[Dict[str, float]]:
    diag48 = load_diag48()
    rows: List[Dict[str, float]] = []
    for weight in WEIGHTS:
        print("=" * 78, flush=True)
        print(f"[diag49] weight {weight:g}; n_geom={n_geom}", flush=True)
        result = diag48.run_pool(n_geom, weight)
        summary = diag48.summarize(result)
        rows.append(summary)
    return rows


def fmt_bool(value: float) -> str:
    return "yes" if bool(value) else "no"


def format_report(rows: List[Dict[str, float]], smoke: bool) -> str:
    lines = [
        "=" * 78,
        "NEUTRINO CP-WEIGHT PARETO SCAN (diagnostic 49)",
        "=" * 78,
        "",
        f"Mode: {'SMOKE' if smoke else 'FULL'}",
        "Fixed weights: " + ", ".join(f"{w:g}" for w in WEIGHTS),
        f"Pass bars: strict >= {DIAG28_STRICT_BAR}/100, signed-J rel err < {CP_REL_ERR_BAR}, sign-match >= {SIGN_MATCH_BAR}",
        "",
        "| w_CP | solved | joint strict | PMNS strict | signed-J rel err | |J| rel err | sign match | mass med | PMNS med | pass? |",
        "|------|--------|--------------|-------------|------------------|-------------|------------|----------|----------|-------|",
    ]
    passing = []
    for row in rows:
        joint_strict = int(row.get("joint_strict", 0))
        requested = int(row.get("requested", 0))
        pass_weight = (
            (requested >= 100 and joint_strict >= DIAG28_STRICT_BAR)
            and row.get("median_signed_j_rel_err", 999.0) < CP_REL_ERR_BAR
            and row.get("sign_match_rate", 0.0) >= SIGN_MATCH_BAR
        )
        if pass_weight:
            passing.append(row)
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{row.get('cp_weight', 0.0):g}",
                    f"{int(row.get('solved', 0))}/{requested}",
                    f"{joint_strict}/{requested}",
                    f"{int(row.get('pmns_strict', 0))}/{int(row.get('solved', 0))}",
                    f"{row.get('median_signed_j_rel_err', float('nan')):.4f}",
                    f"{row.get('median_abs_j_rel_err', float('nan')):.4f}",
                    f"{row.get('sign_match_rate', float('nan')):.3f}",
                    f"{row.get('median_mass_loss', float('nan')):.6f}",
                    f"{row.get('median_pmns_loss', float('nan')):.6f}",
                    "yes" if pass_weight else "no",
                ]
            )
            + " |"
        )

    lines.extend(["", "--- VERDICT ---"])
    if smoke:
        lines.append("Smoke only: do not select a final weight from this run.")
        best = min(rows, key=lambda r: r.get("median_signed_j_rel_err", 999.0))
        lines.append(
            f"Best smoke CP row by signed-J: w={best.get('cp_weight', 0.0):g}, "
            f"rel err={best.get('median_signed_j_rel_err', float('nan')):.4f}, "
            f"joint strict={int(best.get('joint_strict', 0))}/{int(best.get('requested', 0))}."
        )
    elif passing:
        # Choose simplest/minimal CP pressure among passing weights.
        chosen = min(passing, key=lambda r: r.get("cp_weight", 999.0))
        lines.append(
            f"PASS: w={chosen.get('cp_weight', 0.0):g} is the lowest passing CP weight."
        )
    else:
        lines.append("FAIL: no fixed grid weight clears both strict and CP bars.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help=f"Run {SMOKE_N_GEOM} geometries")
    parser.add_argument("--n-geom", type=int, default=None, help="Override geometry count")
    args = parser.parse_args()

    n_geom = args.n_geom if args.n_geom is not None else (SMOKE_N_GEOM if args.smoke else N_GEOM)
    rows = run_scan(n_geom)
    report = format_report(rows, args.smoke)
    print(report)
    if not args.smoke:
        os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            f.write(report + "\n")
        print(f"Saved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
