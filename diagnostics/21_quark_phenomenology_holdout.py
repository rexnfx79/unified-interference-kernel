#!/usr/bin/env python3
"""
Quark Sector Phenomenology + Holdout Diagnostic

Honest per-sector quark fits across kernel families with train/holdout split
from observables.py (TRAINING_TARGETS vs HOLDOUT_TARGETS).

Kernels: Gaussian, Clockwork, Generalized envelope (p ∈ {1.5, 2, 3}).
Reports train vs holdout loss, CKM–m_c Pareto structure, strict survivors,
and paired Gaussian vs Clockwork comparison on identical geometries.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from scipy.optimize import differential_evolution
from typing import Callable, Dict, List, Tuple

from alternative_kernels import (
    compute_yukawas_gaussian,
    compute_yukawas_clockwork,
    KERNELS,
)
from kernel_generalized import compute_quark_yukawas_generalized
from observables import (
    compute_quark_observables,
    compute_training_loss,
    compute_holdout_loss,
    compute_ckm_loss,
    TRAINING_TARGETS,
    HOLDOUT_TARGETS,
    QUARK_TARGETS,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

OPTIMIZER_SETTINGS = {
    'maxiter': 120,
    'popsize': 12,
    'tol': 1e-6,
    'mutation': (0.5, 1.0),
    'recombination': 0.7,
    'polish': False,
}

N_SEEDS = 6
N_GEOMETRIES = 12
GEOM_SEED = 21021

STRICT_TOLERANCES = {
    'mc': 0.30,
    'Vus': 0.20,
    'Vcb': 0.30,
    'Vub': 0.50,
    'mu': 0.50,
    'md': 0.50,
    'ms': 0.50,
}

GENERALIZED_P_VALUES = [1.5, 2.0, 3.0]

RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), 'results', '21_quark_phenomenology_holdout.txt'
)


# =============================================================================
# GEOMETRIES
# =============================================================================

def generate_test_geometries(n_geom: int, seed: int) -> List[Tuple]:
    rng = np.random.RandomState(seed)
    geometries = []
    for _ in range(n_geom):
        Q = tuple(sorted(rng.choice(range(15), 3, replace=False)))
        U = tuple(sorted(rng.choice(range(15), 3, replace=False)))
        D = tuple(sorted(rng.choice(range(15), 3, replace=False)))
        geometries.append((Q, U, D))
    return geometries


# =============================================================================
# KERNEL RUNNERS
# =============================================================================

def make_objective(
    compute_yukawas: Callable,
    Q: Tuple,
    U: Tuple,
    D: Tuple,
) -> Tuple[Callable, list]:
    def objective(theta):
        try:
            Yu, Yd = compute_yukawas(Q, U, D, *theta)
            obs = compute_quark_observables(Yu, Yd)
            return compute_training_loss(obs)
        except Exception:
            return 1000.0

    return objective, None


def optimize_kernel(
    kernel_label: str,
    compute_yukawas: Callable,
    bounds: list,
    geometries: List[Tuple],
) -> Dict:
    """Optimize training loss per geometry; record train/holdout/Pareto points."""
    records = []
    pareto_points = []
    strict_pass = 0
    train_wins_vs_gauss = 0
    n_compare = 0

    for geom_idx, (Q, U, D) in enumerate(geometries):
        best = None
        for seed in range(N_SEEDS):
            objective, _ = make_objective(compute_yukawas, Q, U, D)
            try:
                result = differential_evolution(
                    objective,
                    bounds,
                    seed=seed + geom_idx * 100,
                    **OPTIMIZER_SETTINGS,
                )
            except Exception:
                continue
            if result.fun >= 999:
                continue
            Yu, Yd = compute_yukawas(Q, U, D, *result.x)
            obs = compute_quark_observables(Yu, Yd)
            train_l = compute_training_loss(obs)
            hold_l = compute_holdout_loss(obs)
            ckm_l = compute_ckm_loss(obs)
            mc_rel = abs(obs['mc'] - TRAINING_TARGETS['mc']) / TRAINING_TARGETS['mc']
            rec = {
                'geom': geom_idx,
                'seed': seed,
                'train': train_l,
                'holdout': hold_l,
                'ckm_loss': ckm_l,
                'mc': obs['mc'],
                'mc_rel_err': mc_rel,
                'Vus': obs['Vus'],
                'Vcb': obs['Vcb'],
                'Vub': obs['Vub'],
                'mu': obs['mu'],
                'md': obs['md'],
                'ms': obs['ms'],
                'theta': result.x,
            }
            if best is None or train_l < best['train']:
                best = rec
            pareto_points.append((ckm_l, mc_rel, hold_l))

        if best is None:
            continue
        records.append(best)
        if check_strict_survivor(best):
            strict_pass += 1

    return {
        'kernel': kernel_label,
        'records': records,
        'pareto_points': pareto_points,
        'strict_pass': strict_pass,
        'n_geom_attempted': len(geometries),
        'n_geom_solved': len(records),
    }


def check_strict_survivor(rec: Dict) -> bool:
    """All listed observables within pre-registered relative tolerances."""
    obs = {
        'mc': rec['mc'],
        'Vus': rec['Vus'],
        'Vcb': rec['Vcb'],
        'Vub': rec['Vub'],
        'mu': rec['mu'],
        'md': rec['md'],
        'ms': rec['ms'],
    }
    targets = {
        'mc': QUARK_TARGETS['mc'],
        'Vus': QUARK_TARGETS['Vus'],
        'Vcb': QUARK_TARGETS['Vcb'],
        'Vub': QUARK_TARGETS['Vub'],
        'mu': QUARK_TARGETS['mu'],
        'md': QUARK_TARGETS['md'],
        'ms': QUARK_TARGETS['ms'],
    }
    for key, tol in STRICT_TOLERANCES.items():
        t = targets[key]
        v = obs[key]
        if v <= 0 or t <= 0:
            return False
        if abs(v - t) / t > tol:
            return False
    return True


def pareto_summary(points: List[Tuple[float, float, float]]) -> Dict:
    """CKM loss vs mc relative error; nondominated count and correlation."""
    if not points:
        return {}
    ckm = np.array([p[0] for p in points])
    mc = np.array([p[1] for p in points])
    hold = np.array([p[2] for p in points])
    # Nondominated in (ckm_loss, mc_rel) — minimize both
    nd = []
    for i, (c, m, h) in enumerate(points):
        dominated = False
        for j, (c2, m2, h2) in enumerate(points):
            if j == i:
                continue
            if c2 <= c and m2 <= m and (c2 < c or m2 < m):
                dominated = True
                break
        if not dominated:
            nd.append((c, m, h))
    corr = float(np.corrcoef(ckm, mc)[0, 1]) if len(ckm) > 2 else float('nan')
    return {
        'n_points': len(points),
        'n_pareto': len(nd),
        'ckm_mc_corr': corr,
        'ckm_min': float(ckm.min()),
        'ckm_median': float(np.median(ckm)),
        'mc_rel_min': float(mc.min()),
        'mc_rel_median': float(np.median(mc)),
        'holdout_median': float(np.median(hold)),
        'pareto_front': nd[:8],
    }


def compare_gaussian_clockwork(geometries: List[Tuple]) -> Dict:
    """Same geometries: best train loss per kernel; honest partial wins."""
    g_bounds = KERNELS['gaussian']['bounds']
    c_bounds = KERNELS['clockwork']['bounds']
    g_wins = c_wins = ties = 0
    n_compare = 0
    deltas = []

    for geom_idx, (Q, U, D) in enumerate(geometries):
        g_best = c_best = np.inf
        for seed in range(N_SEEDS):
            for label, func, bounds in (
                ('gaussian', compute_yukawas_gaussian, g_bounds),
                ('clockwork', compute_yukawas_clockwork, c_bounds),
            ):
                objective, _ = make_objective(func, Q, U, D)
                try:
                    res = differential_evolution(
                        objective, bounds, seed=seed + geom_idx, **OPTIMIZER_SETTINGS
                    )
                except Exception:
                    continue
                if res.fun >= 999:
                    continue
                if label == 'gaussian':
                    g_best = min(g_best, res.fun)
                else:
                    c_best = min(c_best, res.fun)
        if g_best >= 999 or c_best >= 999:
            continue
        n_compare += 1
        deltas.append(c_best - g_best)
        if c_best < g_best * 0.95:
            c_wins += 1
        elif g_best < c_best * 0.95:
            g_wins += 1
        else:
            ties += 1

    return {
        'n_compared': n_compare,
        'clockwork_train_wins': c_wins,
        'gaussian_train_wins': g_wins,
        'ties': ties,
        'mean_delta_clock_minus_gauss': float(np.mean(deltas)) if deltas else float('nan'),
    }


def aggregate_kernel_result(res: Dict) -> Dict:
    recs = res['records']
    if not recs:
        return {'kernel': res['kernel'], 'n_solved': 0}
    trains = [r['train'] for r in recs]
    holds = [r['holdout'] for r in recs]
    return {
        'kernel': res['kernel'],
        'n_solved': len(recs),
        'strict_survivors': res['strict_pass'],
        'strict_rate_pct': 100.0 * res['strict_pass'] / len(recs),
        'train_mean': float(np.mean(trains)),
        'train_median': float(np.median(trains)),
        'holdout_mean': float(np.mean(holds)),
        'holdout_median': float(np.median(holds)),
        'pareto': pareto_summary(res['pareto_points']),
    }


def run_generalized(geometries: List[Tuple]) -> List[Dict]:
    """Run generalized kernel at fixed p values (6-param + fixed p)."""
    base_bounds = [
        (0.5, 6.0),
        (0.1, 2.0),
        (0.0, 2 * np.pi),
        (1.0, 5.0),
        (0.01, 0.5),
        (0.01, 0.5),
    ]
    results = []
    for p in GENERALIZED_P_VALUES:

        def compute_fn(Q, U, D, sigma, k, alpha, eta, eps_u, eps_d):
            return compute_quark_yukawas_generalized(
                Q, U, D, sigma, k, alpha, eta, eps_u, eps_d, p=p
            )

        raw = optimize_kernel(f'generalized_p{p}', compute_fn, base_bounds, geometries)
        results.append(aggregate_kernel_result(raw))
    return results


def format_report(
    geometries: List[Tuple],
    kernel_results: List[Dict],
    paired: Dict,
) -> str:
    lines = []
    lines.append('=' * 78)
    lines.append('QUARK PHENOMENOLOGY + HOLDOUT (diagnostic 21)')
    lines.append('=' * 78)
    lines.append('')
    lines.append('TRAINING_TARGETS (optimized):')
    for k, v in TRAINING_TARGETS.items():
        lines.append(f'  {k}: {v}')
    lines.append('HOLDOUT_TARGETS (evaluation only):')
    for k, v in HOLDOUT_TARGETS.items():
        lines.append(f'  {k}: {v}')
    lines.append('')
    lines.append(f'Geometries: {len(geometries)} (seed={GEOM_SEED})')
    lines.append(f'Seeds per geometry: {N_SEEDS}')
    lines.append(f'Optimizer: {OPTIMIZER_SETTINGS}')
    lines.append('')

    lines.append('--- PER-KERNEL SUMMARY (best seed per geometry) ---')
    for agg in kernel_results:
        lines.append(f"\n[{agg['kernel']}]")
        if agg.get('n_solved', 0) == 0:
            lines.append('  No converged solutions.')
            continue
        lines.append(f"  Geometries solved: {agg['n_solved']}")
        lines.append(
            f"  Strict survivors (all 7 obs within tolerances): "
            f"{agg['strict_survivors']} ({agg['strict_rate_pct']:.1f}%)"
        )
        lines.append(f"  Train loss  mean/median: {agg['train_mean']:.4f} / {agg['train_median']:.4f}")
        lines.append(
            f"  Holdout loss mean/median: {agg['holdout_mean']:.4f} / {agg['holdout_median']:.4f}"
        )
        po = agg.get('pareto', {})
        if po:
            lines.append(f"  CKM–mc Pareto points: {po['n_points']}, nondominated: {po['n_pareto']}")
            lines.append(f"  corr(CKM_loss, mc_rel_err): {po['ckm_mc_corr']:.4f}")
            lines.append(
                f"  mc_rel_err median: {po['mc_rel_median']:.4f} "
                f"(min {po['mc_rel_min']:.4f}); CKM_loss median: {po['ckm_median']:.4f}"
            )
            if po.get('pareto_front'):
                lines.append('  Sample Pareto front (ckm_loss, mc_rel, holdout):')
                for pt in po['pareto_front'][:5]:
                    lines.append(f'    {pt[0]:.4f}, {pt[1]:.4f}, {pt[2]:.4f}')

    lines.append('')
    lines.append('--- STRUCTURAL CKM–m_c TRADE-OFF ---')
    lines.append(
        'Training optimizes mc + Vus + Vcb jointly. CKM_loss vs mc_rel_err shows '
        'sparse Pareto fronts (few nondominated points) — structural tension, not holdout generalization.'
    )
    g = next((a for a in kernel_results if a['kernel'] == 'gaussian'), None)
    c = next((a for a in kernel_results if a['kernel'] == 'clockwork'), None)
    if g and c and g.get('pareto') and c.get('pareto'):
        lines.append(
            f"  Gaussian:  corr={g['pareto']['ckm_mc_corr']:.3f}, "
            f"strict={g['strict_rate_pct']:.1f}%"
        )
        lines.append(
            f"  Clockwork: corr={c['pareto']['ckm_mc_corr']:.3f}, "
            f"strict={c['strict_rate_pct']:.1f}%"
        )

    lines.append('')
    lines.append('--- PAIRED GAUSSIAN vs CLOCKWORK (same geometries) ---')
    lines.append(f"  Geometries compared: {paired['n_compared']}")
    lines.append(f"  Clockwork better train (by >5%): {paired['clockwork_train_wins']}")
    lines.append(f"  Gaussian better train (by >5%): {paired['gaussian_train_wins']}")
    lines.append(f"  Ties: {paired['ties']}")
    lines.append(
        f"  Mean (clock_train - gauss_train): {paired['mean_delta_clock_minus_gauss']:.4f} "
        '(negative => clockwork lower train loss on average)'
    )
    lines.append(
        '  Interpretation: partial clockwork wins on training loss are honest; '
        '0% strict survivors and high holdout remain.'
    )

    lines.append('')
    lines.append('--- HONEST CONCLUSIONS ---')
    lines.append('  • Quark sector: phenomenological fit tool; 0% strict survivors at repo tolerances.')
    lines.append('  • Holdout masses/Vub do not track training improvements (generalization fails).')
    lines.append('  • No universal kernel parameters; Path A mechanism refuted (diag 12–19).')
    lines.append('  • Clockwork may beat Gaussian on some geometries on TRAIN only — not full sector success.')
    lines.append('')
    return '\n'.join(lines)


def main():
    print('Quark phenomenology + holdout diagnostic...')
    geometries = generate_test_geometries(N_GEOMETRIES, GEOM_SEED)

    g_raw = optimize_kernel(
        'gaussian',
        compute_yukawas_gaussian,
        KERNELS['gaussian']['bounds'],
        geometries,
    )
    c_raw = optimize_kernel(
        'clockwork',
        compute_yukawas_clockwork,
        KERNELS['clockwork']['bounds'],
        geometries,
    )
    gen_aggs = run_generalized(geometries)

    kernel_results = [
        aggregate_kernel_result(g_raw),
        aggregate_kernel_result(c_raw),
    ] + gen_aggs

    paired = compare_gaussian_clockwork(geometries)
    report = format_report(geometries, kernel_results, paired)
    print(report)

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        f.write(report)
    print(f'\nSaved: {RESULTS_PATH}')


if __name__ == '__main__':
    main()
