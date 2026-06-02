"""
Pre-registered Fisher transfer test helpers (quark → lepton).

Fit quarks, freeze universal parameters, transfer to leptons, and compare
experimental Fisher geometry to actual parameter deviations required for a
good lepton fit.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Sequence, Tuple

from scipy.optimize import differential_evolution

from experimental_fisher import (
    LEPTON_PARAM_NAMES,
    QUARK_PARAM_NAMES,
    UNIVERSAL_PARAM_NAMES,
    align_fisher_subspaces,
    compute_sector_experimental_fisher,
    cramér_rao_bounds,
    fisher_scalar_summaries,
    lepton_observables_from_ye,
)
from kernel import compute_quark_yukawas, compute_yukawa_matrix
from observables import compute_ckm_loss, compute_mass_loss, compute_quark_observables

LEPTON_TARGETS = {"m_e": 0.000511, "m_mu": 0.1057, "m_tau": 1.777}

DEFAULT_QUARK_GEOM = {"name": "standard", "Q": (0, 1, 0), "U": (0, 3, 6), "D": (0, 3, 7)}
DEFAULT_LEPTON_GEOM = {"name": "standard", "L": (0, 1, 0), "E": (0, 3, 6)}


def compute_lepton_loss_from_ye(Ye: np.ndarray) -> float:
    obs = lepton_observables_from_ye(Ye)
    loss = 0.0
    for key in ("m_e", "m_mu", "m_tau"):
        target = LEPTON_TARGETS[key]
        value = obs[key]
        if value > 0 and target > 0:
            loss += float(np.log(value / target) ** 2)
        else:
            loss += 100.0
    return float(loss)


def _quark_objective(Q, U, D, theta: np.ndarray) -> float:
    sigma, k, alpha, eta, eps_u, eps_d = theta
    Yu, Yd = compute_quark_yukawas(Q, U, D, sigma, k, alpha, eta, eps_u, eps_d)
    obs = compute_quark_observables(Yu, Yd)
    L_ckm = compute_ckm_loss(obs)
    L_mass = compute_mass_loss(obs)
    L_md = 2.0 * (np.log(0.002 / obs["md"])) ** 2 if obs["md"] < 0.002 else 0.0
    L_mu = 0.5 * (np.log(0.0005 / obs["mu"])) ** 2 if obs["mu"] < 0.0005 else 0.0
    return float(L_mass + 5.0 * L_ckm + L_md + L_mu)


def fit_quarks(
    geom: Dict = None,
    n_seeds: int = 5,
    maxiter: int = 200,
) -> Dict:
    """Fit quark sector; return parameters and loss."""
    geom = geom or DEFAULT_QUARK_GEOM
    bounds = [
        (0.5, 6.0),
        (0.1, 2.0),
        (0.0, 2 * np.pi),
        (1.0, 5.0),
        (0.01, 0.5),
        (0.01, 0.5),
    ]
    Q, U, D = geom["Q"], geom["U"], geom["D"]

    def objective(theta):
        return _quark_objective(Q, U, D, theta)

    best_loss = np.inf
    best_x = None
    for seed in range(n_seeds):
        result = differential_evolution(
            objective, bounds, maxiter=maxiter, seed=seed, polish=True, atol=1e-8, tol=1e-8
        )
        if result.fun < best_loss:
            best_loss = float(result.fun)
            best_x = result.x

    theta = np.asarray(best_x, dtype=float)
    Yu, Yd = compute_quark_yukawas(Q, U, D, *theta)
    obs = compute_quark_observables(Yu, Yd)
    return {
        "theta": theta,
        "param_names": list(QUARK_PARAM_NAMES),
        "loss_total": best_loss,
        "loss_ckm": compute_ckm_loss(obs),
        "loss_mass": compute_mass_loss(obs),
        **obs,
    }


def transfer_leptons_frozen(
    quark_theta: np.ndarray,
    geom: Dict = None,
    n_seeds: int = 5,
    maxiter: int = 200,
) -> Dict:
    """Freeze σ, k, α, η from quark fit; optimize eps only."""
    geom = geom or DEFAULT_LEPTON_GEOM
    sigma, k, alpha, eta = quark_theta[:4]
    bounds = [(0.01, 0.5)]
    L, E = geom["L"], geom["E"]

    def objective(theta):
        eps = theta[0]
        Ye = compute_yukawa_matrix(L, E, sigma, k, alpha, eta, eps)
        return compute_lepton_loss_from_ye(Ye)

    best_loss = np.inf
    best_eps = None
    for seed in range(n_seeds):
        result = differential_evolution(objective, bounds, maxiter=maxiter, seed=seed, polish=True)
        if result.fun < best_loss:
            best_loss = float(result.fun)
            best_eps = float(result.x[0])

    Ye = compute_yukawa_matrix(L, E, sigma, k, alpha, eta, best_eps)
    obs = lepton_observables_from_ye(Ye)
    lepton_theta = np.array([sigma, k, alpha, eta, best_eps], dtype=float)
    return {
        "theta": lepton_theta,
        "loss": best_loss,
        "eps_e": best_eps,
        **obs,
    }


def fit_leptons_free(
    geom: Dict = None,
    n_seeds: int = 5,
    maxiter: int = 200,
) -> Dict:
    """Independent lepton fit (all parameters free)."""
    geom = geom or DEFAULT_LEPTON_GEOM
    bounds = [
        (0.5, 6.0),
        (0.1, 2.0),
        (0.0, 2 * np.pi),
        (1.0, 5.0),
        (0.01, 0.5),
    ]
    L, E = geom["L"], geom["E"]

    def objective(theta):
        Ye = compute_yukawa_matrix(L, E, *theta)
        return compute_lepton_loss_from_ye(Ye)

    best_loss = np.inf
    best_x = None
    for seed in range(n_seeds):
        result = differential_evolution(objective, bounds, maxiter=maxiter, seed=seed, polish=True)
        if result.fun < best_loss:
            best_loss = float(result.fun)
            best_x = result.x

    theta = np.asarray(best_x, dtype=float)
    Ye = compute_yukawa_matrix(L, E, *theta)
    obs = lepton_observables_from_ye(Ye)
    return {"theta": theta, "loss": best_loss, **obs}


def _fisher_submatrix(F: np.ndarray, param_names: Sequence[str], subset: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    idx = [list(param_names).index(n) for n in subset]
    return F[np.ix_(idx, idx)], list(subset)


def deltas_within_cr_bounds(
    deltas: Dict[str, float],
    cr_bounds: Dict[str, float],
    z: float = 2.0,
) -> Tuple[bool, Dict[str, float]]:
    """Check |Δθ_i| ≤ z √CR_i for each shared parameter."""
    ratios = {}
    all_within = True
    for name, delta in deltas.items():
        cr = cr_bounds.get(name, float("inf"))
        if not np.isfinite(cr) or cr <= 0:
            bound = float("inf")
            within = False
        else:
            bound = z * np.sqrt(cr)
            within = abs(delta) <= bound
        ratios[name] = abs(delta) / bound if np.isfinite(bound) and bound > 0 else float("inf")
        all_within = all_within and within
    return all_within, ratios


def run_fisher_transfer_analysis(
    quark_geom: Dict = None,
    lepton_geom: Dict = None,
    n_seeds: int = 5,
    maxiter: int = 200,
    cr_z: float = 2.0,
) -> Dict:
    """
    Full Fisher transfer pipeline:
    fit quarks → Fisher at quark min → frozen lepton transfer → free lepton fit
    → compare Fisher alignment and CR-predicted vs actual deviations.
    """
    quark_geom = quark_geom or DEFAULT_QUARK_GEOM
    lepton_geom = lepton_geom or DEFAULT_LEPTON_GEOM

    quark_fit = fit_quarks(quark_geom, n_seeds=n_seeds, maxiter=maxiter)
    quark_theta = quark_fit["theta"]

    fisher_quark = compute_sector_experimental_fisher("quark", quark_geom, quark_theta)
    frozen = transfer_leptons_frozen(quark_theta, lepton_geom, n_seeds=n_seeds, maxiter=maxiter)
    free = fit_leptons_free(lepton_geom, n_seeds=n_seeds, maxiter=maxiter)

    fisher_lepton_frozen = compute_sector_experimental_fisher(
        "lepton", lepton_geom, frozen["theta"]
    )
    fisher_lepton_free = compute_sector_experimental_fisher(
        "lepton", lepton_geom, free["theta"]
    )

    cr_quark = cramér_rao_bounds(fisher_quark["fisher"], QUARK_PARAM_NAMES)
    cr_shared = {n: cr_quark[n] for n in UNIVERSAL_PARAM_NAMES}

    deltas = {
        name: float(free["theta"][i] - quark_theta[i])
        for i, name in enumerate(UNIVERSAL_PARAM_NAMES)
    }

    within_cr, cr_ratios = deltas_within_cr_bounds(deltas, cr_shared, z=cr_z)

    align_at_transfer = align_fisher_subspaces(
        fisher_quark["principal_eigenvector"],
        fisher_lepton_frozen["principal_eigenvector"],
        fisher_quark["param_names"],
        fisher_lepton_frozen["param_names"],
    )
    align_at_free_opt = align_fisher_subspaces(
        fisher_quark["principal_eigenvector"],
        fisher_lepton_free["principal_eigenvector"],
        fisher_quark["param_names"],
        fisher_lepton_free["param_names"],
    )

    Fq_sub, _ = _fisher_submatrix(
        fisher_quark["fisher"], QUARK_PARAM_NAMES, UNIVERSAL_PARAM_NAMES
    )
    Fl_sub, _ = _fisher_submatrix(
        fisher_lepton_free["fisher"], LEPTON_PARAM_NAMES, UNIVERSAL_PARAM_NAMES
    )

    return {
        "quark_fit": quark_fit,
        "quark_theta": quark_theta,
        "fisher_quark": fisher_quark,
        "frozen": frozen,
        "free": free,
        "fisher_lepton_frozen": fisher_lepton_frozen,
        "fisher_lepton_free": fisher_lepton_free,
        "cr_shared": cr_shared,
        "universal_deltas": deltas,
        "deltas_within_cr": within_cr,
        "cr_ratios": cr_ratios,
        "alignment_at_transfer": align_at_transfer,
        "alignment_at_free_optimum": align_at_free_opt,
        "fisher_quark_shared_summaries": fisher_scalar_summaries(Fq_sub),
        "fisher_lepton_shared_summaries": fisher_scalar_summaries(Fl_sub),
    }


def evaluate_fisher_transfer_verdict(
    analysis: Dict,
    alignment_threshold: float = 0.50,
    frozen_loss_bad_threshold: float = 797.0,
) -> Dict:
    """
    Pre-registered falsifiers (diagnostic 19):
    A) Fisher-predicted θ ≈ frozen quark θ within CR BUT frozen loss bad → refuted
    B) Fisher alignment at transfer < threshold AND frozen loss bad → refuted
    """
    frozen_loss = analysis["frozen"]["loss"]
    loss_bad = frozen_loss >= frozen_loss_bad_threshold
    align = analysis["alignment_at_transfer"]
    align_low = np.isfinite(align) and align < alignment_threshold

    falsifier_a = analysis["deltas_within_cr"] and loss_bad
    falsifier_b = align_low and loss_bad
    refuted = falsifier_a or falsifier_b

    if refuted:
        if falsifier_a and falsifier_b:
            reason = "both CR-within-delta and low-alignment falsifiers"
        elif falsifier_a:
            reason = "free-fit deltas within quark CR bounds but frozen transfer loss bad"
        else:
            reason = "Fisher alignment below threshold at transfer point with bad frozen loss"
        verdict = "refuted"
    else:
        if not loss_bad:
            reason = "frozen transfer loss below bad threshold — mechanism not falsified on loss axis"
            verdict = "not_refuted_loss_ok"
        else:
            reason = "bad frozen loss but free fit requires deviations beyond CR and/or alignment ok"
            verdict = "inconclusive"

    return {
        "verdict": verdict,
        "refuted": refuted,
        "reason": reason,
        "frozen_loss": frozen_loss,
        "loss_bad": loss_bad,
        "falsifier_a": falsifier_a,
        "falsifier_b": falsifier_b,
        "alignment_at_transfer": align,
        "deltas_within_cr": analysis["deltas_within_cr"],
    }
