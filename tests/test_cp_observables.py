"""
CP observables: Jarlskog, delta_CKM, delta_PMNS.
"""

import sys

sys.path.insert(0, "../src")

import numpy as np
from observables import (
    QUARK_CP_TARGETS,
    NEUTRINO_CP_TARGETS,
    jarlskog_invariant,
    cp_phase_delta_from_unitary,
    pmns_angles_from_unitary,
    compute_quark_observables,
    compute_neutrino_observables,
    fix_svd_phases,
    svd_reconstruction_error,
)
from kernel import compute_quark_yukawas


def _mixing_matrix(theta12, theta23, theta13, delta=0.0):
    c12, s12 = np.cos(theta12), np.sin(theta12)
    c23, s23 = np.cos(theta23), np.sin(theta23)
    c13, s13 = np.cos(theta13), np.sin(theta13)
    cd, sd = np.cos(delta), np.sin(delta)
    return np.array(
        [
            [c12 * c13, s12 * c13, s13 * np.exp(-1j * delta)],
            [
                -s12 * c23 - c12 * s23 * s13 * np.exp(1j * delta),
                c12 * c23 - s12 * s23 * s13 * np.exp(1j * delta),
                s23 * c13,
            ],
            [
                s12 * s23 - c12 * c23 * s13 * np.exp(1j * delta),
                -c12 * s23 - s12 * c23 * s13 * np.exp(1j * delta),
                c23 * c13,
            ],
        ],
        dtype=complex,
    )


def test_jarlskog_pdg_order_of_magnitude():
    """CKM-like unitary with PDG-scale angles yields |J| ~ 10^-5."""
    t12 = np.radians(13.04)
    t23 = np.radians(2.38)
    t13 = np.radians(0.201)
    delta = QUARK_CP_TARGETS["delta_CKM"]
    V = _mixing_matrix(t12, t23, t13, delta)
    J = jarlskog_invariant(V)
    assert 1e-6 < abs(J) < 1e-3
    assert abs(abs(J) - QUARK_CP_TARGETS["J"]) / QUARK_CP_TARGETS["J"] < 5.0


def test_cp_phase_roundtrip_ckm():
    delta = QUARK_CP_TARGETS["delta_CKM"]
    V = _mixing_matrix(0.2, 0.04, 0.003, delta)
    extracted = cp_phase_delta_from_unitary(V)
    assert abs(extracted - delta) < 0.05 or abs(extracted - delta + 2 * np.pi) < 0.05


def test_quark_observables_include_cp():
    Q, U, D = (0, 1, 0), (0, 3, 6), (0, 3, 7)
    Yu, Yd = compute_quark_yukawas(
        Q, U, D, sigma=1.5, k=0.5, alpha=0.0, eta=2.5, eps_u=0.15, eps_d=0.15
    )
    obs = compute_quark_observables(Yu, Yd)
    assert "J" in obs and "delta_CKM" in obs and "J_abs" in obs
    assert obs["J_abs"] >= 0
    Uu, Su, Vuh = np.linalg.svd(Yu, full_matrices=False)
    assert svd_reconstruction_error(Uu, Su, Vuh) < 1e-10


def test_pmns_cp_fields():
    targets = dict(theta12=0.59, theta23=0.785, theta13=0.149)
    delta = NEUTRINO_CP_TARGETS["delta_PMNS"]
    U = _mixing_matrix(targets["theta12"], targets["theta23"], targets["theta13"], delta)
    S = np.diag([1.0, 0.3, 0.01])
    Ynu = U @ S
    Ye = np.eye(3) * np.diag([1.0, 0.1, 1e-6])
    obs = compute_neutrino_observables(Ynu, Ye)
    assert "delta_PMNS" in obs and "J_PMNS_abs" in obs
    t12, t23, t13 = pmns_angles_from_unitary(U)
    assert abs(t12 - targets["theta12"]) < 0.02
