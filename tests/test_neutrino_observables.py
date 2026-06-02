"""
Tests for neutrino observables extraction (PMNS from SVD).
"""

import sys
sys.path.insert(0, '../src')

import numpy as np
from observables import (
    compute_neutrino_observables,
    compute_pmns_loss,
    pmns_angles_from_unitary,
    NEUTRINO_TARGETS,
)


def _pmns_matrix(theta12, theta23, theta13, delta=0.0):
    """Build PMNS unitary from mixing angles (PDG convention)."""
    c12, s12 = np.cos(theta12), np.sin(theta12)
    c23, s23 = np.cos(theta23), np.sin(theta23)
    c13, s13 = np.cos(theta13), np.sin(theta13)
    cd, sd = np.cos(delta), np.sin(delta)
    return np.array([
        [c12 * c13, s12 * c13, s13 * np.exp(-1j * delta)],
        [-s12 * c23 - c12 * s23 * s13 * np.exp(1j * delta),
         c12 * c23 - s12 * s23 * s13 * np.exp(1j * delta),
         s23 * c13],
        [s12 * s23 - c12 * c23 * s13 * np.exp(1j * delta),
         -c12 * s23 - s12 * c23 * s13 * np.exp(1j * delta),
         c23 * c13],
    ], dtype=complex)


def _yukawa_from_left_unitary(U_left, singular_values):
    """Construct Y = U_left diag(S) with arbitrary right factor."""
    S = np.diag(singular_values)
    V = np.eye(3)
    return U_left @ S @ V.conj().T


def test_pmns_angle_roundtrip():
    """Known PMNS angles survive unitary extraction."""
    targets = NEUTRINO_TARGETS
    U = _pmns_matrix(targets['theta12'], targets['theta23'], targets['theta13'])
    t12, t23, t13 = pmns_angles_from_unitary(U)
    assert abs(t12 - targets['theta12']) < 1e-10
    assert abs(t23 - targets['theta23']) < 1e-10
    assert abs(t13 - targets['theta13']) < 1e-10
    print("✓ PMNS angle roundtrip test passed")


def test_neutrino_observables_from_synthetic_yukawas():
    """SVD pipeline recovers planted PMNS angles."""
    U_pmns = _pmns_matrix(
        NEUTRINO_TARGETS['theta12'],
        NEUTRINO_TARGETS['theta23'],
        NEUTRINO_TARGETS['theta13'],
    )
    # Ye -> Ue, Ynu -> Unu with PMNS = Ue† Unu
    Ue = np.eye(3)
    Unu = Ue @ U_pmns
    Ye = _yukawa_from_left_unitary(Ue, [1.0, 0.1, 0.001])
    Ynu = _yukawa_from_left_unitary(Unu, [0.05, 0.02, 0.001])

    obs = compute_neutrino_observables(Ynu, Ye)
    for key in ['theta12', 'theta23', 'theta13']:
        assert key in obs
        assert not np.isnan(obs[key])
        rel_err = abs(obs[key] - NEUTRINO_TARGETS[key]) / NEUTRINO_TARGETS[key]
        assert rel_err < 0.01, f"{key} relative error {rel_err:.4f} too large"

    loss = compute_pmns_loss(obs)
    assert loss < 1e-4, f"PMNS loss should be small for planted angles, got {loss}"
    assert obs['unitarity_violation'] < 1e-10
    print("✓ Synthetic Yukawa PMNS extraction test passed")


def test_pmns_loss_perfect_match():
    """PMNS loss is ~0 when observables match targets."""
    obs = NEUTRINO_TARGETS.copy()
    loss = compute_pmns_loss(obs)
    assert loss < 1e-12, f"Expected ~0 loss, got {loss}"
    print("✓ PMNS loss function test passed")


if __name__ == "__main__":
    test_pmns_angle_roundtrip()
    test_neutrino_observables_from_synthetic_yukawas()
    test_pmns_loss_perfect_match()
    print("\n✓ All neutrino observable tests passed!")
