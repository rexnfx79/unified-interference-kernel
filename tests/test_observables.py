"""
Tests for observables module - focuses on code correctness
"""

import sys
sys.path.insert(0, '../src')

import numpy as np
from kernel import compute_quark_yukawas
from observables import (
    compute_quark_observables,
    compute_ckm_loss,
    compute_mass_loss,
    fix_svd_phases,
    jarlskog_invariant,
    QUARK_TARGETS,
    QUARK_CP_TARGETS,
)


def test_observable_extraction():
    """Test that observable extraction works without errors"""
    Q = (0, 1, 0)
    U = (0, 3, 6)
    D = (0, 3, 7)
    
    Yu, Yd = compute_quark_yukawas(Q, U, D, sigma=1.5, k=0.5, alpha=0.0, eta=2.5, eps_u=0.15, eps_d=0.15)
    obs = compute_quark_observables(Yu, Yd)
    
    # Check all expected keys present
    expected_keys = [
        'Vus', 'Vcb', 'Vub', 'mu', 'mc', 'md', 'ms', 'scale_u', 'scale_d',
        'delta_CKM', 'J', 'J_abs',
    ]
    for key in expected_keys:
        assert key in obs, f"Missing key: {key}"
        assert not np.isnan(obs[key]), f"{key} is NaN"
        assert not np.isinf(obs[key]), f"{key} is Inf"
    
    # Check masses are positive
    for key in ['mu', 'mc', 'md', 'ms']:
        assert obs[key] > 0, f"{key} should be positive"
    
    print("✓ Observable extraction test passed")


def test_loss_functions():
    """Test that loss functions work correctly"""
    # Create observables close to targets
    obs = QUARK_TARGETS.copy()
    
    # CKM loss should be ~0 for perfect match
    ckm_loss = compute_ckm_loss(obs)
    assert ckm_loss < 1e-10, f"CKM loss should be ~0 for perfect match, got {ckm_loss}"
    
    # Mass loss should be ~0 for perfect match
    mass_loss = compute_mass_loss(obs)
    assert mass_loss < 1e-10, f"Mass loss should be ~0 for perfect match, got {mass_loss}"
    
    # Now perturb and check loss increases
    obs_perturbed = obs.copy()
    obs_perturbed['Vus'] *= 1.1
    ckm_loss_perturbed = compute_ckm_loss(obs_perturbed)
    assert ckm_loss_perturbed > ckm_loss, "Loss should increase for perturbed values"
    
    print("✓ Loss function test passed")


def test_fix_svd_phases_preserves_reconstruction():
    """Phase fixing must not break Y = U diag(S) Vh."""
    Y = np.array([
        [1.0 + 0.5j, 0.1 - 0.1j, 0.01],
        [0.2j, 0.5 + 0.3j, 0.05 - 0.02j],
        [0.01 + 0.01j, 0.02, 0.1 - 0.05j],
    ], dtype=complex)
    U, S, Vh = np.linalg.svd(Y, full_matrices=False)
    U_fixed, S_fixed, Vh_fixed = fix_svd_phases(U, S, Vh)
    err = np.max(np.abs(Y - U_fixed @ np.diag(S_fixed) @ Vh_fixed))
    assert err < 1e-10, f"SVD reconstruction error after phase fix: {err:.2e}"


def test_unitarity():
    """Test that CKM matrix is unitary"""
    Q = (0, 1, 0)
    U = (0, 3, 6)
    D = (0, 3, 7)
    
    Yu, Yd = compute_quark_yukawas(Q, U, D, sigma=1.5, k=0.5, alpha=0.0, eta=2.5, eps_u=0.15, eps_d=0.15)
    
    # Get full CKM matrix
    Uu, Su, Vuh = np.linalg.svd(Yu, full_matrices=False)
    Ud, Sd, Vdh = np.linalg.svd(Yd, full_matrices=False)
    Uu_fixed, _, _ = fix_svd_phases(Uu, Su, Vuh)
    Ud_fixed, _, _ = fix_svd_phases(Ud, Sd, Vdh)
    CKM = Uu_fixed.conj().T @ Ud_fixed
    
    # Check unitarity: V V† = I
    identity = CKM @ CKM.conj().T
    deviation = np.linalg.norm(identity - np.eye(3))
    
    assert deviation < 1e-10, f"CKM matrix should be unitary, deviation: {deviation}"
    
    print("✓ Unitarity test passed")


if __name__ == "__main__":
    test_observable_extraction()
    test_loss_functions()
    test_unitarity()
    print("\n✓ All observable tests passed!")
