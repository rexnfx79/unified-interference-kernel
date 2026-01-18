"""
Tests for observables module - focuses on code correctness
"""

import sys
sys.path.insert(0, '../src')

import numpy as np
from kernel import compute_quark_yukawas
from observables import compute_quark_observables, compute_ckm_loss, compute_mass_loss, QUARK_TARGETS


def test_observable_extraction():
    """Test that observable extraction works without errors"""
    Q = (0, 1, 0)
    U = (0, 3, 6)
    D = (0, 3, 7)
    
    Yu, Yd = compute_quark_yukawas(Q, U, D, sigma=1.5, k=0.5, alpha=0.0, eta=2.5, eps_u=0.15, eps_d=0.15)
    obs = compute_quark_observables(Yu, Yd)
    
    # Check all expected keys present
    expected_keys = ['Vus', 'Vcb', 'Vub', 'mu', 'mc', 'md', 'ms', 'scale_u', 'scale_d']
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


def test_unitarity():
    """Test that CKM matrix is unitary"""
    Q = (0, 1, 0)
    U = (0, 3, 6)
    D = (0, 3, 7)
    
    Yu, Yd = compute_quark_yukawas(Q, U, D, sigma=1.5, k=0.5, alpha=0.0, eta=2.5, eps_u=0.15, eps_d=0.15)
    
    # Get full CKM matrix
    Uu, Su, Vuh = np.linalg.svd(Yu, full_matrices=False)
    Ud, Sd, Vdh = np.linalg.svd(Yd, full_matrices=False)
    CKM = Uu.conj().T @ Ud
    
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
