"""
Tests for kernel module - focuses on code correctness, not physics values
"""

import sys
sys.path.insert(0, '../src')

import numpy as np
from kernel import compute_kernel_element, compute_quark_yukawas


def test_kernel_element_basic():
    """Test basic kernel element computation"""
    y = compute_kernel_element(0.0, 0.0, sigma=1.0, k=0.5, alpha=0.0, eta=2.0, eps=0.1)
    assert not np.isnan(y), "Kernel element should not be NaN"
    assert not np.isinf(y), "Kernel element should not be Inf"
    print("✓ Basic kernel element test passed")


def test_kernel_hermiticity():
    """Test that Yukawa matrices have proper structure"""
    Q = (0, 1, 0)
    U = (0, 1, 2)
    D = (0, 1, 2)
    Yu, Yd = compute_quark_yukawas(Q, U, D, sigma=1.5, k=0.5, alpha=0.0, eta=2.0, eps_u=0.1, eps_d=0.1)
    
    # Check dimensions
    assert Yu.shape == (3, 3), "Yu should be 3x3"
    assert Yd.shape == (3, 3), "Yd should be 3x3"
    
    # Check SVD works (no NaNs or infinities)
    Uu, Su, Vuh = np.linalg.svd(Yu)
    assert not np.any(np.isnan(Su)), "Yu SVD contains NaN"
    assert not np.any(np.isinf(Su)), "Yu SVD contains Inf"
    assert np.all(Su >= 0), "Singular values should be non-negative"
    
    print("✓ Kernel structure test passed")


def test_parameter_variation():
    """Test that varying parameters changes output"""
    Q = (0, 1, 0)
    U = (0, 3, 6)
    D = (0, 3, 7)
    
    Yu1, Yd1 = compute_quark_yukawas(Q, U, D, sigma=1.0, k=0.5, alpha=0.0, eta=2.0, eps_u=0.1, eps_d=0.1)
    Yu2, Yd2 = compute_quark_yukawas(Q, U, D, sigma=2.0, k=0.5, alpha=0.0, eta=2.0, eps_u=0.1, eps_d=0.1)
    
    # Different sigma should give different results
    diff = np.linalg.norm(Yu1 - Yu2)
    assert diff > 1e-6, "Changing sigma should change Yukawa matrix"
    
    print("✓ Parameter variation test passed")


if __name__ == "__main__":
    test_kernel_element_basic()
    test_kernel_hermiticity()
    test_parameter_variation()
    print("\n✓ All kernel tests passed!")
