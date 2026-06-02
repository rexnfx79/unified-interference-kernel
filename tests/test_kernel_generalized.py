#!/usr/bin/env python3
"""
Comprehensive tests for generalized kernel module.

QA Strategy:
1. Unit tests for each function
2. Property-based tests (mathematical invariants)
3. Edge case tests
4. Regression tests (p=2 should match Gaussian)
5. Numerical stability tests
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np


def test_envelope_at_zero():
    """Test: envelope(0) = 1 for all p"""
    from kernel_generalized import compute_generalized_envelope
    
    sigma = 1.5
    p_values = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    
    for p in p_values:
        env = compute_generalized_envelope(0.0, sigma, p)
        assert abs(env - 1.0) < 1e-14, f"envelope(0) should be 1 for p={p}, got {env}"
    
    print("✓ test_envelope_at_zero passed")


def test_envelope_at_sigma():
    """Test: envelope(σ) = exp(-1/p) for all p"""
    from kernel_generalized import compute_generalized_envelope
    
    sigma = 2.0
    p_values = [1.0, 1.5, 2.0, 3.0]
    
    for p in p_values:
        env = compute_generalized_envelope(sigma, sigma, p)
        expected = np.exp(-1/p)
        assert abs(env - expected) < 1e-14, f"envelope(σ) should be exp(-1/{p})={expected}, got {env}"
    
    print("✓ test_envelope_at_sigma passed")


def test_envelope_monotonic():
    """Test: envelope decreases with |d|"""
    from kernel_generalized import compute_generalized_envelope
    
    sigma = 1.0
    distances = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    for p in [1.0, 2.0, 3.0]:
        values = [compute_generalized_envelope(d, sigma, p) for d in distances]
        for i in range(len(values) - 1):
            assert values[i] >= values[i+1], f"Not monotonic at p={p}: {values}"
    
    print("✓ test_envelope_monotonic passed")


def test_envelope_symmetric():
    """Test: envelope(d) = envelope(-d)"""
    from kernel_generalized import compute_generalized_envelope
    
    sigma = 1.5
    
    for p in [1.0, 2.0, 3.0]:
        for d in [0.5, 1.0, 2.0, 5.0]:
            env_pos = compute_generalized_envelope(d, sigma, p)
            env_neg = compute_generalized_envelope(-d, sigma, p)
            assert abs(env_pos - env_neg) < 1e-14, f"Not symmetric: {env_pos} vs {env_neg}"
    
    print("✓ test_envelope_symmetric passed")


def test_envelope_positive():
    """Test: envelope is always positive (within numerical precision)"""
    from kernel_generalized import compute_generalized_envelope
    
    sigma = 1.0
    
    # Use realistic distances (not extreme values that underflow)
    for p in [0.5, 1.0, 2.0, 5.0]:
        for d in [0.0, 1.0, 5.0, 10.0]:
            env = compute_generalized_envelope(d, sigma, p)
            # Allow for numerical underflow to exactly 0 at extreme distances
            assert env >= 0, f"Envelope should be non-negative, got {env}"
    
    print("✓ test_envelope_positive passed")


def test_envelope_invalid_inputs():
    """Test: proper error handling for invalid inputs"""
    from kernel_generalized import compute_generalized_envelope
    
    # sigma <= 0 should raise
    try:
        compute_generalized_envelope(1.0, 0.0, 2.0)
        assert False, "Should have raised ValueError for sigma=0"
    except ValueError:
        pass
    
    try:
        compute_generalized_envelope(1.0, -1.0, 2.0)
        assert False, "Should have raised ValueError for sigma<0"
    except ValueError:
        pass
    
    # p <= 0 should raise
    try:
        compute_generalized_envelope(1.0, 1.0, 0.0)
        assert False, "Should have raised ValueError for p=0"
    except ValueError:
        pass
    
    try:
        compute_generalized_envelope(1.0, 1.0, -1.0)
        assert False, "Should have raised ValueError for p<0"
    except ValueError:
        pass
    
    print("✓ test_envelope_invalid_inputs passed")


def test_kernel_element_basic():
    """Test: kernel element computation doesn't crash"""
    from kernel_generalized import compute_kernel_element_generalized
    
    y = compute_kernel_element_generalized(
        x_left=0.0, x_right=1.0,
        sigma=1.5, k=0.5, alpha=0.3, eta=2.0, eps=0.15,
        p=2.0
    )
    
    assert not np.isnan(y), "Kernel element should not be NaN"
    assert not np.isinf(y), "Kernel element should not be Inf"
    assert isinstance(y, (complex, np.complexfloating)), "Should be complex"
    
    print("✓ test_kernel_element_basic passed")


def test_kernel_element_p_variation():
    """Test: kernel element varies smoothly with p"""
    from kernel_generalized import compute_kernel_element_generalized
    
    x_left, x_right = 0.0, 2.0
    sigma, k, alpha, eta, eps = 1.5, 0.5, 0.3, 2.0, 0.15
    
    p_values = np.linspace(1.0, 3.0, 21)
    y_values = [
        compute_kernel_element_generalized(x_left, x_right, sigma, k, alpha, eta, eps, p)
        for p in p_values
    ]
    
    # Check all are finite
    for i, y in enumerate(y_values):
        assert np.isfinite(y), f"Kernel element not finite at p={p_values[i]}"
    
    # Check smoothness (no jumps > 50% between adjacent p)
    for i in range(len(y_values) - 1):
        ratio = abs(y_values[i+1]) / abs(y_values[i]) if abs(y_values[i]) > 1e-10 else 1.0
        assert 0.5 < ratio < 2.0, f"Large jump between p={p_values[i]} and p={p_values[i+1]}"
    
    print("✓ test_kernel_element_p_variation passed")


def test_yukawa_matrix_shape():
    """Test: Yukawa matrix has correct shape"""
    from kernel_generalized import compute_yukawa_matrix_generalized
    
    Y = compute_yukawa_matrix_generalized(
        left_positions=(0, 1, 0),
        right_positions=(0, 3, 6),
        sigma=1.5, k=0.5, alpha=0.3, eta=2.0, eps=0.15, p=2.0
    )
    
    assert Y.shape == (3, 3), f"Expected (3,3), got {Y.shape}"
    assert Y.dtype == np.complex128 or np.issubdtype(Y.dtype, np.complexfloating), "Should be complex"
    
    print("✓ test_yukawa_matrix_shape passed")


def test_yukawa_matrix_svd():
    """Test: Yukawa matrix has valid SVD"""
    from kernel_generalized import compute_yukawa_matrix_generalized
    
    for p in [1.0, 1.5, 2.0, 3.0]:
        Y = compute_yukawa_matrix_generalized(
            left_positions=(0, 1, 0),
            right_positions=(0, 3, 6),
            sigma=1.5, k=0.5, alpha=0.3, eta=2.0, eps=0.15, p=p
        )
        
        U, S, Vh = np.linalg.svd(Y)
        
        assert not np.any(np.isnan(S)), f"SVD contains NaN at p={p}"
        assert not np.any(np.isinf(S)), f"SVD contains Inf at p={p}"
        assert np.all(S >= 0), f"Singular values should be non-negative at p={p}"
    
    print("✓ test_yukawa_matrix_svd passed")


def test_quark_yukawas():
    """Test: quark Yukawa computation"""
    from kernel_generalized import compute_quark_yukawas_generalized
    
    Q = (0, 1, 0)
    U = (0, 3, 6)
    D = (0, 3, 7)
    
    for p in [1.0, 2.0, 3.0]:
        Yu, Yd = compute_quark_yukawas_generalized(
            Q, U, D,
            sigma=1.5, k=0.5, alpha=0.3, eta=2.0,
            eps_u=0.15, eps_d=0.15, p=p
        )
        
        assert Yu.shape == (3, 3), f"Yu wrong shape at p={p}"
        assert Yd.shape == (3, 3), f"Yd wrong shape at p={p}"
        
        # Check SVD works
        _, Su, _ = np.linalg.svd(Yu)
        _, Sd, _ = np.linalg.svd(Yd)
        
        assert np.all(np.isfinite(Su)), f"Yu SVD not finite at p={p}"
        assert np.all(np.isfinite(Sd)), f"Yd SVD not finite at p={p}"
    
    print("✓ test_quark_yukawas passed")


def test_gaussian_equivalence():
    """Test: p=2 matches standard Gaussian kernel"""
    from kernel_generalized import compute_kernel_element_generalized
    from kernel import compute_kernel_element
    
    test_cases = [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 2.0),
        (3.0, 1.0),
        (-2.0, 4.0),
        (5.0, -3.0),
    ]
    
    sigma, k, alpha, eta, eps = 1.5, 0.5, 0.3, 2.0, 0.15
    
    for x_left, x_right in test_cases:
        y_standard = compute_kernel_element(x_left, x_right, sigma, k, alpha, eta, eps)
        y_generalized = compute_kernel_element_generalized(
            x_left, x_right, sigma, k, alpha, eta, eps, p=2.0
        )
        
        diff = abs(y_standard - y_generalized)
        assert diff < 1e-10, f"Mismatch at ({x_left}, {x_right}): {y_standard} vs {y_generalized}, diff={diff}"
    
    print("✓ test_gaussian_equivalence passed")


def test_exponential_limit():
    """Test: p=1 gives exponential decay"""
    from kernel_generalized import compute_generalized_envelope
    
    sigma = 1.0
    d = 2.0
    
    # At p=1: exp(-(|d|/σ)^1 / 1) = exp(-|d|/σ)
    env = compute_generalized_envelope(d, sigma, p=1.0)
    expected = np.exp(-d / sigma)
    
    assert abs(env - expected) < 1e-14, f"p=1 should give exp(-d/σ), got {env} vs {expected}"
    
    print("✓ test_exponential_limit passed")


def test_numerical_stability_large_d():
    """Test: no overflow/underflow for large distances"""
    from kernel_generalized import compute_generalized_envelope, compute_kernel_element_generalized
    
    sigma = 1.0
    
    for p in [1.0, 2.0, 3.0]:
        for d in [10.0, 20.0, 30.0]:  # Reasonable large distances
            env = compute_generalized_envelope(d, sigma, p)
            assert np.isfinite(env), f"Envelope not finite at d={d}, p={p}"
            assert env >= 0, f"Envelope negative at d={d}, p={p}"
            
            y = compute_kernel_element_generalized(0.0, d, sigma, 0.5, 0.3, 2.0, 0.15, p)
            assert np.isfinite(y), f"Kernel element not finite at d={d}, p={p}"
    
    print("✓ test_numerical_stability_large_d passed")


def test_numerical_stability_small_sigma():
    """Test: no issues with small sigma"""
    from kernel_generalized import compute_generalized_envelope
    
    for sigma in [0.1, 0.01, 0.001]:
        for p in [1.0, 2.0]:
            env = compute_generalized_envelope(1.0, sigma, p)
            assert np.isfinite(env), f"Envelope not finite at sigma={sigma}, p={p}"
            assert env >= 0, f"Envelope negative at sigma={sigma}, p={p}"
    
    print("✓ test_numerical_stability_small_sigma passed")


def test_verify_functions():
    """Test: built-in verification functions work"""
    from kernel_generalized import verify_envelope_properties
    
    results = verify_envelope_properties(sigma=1.0)
    
    for test_name, test_results in results.items():
        for p, passed in test_results.items():
            assert passed, f"Verification failed: {test_name} at p={p}"
    
    print("✓ test_verify_functions passed")


def run_all_tests():
    """Run all tests with summary"""
    tests = [
        test_envelope_at_zero,
        test_envelope_at_sigma,
        test_envelope_monotonic,
        test_envelope_symmetric,
        test_envelope_positive,
        test_envelope_invalid_inputs,
        test_kernel_element_basic,
        test_kernel_element_p_variation,
        test_yukawa_matrix_shape,
        test_yukawa_matrix_svd,
        test_quark_yukawas,
        test_gaussian_equivalence,
        test_exponential_limit,
        test_numerical_stability_large_d,
        test_numerical_stability_small_sigma,
        test_verify_functions,
    ]
    
    passed = 0
    failed = 0
    
    print("=" * 60)
    print("GENERALIZED KERNEL TEST SUITE")
    print("=" * 60)
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} ERROR: {e}")
            failed += 1
    
    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
