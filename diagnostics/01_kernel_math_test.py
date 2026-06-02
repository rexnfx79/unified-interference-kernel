#!/usr/bin/env python3
"""
Diagnostic Test 1: Kernel Math Verification

Verify that the kernel formula matches the documented mathematical form:
    Y_ij = exp(-d²/(2σ²)) × [1 + ε exp(iΦ)]

where:
    d = |x_i - x_j| is the distance in internal flavor coordinate
    Φ = α + k(x_i + x_j)/2 + η(x_i - x_j) is the phase structure
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from kernel import compute_kernel_element, compute_yukawa_matrix

# Test results tracking
RESULTS = []


def log_result(test_name: str, passed: bool, details: str = ""):
    """Log a test result."""
    status = "PASS" if passed else "FAIL"
    RESULTS.append((test_name, passed, details))
    print(f"[{status}] {test_name}")
    if details:
        print(f"       {details}")


def test_envelope_suppression():
    """Test that envelope suppression follows exp(-d²/(2σ²))."""
    print("\n" + "="*60)
    print("TEST: Envelope Suppression")
    print("="*60)
    
    # Test parameters
    sigma = 2.0
    k = 0.0      # No phase contribution
    alpha = 0.0
    eta = 0.0
    eps = 0.0    # No interference term
    
    # Test cases: (x_left, x_right, expected_envelope)
    # Formula: exp(-d^2 / (2*sigma^2)) where d = x_left - x_right, sigma = 2.0
    test_cases = [
        (0.0, 0.0, 1.0),                           # d=0, envelope=1
        (1.0, 0.0, np.exp(-1.0 / (2 * 4))),        # d=1, d^2=1, envelope=exp(-1/8)
        (2.0, 0.0, np.exp(-4.0 / (2 * 4))),        # d=2, d^2=4, envelope=exp(-1/2)
        (4.0, 0.0, np.exp(-16.0 / (2 * 4))),       # d=4, d^2=16, envelope=exp(-2)
    ]
    
    all_passed = True
    for x_left, x_right, expected in test_cases:
        result = compute_kernel_element(x_left, x_right, sigma, k, alpha, eta, eps)
        # With eps=0, interference term is (1 + 0) = 1, so result = envelope
        actual = abs(result)
        diff = abs(actual - expected)
        passed = diff < 1e-10
        if not passed:
            all_passed = False
            print(f"  FAIL: x_left={x_left}, x_right={x_right}")
            print(f"        Expected: {expected:.10f}, Got: {actual:.10f}")
    
    log_result("Envelope suppression formula", all_passed,
               f"Tested {len(test_cases)} cases with eps=0, k=0")


def test_phase_structure():
    """Test that phase structure follows Φ = α + k(x_i + x_j)/2 + η(x_i - x_j)."""
    print("\n" + "="*60)
    print("TEST: Phase Structure")
    print("="*60)
    
    sigma = 100.0  # Large sigma so envelope ≈ 1
    eps = 1.0      # Full interference
    
    def compute_expected(x_left, x_right, sigma, k, alpha, eta, eps):
        """Compute expected kernel value including envelope."""
        diff = x_left - x_right
        envelope = np.exp(-diff**2 / (2 * sigma**2))
        phase = alpha + k * (x_left + x_right) / 2 + eta * diff
        interference = 1 + eps * np.exp(1j * phase)
        return envelope * interference
    
    # Test 1: Pure alpha phase
    alpha = np.pi / 4
    k = 0.0
    eta = 0.0
    x_left, x_right = 0.0, 0.0
    
    result = compute_kernel_element(x_left, x_right, sigma, k, alpha, eta, eps)
    expected = compute_expected(x_left, x_right, sigma, k, alpha, eta, eps)
    
    passed1 = abs(result - expected) < 1e-10
    log_result("Phase: pure alpha", passed1,
               f"α={alpha:.4f}, result={result:.4f}, expected={expected:.4f}")
    
    # Test 2: k contribution (center-of-mass term)
    alpha = 0.0
    k = 1.0
    eta = 0.0
    x_left, x_right = 2.0, 4.0
    
    result = compute_kernel_element(x_left, x_right, sigma, k, alpha, eta, eps)
    expected = compute_expected(x_left, x_right, sigma, k, alpha, eta, eps)
    expected_phase = k * (x_left + x_right) / 2
    
    passed2 = abs(result - expected) < 1e-8
    log_result("Phase: k*(x_i+x_j)/2 term", passed2,
               f"k={k}, x_left={x_left}, x_right={x_right}, expected phase={expected_phase:.4f}")
    
    # Test 3: eta contribution (difference term)
    alpha = 0.0
    k = 0.0
    eta = 1.0
    x_left, x_right = 5.0, 2.0
    
    result = compute_kernel_element(x_left, x_right, sigma, k, alpha, eta, eps)
    expected = compute_expected(x_left, x_right, sigma, k, alpha, eta, eps)
    expected_phase = eta * (x_left - x_right)
    
    passed3 = abs(result - expected) < 1e-8
    log_result("Phase: η*(x_i-x_j) term", passed3,
               f"η={eta}, x_left={x_left}, x_right={x_right}, expected phase={expected_phase:.4f}")
    
    # Test 4: Combined phase
    alpha = 0.5
    k = 0.3
    eta = 0.2
    x_left, x_right = 3.0, 7.0
    
    result = compute_kernel_element(x_left, x_right, sigma, k, alpha, eta, eps)
    expected = compute_expected(x_left, x_right, sigma, k, alpha, eta, eps)
    expected_phase = alpha + k * (x_left + x_right) / 2 + eta * (x_left - x_right)
    
    passed4 = abs(result - expected) < 1e-8
    log_result("Phase: combined α + k + η", passed4,
               f"Expected phase={expected_phase:.4f}")


def test_full_kernel_formula():
    """Test the complete kernel formula Y_ij = envelope × interference."""
    print("\n" + "="*60)
    print("TEST: Full Kernel Formula")
    print("="*60)
    
    # Arbitrary parameters
    sigma = 2.5
    k = 0.7
    alpha = 1.2
    eta = 0.4
    eps = 0.6
    x_left = 3.0
    x_right = 5.0
    
    # Compute expected value manually
    diff = x_left - x_right
    envelope = np.exp(-diff**2 / (2 * sigma**2))
    phase = alpha + k * (x_left + x_right) / 2 + eta * diff
    interference = 1 + eps * np.exp(1j * phase)
    expected = envelope * interference
    
    # Get actual value
    actual = compute_kernel_element(x_left, x_right, sigma, k, alpha, eta, eps)
    
    passed = abs(actual - expected) < 1e-10
    log_result("Full kernel formula", passed,
               f"Expected: {expected:.6f}, Got: {actual:.6f}")


def test_yukawa_matrix_construction():
    """Test that compute_yukawa_matrix correctly builds the 3x3 matrix."""
    print("\n" + "="*60)
    print("TEST: Yukawa Matrix Construction")
    print("="*60)
    
    # Test parameters
    left_positions = (1, 3, 5)   # Note: only first 2 are used!
    right_positions = (0, 2, 4)
    sigma = 2.0
    k = 0.5
    alpha = 0.3
    eta = 0.1
    eps = 0.4
    
    # Get the matrix
    Y = compute_yukawa_matrix(left_positions, right_positions, sigma, k, alpha, eta, eps)
    
    # Check shape
    passed_shape = Y.shape == (3, 3)
    log_result("Matrix shape is 3x3", passed_shape, f"Got shape: {Y.shape}")
    
    # CRITICAL CHECK: What left_vec is actually used?
    # From kernel.py line 47: left_vec = np.array([left_positions[0], left_positions[1], 0], dtype=float)
    # This means left_vec = [1, 3, 0], NOT [1, 3, 5]!
    
    actual_left_vec = [left_positions[0], left_positions[1], 0]  # The hardcoded behavior
    
    # Verify each element
    all_elements_correct = True
    for i in range(3):
        for j in range(3):
            expected_element = compute_kernel_element(
                actual_left_vec[i], right_positions[j],
                sigma, k, alpha, eta, eps
            )
            if abs(Y[i, j] - expected_element) > 1e-10:
                all_elements_correct = False
                print(f"  MISMATCH at [{i},{j}]: expected {expected_element}, got {Y[i,j]}")
    
    log_result("Matrix elements match kernel formula", all_elements_correct)
    
    # IMPORTANT: Check if the third left position is ignored
    print("\n  *** CRITICAL FINDING ***")
    print(f"  Input left_positions: {left_positions}")
    print(f"  Actual left_vec used: {actual_left_vec}")
    print(f"  The THIRD left position ({left_positions[2]}) is IGNORED and replaced with 0!")
    
    # This is the hardcoded 0 bug
    log_result("Third left position is used", False,
               f"kernel.py line 47 hardcodes left_vec[2]=0, ignoring input {left_positions[2]}")


def test_complex_output():
    """Verify the kernel produces complex numbers when expected."""
    print("\n" + "="*60)
    print("TEST: Complex Output")
    print("="*60)
    
    sigma = 2.0
    k = 1.0
    alpha = np.pi / 2  # This should give imaginary component
    eta = 0.0
    eps = 1.0
    
    result = compute_kernel_element(0.0, 0.0, sigma, k, alpha, eta, eps)
    
    # With alpha=π/2, exp(iα) = i, so interference = 1 + i
    # envelope = 1 (d=0)
    # result should be 1 + i
    expected = 1 + 1j
    
    passed = abs(result - expected) < 1e-10
    log_result("Complex output with α=π/2", passed,
               f"Expected: {expected}, Got: {result}")
    
    # Check that we get real output when phase is 0
    result_real = compute_kernel_element(0.0, 0.0, sigma, 0.0, 0.0, 0.0, 1.0)
    expected_real = 2.0  # 1 + 1*exp(i*0) = 1 + 1 = 2
    
    passed_real = abs(result_real.imag) < 1e-10 and abs(result_real.real - 2.0) < 1e-10
    log_result("Real output when phase=0", passed_real,
               f"Expected: 2.0, Got: {result_real}")


def main():
    """Run all kernel math tests."""
    print("="*60)
    print("DIAGNOSTIC 1: KERNEL MATH VERIFICATION")
    print("="*60)
    
    test_envelope_suppression()
    test_phase_structure()
    test_full_kernel_formula()
    test_yukawa_matrix_construction()
    test_complex_output()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, p, _ in RESULTS if p)
    total = len(RESULTS)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed < total:
        print("\nFAILED TESTS:")
        for name, p, details in RESULTS:
            if not p:
                print(f"  - {name}")
                if details:
                    print(f"    {details}")
    
    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, '01_kernel_math_results.txt'), 'w') as f:
        f.write("KERNEL MATH VERIFICATION RESULTS\n")
        f.write("="*50 + "\n\n")
        for name, p, details in RESULTS:
            status = "PASS" if p else "FAIL"
            f.write(f"[{status}] {name}\n")
            if details:
                f.write(f"        {details}\n")
        f.write(f"\nTotal: {passed}/{total} tests passed\n")
    
    print(f"\nResults saved to diagnostics/results/01_kernel_math_results.txt")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
