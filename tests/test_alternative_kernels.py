#!/usr/bin/env python3
"""
QA Unit Tests for Alternative Kernels

Tests to verify:
1. Kernel mathematical correctness
2. Reproducibility with fixed seeds
3. Known solution verification
4. Edge cases and numerical stability
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from alternative_kernels import (
    gaussian_kernel_element,
    power_law_kernel_element,
    exponential_kernel_element,
    clockwork_kernel_element,
    hybrid_kernel_element,
    compute_yukawas_clockwork,
    compute_yukawas_gaussian,
    KERNELS,
)
from observables import compute_quark_observables, QUARK_TARGETS


class TestKernelMathematics(unittest.TestCase):
    """Test that kernel formulas are mathematically correct."""
    
    def test_gaussian_at_zero_distance(self):
        """Gaussian kernel at d=0 should give envelope=1."""
        result = gaussian_kernel_element(5.0, 5.0, sigma=2.0, k=0, alpha=0, eta=0, eps=0)
        self.assertAlmostEqual(abs(result), 1.0, places=10)
    
    def test_gaussian_envelope_decay(self):
        """Gaussian envelope should decay as exp(-d²/(2σ²))."""
        sigma = 2.0
        x_left, x_right = 4.0, 0.0  # d = 4
        
        result = gaussian_kernel_element(x_left, x_right, sigma=sigma, k=0, alpha=0, eta=0, eps=0)
        expected = np.exp(-16 / (2 * 4))  # exp(-d²/(2σ²)) = exp(-16/8) = exp(-2)
        
        self.assertAlmostEqual(abs(result), expected, places=10)
    
    def test_clockwork_at_zero_distance(self):
        """Clockwork kernel at d=0 should give envelope=1."""
        result = clockwork_kernel_element(5.0, 5.0, q=3.0, k=0, alpha=0, eta=0, eps=0)
        self.assertAlmostEqual(abs(result), 1.0, places=10)
    
    def test_clockwork_envelope_decay(self):
        """Clockwork envelope should decay as q^(-|d|)."""
        q = 3.0
        x_left, x_right = 5.0, 2.0  # d = 3
        
        result = clockwork_kernel_element(x_left, x_right, q=q, k=0, alpha=0, eta=0, eps=0)
        expected = q ** (-3)  # 3^(-3) = 1/27
        
        self.assertAlmostEqual(abs(result), expected, places=10)
    
    def test_exponential_envelope_decay(self):
        """Exponential envelope should decay as exp(-|d|/λ)."""
        lambda_scale = 2.0
        x_left, x_right = 6.0, 0.0  # d = 6
        
        result = exponential_kernel_element(x_left, x_right, lambda_scale=lambda_scale, 
                                            k=0, alpha=0, eta=0, eps=0)
        expected = np.exp(-6 / 2)  # exp(-3)
        
        self.assertAlmostEqual(abs(result), expected, places=10)
    
    def test_power_law_envelope_decay(self):
        """Power-law envelope should decay as ε^(|d|/λ)."""
        epsilon = 0.22
        lambda_scale = 2.0
        x_left, x_right = 4.0, 0.0  # d = 4
        
        result = power_law_kernel_element(x_left, x_right, epsilon=epsilon, 
                                          lambda_scale=lambda_scale, k=0, alpha=0, 
                                          eta=0, eps_phase=0)
        expected = epsilon ** (4 / 2)  # 0.22^2
        
        self.assertAlmostEqual(abs(result), expected, places=10)
    
    def test_phase_interference(self):
        """Test that phase interference term works correctly."""
        # With alpha=π, exp(iπ) = -1, so interference = 1 + eps*(-1) = 1 - eps
        result = clockwork_kernel_element(0.0, 0.0, q=3.0, k=0, alpha=np.pi, eta=0, eps=0.5)
        expected = 1 - 0.5  # envelope=1, interference=0.5
        
        self.assertAlmostEqual(abs(result), expected, places=10)
    
    def test_phase_constructive(self):
        """Test constructive interference (alpha=0)."""
        result = clockwork_kernel_element(0.0, 0.0, q=3.0, k=0, alpha=0, eta=0, eps=0.5)
        expected = 1 + 0.5  # envelope=1, interference=1.5
        
        self.assertAlmostEqual(abs(result), expected, places=10)


class TestReproducibility(unittest.TestCase):
    """Test that results are reproducible with fixed seeds."""
    
    def test_clockwork_yukawa_reproducible(self):
        """Clockwork Yukawa matrices should be deterministic."""
        Q = (7, 8, 9)
        U = (2, 12, 14)
        D = (1, 4, 7)
        params = (11.64, 3.06, 0.63, 1.49, 1.0, 1.0)
        
        Yu1, Yd1 = compute_yukawas_clockwork(Q, U, D, *params)
        Yu2, Yd2 = compute_yukawas_clockwork(Q, U, D, *params)
        
        np.testing.assert_array_almost_equal(Yu1, Yu2, decimal=15)
        np.testing.assert_array_almost_equal(Yd1, Yd2, decimal=15)
    
    def test_observables_reproducible(self):
        """Observable extraction should be deterministic."""
        Q = (7, 8, 9)
        U = (2, 12, 14)
        D = (1, 4, 7)
        params = (11.64, 3.06, 0.63, 1.49, 1.0, 1.0)
        
        Yu, Yd = compute_yukawas_clockwork(Q, U, D, *params)
        
        obs1 = compute_quark_observables(Yu, Yd)
        obs2 = compute_quark_observables(Yu, Yd)
        
        for key in obs1:
            self.assertAlmostEqual(obs1[key], obs2[key], places=10,
                                   msg=f"Observable {key} not reproducible")


class TestClockworkSolution(unittest.TestCase):
    """Verify the claimed Clockwork solution is correct."""
    
    def setUp(self):
        """Set up the known good solution."""
        self.Q = (7, 8, 9)
        self.U = (2, 12, 14)
        self.D = (1, 4, 7)
        # Optimal parameters from the report
        self.q = 11.636058
        self.k = 3.057852
        self.alpha = 0.627795
        self.eta = 1.486743
        self.eps_u = 0.997599
        self.eps_d = 0.999944
    
    def test_mc_accuracy(self):
        """Test that mc is within 1% of target."""
        Yu, Yd = compute_yukawas_clockwork(
            self.Q, self.U, self.D,
            self.q, self.k, self.alpha, self.eta, self.eps_u, self.eps_d
        )
        obs = compute_quark_observables(Yu, Yd)
        
        mc_error = abs(obs['mc'] - 1.27) / 1.27
        self.assertLess(mc_error, 0.01, f"mc error {mc_error*100:.2f}% exceeds 1%")
    
    def test_vus_accuracy(self):
        """Test that Vus is within 1% of target."""
        Yu, Yd = compute_yukawas_clockwork(
            self.Q, self.U, self.D,
            self.q, self.k, self.alpha, self.eta, self.eps_u, self.eps_d
        )
        obs = compute_quark_observables(Yu, Yd)
        
        vus_error = abs(obs['Vus'] - 0.225) / 0.225
        self.assertLess(vus_error, 0.01, f"Vus error {vus_error*100:.2f}% exceeds 1%")
    
    def test_vcb_accuracy(self):
        """Test that Vcb is within 1% of target."""
        Yu, Yd = compute_yukawas_clockwork(
            self.Q, self.U, self.D,
            self.q, self.k, self.alpha, self.eta, self.eps_u, self.eps_d
        )
        obs = compute_quark_observables(Yu, Yd)
        
        vcb_error = abs(obs['Vcb'] - 0.042) / 0.042
        self.assertLess(vcb_error, 0.01, f"Vcb error {vcb_error*100:.2f}% exceeds 1%")
    
    def test_vub_accuracy(self):
        """Test that Vub is within 1% of target."""
        Yu, Yd = compute_yukawas_clockwork(
            self.Q, self.U, self.D,
            self.q, self.k, self.alpha, self.eta, self.eps_u, self.eps_d
        )
        obs = compute_quark_observables(Yu, Yd)
        
        vub_error = abs(obs['Vub'] - 0.00382) / 0.00382
        self.assertLess(vub_error, 0.01, f"Vub error {vub_error*100:.2f}% exceeds 1%")
    
    def test_svd_ratio(self):
        """Test that SVD ratio S[0]/S[1] is approximately 136."""
        Yu, Yd = compute_yukawas_clockwork(
            self.Q, self.U, self.D,
            self.q, self.k, self.alpha, self.eta, self.eps_u, self.eps_d
        )
        
        _, Su, _ = np.linalg.svd(Yu)
        ratio = Su[0] / Su[1]
        
        # Should be close to mt/mc = 172.5/1.27 ≈ 135.8
        expected_ratio = 172.5 / 1.27
        ratio_error = abs(ratio - expected_ratio) / expected_ratio
        
        self.assertLess(ratio_error, 0.01, 
                        f"SVD ratio {ratio:.2f} differs from expected {expected_ratio:.2f}")


class TestNumericalStability(unittest.TestCase):
    """Test numerical stability and edge cases."""
    
    def test_large_distance(self):
        """Test kernel with large distances doesn't produce NaN/Inf."""
        result = clockwork_kernel_element(100.0, 0.0, q=10.0, k=1.0, alpha=0.5, eta=0.1, eps=0.5)
        self.assertFalse(np.isnan(result))
        self.assertFalse(np.isinf(result))
    
    def test_small_q(self):
        """Test clockwork with q close to 1."""
        result = clockwork_kernel_element(5.0, 0.0, q=1.01, k=1.0, alpha=0.5, eta=0.1, eps=0.5)
        self.assertFalse(np.isnan(result))
        self.assertFalse(np.isinf(result))
    
    def test_large_q(self):
        """Test clockwork with large q."""
        result = clockwork_kernel_element(5.0, 0.0, q=100.0, k=1.0, alpha=0.5, eta=0.1, eps=0.5)
        self.assertFalse(np.isnan(result))
        self.assertFalse(np.isinf(result))
        # Should be very small but not zero
        self.assertGreater(abs(result), 0)
    
    def test_zero_eps(self):
        """Test with zero interference."""
        result = clockwork_kernel_element(5.0, 0.0, q=3.0, k=1.0, alpha=0.5, eta=0.1, eps=0.0)
        # Should just be the envelope
        expected = 3.0 ** (-5)
        self.assertAlmostEqual(abs(result), expected, places=10)
    
    def test_yukawa_matrix_shape(self):
        """Test that Yukawa matrices have correct shape."""
        Q = (0, 1, 2)
        U = (3, 4, 5)
        D = (6, 7, 8)
        
        Yu, Yd = compute_yukawas_clockwork(Q, U, D, 3.0, 1.0, 0.5, 0.1, 0.5, 0.5)
        
        self.assertEqual(Yu.shape, (3, 3))
        self.assertEqual(Yd.shape, (3, 3))
    
    def test_yukawa_matrix_complex(self):
        """Test that Yukawa matrices are complex."""
        Q = (0, 1, 2)
        U = (3, 4, 5)
        D = (6, 7, 8)
        
        Yu, Yd = compute_yukawas_clockwork(Q, U, D, 3.0, 1.0, 0.5, 0.1, 0.5, 0.5)
        
        self.assertEqual(Yu.dtype, np.complex128)
        self.assertEqual(Yd.dtype, np.complex128)


class TestKernelRegistry(unittest.TestCase):
    """Test the kernel registry is complete and correct."""
    
    def test_all_kernels_registered(self):
        """Test that all 5 kernels are in the registry."""
        expected_kernels = ['gaussian', 'power_law', 'exponential', 'clockwork', 'hybrid']
        for kernel in expected_kernels:
            self.assertIn(kernel, KERNELS, f"Kernel {kernel} not in registry")
    
    def test_kernels_have_required_fields(self):
        """Test that all kernels have required fields."""
        required_fields = ['name', 'formula', 'compute_yukawas', 'params', 'bounds']
        
        for kernel_name, kernel_info in KERNELS.items():
            for field in required_fields:
                self.assertIn(field, kernel_info, 
                             f"Kernel {kernel_name} missing field {field}")
    
    def test_bounds_match_params(self):
        """Test that bounds length matches params length."""
        for kernel_name, kernel_info in KERNELS.items():
            self.assertEqual(len(kernel_info['bounds']), len(kernel_info['params']),
                           f"Kernel {kernel_name}: bounds/params length mismatch")


if __name__ == '__main__':
    unittest.main(verbosity=2)
