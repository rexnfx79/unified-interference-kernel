#!/usr/bin/env python3
"""Tests for QED-derived Yukawa information measures."""

import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from flavor_information import yukawa_density_matrix, compute_yukawa_information
from qed_information import (
    coherence_l1_norm,
    off_diagonal_to_diagonal_ratio,
    quantum_fisher_trace,
    symmetric_logarithmic_derivative,
    distinguishability_from_uniform,
    compute_qed_yukawa_information,
    uniform_yukawa_density,
)


def test_density_matrix_normalized():
    Y = np.diag([1.0, 0.3, 0.01]).astype(complex)
    rho = yukawa_density_matrix(Y)
    assert np.allclose(np.trace(rho), 1.0)
    assert np.allclose(rho, rho.conj().T)


def test_coherence_diagonal_dominant():
    Y = np.diag([2.0, 1.0, 0.5]).astype(complex)
    rho = yukawa_density_matrix(Y)
    assert coherence_l1_norm(rho) < 1e-10
    assert off_diagonal_to_diagonal_ratio(rho) < 1e-10


def test_qfi_nonnegative_perturbation():
    Y = np.array([[0.9, 0.1, 0.05], [0.1, 0.8, 0.1], [0.05, 0.1, 0.7]], dtype=complex)
    rho = yukawa_density_matrix(Y)
    eps = 1e-7
    Yp = Y.copy()
    Yp[0, 1] += eps
    Ym = Y.copy()
    Ym[0, 1] -= eps
    drho = (yukawa_density_matrix(Yp) - yukawa_density_matrix(Ym)) / (2 * eps)
    F = quantum_fisher_trace(rho, drho)
    assert F >= 0.0
    assert np.isfinite(F)


def test_sld_hermitian():
    Y = np.eye(3, dtype=complex)
    rho = yukawa_density_matrix(Y)
    drho = 0.01 * (np.random.randn(3, 3) + 1j * np.random.randn(3, 3))
    drho = 0.5 * (drho + drho.conj().T)
    L = symmetric_logarithmic_derivative(rho, drho)
    assert np.allclose(L, L.conj().T)


def test_distinguishability_uniform_zero_for_uniform():
    Y = np.ones((3, 3), dtype=complex) / np.sqrt(3)
    d = distinguishability_from_uniform(Y)
    assert d >= 0.0
    assert d < 1e-6


def test_compute_qed_keys():
    Y = np.diag([1.0, 0.2, 0.01]).astype(complex)
    out = compute_qed_yukawa_information(Y)
    for key in (
        "qfi_mean_elements",
        "qfi_mean_singular_values",
        "coherence_l1",
        "off_diagonal_ratio",
        "distinguishability_uniform",
        "cramer_rao_from_qfi_elements",
    ):
        assert key in out
        assert np.isfinite(out[key]) or out[key] == float("inf")


def test_include_qed_flag():
    Y = np.diag([1.0, 0.2, 0.01]).astype(complex)
    base = compute_yukawa_information(Y, include_qed=False)
    full = compute_yukawa_information(Y, include_qed=True)
    assert "qfi_mean_elements" not in base
    assert "qfi_mean_elements" in full


def test_uniform_reference_trace_one():
    rho_u = uniform_yukawa_density(3)
    assert np.allclose(np.trace(rho_u), 1.0)


if __name__ == "__main__":
    test_density_matrix_normalized()
    test_coherence_diagonal_dominant()
    test_qfi_nonnegative_perturbation()
    test_sld_hermitian()
    test_distinguishability_uniform_zero_for_uniform()
    test_compute_qed_keys()
    test_include_qed_flag()
    test_uniform_reference_trace_one()
    print("All test_qed_information tests passed.")
