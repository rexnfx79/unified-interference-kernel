#!/usr/bin/env python3
"""Tests for open-system decoherence module."""

import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from flavor_information import yukawa_density_matrix
from open_system_decoherence import (
    apply_diagonal_decoherence,
    decoherence_rate_from_g_env,
    decoherence_rate_from_eps,
    external_decoherence_parameter,
    open_system_mixing_proxy,
)


def test_decoherence_rate_bounds():
    assert decoherence_rate_from_g_env(0.40) == 0.0
    assert decoherence_rate_from_g_env(0.80) == 1.0
    assert 0.0 <= decoherence_rate_from_eps(0.25) <= 1.0


def test_diagonal_decoherence_trace_one():
    Y = np.diag([1.0, 0.3, 0.01]).astype(complex)
    rho = yukawa_density_matrix(Y)
    rho_open = apply_diagonal_decoherence(rho, 0.5)
    assert abs(np.trace(rho_open) - 1.0) < 1e-10


def test_full_decoherence_diagonal():
    Y = np.array([[0.9, 0.1, 0.05], [0.1, 0.8, 0.1], [0.05, 0.1, 0.7]], dtype=complex)
    rho = yukawa_density_matrix(Y)
    rho_open = apply_diagonal_decoherence(rho, 1.0)
    off = rho_open.copy()
    np.fill_diagonal(off, 0)
    assert np.sum(np.abs(off)) < 1e-10


def test_mixing_proxy_monotone_in_p():
    Y = np.array([[0.9, 0.2, 0.05], [0.2, 0.7, 0.15], [0.05, 0.15, 0.6]], dtype=complex)
    p0 = open_system_mixing_proxy(Y, 0.0)
    p1 = open_system_mixing_proxy(Y, 1.0)
    assert p0 >= p1


def test_external_quark_parameter():
    p = external_decoherence_parameter("quark", eps_u=0.25, eps_d=0.25)
    assert 0.0 <= p <= 1.0


def test_external_neutrino_parameter():
    p = external_decoherence_parameter("neutrino", g_env=0.55)
    assert 0.0 <= p <= 1.0


if __name__ == "__main__":
    test_decoherence_rate_bounds()
    test_diagonal_decoherence_trace_one()
    test_full_decoherence_diagonal()
    test_mixing_proxy_monotone_in_p()
    test_external_quark_parameter()
    test_external_neutrino_parameter()
    print("All test_open_system_decoherence tests passed.")
