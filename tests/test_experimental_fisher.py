#!/usr/bin/env python3
"""Tests for experimental Fisher / PDG Jacobian module."""

import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from experimental_fisher import (
    QUARK_PARAM_NAMES,
    fisher_information_matrix,
    fisher_scalar_summaries,
    numerical_jacobian,
    compute_sector_experimental_fisher,
    eigenvector_alignment,
    align_fisher_subspaces,
    lepton_observables_from_ye,
)
from kernel import compute_yukawa_matrix


def test_fisher_symmetric_psd():
    def mu_fn(theta):
        return np.array([theta[0] ** 2, theta[1], theta[0] * theta[1]])

    theta = np.array([2.0, 3.0])
    J = numerical_jacobian(mu_fn, theta, eps=1e-6)
    F = fisher_information_matrix(J, ["a", "b", "c"])
    F = 0.5 * (F + F.T)
    eigvals = np.linalg.eigvalsh(F)
    assert np.all(eigvals >= -1e-10)
    assert F.shape == (2, 2)


def test_fisher_summaries_finite():
    F = np.diag([10.0, 5.0, 0.1])
    s = fisher_scalar_summaries(F)
    assert np.isfinite(s["max_eigenvalue"])
    assert s["effective_rank"] >= 1.0
    assert s["logdet_fisher"] > 0


def test_quark_sector_fisher():
    geom = {"name": "t", "Q": (0, 1, 0), "U": (0, 3, 6), "D": (0, 3, 7)}
    theta = np.array([4.0, 1.4, 2.5, 3.0, 0.25, 0.25])
    res = compute_sector_experimental_fisher("quark", geom, theta)
    assert res["fisher"].shape == (len(QUARK_PARAM_NAMES), len(QUARK_PARAM_NAMES))
    assert len(res["principal_eigenvector"]) == len(QUARK_PARAM_NAMES)
    assert res["summaries"]["max_eigenvalue"] >= 0


def test_lepton_log_masses():
    Ye = compute_yukawa_matrix((0, 1, 0), (0, 3, 6), 4.0, 1.75, 2.5, 3.7, 0.41)
    obs = lepton_observables_from_ye(Ye)
    assert "log_m_e" in obs
    assert np.isfinite(obs["log_m_tau"])


def test_eigenvector_alignment_identical():
    v = np.array([1.0, 2.0, 3.0])
    assert abs(eigenvector_alignment(v, v) - 1.0) < 1e-10


def test_align_shared_params():
    v_a = np.array([1.0, 0.0, 2.0, 0.5])
    v_b = np.array([1.0, 0.0, 2.0])
    names_a = ["sigma", "k", "alpha", "eta"]
    names_b = ["sigma", "k", "alpha"]
    a = align_fisher_subspaces(v_a, v_b, names_a, names_b)
    assert np.isfinite(a)
    assert 0.0 <= a <= 1.0


if __name__ == "__main__":
    test_fisher_symmetric_psd()
    test_fisher_summaries_finite()
    test_quark_sector_fisher()
    test_lepton_log_masses()
    test_eigenvector_alignment_identical()
    test_align_shared_params()
    print("All test_experimental_fisher tests passed.")
