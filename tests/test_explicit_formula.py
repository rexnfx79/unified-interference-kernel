"""Tests for explicit_formula module (Path D)."""

import sys

sys.path.insert(0, "../src")

import math

from explicit_formula import (
    RIEMANN_ZERO_IMAG,
    chebyshev_psi,
    psi_from_explicit_formula,
    archimedean_correction,
)


def test_psi_matches_at_medium_x():
    x = 5000.0
    psi = chebyshev_psi(x)
    model = psi_from_explicit_formula(x, RIEMANN_ZERO_IMAG[:40])
    assert abs(psi - model) < 1.0


def test_archimedean_finite():
    assert math.isfinite(archimedean_correction(100.0))
