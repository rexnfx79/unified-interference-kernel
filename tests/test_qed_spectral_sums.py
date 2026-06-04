#!/usr/bin/env python3
"""Sanity tests for standard QED spectral sum helpers."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qed_spectral_sums import (
    SCHWINGER_SUM,
    ZETA2,
    euler_product_zeta,
    integer_partial_sum_zeta,
    schwinger_g2_series_integer,
)


def test_schwinger_series_converges():
    s = schwinger_g2_series_integer(10_000)
    assert abs(s - SCHWINGER_SUM) < 1e-4


def test_zeta2_integer_partial():
    z = integer_partial_sum_zeta(2.0, 50_000)
    assert abs(z - ZETA2) < 1e-3


def test_euler_product_matches_zeta2():
    z_sum = integer_partial_sum_zeta(2.0, 30_000)
    z_prod = euler_product_zeta(2.0, 30_000)
    assert abs(z_sum - z_prod) < 0.05
