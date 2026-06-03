"""Tier-2 kernel smoke tests."""

import sys

sys.path.insert(0, "src")

import numpy as np
from alternative_kernels import (
    compute_yukawas_rank2_clockwork_sum,
    KERNELS,
    TIER2_QUARK_KERNELS,
)


def test_rank2_clockwork_sum_shapes():
    Q = (0, 1, 2)
    U = (0, 2, 4)
    D = (1, 3, 5)
    Yu, Yd = compute_yukawas_rank2_clockwork_sum(
        Q, U, D,
        2.0, 1.0, 0.5, 2.0, 0.1, 0.1,
        3.0, 0.5, 1.0, 3.0, 0.2, 0.2,
        0.6,
    )
    assert Yu.shape == (3, 3)
    assert Yd.shape == (3, 3)
    assert np.all(np.isfinite(np.abs(Yu)))
    assert np.all(np.isfinite(np.abs(Yd)))


def test_tier2_registry():
    for k in TIER2_QUARK_KERNELS:
        assert k in KERNELS
        assert len(KERNELS[k]["bounds"]) == len(KERNELS[k]["params"])
