"""
Tests for charged lepton observables extraction.
"""

import sys
sys.path.insert(0, '../src')

import numpy as np
from observables import (
    compute_lepton_observables,
    compute_lepton_loss,
    LEPTON_TARGETS,
)


def test_lepton_mass_roundtrip():
    """Planted singular values reproduce PDG masses after tau anchoring."""
    mt, mm, me = LEPTON_TARGETS['m_tau'], LEPTON_TARGETS['m_mu'], LEPTON_TARGETS['m_e']
    S = np.array([1.0, mm / mt, me / mt])
    Ye = np.diag(S)
    obs = compute_lepton_observables(Ye)
    for key in ['m_e', 'm_mu', 'm_tau']:
        rel = abs(obs[key] - LEPTON_TARGETS[key]) / LEPTON_TARGETS[key]
        assert rel < 1e-6, f"{key} relative error {rel}"
    print("✓ Lepton mass roundtrip test passed")


def test_lepton_loss_perfect_match():
    mt, mm, me = LEPTON_TARGETS['m_tau'], LEPTON_TARGETS['m_mu'], LEPTON_TARGETS['m_e']
    obs = compute_lepton_observables(np.diag([1.0, mm / mt, me / mt]))
    loss = compute_lepton_loss(obs)
    assert loss < 1e-12
    print("✓ Lepton loss function test passed")


if __name__ == '__main__':
    test_lepton_mass_roundtrip()
    test_lepton_loss_perfect_match()
    print("\n✓ All lepton observable tests passed!")
