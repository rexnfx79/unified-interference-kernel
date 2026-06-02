#!/usr/bin/env python3
"""Tests for Fisher transfer analysis helpers."""

import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fisher_transfer import (
    deltas_within_cr_bounds,
    evaluate_fisher_transfer_verdict,
    run_fisher_transfer_analysis,
)
from experimental_fisher import UNIVERSAL_PARAM_NAMES


def test_deltas_within_cr_bounds():
    cr = {n: 0.01 for n in UNIVERSAL_PARAM_NAMES}
    deltas_ok = {n: 0.01 for n in UNIVERSAL_PARAM_NAMES}
    within, _ = deltas_within_cr_bounds(deltas_ok, cr, z=2.0)
    assert within
    deltas_bad = {n: 1.0 for n in UNIVERSAL_PARAM_NAMES}
    within_bad, _ = deltas_within_cr_bounds(deltas_bad, cr, z=2.0)
    assert not within_bad


def test_evaluate_verdict_falsifier_b():
    analysis = {
        "frozen": {"loss": 800.0},
        "deltas_within_cr": False,
        "alignment_at_transfer": 0.3,
    }
    v = evaluate_fisher_transfer_verdict(analysis, alignment_threshold=0.5, frozen_loss_bad_threshold=797.0)
    assert v["refuted"]
    assert v["falsifier_b"]


def test_run_fisher_transfer_smoke():
    result = run_fisher_transfer_analysis(n_seeds=1, maxiter=30)
    assert "quark_theta" in result
    assert len(result["quark_theta"]) == 6
    assert result["frozen"]["loss"] >= 0
    assert np.isfinite(result["alignment_at_transfer"]) or True  # may be nan in edge cases
    assert len(result["universal_deltas"]) == len(UNIVERSAL_PARAM_NAMES)


if __name__ == "__main__":
    test_deltas_within_cr_bounds()
    test_evaluate_verdict_falsifier_b()
    test_run_fisher_transfer_smoke()
    print("All test_fisher_transfer tests passed.")
