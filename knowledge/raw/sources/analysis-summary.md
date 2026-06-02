# Analysis Summary Snapshot

> **Canonical:** `../../ANALYSIS_SUMMARY.md`

## Survivor rates

### Legacy CSV archives (range-based survivors)

| Sector | Geometries | Survivors | Rate | Best loss |
|--------|------------|-----------|------|-----------|
| Quark | 1000 | 0 | 0% | 0.155 |
| Charged lepton | 100 | 60 | 60% | 2.45×10⁻¹¹ |
| Neutrino | 480 | 216 | 45% | 1.36×10⁻⁸ |

### Diagnostic 21–25 (scaled phenomenology, 2026-06-02e)

| Sector | Script | N geom | Strict survivors | Legacy (range) | Notes |
|--------|--------|--------|------------------|------------------|-------|
| Quark | `21_quark_phenomenology_holdout.py` | 12 | **0%** | — | Holdout masses do not track train |
| Charged lepton | `22_lepton_phenomenology_sweep.py` | **100** | **1%** (1/100) | **1%** | Train m_μ+m_τ perfect 95%; holdout m_e median loss **32**; archived 60% used full 3-mass objective |
| Neutrino | `23_neutrino_phenomenology_sweep.py` | **100** (90 solved) | **78.9%** (71/90) | **78.9%** | g_env mean **0.47**; \|r(g_env, θ₂₃)\|=0.07 (95% CI includes 0); archived 45% on 480 geom |
| Lepton+ν cross-kernel | `24_cross_kernel_paired_lepton_neutrino.py` | 30 paired | — | — | Clockwork partial train wins; holdout m_e poor all kernels |
| Lepton Pareto | `25_lepton_mass_pareto.py` | 24 | — | — | Weighted train/holdout: sparse front (1 nondom. point); holdout-only split fails |

## Z-scores (survivors)

- Lepton masses: Z < 0.85
- PMNS angles: Z(θ₁₂)≈0.62, Z(θ₂₃)≈0.26, Z(θ₁₃)≈0.05

## Adversarial reading

- Quark "0% survivors" contradicts soft "challenging but working" tone in summary — wiki verdict: **refuted validation** at PDG precision ([[manuscript-key-results]])
- σ clustering ~4–5 across sectors: see [[similar-fitted-scales-vs-transfer]] — not transfer universality

## Pipeline

PMNS + lepton train/holdout in `src/observables.py` (diag 22 split: train m_μ+m_τ, holdout m_e).

See wiki: [[analysis-summary]], [[proven-vs-conjecture-ledger]]
