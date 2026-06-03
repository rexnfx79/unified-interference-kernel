---
type: synthesis
title: Pre-Registered Survivor Protocol (Tier A1)
tags: [meta, flavor, manuscript]
related:
  - manuscript-ledger-alignment
  - research-strategy
  - diagnostics-summary
  - yukawa-observables-pipeline
  - projection-regimes
status: established
created: 2026-06-02
updated: 2026-06-02
---

# Pre-Registered Survivor Protocol (Tier A1)

**Single canonical protocol** for reporting kernel fit success across fermion sectors. All new diagnostics (21–28) and manuscript survivor claims must cite this page—not legacy full-objective or range-based rates alone.

## Primary protocol (strict PDG-relative)

A geometry **strict survivor** passes when **every listed observable** for that sector lies within the pre-registered **relative tolerance** vs PDG 2024 targets in `src/observables.py`.

### Strict relative tolerances

| Sector | Observable | Target (PDG 2024) | Relative tol. | Source |
|--------|------------|-------------------|---------------|--------|
| Quark | `mc` | 1.27 GeV | 0.30 | diag 21 `STRICT_TOLERANCES` |
| Quark | `Vus`, `Vcb`, `Vub` | 0.225, 0.04182, 0.00382 | 0.20, 0.30, 0.50 | diag 21 |
| Quark | `mu`, `md`, `ms` | 0.00216, 0.00467, 0.093 GeV | 0.50 each | diag 21 |
| Lepton | `m_e`, `m_mu`, `m_tau` | 0.000511, 0.1057, 1.777 GeV | 0.20, 0.10, 0.05 | diag 22 |
| Neutrino (PMNS) | `theta12`, `theta23`, `theta13` | 0.5903, 0.7850, 0.1490 rad | 0.15, 0.15, 0.20 | diag 23 |
| Neutrino (mass) | `dm21`, `dm31` | 7.53×10⁻⁵, 2.453×10⁻³ eV² | 0.30 each | diag 28 (joint falsifier) |

**Survivor rate** = (strict survivors) / (geometries with at least one converged seed), best seed per geometry.

### Train / holdout splits (optimization objective)

Optimization uses **training targets only**; holdout observables are evaluated post hoc. Strict survivor checks **all** sector observables regardless of split.

| Sector | Optimized (train) | Holdout (eval only) | Diagnostic |
|--------|-------------------|---------------------|------------|
| Quark (split) | `mc`, `Vus`, `Vcb` | `mu`, `md`, `ms`, `Vub` | 21 |
| Quark (joint falsifier A2) | all 7 quark observables via `L_mass + 5 L_CKM` | — (same 7 for strict) | 27 |
| Lepton | `m_mu`, `m_tau` (tau scale-anchored) | `m_e` | 22 |
| Neutrino (PMNS-only) | PMNS angles (`compute_pmns_loss`) | — | 23 |
| Neutrino (joint falsifier B2) | `L_mass + 5 L_PMNS` on Δm² + angles | — | 28 |

Definitions: `src/observables.py` — `TRAINING_TARGETS`, `HOLDOUT_TARGETS`, `LEPTON_TRAINING_TARGETS`, `LEPTON_HOLDOUT_TARGETS`.

### Sweep geometry and optimizer settings

Match diagnostics **21–23** and **26–28**:

| Parameter | Value |
|-----------|-------|
| Geometries per sector | **N = 100** (unique triples from 15×15 grid; rejection sampling) |
| Seeds per geometry | **4** |
| Geometry seeds | quark **21021**, lepton **22022**, neutrino **23023**, joint **26026** |
| Optimizer | `scipy.optimize.differential_evolution` |
| `maxiter` | 120 |
| `popsize` | 12 |
| `tol` | 1e-6 |
| `mutation` | (0.5, 1.0) |
| `recombination` | 0.7 |
| `polish` | False |
| DE seed | `seed + geom_idx * 100` per attempt |

**Kernels (quark, Tier A1):** Gaussian, clockwork, generalized envelope at **p ∈ {1.5, 2.0, 3.0}** — bounds from `alternative_kernels.KERNELS` and `kernel_generalized` (diag 21).

**Kernels (quark, Tier 2 — diag 32):** `alternative_kernels.TIER2_QUARK_KERNELS`: `gaussian` (baseline), `rank2_clockwork_sum`, `clockwork_dual_phase`, `fn_texture`, `fn_texture_split`, `power_law`. Geometries: `generate_quark_geometries`, seed **32032**, N=100. Holdout accept: median holdout improves **>20%** vs Gaussian (diag 09 rule). P2.3 scheme/RGE readout deferred.

**Kernels (lepton / neutrino):** Gaussian; neutrino adds **g_env ∈ [0.45, 0.75]** on effective σ for Y_ν.

Smoke runs (`N=10`) allowed before full N=100 when runtime exceeds ~60 min.

## Legacy protocols — historical reference only

Do **not** use these as primary success metrics in abstract, conclusions, or figures without the strict protocol label.

| Legacy metric | Definition | Typical rate | Why demoted |
|---------------|------------|--------------|-------------|
| Full-objective survivor | All sector observables in optimization loss | lepton ~60%, neutrino ~45% (480 geom archive) | Inflates success; no holdout honesty |
| Range-based survivor | Observable in open interval (e.g. `0.09 < m_mu < 0.12`) | `scripts/04_analyze_results.py` | Not PDG-relative; asymmetric bands |
| Fig. `success_threshold` | Legacy full-objective vs tolerance sweep | 60% / 39% at 10% | Caption now points to diag 21–23 strict rates |

Legacy helpers: `phenomenology_utils.LEGACY_*_RANGES`, `check_legacy_lepton`, `check_legacy_neutrino`.

## Manuscript figure / claim mapping

| Manuscript artifact | Legacy reading | Canonical (this protocol) |
|---------------------|----------------|---------------------------|
| Abstract survivor rates | 60% lepton, 45% neutrino | **0% quark strict**; **1% lepton strict** (holdout m_e); **78.9% neutrino PMNS strict** (100 geom) |
| Fig. `success_threshold` (L407) | 60% / 39% at 10% tol | Legacy caption; Discussion cites diag 21–23 |
| §Results lepton (L326) | 60% legacy | Dual-report: legacy + 1% train/holdout strict |
| §Results neutrino (L348) | 45% legacy | Dual-report: legacy + 78.9% PMNS strict |
| §Conclusions enum (L480–485) | 60% / 39% headline | Lead quark 0%, lepton m_e failure, neutrino PMNS-only caveat |
| Three-regime validation | Regime labels imply mechanism | **Organizational labels only** — not validated (quark refutation) |

## Falsifiers (Tier A2 / B2)

| Tier | Script | Question |
|------|--------|------------|
| A2 | `diagnostics/27_quark_joint_loss_holdout.py` | Is CKM–m_c Pareto a train/holdout split artifact? Joint 7-obs loss vs diag 21 |
| B2 | `diagnostics/28_neutrino_masses_pmns_joint.py` | Does joint PMNS + Δm² objective collapse 78.9% PMNS-only strict rate? |
| A4 | `diagnostics/31_null_geometry_baseline.py` | Do kernel fits beat shuffled-Q / Haar-random Yukawa null baselines? |

Results: `diagnostics/results/27_*`, `28_*`, `31_*`; summarized in [[diagnostics-summary]] and [[research-strategy]].

## Reporting rule

When citing survivor rates in wiki or manuscript:

1. State **protocol** (strict PDG-relative vs legacy).
2. State **N geometries**, **seeds**, and **objective** (train/holdout vs joint).
3. Lead with **quark 0% strict** before any lepton/neutrino partial success.
