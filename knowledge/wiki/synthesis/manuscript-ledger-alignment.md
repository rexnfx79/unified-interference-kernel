---
type: synthesis
title: Manuscript ↔ Ledger Alignment
tags: [meta, flavor, manuscript]
related:
  - proven-vs-conjecture-ledger
  - manuscript-key-results
  - repo-scientific-findings
  - research-strategy
  - survivor-protocol-preregistered
  - interference-kernel-manuscript
status: active
created: 2026-06-02
updated: 2026-06-02i
---

# Manuscript ↔ Ledger Alignment

Maps **manuscript.tex** claims to [[proven-vs-conjecture-ledger]] status. Use when revising abstract, conclusions, or limitations.

**2026-06 integrity pass (2026-06-02i):** Manuscript §Results quark cherry-pick removed (diag 21/27/30 only); §Model regimes relabeled organizational; §Conclusions lead 27.8% joint ν strict; README/SCIENTIFIC_FINDINGS synced; pre-registration = repo-locked protocol note added.

## Claim table

| Manuscript claim (section / line theme) | Ledger status | Repo evidence |
|--------------------------------------|---------------|---------------|
| Single kernel **form** across sectors | **Phenomenological** | [[interference-kernel]], [[projection-regimes]] |
| Universal **parameter** values across sectors | **Refuted** | [[repo-scientific-findings]], `scripts/03_true_transfer_test.py`, diag 19 |
| Three-regime framework **validated** by data | **Refuted** (quarks) / partial (leptons) | manuscript §limitations; 0% quark strict survivors |
| Quark CKM–\(m_c\) Pareto trade-off | **Established** (structural) | [[manuscript-key-results]], `diagnostics/21_quark_phenomenology_holdout.py` |
| Quark sector reproduces PDG precision | **Refuted** | 0% strict survivors; \(m_c\) order-of-magnitude failures |
| Charged lepton 60% survivor rate | **Phenomenological** (legacy full-mass opt) / **1% strict+legacy** (diag 22, 100 geom, train split) | [[analysis-summary]], `diagnostics/22_lepton_phenomenology_sweep.py` |
| Neutrino 45% PMNS survivor rate | **Phenomenological** (legacy 480 geom) / **78.9% strict** (diag 23, 100 geom) | [[analysis-summary]], `diagnostics/23_neutrino_phenomenology_sweep.py` |
| Quark 0% strict survivors (100 geom) | **Established** (structural failure) | `diagnostics/21_quark_phenomenology_holdout.py` (scaled 2026-06-02f) |
| Parameter scale clustering \(\sigma,\alpha\) | **Speculative / artifact** | manuscript §limitations; optimization bias |
| QIT / information-loss sampling narrative | **Refuted** for mechanism | diagnostics 12–19 |
| Exploratory zeta / arithmetic hooks | **Watch only** (Path D) | [[research-strategy]] — no flavor numerology |

## Overclaims to soften (manuscript.tex)

| Location | Issue | Fix pointer |
|----------|-------|-------------|
| Abstract L29 | “coherent phenomenological organization” without quark failure upfront | **Fixed** (2026-06-02f): quark 0% strict lead; dual-protocol survivor rates |
| Introduction L42 | “successfully reproduces experimental observables” | **Fixed**: sector-specific partial success + protocol caveat |
| §Results lepton L326 | 60% survivor without protocol label | **Fixed** (2026-06-02g): legacy vs train/holdout dual-report |
| §Results neutrino L348 | 45% survivor without protocol label | **Fixed** (2026-06-02g): legacy vs diag 23 strict |
| Fig. success_threshold L407 | 60%/39% without protocol | **Fixed** (2026-06-02g): legacy caption + pointer to Discussion |
| §Three-regime L393 | “validates the three-regime framework” | **Fixed**: descriptive only, not validated |
| Title / intro (2026-06-02h) | “Three Projection Regimes” overclaims mechanism | **Fixed**: subtitle + organizational labels; quark 0% lead |
| §Large Mixing L445–455 | Causal bullets quark small / ν large mixing | **Fixed**: removed causal bullets; weak g_env correlation only |
| §Conclusions L478–485 | 60%/39% legacy rates headline | **Fixed**: 0% quark strict, 1% lepton holdout, 78.9% ν PMNS caveat |
| Future work geometry extension | Implies larger grid fixes quarks | **Fixed**: exploratory only, not identified fix |
| §Neutrino–quark unification L455 | “unifies … within a single framework” | **Fixed**: quark failure breaks unification claim |
| Future work L498–502 | UV / statistical validation of “universal scales” | **Fixed**: marked speculative; transfer refuted |
| §Results quark “Robust Fits” (2026-06-02i) | Cherry-picked best geometry contradicts Failure Analysis | **Fixed**: diag 21/27/30 preregistered numbers only |
| §Model L115–121 (2026-06-02i) | “Key insight” / validated regimes | **Fixed**: organizational labels; mechanism not validated |
| §Scope L475 (2026-06-02i) | “successfully organizes” | **Fixed**: partial lepton only; quark failure noted |
| §Conclusions neutrino (2026-06-02i) | 78.9% PMNS headline | **Fixed**: lead 27.8% joint strict; 78.9% PMNS-only sub |
| §Split-fermion L464 (2026-06-02i) | “independent of discrete geometry” | **Fixed**: Pareto persists but no strict match / not geometry-independent |
| §Pre-Registered Falsifiers (2026-06-02i) | External prereg implied | **Fixed**: repo-locked protocol note |
| README / SCIENTIFIC_FINDINGS (2026-06-02i) | 6σ, universal kernel, three-regime mechanism | **Fixed**: ledger pointers; preregistered survivor rates |

## Diagnostics cross-reference

| Diagnostic | Verdict relevant to manuscript |
|------------|-------------------------------|
| `08_controlled_baseline.py` | Gaussian vs clockwork; holdout |
| `09_minimality_ladder.py` | Extra params do not fix holdout |
| `19_fisher_transfer_test.py` | Universal parameters refuted |
| `21_quark_phenomenology_holdout.py` | Train/holdout + CKM–\(m_c\) Pareto (phenomenology tranche) |
| `22_lepton_phenomenology_sweep.py` | 100 geom train/holdout; 1% strict; holdout m_e structural failure |
| `23_neutrino_phenomenology_sweep.py` | 100 geom PMNS; 78.9% strict; g_env≈0.47; bootstrap weak g_env–mixing |
| `24_cross_kernel_paired_lepton_neutrino.py` | Paired Gaussian vs clockwork vs generalized (30 geom; superseded by 26) |
| `25_lepton_mass_pareto.py` | Weighted m_μ–m_e Pareto; holdout tension documented |
| `26_joint_three_sector_cross_kernel.py` | Joint N=100 shared L; paired wins geometry-dependent; no universal envelope |
| `27_quark_joint_loss_holdout.py` | Tier A2: joint 7-obs loss; 2% strict (Gaussian); Pareto persists |
| `28_neutrino_masses_pmns_joint.py` | Tier B2: PMNS + Δm² joint strict 27.8% vs 78.9% PMNS-only |
| `29_quark_geometry_extension.py` | Legacy grid extension; 0% strict; marginal joint-loss gain |
| `30_quark_geometry_followup.py` | Exhaustive legacy re-baseline + shell-5; **0/5759 strict** |
| `31_null_geometry_baseline.py` | Tier A4: shuffled Q + Haar null vs kernel fit |

## Canonical protocol

All survivor rates must cite [[survivor-protocol-preregistered]] (Tier A1). Legacy full-objective / range rates are historical reference only.

## Maintenance

Update when `manuscript.tex` or ledger changes. Honest limitations draft: [[manuscript-honest-limitations-draft]] (raw source).
