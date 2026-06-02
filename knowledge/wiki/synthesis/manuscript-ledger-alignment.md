---
type: synthesis
title: Manuscript ↔ Ledger Alignment
tags: [meta, flavor, manuscript]
related:
  - proven-vs-conjecture-ledger
  - manuscript-key-results
  - repo-scientific-findings
  - research-strategy
  - interference-kernel-manuscript
status: active
created: 2026-06-02
updated: 2026-06-02f
---

# Manuscript ↔ Ledger Alignment

Maps **manuscript.tex** claims to [[proven-vs-conjecture-ledger]] status. Use when revising abstract, conclusions, or limitations.

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
| §Three-regime L393 | “validates the three-regime framework” | **Fixed**: descriptive only, not validated |
| §Neutrino–quark unification L455 | “unifies … within a single framework” | **Fixed**: quark failure breaks unification claim |
| Future work L498–502 | UV / statistical validation of “universal scales” | **Fixed**: marked speculative; transfer refuted |

## Diagnostics cross-reference

| Diagnostic | Verdict relevant to manuscript |
|------------|-------------------------------|
| `08_controlled_baseline.py` | Gaussian vs clockwork; holdout |
| `09_minimality_ladder.py` | Extra params do not fix holdout |
| `19_fisher_transfer_test.py` | Universal parameters refuted |
| `21_quark_phenomenology_holdout.py` | Train/holdout + CKM–\(m_c\) Pareto (phenomenology tranche) |
| `22_lepton_phenomenology_sweep.py` | 100 geom train/holdout; 1% strict; holdout m_e structural failure |
| `23_neutrino_phenomenology_sweep.py` | 100 geom PMNS; 78.9% strict; g_env≈0.47; bootstrap weak g_env–mixing |
| `24_cross_kernel_paired_lepton_neutrino.py` | Paired Gaussian vs clockwork vs generalized (30 geom) |
| `25_lepton_mass_pareto.py` | Weighted m_μ–m_e Pareto; holdout tension documented |

## Maintenance

Update when `manuscript.tex` or ledger changes. Honest limitations draft: [[manuscript-honest-limitations-draft]] (raw source).
