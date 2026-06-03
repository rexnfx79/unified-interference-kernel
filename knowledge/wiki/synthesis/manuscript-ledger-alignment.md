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
  - future-work
status: active
created: 2026-06-02
updated: 2026-06-02o
---

# Manuscript ↔ Ledger Alignment

Maps **manuscript.tex** claims to [[proven-vs-conjecture-ledger]] status. Use when revising abstract, conclusions, or limitations.

**Tier 0 pass (2026-06-02o):** Abstract neutrino headline = **27.8% joint** (diag 28) with **78.9% PMNS-only** (diag 23) qualified; quark **0/5759** (diag 30); Further Study lists diag **32–33** closed; reproduce via `scripts/reproduce_phenomenology_tranche.sh`.

## Claim table

| Manuscript claim (section / line theme) | Ledger status | Repo evidence |
|--------------------------------------|---------------|---------------|
| Single kernel **form** across sectors | **Phenomenological** | [[interference-kernel]], [[projection-regimes]] |
| Universal **parameter** values across sectors | **Refuted** | [[repo-scientific-findings]], `scripts/03_true_transfer_test.py`, diag 19 |
| Three-regime framework **validated** by data | **Refuted** (quarks) / partial (leptons) | manuscript §limitations; 0% quark strict survivors |
| Quark CKM–\(m_c\) Pareto trade-off | **Established** (structural) | [[manuscript-key-results]], `diagnostics/21_quark_phenomenology_holdout.py` |
| Quark sector reproduces PDG precision | **Refuted** | 0% strict survivors; 0/5759 exhaustive (diag 30); \(m_c\) failures |
| Charged lepton 60% survivor rate | **Historical** (legacy) / **1% strict** (diag 22) | [[analysis-summary]], `diagnostics/22_lepton_phenomenology_sweep.py` |
| Neutrino 45% PMNS survivor rate | **Historical** (legacy) / **78.9% PMNS-only strict** (diag 23) | `diagnostics/23_neutrino_phenomenology_sweep.py` |
| Neutrino **27.8% joint** strict (PMNS + Δm²) | **Established** (headline) | `diagnostics/28_neutrino_masses_pmns_joint.py` |
| Quark 0% strict survivors (100 geom) | **Established** (structural failure) | diag 21; diag 27 joint 2% on phenomenology triples only |
| Parameter scale clustering \(\sigma,\alpha\) | **Speculative / artifact** | manuscript §limitations |
| QIT / information-loss sampling narrative | **Refuted** for mechanism | diagnostics 12–19 |
| Rank-2 / FN texture quark ansätze | **Refuted** | diag 32 — 0% strict; holdout worse |
| Split-fermion → kernel derivation | **Refuted** (mechanism) | diag 33 — geometry does not predict \(w/\sigma\) |
| Exploratory zeta / arithmetic hooks | **Watch only** (Path D) | [[research-strategy]], [[adversarial-review-tier5-trace-formula]] |

## Overclaims to soften (manuscript.tex)

All items below marked **Fixed** through 2026-06-02o unless noted.

| Location | Issue | Status |
|----------|-------|--------|
| Abstract neutrino rates | 78.9% without joint 27.8% lead | **Fixed** (2026-06-02o) |
| §Further Study rank-2 ansatz | Implied still testable after diag 32 | **Fixed** (2026-06-02o) — closed |
| §Further Study SVD re-audit | Stale “repair” wording | **Fixed** (2026-06-02o) — repair done; optional small audit |
| §Conclusions quark count | 100-geom only | **Fixed** (2026-06-02o) — 0/5759 cited |

Historical fixes (2026-06-02i–k): quark cherry-pick, three-regime validation, legacy survivor captions, pre-registration wording — see git history.

## Diagnostics cross-reference

| Diagnostic | Verdict relevant to manuscript |
|------------|-------------------------------|
| `21` | 0% strict holdout; CKM–\(m_c\) Pareto |
| `22` | 1% strict lepton; holdout \(m_e\) |
| `23` | 78.9% PMNS-only strict |
| `28` | **27.8%** joint strict (headline ν) |
| `27` | 2% strict joint quark (phenomenology triples) |
| `30` | **0/5759** strict geometry closure |
| `32` | Tier-2 ansätze falsified |
| `33` | Split-fermion / Path D — no mechanism hook |

## Canonical protocol

All survivor rates cite [[survivor-protocol-preregistered]] (Tier A1). Legacy rates are historical only.

## Reproduce

```bash
./scripts/reproduce_phenomenology_tranche.sh
```

## Maintenance

Update when `manuscript.tex` or ledger changes.
