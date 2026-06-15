---
type: synthesis
title: Tangent Research Seeds (Post-Closure)
tags: [meta, flavor, strategy, neutrino]
related:
  - future-work
  - research-strategy
  - proven-vs-conjecture-ledger
  - manuscript-ledger-alignment
  - neutrino-loss-landscape-n1
  - neutrino-haar-null-n2
  - neutrino-pmns-cp-n5
  - neutrino-holdout-geometry-n3
  - chiral-projection-thesis
  - repo-scientific-findings
  - multi-sided-bridge-framework
status: active
created: 2026-06-15
updated: 2026-06-15
---

# Tangent Research Seeds (Post-Closure)

**Context:** In-repo flavor **mechanism search is closed** (2026-06-15). This page distills **novel, tested findings** from diagnostics 21–46 that can seed **tangent** work — not reopenings of dead bridges.

**Thesis seed:**

> The interference kernel is a **sector-local readout family** over discrete internal geometry. Neutrino partial success comes from **shallow joint landscapes** and **PDG-structured PMNS** (N1, N2), not geometry classifiers (N3) or universal parameters. Quark failure is **structural** (CKM–\(m_c\) Pareto, effective rank \(\approx 1\)). CP is **orthogonal** to mass+mixing in the joint neutrino objective (N5). Tangent study should extend **texture rank and CP targets** where landscapes are easy (neutrinos), and **effective rank / EFT operators** where landscapes are rugged (quarks).

## What is novel (repo-tested)

| # | Finding | Evidence | Pointer |
|---|---------|----------|---------|
| 1 | **Sector-asymmetric optimization geography** | Quark joint loss \(\sim\)70× ν; ruggedness \(\sim\)75×; 4/5 metrics \(p<0.05\) — **caveat:** N=50, different loss definitions (scale confound) | [[neutrino-loss-landscape-n1]], diag 39 |
| 2 | **Structured PMNS, not Haar** *(descriptive)* | 3/3 KS reject Haar; PDG dist median **0.24** vs Haar **2.93** — **expected** under PDG-targeted loss; not mechanistic | [[neutrino-haar-null-n2]], diag 41 |
| 3 | **CP decoupled from joint ν objective** | Median \(|\Delta\delta_{\mathrm{PMNS}}| \approx 3.5\) rad under joint loss | [[neutrino-pmns-cp-n5]], diag 46 |
| 4 | **Structural CKM–\(m_c\) Pareto** | Nondominated frontier; **0/5759** strict | diag 21, 30 |
| 5 | **Form shared, parameters not** | Transfer loss 797.5 frozen vs 779.0 free | [[repo-scientific-findings]], diag 19 |
| 6 | **Geometry does not predict success OOS** | Holdout AUC **0.53** (N3); in-sample CV **0.66** only | [[neutrino-holdout-geometry-n3]], [[neutrino-geometry-predictor-n4]] |
| 7 | **Post-fit \(\rho_Y\) information ≠ regimes** | max \|r\| \(\approx 0.05\) | diag 12–19 |
| 8 | **Mirror portals do not rescue flavor** | Diags 42–44 fail pre-registered bars | [[chiral-projection-formalization-program]] |

## What is not novel (do not oversell)

| Item | Note |
|------|------|
| Mirror matter / weak chirality | BSM literature |
| Split-fermion overlaps | Established EFT story |
| Zeta → 3×3 flavor | Dead numerology |
| “Neutrinos easier than quarks” qualitatively | Known; **quantified** landscape is new |

## Ranked tangent seeds

### Seed A — Neutrino CP extension (N5) — **conditional reopen**

**From:** CP orthogonal to joint mass+PMNS fit (N5 — strongest mechanistic tangent).

**Tangent:** Minimal CP extension — extra phase DOF or CP-weighted term in loss (not blind grids).

**Falsifier:** Median \(|\Delta\delta_{\mathrm{PMNS}}| < 1\) rad **and** joint strict rate **≥ 22/100** attempted (no degradation vs diag 28).

**Blockers:** [[future-work]] Tier 1 forbids CP sweeps without new pre-registration; rank-2 quark ansätze already falsified (Tier 2) — any ν rank-2 needs separate justification.

**Adversarial review:** [[adversarial-review-tangent-research-seeds]]

### Seed B — What pins \(\theta_{13}\)? (N2) — **deprioritized**

**From:** Post-fit PMNS rejects Haar; \(\theta_{13}\) clusters at PDG — largely **circular** under PDG loss (see adversarial review).

**Tangent:** Ablation on phase DOF — only if pre-registered with holdout.

**Falsifier:** Removing an identified DOF restores Haar-like \(\theta_{13}\) spread **OOS**.

**Weakness:** N3 killed geometry predictors; in-sample ablation is storytelling.

### Seed C — Quark effective-rank / Pareto anatomy (structural negative)

**From:** CKM–\(m_c\) Pareto + effective rank \(\approx 1\) + 0% strict.

**Tangent:** Analytic characterization of why rank-1 envelope × interference cannot simultaneously hit \(m_c\) and third-generation CKM.

**Falsifier:** Minimal rank-2 extension moves **off** the same Pareto knee with holdout gain \(>20\%\) (Tier 2 already failed — any reopen needs **new** ansatz).

### Seed D — Sector bundle readouts (positive framing of transfer failure)

**From:** Parameters do not transfer; shared geometry (diag 26) is possible.

**Tangent:** UV-flavored story: shared internal **geometry**, sector-local **measurement maps** \(\{R_s\}\) — not one universal \((\sigma,k,\alpha,\eta)\).

**Falsifier:** Any constrained bundle reduces total free params **and** beats sector-independent holdout.

### Seed E — Landscape-aware optimization (N1)

**From:** Neutrino basins shallow/smooth; quark rugged.

**Tangent:** Sector-specific optimizers, warm starts, or ansätze matched to Hessian statistics — meta-optimization research.

**Falsifier:** Same optimizer + same ansatz → landscape metrics indistinguishable.

### Seed F — Methodology export (meta) — **co-top exportable seed**

**From:** Pre-registered holdout, honest denominators (strict/solved/attempted), adversarial closure.

**Tangent:** Template for other EFT fit papers — **most transferable** repo output; not physics mechanism.

## Closed — do not seed from these

| Dead track | Why |
|------------|-----|
| Geometry catalogs / ML on \((L,N)\) | N3 holdout fail |
| QIT on \(\rho_Y\), regime entropy | Diag 12–19 |
| Mirror portals at kernel level | Diags 42–44 |
| Universal kernel parameters | Transfer refuted |
| Zeta / 3×3 GUE → Yukawa | [[why-not-zeta-flavor-numerology]] |
| Chiral projection → flavor | [[adversarial-review-chiral-projection-thesis]] |

## Philosophical watch (external to repo)

[[chiral-projection-thesis]] — oriented reconstruction, mirror sector — **lab portals**, not kernel extensions.

## Recommended tangents (post-adversarial review)

| Priority | Seed | Action |
|----------|------|--------|
| **1** | **F** Methodology | Export protocol + manuscript ledger alignment |
| **2** | **C** Quark Pareto anatomy | Analytic negative / discussion section |
| **3** | **A** ν CP | **Pre-register** before any code; not default reopen |
| — | B, D, E | Deprioritize or philosophical only |

Full critique: [[adversarial-review-tangent-research-seeds]].

## Related

[[future-work]], [[research-strategy]], [[manuscript-ledger-alignment]], [[diagnostics-summary]], [[adversarial-review-tangent-research-seeds]]
