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
  - survivor-protocol-preregistered
status: active
created: 2026-06-15
updated: 2026-06-15
---

# Tangent Research Seeds (Post-Closure)

**Context:** In-repo flavor **mechanism search is closed** (2026-06-15). This page distills **novel, tested findings** from diagnostics 21–46 that can seed **tangent** work — not reopenings of dead bridges.

**Verdict:** **Useful as a constraint map, overstated as a research program.** Keep findings 1, 3, 4, 5, 6, 8; downgrade 2 and 7; demote seeds A–E vs elevate F; block Seed A as default reopen without pre-registration.

**Surviving thesis:**

> The interference kernel is a **sector-local readout family**. **Exportable positives:** pre-registered phenomenology methodology (F), structural quark CKM–\(m_c\) negative (C), landscape asymmetry as **diagnostic** not mechanism (N1, qualified), CP–mixing **decoupling** in joint ν objective (N5). **Do not** build tangents on anti-Haar (N2) or geometry prediction (N3/N4) without new falsifiers. **Program reopen** only via pre-registered Seed A or an explicit non-flavor Path D track.

## What is novel (repo-tested)

| # | Finding | Evidence | Pointer | Review |
|---|---------|----------|---------|--------|
| 1 | **Sector-asymmetric optimization geography** | Quark joint loss \(\sim\)70× ν; ruggedness \(\sim\)75×; 4/5 metrics \(p<0.05\) | [[neutrino-loss-landscape-n1]], diag 39 | **Keep with caveat** — N=50; different loss definitions (scale confound); do not cite 70× as physical constant |
| 2 | **Structured PMNS, not Haar** | 3/3 KS reject Haar; PDG dist median **0.24** vs Haar **2.93** | [[neutrino-haar-null-n2]], diag 41 | **Downgrade** — expected under PDG-targeted loss; descriptive, not mechanistic |
| 3 | **CP decoupled from joint ν objective** | Median \(|\Delta\delta_{\mathrm{PMNS}}| \approx 3.5\) rad under joint loss | [[neutrino-pmns-cp-n5]], diag 46 | **Strongest novel finding** — not SVD convention noise |
| 4 | **Structural CKM–\(m_c\) Pareto** | Nondominated frontier; **0/5759** strict | diag 21, 30 | **Keep** — theorem-style negative for kernel class |
| 5 | **Form shared, parameters not** | Transfer loss 797.5 frozen vs 779.0 free | [[repo-scientific-findings]], diag 19 | **Keep** — not new; supports Seed D narrative only |
| 6 | **Geometry does not predict success OOS** | Holdout AUC **0.53** (N3); in-sample CV **0.66** only | [[neutrino-holdout-geometry-n3]], [[neutrino-geometry-predictor-n4]] | **Solid kill** — blocks geometry-catalog tangents |
| 7 | **Post-fit \(\rho_Y\) information ≠ regimes** | max \|r\| \(\approx 0.05\) | diag 12–19 | **Redundant** — closure doc, not forward seed |
| 8 | **Mirror portals do not rescue flavor** | Diags 42–44 fail pre-registered bars | [[chiral-projection-formalization-program]] | **Solid kill** — P-series dead at kernel level |

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

**Tangent:** Minimal CP extension — extra phase DOF or CP-weighted term in loss (not blind grids). First gate: [[neutrino-cp-invariant-n6]]; first fixed-weight test: [[neutrino-cp-weighted-objective-n7]].

**Falsifier:** Median signed-\(J_{\mathrm{PMNS}}\) relative error < 50% **and** joint strict rate **≥ 22/100** attempted (no degradation vs diag 28) **and** PMNS/mass loss medians not worse than baseline.

**Blockers:** [[future-work]] Tier 1 forbids CP sweeps without new pre-registration; rank-2 quark ansätze already falsified (Tier 2) — any ν rank-2 needs separate justification.

**Review:** Overrecommended as default reopen. Thin base (22/100 strict); CP extension may trade mixing for CP with no net gain. **Watch** — pre-register first. N6 shows raw \(\delta_{\mathrm{PMNS}}\) is rephase-unstable and the invariant signed-\(J\) audit still misses the bar. N7 fixes signed \(J_{\mathrm{PMNS}}\) but drops joint strict to 20/100, so the CP extension is a trade-off, not a clean pass.

### Seed B — What pins \(\theta_{13}\)? (N2) — **deprioritized**

**From:** Post-fit PMNS rejects Haar; \(\theta_{13}\) clusters at PDG — largely **circular** under PDG loss.

**Tangent:** Ablation on phase DOF — only if pre-registered with holdout.

**Falsifier:** Removing an identified DOF restores Haar-like \(\theta_{13}\) spread **OOS**.

**Review:** N3 killed geometry predictors; in-sample ablation is storytelling. θ₁₃ is the easiest angle to hit. Deprioritize below C and F.

### Seed C — Quark effective-rank / Pareto anatomy (structural negative) — **co-top**

**From:** CKM–\(m_c\) Pareto + effective rank \(\approx 1\) + 0% strict.

**Tangent:** Analytic characterization of why rank-1 envelope × interference cannot simultaneously hit \(m_c\) and third-generation CKM.

**Falsifier:** Minimal rank-2 extension moves **off** the same Pareto knee with holdout gain \(>20\%\) (Tier 2 already failed — any reopen needs **new** ansatz).

**Review:** Best quark tangent if framed as analytic negative result (paper/theory note), not new DE sweeps.

### Seed D — Sector bundle readouts — **philosophical**

**From:** Parameters do not transfer; shared geometry (diag 26) is possible.

**Tangent:** UV-flavored story: shared internal **geometry**, sector-local **measurement maps** \(\{R_s\}\) — not one universal \((\sigma,k,\alpha,\eta)\).

**Falsifier:** Any constrained bundle reduces total free params **and** beats sector-independent holdout (same bar as diag 43 L3).

**Review:** "Shared geometry, split parameters" is what transfer test refuted unless \(R_s\) are more structured than free sector fits. Diag 26 shared L did not yield universality.

### Seed E — Landscape-aware optimization (N1) — **deprioritized**

**From:** Neutrino basins shallow/smooth; quark rugged.

**Tangent:** Sector-specific optimizers, warm starts, or ansätze matched to Hessian statistics — meta-optimization research.

**Falsifier:** Same optimizer + same ansatz → landscape metrics indistinguishable.

**Review:** Better optimizers do not change kernel-class identifiability. Computer-science tangent, not repo flavor mission. Replicate N1 with normalized loss before pursuing.

### Seed F — Methodology export (meta) — **co-top**

**From:** Pre-registered holdout, honest denominators (strict/solved/attempted), adversarial closure.

**Tangent:** Template for other EFT fit papers — **most transferable** repo output; not physics mechanism.

**Review:** Strongest exportable seed — elevate for external impact alongside Seed C.

## Mission-creep blocks

| Creep | Status |
|-------|--------|
| Seed A without pre-registration | **Block** — violates Tier 1 spirit |
| N2 → "texture mechanism proved" | **Block** |
| N1 70× → physical coupling ratio | **Block** |
| Seed D → UV completion claim | **Block** without bundle test |
| Reopen quark rank-2 DE sweeps | **Block** — Tier 2 falsified |

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

## Recommended tangents

| Priority | Seed | Action |
|----------|------|--------|
| **1** | **F** Methodology | [[phenomenology-methodology-export]] + `BUILD_MANUSCRIPT.md` + bundle script |
| **2** | **C** Quark Pareto anatomy | Analytic negative / discussion section |
| **3** | **A** ν CP | **Pre-register** before any code; not default reopen |
| 4 | **D** Sector bundle | Philosophical until bundle beats holdout |
| 5 | **B** θ₁₃ ablation | Deprioritize |
| 6 | **E** Landscape meta-opt | Out of scope |

## Related

[[future-work]], [[research-strategy]], [[manuscript-ledger-alignment]], [[diagnostics-summary]], [[survivor-protocol-preregistered]]
