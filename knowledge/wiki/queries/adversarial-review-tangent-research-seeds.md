---
type: query
title: Adversarial Review — Tangent Research Seeds
tags: [meta, flavor, strategy]
related:
  - tangent-research-seeds
  - future-work
  - research-strategy
  - neutrino-haar-null-n2
  - neutrino-loss-landscape-n1
  - plausibility-register
status: open
created: 2026-06-15
updated: 2026-06-15
---

# Adversarial Review — Tangent Research Seeds

**Scope:** [[tangent-research-seeds]] — eight "novel" findings and ranked seeds A–F.

**Verdict:** **Useful as a constraint map, overstated as a research program.** Keep findings 1, 3, 4, 5, 6, 8; **downgrade** 2 and 7; **demote** seeds A–E vs **elevate** F; **block** Seed A as "highest leverage" without explicit program reopen + new pre-registration.

---

## Executive summary

| Item | Verdict |
|------|---------|
| Thesis seed (sector-local readout family) | **Keep** — best honest framing |
| Finding 2 (anti-Haar PMNS) | **Downgrade** — largely circular (PDG-targeted loss) |
| Finding 1 (landscape asymmetry) | **Keep with caveat** — N=50, loss-scale confound |
| Seed A (ν CP extension) | **Conditional** — conflicts with Tier 1 "no CP sweeps" unless re-registered |
| Seed B (θ₁₃ pinning) | **Weak** — N3 killed geometry; ablation is post-hoc |
| Seed C (quark Pareto anatomy) | **Keep** — theory/negative-result only |
| Seed D (sector bundle) | **Philosophy** until bundle beats holdout |
| Seed E (landscape optimizers) | **Meta** — not flavor physics |
| Seed F (methodology) | **Strongest exportable seed** |

---

## 1. Attacks on "novel findings"

### 1.1 Finding 2 (N2) — circularity

**Attack:** Optimizer minimizes PMNS loss against **PDG targets**. Rejecting Haar and clustering near PDG is **expected** under any loss \(\sum ( \theta_i - \theta_i^{\text{PDG}})^2\). This is not evidence of a **texture mechanism** or non-anarchy in nature — it is evidence the fit **works as designed**.

**Verdict:** **Descriptive, not mechanistic.** Useful to **refute loose "ν anarchy" claims for this ansatz**; **do not** seed mechanism work from N2 alone. Pair with Seed B only if ablations are **pre-registered** and geometry-agnostic.

### 1.2 Finding 1 (N1) — scale confound

**Attack:** Quark joint loss (diag 27) and neutrino joint loss (diag 28) are **different objectives** on different observables with different typical magnitudes. Ratio "~70×" mixes **physics difficulty** with **loss definition**. Ruggedness comparison is stronger but N=**50** geometries only.

**Verdict:** **Keep** as exploratory sector comparison; **do not** cite 70× as a physical constant. Replicate with **normalized** loss or common observable subset before seeding Seed E.

### 1.3 Finding 3 (N5) — solid

**Attack:** SVD phase conventions could rotate \(\delta_{\mathrm{PMNS}}\) arbitrarily?

**Defense:** Same pipeline used for angle fits; CP misalignment is **consistent** across solved geoms. Median \(|\Delta\delta| \approx 3.5\) rad is not convention noise.

**Verdict:** **Strongest novel finding** for tangent work — CP is not entailed by current joint objective.

### 1.4 Finding 4 — established negative (high value)

**Attack:** None substantive — 0/5759 + Pareto is the paper's structural quark contribution.

**Verdict:** **Keep** — export as **theorem-style negative** for the kernel class.

### 1.5 Finding 5 — replication of transfer refutation

**Attack:** Already in [[repo-scientific-findings]] before tangent doc.

**Verdict:** **Keep** in seed list but not "new" — supports Seed D narrative only.

### 1.6 Finding 6 — solid kill

**Verdict:** **Keep** — correctly kills geometry-catalog tangents.

### 1.7 Finding 7 — already dead in register

**Verdict:** **Redundant** as a "seed" — documents a **closure**, not a forward direction.

### 1.8 Finding 8 — solid kill

**Verdict:** **Keep** — closes P-series at kernel level.

---

## 2. Attacks on ranked seeds

### Seed A — CP extension (**overrecommended**)

| Attack | Detail |
|--------|--------|
| Protocol conflict | [[future-work]] Tier 1: "do not scale full CP optimization sweeps without new falsifier" |
| Rank-2 mention | Tier 2 **failed** rank-2 quark ansätze; ν rank-2 untested but not justified by one N5 point |
| Thin base rate | 22/100 joint strict — CP extension may trade mixing for CP with no net gain |
| Falsifier too weak | "≥ 20/100 strict" allows degradation from 22 → 20 |

**Revised bar:** Pre-register **CP-only** extension: \(|\Delta\delta| < 1\) rad **and** joint strict **≥ 22/100** (no degradation) **and** PMNS loss median not worse than baseline.

**Verdict:** **Conditional reopen** — not default "highest leverage" until pre-registration written.

### Seed B — θ₁₃ pinning (**weak**)

**Attack:** N3 shows geometry does not predict success OOS. Ablation on \(\Phi_{ij}\) without holdout is **in-sample storytelling**. θ₁₃ is the **easiest** of three angles to hit (smallest target).

**Verdict:** **Deprioritize** below Seed C and F.

### Seed C — Pareto anatomy (**keep**)

**Attack:** Tier 2 already tested rank-2 — falsifier text admits this.

**Verdict:** **Best quark tangent** if framed as **analytic negative result** (paper/theory note), not new DE sweeps.

### Seed D — sector bundle (**untested philosophy**)

**Attack:** "Shared geometry, split parameters" is exactly what **transfer test refuted** unless \(R_s\) are **more structured** than free sector fits. Diag 26 shared L did not yield universality.

**Verdict:** **Philosophical** until a **parametrically smaller** bundle beats sector-independent holdout (same bar as diag 43 L3).

### Seed E — landscape optimizers (**meta**)

**Attack:** Better optimizers do not change **identifiability** of the kernel class. N1 explains **why** sectors differ under DE, not **what** physics to add.

**Verdict:** **Computer-science tangent**, not repo flavor mission.

### Seed F — methodology (**underweighted**)

**Attack:** None — pre-registered holdout, honest denominators, adversarial closure is the **most transferable** output.

**Verdict:** **Elevate to co-equal** with Seed C for external impact.

---

## 3. Mission-creep blocks

| Creep | Status |
|-------|--------|
| Seed A without pre-registration | **Block** — violates Tier 1 spirit |
| N2 → "texture mechanism proved" | **Block** |
| N1 70× → physical coupling ratio | **Block** |
| Seed D → UV completion claim | **Block** without bundle test |
| Reopen quark rank-2 DE sweeps | **Block** — Tier 2 falsified |

---

## 4. Revised seed ranking

| Rank | Seed | Status |
|------|------|--------|
| 1 | **F** Methodology export | **Pursue** (paper + protocol doc) |
| 2 | **C** Quark Pareto / rank anatomy | **Pursue** (analytic / discussion) |
| 3 | **A** ν CP extension | **Watch** — pre-register first |
| 4 | **D** Sector bundle | **Philosophical** |
| 5 | **B** θ₁₃ ablation | **Deprioritize** |
| 6 | **E** Landscape meta-opt | **Deprioritize** (out of scope) |

---

## 5. Revised thesis seed (surviving form)

> The interference kernel is a **sector-local readout family**. **Exportable positives:** pre-registered phenomenology methodology (F), structural quark CKM–\(m_c\) negative (C), landscape asymmetry as **diagnostic** not mechanism (N1, qualified), CP–mixing **decoupling** in joint ν objective (N5). **Do not** build tangents on anti-Haar (N2) or geometry prediction (N3/N4) without new falsifiers. **Program reopen** only via pre-registered Seed A or an explicit non-flavor Path D track.

---

## 6. Required edits to [[tangent-research-seeds]]

1. Downgrade Finding 2 to **"descriptive / circular under PDG loss"**.
2. Add N=50 and loss-scale caveat to Finding 1.
3. Demote Seed A; add Tier 1 protocol conflict note.
4. Elevate Seed F; co-recommend with Seed C for publication.
5. Tighten Seed A falsifier: no strict-rate degradation vs 22/100.

## Related

[[tangent-research-seeds]], [[future-work]], [[survivor-protocol-preregistered]]
