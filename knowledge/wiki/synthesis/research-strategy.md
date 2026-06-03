---
type: synthesis
title: Research Strategy (User Decisions)
tags: [meta, strategy]
related:
  - multi-sided-bridge-framework
  - plausibility-register
  - manuscript-ledger-alignment
  - survivor-protocol-preregistered
  - qed-qm-information
  - hilbert-polya-conjecture
  - explicit-formula-primes-zeros
  - information-measure-for-projection-regimes
  - fisher-transfer-universality-test
status: active
created: 2026-06-01
updated: 2026-06-02k
---

# Research Strategy (User Decisions)

Living record of **strategic choices** for unified-interference-kernel exploration. Code-only; SM flavor physics only.

## Future work (canonical)

Full tiered plan: **[[future-work]]** (2026-06-02k). Phenomenology tranche **21–31 complete**; combinatorial quark search **closed**.

## Active priorities (2026-06-02k)

| Priority | Action | Rationale |
|----------|--------|-----------|
| **1** | **Tier 0** — publication package per [[future-work]] | Ledger-aligned manuscript + artifact paths |
| **2** | **Tier 1** — post phase-fix audit (small N) + CP observables | Confirm refutations; then CP/Jarlskog |
| **3** | ~~**Tier 2** diag 32~~ | **Complete — falsified** (0% strict; holdout worse than Gaussian) |
| **4** | **Tier 5** — conjecture↔physics ([[conjecture-to-physics-avenues]]) | HP / explicit formula / Jacobi inverse; **not** flavor numerology |
| **5** | **Tier 4** — repo hygiene | Parallel |

## Closed — quark search dead ends (do not fund without new falsifier)

| Approach | Evidence | Pointer |
|----------|----------|---------|
| Legacy geometry extension (coords 5–8, 86k pool) | 0% strict; N=100 sample 0% strict (diag 29 rerun) | `29`, `30` |
| Exhaustive max_coord=5 re-baseline + shell-5 | 0/5759 strict; shell joint min 3.06 vs 1k 5.01 | `30` |
| Independent \(Q_u, Q_d\) / minimality relaxations | All levels **worsen holdout** (+110–456%) | `09`, [[MINIMALITY_REPORT]] |
| Envelope sweep only (G / clockwork / \(p\)) | 0% strict split (diag 21); joint 2% on **phenomenology** triples only (diag 27) | `21`, `27` |
| Shared-\(Q\) as bottleneck | Refuted | `09` |
| More DE seeds / CSV-era optimizer parity | Re-baseline shifts bests; does not create strict survivors | `30` |

Re-open **only** with: new matrix-level ansatz (rank > 1), pre-registered holdout rule, and [[quark-geometry-conventions]] convention stated explicitly.

## Primary track (2026-06-02c): **Phenomenology**

| Track | Role |
|-------|------|
| **Phenomenology** | **Primary** — per-sector kernel fits, train/holdout honesty, envelope/kernel comparisons, manuscript–ledger alignment |
| **Path A** (QIT→flavor) | **Deprioritized** — diagnostics 12–19 refuted; no mechanism claims |
| **Path D** (arithmetic/spectral) | **Watch only** — no flavor numerology; no promotion without non–ρ_Y hook |

**Deprioritized / dead:** split-fermion→kernel derivation; universal parameters; zeta→flavor; QIT sampling narrative.

## Phenomenology worklist

| Item | Script / doc | Status |
|------|--------------|--------|
| Unified observables (`observables.py`) | CKM, PMNS, lepton masses | ✓ PMNS + lepton in library |
| Quark train/holdout + Pareto | `diagnostics/21_quark_phenomenology_holdout.py` | ✓ scaled 100 geom |
| Manuscript ↔ ledger | [[manuscript-ledger-alignment]] | ✓ |
| **Survivor protocol (Tier A1)** | [[survivor-protocol-preregistered]] | ✓ 2026-06-02h |
| Lepton sector phenomenology sweep | `diagnostics/22_lepton_phenomenology_sweep.py` | ✓ scaled 100 geom |
| Neutrino PMNS sweep | `diagnostics/23_neutrino_phenomenology_sweep.py` | ✓ scaled 100 geom |
| Cross-kernel lepton+ν paired | `diagnostics/24_cross_kernel_paired_lepton_neutrino.py` | ✓ (30 geom; superseded by 26) |
| Joint 3-sector cross-kernel | `diagnostics/26_joint_three_sector_cross_kernel.py` | ✓ N=100 (45 min) |
| Lepton m_μ–m_e Pareto | `diagnostics/25_lepton_mass_pareto.py` | ✓ |
| Refactor transfer script → observables.py | `scripts/03_true_transfer_test.py` | ✓ |

## Goal (revised)

Document **what the kernel can and cannot fit** with pre-registered splits and holdouts. Do **not** claim fundamental mechanism or cross-sector universality.

| Accept | Reject |
|--------|--------|
| Sector-specific fits with holdout reporting | “Universal kernel” parameter claims |
| Structural trade-offs (CKM–\(m_c\)) with numbers | Three-regime “validation” while quarks fail |
| Negative results as **refuted** | QIT→flavor mechanism without new falsifier |

## Scope

- **Physics:** Standard Model flavor only (quark, charged lepton, neutrino).
- **Path D:** Decoupled from flavor; watch [[hilbert-polya-conjecture]], [[explicit-formula-primes-zeros]].

## Budget & resources

- **Long** — full sweeps (`diagnostics/21_*`, envelope scans).
- **Code only** — `src/observables.py`, `src/kernel*.py`, `src/alternative_kernels.py`, `diagnostics/`.

## Completed diagnostics (mechanism — closed)

| Script | Result |
|--------|--------|
| `12`–`16` | ρ_Y / QFI / decoherence vs mixing — **refuted** |
| `17` | Experimental Fisher alignment 0.46 — **dead for mechanism** |
| `18` | Open-system pooled — **refuted** |
| `19` | Fisher transfer — **refuted** (frozen loss 805.8) |
| `20` | Collider Fisher — **out of scope** |

## Phenomenology diagnostics

| Script | Purpose |
|--------|---------|
| `08_controlled_baseline.py` | Gaussian vs clockwork + holdout |
| `09_minimality_ladder.py` | Extra params vs holdout |
| `21_quark_phenomenology_holdout.py` | 100 geom train/holdout; 0% strict; CKM–\(m_c\) Pareto |
| `22_lepton_phenomenology_sweep.py` | 100 geom train/holdout; 1% strict |
| `23_neutrino_phenomenology_sweep.py` | 100 geom PMNS; 78.9% strict |
| `24_cross_kernel_paired_lepton_neutrino.py` | Cross-kernel paired lepton+ν (30 geom) |
| `26_joint_three_sector_cross_kernel.py` | Joint corpus N=100; shared L; 3-sector paired |
| `25_lepton_mass_pareto.py` | m_μ–m_e weighted Pareto |
| `27_quark_joint_loss_holdout.py` | Tier A2: joint 7-obs quark loss vs split Pareto |
| `28_neutrino_masses_pmns_joint.py` | Tier B2: PMNS + Δm² joint vs 78.9% PMNS-only |
| `29_quark_geometry_extension.py` | Legacy grid extension; 0% strict |
| `30_quark_geometry_followup.py` | 989+4770 legacy re-baseline; 0/5759 strict |
| `31_null_geometry_baseline.py` | Tier A4: null/scrambled geometry baseline |
| `32_quark_tier2_ansatz.py` | Tier 2: rank2 sum, FN texture, dual-phase, power-law |

## Phenomenology tranche — completed

All items through diag **31**, geometry closure, and phase-fix are **done**. See [[future-work]] for next tiers.

## Path D — watch only (unchanged)

Promotion requires non–ρ_Y Path A hook — **not found** (diag 17–20). See prior criteria table in [[proven-vs-conjecture-ledger]].

## Explicit do-nots

- Zeta zeros → Yukawa / CKM
- Universal kernel **parameters**
- QIT→flavor mechanism from sector-local statistics
- Manuscript overclaims without [[manuscript-ledger-alignment]] check
- Further **legacy quark geometry** combinatorics (29–30 closed)
- **Split \(Q_u,Q_d\)** or minimality ladder re-runs without new ansatz + falsifier
- CP / Jarlskog targets until observables audit passes (post phase-fix)

## Success criteria (phenomenology tranche)

1. [[research-strategy]] + `knowledge/purpose.md` aligned ✓
2. `observables.py` train/holdout + lepton helpers ✓
3. Diagnostic 21 results on disk ✓
4. [[manuscript-ledger-alignment]] + honest limitations draft ✓
5. QA + wiki lint pass ✓
