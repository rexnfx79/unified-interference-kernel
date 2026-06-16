---
type: synthesis
title: Phenomenology Methodology Export (Seed F)
tags: [meta, flavor, manuscript, strategy]
related:
  - survivor-protocol-preregistered
  - manuscript-ledger-alignment
  - tangent-research-seeds
status: active
created: 2026-06-15
updated: 2026-06-15
---

# Phenomenology Methodology Export (Seed F)

**Purpose:** Portable template for EFT / kernel-fit papers — pre-registered holdout, honest denominators, adversarial closure. **Not** a physics mechanism claim.

**Canonical sources:** [[survivor-protocol-preregistered]], [[manuscript-ledger-alignment]], `manuscript.tex`.

## Core protocol (copy for other projects)

### 1. Pre-register before large runs

| Element | This repo |
|---------|-----------|
| Falsifier script | Committed under `diagnostics/` before result ingestion |
| Holdout rule | Train vs holdout observables fixed in `src/observables.py` |
| Accept bar | Sector-specific (e.g. median holdout improves >20% vs baseline) |
| Geometry seed | Fixed per diagnostic (e.g. quark **21021**, ν **23023**) |
| Wiki lock | [[survivor-protocol-preregistered]] |

**Rule:** No headline rate changes without updating the protocol page and [[manuscript-ledger-alignment]].

### 2. Honest denominators

Always report **three counts** where applicable:

| Count | Meaning |
|-------|---------|
| **Attempted** | Geometries drawn (e.g. N=100) |
| **Solved** | DE converged with physical readout (e.g. θ₂₃ > 0) |
| **Strict** | All sector observables within PDG-relative tolerances |

**Example (neutrino):** 22/100 joint strict attempted; 22/79 among solved (27.8%); 71/100 PMNS-only attempted — **different objectives**, not interchangeable headlines.

### 3. Lead with failures

Abstract and conclusions order:

1. Quark **0% strict** (0/5759 exhaustive)
2. Lepton **1% strict** (holdout m_e)
3. Neutrino **qualified** partial success with objective caveat

### 4. Dual-report legacy vs strict

Legacy full-objective rates (~60% lepton, ~45% ν) are **historical only**. Every figure/table caption must point to strict protocol or label "legacy."

### 5. Adversarial closure

Before claiming a tangent direction:

| Step | Action |
|------|--------|
| Inventory | List what diagnostics already falsified |
| Attack | Circular metrics, scale confounds, in-sample ML |
| Block | Mission-creep table ([[tangent-research-seeds]]) |
| Reopen bar | New pre-registered falsifier only |

### 6. Frozen artifacts + reproduce

```bash
./scripts/reproduce_phenomenology_tranche.sh   # Unix
./scripts/reproduce_phenomenology_tranche.ps1  # Windows check
```

Headline reports: `diagnostics/results/21_*` through `46_*` (see script manifest).

### 7. Manuscript ↔ ledger sync

On every TeX edit:

1. Check claim against [[manuscript-ledger-alignment]] table
2. Update [[proven-vs-conjecture-ledger]] if status changes
3. Rebuild PDF — see `BUILD_MANUSCRIPT.md`

## Headline numbers (2026-06-15)

| Sector | Strict rate | Diagnostic | Denominator note |
|--------|-------------|------------|------------------|
| Quark | **0%** | 21, 30 | 0/100 gaussian; 0/5759 exhaustive |
| Quark joint | **2%** | 27 | Phenomenology triples; Pareto persists |
| Lepton | **1%** | 22 | Holdout m_e structural failure |
| ν PMNS-only | **71%** attempted | 23 | 71/100; 78.9% of 90 solved |
| ν joint | **22%** attempted | 28 | 22/100; 27.8% of 79 solved |
| N3 geometry predictor | **FAIL** | 45 | Holdout AUC 0.53 |
| N5 CP (descriptive) | **misaligned** | 46 | Median \|Δδ\| ≈ 3.5 rad |

## Submission bundle

```bash
./scripts/bundle_submission_artifacts.sh   # creates submission_bundle/
```

Includes: `manuscript.tex`, frozen `diagnostics/results/*.txt`, protocol wiki paths, reproduce scripts.

## What not to export

| Anti-pattern | Why |
|--------------|-----|
| 70× landscape ratio as physics | Loss-scale confound (N1) |
| Anti-Haar as mechanism | Circular under PDG loss (N2) |
| Geometry ML without holdout | N3 refuted |
| Universal parameters | Transfer refuted |
| Three-regime "validation" | Quark refutation |

## Related

[[tangent-research-seeds]], [[future-work]], [[multi-sided-bridge-framework]]
