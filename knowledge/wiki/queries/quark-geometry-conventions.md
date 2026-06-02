---
type: query
title: Quark geometry conventions (legacy vs phenomenology)
tags: [flavor, diagnostics]
related:
  - survivor-protocol-preregistered
  - yukawa-observables-pipeline
status: established
created: 2026-06-02
updated: 2026-06-02i
---

# Quark geometry conventions (legacy vs phenomenology)

Two **non-equivalent** discrete geometry spaces appear in this repo. Do not compare results across them without translation.

## Legacy enumeration (scripts/01, data/quark_results.csv, diagnostics 29–30)

- **Q:** `(q1, q2, 0)` with `q1 < q2`
- **U, D:** strictly increasing triples `(u1, u2, u3)`, `(d1, d2, d3)`
- **max_coord=5** in `generate_geometries(5)` means coordinate **values 0..4** → exactly **1000** geometries
- Extension shell: coordinate value **5** first appears when enumerating with `max_coord=6` (+5000 new triples)

## Phenomenology sampler (diagnostics 21, 27, 31)

- **Q, U, D:** independent sorted 3-tuples from `range(15)` (rejection sampling)
- No trailing `0` on Q; not a subset of the legacy 1000-grid

## Falsifiers

| Claim | Test | Result (2026-06-02) |
|-------|------|---------------------|
| Exhaustive legacy + shell-5 fixes strict quarks | Diagnostic **30** — 989 + 4770 geoms, unified DE | **0/5759 strict**; Wilson 95% UB ≈ **0.07%** |
| Shell-5 improves joint compromise only | Diag 30 joint min shell **3.06** vs 1k **5.01** | Yes; best \(m_c \approx 4.2\) GeV — still not PDG |
| Extension sample (diag 29) | N=100 from 86k pool (rerun, penalty fix) | **92/100** solved, **0% strict**; best joint **4.28** |
| Random **15-wide** triples | Diagnostics **21, 27** | **0% strict** (split); **2% strict** (joint, Gaussian) |
| Kernel beats null/scrambled geometry | Diagnostic **31** | See `diagnostics/results/31_*` |

## Diagnostic 30 headline numbers

- Unified protocol: DE maxiter=120, popsize=12, 4 seeds, joint 7-obs loss
- Re-baselined 1k best: `Q=(2,4)`, `U=(0,1,4)`, `D=(2,3,4)`, joint **5.01** (legacy CSV champion re-opts to **6.29** under same protocol)
- Shell-5 best: `Q=(1,2)`, `U=(2,4,5)`, `D=(2,3,4)`, joint **3.06**, \(m_c \approx 4.24\) GeV
- θ* stored in `data/quark_geometry_followup_bests.json`; reproducibility diff **0**

## See also

- `diagnostics/results/30_quark_geometry_followup.txt`
- `data/quark_geometry_followup_baseline.csv`, `quark_geometry_followup_shell5.csv`
- `diagnostics/results/29_quark_geometry_extension.txt` (superseded sample)
