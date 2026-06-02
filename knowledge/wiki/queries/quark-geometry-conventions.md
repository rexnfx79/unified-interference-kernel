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
| Larger **legacy** grid fixes strict quarks | Diagnostic **29** — 100 extension geoms (max_coord 6–8) | **0% strict**; best joint 4.28 vs legacy CSV 4.90 — marginal loss gain, no PDG match |
| Exhaustive legacy re-baseline + shell-5 | Diagnostic **30** — unified DE on 1k + 5k shell | **Smoke only** (10+10 geoms); full run deferred (~6000 geom). Preliminary 0/20 strict |
| Random **15-wide** triples fix strict quarks | Diagnostics **21, 27** — N=100 | **0% strict** (split); **2% strict** (joint, Gaussian) |
| Kernel beats null/scrambled geometry | Diagnostic **31** — shuffled Q + Haar Yu/Yd | See `diagnostics/results/31_*` |

## Diagnostic 29 headline numbers

- Extension pool: 86,808 geoms; sampled N=100 (seed 29029)
- Solved 68/100; **0% strict** survivors
- Best extension joint loss **4.28** (beats legacy CSV best **4.90**) but still no simultaneous PDG match

## See also

- `diagnostics/results/29_quark_geometry_extension.txt`
- `diagnostics/results/30_quark_geometry_followup.txt` (smoke; not exhaustive)
- `data/quark_geometry_followup_bests.json` (stored θ* for smoke bests)
