---
type: source
title: Split-Fermion Overlap Derivation Test
tags: [flavor, qm]
related:
  - derive-interference-kernel-from-overlaps
  - split-fermion-overlaps
  - interference-kernel
sources:
  - raw/sources/split-fermion-overlap-test.md
status: established
created: 2026-06-01
updated: 2026-06-01
---

# Split-Fermion Overlap Derivation Test

Extended numeric test of [[derive-interference-kernel-from-overlaps]].

## Result (2026-06-01 extended run)

4 geometries × Yu and Yd — see `diagnostics/results/10_split_fermion_overlap.txt`.

| Geometry | Yu r | Yd r | Yu w/σ | Yd w/σ |
|----------|------|------|--------|--------|
| standard | 0.99996 | 0.999998 | 0.694 | 0.703 |
| csv_compact | **0.967** | 0.999997 | **1.093** | 0.706 |
| kernel_comparison | 0.999 | 0.99996 | 0.663 | 0.694 |
| rigorous_validation | 0.998 | 0.99986 | 0.656 | 0.684 |

**Phase:** 0 rad error all cases.

**w/σ stability:** mean 0.737, rel spread **18%** — **not stable** across all geometries (csv_compact Yu outlier). Standard + spread geometries cluster near **w ≈ 0.69 σ**.

## Script

`diagnostics/10_split_fermion_overlap_derivation.py`

## Next

Derive w from potential V(y); explain csv_compact Yu mismatch.
