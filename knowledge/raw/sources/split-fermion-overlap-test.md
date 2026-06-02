# Split-Fermion Overlap Test Snapshot

> **Script:** `diagnostics/10_split_fermion_overlap_derivation.py`  
> **Output:** `diagnostics/results/10_split_fermion_overlap.txt`

## Extended test (2026-06-01)

4 geometries from repo optimization / validation suites:

- standard: Q=(0,1,0), U=(0,3,6), D=(0,3,7)
- csv_compact: Q=(0,1,0), U=(0,1,2), D=(0,1,3)
- kernel_comparison: Q=(0,1,3), U=(2,4,5), D=(0,3,6)
- rigorous_validation: Q=(5,7,9), U=(1,10,13), D=(0,3,6)

Each geometry tests **Yu** and **Yd** overlap fit to kernel magnitudes.

Kernel params: σ=4, k=1.4, α=2.5, η=2.0, ε=0.15.

## Expected metrics

- magnitude correlation r > 0.99
- mean phase error < 0.05 rad
- w/σ ratio stable across cases (~0.69 mean)

See wiki: [[split-fermion-overlap-test]], [[derive-interference-kernel-from-overlaps]].
