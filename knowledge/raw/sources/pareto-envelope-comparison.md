# Pareto Envelope Comparison Snapshot

> **Canonical:** `../../data/pareto_envelope_comparison_summary.txt`  
> **Script:** `../../scripts/02_pareto_envelope_comparison.py`

## Test

Generalized envelope \( \exp(-(|d|/\sigma)^p / p) \) for \(p \in \{1, 2, 3\}\) — same geometry/weights, only \(p\) varies.

## Results (from summary file)

| p | Knee found? | Notes |
|---|-------------|-------|
| 1.0 (exponential) | **No** | No Pareto knee detected in scan |
| 2.0 (Gaussian) | **No** | Same — limited scan config |
| 3.0 (super-Gaussian) | **Yes** | CKM loss ≈ 2.32, mc ≈ 5.42 GeV (still far from target 1.27) |

## Verdict

- **Does not** rescue full quark sector — knee at p=3 still has mc ~4× too high
- **Does not** restore parameter universality
- Envelope shape alone is **not** the primary bottleneck; rank/phase structure dominates ([[diagnostics-summary]])

## Caveat

Scan used small geometry/weight grid (3 geometries, 10 weights, 2 seeds) — treat as exploratory, not exhaustive.

See wiki: [[pareto-envelope-comparison]], [[generalized-envelope-kernel]]
