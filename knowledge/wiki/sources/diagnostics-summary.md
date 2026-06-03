---
type: source
title: Diagnostics Summary (Repo)
tags: [flavor]
related:
  - proven-vs-conjecture-ledger
  - plausibility-register
  - repo-scientific-findings
  - interference-kernel
sources:
  - raw/sources/diagnostics-summary.md
authors: [Alexander Seto]
year: 2026
status: established
created: 2026-06-01
updated: 2026-06-02m
---

# Diagnostics Summary (Repo)

Synthesis of `diagnostics/` reports (2026 ingest).

## Headline verdicts

| Result | Status |
|--------|--------|
| Code/math correct | **established** (56/56 tests) |
| Gaussian kernel full quarks | **refuted** (structural CKM–\(m_c\) trade-off) |
| Clockwork partial quark fit | **phenomenological** (\(m_c\), \(|V_{us}|\) OK; light quarks, CP fail) |
| Shared-\(Q\) bottleneck | **refuted** (minimality ladder) |
| Parameter universality | **refuted** (transfer test) |
| Three-regime validation | **refuted** for quarks; partial lepton/neutrino fits |

## Phenomenology tranche (21–28)

| Script | Scaled result |
|--------|---------------|
| `21` | 100 geom: **0% strict** all kernels; CKM–\(m_c\) Pareto (corr≈0); G/C paired 53/38/9 train wins |
| `22` | 100 geom: 1% strict/legacy (train split); holdout m_e fails |
| `23` | 100 geom: 78.9% strict PMNS; g_env≈0.47 |
| `24` | Cross-kernel paired lepton+ν (30 geom; superseded by 26) |
| `25` | Lepton m_μ–m_e weighted Pareto (N=100: corr **0.31**, 7 nondominated pts; N=24 snapshot corr **0.58**, 1 pt) |
| `26` | Joint 3-sector N=100 shared L; see paired wins below |
| `27` | Tier A2: joint 7-obs quark loss — Gaussian **2.0%** strict (2/100), gp2 **1.4%**; Pareto corr≈0.1; holdout median **0.69** (G) vs diag 21 **32.8** |
| `28` | Tier B2: joint PMNS+Δm² — **27.8%** strict PMNS (22/79 solved) vs diag 23 **78.9%**; joint strict = PMNS strict |
| `29` | Legacy extension pool N=100 (seed 29029): **92/100** solved, **0% strict**; best joint **4.28** vs CSV **4.90** (diag 30 shell-5 **3.06** on exhaustive legacy) |
| `30` | **5759** legacy geoms (989+4770 solved): **0% strict**; shell-5 joint min **3.06** vs 1k **5.01**; θ* in `quark_geometry_followup_bests.json` |
| `31` | Tier A4: null baseline N=30 — kernel **0% strict**; Haar train median **~360** vs kernel **~2.0**; shuffled Q similar to kernel |
| `32` | Tier 2: rank2 sum, FN texture, dual-phase, power-law — **0% strict**; holdout worse than Gaussian; effective rank ≈1 |
| `33` | Tier 3: split-fermion overlap N=50 — \(w/\sigma\) stable at fixed \(\sigma\) (rel spread **0.067**); geometry→\(w/\sigma\) R² **0.045**; Path D 3×3 GUE **not testable** |
| `36` | Tier 1: phase-fix audit N=15 — **0/15 strict** (repaired & legacy); masses invariant under phase fix; CKM differs legacy vs repaired |

### Diagnostic 31 — null geometry (N=30, seed 21021)

| Condition | Strict | Train median | Holdout median |
|-----------|--------|--------------|----------------|
| Kernel (real Q) | 0% | ~2.0 | ~898 |
| Shuffled Q | 0% | ~5.3 | ~213 |
| Haar Yu/Yd | 0% | ~362 | ~25,595 |

Pass/fail: kernel beats Haar on train loss (geometry signal real) but all 0% strict (no spurious survivor inflation).

### Diagnostic 29 — legacy geometry extension (N=100 sample)

- 0% strict; best extension joint **4.28** vs legacy CSV **4.90**
- Verdict: marginal loss improvement, not PDG simultaneous match

Canonical survivor protocol: [[survivor-protocol-preregistered]].

### Diagnostic 26 — paired wins vs Gaussian (N=100, >5% better)

| Sector | Metric | clockwork | gp1.5 | gp2.0 | gp3.0 |
|--------|--------|-----------|-------|-------|-------|
| Quark | train | 37/62/1 | 15/83/2 | 14/83/3 | 12/85/3 |
| Lepton | holdout | 30/69/1 | 41/56/3 | 0/1/99 | 50/49/1 |
| Neutrino | PMNS | 17/61/1 (n=79) | 25/54/4 (n=83) | 0/0/87 (n=87) | 41/44/2 (n=87) |

Format: alt-kernel wins / Gaussian wins / ties. Holdout medians ~29 (quark G), ~34 (lepton, all kernels). No universal envelope.

## Reports ingested

- `QA_SUMMARY.md`, `GAUSSIAN_KERNEL_FINAL_REPORT.md`, `KERNEL_COMPARISON_REPORT_v2.md`, `MINIMALITY_REPORT.md`
- `diagnostics/results/21_*` through `26_*` (2026-06-02g)
