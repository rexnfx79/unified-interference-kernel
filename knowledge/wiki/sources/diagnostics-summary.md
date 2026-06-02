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
updated: 2026-06-02g
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

## Phenomenology tranche (21–26)

| Script | Scaled result |
|--------|---------------|
| `21` | 100 geom: **0% strict** all kernels; CKM–\(m_c\) Pareto (corr≈0); G/C paired 53/38/9 train wins |
| `22` | 100 geom: 1% strict/legacy (train split); holdout m_e fails |
| `23` | 100 geom: 78.9% strict PMNS; g_env≈0.47 |
| `24` | Cross-kernel paired lepton+ν (30 geom; superseded by 26) |
| `25` | Lepton m_μ–m_e weighted Pareto |
| `26` | Joint 3-sector N=100 shared L; see paired wins below |

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
