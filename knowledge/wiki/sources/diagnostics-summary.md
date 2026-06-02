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
updated: 2026-06-02e
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

## Phenomenology tranche (21–25)

| Script | Scaled result |
|--------|---------------|
| `21` | 100 geom: **0% strict** all kernels; CKM–\(m_c\) Pareto (corr≈0); G/C paired 53/38/9 train wins |
| `22` | 100 geom: 1% strict/legacy (train split); holdout m_e fails |
| `23` | 100 geom: 78.9% strict PMNS; g_env≈0.47 |
| `24` | Cross-kernel paired lepton+ν (30 geom) |
| `26` | Joint 3-sector N=100: quark G wins 62/37; lepton holdout med ~34; ν 87–95 solved |
| `25` | Lepton m_μ–m_e weighted Pareto |

## Reports ingested

- `QA_SUMMARY.md`, `GAUSSIAN_KERNEL_FINAL_REPORT.md`, `KERNEL_COMPARISON_REPORT_v2.md`, `MINIMALITY_REPORT.md`
- `diagnostics/results/21_*` through `25_*` (2026-06-02f)
