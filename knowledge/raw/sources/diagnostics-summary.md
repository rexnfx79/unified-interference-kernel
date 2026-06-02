# Diagnostics Summary Snapshot

> **Canonical sources:** `../../diagnostics/` reports (QA_SUMMARY, KERNEL_COMPARISON_REPORT_v2, MINIMALITY_REPORT, GAUSSIAN_KERNEL_FINAL_REPORT). Synthesis snapshot — 2026-06 ingest.

## QA / reproducibility (established)

- **56/56** unit tests passed (`tests/run_qa_tests.py`).
- Kernel computation deterministic; optimization reproducible with fixed seeds.
- Claimed clockwork solution verified independently (mc, CKM elements within ~0.5% on training-style targets).

## Gaussian kernel (refuted as full quark model)

- Math implementation **correct**; failure is **structural**.
- Trade-off: large \(S_0/S_1\) (good \(m_c\)) ⟺ nearly rank-1 \(Y\) ⟺ bad CKM; full-rank ⟺ bad \(m_c\).
- Cannot simultaneously hit \(m_c\) and CKM with Gaussian envelope (Pareto knee documented).

## Clockwork kernel \(q^{-|d|}\) (partial phenomenology)

- Improves \(m_c\) + \(|V_{us}|\) vs Gaussian (combined error ~0.77 vs ~1.84).
- **Fails** light quarks (\(m_u \sim 10^3\times\) target), many third-gen CKM elements, Jarlskog sign/magnitude.
- Fragile to \(k, \eta\) perturbations (40–54% \(m_c\) error at ±1%).

## Minimality ladder (established negative result)

- Relaxing shared-\(Q\), Higgs shift, gear-ratio split **degrades** holdout loss vs baseline.
- Shared-\(Q\) is **not** the primary bottleneck; kernel functional form is.

## Transfer / universality (refuted)

- Frozen quark → lepton params: loss **797.5**; free all: **779.0** (~2.3% gain) — see `SCIENTIFIC_FINDINGS.md`.

## Manuscript-aligned honest quark failure

- Random geometry survey: \(m_c \approx 19.4 \pm 17.9\) GeV vs 1.27 GeV target; 0% strict quark survivors at PDG precision.

## Phenomenology tranche (diagnostics 21–25, 2026-06-02e)

| Script | Result |
|--------|--------|
| `21_quark_phenomenology_holdout.py` | 0% strict survivors; CKM–\(m_c\) Pareto; holdout fails |
| `22_lepton_phenomenology_sweep.py` | **100 geom:** 1% strict/legacy (train-only opt); 95% perfect train; 0% holdout m_e <5% |
| `23_neutrino_phenomenology_sweep.py` | **100 geom (90 solved):** 78.9% strict PMNS; g_env≈0.47; weak g_env–mixing (bootstrap CI) |
| `24_cross_kernel_paired_lepton_neutrino.py` | 30 paired geom; clockwork/generalized partial wins; holdout m_e fails all kernels |
| `25_lepton_mass_pareto.py` | Weighted m_μ–m_e Pareto; holdout-only split structural; 1 nondom. point when holdout weighted |

See wiki: [[diagnostics-summary]], [[proven-vs-conjecture-ledger]], [[plausibility-register]].
