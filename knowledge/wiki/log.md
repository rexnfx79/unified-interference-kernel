updated: 2026-06-02e
---

# Research Log

## 2026-06-02 | implement | Scaled phenomenology + cross-kernel (diag 22–25)

- **Scaled sweeps:**
  - `22_lepton_phenomenology_sweep.py` (100 geom) → **1% strict/legacy**; 95% perfect train; 0% holdout m_e <5%; median holdout loss **32**
  - `23_neutrino_phenomenology_sweep.py` (100 geom, 90 solved) → **78.9% strict** PMNS; g_env mean **0.47**; weak g_env–mixing (bootstrap CI consistent)
- **New diagnostics:**
  - `24_cross_kernel_paired_lepton_neutrino.py` (30 geom) — clockwork partial train wins; holdout m_e poor all kernels; generalized p=1.5 wins ν PMNS 14/24
  - `25_lepton_mass_pareto.py` — weighted m_μ–m_e Pareto; holdout-only split structural; figure `figures/lepton_mass_pareto_diag25.png`
- **Wiki:** [[analysis-summary]], [[diagnostics-summary]], [[manuscript-ledger-alignment]], [[projection-regimes]], [[research-strategy]]
- **Status:** Phenomenology tranche scaled; next = scale quark diag 21 + reconcile survivor objective protocols

## 2026-06-02 | implement | Lepton + neutrino phenomenology (diag 22–23)

- **Code:** `LEPTON_TRAINING_TARGETS` / `LEPTON_HOLDOUT_TARGETS`, `compute_lepton_training_loss`, `compute_lepton_holdout_loss` in `src/observables.py`
- **Diagnostics:**
  - `22_lepton_phenomenology_sweep.py` → **0% strict** survivors; train m_μ perfect, holdout m_e fails (median loss 149)
  - `23_neutrino_phenomenology_sweep.py` → **91.7% strict** PMNS (11/12 geom); g_env mean 0.47; |r(g_env, θ₂₃)|≈0.10
- **Refactor:** `scripts/03_true_transfer_test.py` imports lepton helpers from `observables.py`
- **Wiki:** [[manuscript-ledger-alignment]], [[projection-regimes]], [[research-strategy]], `analysis-summary.md`, `diagnostics-summary.md`
- **Status:** Phenomenology tranche complete for all three sectors; next = scale sweeps + cross-kernel paired tests

## 2026-06-02 | implement | Phenomenology tranche (diagnostic 21 + observables)

- **Code:** `compute_lepton_observables`, `compute_lepton_loss`, `LEPTON_TARGETS` in `src/observables.py`; TRAINING/HOLDOUT PDG-aligned
- **Tests:** `tests/test_lepton_observables.py`; neutrino tests wired in `run_qa_tests.py`
- **Diagnostic:** `diagnostics/21_quark_phenomenology_holdout.py` → `diagnostics/results/21_quark_phenomenology_holdout.txt`
- **Wiki:** [[manuscript-ledger-alignment]], `knowledge/raw/sources/manuscript-honest-limitations-draft.md`, [[research-strategy]] (phenomenology primary), `knowledge/purpose.md`
- **Status:** Primary track = phenomenology; Path A mechanism closed; Path D watch

## 2026-06-02 | implement | Fisher transfer + strategy pivot (diagnostics 19–20)

- **Code:** `src/fisher_transfer.py` — quark fit, frozen transfer, Fisher CR comparison; extends `src/experimental_fisher.py` (`UNIVERSAL_PARAM_NAMES`)
- **Tests:** `tests/test_fisher_transfer.py` (3 tests)
- **Diagnostics:**
  - `diagnostics/19_fisher_transfer_test.py` — frozen loss **805.84**, alignment **0.413**, falsifier B → **REFUTED**
  - `diagnostics/20_collider_fisher_sketch.py` — mixing-only Fisher under-identifies kernel; full collider Fisher **out of scope**
- **Wiki:** [[fisher-transfer-universality-test]], [[research-strategy]] (Path A deprioritized), [[plausibility-register]], [[proven-vs-conjecture-ledger]], [[information-creates-reality]], `knowledge/purpose.md`
- **QA:** 54/54 pass; manifest + wiki links OK
- **Status:** Path A QIT→flavor mechanism **deprioritized**; kernel = phenomenology; Path D **watch only** (no promotion hook)

## 2026-06-02 | implement | Experimental Fisher + open-system (diagnostics 17–18)

- **Code:** `src/experimental_fisher.py` — PDG-weighted Fisher, Jacobian, CR bounds, cross-sector alignment; `src/open_system_decoherence.py` — external \(p(g_{\mathrm{env}}, \varepsilon)\) Lindblad sketch
- **Tests:** `tests/test_experimental_fisher.py`, `tests/test_open_system_decoherence.py`
- **Diagnostics:** `diagnostics/17_experimental_fisher_pdg.py` — n=7,920; alignment 0.46; rank–regime |r|=0.72 (confounded); cross-sector mechanism **weak**
  - `diagnostics/18_open_system_mixing.py` — n=7,776; pooled |r|=0.007 **refuted**; quark-local |r|=0.26 exploratory
- **Wiki:** [[qm-to-information-what-is-measurable]], [[research-strategy]] (Path D criteria), [[plausibility-register]], [[proven-vs-conjecture-ledger]]
- **QA:** 51/51 pass; manifest + wiki links OK
- **Status:** Post-hoc ρ_Y QFI path **dead**; experimental Fisher does not support sector-split prediction; open-system pooled **refuted**

## 2026-06-02 | implement | Path A QED-info pivot (diagnostics 15–16)

- **Code:** `src/qed_information.py` — QFI (SLD), coherence l1, off-diagonal ratio, distinguishability vs uniform; `flavor_information.compute_yukawa_information(..., include_qed=True)`
- **Tests:** `tests/test_qed_information.py` (8 tests)
- **Diagnostics:**
  - `diagnostics/15_qed_fisher_yukawa.py` — n=10,080; pooled max \|r\| vs mixing = **0.017** → **REFUTED** (< 0.25)
  - `diagnostics/16_decoherence_mixing_bound.py` — n=2,592; max \|r\| = **0.241** → **REFUTED**; upper-bound fraction trivial
- **Wiki:** [[information-measure-for-projection-regimes]], [[qm-to-information-what-is-measurable]], [[research-strategy]], [[plausibility-register]], [[proven-vs-conjecture-ledger]]
- **QA:** 34/34 pass; manifest + wiki links OK

## 2026-06-01 | strategy | Path A+D user decisions + mechanism diagnostics

- **Strategy:** [[research-strategy]] — Path A (QED→info) + Path D (watch); SM flavor only; code-only; long budget
- **Deprioritized:** split-fermion→kernel on pursue list ([[derive-interference-kernel-from-overlaps]], [[plausibility-register]])
- **Path A diagnostics:**
  - `diagnostics/12_regime_entropy_correlation.py` — 10,080 samples; **REFUTED** (max \|r\| vs regime = 0.047 < 0.30 threshold)
  - `diagnostics/13_yukawa_information_inequality.py` — no cross-sector mechanism (pooled \|r\| < 0.1; sign-flip on off-diag candidate)
- **Path D:** `diagnostics/14_explicit_formula_numerics.py`; SM decoupling notes on [[explicit-formula-primes-zeros]], [[hilbert-polya-conjecture]]
- Updated `knowledge/purpose.md`, [[overview]], [[information-measure-for-projection-regimes]], [[qm-to-information-what-is-measurable]]
- **Lint:** manifest OK; wiki links OK after fix

## 2026-06-01 | implement | Open issues batch

- **S(ρ_Y):** `src/flavor_information.py`, `diagnostics/11_flavor_information_entropy.py`
- **PMNS:** `compute_neutrino_observables` in `src/observables.py`, `tests/test_neutrino_observables.py`; [[neutrino-observables-gap]] resolved
- **Overlap test:** extended `diagnostics/10_split_fermion_overlap_derivation.py` — 4 geometries, Yu+Yd, w/σ stability
- **Curated ingests:** [[montgomery-pair-correlation]], [[connes-spectral-triple]]
- **Lint:** `scripts/lint_wiki_links.py`; documented in `knowledge/AGENTS.md` and schema
- Updated [[knowledge-gaps-audit]], [[quantum-information]], [[information-measure-for-projection-regimes]]

## 2026-06-01 | implement | Audit recommendations

- Curated ingests: [[bekenstein-holographic-bound]], [[preskill-qit-entropy]]
- Repo ingests: [[analysis-summary]], [[pareto-envelope-comparison]], [[split-fermion-overlap-test]]
- `knowledge/manifest.yaml` + `scripts/lint_wiki_manifest.py`
- [[neutrino-observables-gap]] documented
- Overlap derivation: r=0.99996 ([[derive-interference-kernel-from-overlaps]] partial win)
- Updated [[knowledge-gaps-audit]] grade B+

## 2026-06-01 | adversarial-audit | Gaps + contradiction fixes

- [[knowledge-gaps-audit]] — full adversarial review of post-ingest wiki
- [[boundary-analysis-report]] ingested; [[similar-fitted-scales-vs-transfer]] disambiguates σ clustering vs transfer refutation
- New concepts: [[holographic-principle]], [[explicit-formula-primes-zeros]], [[clockwork-kernel]], [[generalized-envelope-kernel]]
- Fixed [[interference-kernel]] status → phenomenological; [[split-fermion-overlaps]] kernel identification → conjecture

## 2026-06-01 | ingest | Repo code, diagnostics, manuscript

- Raw snapshots: `kernel-implementation`, `observables-extraction`, `diagnostics-summary`, `manuscript-key-results`
- Wiki sources: [[kernel-implementation]], [[observables-extraction]], [[diagnostics-summary]], [[manuscript-key-results]]
- New concepts: [[split-fermion-localization]], [[split-fermion-overlaps]], [[yukawa-observables-pipeline]]
- New synthesis: [[proven-vs-conjecture-ledger]]
- New queries: [[derive-interference-kernel-from-overlaps]], [[information-measure-for-projection-regimes]]
- Updated Proven/Conjecture sections on [[interference-kernel]], [[projection-regimes]], [[hilbert-spaces-qm]], [[qed-qm-information]], [[quantum-information]], [[prime-numbers-and-physics]], [[riemann-zeta-function]], [[random-matrix-theory]]
- [[plausibility-register]]: Gaussian full quarks, shared-Q bottleneck, clockwork completeness, three-regime quark validation marked **dead**; pipeline QA **established**
- No `[[scientific-findings]]` broken links found (already [[repo-scientific-findings]])

## 2026-06-01 | refactor | Multi-sided framework + plausibility register

- Added [[multi-sided-bridge-framework]] — three approaches (arithmetic, Hilbert, QED/info) + gap-crossing protocol
- Added [[plausibility-register]] — dead/refuted/pursue verdicts
- New islands: [[prime-numbers-and-physics]], [[p-adic-analysis]], [[hilbert-spaces-qm]], [[qed-qm-information]], [[quantum-information]], [[primes-via-quantum-effects]]
- Adversarial queries: [[why-not-zeta-flavor-numerology]], [[qm-to-information-what-is-measurable]]
- Comparison: [[comparison-hilbert-vs-p-adic-approach]]
- Marked **dead**: zeta→flavor direct, 3×3 GUE test, universal kernel transfer
- Renamed source page → [[repo-scientific-findings]]
- Primary pursue path: C → B → (maybe) A → explain D

## 2026-06-01 | init | Knowledge base bootstrap

- LLM Wiki structure; submodule `tools/llm_wiki`
- Seed islands and [[information-reality-bridge-map]]
