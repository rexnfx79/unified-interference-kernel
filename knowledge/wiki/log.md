updated: 2026-06-12
---

# Research Log

## 2026-06-12 | implement | Diag 43 joint 6×6 three-sector fit

- **Diag 43:** L3 FAIL — 6×6 Schur joint holdout sum **1387** vs independent **670** (N=30, diag 26 corpus)
- **Wiki:** [[chiral-projection-formalization-program]] L3 failed

## 2026-06-12 | ingest | Chiral projection thesis + diag 42

- **Wiki:** [[chiral-projection-thesis]], [[adversarial-review-chiral-projection-thesis]], [[chiral-projection-formalization-program]]
- **Diag 42:** quark portal audit N=20 — simple portals fail \(J\) falsifier; Schur overfits holdout
- **Register:** mirror sector **watch**; orientation map **philosophical**; kernel bridge **dead**

## 2026-06-02 | explore | N2 Haar PMNS null (diag 41)

- **Diag 41:** post-fit angles reject Haar (3/3 KS); PDG distance ≪ Haar null
- **Wiki:** [[neutrino-haar-null-n2]]

## 2026-06-02 | explore | N4 geometry strict predictor (diag 40)

- **Diag 40:** diag 28 pool — CV AUC **0.658**, best uni **0.572** (`overlap_count`); falsifier rejected
- **Wiki:** [[neutrino-geometry-predictor-n4]]

## 2026-06-02 | explore | N1 joint loss landscape (diag 39)

- **Diag 39:** quark vs neutrino joint loss cartography — 4/5 metrics differ (\(p<0.05\)); ν strict 34% vs quark 0%
- **Wiki:** [[neutrino-loss-landscape-n1]]

## 2026-06-02 | implement | Tier 4 hygiene + manuscript sync

- **data/README.md** — legacy CSV vs canonical survivor protocol
- **manuscript.tex** — Further Study: Tier 5 (34–38) closed, CP/diag 36 done
- **Wiki/strategy:** Tier 5 complete; [[future-work]], [[manuscript-ledger-alignment]], README reproduce block

## 2026-06-02 | implement | Tier 5.5 QED prime spectral audit (diag 38)

- **Code:** `src/qed_spectral_sums.py`; `38_tier5_qed_prime_spectral_audit.py`
- **Verdict:** FAIL — Schwinger/ζ sums converge on integer n; prime-only surrogates **67–93%** off; Euler product analytic only
- **Wiki:** [[can-primes-enter-via-qed-spectral-sums]], [[future-work]], [[diagnostics-summary]]

## 2026-06-02 | implement | Tier 5.4 landscape RMT (diag 37)

- **Diag 37:** `37_tier5_landscape_rmt.py` — N=60 Gaussian minima, pooled Hessian unfolded spacings
- **Verdict:** FAIL GUE — frac(s<0.1)=**0.42** vs Poisson **0.10**, GOE **0**; do not link to CKM/zeros
- **Wiki:** [[future-work]], [[diagnostics-summary]]

## 2026-06-02 | implement | Tier 5.2–5.3 (diag 34–35)

- **T5.2:** `34_explicit_formula_spectral_audit.py` — PASS: true zeros beat random-frequency null; `src/explicit_formula.py`
- **T5.3:** `35_jacobi_inverse_kernel_phase.py` — FAIL: Yukawa not reducible to 3-site Hermitian/tridiagonal proxy
- **Wiki:** [[future-work]], [[diagnostics-summary]], [[conjecture-to-physics-avenues]]

## 2026-06-02 | implement | Tier 1 CP + phase-fix audit (diag 36)

- **Code:** `delta_CKM`, `J`, `delta_PMNS`, `J_PMNS` in `observables.py`; `tests/test_cp_observables.py`
- **Diag 36:** N=15 — 0/15 strict repaired & legacy; masses unchanged; CKM legacy vs repaired differs
- **Verdict:** Quark refutation **stable** after SVD phase fix (P1.1 pass)

## 2026-06-02 | implement | Tier 0 publication package

- **Manuscript:** abstract 27.8% joint ν lead; 0/5759 quark; Further Study closed diag 32–33; reproduce script cited
- **Wiki:** [[manuscript-ledger-alignment]], [[future-work]] Tier 0 complete; README + reproduce path
- **Review fixes:** [[adversarial-review-tier5-trace-formula]]; T5.2 blocked; ladder status active

## 2026-06-02 | review | Adversarial Tier 5 trace-formula program

- **Query:** [[adversarial-review-tier5-trace-formula]] — T5.2 tautology risk; “physics” framing overstated; Tier 0 priority
- **Wiki fixes:** [[trace-formula-bridge-ladder]] status→active; Selberg arrow analogy-only

## 2026-06-02 | ingest | Tier 5.1 trace-formula bridge (wiki)

- **Synthesis:** [[trace-formula-bridge-ladder]] — Selberg → explicit formula → Montgomery → HP / Connes
- **New:** [[selberg-trace-formula]] concept + raw/source ingests; expanded [[explicit-formula-primes-zeros]], [[montgomery-pair-correlation]], [[connes-spectral-triple]]
- **Ledger:** [[proven-vs-conjecture-ledger]] — explicit formula, Selberg, Montgomery (conditional), Connes framework

## 2026-06-02 | strategy | Conjecture-to-physics program (Tier 5)

- **Wiki:** [[conjecture-to-physics-avenues]] — ranked B↔A trace formula, Jacobi inverse, landscape RMT; dead ends explicit
- **Strategy:** [[future-work]] Tier 5, [[research-strategy]] priority 4, [[multi-sided-bridge-framework]] order updated
- **Git:** pushed `7c6b923` (Tier 2–3 closure, SVD fix, diagnostics 32–33)

## 2026-06-02 | implement | Tier 3 theory bridges (diag 33)

- **Code:** `src/split_fermion_overlap.py`; `diagnostics/33_tier3_theory_bridges.py` — N=50, seed 33033
- **Track A:** \(w/\sigma\) rel spread 0.067 (stable at fixed params); geometry→\(w/\sigma\) R² **0.045** — mechanism **refuted**
- **Track B:** 3×3 GUE spacing not testable; Path D watch-only
- **Wiki:** [[future-work]], [[derive-interference-kernel-from-overlaps]], [[diagnostics-summary]], [[plausibility-register]]

## 2026-06-02 | implement | Tier 2 quark ansatz (diag 32)

- **Code:** `compute_yukawas_rank2_clockwork_sum`, `TIER2_QUARK_KERNELS` in `alternative_kernels.py`; `generate_quark_geometries` in `phenomenology_utils.py`
- **Diagnostic:** `32_quark_tier2_ansatz.py` — N=100, seed 32032, holdout >20% vs Gaussian rule
- **Smoke:** 0/10 strict all kernels; rank2/FN worsen holdout vs Gaussian on small sample
- **Full N=100:** 0% strict all kernels; Gaussian holdout median 25.5; rank2/FN/power-law holdout ≫ baseline — **Tier 2 falsified**

## 2026-06-02 | implement | Future work tier plan (post-tranche)

- **Wiki:** [[future-work]] — Tier 0 publication, Tier 1 CP/post-fix audit, Tier 2 rank-2 pre-reg, Tier 3 theory (deprioritized)
- **Manuscript:** §Further Study rewritten — closed geometry/split-Q/envelope paths; honest forward items
- **Strategy:** [[research-strategy]] points to [[future-work]]; phenomenology checklist collapsed to “complete”

## 2026-06-02 | implement | Strategy closure + observables phase fix

- **Strategy:** [[research-strategy]] — quark dead-ends table, active priorities (honest phenomenology; no more geometry grids)
- **Code:** `fix_svd_phases` — column/row paired phases; preserves \(Y=U\Sigma V^\dagger\); `test_fix_svd_phases_preserves_reconstruction`
- **Register:** [[plausibility-register]] — geometry extension + split \(Q_u,Q_d\) marked dead
- **Diag 29:** penalty-fix rerun documented in [[diagnostics-summary]]

## 2026-06-02 | implement | Geometry extension hardened follow-up (diag 30)

- **Diagnostic 30:** exhaustive legacy re-baseline **989/1000** + shell-5 **4770/5000** (unified DE); **0/5759 strict**; Wilson 95% UB **~0.07%**; shell-5 joint min **3.06** vs re-baselined 1k **5.01** (legacy CSV **4.90** not comparable)
- **Bugfix:** diag 29/30 — accept only `fun≈1000` penalty rejects, not high joint loss
- **Artifacts:** `data/quark_geometry_followup_*.csv`, `quark_geometry_followup_bests.json`
- **Note:** diag 29 N=100 sample superseded; [[quark-geometry-conventions]] documents legacy vs phenomenology grids

## 2026-06-02 | implement | Geometry extension + null baseline (diag 29–31)

- **Diagnostic 29:** legacy grid extension (max_coord 6–8), N=100 sample — **0% strict** (see fun-filter bug; use 30); best joint **4.28** vs CSV **4.90**
- **Diagnostic 25:** scaled to N=100 — corr **0.31** (vs **0.58** at N=24); 7 nondominated Pareto pts (vs 1); structural m_e tension persists
- **Diagnostic 31:** Tier A4 N=30 — kernel train median **1.13** vs Haar **652**; all **0% strict**
- **Manuscript:** Discussion §Pre-Registered Falsifiers (diag 27/28 numbers)
- **Wiki:** [[quark-geometry-conventions]], [[diagnostics-summary]], [[manuscript-ledger-alignment]], [[research-strategy]]
- **Tests:** Δm² unit test in `test_neutrino_observables.py`

## 2026-06-02 | implement | Adversarial review follow-up (Tier A1/A2/B2)

- **Protocol:** [[survivor-protocol-preregistered]] — single strict PDG-relative survivor definition; legacy rates demoted
- **Manuscript:** title subtitle, intro regime labels, §Large Mixing causal bullets removed, conclusions lead quark 0% / lepton m_e / ν PMNS caveat, geometry extension softened
- **Code:** `observables.py` — neutrino Δm² extraction, `compute_neutrino_mass_loss`, `compute_neutrino_joint_loss`
- **Diagnostics:** `27_quark_joint_loss_holdout.py` → Gaussian **2%** strict, sparse Pareto persists (A2: not split artifact); `28_neutrino_masses_pmns_joint.py` → **27.8%** strict vs 78.9% PMNS-only (B2)
- **Wiki:** [[manuscript-ledger-alignment]], [[research-strategy]], [[diagnostics-summary]], `index.md`
- **Status:** Tier A1 locked; A2/B2 results in `diagnostics/results/27_*`, `28_*`

## 2026-06-02 | implement | Joint 3-sector cross-kernel corpus (diag 26)

- **Code:** `generate_joint_three_sector_geometries` in `phenomenology_utils.py` — shared L (= quark Q), independent E/N/U/D; seed **26026**
- **Diagnostic:** `26_joint_three_sector_cross_kernel.py` — Gaussian/clockwork/generalized p∈{1.5,2,3} on quark+lepton+neutrino at equal N; `--smoke` (N=3) verified in ~1.5 min
- **Full run:** N=100, 4 seeds — **complete** (45.4 min); shared L across sectors
- **Paired wins vs Gaussian (>5% better):**
  - Quark (train): C 37/62/1, gp1.5 15/83/2, gp2.0 14/83/3, gp3.0 12/85/3
  - Lepton (holdout): C 30/69/1, gp1.5 41/56/3, gp2.0 0/1/99, gp3.0 50/49/1
  - Neutrino (PMNS): C 17/61/1 (n=79), gp1.5 25/54/4 (n=83), gp2.0 0/0/87 (n=87), gp3.0 41/44/2 (n=87)
- **Holdout medians:** quark G ~29, lepton G ~34; no kernel fixes m_e or quark holdout
- **Status:** Joint 3-sector corpus complete; supersedes diag 24 for cross-sector paired comparison

## 2026-06-02 | implement | Scaled quark holdout + manuscript alignment

- **Diagnostic 21:** scaled to 100 geom, 4 seeds (parity with 22/23); **0% strict** all kernels; Gaussian holdout median **32.8**; paired G vs C: 53/38/9 wins; ~26 min runtime
- **Manuscript:** applied [[manuscript-honest-limitations-draft]] — abstract/intro limitations paragraph; dual-protocol survivor language; softened three-regime validation, unification, UV/statistical future work
- **Wiki:** [[manuscript-ledger-alignment]], [[research-strategy]], [[diagnostics-summary]]
- **Status:** Phenomenology tranche quark scale complete; joint 3-sector corpus remains

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
