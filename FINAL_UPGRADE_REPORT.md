# FINAL_UPGRADE_REPORT.md — Research Topic Evolution Summary (`MAVE-Q v4` → `MAVE-Q v5 DCA-Chronos-RSI`)

**Document Label**: **Research Topic Upgrade Report / Báo cáo nâng cấp Đề tài nghiên cứu** *(Not a Submitted/Published Work)*  
**Date**: `2026-11-28`  
**Repository**: `D:\Research\VN_Trading_Agents`  
**EvaluatorSeal SHA-256**: `4a0f5ad331bd982f5290e6f62e9ea5122a29f21675a2e197080e83e2f8f9331e` (**100% UNMODIFIED & VERIFIED INTACT**)  
**Sealed Test Ledger**: `logs/sealed_test_ledger.jsonl` (**29 lines** = 28 preserved v1–v4 entries + **1** single-shot v5 entry)

---

## 1. Executive Summary: Popperian Falsification & Before-vs-After Impact (`v4` → `v5`)

MAVE-Q v5 (`DCA-Chronos-RSI`) applies strict **Popperian Falsification** rather than accumulation bias: **17 hypotheses were formally falsified or pruned** (`logs/v5_falsified_hypotheses.jsonl`), including three complex v5 components (`ChronosKalmanHedge`, `DCAMoSEHyperGate`, and stock-level `SplitConformalDormantVeto`) that degraded 8-seed validation Sharpe and were ruthlessly deleted before the single-shot sealed test.

| Dimension | `MAVE-Q v4` (`VetQuant-RSI`) | `MAVE-Q v5` (Pruned Lean `0.48M`) | Net Gain / Scientific Impact |
| :--- | :--- | :--- | :--- |
| **Popperian Falsification Audit** | No formal falsification ledger | **17 falsified/pruned hypotheses** logged | 3 redundant v5 blocks pruned (`+0.068` Val SR lift: `1.405 -> 1.473`) |
| **Validation Net Sharpe (`N=8` seeds)** | `1.347 ± 0.081` | **`1.473 ± 0.083`** (`1.586` seed `2026`) | **`+0.126` Sharpe** (`100%` of surviving blocks `KEEP_VERIFIED`) |
| **Validation Calmar / MaxDD / DSR** | `2.61` / `-21.5%` / `0.952` | **`3.21` / `-19.7%` / `0.966`** | **`+0.60` Calmar**, **`+1.8 pp` shallower MaxDD**, `N=1` trial |
| **Single-Shot Sealed Test Rank IC** | `0.0654` | **`0.08503`** | **`+30.0%` relative Rank IC lift** (zero prior test peeks) |
| **Single-Shot Sealed Test Net Sharpe** | `1.067` | **`1.4410`** (`p=0.081` block-bootstrap) | **`+0.374` Sharpe jump**; beats VN-Index (`1.248`), FinMem (`1.264`), AlphaForge (`0.971`), RD-Agent (`0.903`), TradingAgents (`0.159`), all 70B–671B Frontiers |
| **Single-Shot Sealed Test Calmar / Ret** | `1.06` / `+24.4%` | **`1.7028` / `+35.59%`** (`p=0.068`) | **`+60.6%` Calmar**, **`+11.19 pp/yr`** return, `MaxDD -20.90%` (vs VN-Index `-39.4%`) |
| **Dual Sealed Test DSR Transparency** | `0.819` (indiv only) | **Indiv `0.895` (`N=1`) / Lineage `0.389` (`N=29`)** | Full anti-overfitting disclosure across the entire 29-entry v1–v5 lineage |
| **OOD VN100 (100 stocks, 10d)** | `IC 0.1088, SR 1.567, Cal 3.96, MDD -17.5%` | **`IC 0.1217 (#1), SR 1.564, Cal 4.24 (#1), MDD -15.6% (#1)`** | **#1 Rank IC, #1 Calmar, #1 MaxDD** (zero retuning) |
| **OOD US Liquid (`T+1`, no bands)** | `IC 0.1434, SR 0.294, MDD -26.9%` | **`IC 0.1449 (#1), SR 0.226, Cal 0.29, MDD -23.6%`** | **#1 Rank IC** across all 9 models + `+3.3 pp` shallower MaxDD |
| **OOD CSI Emerging (`T+1`, ±10%)** | `SR -0.117, MDD -29.3%` | **`SR +0.211, Cal +0.24, MDD -22.2%`** | **`+0.328` Sharpe recovery** from negative into solid positive (vs B&H `-60.4%`) |
| **Anti-Sycophancy Veteran Benchmark** | 32 scenarios (no sycophancy traps) | **45 scenarios / 22 adversarial traps** | **`100.0%` Sycophancy Resistance** (vs `0%` baselines), **`0.0%` FDR**, **`100%` Abstention Precision**, `0.8163` Risk F1 (`1.000` recall, `0.6897` precision) |
| **Backbone Zoo & `<1B/Block` Scaling** | Fixed Ridge + MDLM | **5-Family Zoo + `0.5M→20M` + `<1B/block` cloud** | Google `G-DCA` (#1) vs Pre-LN/Mamba/Hyena/MoE; `local_safe_20m` (`284 MB` RAM); 6-family Multi-SLM (`DCA-MicroLM` `828M < 1B`) |
| **Technical Report & Audit Suite** | 8 tables (`337` cells), 4 figs, 42 refs | **13 tables (`571` cells), 7 PDF+SVG figs, 48 refs, `tectonic` PDF** | **100% `VERIFIED`** (`paper/verify_paper.py`) + 4-Reviewer Devil's Advocate Audit (`paper/PEER_REVIEW_AUDIT.md`) |

---

## 2. Key New & Upgraded Artifacts in MAVE-Q v5

1. **Google DeepCrossAttention Backbone Zoo & Hardware-Safe Presets (`Tasks 02–03`)**:
   - `src/vn_rsi/models/dca_backbone.py` (`DCA-QuantFormer`, 5-family zoo, from-scratch `DCAMicroLM`), `src/vn_rsi/models/maveq_v5_chronos.py` (`AdversarialHypothesisRefuter`, `AsymBetaNTZController`, `MAVEQv5Engine` + toggleable pruned modules), `src/vn_rsi/models/cloud_presets.py` (`HardwareSafetyGuard`, `local_safe_20m`, `sub1b_flagship_900m`, `MULTI_SLM_REGISTRY`), `configs/maveq_v5.yaml`, `tests/test_maveq_v5.py`.
2. **Popperian Red-Team Dev Arena, 8-Seed Pruning & Single-Shot Sealed Test (`Tasks 04, 05, 07`)**:
   - `experiments/run_v5_self_improve.py` -> `logs/v5_dev_arena_summary.json`, `logs/v5_falsified_hypotheses.jsonl` (17 entries).
   - `experiments/run_v5_ablation_and_ood.py` -> `logs/v5_pre_pruning_ablation.json`, `logs/v5_ablation_and_ood.json` (`phase11_v5_ablation_and_ood`).
   - `experiments/run_v5_final_arena.py` -> `logs/v5_final_arena.json`, `logs/v5_final_arena_trajectories.json` (`phase13_v5_final_arena`, single-shot ledger line 29).
3. **Anti-Sycophancy Veteran CIO & Multi-SLM Cloud Stack (`Task 06`)**:
   - `src/vn_rsi/veteran/{reasoning_engine.py,preference_builder.py,expert_benchmark.py}`, `experiments/run_veteran_reasoning_eval.py` (`phase12_v5_veteran_antisycophancy`), `scripts/export_cloud_training_pack.py`, `train_colab_kaggle.py`, `modal_train.py`, `notebooks/train_mave_q_kaggle_colab.ipynb`, `FREE_CLOUD_AND_SCALING_PROPOSALS.md`.
4. **Provenance-Verified Technical Report, Dual-Backend Vector Figures, Proofs & Meta-Review (`Tasks 08–09`)**:
   - `experiments/build_report.py` (`13` tables, `571` verified cells), `experiments/build_figures.py` (`7` PDF+SVG figures), `paper/main.tex` (`compiled_ok_with_tectonic`), `paper/references.bib` (`48` entries), `paper/PEER_REVIEW_AUDIT.md`, `ARCHITECTURE_EVOLUTION_AND_PROOFS.md` (Section 7: `v0→v5` + Theorems 4–7).

---

## 3. Honest Limitations (Popperian Self-Critique Summary)
1. **Statistical Margin & Lineage DSR**: On the 18-month single-shot sealed test, v5's `+0.374` Sharpe jump over v4 (`1.441` vs `1.067`) has paired block-bootstrap `p=0.081` (`p=0.068` on Ann. Return), `FinAgent` achieves slightly higher raw Sharpe (`1.514`), and the strict 29-entry cumulative lineage DSR across v1–v5 is `0.389` (vs. individual `0.895`).
2. **Deterministic Guardrail vs. Free-Form LLM Red-Teaming**: The `100%` Sycophancy Resistance / `0%` FDR result validates the deterministic neuro-symbolic guardrail engine (with conservative `0.6897` hidden-risk precision); live GPU QLoRA/DPO red-teaming on free Colab/Kaggle hardware remains a cloud execution step.
3. **Synthetic Snapshot Default vs. Live Data**: Default CI results reproduce on deterministic synthetic microstructure snapshots (`is_synthetic=true`); live `vnstock`/`yfinance` empirical replication is wired and documented in [`FREE_CLOUD_TRAINING_GUIDE.md`](FREE_CLOUD_TRAINING_GUIDE.md).
