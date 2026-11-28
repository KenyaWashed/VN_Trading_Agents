# Project Progress & Phase Acceptance Checklist (`PROGRESS.md`)

**Document Label**: **Research Topic Progress Log / Nhật ký tiến độ Đề tài nghiên cứu MAVE-Q v5 (`DCA-Chronos-RSI`)** *(Not a Submitted/Published Work)*  
**Author**: **Tran Duy Khang** — *AI Software Engineer / Independent Quantitative AI Researcher*  
**Last Updated**: `2026-11-28`  
**EvaluatorSeal SHA-256**: `4a0f5ad331bd982f5290e6f62e9ea5122a29f21675a2e197080e83e2f8f9331e` (**100% UNMODIFIED**)

---

## Phase-by-Phase Acceptance Status (Phases 0–13 / v0 → v5)

| Phase / Track | Deliverable | Acceptance Check | Status |
| :--- | :--- | :--- | :--- |
| **Phase 0–2** | `docs/`, `src/vn_rsi/{data,sandbox,eval}/` | Immutable Parquet snapshot (`54,780` rows, `30` tickers incl. `ROS_DELIST`, `FLC_HALT`, `NVL_STRESS`), SHA-256 `EvaluatorSeal` locked (`4a0f5ad3...31e`), 3-zone guard + single-shot `SealedTestLedger`, 7 baselines (`phase1_data_qa`, `phase2_baselines`) | **PASS (VERIFIED)** |
| **Phase 3–7 (`v0→v3`)** | `src/vn_rsi/{dsl,memory,agents,rsi,models}/` | 7-role `v0`, discrete AST diffusion `v1`, QD `v2`, Holo-MAVE-Q `v3` vs 7 alpha miners & 70B–671B frontiers (`phase3_agent_v0`..`phase7_holo_maveq_ablation`) | **PASS (VERIFIED)** |
| **Phase 8–10 (`v4`)** | `src/vn_rsi/{benchmarks,models/maveq_v4_veteran.py,veteran}/` | 25+ extended metrics, 5 causal Global SOTA rivals (`TradingAgents`, `FinMem`, `FinAgent`, `RD-Agent(Q)`, `AlphaForge-DE`), 3-market OOD, Zero-Val-Burn CV, 32-scenario veteran engine, single-shot v4 arena (`phase8`..`phase10_v4_final_arena`) | **PASS (VERIFIED)** |
| **v5 Task 01** | `AUTONOMOUS_STATE_V5.md`, `logs/v5_falsified_hypotheses.jsonl` | Post-mortem of v4's 7 flaws, live SHA-256 seal verified, Popperian Falsification Constitution + 3 initial falsified hypotheses logged | **PASS (VERIFIED)** |
| **v5 Task 02–03** | `src/vn_rsi/models/{dca_backbone.py,maveq_v5_chronos.py,cloud_presets.py}`, `configs/maveq_v5.yaml`, `tests/test_maveq_v5.py` | Google `G-DCA` (`DCA-QuantFormer`, 5-Family Backbone Zoo, `DCAMicroLM`), `HardwareSafetyGuard` (`local_safe_20m` `<1.2GB` RAM, `sub1b_flagship_900m` `<1B/block` 0 MB meta-verified, 6-family Multi-SLM), `AsymBetaNTZController`, 7/7 `atol=1e-9` future-perturbation shift-invariance tests green | **PASS (VERIFIED)** |
| **v5 Task 04** | `experiments/run_v5_self_improve.py`, `logs/v5_dev_arena_summary.json` | 14-candidate Popperian Red-Team Dev Arena (`--dev` ONLY, peak RAM `284 MB`); 11 rejected hypotheses (including all 6 adversarial controls) logged honestly (`phase10b_v5_dev_arena_and_falsifications`) | **PASS (VERIFIED)** |
| **v5 Task 05** | `experiments/run_v5_ablation_and_ood.py`, `logs/v5_ablation_and_ood.json` | Ruthless 8-seed Pruning Audit: 3 redundant modules (`ChronosKalmanHedge`, `DCAMoSEHyperGate`, stock-level `SplitConformalDormantVeto`) permanently pruned (`17` total falsified); Lean `0.48M` v5 achieves `Val SR 1.473±0.083, Cal 3.21, MDD -19.7%, DSR 0.966`, 100% surviving ablations `KEEP_VERIFIED`, and improves all 3 OOD markets (`VN100 Cal 4.24 #1`, `US IC 0.1449 #1`, `CSI SR +0.211`) (`phase11_v5_ablation_and_ood`) | **PASS (VERIFIED)** |
| **v5 Task 06** | `src/vn_rsi/veteran/`, `train_colab_kaggle.py`, `modal_train.py`, `scripts/export_cloud_training_pack.py`, `FREE_CLOUD_AND_SCALING_PROPOSALS.md` | 45-scenario / 22-trap Anti-Sycophancy benchmark: `100%` Decision Acc, `100%` Sycophancy Resistance (vs `0%` baselines), `0.0%` FDR, `100%` Abstention Precision, `0.8163` Risk F1 (`phase12_v5_veteran_antisycophancy`); 100 search-zone SFT/DPO pairs exported; local `--smoke-test` verified under `local_safe_20m` | **PASS (VERIFIED)** |
| **v5 Task 07** | `experiments/run_v5_final_arena.py`, `logs/v5_final_arena.json`, `logs/sealed_test_ledger.jsonl` | Pre-Seal Audit 100% passed; Single-Shot Sealed Test scored **ONCE** (ledger `28 -> 29`): **`Test SR 1.441 (+0.374 vs v4, p=0.081), IC 0.0850 (+30%), Calmar 1.70 (+61%), Ret +35.6% (p=0.068), MDD -20.9%, Indiv DSR 0.895 / Lineage DSR 0.389`** (`phase13_v5_final_arena`) | **PASS (VERIFIED)** |
| **v5 Task 08–09** | `experiments/{build_report.py,build_figures.py}`, `paper/{main.tex,main.pdf,references.bib,verify_paper.py,PEER_REVIEW_AUDIT.md}`, `ARCHITECTURE_EVOLUTION_AND_PROOFS.md` | 13 auto-generated LaTeX tables (`571/571` provenance cells), 7 dual-backend PDF+SVG figures, 48 BibTeX refs, Theorems 1–7 + proofs, Tectonic PDF compiled (`compiled_ok_with_tectonic`), ruthless 4-Reviewer Devil's Advocate Audit & 7 self-fixes verified 100% | **PASS (VERIFIED)** |
| **v5 Task 10** | `README.md`, `MODULAR_RSI_ARCHITECTURE.md`, `PROGRESS.md`, `FREE_CLOUD_TRAINING_GUIDE.md`, `FINAL_UPGRADE_REPORT.md`, `scripts/ci_check.py` | Full documentation sync strictly labeled **Research Topic / Đề tài nghiên cứu** + full `scripts/ci_check.py` (`ruff` + `mypy` + `55/55 pytest` + `571-cell` report/PDF verification + SHA-256 `EvaluatorSeal`) 100% GREEN | **PASS (VERIFIED)** |

---

## Verified vs. Explicitly Flagged UNVERIFIED / NOT RUN Inventory

### Directly Verified on Current Local Host (`AMD Ryzen 5 5500U`, `7.35 GB RAM`, `0 CUDA GPUs`)
- `scripts/ci_check.py`: `ruff check`, `mypy` (0 errors across `src/vn_rsi`), `pytest` (**55/55 tests** passing: causality, `atol=1e-9` future-perturbation shift-invariance across all 5 backbones + global rivals, SHA-256 seal integrity, single-shot ledger guard, anti-sycophancy benchmark), and `paper/verify_paper.py` (`48/48` citations, `27/27` cross-refs, `13/13` tables, `7/7` PDF+SVG figures, **`571/571` log-verified cells**, **`compiled_ok_with_tectonic`**).
- Peak local memory telemetry: **`284 MB`** RAM during the 14-candidate v5 Dev Arena (`<1.2 GB` `HardwareSafetyGuard` cap); `sub1b_flagship_900m` parameter budget verified at `0 MB RAM` via `meta` tensor counting.
- `train_colab_kaggle.py --mode maveq_v5 --preset auto --slm-family dca_microlm --enable-veteran-dpo --smoke-test`, `scripts/export_cloud_training_pack.py` (100 SFT/DPO pairs), and `scripts/ask_veteran_agent.py` (Vietnamese/English CLI).

### Explicitly Flagged UNVERIFIED / NOT RUN on Current 0-GPU Host
1. **Live Empirical Market Data (`NOT RUN` by default)**: `VnstockDataProvider` and `yfinance` live fetchers are implemented and ready; default CI tables use deterministic synthetic fixtures (`is_synthetic: true`) for reproducibility.
2. **Broker Square-Root Market Impact Coefficient (`UNVERIFIED`)**: `configs/market_rules.yaml` marks `illiquidity_impact.verified: false` (conservative proxy), while statutory commission (`15 bps`), sell tax (`10 bps`), T+2.5 holding locks, 100-share board lots, and `±7%/10%/15%` exchange price limits are `verified: true`.
3. **Physical Cloud GPU / TPU Execution (`NOT RUN` on 0-GPU laptop)**: VRAM/HBM budgets for `kaggle_t4_optimal`, `kaggle_max_100m`, `tpu_v5e_350m`, and `sub1b_flagship_900m` plus live QLoRA/DPO red-teaming of generative SLMs are analytical configurations prepared for free Colab/Kaggle/Modal runs (see [`FREE_CLOUD_TRAINING_GUIDE.md`](FREE_CLOUD_TRAINING_GUIDE.md) and [`FREE_CLOUD_AND_SCALING_PROPOSALS.md`](FREE_CLOUD_AND_SCALING_PROPOSALS.md)).
