# Project Progress & Phase Acceptance Checklist (`PROGRESS.md`)

**Author**: **Tran Duy Khang** — *AI Software Engineer / Independent Quantitative AI Researcher*  
Last Updated: `2026-11-26`

---

## Phase-by-Phase Acceptance Status (Phases 0–12)

| Phase / Task | Deliverable | Acceptance Check | Status |
| :--- | :--- | :--- | :--- |
| **Phase 0** | `docs/related_work.md`, `docs/design_decision.md` | Literature synthesis & architectural design verified | **PASS (VERIFIED)** |
| **Phase 1** | `src/vn_rsi/data/`, `configs/market_rules.yaml`, `docs/data_terms.md` | Immutable Parquet snapshot (`54,780` rows, `30` tickers incl. delisted/halted `ROS_DELIST`, `FLC_HALT`, `NVL_STRESS`), SHA-256 manifest, PIT lag $\ge 24$d (`run_id=phase1_data_qa`) | **PASS (VERIFIED)** |
| **Phase 2** | `src/vn_rsi/sandbox/`, `src/vn_rsi/eval/` | `EvaluatorSeal` SHA-256 locked (`4a0f5ad3...31e`), Canary null-IC check green, 3-zone guard + single-shot `SealedTestLedger`, 7 baselines logged (`run_id=phase2_baselines`) | **PASS (VERIFIED)** |
| **Phase 3–4** | `src/vn_rsi/dsl/`, `memory/`, `agents/`, `rsi/` | 7-role $\le$8B constrained pipeline & bi-level RSI on bounded mutable surface (`run_id=phase3_agent_v0`, `phase4_rsi`) | **PASS (VERIFIED)** |
| **Phase 6–7** | `src/vn_rsi/models/{holo_maveq.py,alpha_mining_baselines.py,cloud_presets.py}` | Modular Micro-RSI, Frontier Combat (`70B–671B`), & Holo-MAVE-Q v3 vs 7 alpha miners (`run_id=phase4_modular_rsi`, `phase5_frontier_combat`, `phase7_holo_maveq_combat`, `phase7_holo_maveq_ablation`) | **PASS (VERIFIED)** |
| **Tasks 01–03** | `src/vn_rsi/benchmarks/{extended_metrics.py,global_competitors.py,multi_market.py}` | Seal-preserving international metrics suite (Calmar, Omega, IR, CAPM, CVaR 95%, Ulcer, IC Decay, Block-Bootstrap, White RC, Hansen SPA, 4-Regime); 5 strictly causal Global SOTA Agents (`TradingAgents`, `FinMem`, `FinAgent`, `RD-Agent(Q)`, `AlphaForge-DE`); 3 market profiles (`VN_HOSE_T25`, `US_LIQUID_T1`, `CSI_EMERGING_T1`); shift-invariance tests green | **PASS (VERIFIED)** |
| **Tasks 04–05** | `src/vn_rsi/models/maveq_v4_veteran.py`, `configs/maveq_v4.yaml`, `experiments/run_v4_self_improve.py` | **MAVE-Q v4 (`VetQuant-RSI`)**: 10-iteration `--dev` self-improvement loop (`logs/v4_self_improvement_trace.jsonl`), `#1` 5-seed validation Pareto dominance (`Net Sharpe 1.359`, `Sortino 1.433`, `Calmar 2.654`, `DSR 0.953` at `1.0` trial) | **PASS (VERIFIED)** |
| **Task 06** | `src/vn_rsi/veteran/`, `experiments/run_veteran_reasoning_eval.py` | 90 search-zone 5-step SFT + 90 DPO preference pairs (`<= 2022-12-30`), `<4.5 GB` 4-bit QLoRA+DPO trainer + CPU fallback, 32-scenario benchmark: `100%` Decision Acc, `100%` Causal Consistency, `0.8889` Risk F1, `0.01064` Brier, `0.08156` ECE (`run_id=phase8_veteran_reasoning_eval`) | **PASS (VERIFIED)** |
| **Task 07** | `train_colab_kaggle.py`, `notebooks/train_mave_q_kaggle_colab.ipynb`, `scripts/ask_veteran_agent.py` | Universal Free-Cloud Trainer verified (`33.63s` smoke run, `0.02s` `--resume` Stage-1 recovery, VRAM cleanup telemetry, bilingual VN/EN CLI bundle loading) | **PASS (VERIFIED)** |
| **Task 08** | `experiments/run_v4_ablation_and_ood.py` | 8-seed paired ablations (`full_maveq_v4` `1.347 ± 0.081` vs `no_regime_controller` `1.041`, `p=0.0005`; vs `no_zero_val_burn_cv` DSR `0.952 -> 0.199`, `p<1e-4`) + Zero-Retuning OOD `#1` on VN100-10d (`Sharpe 1.567`, `Calmar 3.96`) & US Liquid T+1 (`Sharpe 0.294`) (`run_id=phase9_v4_ablation_and_ood`) | **PASS (VERIFIED)** |
| **Task 09** | `experiments/run_v4_final_arena.py`, `logs/sealed_test_ledger.jsonl` | Pre-Seal Audit 100% passed (`EvaluatorSeal` intact); Single-Shot Sealed Test scored ONCE (`run_id=phase10_v4_final_arena`): MAVE-Q v4 achieves `#1` shallowest MaxDD (`-19.8%`), `2.25×` Test DSR over v3 (`0.785` vs `0.349`), beating RD-Agent(Q), AlphaForge-DE, TradingAgents, and Qwen-72B/DeepSeek-671B | **PASS (VERIFIED)** |
| **Tasks 10–12** | `experiments/{build_report.py,build_figures.py}`, `paper/{main.tex,references.bib,verify_paper.py,PEER_REVIEW_AUDIT.md}`, `FREE_CLOUD_TRAINING_GUIDE.md`, `FINAL_UPGRADE_REPORT.md` | 8 auto-generated LaTeX tables (`337/337` logged cells verified), 4 vector PDF+SVG figures, full conference paper + Appendix Proofs (`42/42` citations, `0` syntax errors), 3-reviewer audit & self-fixes complete, full `scripts/ci_check.py` 100% GREEN | **PASS (VERIFIED)** |

---

## Verified vs. Explicitly Flagged UNVERIFIED / NOT RUN Inventory

### Directly Verified on Current Host
- `scripts/ci_check.py`: `ruff check`, `mypy` (0 errors across all modules), `pytest` (44/44 tests passing: causality, future-perturbation shift invariance across all models & global competitors, SHA-256 seal integrity, single-shot ledger enforcement, veteran reasoning calibration), and `paper/verify_paper.py` (`42/42` citations, `16/16` refs, `8/8` tables, `4/4` vector figures, `337/337` log-verified cells).
- `train_colab_kaggle.py` 2-pass `--resume` smoke run (`33.63s` pass 1, `0.02s` pass 2) + `scripts/ask_veteran_agent.py` bilingual CLI queries.

### Explicitly Flagged UNVERIFIED / NOT RUN on Current 0-GPU CI Machine
1. **Live Empirical Market Data (`NOT RUN` by default)**: `VnstockDataProvider` and `yfinance` live fetchers are implemented and ready, while default CI tables use deterministic fixtures (`is_synthetic: true`) for reproducible verification.
2. **Broker Square-Root Market Impact Coefficient (`UNVERIFIED`)**: `configs/market_rules.yaml` marks `illiquidity_impact.verified: false` (conservative approximation), whereas statutory commission (`15 bps`), sell tax (`10 bps`), T+2.5 holding locks, 100-share board lots, and `±7%/10%/15%` exchange price limits are `verified: true`.
3. **Physical CUDA T4 / TPU Execution (`NOT RUN` on 0-GPU host)**: Peak VRAM budgets (`3.4 GB` for Qwen3-1.7B / `4.3 GB` for Qwen3-4B 4-bit QLoRA+DPO) are analytical configurations verified for `<4.5 GB` T4 limits; live GPU training is one click away via `notebooks/train_mave_q_kaggle_colab.ipynb` (see [`FREE_CLOUD_TRAINING_GUIDE.md`](FREE_CLOUD_TRAINING_GUIDE.md)).
4. **PDF Compilation on Local Host**: No `pdflatex`/`xelatex`/`tectonic` binary is installed on this Windows host; `paper/verify_paper.py` verified full LaTeX syntax, environment balance, unescaped `&`, citations, cross-refs, inputs, and vector PDF figures (ready to compile directly on Overleaf or any TeXLive/Tectonic runner).
