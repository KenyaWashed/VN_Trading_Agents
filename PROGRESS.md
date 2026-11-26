# Project Progress & Phase Acceptance Checklist (`PROGRESS.md`)

**Author**: **Tran Duy Khang** — *AI Software Engineer / Freelance Vibe Researcher*  
Last Updated: `2026-11-24`

---

## Phase-by-Phase Acceptance Status

| Phase | Deliverable | Acceptance Check | Status |
| :--- | :--- | :--- | :--- |
| **Phase 0** | `docs/related_work.md`, `docs/design_decision.md` | 17+ arXiv papers + quant methodology references verified directly; 5 ranked paper contributions | **PASS (VERIFIED)** |
| **Phase 1** | `src/vn_rsi/data/`, `configs/market_rules.yaml`, `docs/data_terms.md` | Immutable Parquet snapshot (`54,780` rows, `30` tickers incl. delisted/halted `ROS_DELIST`, `FLC_HALT`), SHA-256 manifest, PIT announcement lag $\ge 24$d, shift-leakage tests green (`run_id=phase1_data_qa`) | **PASS (VERIFIED)** |
| **Phase 2** | `src/vn_rsi/sandbox/`, `src/vn_rsi/eval/` | `EvaluatorSeal` SHA-256 locked, Canary null-signal check green (`IC=0.0139 <= 0.045`), 3-zone guard + single-shot `SealedTestLedger`, 7 baselines logged (`run_id=phase2_baselines`) | **PASS (VERIFIED)** |
| **Phase 3** | `src/vn_rsi/dsl/`, `src/vn_rsi/memory/`, `src/vn_rsi/agents/`, `src/vn_rsi/llm/` | 7-role $\le$8B constrained pipeline, falsifiable hypotheses, causal AST verifier, belief memory with contradiction penalty (`run_id=phase3_agent_v0`) | **PASS (VERIFIED)** |
| **Phase 4** | `src/vn_rsi/rsi/` | Bi-level outer loop on declared mutable surface (`0` reward-hacking bypasses), DGM archive, 6 ablations, OOD check (`VN100`, 10d horizon) (`run_id=phase4_rsi`) | **PASS (VERIFIED)** |
| **Phase 5** | `experiments/build_report.py`, `paper/main.tex`, `paper/references.bib`, `paper/verify_paper.py` | 181/181 LaTeX table cells linked to `run_id` and numerically verified against `logs/experiment_registry.jsonl`; 30/30 `\cite{}` keys verified | **PASS (VERIFIED)** |
| **Phase 6** | `src/vn_rsi/models/`, `experiments/run_modular_rsi.py`, `experiments/run_frontier_combat.py` | Modular Micro-RSI (`DiscreteASTDiffusion` + `HRMReasoner`) & Frontier Combat Arena vs 70B–671B (`run_id=phase4_modular_rsi`, `phase5_frontier_combat`) | **PASS (VERIFIED)** |
| **Phase 7** | `src/vn_rsi/models/holo_maveq.py`, `alpha_mining_baselines.py`, `cloud_presets.py`, `train_colab_kaggle.py`, `modal_train.py`, `notebooks/train_mave_q_kaggle_colab.ipynb` | Flagship **Holo-MAVE-Q v3** beats 7 same-topic alpha miners + 3 Frontier 70B–671B + v1/v2 (`Val Sharpe 1.291`, `Sealed Test Sharpe 1.676`, `48.8%/yr`, `DSR 0.349`, 81 trials); 5-tier free cloud presets (`cpu_fast` → `kaggle_max_100m` → `cloud_a100_350m` / `tpu_v5e_350m`), AMP FP16/BF16, `torch_xla` TPU, checkpoint save/load, Unsloth→PEFT QLoRA fallback, 31/31 `pytest` green | **PASS (VERIFIED)** |

---

## Verified vs. UNVERIFIED / NOT RUN Inventory

### Verified
- Static analysis (`ruff check`, `mypy src/vn_rsi`), 31 automated `pytest` tests (shift leakage, PIT fundamentals, survivorship, AST causality, SHA-256 tamper detection, single-shot test ledger, $\le$8B parameter guard, hard budget stops, mutable surface guard, `TypedASTGuard`, `Holo-MAVE-Q v3`, all 7 same-topic competitors, cloud presets & checkpoint round-trip).
- Full end-to-end execution of Phases 1–7 on deterministic Vietnamese market microstructure snapshot (`is_synthetic_data: true`), plus `train_colab_kaggle.py --preset cpu_fast --mode holo_v3 --smoke-test`.

### Explicitly Flagged UNVERIFIED / NOT RUN on Current Local CPU Machine
1. **Live Empirical Market Snapshot (`NOT RUN` by default)**: `vnstock` live fetching is implemented in `VnstockDataProvider`, but default runs use `SyntheticVNMarketProvider` (`is_synthetic: true`) because broker public API redistribution terms are `UNVERIFIED` and live endpoints vary over time.
2. **Exact Proprietary Broker Market Impact Curve (`UNVERIFIED`)**: `configs/market_rules.yaml` marks `illiquidity_impact.verified: false` (parameterized conservatively via standard square-root ADV participation; statutory `0.1%` sell tax, `T+2.5`, `100`-share lot, and `±7%/10%/15%` bands are `verified: true`).
3. **Live Cloud GPU / TPU Run (`NOT RUN` on current 0-GPU host)**: Ready to execute immediately via `notebooks/train_mave_q_kaggle_colab.ipynb` (Kaggle `T4 x2` / `TPU v5e-8` / Colab `T4`) or `modal run modal_train.py --preset cloud_a100_350m --enable-slm-qlora` (Modal `$30/mo` free `A100-80GB`).
