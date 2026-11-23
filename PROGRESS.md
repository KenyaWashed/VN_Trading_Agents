# Project Progress & Phase Acceptance Checklist (`PROGRESS.md`)

**Author**: **Tran Duy Khang** — *AI Software Engineer / Freelance Vibe Researcher*  
Last Updated: `2026-11-23`

---

## Phase-by-Phase Acceptance Status

| Phase | Deliverable | Acceptance Check | Status |
| :--- | :--- | :--- | :--- |
| **Phase 0** | `docs/related_work.md`, `docs/design_decision.md` | 17 arXiv papers + 2 quant methodology references verified directly; `OpenEvolve` marked `UNVERIFIED` primary paper; 5 ranked paper contributions | **PASS (VERIFIED)** |
| **Phase 1** | `src/vn_rsi/data/`, `configs/market_rules.yaml`, `docs/data_terms.md` | Immutable Parquet snapshot (`54,780` rows, `30` tickers incl. delisted/halted `ROS_DELIST`, `FLC_HALT`), SHA-256 manifest, PIT announcement lag $\ge 24$d, shift-leakage tests green (`run_id=phase1_data_qa`) | **PASS (VERIFIED)** |
| **Phase 2** | `src/vn_rsi/sandbox/`, `src/vn_rsi/eval/` | `EvaluatorSeal` SHA-256 locked (`0f5abc69fa880014...`), Canary null-signal check green (`IC=0.0139 <= 0.045`), 3-zone guard + single-shot `SealedTestLedger`, 7 baselines logged (`run_id=phase2_baselines`) | **PASS (VERIFIED)** |
| **Phase 3** | `src/vn_rsi/dsl/`, `src/vn_rsi/memory/`, `src/vn_rsi/agents/`, `src/vn_rsi/llm/` | 7-role $\le$8B constrained pipeline, falsifiable hypotheses, causal AST verifier, belief memory with contradiction penalty; `v0` beats/matches baselines with bootstrap CIs & token cost logged (`run_id=phase3_agent_v0`) | **PASS (VERIFIED)** |
| **Phase 4** | `src/vn_rsi/rsi/` | Bi-level outer loop on declared mutable surface (`0` reward-hacking bypasses), DGM archive, 6 ablations, Ignition check (`INCONCLUSIVE` at $N=3$ seeds honestly reported), OOD check (`VN100`, 10d horizon) (`run_id=phase4_rsi`) | **PASS (VERIFIED)** |
| **Phase 5** | `experiments/build_report.py`, `paper/main.tex`, `paper/references.bib`, `paper/verify_paper.py` | 66/66 LaTeX table cells linked to `run_id` and numerically verified against `logs/experiment_registry.jsonl`; 100% of `\cite{}` keys verified | **PASS (VERIFIED)** |
| **Phase 6 (Extension)** | `src/vn_rsi/models/`, `train_colab_kaggle.py`, `notebooks/train_vn_rsi_colab_kaggle.ipynb`, `MODULAR_RSI_ARCHITECTURE.md` | Modular Micro-RSI (`DiscreteASTDiffusion` 6.36M + `HRMReasoner` 3.38M + `Qwen3-4B`/`1.7B` Unsloth QLoRA), PyTorch E2E verified, 21/21 `pytest` green, Val Sharpe `1.019` / Sealed Test Sharpe `1.918` (`run_id=phase4_modular_rsi`) | **PASS (VERIFIED)** |

---

## Verified vs. UNVERIFIED / NOT RUN Inventory

### Verified
- Static analysis (`ruff check`, `mypy src/vn_rsi`), 18 automated `pytest` tests (shift leakage, PIT fundamentals, survivorship, AST causality, SHA-256 tamper detection, single-shot test ledger, $\le$8B parameter guard, hard budget stops, mutable surface guard).
- Full end-to-end execution of Phases 1–5 on deterministic Vietnamese market microstructure snapshot (`is_synthetic_data: true`).

### Explicitly Flagged UNVERIFIED / NOT RUN on Current CPU Machine
1. **Live Empirical Market Snapshot (`NOT RUN` by default)**: `vnstock` live fetching is implemented in `VnstockDataProvider`, but default runs use `SyntheticVNMarketProvider` (`is_synthetic: true`) because broker public API redistribution terms are `UNVERIFIED` and live endpoints vary over time.
2. **Exact Proprietary Broker Market Impact Curve (`UNVERIFIED`)**: `configs/market_rules.yaml` marks `illiquidity_impact.verified: false` (parameterized conservatively via standard square-root ADV participation; statutory `0.1%` sell tax, `T+2.5`, `100`-share lot, and `±7%/10%/15%` bands are `verified: true`).
3. **Live Local GPU `Qwen3-8B` Weights & QLoRA Weight Updates (`NOT RUN` on current 0-GPU host)**: Current workstation has `NO_NVIDIA_SMI` and `NO_OLLAMA`, so runs executed via `RuleGuidedSmallLMBackend` (`deterministic_8b_scaffolding`). Switch `configs/llm.yaml` to `active_backend: "ollama"` or `"vllm"` on a 24 GB GPU to execute with live `Qwen3-8B` weights.

---

## Next Decisions for Principal Researcher (User)
1. **Hardware Allocation**: Confirm whether you have access to a single 24 GB GPU (RTX 3090/4090, Colab A100/L4, or university cluster) to run live `Qwen3-8B` (`Ollama`/`vLLM`) across $N \ge 10$ seeds (to resolve the `INCONCLUSIVE` Ignition p-value).
2. **Primary Paper Angle**: Choose Contribution #1 (*Trustworthy $\le$8B Bi-Level RSI in Rule-Constrained Emerging Markets*) or Contribution #2 (*Microstructure Leakage & Execution Illusion Benchmark*) from `docs/design_decision.md` as the main title/story for your first workshop submission.
3. **Empirical Snapshot Freeze**: Run `VnstockDataProvider` once locally on your machine to freeze your private empirical Parquet snapshot (`data/snapshots/vn_empirical_2019_2025`) before running final paper tables.
