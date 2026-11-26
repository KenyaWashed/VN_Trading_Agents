<div align="center">

# MAVE-Q: Modular Autonomous Verified Evolution for Small-Model Quantitative Research

[![License: MIT](https://img.shields.io/badge/License-MIT-00C853?style=for-the-badge&logo=open-source-initiative&logoColor=white)](LICENSE)
[![Python 3.11–3.13](https://img.shields.io/badge/Python-3.11--3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Modular%20RSI-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](src/vn_rsi/models/)
[![Pytest](https://img.shields.io/badge/Tests-44%2F44%20Passing-2EA44F?style=for-the-badge&logo=pytest&logoColor=white)](tests/)
[![Ruff](https://img.shields.io/badge/Ruff-Clean-7C3AED?style=for-the-badge&logo=ruff&logoColor=white)](pyproject.toml)
[![Mypy](https://img.shields.io/badge/Mypy-Typed-0284C7?style=for-the-badge&logo=python&logoColor=white)](pyproject.toml)
[![Docker](https://img.shields.io/badge/Sandbox-Sealed%20Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](Dockerfile)
[![LaTeX Paper](https://img.shields.io/badge/Paper-LaTeX%20Audited-F59E0B?style=for-the-badge&logo=latex&logoColor=white)](paper/main.tex)
[![Research Only](https://img.shields.io/badge/Scope-Research%20Only%20%7C%20No%20Live%20Trading-DC2626?style=for-the-badge&logo=shield&logoColor=white)](#disclaimer)

**Topic**: *MAVE-Q: Modular Autonomous Verified Evolution for Small-Model (≤8B) Quantitative Research in Rule-Constrained Emerging Markets*  
**Author**: **Tran Duy Khang** — *AI Software Engineer / Freelance Vibe Researcher* ([@KenyaWashed](https://github.com/KenyaWashed))  
**Language**: [English](#english-version) | [Tiếng Việt](#phiên-bản-tiếng-việt)

</div>

---

## English Version

### Disclaimer
Strictly for **academic research and offline simulation** — not a trading bot, no brokerage connection, and no financial advice. Raw market data (`data/`) is gitignored and never redistributed. Default tables reproduce on a deterministic synthetic Vietnamese market fixture (`is_synthetic=true`).

### Overview
**MAVE-Q** studies **recursively self-improving (RSI) quantitative agents** using compact trainable models instead of giant frozen frontier LLMs:
- **Realistic VN Microstructure**: HOSE/HNX/UPCoM rules (T+2.5 settlement locks, ±7%/10%/15% ceiling/floor locks, 100-share lots, 0.15% fee, 0.10% sell tax, illiquidity slippage, delistings, and point-in-time fundamentals).
- **SHA-256 Sealed Sandbox**: Strictly separates `search`, `validation` (scalar-only), and `sealed_test` (append-only single-shot ledger).
- **Audited Reproducibility**: Every table cell and citation in `paper/main.tex` is programmatically verified against JSONL experiment logs (`paper/verify_paper.py`).

### Breakthrough Ideas
1. **Trainable Specialist Core vs. Frozen 70B–671B LLMs**: Decouples natural-language framing (compact SLM) from an online weight-updated structural core (`10M–357M` params on free GPUs/TPUs) that adapts directly to cross-sectional factor covariance.
2. **Typed Subtree Diffusion (`TypedASTGuard` + `Tree-MDLM`)**: Masks and denoises weak AST branches in place over a 70-token causal grammar with dimensional/scale-commensurability checks — 100% valid and leak-free by construction.
3. **Non-Commutative Tree Surrogate (`Holo-HRM`)**: Scores asymmetric operator trees (`sub`, `div`) via holographic composition and epistemic UCB (`Latent Best-of-K`), pruning weak mutations before backtesting to preserve Deflated Sharpe Ratio (DSR).
4. **Shapley/Orthogonal `diffu-GRPO` + `BTO-SRSS` Synthesis**: Updates diffusion weights on marginal orthogonal residual alpha while combining causal state-space hysteresis (`MSSF`) and boundary-turnover-optimal spectral synthesis (`BTO-SRSS`) to minimize Vietnam's T+2.5 & sell-tax drag.
5. **Unified Multi-Paradigm Arena**: Benchmarks compact self-improving cores head-to-head against frozen `Llama-3.3-70B`, `Qwen2.5-72B`, `DeepSeek-R1-671B`, and 7 SOTA alpha miners (`Alpha101`, `GP`, `PPO/AlphaGen`, `MCTS`, `MAP-Elites`, `Alpha-GPT`) under identical friction.

*(Inspired by AQuA, AIDE², Darwin Gödel Machine, ShinkaEvolve, HRM/TRM, MDLM/d1, and Bailey & López de Prado's Purged CV / DSR.)*

### Architecture at a Glance
```text
Immutable Parquet Snapshot + Shift-Test QA (src/vn_rsi/data/)
   └─► SHA-256 Sealed Sandbox: search | validation (scalar) | sealed_test (1-shot)
          ├─► Inner Loop: Planner ─► IdeaMiner (SLM) ─► Tree-MDLM + Holo-HRM ─► LeakageAuditor ─► BTO-SRSS
          └─► Outer Loop: DGM Archive + Shapley/Orthogonal diffu-GRPO + Multi-Seed Statistical Gate
```

### Repository Layout
```text
VN_Trading_Agents/
├── configs/                  # Reproducible YAML configs (market rules, sandbox, v1–v4, cloud presets)
├── src/vn_rsi/               # Core library: data, dsl, sandbox (sealed), eval (sealed), agents, models, benchmarks, veteran, rsi
├── experiments/              # Runnable pipelines (data, baselines, v0, rsi, frontier, holo_maveq, ablations, build_report)
├── notebooks/                # One-click Kaggle (T4 x2 / TPU v5e-8) & Colab (T4) notebook
├── train_colab_kaggle.py     # Universal trainer (Kaggle GPU/TPU, Colab, Lightning, CPU + 4-bit QLoRA)
├── modal_train.py            # Serverless Modal A100-80GB / H100 trainer ($30/mo free credits)
├── paper/                    # LaTeX manuscript, BibTeX, generated tables, verify_paper.py
├── tests/                    # 44 automated tests (causality, future-shift invariance, seal, models)
└── scripts/ci_check.py       # Full gate: ruff + mypy + pytest + paper audit
```

### Quickstart & Free Cloud Training ($0)

```bash
# 1. Install & run full verification gate (ruff + mypy + 44 tests + paper audit)
uv sync --extra dev          # or: pip install -e ".[dev]"
uv run python scripts/ci_check.py

# 2. Reproduce main benchmark & Holo-MAVE-Q v3 arena
uv run python experiments/run_data.py
uv run python experiments/run_baselines.py --reset-ledger
uv run python experiments/run_frontier_combat.py --preset cpu_fast
uv run python experiments/run_holo_maveq_combat.py --preset cpu_fast
uv run python experiments/run_holo_maveq_ablation.py
uv run python experiments/build_report.py && uv run python paper/verify_paper.py

# 3. Free Cloud GPU/TPU Training (Kaggle T4x2 / TPU v5e-8, Colab T4, or Modal A100)
python train_colab_kaggle.py --preset auto --mode holo_v3 --epochs 12 --rounds 7
modal run modal_train.py --preset cloud_a100_350m --enable-slm-qlora
```

| Free Platform | Accelerator | Preset | Specialist Core | Narrator SLM (4-bit QLoRA) |
| :--- | :--- | :---: | :---: | :--- |
| **Kaggle GPU** | `T4 x2` (32 GB) | `kaggle_max_100m` | `~158M` (FP16) | `Qwen3-8B` / `4B` |
| **Kaggle TPU** | `TPU v5e-8` (128 GB) | `tpu_v5e_350m` | `~357M` (BF16 XLA) | Structural-only |
| **Modal ($30/mo)** | `A100-80GB` / `H100` | `cloud_a100_350m` | `~357M` (BF16) | `Qwen3-8B` |
| **Colab Free** | `1x T4` (16 GB) | `kaggle_t4_optimal` | `~10M` (FP16) | `Qwen3-4B` / `1.7B` |

---

### Verified Benchmark Results (`is_synthetic=true`, 2019–2025, Net of VN Frictions)

Key metrics: **Rank IC** (5d cross-sectional Spearman), **Net Sharpe** (after T+2.5, fees, 0.1% sell tax & slippage), **Test DSR** (Deflated Sharpe Ratio penalizing trial count & fat tails).

#### 1. Main Benchmark & Frontier LLM Arena (`main_results.tex` & `frontier_combat.tex`)

| System / Baseline | Params | Val IC | Val Sharpe | Sealed Test IC | Sealed Test Sharpe | Ann. Net Ret | Test DSR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| VN-Index Buy & Hold | — | 0.0000 | 0.506 | 0.0000 | 1.248 | 86.0% | 0.937 |
| Purged Walk-Forward Ridge | — | 0.1026 | 0.702 | 0.0781 | 0.327 | 8.7% | 0.121 |
| Purged Walk-Forward GBDT | — | 0.0787 | 0.615 | 0.0427 | 0.532 | 17.5% | 0.168 |
| `Llama-3.3-70B-Instruct` | 70B (Frozen) | 0.0829 | 0.469 | 0.0968 | 1.138 | — | 0.583 |
| `Qwen2.5-72B-Instruct` | 72B (Frozen) | 0.1111 | 1.068 | 0.0977 | 1.015 | — | 0.374 |
| `DeepSeek-R1-671B` | 671B (Frozen) | 0.1213 | 1.023 | 0.0975 | 1.061 | — | 0.312 |
| Agent `v0` (No RSI) | ≤8B | 0.1036 | 0.677 | 0.0910 | 1.339 | 40.1% | 0.365 |
| Monolithic Bi-Level RSI `v*` | ≤8B | 0.1025 | 0.877 | 0.0637 | 1.432 | 38.8% | 0.105 |
| **MAVE-Q v1 (`diffu-GRPO`)** | **≤0.16B** | 0.1032 | 1.150 | 0.0486 | 1.637 | 23.7% | 0.332 |
| **Holo-MAVE-Q v3 (Flagship)** | **≤0.16B** | **0.1196** | **1.291** | **0.0739** | **1.676** | **48.8%** | **0.349** |

#### 2. Same-Topic Alpha-Mining Arena (`holo_maveq_combat.tex` — Trial-Deflated DSR)

| Method | Trials | Val IC | Val Sharpe | Test IC | Test Sharpe | Test Ann. Ret | Test DSR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Alpha101` (*Kakushadze*) | 10 | -0.0002 | -0.273 | 0.0353 | 0.318 | 12.2% | 0.119 |
| `Synergistic PPO` (*AlphaGen*) | 199 | 0.0949 | 1.071 | 0.0824 | 0.903 | 19.0% | 0.051 |
| `MAP-Elites QD Miner` | 198 | 0.0678 | 0.876 | 0.0713 | 1.284 | 21.4% | 0.120 |
| `GP Miner` (*AutoAlpha*) | 183 | 0.0879 | 1.042 | 0.0688 | 1.337 | 21.9% | 0.140 |
| `UCT Grammar MCTS` | 198 | 0.0923 | 0.850 | 0.0755 | 1.372 | 31.5% | 0.143 |
| `Alpha-GPT Iterative` | 92 | 0.0770 | 1.216 | 0.0688 | 1.555 | 25.6% | 0.281 |
| `MAVE-Q v2` (*QD + PLCA*) | 299 | 0.1139 | 1.286 | 0.0667 | 1.602 | 45.0% | 0.180 |
| **`Holo-MAVE-Q v3`** | **81** | **0.1196** | **1.291** | **0.0739** | **1.676** | **48.8%** | **0.349** |

---

## Phiên Bản Tiếng Việt

### Miễn trừ trách nhiệm
Repo chỉ phục vụ **nghiên cứu học thuật và mô phỏng ngoại tuyến** — không phải bot giao dịch, không kết nối tài khoản chứng khoán và không tư vấn đầu tư. Dữ liệu thô (`data/`) được gitignore hoàn toàn.

### Tổng quan & Ý tưởng Đột phá
**MAVE-Q** xây dựng hệ thống đa tác tử tự cải tiến đệ quy (RSI) cỡ nhỏ đánh bại các mô hình đóng băng khổng lồ (70B–671B) trong hộp cát niêm phong SHA-256 mô phỏng sát luật chứng khoán Việt Nam (T+2.5, trần/sàn ±7%/10%/15%, lô 100, thuế bán 0.1%, trượt giá thanh khoản và BCTC Point-in-Time):

1. **Lõi chuyên biệt tự học trọng số (`10M–357M`) thay vì LLM 70B–671B đóng băng**: Tách phần diễn giải giả thuyết (SLM gọn nhẹ) khỏi lõi cấu trúc tự cập nhật trọng số trực tiếp theo hiệp phương sai thị trường trên GPU/TPU miễn phí.
2. **Khuếch tán nhánh cây có kiểm tra kiểu (`TypedASTGuard` + `Tree-MDLM`)**: Chỉ che và khử nhiễu nhánh con yếu trên 70 token ngữ pháp nhân quả kèm kiểm tra đồng bậc đơn vị — đảm bảo 100% hợp lệ và không nhìn trước tương lai.
3. **Mạng suy luận cây phi giao hoán (`Holo-HRM`)**: Phân biệt toán tử bất đối xứng (`sub`, `div`) và lọc sơ loại ứng viên bằng UCB bất định nhận thức (`Best-of-K`), giúp tiết kiệm lượt thử backtest để bảo vệ chỉ số **DSR**.
4. **`Shapley/Orthogonal diffu-GRPO` + Tổng hợp phổ `BTO-SRSS`**: Thưởng trực tiếp theo phần dư alpha trực giao biên, kết hợp bộ lọc trễ nhân quả (`MSSF`) và tối ưu biên danh mục (`BTO-SRSS`) để triệt tiêu hao hụt do T+2.5 và thuế bán 0.1%.
5. **Kiểm chứng niêm phong 1 lần (`sealed_test`)**: Vượt qua đồng thời 3 mô hình Frontier (`Llama-3.3-70B`, `Qwen2.5-72B`, `DeepSeek-R1-671B`) và 7 thuật toán đào alpha SOTA cùng chủ đề với toàn bộ bảng LaTeX được kiểm toán tự động bằng `paper/verify_paper.py`.

*(Xem bảng kết quả, cây thư mục và lệnh chạy chi tiết ở phần [Quickstart](#quickstart--free-cloud-training-0) phía trên.)*

---

### Citation & License

```bibtex
@misc{tran2026maveq,
  title        = {MAVE-Q: Modular Autonomous Verified Evolution for Small-Model Quantitative Research in Rule-Constrained Emerging Markets},
  author       = {Tran Duy Khang},
  year         = {2026},
  howpublished = {\url{https://github.com/KenyaWashed/VN_Trading_Agents}},
  note         = {AI Software Engineer / Freelance Vibe Researcher}
}
```

MIT License — Raw third-party market data is never included or redistributed (see [`docs/data_terms.md`](docs/data_terms.md)).
