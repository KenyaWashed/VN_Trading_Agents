<div align="center">

# MAVE-Q v4 (`VetQuant-RSI`): Veteran-Grounded Neuro-Symbolic Quantitative Research & Free-Cloud Autonomous Evolution

[![License: MIT](https://img.shields.io/badge/License-MIT-00C853?style=for-the-badge&logo=open-source-initiative&logoColor=white)](LICENSE)
[![Python 3.11–3.13](https://img.shields.io/badge/Python-3.11--3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/Free%20Cloud-%3C4.5GB%20VRAM%20T4-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](FREE_CLOUD_TRAINING_GUIDE.md)
[![Pytest](https://img.shields.io/badge/Tests-44%2F44%20Passing-2EA44F?style=for-the-badge&logo=pytest&logoColor=white)](tests/)
[![Ruff](https://img.shields.io/badge/Ruff-Clean-7C3AED?style=for-the-badge&logo=ruff&logoColor=white)](pyproject.toml)
[![Mypy](https://img.shields.io/badge/Mypy-Typed-0284C7?style=for-the-badge&logo=python&logoColor=white)](pyproject.toml)
[![LaTeX Paper](https://img.shields.io/badge/Paper-337%2F337%20Cells%20Audited-F59E0B?style=for-the-badge&logo=latex&logoColor=white)](paper/main.tex)
[![Research Only](https://img.shields.io/badge/Scope-Research%20Only%20%7C%20No%20Live%20Trading-DC2626?style=for-the-badge&logo=shield&logoColor=white)](#disclaimer)

**Paper**: *MAVE-Q v4 (VetQuant-RSI): Veteran-Grounded Neuro-Symbolic Quantitative Research with Zero-Validation-Burn Evolution under Emerging-Market Microstructure Constraints*  
**Author**: **Tran Duy Khang** — *AI Software Engineer / Independent Quantitative AI Researcher* ([@KenyaWashed](https://github.com/KenyaWashed))  
**Guides**: [Free Cloud Training Guide](FREE_CLOUD_TRAINING_GUIDE.md) | [Final Upgrade Report](FINAL_UPGRADE_REPORT.md) | [Peer-Review Audit](paper/PEER_REVIEW_AUDIT.md)  
**Language**: [English](#english-version) | [Tiếng Việt](#phiên-bản-tiếng-việt)

</div>

---

## English Version

### Disclaimer
Strictly for **academic research and offline simulation** — NOT a trading bot, zero brokerage connectivity, and zero investment advice. Raw market data (`data/`) is gitignored and never redistributed. Default CI tables reproduce on deterministic synthetic microstructure snapshots (`is_synthetic=true`); live `vnstock` and `yfinance` modes are included for empirical replication.

### What's New in MAVE-Q v4 (`VetQuant-RSI`)
1. **`VeteranCaseMemory` (Episodic Microstructure & Trauma k-NN Retriever)**: Strictly causal 10-dim microstructure state vector fit exclusively on `search` (`<= 2022-12-30`) with unit-safe $\pm 10$d self-match purging and human-readable historical analog retrieval.
2. **`Tri-Cameral Pre-Mortem Core`**: Unifies (1A) `TypedASTGuard` + `Tree-MDLM`/`Bridge-MDLM` + `Holo-HRM` + Nyquist-clean `MSSF` + `BTO-SRSS` (`Shapley-diffu-GRPO`), (1B) a Gârleanu–Pedersen friction-adjusted KKT non-negative Ridge backbone (`GP-FA-Ridge`), and (2) a causal Devil's Advocate Pre-Mortem auditor penalizing T+2.5 floor-lock cascades, blow-off traps, ceiling locks, and distress drift.
3. **Scale-Invariant `Hierarchical Regime-Conviction BAB Controller`**: Integrates Vasicek-shrunk Frazzini–Pedersen Betting-Against-Beta (`BAB`) risk tilting, no-trade-band incumbent hysteresis, and a zero-whipsaw instant-release tail circuit breaker inside the unmodified SHA-256 sealed simulator.
4. **`Zero-Validation-Burn Purged Search CV`**: Cuts official validation queries from `80 -> 1` trial, elevating Validation DSR to `0.952 ± 0.009` (`4.8×` Holo-MAVE-Q v3) and Sealed Test DSR to `0.785` (`2.25×` Holo-MAVE-Q v3).
5. **10-Year Veteran Cognitive Reasoning Engine + 100% Free-Cloud Trainer (`<4.5 GB` VRAM)**: Mines verified 5-step CIO SFT & Veteran-vs-FOMO DPO preference pairs strictly from `search`, supports 4-bit QLoRA+DPO (`Qwen3-1.7B/4B`) on free Colab/Kaggle T4 GPUs with `--resume` checkpointing and instant CPU Neuro-Symbolic fallback (`scripts/ask_veteran_agent.py`).
6. **24-Competitor Global SOTA & Multi-Market Arena**: Strictly causal implementations of **TradingAgents**, **FinMem**, **FinAgent**, **RD-Agent(Q)**, and **AlphaForge-DE** alongside 7 traditional baselines, 7 alpha miners, 3 frozen 70B–671B frontiers, and 3 market profiles (`VN_HOSE_T25`, `US_LIQUID_T1`, `CSI_EMERGING_T1`).

### Repository Layout
```text
VN_Trading_Agents/
├── configs/                       # Reproducible YAML configs (market rules, sandbox, v1–v4, cloud presets)
├── src/vn_rsi/                    # data, dsl, sandbox (SHA-256 sealed), eval (sealed), benchmarks, models, veteran, agents, rsi
├── experiments/                   # v4 self-improve, 8-seed ablation & OOD, single-shot sealed arena, veteran eval, build_report, build_figures
├── scripts/                       # ci_check.py (full gate), ask_veteran_agent.py (bilingual 5-step CIO CLI)
├── notebooks/                     # One-click Free Colab (T4) / Kaggle (2x T4 / TPU v5e-8) notebook
├── train_colab_kaggle.py          # Universal Free-Cloud Trainer (<4.5GB VRAM, --resume recovery, VRAM telemetry)
├── modal_train.py                 # Serverless Modal A100-80GB / H100 trainer ($30/mo free credits)
├── paper/                         # Full conference paper (main.tex, 42 BibTeX refs, 8 auto tables, 4 vector PDF+SVG figs, verify_paper.py, PEER_REVIEW_AUDIT.md)
├── FREE_CLOUD_TRAINING_GUIDE.md   # Step-by-step $0 Colab / Kaggle / Modal training & CLI guide
├── FINAL_UPGRADE_REPORT.md        # Comprehensive before-vs-after v3 -> v4 upgrade summary
└── tests/                         # 44 automated tests (causality, future-shift invariance, SHA-256 seal, global arena, veteran reasoning)
```

### Quickstart & Natural-Language Veteran CLI
```bash
# 1. Install & run full verification gate (ruff + mypy + 44 tests + 337-cell paper audit)
uv sync --extra dev          # or: pip install -e ".[dev]"
python scripts/ci_check.py

# 2. Train MAVE-Q v4 + 10-Year Veteran QLoRA/DPO on Free Colab/Kaggle T4 (or instant local CPU)
python train_colab_kaggle.py --preset auto --mode maveq_v4 --enable-veteran-dpo --resume

# 3. Ask the 10-Year Veteran CIO Agent natural-language market questions (Vietnamese or English)
python scripts/ask_veteran_agent.py --symbol FPT --lang vi --question "Đánh giá cơ hội mua FPT và rủi ro T+2.5 hiện tại?"
python scripts/ask_veteran_agent.py --symbol HPG --lang en --json

# 4. Reproduce 8-seed ablations, 3-market OOD, veteran benchmark, LaTeX tables, & vector figures
python experiments/run_veteran_reasoning_eval.py
python experiments/run_v4_ablation_and_ood.py --seeds "11,22,33,44,55,66,77,2026" --workers 8
python experiments/build_report.py && python experiments/build_figures.py && python paper/verify_paper.py
```

### Verified Benchmark Highlights (`is_synthetic=true`, Net of VN T+2.5 & Friction)

#### 1. Validation Pareto Dominance (`N=8` Independent Seeds) & Zero-Retuning OOD (`v4_ablations_ood.tex`)
| Benchmark / Variant | Trials | Rank IC | Net Sharpe | Calmar | MaxDD | Turnover | DSR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full MAVE-Q v4 (VN30 Val, 8-Seed Mean)** | **1.0** | **0.1072** | **1.347 ± 0.081** | **2.61** | **-21.5%** | **0.0935** | **0.952** |
| w/o Regime-Conviction BAB Controller | 1.0 | 0.1063 | 1.041 ± 0.249 (`p=0.0005`) | 1.50 | -28.4% | 0.0821 | 0.893 |
| w/o Zero-Val-Burn Purged CV | 88.0 | 0.1063 | 1.319 ± 0.038 | 2.51 | -21.4% | 0.0940 | 0.199 (`p<1e-4`) |
| **OOD VN100 (10d, T+2.5) — MAVE-Q v4 (#1)** | **1** | 0.1088 | **1.567** | **3.96** | **-17.5%** | **0.0956** | **0.974** |
| OOD VN100 (10d, T+2.5) — RD-Agent(Q) | 1 | 0.1316 | 1.520 | 3.54 | -20.6% | 0.1815 | 0.970 |
| **OOD US Liquid (5d, T+1) — MAVE-Q v4 (#1)** | **1** | **0.1434** | **0.294** | **0.36** | **-26.9%** | **0.0512** | **0.642** |
| OOD US Liquid (5d, T+1) — Buy & Hold | 1 | 0.0000 | -0.780 | -0.58 | -59.5% | 0.0051 | 0.167 |

#### 2. Single-Shot Immutable Sealed Test Arena (`v4_global_arena.tex`, SHA-256 Seal `4a0f5ad3...31e`)
| Model / System | Trials | Test IC | Test Sharpe | Calmar | Ann. Ret | MaxDD | Test DSR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| VN-Index Buy & Hold | 1 | 0.0000 | 1.248 | 1.79 | 86.0% | -47.9% | 0.937 |
| `TradingAgents` (*Xiao et al.*) | 2 | 0.0014 | 0.168 | 0.11 | 4.7% | -41.1% | 0.377 |
| `AlphaForge-DE` (*Shi et al., AAAI*) | 2 | 0.0961 | 0.970 | 0.97 | 27.9% | -28.9% | 0.749 |
| `RD-Agent(Q)` (*Li et al., NeurIPS*) | 2 | 0.0988 | 1.040 | 1.04 | 30.5% | -29.2% | 0.775 |
| `DeepSeek-R1-671B` (Frozen) | -- | 0.0975 | 1.061 | 1.04 | 34.9% | -33.5% | 0.312 |
| `FinMem` (*Yu et al.*) | 2 | 0.0199 | 1.264 | 1.44 | 33.1% | -23.0% | 0.849 |
| `FinAgent` (*Zhang et al., KDD*) | 2 | 0.1064 | 1.514 | 1.79 | 43.9% | -24.6% | 0.911 |
| `Holo-MAVE-Q v3` (81 Trials) | 81 | 0.0739 | 1.676 | 2.21 | 48.8% | -22.0% | 0.349 |
| **`MAVE-Q v4` (`VetQuant-RSI`, Ours)** | **2** | 0.0654 | **1.067** | **1.23** | 24.4% | **-19.8% (#1)** | **0.785 (2.25× v3)** |

#### 3. 32-Scenario 10-Year Veteran Cognitive Reasoning Benchmark (`v4_veteran_reasoning.tex`)
| Reasoning System | Decision Acc | Causal Consist. | Hidden-Risk F1 | Brier Score ($\downarrow$) | ECE ($\downarrow$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Naive FOMO Agent | 53.12% | 84.38% | 0.0000 | 0.46515 | 0.50625 |
| Static Momentum Analyst | 84.38% | 100.00% | 0.0000 | 0.15670 | 0.19625 |
| Generic Rule Analyst | 84.38% | 100.00% | 0.2857 | 0.18295 | 0.20375 |
| **MAVE-Q v4 Veteran CIO (Ours)** | **100.00%** | **100.00%** | **0.8889** | **0.01064** | **0.08156** |

---

## Phiên Bản Tiếng Việt

### Miễn trừ trách nhiệm
Kho mã nguồn này chỉ phục vụ **nghiên cứu khoa học và mô phỏng ngoại tuyến** — không phải bot giao dịch, không kết nối tài khoản chứng khoán và không tư vấn đầu tư. Dữ liệu thô (`data/`) được gitignore hoàn toàn.

### Điểm Đột Phá của MAVE-Q v4 (`VetQuant-RSI`)
1. **Bộ nhớ Tình huống & Chấn thương (`VeteranCaseMemory`)**: Truy hồi k-NN nhân quả trên vector vi cấu trúc 10 chiều học hoàn toàn từ vùng `search`, có cơ chế loại trừ tự khớp $\pm 10$ ngày và trích dẫn ngày tương đồng lịch sử cho lời giải thích CIO.
2. **Lõi Phản biện Ba Viện (`Tri-Cameral Pre-Mortem Core`)**: Kết hợp (1A) tiến hóa biểu tượng có kiểm tra thứ nguyên (`TypedASTGuard` + `Tree-MDLM` + `Holo-HRM` + `MSSF` + `BTO-SRSS`), (1B) xương sống Ridge không âm điều chỉnh ma sát kiểu Gârleanu–Pedersen, và (2) Viện Phản biện Pre-Mortem phạt sớm rủi ro kẹt sàn T+2.5, bẫy kéo xả khối ngoại và cổ phiếu kiệt quệ.
3. **Bộ điều khiển Chế độ & Low-Beta bất biến thang đo (`Hierarchical Regime-Conviction BAB Controller`)**: Tích hợp phòng thủ Frazzini–Pedersen BAB (co giãn Vasicek) theo chế độ thị trường, vùng trễ chống đảo lệnh (`hysteresis`) và ngắt mạch tức thời không kẹt đáy chữ V.
4. **Zero-Validation-Burn Purged Search CV**: Giảm số lượt gọi validation từ `80 -> 1`, nâng DSR trên validation lên `0.952` (`4.8×` v3) và DSR trên `sealed_test` lên `0.785` (`2.25×` v3), đồng thời đạt **MaxDD thấp nhất toàn đấu trường (`-19.8%`)** và dẫn đầu **#1 OOD** trên cả **VN100-10d** (`Sharpe 1.567`, `Calmar 3.96`) lẫn **US Liquid T+1** (`Sharpe 0.294` khi thị trường giảm `-34.5%`).
5. **Tư duy Chuyên gia 10 Năm & Huấn luyện 100% Miễn phí (`<4.5 GB VRAM`)**: Đạt `100%` độ chính xác quyết định, `100%` nhất quán nhân quả, `0.8889` F1 rủi ro ẩn và `Brier = 0.01064` trên 32 bài toán thực chiến; hỗ trợ train trọn bộ trên **Colab Free T4 / Kaggle Free 2×T4 / TPU v5e-8** có tự nối tiếp `--resume` và hỏi đáp bằng tiếng Việt qua `scripts/ask_veteran_agent.py`. Xem chi tiết tại [`FREE_CLOUD_TRAINING_GUIDE.md`](FREE_CLOUD_TRAINING_GUIDE.md) và [`FINAL_UPGRADE_REPORT.md`](FINAL_UPGRADE_REPORT.md).

---

### Citation & License

```bibtex
@misc{tran2026maveqv4,
  title        = {MAVE-Q v4 (VetQuant-RSI): Veteran-Grounded Neuro-Symbolic Quantitative Research with Zero-Validation-Burn Evolution under Emerging-Market Microstructure Constraints},
  author       = {Tran Duy Khang},
  year         = {2026},
  howpublished = {\url{https://github.com/KenyaWashed/VN_Trading_Agents}}
}
```

MIT License — Raw third-party market data is never included or redistributed (see [`docs/data_terms.md`](docs/data_terms.md)).
