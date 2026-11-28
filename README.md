<div align="center">

# MAVE-Q v5 (`DCA-Chronos-RSI`): Popperian Falsification, Google DeepCrossAttention QuantFormer & Anti-Sycophancy Veteran Research

[![License: MIT](https://img.shields.io/badge/License-MIT-00C853?style=for-the-badge&logo=open-source-initiative&logoColor=white)](LICENSE)
[![Python 3.11–3.13](https://img.shields.io/badge/Python-3.11--3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Local Safe](https://img.shields.io/badge/Local%20CPU-284MB%20RAM%20Safe-0EA5E9?style=for-the-badge&logo=amd&logoColor=white)](MODULAR_RSI_ARCHITECTURE.md)
[![Free Cloud](https://img.shields.io/badge/Free%20Cloud-%3C1B%2FBlock%20T4%2FTPU-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](FREE_CLOUD_AND_SCALING_PROPOSALS.md)
[![Pytest](https://img.shields.io/badge/Tests-55%2F55%20Passing-2EA44F?style=for-the-badge&logo=pytest&logoColor=white)](tests/)
[![Ruff](https://img.shields.io/badge/Ruff-Clean-7C3AED?style=for-the-badge&logo=ruff&logoColor=white)](pyproject.toml)
[![Mypy](https://img.shields.io/badge/Mypy-Typed-0284C7?style=for-the-badge&logo=python&logoColor=white)](pyproject.toml)
[![Research Topic](https://img.shields.io/badge/Research%20Topic-571%2F571%20Cells%20Verified-F59E0B?style=for-the-badge&logo=latex&logoColor=white)](paper/main.tex)
[![Research Only](https://img.shields.io/badge/Scope-Research%20Only%20%7C%20No%20Live%20Trading-DC2626?style=for-the-badge&logo=shield&logoColor=white)](#disclaimer)

**Research Topic / Đề tài nghiên cứu**: *MAVE-Q v5 (DCA-Chronos-RSI): Popperian Falsification, Google DeepCrossAttention QuantFormer, and Asymmetric Beta No-Trade-Zone Control under Emerging-Market Microstructure Constraints* *(Active Research Topic — Not a Submitted/Published Work)*  
**Author**: **Tran Duy Khang** — *AI Software Engineer / Independent Quantitative AI Researcher* ([@KenyaWashed](https://github.com/KenyaWashed))  
**Core Guides**: [**Architecture Evolution & Math Proofs (`v0→v5`)**](ARCHITECTURE_EVOLUTION_AND_PROOFS.md) | [**`<1B/Block` Free-Cloud Scaling Proposals**](FREE_CLOUD_AND_SCALING_PROPOSALS.md) | [Free Cloud Training Guide](FREE_CLOUD_TRAINING_GUIDE.md) | [Upgrade Report (`v4→v5`)](FINAL_UPGRADE_REPORT.md) | [4-Reviewer Self-Audit](paper/PEER_REVIEW_AUDIT.md)  
**Language**: [English](#english-version) | [Tiếng Việt](#phiên-bản-tiếng-việt)

</div>

---

## English Version

### Disclaimer
Strictly for **academic research and offline simulation** — NOT a trading bot, zero brokerage connectivity, and zero investment advice. Raw market data (`data/`) is gitignored and never redistributed. Default CI benchmarks reproduce on deterministic synthetic microstructure snapshots (`is_synthetic=true`); live `vnstock` and `yfinance` modes are wired for empirical replication.

### What's New in MAVE-Q v5 (`DCA-Chronos-RSI` + Popperian Falsification)
1. **Popperian Falsification Constitution & Ruthless 8-Seed Pruning**: Every hypothesis is tested against negative controls (randomized alphas, scrambled timestamps, future-perturbation shift invariance `atol=1e-9`) and logged transparently in [`logs/v5_falsified_hypotheses.jsonl`](logs/v5_falsified_hypotheses.jsonl) (**17 falsified/pruned hypotheses**). Removing three redundant v5 blocks (`ChronosKalmanHedge`, `DCAMoSEHyperGate`, stock-level `SplitConformalDormantVeto`) lifted 8-seed Validation Sharpe from `1.405 -> 1.473`, improved **all 3 OOD markets**, and unlocked a **`+0.374`** Single-Shot Sealed Test Sharpe leap over v4 (`1.441` vs `1.067`).
2. **Google DeepCrossAttention `DCA-QuantFormer` & 5-Family Backbone Zoo**: Adapts **Google Research's DeepCrossAttention (`G-DCA`, ICML 2025 `arXiv:2502.06785`)** with a raw-target orthogonal Ridge anchor across 16 causal microstructure channels (`0.48M` Occam-optimal), outperforming Pre-LN Transformer, Mamba-SSM, Hyena-Conv, and Sparse MoE on small-N VN30 (`IC = 0.1152`, `p_IC = 0.0000` vs Pre-LN).
3. **Asymmetric Beta No-Trade-Zone Controller (`AsymBeta-NTZ`)**: Unifies bull/bear asymmetric beta-budget release, exchange-band epistemic abstention (`band_tail_scale` adapting HOSE `±7%` vs wide US/CSI bands), and L1 no-trade-zone hysteresis (`7.76%` Test turnover; ablation crash `-0.4469` Sharpe, `p = 0.0000`).
4. **Anti-Sycophancy 10-Year Veteran CIO Engine (45 Scenarios / 22 Adversarial Traps)**: Refutes 6 quantitative user traps (FOMO margin, unannounced rumor/PIT, look-ahead formula leakage, 1.5x VN fee mirage, falling knife, overfit complexity) with **`100.0%` Sycophancy Resistance**, **`0.0%` False Discovery Rate**, and **`100.0%` Abstention Precision** (`0.8163` Risk F1; deterministic neuro-symbolic guardrail scope).
5. **Hardware-Safe Local (`284 MB` RAM) + `<1B/Block` Free-Cloud Multi-SLM Stack**: `HardwareSafetyGuard` auto-locks `<16 GB` RAM laptops to `local_safe_20m` (`<1.2 GB` cap), while free Kaggle/Colab/Modal blueprints scale to `sub1b_flagship_900m` (every block strictly `<1B`, `0 MB` meta-verified) across **6 SLM families** (`DCA-MicroLM`, `Qwen3`, `Gemma-3`, `SmolLM2`, `Llama-3.2`, `Phi-4-mini`).
6. **Immutable SHA-256 `EvaluatorSeal` & Dual DSR Transparency**: `EvaluatorSeal` (`4a0f5ad3...31e`) remains 100% unmodified; `sealed_test` was run **ONCE** for frozen v5 (ledger entry `#29`), reporting both **Individual DSR (`0.895`, `N=1`)** and **Strict 29-Entry Lineage DSR (`0.389`)**.

### Repository Layout
```text
VN_Trading_Agents/
├── configs/                          # Reproducible YAML configs (market rules, sandbox, v1–v5, cloud presets)
├── src/vn_rsi/                       # data, dsl, sandbox (SHA-256 sealed), eval (sealed), benchmarks, models, veteran, agents, rsi
├── experiments/                      # v5 Popperian dev arena, 8-seed pruning & OOD, single-shot sealed arena, report & figure builders
├── scripts/                          # ci_check.py (full gate), ask_veteran_agent.py (VN/EN CLI), export_cloud_training_pack.py
├── notebooks/                        # One-click Free Colab (T4) / Kaggle (2x T4 / TPU v5e-8) notebook
├── train_colab_kaggle.py             # Universal Free-Cloud Trainer (6 SLM families, HardwareSafetyGuard, --resume)
├── modal_train.py                    # Serverless Modal A100-80GB / H100 trainer ($30/mo free credits)
├── ARCHITECTURE_EVOLUTION_AND_PROOFS.md # Full v0->v5 architectural evolution, Theorems 1–7 & Vietnamese proofs
├── FREE_CLOUD_AND_SCALING_PROPOSALS.md  # <1B/block Kaggle/Colab/Modal blueprints & 6-family Multi-SLM guide
├── paper/                            # Research Topic Technical Report (main.tex -> tectonic PDF, 48 refs, 13 tables, 7 PDF+SVG figs)
└── tests/                            # 55 automated tests (causality, atol=1e-9 shift-invariance, SHA-256 seal, anti-sycophancy)
```

### Quickstart
```bash
# 1. Install & run full verification gate (ruff + mypy + 55 tests + 571-cell report & tectonic PDF audit)
uv sync --extra dev          # or: pip install -e ".[dev]"
python scripts/ci_check.py

# 2. Hardware-safe local smoke test or Free Cloud 6-Family SLM QLoRA/DPO training
python train_colab_kaggle.py --preset auto --mode maveq_v5 --slm-family dca_microlm --enable-veteran-dpo --smoke-test
python scripts/export_cloud_training_pack.py

# 3. Query the Anti-Sycophancy 10-Year Veteran CIO Agent (Vietnamese or English)
python scripts/ask_veteran_agent.py --symbol FPT --lang vi --question "Nghe tin đồn FPT sắp ký hợp đồng lớn, có nên full margin đuổi trần không?"
python scripts/ask_veteran_agent.py --symbol HPG --lang en --json

# 4. Reproduce 8-seed ablations, 3-market OOD, 45-scenario veteran benchmark, 13 tables & 7 vector figures
python experiments/run_veteran_reasoning_eval.py
python experiments/run_v5_ablation_and_ood.py --workers 4
python experiments/build_report.py && python experiments/build_figures.py && python paper/verify_paper.py
```

### Verified Benchmark Highlights (`is_synthetic=true`, Net of VN T+2.5 & Friction)

#### 1. 8-Seed Pruning Ablations & Zero-Retuning 3-Market OOD (`v5_ablations_ood.tex`)
| Benchmark / Variant | Trials | Rank IC | Net Sharpe | Calmar | MaxDD | Turnover | DSR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Pre-Pruning MAVE-Q v5 (`8-seed`) | 1.0 | 0.1138 | 1.405 ± 0.106 | 2.87 | -20.5% | 0.1095 | 0.959 |
| **Pruned Lean MAVE-Q v5 (`0.48M`, Official)** | **1.0** | **0.1152** | **1.473 ± 0.083** | **3.21** | **-19.7%** | **0.1088** | **0.966** |
| w/o `DCA-QuantFormer` (`Pre-LN`) | 1.0 | 0.1098 (`p=0.000`) | 1.431 ± 0.099 (`p=0.112`) | 3.01 | -20.2% | 0.1069 | 0.962 |
| w/o `AsymBeta-NTZ` (`Static Linear`) | 1.0 | 0.1152 | 1.026 ± 0.032 (`p=0.000`) | 1.45 | -27.1% | 0.0972 | 0.896 |
| **OOD VN100 (10d, T+2.5) — MAVE-Q v5** | **1** | **0.1217 (#1)** | **1.564** | **4.24 (#1)** | **-15.6% (#1)** | 0.1080 | **0.974** |
| **OOD US Liquid (5d, T+1) — MAVE-Q v5** | **1** | **0.1449 (#1)** | **0.226** | **0.29** | **-23.6%** | 0.0725 | 0.597 |
| **OOD CSI Emerging (5d, T+1) — MAVE-Q v5** | **1** | 0.1069 | **+0.211** (`+0.328` vs v4) | **+0.24** | **-22.2%** | 0.0681 | 0.584 |

#### 2. Single-Shot Immutable Sealed Test Arena (`v5_global_arena.tex`, SHA-256 `4a0f5ad3...31e`, Ledger #29)
| Model / System | Trials | Test IC | Test Sharpe | Calmar | Ann. Ret | MaxDD | Test DSR (Indiv / Lineage) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| VN-Index Buy & Hold | 1 | 0.0000 | 1.248 | 1.79 | +86.0% | -39.4% | 0.937 / -- |
| `TradingAgents` (*Xiao et al.*) | 2 | 0.0014 | 0.159 | 0.11 | +4.7% | -41.1% | 0.377 / -- |
| `RD-Agent(Q)` (*Li et al., NeurIPS*) | 2 | 0.0988 | 0.903 | 1.04 | +30.5% | -29.2% | 0.775 / -- |
| `AlphaForge-DE` (*Shi et al., AAAI*) | 2 | 0.0961 | 0.971 | 0.97 | +27.9% | -28.9% | 0.749 / -- |
| `MAVE-Q v4` (`VetQuant-RSI`) | 2 | 0.0654 | 1.067 | 1.06 | +24.4% | -23.1% | 0.819 / -- |
| `FinMem` (*Yu et al.*) | 2 | 0.0199 | 1.264 | 1.44 | +33.1% | -23.0% | 0.849 / -- |
| `FinAgent` (*Zhang et al., KDD*) | 2 | 0.1064 | 1.514 | 1.79 | +43.9% | -24.6% | 0.911 / -- |
| `Holo-MAVE-Q v3` (81 Trials) | 81 | 0.0739 | 1.676 | 2.21 | +48.8% | -22.0% | 0.388 / -- |
| **`MAVE-Q v5` (Pruned Lean `0.48M`, Ours)** | **1** | **0.0850** (`+30%` v4) | **1.441** (`+0.374` v4, `p=0.081`) | **1.70** (`+61%`) | **+35.6%** (`p=0.068`) | **-20.9%** | **0.895 / 0.389** (29-entry) |

#### 3. 45-Scenario / 22-Trap Anti-Sycophancy Veteran Benchmark (`v5_veteran_antisycophancy.tex`)
| Reasoning System | Decision Acc | Sycophancy Resist. | FDR ($\downarrow$) | Abstention Prec. | Hidden-Risk F1 | Brier ($\downarrow$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Sycophantic Yes-Man LLM | 46.67% | 0.00% | 73.81% | 0.00% | 0.0000 | 0.4910 |
| Naive FOMO Agent | 48.89% | 0.00% | 70.00% | 0.00% | 0.0000 | 0.4590 |
| Static Momentum Analyst | 57.78% | 0.00% | 58.06% | 0.00% | 0.0000 | 0.3820 |
| Generic Rule Analyst | 57.78% | 0.00% | 56.00% | 0.00% | 0.2353 | 0.3760 |
| **MAVE-Q v5 Anti-Sycophancy Veteran CIO** | **100.00%** | **100.00%** (22/22) | **0.00%** | **100.00%** | **0.8163** (`R=1.0, P=0.69`) | **0.0162** |

#### 4. Longitudinal Evolution (`v0 → v5`) & 5-Family Backbone Zoo Summary
| Generation / Backbone | Active Core | Val Queries | Val Sharpe | Sealed Test Sharpe | Sealed MaxDD | Sealed DSR (Indiv / Lineage) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Agent `v0` → `v1` → `v2` | Monolithic 8B → AST Diffusion → QD | `1 → 40+ → 299` | `0.677 → 1.286` | `1.339 → 1.602` | `-25.1% → -24.1%` | `0.365 → 0.180` |
| Holo-MAVE-Q `v3` | `Tree-MDLM` + `Holo-HRM` + `MSSF` + `BTO-SRSS` | `81` | `1.291` | `1.676` | `-22.0%` | `0.388` (81t) / -- |
| MAVE-Q `v4` (`VetQuant-RSI`) | `VeteranCaseMemory` + `GP-FA-Ridge` + `BAB` | `1` (2 total) | `1.347 ± 0.081` | `1.067` | `-23.1%` | `0.819` / -- |
| **MAVE-Q `v5` (Pruned Lean `0.48M`)** | **Google `DCA-QuantFormer` + `AsymBeta-NTZ` (17 falsified)** | **`1` (`N=1`)** | **`1.473 ± 0.083`** | **`1.441` (`+0.374` v4)** | **`-20.9%`** | **`0.895 / 0.389`** (29-entry) |

- **5-Family Backbone Zoo (`v5_backbone_zoo_and_scaling.tex`)**: Google `DCA-QuantFormer` (`0.48M`, #1) outperformed `Pre-LN Transformer`, `Mamba-SSM`, `Hyena-Conv`, and `Sparse MoE` on small-N VN30 (`p_IC = 0.0000` vs Pre-LN), while `0.48M` beat `5M` and `20M` by Occam BIC (`42×` params gave only `+0.011` SR with worse Calmar/MaxDD).
- **Popperian Honesty ("What Failed & Why")**: All **17 falsified/pruned hypotheses** — including 3 pruned v5 modules (`ChronosKalmanHedge`, `DCAMoSEHyperGate`, stock-level `SplitConformalDormantVeto`), 6 adversarial controls, and 3 v4 post-mortem flaws — are logged in [`logs/v5_falsified_hypotheses.jsonl`](logs/v5_falsified_hypotheses.jsonl) and analyzed in [`ARCHITECTURE_EVOLUTION_AND_PROOFS.md`](ARCHITECTURE_EVOLUTION_AND_PROOFS.md).

#### 5. Local Hardware Safety (`284 MB` RAM) & Free-Cloud `<1B/Block` Multi-SLM Scaling
| Target Tier | Platform / Hardware | Active Preset | Max Block Size | Memory Profile |
| :--- | :--- | :--- | :---: | :--- |
| **Local Laptop (CPU)** | Any `<16 GB` RAM laptop (`0 GPU`) | `local_safe_20m` (auto-locked) | `0.48M`–`20M` | **Measured `284 MB` RAM** (`<1.2 GB` cap) |
| **Colab Free (`1x T4`)** | `16 GB` VRAM (`$0`) | `kaggle_t4_optimal` | `12M` core / `125M` MicroLM | Analytical `<4.5 GB` VRAM (FP16 + 4-bit QLoRA) |
| **Kaggle Free (`T4 x2` / `TPU v5e-8`)** | `32 GB` VRAM / `128 GB` HBM (`$0`) | `sub1b_flagship_900m` | **`<1B/block`** (`880M` QuantFormer, `828M` MicroLM) | Analytical BF16 (`0 MB` meta-verified; 6 SLM families) |
| **Modal Free (`$30/mo`)** | `A100-80GB` / `H100` | `sub1b_flagship_900m` | **`<1B/block`** + `Qwen3-4B / Gemma-3-4B` | Analytical BF16 (`FREE_CLOUD_AND_SCALING_PROPOSALS.md`) |

**6 Open-Source SLM Families Supported**: `DCA-MicroLM` (from-scratch Google `G-DCA`), `Qwen3`, `Gemma-3`, `SmolLM2`, `Llama-3.2`, and `Phi-4-mini`.

---

## Phiên Bản Tiếng Việt

### Miễn trừ trách nhiệm
Kho mã nguồn này là một **Đề tài nghiên cứu (Research Topic)** phục vụ **nghiên cứu khoa học và mô phỏng ngoại tuyến** — không phải bot giao dịch, không kết nối tài khoản chứng khoán và không tư vấn đầu tư. Dữ liệu thô (`data/`) được gitignore hoàn toàn.

### Điểm Đột Phá của Đề Tài Nghiên Cứu MAVE-Q v5 (`DCA-Chronos-RSI` + Popperian Falsification)
1. **Hiến pháp Phản nghiệm Popperian & Cắt tỉa không khoan nhượng (17 giả thuyết bị bác bỏ)**: Toàn bộ 17 giả thuyết thất bại/dư thừa được ghi nhận công khai tại [`logs/v5_falsified_hypotheses.jsonl`](logs/v5_falsified_hypotheses.jsonl). Khi mạnh tay cắt bỏ 3 mô-đun làm giảm Sharpe trên 8 seeds (`ChronosKalmanHedge`, `DCAMoSEHyperGate`, `SplitConformalDormantVeto` cấp cổ phiếu), mô hình tinh gọn **`0.48M`** tăng Validation Sharpe từ `1.405 -> 1.473`, cải thiện **cả 3 thị trường OOD**, và bật tăng **`+0.374`** Sharpe trên `sealed_test` bắn 1 phát duy nhất (`1.441` so với v4 `1.067`, `p=0.081`).
2. **Xương sống `DCA-QuantFormer` (Google DeepCrossAttention, ICML 2025) & Bộ điều khiển `AsymBeta-NTZ`**: Vượt 4 kiến trúc đối chứng (Pre-LN, Mamba-SSM, Hyena-Conv, MoE) về Rank IC (`p=0.0000`), giữ MaxDD `-20.9%` (VN-Index `-39.4%`), nâng Test Calmar `+60.6%` (`1.70`) và đảo chiều thị trường OOD CSI Trung Quốc từ `-0.117` lên **`+0.211`**.
3. **Chuyên gia Phản biện Chống Nịnh hót (`Anti-Sycophancy Veteran CIO`, 45 kịch bản / 22 bẫy)**: Đạt **`100%` Kháng nịnh hót**, **`0%` Tỷ lệ phát hiện giả (FDR)** và **`100%` Độ chính xác đứng ngoài** trước các bẫy full margin, tin đồn chưa công bố PIT, công thức rò rỉ tương lai và ảo giác phí giao dịch T+2.5 (phạm vi tường lửa neuro-symbolic tất định; F1 rủi ro ẩn `0.8163`, Recall `1.000`, Precision `0.6897`).
4. **An toàn phần cứng Laptop (`284 MB` RAM) & Hệ sinh thái Cloud Miễn phí `<1B/khối` (6 họ SLM)**: Tự động khóa `local_safe_20m` (`<1.2 GB` RAM) trên laptop không GPU, đồng thời cung cấp bản thiết kế `<1B/khối` (`sub1b_flagship_900m`) cho Kaggle `T4 x2` / `TPU v5e-8` / Colab / Modal.
5. **Tài liệu Chuyên sâu & Chứng minh Toán học**:
   - [**Tiến hóa kiến trúc `v0→v5` & Định lý 1–7 (`ARCHITECTURE_EVOLUTION_AND_PROOFS.md`)**](ARCHITECTURE_EVOLUTION_AND_PROOFS.md)
   - [**Đề xuất Cloud Miễn phí `<1B/khối` & 6 họ SLM (`FREE_CLOUD_AND_SCALING_PROPOSALS.md`)**](FREE_CLOUD_AND_SCALING_PROPOSALS.md)
   - [**Báo cáo nâng cấp `v4→v5` (`FINAL_UPGRADE_REPORT.md`)**](FINAL_UPGRADE_REPORT.md) | [**Biên bản Tự phản biện 4 Reviewer (`paper/PEER_REVIEW_AUDIT.md`)**](paper/PEER_REVIEW_AUDIT.md)

### Lệnh Chạy Nhanh (Tiếng Việt)
```bash
# Kiểm thử toàn diện (ruff + mypy + 55 pytest + xác thực 571 ô số liệu & biên dịch PDF Tectonic)
python scripts/ci_check.py

# Hỏi Chuyên gia Phản biện Chống Nịnh hót bằng Tiếng Việt (chạy tức thì trên CPU)
python scripts/ask_veteran_agent.py --symbol FPT --lang vi --question "Đánh giá FPT và bẫy kẹt sàn T+2.5?"

# Đóng gói dữ liệu Search-Zone sạch để đẩy lên Kaggle / Colab Free T4 & TPU v5e-8
python scripts/export_cloud_training_pack.py
```

---

### Citation & License

```bibtex
@misc{tran2026maveqv5,
  title        = {MAVE-Q v5 (DCA-Chronos-RSI): Popperian Falsification, Google DeepCrossAttention QuantFormer, and Asymmetric Beta No-Trade-Zone Control under Emerging-Market Microstructure Constraints},
  author       = {Tran Duy Khang},
  year         = {2026},
  note         = {Independent Research Topic Technical Report},
  howpublished = {\url{https://github.com/KenyaWashed/VN_Trading_Agents}}
}
```

MIT License — Raw third-party market data is never included or redistributed (see [`docs/data_terms.md`](docs/data_terms.md)).
