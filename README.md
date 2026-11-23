<div align="center">

# MAVE-Q: Modular Autonomous Verified Evolution for Small-Model Quantitative Research

[![License: MIT](https://img.shields.io/badge/License-MIT-00C853?style=for-the-badge&logo=open-source-initiative&logoColor=white)](LICENSE)
[![Python 3.11–3.13](https://img.shields.io/badge/Python-3.11--3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Modular%20RSI-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](src/vn_rsi/models/)
[![Pytest](https://img.shields.io/badge/Tests-21%2F21%20Passing-2EA44F?style=for-the-badge&logo=pytest&logoColor=white)](tests/)
[![Ruff](https://img.shields.io/badge/Ruff-Clean-7C3AED?style=for-the-badge&logo=ruff&logoColor=white)](pyproject.toml)
[![Mypy](https://img.shields.io/badge/Mypy-Typed-0284C7?style=for-the-badge&logo=python&logoColor=white)](pyproject.toml)
[![Docker](https://img.shields.io/badge/Sandbox-Sealed%20Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](Dockerfile)
[![LaTeX Paper](https://img.shields.io/badge/Paper-LaTeX%20Audited-F59E0B?style=for-the-badge&logo=latex&logoColor=white)](paper/main.tex)
[![Research Only](https://img.shields.io/badge/Scope-Research%20Only%20%7C%20No%20Live%20Trading-DC2626?style=for-the-badge&logo=shield&logoColor=white)](#disclaimer)

**Paper**: *MAVE-Q: Modular Autonomous Verified Evolution for Small-Model (≤8B) Quantitative Research in Rule-Constrained Emerging Markets*  
**Author**: **Tran Duy Khang** — *AI Software Engineer / Freelance Vibe Researcher* ([@KenyaWashed](https://github.com/KenyaWashed))  
**Language**: [English](#english-version) | [Tiếng Việt](#phiên-bản-tiếng-việt)

</div>

---

## English Version

### Disclaimer

This repository is strictly for **academic research and simulation**. It is **not** a trading bot, has no brokerage or order-execution connections, and offers no financial advice. All experiments run offline on versioned local Parquet snapshots (`data/` is gitignored; raw third-party market data is never redistributed). Default benchmark tables are produced on a deterministic synthetic Vietnamese market fixture (`is_synthetic=true`) so anyone can reproduce and verify the full pipeline before running on their own empirical data snapshot.

---

### What This Repository Does

**MAVE-Q** is a self-contained research codebase for studying **recursively self-improving (RSI) multi-agent systems** in quantitative finance without relying on frontier-scale closed models or fragile, leaky backtests. Concretely, this repo gives you:

1. **A leak-free Vietnamese equity simulator** covering HOSE, HNX, and UPCoM rules (T+2.5 settlement locks, ±7%/10%/15% ceiling/floor unfillable bars, 100-share board lots, 0.15% brokerage fee, 0.10% statutory sell tax, illiquidity slippage, delisted/halted tickers, and point-in-time fundamental announcement lags).
2. **A sealed, tamper-checked evaluation sandbox** that locks the splits, cost model, and evaluator behind a SHA-256 hash and allows each frozen configuration to touch the sealed test window **only once**.
3. **Two complete small-model RSI implementations**: a standard ≤8B multi-agent loop and a decomposed **Modular Micro-RSI** core (`Qwen3-4B/1.7B` + `6.36M Discrete AST Diffusion` + `3.38M HRM Reasoner`) that anyone can train independently on whatever hardware they have.
4. **An automated paper-verification pipeline** that builds LaTeX tables directly from JSONL logs and checks that every citation and every table cell matches a recorded run.

---

### Core Idea & Research Inspirations

The central question behind this work is simple: **Can a small multi-agent system (around a few billion parameters — or a modular core with under 10M structural parameters) improve its own research strategy from validated evidence, while keeping evaluation genuinely leak-free in a thin, rule-constrained emerging market?**

MAVE-Q pulls together ideas from several recent papers and adapts them to work cleanly with compact models:

- **AQuA** (`arXiv:2608.12841`): Inspires the sealed sandbox design, fixed causal operator registry, persistent belief memory, and single-shot test discipline.
- **AIDE²** (`arXiv:2609.26457`): Inspires the bi-level setup where an inner agent searches for alpha factors under a fixed budget, while an outer loop updates the agent itself based on held-out validation scores.
- **Darwin Gödel Machine** (`arXiv:2505.22954`) & **ShinkaEvolve** (`arXiv:2509.19349`): Inspire keeping an open-ended archive of diverse agent variants (balancing fitness and novelty) and filtering duplicate structures early before running costly backtests.
- **HRM** (`arXiv:2506.21734`) & **TRM** (`arXiv:2510.04871`): Inspire replacing a heavy monolithic planner with a tiny recursive network (`3.38M` params) that separates slow regime-level planning (`z_H`) from fast candidate scoring (`z_L`).
- **Discrete Masked Diffusion (MDLM / LLaDA)** & **Qwen3** (`arXiv:2505.09388`): Inspire handling factor synthesis with a `6.36M` discrete masked diffusion model over a closed 70-token causal grammar, while leaving natural-language hypothesis framing to a compact `Qwen3` model.
- **Bailey & López de Prado (2014, 2018)**: Provide the statistical foundation — purged and embargoed walk-forward splits plus the Deflated Sharpe Ratio to catch overfitting across many agent trials.

---

### What Makes MAVE-Q Different (Breakthrough Ideas)

1. **Breaking the monolith into three tiny specialists instead of scaling one giant LLM**:  
   Most quantitative agent papers ask one large autoregressive LLM to do everything — read market context, invent hypotheses, write Python code, debug crashes, and reflect on errors. MAVE-Q splits those jobs by what each model is actually good at: a compact language model handles economic reasoning, a `6.36M` Discrete Masked AST Diffusion model synthesizes and mutates causal factor trees, and a `3.38M` Hierarchical Recursive Model (`HRM`) handles regime allocation and pre-backtest filtering. Because the structural core is under `10M` parameters, the system can perform **both outer-loop policy evolution and real weight-level self-training (ReST)** cleanly and quickly.
2. **Grammar-constrained Subtree Infilling instead of blind regeneration**:  
   Autoregressive models write formulas token-by-token from left to right; one wrong operator breaks the tree or sneaks in a look-ahead shift. Our AST diffusion model operates over a closed vocabulary of 70 strictly causal tokens. To mutate a promising factor, it masks only a weak subtree and denoises that branch in place under grammar constraints — guaranteeing 100% valid, leak-free expressions by construction.
3. **Bounded mutable surface to stop small-model self-destruction and reward hacking**:  
   Letting a small model freely rewrite its own Python source code usually breaks the repo or leads to metric tampering. MAVE-Q defines an explicit mutable genome (focus weights, prompts, lookback windows, AST depth, `best_of_n`, memory decay) and physically blocks edits to the evaluator, splits, or cost simulator.
4. **Realistic emerging-market friction as a first-class scientific testbed**:  
   Strategies that look great in frictionless backtests often collapse once Vietnam's T+2.5 holding lock, unfillable ceiling buys / floor sells, 0.1% sell tax, delisted stocks, and earnings announcement lags are enforced. Building these rules into the sealed evaluator exposes failure modes that standard US/crypto benchmarks hide.

---

### Architecture (Plain Text Overview)

MAVE-Q runs as a two-level loop wrapped around a locked evaluation sandbox:

1. **Data & Market Layer (`src/vn_rsi/data/`)**: Builds an immutable Parquet snapshot (either from live rate-limited `vnstock` feeds or the deterministic synthetic VN fixture), checks SHA-256 checksums, verifies announcement-date causality on fundamentals, and runs future-perturbation shift tests to confirm zero look-ahead bias.
2. **Sealed Sandbox & Evaluator (`src/vn_rsi/sandbox/`, `src/vn_rsi/eval/`)**: Partitions data strictly into three chronological zones — `search` (open for agent inspection), `validation` (returns scalar scores only), and `sealed_test` (guarded by an append-only ledger that refuses any second look at the same config hash). Every candidate is simulated under Vietnamese cash-equity rules (T+2.5, price bands, board lots, taxes, and illiquidity impact).
3. **Inner Research Loop (`src/vn_rsi/agents/`, `dsl/`, `memory/`, `models/`)**: Seven collaborating roles run under hard token/time/compute budgets:
   - `Planner` & `DataSteward` inspect search-zone regime stats and allocate exploration across four economic families (liquidity momentum, post-T+2.5 reversal, point-in-time value/quality, and foreign flows).
   - `IdeaMiner` (Model 1: compact SLM) writes a falsifiable hypothesis with an expected sign and refutation criteria *before* building any factor.
   - `Implementer` (Model 2: `DiscreteASTDiffusion`, `6.36M` + Model 3: `HRMReasoner` fast head `z_L`) generates or subtree-mutates candidate causal ASTs and pre-ranks them before backtesting.
   - `LeakageAuditor` verifies grammar rules, runs a dynamic shift-test, and rejects structural duplicates via AST fingerprinting.
   - `EvaluatorProxy` checks search-zone refutation rules and scores surviving ensembles on the scalar validation view.
   - `Librarian` updates `EpistemicBeliefMemory` (with confidence decay and contradiction penalties) and exports verified search-zone trajectories for ReST / QLoRA self-training.
4. **Outer Self-Improvement Loop (`src/vn_rsi/rsi/`)**: Samples a parent genome from the open-ended `DGMArchive`, proposes bounded edits inside the mutable surface (via `HRM` slow head `z_H` or an ≤8B meta-proposer), evaluates the child across multiple paired seeds, and accepts the rewrite only if it clears the statistical acceptance gate.

---

### Evaluation Metrics & What They Are Used For

Every method in this repo is judged on a consistent set of metrics designed to separate genuine signal from lucky backtest noise:

| Metric | What It Measures | Why We Use It Here |
| :--- | :--- | :--- |
| **Rank IC** (Spearman Information Coefficient) | Daily cross-sectional rank correlation between the predicted signal at close `t` and realized 5-day forward returns. | Measures raw predictive power across stocks without getting distorted by extreme single-stock outliers. |
| **ICIR** (IC Information Ratio) | Annualized mean Rank IC divided by the standard deviation of daily Rank IC. | Checks consistency — a factor with decent average IC that swings wildly from week to week is unreliable. |
| **Net Sharpe Ratio** | Annualized excess return divided by volatility **after** brokerage fees, 0.1% sell tax, spread, and illiquidity slippage. | Core risk-adjusted profitability measure under realistic Vietnamese trading frictions. |
| **Net Sortino Ratio** | Similar to Net Sharpe, but only penalizes downside volatility. | Captures asymmetric crash risk (especially important when floor-limit locks trap positions during sell-offs). |
| **Max Drawdown (MaxDD)** | Worst peak-to-trough portfolio loss over the evaluation window. | Measures real-world survivability during market stress and delisting/halt events. |
| **Daily Turnover** | Fraction of portfolio value traded per day. | Directly tracks cost sensitivity; in Vietnam, high turnover quickly gets eaten alive by the 0.1% sell tax + spread. |
| **Deflated Sharpe Ratio (DSR)** | Probability that the true Sharpe ratio is > 0 after adjusting for skewness, fat tails, track record length, and **total number of trials tested** (`Bailey & López de Prado, 2014`). | Crucial for RSI agents: because an agent tests dozens or hundreds of ideas, raw Sharpe is inflated by selection bias. DSR keeps us honest. |
| **Paired Block-Bootstrap 95% CI** | Confidence interval on the Sharpe difference between an agent and a baseline using block resampling (preserving time-series autocorrelation). | Verifies whether a performance gain is statistically meaningful rather than random seed luck. |

---

### Repository Structure

```text
VN_Trading_Agents/
├── README.md                                # Project overview (English + Vietnamese)
├── MODULAR_RSI_ARCHITECTURE.md              # Notes on the 3-model split, parameter counts, and training setup
├── REPORT_TONG_HOP_VN_RSI.md                # Detailed Vietnamese summary report
├── PROGRESS.md                              # Phase-by-phase progress and verification checklist
├── pyproject.toml                           # Dependencies (uv/pip), ruff, mypy, and pytest config
├── Makefile                                 # Shortcuts: data, test, lint, baseline, agent, rsi, modular, report
├── Dockerfile                               # Isolated no-network container for running the sealed sandbox
├── train_colab_kaggle.py                    # Standalone training script (handles CPU, single-GPU, and multi-GPU)
│
├── notebooks/
│   └── train_mave_q_kaggle_colab.ipynb      # Ready-to-run Jupyter notebook for training & full evaluation
│
├── configs/                                 # All experiment configs (reproducible via SHA-256 config hashing)
│   ├── market_rules.yaml                    # HOSE/HNX/UPCoM price limits, T+2.5, lot size, tax, slippage, FOL
│   ├── data.yaml                            # Snapshot ID, universe tier (vn30/vn100/all_liquid), QA toggles
│   ├── sandbox.yaml                         # Search / validation / sealed_test dates, purge & embargo, canary
│   ├── llm.yaml                             # Model backend settings and hard token/time/GPU-hour budgets
│   ├── agent_v0.yaml                        # Starting genome and search-zone acceptance thresholds
│   ├── rsi.yaml                             # Mutable surface whitelist, archive weights, ablations, OOD config
│   └── kaggle_modular_rsi.yaml              # Presets for Modular Micro-RSI (cpu_fast / kaggle_t4_optimal)
│
├── src/vn_rsi/                              # Core Python package
│   ├── data/                                # vnstock & synthetic providers, Parquet snapshot manager, leakage QA
│   ├── dsl/                                 # Closed causal operator registry (70-token grammar) and AST verifier
│   ├── sandbox/                             # SHA-256 evaluator seal, zone guards, VN simulator, sealed ledger
│   ├── eval/                                # Metrics (Rank IC, DSR, bootstrap CIs), 7 baselines, sealed evaluator
│   ├── llm/                                 # Unified LLM backend (local Ollama/vLLM/llama.cpp + CPU scaffold)
│   ├── memory/                              # Persistent epistemic belief store with decay & contradiction checks
│   ├── agents/                              # 7-role hypothesis-to-factor pipeline
│   ├── models/                              # ASTVocab, DiscreteASTDiffusion (6.36M), HRMReasoner (3.38M), ModularBackend
│   ├── rsi/                                 # Bi-level outer loop, mutable surface guard, DGM archive, ignition test
│   └── utils/                               # Seeding, hashing, hard budget enforcement, structured JSONL logging
│
├── experiments/                             # Entry scripts for every stage of the research workflow
│   ├── run_data.py                          # Build snapshot and run automated leakage & QA checks
│   ├── run_baselines.py                     # Run canary checks and all 7 reference baselines
│   ├── run_agent_v0.py                      # Run the base multi-agent pipeline (no outer self-improvement)
│   ├── run_rsi.py                           # Run bi-level RSI, 6 ablations, ignition test, and VN100 OOD test
│   ├── run_modular_rsi.py                   # Run Modular Micro-RSI (HRM + Discrete AST Diffusion + ReST)
│   └── build_report.py                      # Generate LaTeX tables directly from JSONL experiment logs
│
├── tests/                                   # Automated test suite (21 tests: leakage, causality, seal, models)
├── paper/                                   # LaTeX manuscript (main.tex), verified BibTeX, tables, verify_paper.py
├── docs/                                    # Literature notes, design trade-offs, data terms, and scale-up guide
└── scripts/
    └── ci_check.py                          # One-shot check: ruff + mypy + pytest + paper audit
```

---

### How to Run Everything

#### 1. Install & Verify Locally

```bash
git clone https://github.com/KenyaWashed/VN_Trading_Agents.git
cd VN_Trading_Agents

# Using uv (fast, reproducible lockfile)
uv sync --extra dev

# Or standard pip
pip install -e ".[dev]"

# Run linter, type checks, all 21 tests, and paper provenance audit
uv run python scripts/ci_check.py
```

#### 2. Run the Full Research Pipeline (End-to-End)

```bash
uv run python experiments/run_data.py                           # 1. Build immutable snapshot + leakage QA
uv run python experiments/run_baselines.py --reset-ledger       # 2. Sealed sandbox canary + 7 baselines
uv run python experiments/run_agent_v0.py                       # 3. Base multi-agent pipeline (v0)
uv run python experiments/run_rsi.py                            # 4. Bi-level RSI + 6 ablations + Ignition + OOD
uv run python experiments/run_modular_rsi.py --preset cpu_fast  # 5. Modular Micro-RSI (HRM + AST Diffusion)
uv run python experiments/build_report.py                       # 6. Auto-generate LaTeX result tables
uv run python paper/verify_paper.py                             # 7. Verify all citations & table numbers vs logs
```

*(If you have `make` installed, `make all` runs the full chain.)*

#### 3. Training & Isolated Sandbox

```bash
# Train via notebook: notebooks/train_mave_q_kaggle_colab.ipynb

# Or run the training script directly (install optional GPU deps if using QLoRA):
pip install -q unsloth trl datasets bitsandbytes accelerate
python train_colab_kaggle.py \
    --preset kaggle_t4_optimal \
    --enable-unsloth-qlora \
    --slm-model unsloth/Qwen3-4B-bnb-4bit \
    --epochs 25 \
    --outer-iters 4

# Run inside an isolated no-network Docker container:
docker build -t mave-q-sandbox .
docker run --rm --network=none --read-only --tmpfs /tmp -v "$(pwd)/logs:/workspace/logs" mave-q-sandbox
```

---

### Benchmark Snapshot (Verified from Logs)

Auto-generated from [`paper/tables/main_results.tex`](paper/tables/main_results.tex) (`is_synthetic=true` Vietnamese microstructure benchmark, 2019–2025, net of T+2.5, fees, 0.1% sell tax, and illiquidity impact):

| Method | Val Rank IC | Val Net Sharpe | Sealed Test Rank IC | Sealed Test Net Sharpe | Test Ann. Net Ret | Test DSR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| VN-Index Buy & Hold | 0.0000 | 0.506 | 0.0000 | 1.248 | 86.0% | 0.937 |
| Equal-Weight Universe | 0.0000 | 0.454 | 0.0000 | 1.195 | 68.3% | 0.534 |
| Momentum (20d) | 0.0514 | 0.462 | 0.0551 | 1.179 | 54.0% | 0.498 |
| Short-Term Reversal (5d) | -0.0087 | -0.408 | -0.0580 | 0.366 | 15.4% | 0.143 |
| Purged Walk-Forward Ridge | 0.1026 | 0.702 | 0.0781 | 0.327 | 8.7% | 0.121 |
| Purged Walk-Forward GBDT | 0.0787 | 0.615 | 0.0427 | 0.532 | 17.5% | 0.168 |
| Purged Temporal GRU | 0.0976 | 0.450 | 0.0720 | 0.396 | 11.5% | 0.120 |
| Agent `v0` (≤8B, No RSI) | 0.1036 | 0.677 | 0.0910 | 1.339 | 40.1% | 0.365 |
| Monolithic Bi-Level RSI `v*` (≤8B) | 0.1025 | 0.877 | 0.0637 | 1.432 | 38.8% | 0.105 |
| **MAVE-Q Modular (`HRM + AST Diff.`)** | **0.1045** | **1.019** | 0.0539 | **1.918** | 31.0% | **0.389** |

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

---
---

## Phiên Bản Tiếng Việt

### Tuyên bố Miễn trừ Trách nhiệm

Kho mã nguồn này chỉ phục vụ **nghiên cứu học thuật và mô phỏng định lượng**. Đây **không phải** bot giao dịch thực tế, không kết nối tài khoản chứng khoán, không có chức năng đặt lệnh và không đưa ra lời khuyên đầu tư. Mọi thực nghiệm đều chạy ngoại tuyến trên các bản chụp dữ liệu Parquet cục bộ có kiểm tra mã băm (`data/` được gitignore; tuyệt đối không phân phối lại dữ liệu thô từ bên thứ ba). Các bảng kết quả mặc định trong repo chạy trên bộ dữ liệu mô phỏng vi cấu trúc thị trường Việt Nam tất định (`is_synthetic=true`) để bất kỳ ai cũng có thể kiểm chứng trọn vẹn quy trình trước khi tự tải bản chụp dữ liệu lịch sử thực tế.

---

### Repo Này Dùng Để Làm Gì?

**MAVE-Q** là một bộ khung nghiên cứu hoàn chỉnh giúp khảo sát **hệ thống đa tác tử tự cải tiến đệ quy (RSI)** trong tài chính định lượng mà không phụ thuộc vào các mô hình đóng khổng lồ hay các bài backtest dễ bị rò rỉ dữ liệu. Cụ thể, repo này cung cấp:

1. **Bộ mô phỏng thị trường chứng khoán Việt Nam sát thực tế** trên cả HOSE, HNX và UPCoM (khóa thanh toán T+2.5, chặn khớp lệnh khi kịch trần/nằm sàn ±7%/10%/15%, lô chẵn 100 cổ phiếu, phí môi giới 0.15%, thuế bán 0.10%, trượt giá theo thanh khoản, cổ phiếu bị đình chỉ/hủy niêm yết và báo cáo tài chính theo đúng ngày công bố Point-in-Time).
2. **Hộp cát đánh giá niêm phong chống gian lận** khóa mã băm SHA-256 toàn bộ bộ chia dữ liệu, mô hình chi phí và hàm chấm điểm, đồng thời chỉ cho phép mỗi cấu hình đóng băng được chấm trên tập `sealed_test` **duy nhất 1 lần**.
3. **Hai cài đặt hoàn chỉnh cho mô hình cỡ nhỏ**: vòng lặp đa tác tử ≤8B truyền thống và lõi **MAVE-Q Modular** chia nhỏ 3 mô hình (`Qwen3-4B/1.7B` + `6.36M Discrete AST Diffusion` + `3.38M HRM Reasoner`) để bạn tự do huấn luyện độc lập trên bất kỳ phần cứng nào mình có.
4. **Quy trình kiểm toán bài báo tự động** sinh bảng LaTeX trực tiếp từ file log JSONL và đối chiếu từng trích dẫn, từng con số trong bảng với log gốc.

---

### Ý Tưởng Chính & Nguồn Cảm Hứng Học Thuật

Câu hỏi trung tâm của dự án là: **Liệu một hệ thống đa tác tử cỡ nhỏ khoảng vài tỷ tham số (hoặc kiến trúc phân rã với lõi cấu trúc dưới 10 triệu tham số) có thể tự cải tiến chiến lược nghiên cứu của chính nó dựa trên bằng chứng thực nghiệm hợp lệ, trong khi vẫn giữ quá trình đánh giá hoàn toàn không rò rỉ dữ liệu ở một thị trường mới nổi kém thanh khoản và nhiều quy định chặt như Việt Nam?**

MAVE-Q tổng hợp và điều chỉnh ý tưởng từ 6 nhóm công trình mới nhất:

- **AQuA** (`arXiv:2608.12841`): Lấy cảm hứng về hộp cát niêm phong, tập toán tử nhân quả cố định, bộ nhớ niềm tin xuyên suốt các vòng lặp và kỷ luật chấm tập test đúng 1 lần.
- **AIDE²** (`arXiv:2609.26457`): Lấy cảm hứng về cấu trúc tối ưu hai cấp — vòng trong tìm kiếm nhân tố dưới ngân sách cố định, vòng ngoài tự điều chỉnh chính tác tử dựa trên điểm đánh giá ẩn.
- **Darwin Gödel Machine** (`arXiv:2505.22954`) & **ShinkaEvolve** (`arXiv:2509.19349`): Duy trì kho lưu trữ mở gồm nhiều phiên bản tác tử đa dạng (cân bằng giữa hiệu năng và độ mới) kèm bộ lọc loại bỏ công thức trùng lặp ngay trước khi chạy backtest.
- **HRM** (`arXiv:2506.21734`) & **TRM** (`arXiv:2510.04871`): Thay thế bộ lập kế hoạch cồng kềnh bằng mạng đệ quy siêu nhỏ (`3.38M` tham số) tách biệt tầng lập kế hoạch chậm theo chế độ thị trường (`z_H`) và tầng lọc ứng viên nhanh (`z_L`).
- **Discrete Masked Diffusion (MDLM / LLaDA)** & **Qwen3** (`arXiv:2505.09388`): Dùng mô hình khuếch tán rời rạc `6.36M` tham số trên đúng 70 token ngữ pháp nhân quả để sinh và đột biến công thức, còn phần diễn giải giả thuyết kinh tế giao cho mô hình ngôn ngữ gọn nhẹ như `Qwen3`.
- **Bailey & López de Prado (2014, 2018)**: Nền tảng thống kê gồm chia tập thời gian có loại chồng lấn (*purge*) & khoảng đệm (*embargo*) kết hợp chỉ số **Deflated Sharpe Ratio** để chống ảo giác do thử nghiệm quá nhiều lần.

---

### Điểm Đột Phá & Khác Biệt Của MAVE-Q

1. **Chia nhỏ mô hình nguyên khối thành 3 chuyên gia siêu nhẹ thay vì chạy đua tăng kích thước LLM**:  
   Hầu hết các nghiên cứu agent định lượng hiện nay bắt một LLM lớn làm hết mọi khâu từ đọc thị trường, nghĩ ý tưởng, viết code Python, sửa lỗi cho tới tự rút kinh nghiệm. MAVE-Q tách đúng việc cho đúng mô hình: SLM gọn nhẹ lo tư duy kinh tế, `DiscreteASTDiffusion` (`6.36M`) chuyên sinh và đột biến cây công thức nhân quả, còn `HRMReasoner` (`3.38M`) chuyên lập kế hoạch và chấm điểm sơ loại. Nhờ lõi cấu trúc dưới `10M` tham số, hệ thống làm được **cả tự cải tiến chiến lược vòng ngoài lẫn tự huấn luyện cập nhật trực tiếp trọng số (ReST)** cực nhanh và ổn định.
2. **Đột biến nhánh con có kiểm soát ngữ pháp (Subtree Infilling) thay vì viết lại mù quáng từ đầu**:  
   Mô hình sinh từ trái sang phải chỉ cần sai một toán tử là hỏng cả công thức hoặc vô tình kéo dữ liệu tương lai vào quá khứ. Mạng AST Diffusion của MAVE-Q hoạt động trên bộ từ vựng đóng 70 token hoàn toàn nhân quả: khi cần cải tiến một nhân tố tốt, nó chỉ che mặt nạ đúng nhánh con yếu rồi khử nhiễu lại theo ràng buộc ngữ pháp — đảm bảo 100% công thức sinh ra đều hợp lệ và không nhìn trước tương lai.
3. **Giới hạn bề mặt tự chỉnh sửa để chặn tự phá code và gian lận điểm số (reward hacking)**:  
   Cho mô hình nhỏ tự sửa mã nguồn Python thường dẫn đến lỗi sập chương trình hoặc âm thầm nới lỏng bộ đánh giá. MAVE-Q quy định rõ danh sách tham số được phép sửa (trọng số ưu tiên cơ chế, prompt, cửa sổ thời gian, độ sâu AST, `best_of_n`, hệ số suy giảm bộ nhớ) và khóa cứng mọi thành phần chấm điểm, chia tập và tính chi phí.
4. **Đưa vi cấu trúc thị trường mới nổi làm bài kiểm tra khoa học thực thụ**:  
   Nhiều chiến lược đẹp lung linh khi backtest không ma sát sẽ sụp đổ ngay khi áp luật T+2.5, không mua được cổ phiếu kịch trần / không bán được cổ phiếu nằm sàn, thuế bán 0.1%, mã hủy niêm yết và độ trễ công bố BCTC. Hộp cát của MAVE-Q phơi bày trung thực những điểm gãy này.

---

### Kiến Trúc Hệ Thống (Tóm Tắt Dạng Văn Bản)

MAVE-Q vận hành theo vòng lặp hai cấp bao quanh một hộp cát đánh giá được khóa cứng:

1. **Tầng Dữ liệu & Thị trường (`src/vn_rsi/data/`)**: Tạo bản chụp Parquet bất biến (từ `vnstock` có giới hạn tốc độ hoặc bộ sinh mô phỏng VN tất định), kiểm tra mã băm SHA-256, xác nhận BCTC chỉ nối sau ngày công bố và chạy bài test xáo trộn dữ liệu tương lai để đảm bảo không có rò rỉ.
2. **Hộp cát & Bộ Đánh giá Niêm phong (`src/vn_rsi/sandbox/`, `src/vn_rsi/eval/`)**: Chia thời gian thành 3 vùng rạch ròi — `search` (agent được xem thống kê), `validation` (chỉ trả về các con số điểm tổng hợp) và `sealed_test` (có sổ cái ghi nhật ký, từ chối mọi yêu cầu chấm lại lần 2 trên cùng cấu hình). Mọi tín hiệu đều đi qua bộ mô phỏng khớp lệnh theo luật chứng khoán Việt Nam.
3. **Vòng Nghiên cứu Bên trong (`src/vn_rsi/agents/`, `dsl/`, `memory/`, `models/`)**: Gồm 7 vai trò phối hợp dưới ngân sách cứng về token và thời gian:
   - `Planner` & `DataSteward` đọc thống kê vùng `search` và chia nguồn lực khám phá theo 4 nhóm cơ chế kinh tế (đà tăng kèm thanh khoản, đảo chiều sau T+2.5, giá trị/chất lượng BCTC Point-in-Time và dòng tiền khối ngoại).
   - `IdeaMiner` (Model 1: SLM gọn nhẹ) viết giả thuyết có thể bác bỏ kèm dấu kỳ vọng và điều kiện loại bỏ *trước khi* tạo công thức.
   - `Implementer` (Model 2: `DiscreteASTDiffusion 6.36M` + Model 3: đầu nhanh `z_L` của `HRMReasoner`) sinh mới hoặc đột biến nhánh con các cây AST nhân quả rồi chấm sơ loại trước khi backtest.
   - `LeakageAuditor` kiểm tra ngữ pháp, chạy shift-test động và lọc bỏ các cây trùng cấu trúc.
   - `EvaluatorProxy` kiểm định giả thuyết trên `search` và chấm tổ hợp nhân tố qua điểm vô hướng của `validation`.
   - `Librarian` cập nhật bộ nhớ niềm tin `EpistemicBeliefMemory` (có suy giảm theo thời gian và phạt khi mâu thuẫn dấu) đồng thời xuất các mẫu thành công để tự huấn luyện ReST / QLoRA.
4. **Vòng Tự Cải Tiến Bên ngoài (`src/vn_rsi/rsi/`)**: Chọn cá thể cha từ kho `DGMArchive`, đề xuất tinh chỉnh trong phạm vi bề mặt khả biến (qua đầu chậm `z_H` của `HRM` hoặc meta-proposer ≤8B), đánh giá ứng viên con trên nhiều seed ghép cặp và chỉ giữ lại khi vượt qua cổng kiểm định thống kê.

---

### Các Độ Đo Sử Dụng & Tác Dụng Của Từng Độ Đo

| Độ đo | Ý nghĩa | Dùng để làm gì trong nghiên cứu này? |
| :--- | :--- | :--- |
| **Rank IC** (Hệ số tương quan hạng Spearman) | Độ tương quan thứ hạng hàng ngày giữa tín hiệu dự báo cuối phiên `t` và tỷ suất sinh lời thực tế 5 phiên sau đó. | Đo sức mạnh phân loại cổ phiếu tốt/xấu mà không bị méo do vài mã tăng giảm cực đoan. |
| **ICIR** (Tỷ số thông tin của IC) | Giá trị trung bình Rank IC chia cho độ lệch chuẩn của nó (quy năm). | Đánh giá độ ổn định của tín hiệu; nhân tố có IC trung bình cao nhưng trồi sụt thất thường sẽ khó dùng thực tế. |
| **Net Sharpe Ratio** | Lợi nhuận vượt trội chia cho độ biến động quy năm **sau khi đã trừ sạch** phí môi giới, thuế bán 0.1%, chênh lệch mua/bán và trượt giá thanh khoản. | Thước đo hiệu quả sinh lời điều chỉnh theo rủi ro sát với thực tế giao dịch tại Việt Nam. |
| **Net Sortino Ratio** | Tương tự Sharpe ròng nhưng chỉ phạt phần biến động giảm giá. | Phản ánh rõ rủi ro sập sâu, nhất là khi cổ phiếu nằm sàn liên tiếp không bán được. |
| **Max Drawdown (MaxDD)** | Mức sụt giảm lớn nhất từ đỉnh xuống đáy của danh mục. | Cho biết khả năng chịu đòn của chiến lược qua các giai đoạn thị trường hoảng loạn hoặc có mã hủy niêm yết. |
| **Daily Turnover** | Tỷ lệ giá trị danh mục mua bán xoay vòng mỗi ngày. | Kiểm soát chi phí ẩn; ở Việt Nam, xoay vòng quá dày sẽ bị thuế bán 0.1% và trượt giá bào mòn hết lợi nhuận. |
| **Deflated Sharpe Ratio (DSR)** | Xác suất Sharpe thật > 0 sau khi hiệu chỉnh độ lệch, đuôi dày, độ dài chuỗi thời gian và **tổng số phép thử đã chạy** (`Bailey & López de Prado, 2014`). | Rất quan trọng với hệ thống AI tự thử hàng chục/hàng trăm ý tưởng: tránh tự đánh lừa mình bởi kết quả ăn may do thử nhiều lần. |
| **Paired Block-Bootstrap 95% CI** | Khoảng tin cậy 95% cho mức chênh lệch Sharpe giữa agent và baseline bằng lấy mẫu lại theo khối (giữ nguyên tương quan chuỗi thời gian). | Xác minh mức cải thiện là có ý nghĩa thống kê chứ không phải ăn may ở một seed ngẫu nhiên. |

---

### Cấu Trúc Thư Mục & Cách Chạy Nhanh

Xem sơ đồ cây thư mục chi tiết ở mục [Repository Structure](#repository-structure) phía trên. Mọi lệnh chạy hoàn toàn giống nhau:

1. **Cài đặt & kiểm tra toàn bộ hệ thống** (lint + kiểu dữ liệu + 21 bài test chống rò rỉ + đối chiếu bảng bài báo):
   ```bash
   git clone https://github.com/KenyaWashed/VN_Trading_Agents.git
   cd VN_Trading_Agents
   uv sync --extra dev          # hoặc: pip install -e ".[dev]"
   uv run python scripts/ci_check.py
   ```
2. **Chạy trọn bộ thực nghiệm từ tạo dữ liệu đến xuất bảng LaTeX**:
   ```bash
   uv run python experiments/run_data.py
   uv run python experiments/run_baselines.py --reset-ledger
   uv run python experiments/run_agent_v0.py
   uv run python experiments/run_rsi.py
   uv run python experiments/run_modular_rsi.py --preset cpu_fast
   uv run python experiments/build_report.py
   uv run python paper/verify_paper.py
   ```
3. **Huấn luyện mô hình & chạy trong Docker cô lập**:
   ```bash
   # Chạy trực tiếp bằng notebook: notebooks/train_mave_q_kaggle_colab.ipynb

   # Hoặc chạy script huấn luyện:
   pip install -q unsloth trl datasets bitsandbytes accelerate
   python train_colab_kaggle.py \
       --preset kaggle_t4_optimal \
       --enable-unsloth-qlora \
       --slm-model unsloth/Qwen3-4B-bnb-4bit \
       --epochs 25 \
       --outer-iters 4

   # Chạy cô lập trong Docker không kết nối mạng:
   docker build -t mave-q-sandbox .
   docker run --rm --network=none --read-only --tmpfs /tmp -v "$(pwd)/logs:/workspace/logs" mave-q-sandbox
   ```
