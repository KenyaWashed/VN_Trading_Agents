# Modular Micro-RSI Architecture & Free Cloud GPU (Kaggle / Colab) Training Guide

**Author**: **Tran Duy Khang** — *AI Software Engineer / Freelance Vibe Researcher*  
**Project**: `VN-RSI` (Leak-Free Recursively Self-Improving Multi-Agent Research System for Vietnamese Equities)

---

## 1. Motivation: Breaking Down the 8B Monolith for Free Compute ($0)

Training or fine-tuning a single monolithic 8B LLM inside a bi-level Recursive Self-Improvement (RSI) loop is bottlenecked by VRAM and rollout latency on consumer laptops or free cloud GPUs. Because our causal factor discovery grammar ([`src/vn_rsi/dsl/operators.py`](src/vn_rsi/dsl/operators.py)) spans only **70 closed causal DSL tokens** (`18` operators $\times$ canonical VN lookback windows + `18` causal terminals), decomposing the monolith into **3 specialized micro-models** achieves higher sample efficiency, **true weight-level RSI**, and **100% compatibility with FREE Kaggle & Colab GPUs**.

---

## 2. Exact Calibrated Model Breakdown & PyTorch VRAM Measurements

Verified directly via PyTorch (`src/vn_rsi/models/`):

| Model Role | Module Path | Architecture | `cpu_fast` Preset | `kaggle_t4_optimal` Preset | Est. Training VRAM |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Model 1: Financial Ideation & Critique** (`IdeaMiner`) | [`train_colab_kaggle.py`](train_colab_kaggle.py) (`Unsloth 4-bit QLoRA`) | Pretrained SLM (`Qwen3-4B-bnb-4bit` or `Qwen3-1.7B-bnb-4bit`) | Scaffold fallback | **4.0B** *(or **1.7B** fast alt)* | **~4.2 – 4.5 GB** *(~2.0 GB for 1.7B)* |
| **Model 2: Factor Synthesizer & Mutator** (`Implementer`) | [`src/vn_rsi/models/ast_diffusion.py`](src/vn_rsi/models/ast_diffusion.py) | **Discrete Masked AST Diffusion (Tree-MDLM)** w/ Grammar-Constrained Unmasking & Subtree Infilling | `110,018` params (`0.11M`) | **6,358,082 params (`6.36M`)** | **129.02 MB** (`0.13 GB`) |
| **Model 3: Regime Planner & Surrogate Verifier** (`Planner` / `Outer Proposer`) | [`src/vn_rsi/models/hrm_reasoner.py`](src/vn_rsi/models/hrm_reasoner.py) | **HRM / TRM Hierarchical Recursive Reasoner** (Slow $z_H$ + Fast $z_L$) | `117,575` params (`0.12M`) | **3,377,927 params (`3.38M`)** | **75.54 MB** (`0.08 GB`) |
| **Total Trainable Micro-Core (Models 2 + 3)** | [`src/vn_rsi/models/modular_backend.py`](src/vn_rsi/models/modular_backend.py) | Combined Structural RSI Core | `~0.23M` | **9,736,009 params (`9.74M`)** | **~204.56 MB (`< 0.21 GB`)** |

---

## 3. Environment Compatibility Verdict: Kaggle vs. Google Colab

| Cloud Environment (FREE Tier) | Free Hardware | Quota | Compatibility Verdict | Recommended Device Allocation |
| :--- | :--- | :--- | :---: | :--- |
| **Kaggle Notebooks Free** *(Recommended #1)* | **Dual NVIDIA T4 (`T4 x2` = 2 $\times$ 16 GB VRAM)**, 30 GB RAM | **30 hrs / week** (up to 12h background commits) | **EXCELLENT (100% Supported)** | `cuda:0`: `Qwen3-4B-bnb-4bit` QLoRA (~4.5 GB / 16 GB)<br>`cuda:1`: `DiscreteASTDiffusion (6.36M)` + `HRMReasoner (3.38M)` (~0.21 GB / 16 GB) |
| **Google Colab Free** *(Recommended #2)* | **Single NVIDIA T4 (`1x 16 GB VRAM`)**, 12 GB RAM | Dynamic (interactive session) | **FULLY SUPPORTED** | `cuda:0`: Runs `DiscreteASTDiffusion` + `HRMReasoner` (~0.21 GB), then `Qwen3-4B` or `Qwen3-1.7B` Unsloth 4-bit QLoRA (~4.5 GB / 2.0 GB) with automatic `gc.collect()` + `torch.cuda.empty_cache()` cleanup |
| **Local Laptop CPU** | 4+ CPU cores, 16 GB RAM, 0 GPU | Unlimited | **SUPPORTED (`cpu_fast`)** | Automatic fallback (`preset="cpu_fast"`), zero CUDA required for CI, unit tests, and prototyping |

---

## 4. Verified End-to-End Test & Sealed Sandbox Results

All **21 automated unit/leakage/seal/model tests** (`pytest`), `ruff`, `mypy` (37 files), PyTorch E2E training checks, and `paper/verify_paper.py` (19/19 citations, 66/66 table cells) passed with **zero errors**:

| Method (`is_synthetic=true` VN Microstructure Benchmark) | Val Rank IC | Val Net Sharpe | Sealed Test Rank IC | Sealed Test Net Sharpe | Provenance `run_id` |
| :--- | :---: | :---: | :---: | :---: | :--- |
| VN-Index Buy & Hold | 0.0000 | 0.506 | 0.0000 | 1.248 | `phase2_baselines` |
| Equal-Weight Universe | 0.0000 | 0.454 | 0.0000 | 1.195 | `phase2_baselines` |
| Purged Walk-Forward Ridge | 0.1026 | 0.702 | 0.0781 | 0.327 | `phase2_baselines` |
| Agent $v_0$ ($\le$8B, No RSI) | 0.1036 | 0.677 | 0.0910 | 1.339 | `phase3_agent_v0` |
| Monolithic VN-RSI $v^*$ ($\le$8B Bi-Level) | 0.1025 | 0.877 | 0.0637 | 1.432 | `phase4_rsi` |
| **Modular Micro-RSI (`HRM + Discrete AST Diffusion`)** | **0.1045** | **1.019** | **0.0539** | **1.918** *(DSR 0.389)* | `phase4_modular_rsi` |

> **Honest Disclosure**: Results above were collected on our deterministic synthetic Vietnamese microstructure benchmark (`is_synthetic=true`, enforcing T+2.5 settlement, $\pm 7\%/10\%/15\%$ ceiling/floor blocks, 0.15% fee, 0.10% sell tax, and delisted tickers). Before submitting to a workshop/arXiv, re-run on Kaggle `T4 x2` using your local frozen empirical `vnstock` snapshot and live `Qwen3-4B` QLoRA weights.

---

## 5. How to Train & Reproduce on Kaggle or Google Colab

### Option A: Drag-and-Drop Notebook (Fastest)
Upload [`notebooks/train_vn_rsi_colab_kaggle.ipynb`](notebooks/train_vn_rsi_colab_kaggle.ipynb) directly to **Kaggle** (select Accelerator: **GPU T4 x2**) or **Google Colab** (Runtime $\to$ **T4 GPU**), set your GitHub repo URL in Cell 2, and click **Run All**.

### Option B: Direct CLI Script (`train_colab_kaggle.py`)
```bash
# Install dependencies on Kaggle / Colab
pip install -q -e ".[dev]"
pip install -q unsloth trl datasets bitsandbytes accelerate

# Run full Modular Micro-RSI training (Qwen3-4B 4-bit QLoRA + 6.36M AST Diffusion + 3.38M HRM)
python train_colab_kaggle.py \
    --preset kaggle_t4_optimal \
    --enable-unsloth-qlora \
    --slm-model unsloth/Qwen3-4B-bnb-4bit \
    --epochs 25 \
    --outer-iters 4

# Faster 2x variant on Colab 1x T4:
python train_colab_kaggle.py \
    --preset kaggle_t4_optimal \
    --enable-unsloth-qlora \
    --slm-model unsloth/Qwen3-1.7B-bnb-4bit \
    --epochs 25 \
    --outer-iters 4
```
