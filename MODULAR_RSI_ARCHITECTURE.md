# MAVE-Q v4 (`VetQuant-RSI`) Modular Architecture & Free-Cloud Scaling Guide

**Paper**: *MAVE-Q v4 (VetQuant-RSI): Veteran-Grounded Neuro-Symbolic Quantitative Research with Zero-Validation-Burn Evolution under Emerging-Market Microstructure Constraints*  
**Author**: **Tran Duy Khang** — *AI Software Engineer / Independent Quantitative AI Researcher*  
**Notebook**: [`notebooks/train_mave_q_kaggle_colab.ipynb`](notebooks/train_mave_q_kaggle_colab.ipynb) | **Step-by-Step Guide**: [`FREE_CLOUD_TRAINING_GUIDE.md`](FREE_CLOUD_TRAINING_GUIDE.md) | **Serverless Modal**: [`modal_train.py`](modal_train.py)

---

## 1. Architectural Evolution: From Monolithic 8B (`v0`) to VetQuant-RSI (`v4`)

| Generation | Core Mechanism | Official Val Queries | Val Sharpe | Sealed Test Sharpe | Sealed MaxDD | Sealed DSR |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Agent `v0` | Monolithic $\le$8B constrained pipeline | 1 | 0.677 | 1.339 | -25.1% | 0.365 |
| MAVE-Q `v1` | Discrete AST Diffusion + HRM (`diffu-GRPO`) | 40+ | 1.150 | 1.637 | -20.5% | 0.332 |
| MAVE-Q `v2` | Friction-Aware QD + Greedy PLCA | 299 | 1.286 | 1.602 | -24.1% | 0.180 |
| Holo-MAVE-Q `v3` | `TypedASTGuard` + `Tree-MDLM` + `Holo-HRM` + `MSSF` + `BTO-SRSS` | 81 | 1.291 | 1.676 | -22.0% | 0.349 |
| **MAVE-Q `v4` (`VetQuant-RSI`)** | **4-Module Veteran CIO Core + `GP-FA-Ridge` + Scale-Invariant `BAB` + Zero-Val-Burn CV + 10-Yr Veteran QLoRA/DPO** | **1** (2 total) | **1.347 ± 0.081** (8-seed) | **1.067** | **-19.8% (#1)** | **0.785 (2.25× v3)** |

---

## 2. Four Coupled MAVE-Q v4 Modules (`src/vn_rsi/models/maveq_v4_veteran.py`)

```text
Strictly Causal Feature Panel (<= t, PIT Announcement Lag Enforced)
   ├─► Module 1: VeteranCaseMemory (10-dim microstructure state -> purged search-zone k-NN trauma & analogs)
   ├─► Module 2: Tri-Cameral Pre-Mortem Core
   │      ├─► Chamber 1A: TypedASTGuard + Tree-MDLM / Bridge-MDLM + Holo-HRM + Nyquist MSSF + BTO-SRSS (Shapley-diffu-GRPO)
   │      ├─► Chamber 1B: Garleanu-Pedersen Friction-Adjusted Non-Negative FWL Ridge Backbone (7 orthogonal causal views)
   │      └─► Chamber 2 : Devil's Advocate Pre-Mortem Auditor (floor-lock cascade, blow-off T+2.5 supply, ceiling lock, distress)
   ├─► Module 3: Scale-Invariant Hierarchical Regime-Conviction Controller
   │      └─► Layer 1 (Alpha Fusion via 30d causal reflection) + Layer 2 (Vasicek-Shrunk Frazzini-Pedersen BAB + Hysteresis + Instant-Release Tail Shield)
   └─► Module 4: Zero-Validation-Burn Purged Search CV (all 62 screenings & 4 controller profiles inside local search simulator -> M=1 val query!)
```

Alongside the quantitative core, `src/vn_rsi/veteran/` implements the **10-Year Veteran Cognitive Reasoning Module**:
- `preference_builder.py`: Mines 90 strictly search-zone episodes (`<= 2022-12-30`) into 5-step CIO SFT trajectories and Veteran-vs-FOMO DPO pairs + 2-stage `<4.5 GB` VRAM 4-bit QLoRA SFT -> reference-free PEFT DPO trainer.
- `reasoning_engine.py`: 5-step bilingual (Vietnamese/English) Veteran CIO reasoning engine with calibrated probability estimation (`Brier = 0.01064`, `ECE = 0.08156`) and automatic CPU Neuro-Symbolic fallback.
- `expert_benchmark.py`: 32-scenario adversarial benchmark (`100%` Decision Accuracy, `100%` Causal Consistency, `0.8889` Hidden-Risk F1).

---

## 3. Calibrated 5-Tier Free Cloud Presets (`src/vn_rsi/models/cloud_presets.py`)

| Preset | Target Free Platform | `Tree-MDLM` Config | `Holo-HRM` Config | Structural Params | Peak VRAM (Core + QLoRA/DPO) | Precision |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `cpu_fast` | Local CPU / CI (0 GPU) | `d=64, L=2, H=4` | `d=64, L=1` | `96.5K` | `~85 MB RAM` (measured) | FP32 |
| `kaggle_t4_optimal` | **Google Colab Free (`1x T4 16GB`)** | `d=256, L=8, H=8` | `d=256, L=4` | `~10M` | `<0.25 GB` core + `3.4–4.3 GB` Qwen3-1.7B/4B (`<4.5 GB`) | FP16 AMP |
| `kaggle_max_100m` | **Kaggle Free (`T4 x2 32GB`, 30h/wk)** | `d=768, L=14, H=12` | `d=768, L=10` | `~158M` | `~1.6 GB` (`cuda:1`) + `~4.3–6.0 GB` (`cuda:0`) | FP16 AMP |
| `tpu_v5e_350m` | **Kaggle Free (`TPU v5e-8 128GB HBM`)** | `d=1024, L=20, H=16` | `d=1024, L=10` | `~357M` | `~3.6 GB HBM` | Native BF16 (`torch_xla`) |
| `cloud_a100_350m` | **Modal Free (`$30/mo A100-80GB`)** | `d=1024, L=20, H=16` | `d=1024, L=10` | `~357M` | `~3.6 GB` + `Qwen3-8B` QLoRA | Native BF16 |

---

## 4. Quickstart Commands
```bash
# Universal Free-Cloud Trainer with inter-stage VRAM cleanup & automatic --resume recovery
python train_colab_kaggle.py --preset auto --mode maveq_v4 --enable-veteran-dpo --slm-model Qwen/Qwen3-1.7B --resume

# Ask the trained 10-Year Veteran Agent natural-language questions from CLI
python scripts/ask_veteran_agent.py --symbol FPT --checkpoint-dir artifacts/checkpoints --lang vi --question "Có nên mua FPT giai đoạn này không?"
```
