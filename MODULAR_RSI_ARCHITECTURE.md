# MAVE-Q & Holo-MAVE-Q (v3): Free Cloud Training & Scaling Guide

**Paper**: *MAVE-Q: Modular Autonomous Verified Evolution for Small-Model Quantitative Research in Rule-Constrained Emerging Markets*  
**Author**: **Tran Duy Khang** — *AI Software Engineer / Freelance Vibe Researcher*  
**Notebook**: [`notebooks/train_mave_q_kaggle_colab.ipynb`](notebooks/train_mave_q_kaggle_colab.ipynb) | **Serverless Modal**: [`modal_train.py`](modal_train.py)

---

## 1. Motivation: Breaking Down the 8B Monolith for Free Cloud Compute ($0)

Training a single monolithic 8B LLM inside a bi-level Recursive Self-Improvement (RSI) loop is bottlenecked by VRAM and rollout latency on consumer laptops or free cloud GPUs. Because our causal factor discovery grammar ([`src/vn_rsi/dsl/operators.py`](src/vn_rsi/dsl/operators.py)) spans **70 closed causal DSL tokens**, decomposing the system into specialized trainable modules (`Tree-MDLM` + `Holo-HRM` + optional 4-bit QLoRA `Qwen3` narrator) achieves higher sample efficiency, **true weight-level RSI**, and **100% compatibility with FREE cloud GPUs & TPUs**.

---

## 2. Calibrated 5-Tier Scale Presets (`src/vn_rsi/models/cloud_presets.py`)

| Preset | Target Free Platform | `Tree-MDLM` / `AST-Diff` Config | `Holo-HRM` / `HRM` Config | Total Structural Params | Est. Train VRAM | Precision |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `cpu_fast` | Local CPU (0 GPU) | `d=64, L=2, H=4` | `d=64, L=1` | `~0.2M` | CPU RAM | FP32 |
| `kaggle_t4_optimal` | **Google Colab Free (`1x T4 16GB`)** | `d=256, L=8, H=8` (`6.4M`) | `d=256, L=4` (`3.4M`) | **`~10M`** | **`< 0.25 GB`** | FP16 AMP |
| `kaggle_max_100m` | **Kaggle Free (`T4 x2 32GB`, 30h/wk)** | `d=768, L=14, H=12` (`99.4M`) | `d=768, L=10` (`58.6M`) | **`~158M`** | **`~1.6 GB`** | FP16 AMP |
| `tpu_v5e_350m` | **Kaggle Free (`TPU v5e-8 128GB HBM`, 20h/wk)** | `d=1024, L=20, H=16` (`252M`) | `d=1024, L=10` (`105M`) | **`~357M`** | **`~3.6 GB HBM`** | Native BF16 (`torch_xla`) |
| `cloud_a100_350m` | **Modal Free ($30/mo `A100-80GB`/`H100`)** | `d=1024, L=20, H=16` (`252M`) | `d=1024, L=10` (`105M`) | **`~357M`** | **`~3.6 GB HBM`** | Native BF16 |

---

## 3. Free Cloud Platform Comparison (Where to Train Larger Models for $0)

| Free Cloud Platform | Free Hardware & Quota | Credit Card Needed? | Best Use Case for MAVE-Q / Holo-MAVE-Q |
| :--- | :--- | :---: | :--- |
| **Kaggle Notebooks Free (GPU)** | **`T4 x2` (`32 GB VRAM`), 30 hrs/week**, 12h background commits | No (phone verify) | Best everyday zero-cost GPU: `cuda:0` runs `Qwen3-8B/4B` 4-bit QLoRA (~4.5–6.5 GB), `cuda:1` runs `kaggle_max_100m` (`~158M`). |
| **Kaggle Notebooks Free (TPU)** | **`TPU v5e-8` (`128 GB HBM`), 20 hrs/week**, 9h sessions | No (phone verify) | Largest zero-card memory pool: runs `tpu_v5e_350m` (`~357M` native BF16 via `torch_xla`). |
| **Modal Serverless Free** | **$30/month recurring credits** (`A100-80GB` ~12h/mo, `H100-80GB` ~7.5h/mo), billed per second | Card verify ($0 spent) | Best single-GPU speed & scale: `modal run modal_train.py --preset cloud_a100_350m --enable-slm-qlora` trains `~357M` core + `Qwen3-8B` BF16 QLoRA in ~4–8 min (~$0.30/run!). |
| **Google Colab Free** | **`1x T4` (`16 GB VRAM`)**, dynamic quota | No | Fast interactive notebook debugging (`kaggle_t4_optimal` + `Qwen3-4B/1.7B` QLoRA). |
| **Lightning AI Studios** | **~$30 starter credits (`L4`/`A10G`/`A100`)** + free CPU IDE | Card verify | Persistent VS Code cloud workspace for multi-file development. |
| **Google TPU Research Cloud (TRC)** | Free 30-day TPU v4/v5e pods for open research papers | Academic/open app | Full `N=20` multi-seed empirical `vnstock` runs for arXiv / conference submission. |

---

## 4. Quickstart Commands

```bash
# 1. Universal CLI Trainer (auto-detects Kaggle GPU/TPU, Colab, Modal, Lightning, CPU):
python train_colab_kaggle.py --preset auto --mode holo_v3 --epochs 12 --rounds 7

# 2. Full Kaggle T4 x2 Run (Holo-MAVE-Q v3 + Modular v1 + Qwen3-4B/8B 4-bit QLoRA with auto PEFT fallback):
python train_colab_kaggle.py \
    --preset kaggle_max_100m \
    --mode full \
    --enable-slm-qlora \
    --slm-model unsloth/Qwen3-4B-bnb-4bit \
    --epochs 16

# 3. Serverless Modal A100-80GB Run (~357M Holo-MAVE-Q + Qwen3-8B QLoRA):
pip install modal && modal setup
modal run modal_train.py --preset cloud_a100_350m --enable-slm-qlora
modal volume get mave-q-artifacts /checkpoints ./artifacts/modal_checkpoints
```
