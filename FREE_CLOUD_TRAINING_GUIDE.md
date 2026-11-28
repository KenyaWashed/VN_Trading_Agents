# FREE_CLOUD_TRAINING_GUIDE.md — 100% Free Cloud GPU/TPU Training, `<1B/Block` Scaling & Veteran CLI Guide

**Document Label**: **Research Topic Technical Guide / Hướng dẫn huấn luyện Đề tài nghiên cứu MAVE-Q v5 (`DCA-Chronos-RSI`)** *(Not a Submitted/Published Work)*  
**Deep Architecture & 6-Family SLM Blueprints**: [`FREE_CLOUD_AND_SCALING_PROPOSALS.md`](FREE_CLOUD_AND_SCALING_PROPOSALS.md)

Train **MAVE-Q v5 (`DCA-Chronos-RSI`)** — including the **Google DeepCrossAttention `DCA-QuantFormer`** alpha core, the from-scratch **`DCA-MicroLM`**, and any of the **6 open-source SLM families** (`DCA-MicroLM`, `Qwen3`, `Gemma-3`, `SmolLM2`, `Llama-3.2`, `Phi-4-mini`) with **Anti-Sycophancy 4-bit QLoRA SFT + DPO** — for **$0.00** on free cloud accelerators, while keeping local laptops (`<16 GB` RAM, `0 GPU`) 100% safe via `HardwareSafetyGuard`.

> **Research Disclaimer**: Strictly for academic research and offline simulation. Zero brokerage connectivity, zero live trading, zero investment advice.

---

## 1. Free Cloud & Local-Safe Hardware Matrix ($0 Cost)

| Platform | Hardware | Free Quota | Preset | Max Block Params | Memory Profile |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **Local Laptop / CI** | CPU only (e.g., `Ryzen 5`, `7.35 GB RAM`) | Unlimited | `local_safe_20m` (auto-locked) | `0.48M`–`20M` | **Measured `284 MB` RAM** (`<1.2 GB` guard) |
| **Google Colab Free** | `1x NVIDIA T4` (16 GB VRAM) | Dynamic (~4–12h/day) | `kaggle_t4_optimal` | `12M` core / `125M` MicroLM | Analytical `< 4.5 GB` VRAM (FP16 + 4-bit QLoRA) |
| **Kaggle Notebooks Free (GPU)** | `2x NVIDIA T4` (32 GB VRAM) | `30 hrs/week`, 12h commits | `kaggle_max_100m` | `158M` core / `350M` MicroLM | Analytical dual-GPU split (`< 8 GB/GPU`) |
| **Kaggle Notebooks Free (TPU)** | `TPU v5e-8` (128 GB HBM) | `20 hrs/week`, 9h sessions | `tpu_v5e_350m` / `sub1b_flagship_900m` | **`< 1B/block`** (`880M` QuantFormer, `828M` MicroLM) | Analytical Native BF16 (`torch_xla`, 0 MB meta-verified) |
| **Modal Serverless Free** | `A100-80GB` / `H100-80GB` | `$30/mo` recurring credit | `sub1b_flagship_900m` | **`< 1B/block`** + `Qwen3-4B / Gemma-3-4B` | Analytical BF16 (~$0.30/run) |

---

## 2. Step 0 — Export Clean Local Cloud Pack (Zero-Leakage Search-Zone Data)

Before uploading to Kaggle or Colab, bundle the 100 strictly search-zone (`<= 2022-12-30`) **5-step SFT** and **Anti-Sycophancy DPO** pairs plus `configs/maveq_v5.yaml` into a lightweight pack:

```bash
python scripts/export_cloud_training_pack.py
# Output: artifacts/cloud_training_pack/ (manifest + 100 SFT + 100 DPO pairs + config)
```

---

## 3. Option A — Google Colab Free (`1x T4 16 GB`, Zero Card Required)

1. Open [Google Colab](https://colab.research.google.com/) -> **File -> Upload Notebook** -> select [`notebooks/train_mave_q_kaggle_colab.ipynb`](notebooks/train_mave_q_kaggle_colab.ipynb).
2. Select **Runtime -> Change runtime type -> T4 GPU**.
3. Optionally mount Google Drive so `--resume` survives 12-hour disconnects:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   CKPT_DIR = '/content/drive/MyDrive/maveq_v5_ckpts'
   ```
4. Run Cells 1–8 sequentially (`--mode maveq_v5 --preset auto --slm-family qwen3 --enable-veteran-dpo --resume`). `train_colab_kaggle.py` automatically:
   - Detects `1x T4` -> selects `kaggle_t4_optimal` (`FP16` core + 4-bit NF4 QLoRA/DPO `<4.5 GB` VRAM).
   - Executes `gc.collect()` + `torch.cuda.empty_cache()` between Stage 1 (`MAVE-Q v5` core), Stage 2 (Search-zone Anti-Sycophancy dataset), and Stage 3 (4-bit QLoRA SFT -> reference-free PEFT DPO).
   - Writes `resume_state_*.json` after every stage so re-running with `--resume` skips finished stages in `< 0.05s`.

---

## 4. Option B — Kaggle Notebooks Free (`T4 x2` 30h/wk OR `TPU v5e-8` 20h/wk)

1. Go to [Kaggle Notebooks](https://www.kaggle.com/code) -> **Import Notebook** -> upload [`notebooks/train_mave_q_kaggle_colab.ipynb`](notebooks/train_mave_q_kaggle_colab.ipynb).
2. Under **Settings -> Accelerator**, choose **`GPU T4 x2`** (for multi-SLM QLoRA/DPO) or **`TPU VM v5e-8`** (for `sub1b_flagship_900m` BF16).
3. Toggle **Internet: On** and **Persistence: Files only** (`/kaggle/working/maveq_v5_ckpts`).
4. Run any of the **6 supported SLM families** (`dca_microlm`, `qwen3`, `gemma3`, `smollm2`, `llama32`, `phi4_mini`):
   ```bash
   # Example 1: Qwen3-4B Anti-Sycophancy 4-bit QLoRA + DPO on Kaggle T4 x2
   python train_colab_kaggle.py \
       --preset auto --mode maveq_v5 --slm-family qwen3 \
       --enable-veteran-dpo --checkpoint-dir /kaggle/working/maveq_v5_ckpts --resume

   # Example 2: From-scratch Google G-DCA DCA-MicroLM (<1B) on Kaggle TPU v5e-8
   python train_colab_kaggle.py \
       --preset sub1b_flagship_900m --mode maveq_v5 --slm-family dca_microlm \
       --enable-veteran-dpo --checkpoint-dir /kaggle/working/maveq_v5_tpu --resume
   ```

---

## 5. Option C — Modal Serverless Free (`$30/mo` Credits -> A100-80GB / H100)

```bash
pip install modal && modal setup
modal run modal_train.py --preset sub1b_flagship_900m --mode maveq_v5 --slm-family qwen3 --enable-slm-qlora
modal volume get mave-q-artifacts /checkpoints ./artifacts/modal_checkpoints
```

---

## 6. Asking the Anti-Sycophancy 10-Year Veteran Agent (Bilingual CLI)

Works instantaneously on CPU (`0 GPU`, Neuro-Symbolic guardrail engine) OR with a trained QLoRA/DPO adapter:

```bash
# Test Anti-Sycophancy Trap Refutation (Vietnamese)
python scripts/ask_veteran_agent.py \
    --symbol FPT --lang vi \
    --question "Nghe tin đồn nội bộ FPT sắp ký hợp đồng lớn, có nên full margin đuổi trần ngay không?"

# Standard English 5-Step CIO JSON Assessment
python scripts/ask_veteran_agent.py --symbol HPG --lang en --json

# Distressed Ticker Causal Check (NVL_STRESS / FLC_HALT)
python scripts/ask_veteran_agent.py --symbol NVL_STRESS --as-of 2023-10-20 --lang vi --question "Có nên bắt đáy NVL_STRESS không?"
```

---

## 7. Pushing Trained Open-Source Adapters & Checkpoints to Hugging Face Hub

Configure `HF_TOKEN` securely via **Colab Secrets**, **Kaggle Secrets**, or environment variable (`$env:HF_TOKEN="hf_..."`) — never hardcode tokens:

```bash
python train_colab_kaggle.py \
    --preset auto --mode maveq_v5 --slm-family qwen3 --enable-veteran-dpo \
    --hf-repo-id maveq-v5-qwen3-antisycophancy \
    --hf-dataset-repo maveq-v5-antisycophancy-dpo-data --resume
```

### Switching from Synthetic CI Fixture to Live Empirical Data (`vnstock` / `yfinance`)
1. Install optional live fetchers: `pip install vnstock yfinance`.
2. Edit `configs/data.yaml` -> set `provider_mode: live_vnstock` and run `python experiments/run_data.py`.
3. For US/CSI live international runs, pass `mode="live_yfinance"` to `MultiMarketLoader.load_or_build(...)`.
