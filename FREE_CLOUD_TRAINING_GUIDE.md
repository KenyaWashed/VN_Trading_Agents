# FREE_CLOUD_TRAINING_GUIDE.md — 100% Free Cloud GPU/TPU Training & Veteran CLI Guide

Train **MAVE-Q v4 (`VetQuant-RSI`)** — including both the Neuro-Symbolic Alpha Core (`Tree-MDLM` + `Holo-HRM` + `VeteranCaseMemory`) AND the **10-Year Veteran CIO Reasoning SLM** (`Qwen3-1.7B/4B` 4-bit QLoRA SFT + DPO) — for **$0.00** on free cloud accelerators (`< 4.5 GB` peak VRAM on T4).

> **Research Disclaimer**: Strictly for academic research and offline simulation. Zero brokerage connectivity, zero live trading, zero investment advice.

---

## 1. Free Cloud Platform Matrix ($0 Cost)

| Platform | Free Hardware | Free Quota | Recommended Preset | Structural Params | Peak VRAM (Est.) |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **Google Colab Free** | `1x NVIDIA T4` (16 GB) | Dynamic (~4–12h/day) | `kaggle_t4_optimal` | `~10M` | `< 0.25 GB` core + `3.4 GB` (`Qwen3-1.7B`) or `4.3 GB` (`Qwen3-4B`) |
| **Kaggle Notebooks Free (GPU)** | `2x NVIDIA T4` (32 GB) | `30 hrs/week`, 12h commits | `kaggle_max_100m` | `~158M` | `~1.6 GB` (`cuda:1`) + `~4.3 GB` (`cuda:0`) |
| **Kaggle Notebooks Free (TPU)** | `TPU v5e-8` (128 GB HBM) | `20 hrs/week`, 9h sessions | `tpu_v5e_350m` | `~357M` | `~3.6 GB HBM` (Native BF16 via `torch_xla`) |
| **Modal Serverless Free** | `A100-80GB` / `H100-80GB` | `$30/mo` recurring free credit | `cloud_a100_350m` | `~357M` | `~3.6 GB` + `Qwen3-8B` QLoRA (~$0.30/run) |
| **Local Laptop / CPU** | Any CPU (0 GPU) | Unlimited | `cpu_fast` | `96.5K` | `~85 MB RAM` (88s full run / 33s smoke) |

---

## 2. Option A — Google Colab Free (`1x T4 16 GB`, Zero Card Required)

1. Open [Google Colab](https://colab.research.google.com/) -> **File -> Upload Notebook** -> select [`notebooks/train_mave_q_kaggle_colab.ipynb`](notebooks/train_mave_q_kaggle_colab.ipynb).
2. Menu **Runtime -> Change runtime type -> T4 GPU**.
3. Mount Google Drive (optional, for persistent `--resume` across 12-hour disconnects):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   CKPT_DIR = '/content/drive/MyDrive/maveq_v4_ckpts'
   ```
4. Run Cells 1–7 sequentially. `train_colab_kaggle.py` automatically:
   - Detects `1x T4` and selects `kaggle_t4_optimal` (`~10M` FP16 structural core, `< 0.25 GB` VRAM).
   - Trains Stage 1 (`MAVE-Q v4`), exports checkpoints (`v4_tree_mdlm_*.pt`, `v4_holo_hrm_*.pt`, `v4_case_memory_*.npz`, `maveq_v4_bundle_*.json`), and executes `gc.collect()` + `torch.cuda.empty_cache()`.
   - Mines 90 strictly search-zone 5-step SFT + DPO trajectories (`<= 2022-12-30`).
   - Trains Stage 4: 4-bit NF4 QLoRA SFT -> reference-free PEFT DPO on `Qwen/Qwen3-1.7B` (`~3.4 GB` peak VRAM) or `Qwen/Qwen3-4B` (`~4.3 GB` peak VRAM).
   - Saves `resume_state_*.json` after every stage so re-running the exact same command with `--resume` after a session timeout skips completed stages in `< 0.05s`!

---

## 3. Option B — Kaggle Notebooks Free (`T4 x2` 30h/wk OR `TPU v5e-8` 20h/wk)

1. Go to [Kaggle Notebooks](https://www.kaggle.com/code) -> **Import Notebook** -> upload [`notebooks/train_mave_q_kaggle_colab.ipynb`](notebooks/train_mave_q_kaggle_colab.ipynb).
2. In the right sidebar -> **Settings -> Accelerator**:
   - Choose **`GPU T4 x2`** (recommended for full core + Veteran QLoRA/DPO) OR **`TPU VM v5e-8`** (for `~357M` BF16 structural core).
   - Toggle **Internet: On** and **Persistence: Files only** (preserves `/kaggle/working/maveq_v4_ckpts` across sessions!).
3. Click **Save Version -> Save & Run All (Commit)** to train unattended in the background up to 12 hours!
4. Or run via Kaggle terminal / cell:
   ```bash
   python train_colab_kaggle.py \
       --preset auto \
       --mode maveq_v4 \
       --slm-model Qwen/Qwen3-4B \
       --enable-veteran-dpo \
       --rounds 6 \
       --epochs 10 \
       --checkpoint-dir /kaggle/working/maveq_v4_ckpts \
       --resume
   ```

---

## 4. Option C — Modal Serverless Free (`$30/mo` Credits -> A100-80GB)

```bash
pip install modal && modal setup
modal run modal_train.py --preset cloud_a100_350m --enable-slm-qlora
modal volume get mave-q-artifacts /checkpoints ./artifacts/modal_checkpoints
```

---

## 5. Resuming After Free-Tier Timeout (`--resume`)

If Colab or Kaggle disconnects mid-run, simply re-run the **exact same command** with `--resume` pointing to the same `--checkpoint-dir`:
```bash
python train_colab_kaggle.py --preset auto --mode maveq_v4 --enable-veteran-dpo --checkpoint-dir artifacts/checkpoints --resume
```
- Completed stages are listed under `"resumed_stages"` in the telemetry JSON (`stage1_maveq_v4` resumes in `0.02s`).
- Peak VRAM and per-stage wall-clock timings are printed in `"stage_vram_telemetry"` and `"stage_timings_sec"`.

---

## 6. Asking the 10-Year Veteran Agent Natural-Language Questions (CLI)

Once checkpoints are exported (or even without a GPU, using the instant CPU Neuro-Symbolic fallback), query the agent in **Vietnamese** or **English**:

```bash
# Vietnamese 5-step Veteran CIO Analysis (loads bundle + VeteranCaseMemory analogs + optional QLoRA/DPO adapter)
python scripts/ask_veteran_agent.py \
    --symbol FPT \
    --checkpoint-dir artifacts/checkpoints \
    --lang vi \
    --question "Đánh giá cơ hội mua FPT và rủi ro kẹt T+2.5 hiện tại?"

# English structured JSON output for automated workflows
python scripts/ask_veteran_agent.py \
    --symbol HPG \
    --checkpoint-dir artifacts/checkpoints \
    --lang en \
    --json

# Stress-test on distressed ticker (e.g. NVL_STRESS, FLC_HALT) on a specific causal date
python scripts/ask_veteran_agent.py \
    --symbol NVL_STRESS \
    --as-of 2023-10-20 \
    --lang vi \
    --question "Có nên bắt đáy NVL_STRESS không?"
```

### Switching from Synthetic CI Fixture to Live Empirical Data (`vnstock` / `yfinance`)
1. Install optional live provider: `pip install vnstock yfinance`.
2. Edit `configs/data.yaml` -> set `provider_mode: live_vnstock` and run `python experiments/run_data.py`.
3. For US/CSI international live benchmarks, pass `mode="live_yfinance"` to `MultiMarketLoader.load_or_build(...)` in `src/vn_rsi/benchmarks/multi_market.py`.


---

## 7. Pushing Your Trained Open-Source Model & Adapter to Hugging Face Hub

Both [`notebooks/train_mave_q_kaggle_colab.ipynb`](notebooks/train_mave_q_kaggle_colab.ipynb) (Cells 5, 6, 8) and [`train_colab_kaggle.py`](train_colab_kaggle.py) natively push your fine-tuned open-source LoRA/DPO adapter (or full merged FP16 model) + MAVE-Q v4 specialist checkpoints + auto-generated Model Card (`README.md`) to your Hugging Face account.

### Step 1: Configure `HF_TOKEN` Securely (Never Hardcode Tokens!)
- **Google Colab**: Click the **🔑 Secrets** icon in the left sidebar -> add `HF_TOKEN` with a **Write** token from `https://huggingface.co/settings/tokens` -> toggle **Notebook access** ON.
- **Kaggle Notebooks**: Menu **Add-ons -> Secrets** -> add `HF_TOKEN` -> check the box to attach it to the notebook.
- **Terminal / Local CLI**: `export HF_TOKEN="hf_..."` (Linux/macOS) or `$env:HF_TOKEN="hf_..."` (PowerShell).

### Step 2: Train & Push in One Command (or Run Cells 5–8 in the Notebook)
```bash
# Push lightweight PEFT LoRA+DPO adapter (~30 MB) + MAVE-Q v4 specialist weights
python train_colab_kaggle.py \
    --preset auto \
    --mode maveq_v4 \
    --slm-model Qwen/Qwen3-1.7B \
    --enable-veteran-dpo \
    --strict-slm \
    --hf-repo-id maveq-v4-qwen3-veteran \
    --hf-dataset-repo maveq-v4-veteran-dpo-data \
    --resume

# Or merge LoRA weights into a standalone FP16 model (~3.4 GB) before pushing:
python train_colab_kaggle.py \
    --preset auto \
    --mode maveq_v4 \
    --slm-model Qwen/Qwen3-1.7B \
    --enable-veteran-dpo \
    --hf-repo-id your-username/maveq-v4-qwen3-veteran-merged \
    --hf-merge-16bit \
    --resume
```
*(If you pass just `--hf-repo-id maveq-v4-qwen3-veteran` without a slash, `HfApi.whoami()` automatically prepends your Hugging Face username!)*
