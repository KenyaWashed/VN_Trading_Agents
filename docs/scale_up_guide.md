# Hardware Requirements & Scale-Up Guide

## 1. Hardware Tiers

| Tier | Hardware Spec | Configuration (`configs/llm.yaml` & `configs/data.yaml`) | Use Case |
| :--- | :--- | :--- | :--- |
| **Tier 0: CPU / Laptop Fallback** *(Verified on current machine)* | 4+ CPU cores, 16 GB RAM, 0 GPU | `active_backend: "local_8b_sim_or_ollama"` (or `llamacpp` GGUF Q4_K_M on CPU), `configured_vram_gb: 0.0`, `universe_tier: "vn30"` | Fast CI, leakage tests, unit tests, deterministic pipeline & paper build (<2 min end-to-end). |
| **Tier 1: Minimum Recommended GPU** *(Target Paper Setup)* | 1x 24 GB GPU (e.g., RTX 3090 / RTX 4090 / L4), 32 GB RAM | Serve `Qwen3-8B` (4-bit AWQ/GGUF) via `Ollama` or `vLLM`; set `active_backend: "ollama"` or `"vllm"`, `configured_vram_gb: 24.0`, `universe_tier: "vn100"` | Full live-weight ≤8B multi-agent + bi-level RSI experiments + QLoRA trajectory fine-tuning. |
| **Tier 2: Multi-GPU / Lab Server** *(Scale-Up & Teacher Ablations)* | 2x–4x 48–80 GB GPUs (A6000 / A100 / H100), 128 GB RAM | Serve `Qwen3-8B` FP16 on GPU 0 via `vLLM` continuous batching; optionally enable `teacher_backend.enabled: true` (`Qwen2.5-72B-Instruct` AWQ) on GPUs 1–2 for teacher upper-bound ablation ONLY; set `universe_tier: "all_liquid"` | Full HOSE+HNX+UPCoM liquid universe (`220+` tickers), 15+ outer RSI iterations, 10+ paired seeds. |

---

## 2. Step-by-Step Scale-Up Without Code Rewrites

### 2.1 Switching from Offline Scaffolding to Live Local `Qwen3-8B`
1. Start Ollama or vLLM locally:
   ```bash
   # Using Ollama (4-bit quantized, ~5.2 GB VRAM):
   ollama pull qwen3:8b
   ollama serve

   # Or using vLLM on Linux / WSL2 (24 GB GPU):
   vllm serve Qwen/Qwen3-8B-AWQ --port 11434 --max-model-len 4096 --gpu-memory-utilization 0.85
   ```
2. Update `configs/llm.yaml`:
   ```yaml
   active_backend: "ollama"   # or "vllm"
   model:
     name: "qwen3:8b"
     parameter_count_billions: 8.0
   hardware:
     configured_vram_gb: 24.0
   ```
3. Re-run `make agent rsi report` — zero Python changes required.

### 2.2 Scaling Universe (`VN30` z_H\toz_H `VN100` z_H\toz_H `All Liquid`)
Edit `configs/data.yaml`:
```yaml
universe_tier: "all_liquid"   # expands from 30 -> 220 symbols across HOSE, HNX, UPCoM
```
Then rebuild a new versioned snapshot (`snapshot_id: "vn_equities_all_v2"`) via `make data`.

### 2.3 Enabling Closed-Loop QLoRA / ReST Self-Training on Verified Trajectories
`MultiAgentResearchPipeline` automatically exports verified search-zone trajectories to `logs/sft_verified_trajectories.jsonl`. On a 24 GB GPU, fine-tune a LoRA adapter (`r=16, alpha=32`) with `trl`/`peft` on `sft_verified_trajectories.jsonl` between outer generations and point `model.name` to the merged/LoRA checkpoint in `vLLM` (`--enable-lora`). **Safety rule**: `sft_verified_trajectories.jsonl` contains *only* `search`-zone outcomes, preserving strict validation and sealed-test isolation.
