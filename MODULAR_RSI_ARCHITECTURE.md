# MAVE-Q v5 (`DCA-Chronos-RSI` + Popperian Falsification) — Modular Architecture & Free-Cloud Scaling Guide

**Research Topic / Đề tài nghiên cứu**: *MAVE-Q v5 (DCA-Chronos-RSI): Popperian Falsification, Google DeepCrossAttention QuantFormer, and Asymmetric Beta No-Trade-Zone Control under Emerging-Market Microstructure Constraints* *(Ongoing Research Topic — Not a Submitted/Published Work)*  
**Author**: **Tran Duy Khang** — *AI Software Engineer / Independent Quantitative AI Researcher*  
**Deep Evolution & Math Proofs (`v0→v5`)**: [`ARCHITECTURE_EVOLUTION_AND_PROOFS.md`](ARCHITECTURE_EVOLUTION_AND_PROOFS.md) | **Cloud Blueprints (`<1B/block`)**: [`FREE_CLOUD_AND_SCALING_PROPOSALS.md`](FREE_CLOUD_AND_SCALING_PROPOSALS.md) | **Training Guide**: [`FREE_CLOUD_TRAINING_GUIDE.md`](FREE_CLOUD_TRAINING_GUIDE.md) | **Self-Audit**: [`paper/PEER_REVIEW_AUDIT.md`](paper/PEER_REVIEW_AUDIT.md)

---

## 1. Architectural Evolution: From Monolithic 8B (`v0`) to Pruned Lean `MAVE-Q v5`

| Generation | Core Active Mechanism | Official Val Queries | Val Sharpe | Sealed Test Sharpe | Sealed Calmar | Sealed MaxDD | Sealed DSR (Indiv / Lineage) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Agent `v0` | Monolithic $\le$8B constrained pipeline | 1 | 0.677 | 1.339 | 1.98 | -25.1% | 0.365 / -- |
| MAVE-Q `v1` | Discrete AST Diffusion + HRM (`diffu-GRPO`) | 40+ | 1.150 | 1.637 | 2.38 | -20.5% | 0.332 / -- |
| MAVE-Q `v2` | Friction-Aware QD + Greedy PLCA | 299 | 1.286 | 1.602 | 2.12 | -24.1% | 0.180 / -- |
| Holo-MAVE-Q `v3` | `TypedASTGuard` + `Tree-MDLM` + `Holo-HRM` + `MSSF` + `BTO-SRSS` | 81 | 1.291 | 1.676 | 2.21 | -22.0% | 0.349 / -- |
| MAVE-Q `v4` (`VetQuant-RSI`) | `VeteranCaseMemory` + `GP-FA-Ridge` + `BAB` + Zero-Val-Burn CV | 1 (2 total) | 1.347 ± 0.081 (8-seed) | 1.067 | 1.06 | -23.1%* | 0.819 / -- |
| **MAVE-Q `v5` (Pruned Lean)** | **`DCA-QuantFormer` (`G-DCA`, `0.48M`) + `AsymBeta-NTZ` + Popperian Pruning (17 falsified)** | **1** (`N=1`) | **1.473 ± 0.083** (8-seed) | **1.441** (`+0.374` vs v4) | **1.70** (`+60.6%`) | **-20.9%** | **0.895 / 0.389** (29-entry lineage) |

*\*Note: v4's `phase13_v5_final_arena` uniform comparison window (`2024-07-15` to `2025-12-31`, `v5_global_arena.tex`) records v4 Test `SR=1.067, Calmar=1.06, MaxDD=-23.1%` vs v5 `SR=1.441, Calmar=1.70, MaxDD=-20.9%` (single-shot ledger entry #29).*

---

## 2. Popperian Pruned MAVE-Q v5 Pipeline (`src/vn_rsi/models/`)

Unlike ad-hoc ensembles that keep every proposed block, **MAVE-Q v5** enforces a **Popperian Falsification Constitution** (`logs/v5_falsified_hypotheses.jsonl` — **17 falsified/pruned hypotheses**). Three complex v5 hypotheses (`ChronosKalmanHedge`, `DCAMoSEHyperGate`, and stock-level `SplitConformalDormantVeto`) degraded 8-seed validation Sharpe and were **permanently pruned** (`false` in `configs/maveq_v5.yaml`), leaving the verified Occam-optimal lean core:

```text
Strictly Causal Feature Panel (<= t, PIT Announcement Lag Enforced, atol=1e-9 Shift-Invariant)
   ├─► Module 1 (ACTIVE — KEEP_VERIFIED): DCA-QuantFormer (`src/vn_rsi/models/dca_backbone.py`, 0.48M Occam-optimal)
   │      ├─► Shallow Orthogonal Ridge Anchor (7 clean causal microstructure views, raw-target calibrated)
   │      ├─► Google DeepCrossAttention (G-DCA, ICML 2025 arXiv:2502.06785) over 16 causal channels
   │      └─► 5-Family Backbone Zoo Benchmarked: G-DCA (#1) vs Pre-LN, Mamba-SSM, Hyena-Conv, MoE-Trans
   ├─► Module 2 (ACTIVE — KEEP_VERIFIED): AsymBeta-NTZ Controller (`src/vn_rsi/models/maveq_v5_chronos.py`)
   │      ├─► Asymmetric Bull/Bear Beta-Budget Release (captures rallies without sacrificing tail defense)
   │      ├─► Exchange-Band Epistemic Abstention (`band_tail_scale` adapts VN ±7% vs US/CSI wide bands)
   │      └─► L1 No-Trade-Zone (NTZ) Incumbent Hysteresis (cuts T+2.5 churn: 7.76% daily turnover on Test)
   ├─► Module 3 (PRUNED BY POPPERIAN AUDIT — kept toggleable for ablation reproducibility only):
   │      ✗ ChronosKalmanHedge (removed: +0.0740 Val SR gain on 8 seeds)
   │      ✗ DCAMoSEHyperGate   (removed: +0.0584 Val SR gain on 8 seeds)
   │      ✗ SplitConformalDormantVeto stock penalty (removed: +0.0416 Val SR gain, -61% SD)
   └─► Module 4 (ACTIVE — Anti-Sycophancy Veteran CIO, `src/vn_rsi/veteran/`):
          ├─► 6 Quantitative Trap Detectors (FOMO Margin, Rumor/PIT, Look-Ahead, 1.5x Fee Mirage, Falling Knife, Overfit)
          ├─► 45-Scenario / 22-Trap Benchmark: 100% Sycophancy Resistance, 0.0% FDR, 100% Abstention Precision
          └─► 100 Strictly Search-Zone SFT + Anti-Sycophancy DPO Pairs + 6-Family Multi-SLM Stack
```

---

## 3. Hardware-Safe Local & `<1B/Block` Free-Cloud Preset Matrix (`src/vn_rsi/models/cloud_presets.py`)

`HardwareSafetyGuard` inspects host RAM and CUDA availability automatically: on laptops with `<16 GB` RAM and `0` GPUs (e.g., `AMD Ryzen 5 5500U`, `7.35 GB` RAM), it auto-locks to `local_safe_20m` (`<1.2 GB` RAM ceiling; measured peak **`284 MB`** RAM) and refuses OOM launches.

| Preset | Target Platform | `DCA-QuantFormer` | `DCA-MicroLM` | `Tree-MDLM` | `Holo-HRM` | Max Block Constraint | Telemetry / Budget |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `local_safe_20m` | **Local Laptop / CI (0 GPU, <16GB RAM)** | `0.48M`–`20M` | `18.4M` | `96.5K` | `85K` | Strictly `<= 20M` | **Measured `284 MB` RAM** (`<1.2 GB` cap) |
| `kaggle_t4_optimal` | **Google Colab Free (`1x T4 16GB`)** | `12M` | `125M` | `10M` | `8M` | `< 1B/block` | Analytical `< 4.5 GB` VRAM (FP16 + 4-bit QLoRA) |
| `kaggle_max_100m` | **Kaggle Free (`2x T4 32GB`, 30h/wk)** | `120M` | `350M` | `158M` | `95M` | `< 1B/block` | Analytical `cuda:0` + `cuda:1` split (`< 8 GB/GPU`) |
| `tpu_v5e_350m` | **Kaggle Free (`TPU v5e-8 128GB HBM`)** | `380M` | `500M` | `357M` | `260M` | `< 1B/block` | Analytical Native BF16 (`torch_xla`) |
| `sub1b_flagship_900m` | **Kaggle `TPU v5e-8` / `T4x2` / Modal A100** | **`880M`** | **`828M`** | **`650M`** | **`520M`** | **Strictly `< 1B/block`** (0 MB meta-verified) | Analytical BF16 / 4-bit QLoRA (`FREE_CLOUD_AND_SCALING_PROPOSALS.md`) |

**6-Family Multi-SLM Registry (`MULTI_SLM_REGISTRY`)**: `dca_microlm` (from-scratch Google `G-DCA` `18.4M / 828M`), `qwen3` (`Qwen3-1.7B / 4B`), `gemma3` (`Gemma-3-1B / 4B`), `smollm2` (`SmolLM2-360M / 1.7B`), `llama32` (`Llama-3.2-1B / 3B`), and `phi4_mini` (`Phi-4-mini-3.8B`).

---

## 4. Quickstart Commands
```bash
# 1. Hardware-safe local smoke test (auto-locks to local_safe_20m, <1.2 GB RAM)
python train_colab_kaggle.py --preset auto --mode maveq_v5 --slm-family dca_microlm --enable-veteran-dpo --smoke-test

# 2. Export 100 search-zone Anti-Sycophancy SFT/DPO pairs & cloud training bundle
python scripts/export_cloud_training_pack.py

# 3. Query the Anti-Sycophancy 10-Year Veteran CIO Agent from CLI (Vietnamese or English)
python scripts/ask_veteran_agent.py --symbol FPT --lang vi --question "Nghe tin đồn nội bộ FPT sắp ký hợp đồng lớn, có nên full margin đuổi trần ngay không?"
```
