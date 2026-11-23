"""All-in-One Modular Micro-RSI Training & Sealed Evaluation Script for Google Colab (T4) & Kaggle (T4 x2).

Author: Tran Duy Khang — AI Software Engineer / Freelance Vibe Researcher

Supports:
- Kaggle Notebooks FREE (`T4 x2`: 2x 16GB VRAM, 30h/week): GPU 0 -> Qwen3-4B QLoRA, GPU 1 -> AST Diffusion + HRM
- Google Colab FREE (`1x T4`: 16GB VRAM): GPU 0 -> Qwen3-4B/1.7B QLoRA + AST Diffusion + HRM (sequential VRAM clean)
- Local CPU laptop: automatic `cpu_fast` preset for instant smoke-testing

Usage on Kaggle / Colab:
    !python train_colab_kaggle.py --preset kaggle_t4_optimal --enable-unsloth-qlora --slm-model unsloth/Qwen3-4B-bnb-4bit
Usage on CPU (smoke test):
    python train_colab_kaggle.py --preset cpu_fast --epochs 5
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

# Ensure `src/` is importable when executed directly on Kaggle/Colab without `pip install -e .`
REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vn_rsi.agents import MultiAgentResearchPipeline  # noqa: E402
from vn_rsi.data import DataQASuite, SnapshotManager, build_provider, load_market_rules  # noqa: E402
from vn_rsi.eval import SealedEvaluator  # noqa: E402
from vn_rsi.models import ModularMicroRSIBackend  # noqa: E402
from vn_rsi.rsi import BiLevelRSIRunner  # noqa: E402
from vn_rsi.utils import BudgetConfig, BudgetTracker, load_yaml, set_global_seed  # noqa: E402


def detect_compute_environment() -> dict[str, Any]:
    """Detect Kaggle / Colab / Local environment and exact CUDA VRAM topology."""
    platform = "local"
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle/working").exists():
        platform = "kaggle"
    elif os.environ.get("COLAB_RELEASE_TAG") or Path("/content").exists():
        platform = "colab"

    gpus: list[dict[str, Any]] = []
    has_torch = False
    try:
        import torch

        has_torch = True
        if torch.cuda.is_available():
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                gpus.append(
                    {
                        "index": idx,
                        "name": props.name,
                        "vram_gb": round(props.total_memory / (1024**3), 2),
                    }
                )
    except Exception:
        pass

    if len(gpus) >= 2:
        slm_device, micro_device = "cuda:0", "cuda:1"
    elif len(gpus) == 1:
        slm_device, micro_device = "cuda:0", "cuda:0"
    else:
        slm_device, micro_device = "cpu", "cpu"

    return {
        "platform": platform,
        "has_torch": has_torch,
        "gpu_count": len(gpus),
        "gpus": gpus,
        "slm_device": slm_device,
        "micro_device": micro_device,
    }


def finetune_slm_with_unsloth_qlora(
    sft_jsonl_path: Path,
    model_id: str,
    output_dir: Path,
    max_seq_len: int = 2048,
    epochs: int = 2,
    batch_size: int = 2,
    grad_accum: int = 4,
) -> dict[str, Any]:
    """Fine-tune Model 1 (Qwen3-4B or Qwen3-1.7B) via Unsloth 4-bit QLoRA on verified search-zone trajectories.

    Fits comfortably inside a single FREE T4 16GB GPU (~4.2 GB VRAM for Qwen3-4B, ~2.0 GB for Qwen3-1.7B).
    """
    if not sft_jsonl_path.exists():
        return {"status": "skipped", "reason": f"SFT file not found: {sft_jsonl_path}"}

    rows: list[dict[str, str]] = []
    with sft_jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            prompt = (
                f"Mechanism family: {rec.get('family')}. "
                "Formulate a falsifiable Vietnamese equity hypothesis and strictly causal FactorSpec AST."
            )
            completion = json.dumps(
                {"hypothesis": rec.get("hypothesis"), "factor_spec": rec.get("factor_spec")},
                sort_keys=True,
            )
            rows.append({"text": f"<|user|>\n{prompt}\n<|assistant|>\n{completion}<|end|>"})

    if len(rows) < 2:
        return {"status": "skipped", "reason": f"Need >= 2 verified trajectories, found {len(rows)}"}

    try:
        import torch
        from datasets import Dataset  # type: ignore[import-not-found]
        from trl import SFTConfig, SFTTrainer  # type: ignore[import-not-found]
        from unsloth import FastLanguageModel  # type: ignore[import-not-found]
    except Exception as exc:
        return {
            "status": "unsloth_not_installed",
            "reason": str(exc),
            "install_hint": "pip install -q unsloth trl datasets bitsandbytes accelerate",
            "trajectories_ready": len(rows),
        }

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=max_seq_len,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=2026,
    )

    ds = Dataset.from_list(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        dataset_text_field="text",
        args=SFTConfig(
            output_dir=str(output_dir),
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            num_train_epochs=epochs,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            save_strategy="no",
            report_to="none",
        ),
    )
    train_out = trainer.train()
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Free GPU memory cleanly for single-T4 Colab setups
    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "status": "trained",
        "model_id": model_id,
        "trajectories_used": len(rows),
        "train_loss": round(float(train_out.training_loss), 5),
        "adapter_saved_to": str(output_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Modular Micro-RSI on Google Colab / Kaggle / CPU")
    parser.add_argument("--preset", default="auto", choices=["auto", "cpu_fast", "kaggle_t4_optimal"])
    parser.add_argument("--slm-model", default="unsloth/Qwen3-4B-bnb-4bit")
    parser.add_argument("--enable-unsloth-qlora", action="store_true")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--outer-iters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--checkpoint-dir", default="artifacts/checkpoints")
    args = parser.parse_args()

    set_global_seed(args.seed)
    env = detect_compute_environment()
    preset = args.preset
    if preset == "auto":
        preset = "kaggle_t4_optimal" if env["gpu_count"] >= 1 else "cpu_fast"

    print("=== [VN-RSI] Environment & Hardware Diagnostics ===")
    print(json.dumps({**env, "selected_preset": preset}, indent=2))

    data_cfg = load_yaml(REPO_ROOT / "configs" / "data.yaml")
    sandbox_cfg = load_yaml(REPO_ROOT / "configs" / "sandbox.yaml")
    agent_cfg = load_yaml(REPO_ROOT / "configs" / "agent_v0.yaml")
    llm_cfg = load_yaml(REPO_ROOT / "configs" / "llm.yaml")
    rsi_cfg = load_yaml(REPO_ROOT / "configs" / "rsi.yaml")
    rules = load_market_rules(REPO_ROOT / "configs" / "market_rules.yaml")

    # Stage 1: Ensure immutable leak-free snapshot exists & passes Data QA
    snap_mgr = SnapshotManager(REPO_ROOT / str(data_cfg.get("snapshot_root", "data/snapshots")))
    snap_id = str(data_cfg["snapshot_id"])
    try:
        snapshot = snap_mgr.load_snapshot(snap_id)
    except FileNotFoundError:
        provider = build_provider(str(data_cfg.get("provider_mode", "auto")))
        bundle = provider.fetch_bundle(
            universe_tier=str(data_cfg.get("universe_tier", "vn30")),
            start_date=str(data_cfg["start_date"]),
            end_date=str(data_cfg["end_date"]),
            market_rules=rules,
            seed=int(data_cfg.get("seed", 20261123)),
        )
        snapshot = snap_mgr.write_snapshot(snap_id, bundle)

    qa = DataQASuite(rules).run_all(snapshot)
    if not qa.passed:
        raise SystemExit(f"Data QA FAILED: {qa.checks}")

    # Stage 2: Initialize Sealed Evaluator + Modular Micro-RSI Models (AST Diffusion + HRM)
    sandbox_cfg["ledger_path"] = str(REPO_ROOT / "logs" / "cloud_train_sealed_ledger.jsonl")
    evaluator = SealedEvaluator(snapshot, rules, sandbox_cfg)

    modular = ModularMicroRSIBackend(
        preset=preset,
        diffusion_device=env["micro_device"],
        hrm_device=env["micro_device"],
        seed=args.seed,
    )
    print("\n=== [VN-RSI] Calibrated Micro-Model Breakdown ===")
    print(
        json.dumps(
            {
                "ast_diffusion_runtime": modular.diffusion.backend_runtime,
                "ast_diffusion_params": modular.diffusion.parameter_count(),
                "ast_diffusion_est_train_vram_mb": modular.diffusion.estimate_training_vram_mb(),
                "hrm_reasoner_runtime": modular.hrm.backend_runtime,
                "hrm_reasoner_params": modular.hrm.parameter_count(),
                "hrm_reasoner_est_train_vram_mb": modular.hrm.estimate_training_vram_mb(),
            },
            indent=2,
        )
    )

    # Stage 3: Harvest verified search-zone trajectories & train AST Diffusion + HRM weights (search-only!)
    sft_path = REPO_ROOT / "logs" / "cloud_sft_verified_trajectories.jsonl"
    if sft_path.exists():
        sft_path.unlink()

    warmup_budget = BudgetTracker(BudgetConfig(max_tokens=100_000, max_wall_seconds=300.0, max_gpu_hours=1.0), "cloud_warmup")
    pipe = MultiAgentResearchPipeline(
        evaluator=evaluator,
        llm=modular,
        genome=dict(agent_cfg["genome"]),
        acceptance_cfg=dict(agent_cfg.get("acceptance_criteria", {})),
    )
    warm = pipe.run(
        agent_version="cloud_warmup",
        n_rounds=3,
        n_hypotheses_per_round=4,
        budget=warmup_budget,
        seed=args.seed,
        run_id="cloud_warmup",
        sft_export_path=sft_path,
    )

    steward = pipe._data_steward_search_summary()
    hrm_obs = []
    for rec in warm.candidate_records:
        if rec.audit.passed:
            hrm_obs.append(
                {
                    "regime_vec": modular.hrm.encode_regime_state(steward, warm.family_ic_summary),
                    "ast_tokens": modular.diffusion.vocab.encode_ast(rec.spec.root),
                    "search_rank_ic": rec.search_rank_ic,
                    "family": rec.hypothesis.mechanism_family,
                }
            )
            if rec.accepted_on_search:
                modular.register_verified_search_winner(rec.hypothesis.mechanism_family, rec.spec)

    micro_train_stats = modular.self_improve_weights(
        verified_trajectories=warm.verified_sft_trajectories,
        hrm_observations=hrm_obs,
        epochs=args.epochs,
    )

    ckpt_dir = REPO_ROOT / args.checkpoint_dir
    diff_ckpt = modular.diffusion.save_checkpoint(ckpt_dir / f"ast_diffusion_{preset}.pt")
    hrm_ckpt = modular.hrm.save_checkpoint(ckpt_dir / f"hrm_reasoner_{preset}.pt")

    # Stage 4: Optional Unsloth 4-bit QLoRA Fine-Tuning of Qwen3-4B / 1.7B Narrator on Free T4 GPU
    qlora_report: dict[str, Any] = {"status": "disabled_by_flag"}
    if args.enable_unsloth_qlora:
        qlora_report = finetune_slm_with_unsloth_qlora(
            sft_jsonl_path=sft_path,
            model_id=args.slm_model,
            output_dir=ckpt_dir / "qwen3_narrator_qlora",
            epochs=2,
        )

    # Stage 5: Bi-Level RSI Validation Evaluation of Trained Micro-Models
    rsi_local = dict(rsi_cfg)
    rsi_local["outer_iterations"] = args.outer_iters
    rsi_local["inner_rounds_per_eval"] = 2
    rsi_local["inner_hypotheses_per_round"] = 3
    rsi_local["selection_seeds"] = [101, 202, 303]
    rsi_local["self_training"] = {"export_sft_trajectories": False}

    runner = BiLevelRSIRunner(evaluator, modular, modular, rsi_local, agent_cfg, llm_cfg)
    summary = runner.run_mode("cloud_modular_rsi", seed_override=args.seed)

    report = {
        "status": "COMPLETED",
        "environment": env,
        "preset": preset,
        "micro_weight_training": micro_train_stats,
        "slm_qlora_training": qlora_report,
        "checkpoints": {"ast_diffusion": str(diff_ckpt), "hrm_reasoner": str(hrm_ckpt)},
        "rsi_results": {
            "seed_val_composite": summary.seed_entry.val_composite_mean,
            "best_val_composite": summary.best_entry.val_composite_mean,
            "best_val_sharpe": summary.best_entry.val_sharpe_mean,
            "best_val_rank_ic": summary.best_entry.val_rank_ic_mean,
            "accepted_rewrites": summary.accepted_rewrites,
            "reward_hacking_attempts": summary.reward_hacking_attempts,
        },
    }
    print("\n=== [VN-RSI] Training & Evaluation Summary ===")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
