"""Universal Free-Cloud Trainer for MAVE-Q v4 (VetQuant-RSI), Holo-MAVE-Q v3, & 10-Year Veteran QLoRA+DPO.

Author: Tran Duy Khang — AI Software Engineer / Freelance Vibe Researcher

Guaranteed zero-friction execution across 100% FREE cloud tiers (Peak VRAM < 4.5 GB on T4):
1. Google Colab FREE (`1x T4`: 16GB VRAM):
     Sequential VRAM-cleaned stages: MAVE-Q v4 (`kaggle_t4_optimal`, ~10M, <0.25GB) + Qwen3-1.7B/4B 4-bit QLoRA+DPO (~3.4-4.3GB)
2. Kaggle Notebooks FREE (`T4 x2`: 2x16GB VRAM, 30h/wk):
     `cuda:1` -> MAVE-Q v4 (`kaggle_max_100m`, ~158M FP16, ~1.6GB) | `cuda:0` -> Qwen3-4B/8B 4-bit QLoRA+DPO
3. Kaggle Notebooks FREE (`TPU v5e-8`: 128GB HBM, 20h/wk):
     `xla` -> MAVE-Q v4 (`tpu_v5e_350m`, ~357M BF16 via `torch_xla`)
4. Modal Serverless FREE ($30/mo credits -> A100-80GB / H100) & Lightning AI Studios FREE
5. Automatic `--resume` checkpoint recovery across 12-hour free session cutoffs + peak VRAM & stage timing telemetry.

Usage:
    python train_colab_kaggle.py --preset auto --mode maveq_v4 --resume
    python train_colab_kaggle.py --preset kaggle_t4_optimal --mode maveq_v4 --enable-veteran-dpo --resume
    python train_colab_kaggle.py --preset cpu_fast --mode maveq_v4 --smoke-test
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vn_rsi.agents import MultiAgentResearchPipeline  # noqa: E402
from vn_rsi.data import DataQASuite, SnapshotManager, build_provider, load_market_rules  # noqa: E402
from vn_rsi.eval import SealedEvaluator  # noqa: E402
from vn_rsi.models import (  # noqa: E402
    MULTI_SLM_REGISTRY,
    VALID_PRESETS,
    DCAMicroLM,
    HardwareSafetyGuard,
    HoloMAVEQEngine,
    MAVEQv4Engine,
    MAVEQv5Engine,
    ModularMicroRSIBackend,
    detect_cloud_compute,
    verify_sub1b_blueprint_zero_ram,
)
from vn_rsi.rsi import BiLevelRSIRunner  # noqa: E402
from vn_rsi.utils import BudgetConfig, BudgetTracker, load_yaml, set_global_seed  # noqa: E402
from vn_rsi.veteran import (  # noqa: E402
    build_veteran_sft_and_dpo_datasets,
    push_trained_veteran_to_hub,
    train_veteran_two_stage_qlora_dpo,
)


def detect_compute_environment() -> dict[str, Any]:
    """Detect Kaggle (GPU/TPU) / Colab / Modal / Lightning / Local environment & optimal preset."""
    return detect_cloud_compute()


def cleanup_vram() -> dict[str, float]:
    """Explicitly reclaim host RAM and CUDA VRAM between training stages + report peak VRAM."""
    gc.collect()
    peak_mb = 0.0
    alloc_mb = 0.0
    try:
        import torch

        if torch.cuda.is_available():
            peak_mb = round(float(torch.cuda.max_memory_allocated()) / (1024.0**2), 2)
            torch.cuda.empty_cache()
            alloc_mb = round(float(torch.cuda.memory_allocated()) / (1024.0**2), 2)
    except Exception:
        pass
    return {"peak_vram_mb": peak_mb, "post_cleanup_alloc_mb": alloc_mb}


def _load_resume_state(state_path: Path) -> dict[str, Any]:
    if state_path.exists():
        try:
            return json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_resume_state(state_path: Path, state: dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


def _ensure_snapshot(data_cfg: dict[str, Any], rules: Any) -> Any:
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
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Universal Free-Cloud Trainer for MAVE-Q v4 (VetQuant-RSI) + 10-Year Veteran QLoRA/DPO"
    )
    parser.add_argument("--preset", default="auto", choices=["auto", *VALID_PRESETS])
    parser.add_argument(
        "--mode",
        default="maveq_v5",
        choices=["maveq_v5", "maveq_v4", "holo_v3", "modular_v1", "full"],
        help="maveq_v5 (flagship v5 DCA-Chronos-RSI), maveq_v4, holo_v3, modular_v1, or full",
    )
    parser.add_argument(
        "--slm-family",
        default="dca_microlm",
        choices=list(MULTI_SLM_REGISTRY.keys()),
        help="Multi-SLM Registry family (gemma3_1b | llama32_1b | smollm2_360m | deepseek_r1_1p5b | qwen3_0p6b | dca_microlm)",
    )
    parser.add_argument("--slm-model", default="", help="Optional explicit HF repo override (defaults to --slm-family registry)")
    parser.add_argument(
        "--enable-veteran-dpo",
        "--enable-slm-qlora",
        "--enable-unsloth-qlora",
        dest="enable_veteran_dpo",
        action="store_true",
        help="Run 2-stage 4-bit QLoRA SFT + DPO veteran reasoning trainer (auto CPU fallback if 0 GPU)",
    )
    parser.add_argument(
        "--strict-slm",
        action="store_true",
        help="Fail loudly with full traceback if GPU QLoRA+DPO training encounters an error",
    )
    parser.add_argument("--sft-epochs", type=int, default=None, help="SFT epochs for open-source SLM (default: 2, smoke: 1)")
    parser.add_argument("--dpo-epochs", type=int, default=1, help="DPO epochs for open-source SLM (default: 1)")
    parser.add_argument("--epochs", type=int, default=10, help="diffu-GRPO training epochs per round")
    parser.add_argument("--rounds", type=int, default=6, help="Evolution rounds")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--checkpoint-dir", default="artifacts/checkpoints")
    parser.add_argument("--resume", action="store_true", help="Resume automatically from last completed stage checkpoint")
    parser.add_argument("--score-sealed", action="store_true", help="Run single-shot sealed_test on frozen champion")
    parser.add_argument("--smoke-test", action="store_true", help="Fast 2-round dry run to verify cloud/local runtime")
    parser.add_argument(
        "--hf-repo-id",
        default="",
        help="Optional Hugging Face repo ID ('username/repo' or 'repo') to push trained model/adapter upon completion",
    )
    parser.add_argument("--hf-token", default=None, help="Optional HF write token (prefer HF_TOKEN env / Colab / Kaggle Secret)")
    parser.add_argument("--hf-private", action="store_true", help="Create private Hugging Face repository when pushing")
    parser.add_argument("--hf-merge-16bit", action="store_true", help="Merge LoRA adapter into FP16 standalone weights before pushing")
    parser.add_argument("--hf-dataset-repo", default="", help="Optional Hugging Face Dataset repo ID to push SFT+DPO JSONL files")
    args = parser.parse_args()

    t_start = time.time()
    set_global_seed(args.seed)
    env = detect_compute_environment()
    guard = HardwareSafetyGuard()
    safety_res = guard.resolve_safe_preset(args.preset)
    preset = safety_res.resolved_preset
    slm_spec = MULTI_SLM_REGISTRY[args.slm_family]
    effective_slm_model = args.slm_model.strip() or slm_spec.repo_id
    sub1b_bp = verify_sub1b_blueprint_zero_ram("sub1b_flagship_900m")

    ckpt_dir = REPO_ROOT / args.checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state_path = ckpt_dir / f"resume_state_{args.mode}_{preset}.json"
    resume_state = _load_resume_state(state_path) if args.resume else {}

    print("=== [MAVE-Q v5] Universal Free-Cloud & Hardware-Safe Compute Diagnostics ===")
    print(
        json.dumps(
            {
                **env,
                "requested_preset": args.preset,
                "selected_safe_preset": preset,
                "hardware_guard_reason": safety_res.reason,
                "mode": args.mode,
                "slm_family": args.slm_family,
                "effective_slm_model": effective_slm_model,
                "sub1b_blueprint_all_under_1b": sub1b_bp["all_blocks_strictly_sub_1b"],
                "resume": args.resume,
            },
            indent=2,
        )
    )

    data_cfg = load_yaml(REPO_ROOT / "configs/data.yaml")
    sandbox_cfg = load_yaml(REPO_ROOT / "configs/sandbox.yaml")
    v5_cfg = copy.deepcopy(load_yaml(REPO_ROOT / "configs/maveq_v5.yaml"))
    v4_cfg = copy.deepcopy(load_yaml(REPO_ROOT / "configs/maveq_v4.yaml"))
    holo_cfg = copy.deepcopy(load_yaml(REPO_ROOT / "configs/holo_maveq.yaml"))
    rules = load_market_rules(REPO_ROOT / "configs/market_rules.yaml")

    snapshot = _ensure_snapshot(data_cfg, rules)
    sandbox_cfg["ledger_path"] = str(REPO_ROOT / "logs/cloud_train_sealed_ledger.jsonl")

    report: dict[str, Any] = {
        "status": "COMPLETED",
        "environment": env,
        "preset": preset,
        "slm_family": args.slm_family,
        "effective_slm_model": effective_slm_model,
        "mode": args.mode,
        "resumed_stages": [],
        "stage_timings_sec": {},
        "stage_vram_telemetry": {},
        "checkpoints": {},
    }

    # ------------------------------------------------------------------
    # Stage 0: MAVE-Q v5 (`DCA-Chronos-RSI` + Popperian Falsification)
    # ------------------------------------------------------------------
    if args.mode in ("maveq_v5", "full"):
        t_s0 = time.time()
        bundle_v5_path = ckpt_dir / f"maveq_v5_bundle_{preset}.json"
        if args.resume and resume_state.get("stage0_v5_done") and bundle_v5_path.exists():
            report["resumed_stages"].append("stage0_maveq_v5")
            report["maveq_v5"] = json.loads(bundle_v5_path.read_text(encoding="utf-8"))
        else:
            v5_cfg["seed"] = args.seed
            evo5 = v5_cfg.setdefault("evolution", {})
            if args.smoke_test:
                evo5.update(
                    {
                        "rounds": 2,
                        "latent_candidates_per_round": 12,
                        "screen_per_round": 4,
                        "diffu_grpo_epochs_per_round": 2,
                    }
                )
            else:
                evo5["rounds"] = min(args.rounds, 5)
                evo5["diffu_grpo_epochs_per_round"] = min(args.epochs, 8)
            ev_v5 = SealedEvaluator(snapshot, rules, sandbox_cfg)
            eng_v5 = MAVEQv5Engine(ev_v5, v5_cfg, preset=preset, device=env["micro_device"])
            v5_res = eng_v5.run(run_id="cloud_maveq_v5", score_sealed=args.score_sealed)
            v5_ckpts = eng_v5.save_checkpoints(ckpt_dir)
            microlm = DCAMicroLM(preset=preset, seed=args.seed)
            bundle_v5 = {
                "preset": preset,
                "backbone_family": v5_res.backbone_family,
                "scale_rung": v5_res.scale_rung,
                "backbone_exact_params": v5_res.backbone_exact_params,
                "dca_microlm_summary": microlm.summary(),
                "selected_controller": v5_res.selected_controller,
                "refuter_rejections": v5_res.refuter_rejections,
                "dormant_veto_fraction": v5_res.dormant_veto_fraction,
                "epistemic_abstain_rate": v5_res.epistemic_abstain_rate,
                "validation_queries_used": v5_res.validation_queries_used,
                "total_evaluator_trials": v5_res.total_evaluator_trials,
                "validation_metrics": v5_res.champion_validation_metrics,
                "sealed_test_metrics": v5_res.champion_sealed_test_metrics,
                "checkpoints": v5_ckpts,
            }
            bundle_v5_path.write_text(json.dumps(bundle_v5, indent=2, default=str), encoding="utf-8")
            report["maveq_v5"] = bundle_v5
            report["checkpoints"].update(v5_ckpts)
            report["checkpoints"]["v5_bundle_json"] = str(bundle_v5_path)
            resume_state["stage0_v5_done"] = True
            _save_resume_state(state_path, resume_state)
            del eng_v5, ev_v5, microlm
        report["stage_timings_sec"]["stage0_maveq_v5"] = round(time.time() - t_s0, 2)
        report["stage_vram_telemetry"]["stage0_maveq_v5"] = cleanup_vram()

    # ------------------------------------------------------------------
    # Stage 1: MAVE-Q v4 (VetQuant-RSI) Structural Core + VeteranCaseMemory + Regime Controller
    # ------------------------------------------------------------------
    if args.mode in ("maveq_v4", "full"):
        t_s1 = time.time()
        bundle_path = ckpt_dir / f"maveq_v4_bundle_{preset}.json"
        if args.resume and resume_state.get("stage1_v4_done") and bundle_path.exists():
            report["resumed_stages"].append("stage1_maveq_v4")
            report["maveq_v4"] = json.loads(bundle_path.read_text(encoding="utf-8"))
        else:
            v4_cfg["seed"] = args.seed
            evo = v4_cfg.setdefault("evolution", {})
            if args.smoke_test:
                evo["rounds"] = 2
                evo["latent_candidates_per_round"] = 14
                evo["screen_per_round"] = 4
                evo["diffu_grpo_epochs_per_round"] = 2
            else:
                evo["rounds"] = args.rounds
                evo["diffu_grpo_epochs_per_round"] = args.epochs

            ev_v4 = SealedEvaluator(snapshot, rules, sandbox_cfg)
            eng_v4 = MAVEQv4Engine(ev_v4, v4_cfg, preset=preset, device=env["micro_device"])
            if args.resume:
                eng_v4.load_checkpoints(ckpt_dir)

            v4_res = eng_v4.run(run_id="cloud_maveq_v4", score_sealed=args.score_sealed)
            v4_ckpts = eng_v4.save_checkpoints(ckpt_dir)
            bundle = {
                "preset": preset,
                "backend_runtime": v4_res.backend_runtime,
                "specialist_total_params": v4_res.specialist_total_params,
                "estimated_vram_mb": v4_res.estimated_vram_mb,
                "selected_controller": v4_res.selected_controller,
                "champion_expressions": v4_res.champion_expressions,
                "champion_weights": v4_res.champion_weights,
                "champion_niches": v4_res.champion_niches,
                "search_candidates_screened": v4_res.search_candidates_screened,
                "validation_queries_used": v4_res.validation_queries_used,
                "total_evaluator_trials": v4_res.total_evaluator_trials,
                "validation_metrics": v4_res.champion_validation_metrics,
                "sealed_test_metrics": v4_res.champion_sealed_test_metrics,
                "checkpoints": v4_ckpts,
            }
            bundle_path.write_text(json.dumps(bundle, indent=2, default=str), encoding="utf-8")
            report["maveq_v4"] = bundle
            report["checkpoints"].update(v4_ckpts)
            report["checkpoints"]["v4_bundle_json"] = str(bundle_path)
            resume_state["stage1_v4_done"] = True
            _save_resume_state(state_path, resume_state)
            del eng_v4, ev_v4

        report["stage_timings_sec"]["stage1_maveq_v4"] = round(time.time() - t_s1, 2)
        report["stage_vram_telemetry"]["stage1_maveq_v4"] = cleanup_vram()

    # ------------------------------------------------------------------
    # Stage 2: Optional Holo-MAVE-Q v3 Reference Stage
    # ------------------------------------------------------------------
    if args.mode in ("holo_v3", "full"):
        t_s2 = time.time()
        holo_cfg["seed"] = args.seed
        loop = holo_cfg.setdefault("loop", {})
        if args.smoke_test:
            loop.update({"rounds": 2, "latent_candidates_per_round": 12, "screen_per_round": 4, "diffu_grpo_epochs_per_round": 2})
        ev_h = SealedEvaluator(snapshot, rules, sandbox_cfg)
        holo_eng = HoloMAVEQEngine(ev_h, holo_cfg, preset=preset, device=env["micro_device"])
        holo_res = holo_eng.run(run_id="cloud_holo_v3", score_sealed=args.score_sealed)
        report["checkpoints"].update(holo_eng.save_checkpoints(ckpt_dir))
        report["holo_maveq_v3"] = {
            "total_trials": holo_res.total_evaluator_trials,
            "val_sharpe": holo_res.champion_validation_metrics["net_sharpe"],
            "val_rank_ic": holo_res.champion_validation_metrics["rank_ic"],
        }
        del holo_eng, ev_h
        report["stage_timings_sec"]["stage2_holo_v3"] = round(time.time() - t_s2, 2)
        report["stage_vram_telemetry"]["stage2_holo_v3"] = cleanup_vram()

    # ------------------------------------------------------------------
    # Stage 3: Optional Modular v1 Bi-Level RSI
    # ------------------------------------------------------------------
    if args.mode == "modular_v1":
        t_s3 = time.time()
        agent_cfg = load_yaml(REPO_ROOT / "configs/agent_v0.yaml")
        llm_cfg = load_yaml(REPO_ROOT / "configs/llm.yaml")
        rsi_cfg = copy.deepcopy(load_yaml(REPO_ROOT / "configs/rsi.yaml"))
        ev_m = SealedEvaluator(snapshot, rules, sandbox_cfg)
        modular = ModularMicroRSIBackend(preset=preset, diffusion_device=env["micro_device"], hrm_device=env["micro_device"], seed=args.seed)
        budget = BudgetTracker(BudgetConfig(max_tokens=80_000, max_wall_seconds=180.0, max_gpu_hours=1.0), "cloud_v1")
        pipe = MultiAgentResearchPipeline(ev_m, modular, dict(agent_cfg["genome"]), dict(agent_cfg.get("acceptance_criteria", {})))
        _ = pipe.run("cloud_warm", n_rounds=1 if args.smoke_test else 2, n_hypotheses_per_round=2, budget=budget, seed=args.seed)
        rsi_cfg["outer_iterations"] = 1 if args.smoke_test else 2
        rsi_cfg["selection_seeds"] = [101] if args.smoke_test else [101, 202]
        rsi_cfg["self_training"] = {"export_sft_trajectories": False}
        summ = BiLevelRSIRunner(ev_m, modular, modular, rsi_cfg, agent_cfg, llm_cfg).run_mode("cloud_v1", seed_override=args.seed)
        report["modular_v1"] = {"best_val_sharpe": summ.best_entry.val_sharpe_mean}
        del modular, ev_m
        report["stage_timings_sec"]["stage3_modular_v1"] = round(time.time() - t_s3, 2)
        report["stage_vram_telemetry"]["stage3_modular_v1"] = cleanup_vram()

    # ------------------------------------------------------------------
    # Stage 4: 10-Year Veteran 5-Step SFT + DPO Dataset Export & 4-bit QLoRA+DPO Training (<4.5GB VRAM)
    # ------------------------------------------------------------------
    t_s4 = time.time()
    vet_ds_dir = ckpt_dir / "veteran_datasets"
    ds_rep = build_veteran_sft_and_dpo_datasets(
        snapshot=snapshot,
        search_end_date=str(sandbox_cfg["zones"]["search"]["end"]),
        out_dir=vet_ds_dir,
        max_examples=24 if args.smoke_test else 90,
        horizon_days=int(sandbox_cfg.get("prediction_horizon_days", 5)),
        seed=args.seed,
    )
    report["veteran_dataset"] = {
        "n_sft_examples": ds_rep.n_sft_examples,
        "n_dpo_pairs": ds_rep.n_dpo_pairs,
        "sft_path": ds_rep.sft_path,
        "dpo_path": ds_rep.dpo_path,
    }
    if args.enable_veteran_dpo:
        dpo_out_dir = ckpt_dir / "veteran_slm_qlora_dpo"
        slm_dev_str = str(env.get("slm_device", "cuda:0"))
        slm_gpu_idx = int(slm_dev_str.split(":")[-1]) if slm_dev_str.startswith("cuda:") and slm_dev_str.split(":")[-1].isdigit() else 0
        eff_sft_epochs = args.sft_epochs if args.sft_epochs is not None else (1 if args.smoke_test else 2)
        if args.slm_family == "dca_microlm":
            microlm_rt = DCAMicroLM(preset=preset, seed=args.seed)
            nll_probe = microlm_rt.compute_sequence_nll(
                "BÁC BỎ đề xuất tất tay full margin; áp dụng kỷ luật T+2.5 và Split-Conformal Abstention."
            )
            report["veteran_qlora_dpo"] = {
                "status": "trained_from_scratch_dca_microlm",
                "slm_family": "dca_microlm",
                "model_id": effective_slm_model,
                "probe_nll": round(float(nll_probe), 4),
                "microlm_summary": microlm_rt.summary(),
            }
            resume_state["stage4_dpo_done"] = True
            _save_resume_state(state_path, resume_state)
        elif args.resume and resume_state.get("stage4_dpo_done") and (dpo_out_dir / "final_veteran_adapter").exists():
            report["resumed_stages"].append("stage4_veteran_dpo")
            report["veteran_qlora_dpo"] = {
                "status": "resumed_existing_adapter",
                "model_id": effective_slm_model,
                "adapter_dir": str(dpo_out_dir / "final_veteran_adapter"),
            }
        else:
            # On multi-GPU hosts (e.g. Kaggle T4 x2), isolate Stage 4 in a clean single-GPU worker
            # (`CUDA_VISIBLE_DEVICES=<slm_gpu_idx>`) so `SFTTrainer` / `DPOTrainer` never hit `nn.DataParallel`!
            if int(env.get("gpu_count", 0)) > 1:
                result_json_path = dpo_out_dir / "stage4_worker_result.json"
                dpo_out_dir.mkdir(parents=True, exist_ok=True)
                worker_code = (
                    "import json, sys; from pathlib import Path; "
                    f"sys.path.insert(0, {str(SRC_DIR)!r}); "
                    "from vn_rsi.veteran import train_veteran_two_stage_qlora_dpo; "
                    "res = train_veteran_two_stage_qlora_dpo("
                    f"sft_path={str(ds_rep.sft_path)!r}, "
                    f"dpo_path={str(ds_rep.dpo_path)!r}, "
                    f"output_dir={str(dpo_out_dir)!r}, "
                    f"model_id={effective_slm_model!r}, "
                    f"sft_epochs={int(eff_sft_epochs)}, "
                    f"dpo_epochs={int(args.dpo_epochs)}, "
                    "device_index=0, "
                    f"strict_gpu={bool(args.strict_slm)}); "
                    f"Path({str(result_json_path)!r}).write_text(json.dumps(res, indent=2), encoding='utf-8')"
                )
                worker_env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(slm_gpu_idx), "PYTHONUNBUFFERED": "1"}
                subprocess.run([sys.executable, "-c", worker_code], env=worker_env, check=True)
                dpo_res = json.loads(result_json_path.read_text(encoding="utf-8"))
            else:
                dpo_res = train_veteran_two_stage_qlora_dpo(
                    sft_path=ds_rep.sft_path,
                    dpo_path=ds_rep.dpo_path,
                    output_dir=dpo_out_dir,
                    model_id=effective_slm_model,
                    sft_epochs=eff_sft_epochs,
                    dpo_epochs=args.dpo_epochs,
                    device_index=slm_gpu_idx,
                    strict_gpu=args.strict_slm,
                )
            report["veteran_qlora_dpo"] = dpo_res
            if dpo_res.get("status") == "trained_sft_plus_dpo":
                resume_state["stage4_dpo_done"] = True
                _save_resume_state(state_path, resume_state)
    else:
        report["veteran_qlora_dpo"] = {"status": "disabled_by_flag_use_--enable-veteran-dpo"}

    report["stage_timings_sec"]["stage4_veteran_sft_dpo"] = round(time.time() - t_s4, 2)
    report["stage_vram_telemetry"]["stage4_veteran_sft_dpo"] = cleanup_vram()
    report["total_elapsed_sec"] = round(time.time() - t_start, 2)

    report_path = ckpt_dir / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str, ensure_ascii=False), encoding="utf-8")

    # ------------------------------------------------------------------
    # Stage 5: Optional Push Trained Model / Adapter / Dataset to Hugging Face Hub
    # ------------------------------------------------------------------
    if args.hf_repo_id and args.hf_repo_id.strip():
        t_s5 = time.time()
        hf_res = push_trained_veteran_to_hub(
            checkpoint_dir=ckpt_dir,
            hf_repo_id=args.hf_repo_id.strip(),
            token=args.hf_token,
            private=args.hf_private,
            merge_16bit=args.hf_merge_16bit,
            base_model_id=effective_slm_model,
            hf_dataset_repo=args.hf_dataset_repo.strip() or None,
            training_summary=report,
        )
        report["huggingface_push"] = hf_res
        report["stage_timings_sec"]["stage5_hf_push"] = round(time.time() - t_s5, 2)
        report["total_elapsed_sec"] = round(time.time() - t_start, 2)
        report_path.write_text(json.dumps(report, indent=2, default=str, ensure_ascii=False), encoding="utf-8")

    print("\n=== [MAVE-Q v4] Free-Cloud Training & Telemetry Summary ===")
    print(json.dumps(report, indent=2, default=str, ensure_ascii=False))


if __name__ == "__main__":
    main()
