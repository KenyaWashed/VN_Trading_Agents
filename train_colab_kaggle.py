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
    VALID_PRESETS,
    HoloMAVEQEngine,
    MAVEQv4Engine,
    ModularMicroRSIBackend,
    detect_cloud_compute,
)
from vn_rsi.rsi import BiLevelRSIRunner  # noqa: E402
from vn_rsi.utils import BudgetConfig, BudgetTracker, load_yaml, set_global_seed  # noqa: E402
from vn_rsi.veteran import (  # noqa: E402
    build_veteran_sft_and_dpo_datasets,
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
        default="maveq_v4",
        choices=["maveq_v4", "holo_v3", "modular_v1", "full"],
        help="maveq_v4 (flagship v4 VetQuant-RSI), holo_v3, modular_v1, or full (v4 + v3 + veteran DPO)",
    )
    parser.add_argument("--slm-model", default="Qwen/Qwen3-1.7B", help="Compact SLM for <4.5GB VRAM 4-bit QLoRA+DPO")
    parser.add_argument(
        "--enable-veteran-dpo",
        "--enable-slm-qlora",
        "--enable-unsloth-qlora",
        dest="enable_veteran_dpo",
        action="store_true",
        help="Run 2-stage 4-bit QLoRA SFT + DPO veteran reasoning trainer (auto CPU fallback if 0 GPU)",
    )
    parser.add_argument("--epochs", type=int, default=10, help="diffu-GRPO training epochs per round")
    parser.add_argument("--rounds", type=int, default=6, help="Evolution rounds")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--checkpoint-dir", default="artifacts/checkpoints")
    parser.add_argument("--resume", action="store_true", help="Resume automatically from last completed stage checkpoint")
    parser.add_argument("--score-sealed", action="store_true", help="Run single-shot sealed_test on frozen champion")
    parser.add_argument("--smoke-test", action="store_true", help="Fast 2-round dry run to verify cloud/local runtime")
    args = parser.parse_args()

    t_start = time.time()
    set_global_seed(args.seed)
    env = detect_compute_environment()
    preset = env["recommended_preset"] if args.preset == "auto" else args.preset

    ckpt_dir = REPO_ROOT / args.checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state_path = ckpt_dir / f"resume_state_{args.mode}_{preset}.json"
    resume_state = _load_resume_state(state_path) if args.resume else {}

    print("=== [MAVE-Q v4] Universal Free-Cloud Compute Diagnostics ===")
    print(json.dumps({**env, "selected_preset": preset, "mode": args.mode, "resume": args.resume}, indent=2))

    data_cfg = load_yaml(REPO_ROOT / "configs/data.yaml")
    sandbox_cfg = load_yaml(REPO_ROOT / "configs/sandbox.yaml")
    v4_cfg = copy.deepcopy(load_yaml(REPO_ROOT / "configs/maveq_v4.yaml"))
    holo_cfg = copy.deepcopy(load_yaml(REPO_ROOT / "configs/holo_maveq.yaml"))
    rules = load_market_rules(REPO_ROOT / "configs/market_rules.yaml")

    snapshot = _ensure_snapshot(data_cfg, rules)
    sandbox_cfg["ledger_path"] = str(REPO_ROOT / "logs/cloud_train_sealed_ledger.jsonl")

    report: dict[str, Any] = {
        "status": "COMPLETED",
        "environment": env,
        "preset": preset,
        "mode": args.mode,
        "resumed_stages": [],
        "stage_timings_sec": {},
        "stage_vram_telemetry": {},
        "checkpoints": {},
    }

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
        if args.resume and resume_state.get("stage4_dpo_done") and (dpo_out_dir / "final_veteran_adapter").exists():
            report["resumed_stages"].append("stage4_veteran_dpo")
            report["veteran_qlora_dpo"] = {"status": "resumed_existing_adapter", "adapter_dir": str(dpo_out_dir / "final_veteran_adapter")}
        else:
            dpo_res = train_veteran_two_stage_qlora_dpo(
                sft_path=ds_rep.sft_path,
                dpo_path=ds_rep.dpo_path,
                output_dir=dpo_out_dir,
                model_id=args.slm_model,
                sft_epochs=1 if args.smoke_test else 2,
                dpo_epochs=1,
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

    print("\n=== [MAVE-Q v4] Free-Cloud Training & Telemetry Summary ===")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
