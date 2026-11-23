"""Modular Micro-RSI Runner: Evaluates HRM + Discrete AST Diffusion with Weight-Level ReST inside Sealed Sandbox."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from vn_rsi.agents import MultiAgentResearchPipeline
from vn_rsi.data import SnapshotManager, load_market_rules
from vn_rsi.eval import SealedEvaluator, paired_bootstrap_ci
from vn_rsi.models import ModularMicroRSIBackend
from vn_rsi.rsi import BiLevelRSIRunner
from vn_rsi.sandbox import SealedTestReuseError
from vn_rsi.utils import (
    BudgetConfig,
    BudgetTracker,
    RunLogger,
    compute_config_hash,
    get_git_commit_hash,
    load_yaml,
    set_global_seed,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Modular Micro-RSI (HRM + Discrete AST Diffusion)")
    parser.add_argument("--modular-config", default="configs/kaggle_modular_rsi.yaml")
    parser.add_argument("--rsi-config", default="configs/rsi.yaml")
    parser.add_argument("--agent-config", default="configs/agent_v0.yaml")
    parser.add_argument("--llm-config", default="configs/llm.yaml")
    parser.add_argument("--sandbox-config", default="configs/sandbox.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--market-rules", default="configs/market_rules.yaml")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--preset", default=None, help="Override preset: cpu_fast | kaggle_t4_optimal")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    mod_cfg = load_yaml(repo_root / args.modular_config)
    rsi_cfg = load_yaml(repo_root / args.rsi_config)
    agent_cfg = load_yaml(repo_root / args.agent_config)
    llm_cfg = load_yaml(repo_root / args.llm_config)
    sandbox_cfg = load_yaml(repo_root / args.sandbox_config)
    data_cfg = load_yaml(repo_root / args.data_config)
    rules = load_market_rules(repo_root / args.market_rules)

    seed = int(mod_cfg.get("seed", 2026))
    set_global_seed(seed)

    preset = args.preset or str(mod_cfg.get("model_preset", "auto"))
    if preset == "auto":
        try:
            import torch

            preset = "kaggle_t4_optimal" if torch.cuda.is_available() else "cpu_fast"
        except Exception:
            preset = "cpu_fast"

    sandbox_cfg["ledger_path"] = str(repo_root / str(sandbox_cfg.get("ledger_path", "logs/sealed_test_ledger.jsonl")))
    snap_mgr = SnapshotManager(repo_root / str(data_cfg.get("snapshot_root", "data/snapshots")))
    snapshot = snap_mgr.load_snapshot(str(data_cfg["snapshot_id"]))

    protected = [
        repo_root / "src" / "vn_rsi" / "sandbox",
        repo_root / "src" / "vn_rsi" / "eval",
        repo_root / args.sandbox_config,
        repo_root / args.market_rules,
    ]
    evaluator = SealedEvaluator(snapshot, rules, sandbox_cfg, protected_paths=protected)

    modular_llm = ModularMicroRSIBackend(preset=preset, seed=seed)

    # Phase A: Warm-up Search-Zone Exploration + Online Weight-Level ReST Self-Training
    # Strictly uses search-zone trajectories ONLY (never validation or sealed_test).
    warmup_budget = BudgetTracker(BudgetConfig(max_tokens=80_000, max_wall_seconds=180.0, max_gpu_hours=0.5), "modular_warmup")
    warmup_pipe = MultiAgentResearchPipeline(
        evaluator=evaluator,
        llm=modular_llm,
        genome=dict(agent_cfg["genome"]),
        acceptance_cfg=dict(agent_cfg.get("acceptance_criteria", {})),
    )
    warmup_res = warmup_pipe.run(
        agent_version="modular_warmup",
        n_rounds=2,
        n_hypotheses_per_round=4,
        budget=warmup_budget,
        seed=seed,
        run_id="phase4_modular_warmup",
    )

    # Register winning ASTs into Diffusion subtree-infilling pool & construct HRM search observations
    steward_stats = warmup_pipe._data_steward_search_summary()
    hrm_obs: list[dict[str, Any]] = []
    for rec in warmup_res.candidate_records:
        if rec.audit.passed:
            rv = modular_llm.hrm.encode_regime_state(steward_stats, warmup_res.family_ic_summary)
            hrm_obs.append(
                {
                    "regime_vec": rv,
                    "ast_tokens": modular_llm.diffusion.vocab.encode_ast(rec.spec.root),
                    "search_rank_ic": rec.search_rank_ic,
                    "family": rec.hypothesis.mechanism_family,
                }
            )
            if rec.accepted_on_search:
                modular_llm.register_verified_search_winner(rec.hypothesis.mechanism_family, rec.spec)

    weight_update_stats = modular_llm.self_improve_weights(
        verified_trajectories=warmup_res.verified_sft_trajectories,
        hrm_observations=hrm_obs,
        epochs=int(mod_cfg.get("models", {}).get("ast_diffusion", {}).get("rest_epochs_per_outer_iter", 8)),
    )

    # Phase B: Bi-Level Outer Loop with HRM + Discrete AST Diffusion
    mod_rsi_cfg = dict(rsi_cfg)
    mod_rsi_cfg["outer_iterations"] = int(mod_cfg.get("rsi_loop", {}).get("outer_iterations", 4))
    mod_rsi_cfg["inner_rounds_per_eval"] = int(mod_cfg.get("rsi_loop", {}).get("inner_rounds_per_eval", 2))
    mod_rsi_cfg["inner_hypotheses_per_round"] = int(mod_cfg.get("rsi_loop", {}).get("inner_hypotheses_per_round", 4))
    mod_rsi_cfg["selection_seeds"] = list(mod_cfg.get("rsi_loop", {}).get("selection_seeds", [101, 202, 303]))
    mod_rsi_cfg["self_training"] = {"export_sft_trajectories": False}

    runner = BiLevelRSIRunner(
        evaluator=evaluator,
        inner_llm=modular_llm,
        outer_proposer_llm=modular_llm,
        rsi_cfg=mod_rsi_cfg,
        agent_cfg=agent_cfg,
        llm_cfg=llm_cfg,
    )
    rsi_summary = runner.run_mode(mode_name="modular_micro_rsi", seed_override=seed)

    # Single-shot sealed_test evaluation of frozen best modular ensemble
    best = rsi_summary.best_entry
    try:
        sealed_out = evaluator.evaluate_factor_ensemble(
            specs=best.representative_specs,
            weights=best.representative_weights,
            zone="sealed_test",
            candidate_id="modular_micro_rsi_vstar",
            run_id="phase4_modular_rsi",
            allow_sealed=True,
        )
        sealed_metrics: dict[str, Any] = sealed_out.metrics
    except SealedTestReuseError:
        sealed_metrics = {"note": "ensemble_hash_already_scored_in_ledger"}

    eq_val = evaluator.evaluate_signal_matrix(
        evaluator.feature_panel["close"] * 0.0 + 1.0, zone="validation", candidate_id="eq_val_mod_ref"
    )
    mod_val = evaluator.evaluate_factor_ensemble(
        best.representative_specs, best.representative_weights, zone="validation", candidate_id="mod_val_ci"
    )
    ci_vs_eq = paired_bootstrap_ci(mod_val.daily_net_returns, eq_val.daily_net_returns)

    cfg_hash = compute_config_hash({"modular": mod_cfg, "preset": preset})[:16]
    logger = RunLogger(
        run_id="phase4_modular_rsi",
        logs_dir=repo_root / args.logs_dir,
        git_commit=get_git_commit_hash(repo_root),
        config_hash=cfg_hash,
        evaluator_seal_hash=evaluator.seal_hash,
        metadata={
            "phase": 4,
            "preset": preset,
            "backend": modular_llm.backend_id,
            "diffusion_runtime": modular_llm.diffusion.backend_runtime,
            "hrm_runtime": modular_llm.hrm.backend_runtime,
            "is_synthetic_data": snapshot.is_synthetic,
        },
    )
    summary = logger.finalize_summary(
        {
            "parameter_breakdown": {
                "narrator_slm_b": modular_llm.narrator.parameter_count_billions,
                "ast_diffusion_params": modular_llm.diffusion.parameter_count(),
                "ast_diffusion_vram_mb": modular_llm.diffusion.estimate_training_vram_mb(),
                "hrm_reasoner_params": modular_llm.hrm.parameter_count(),
                "hrm_reasoner_vram_mb": modular_llm.hrm.estimate_training_vram_mb(),
            },
            "weight_level_self_training": weight_update_stats,
            "seed_val_composite": rsi_summary.seed_entry.val_composite_mean,
            "best_val_composite": best.val_composite_mean,
            "best_val_composite_std": best.val_composite_std,
            "best_val_sharpe": best.val_sharpe_mean,
            "best_val_rank_ic": best.val_rank_ic_mean,
            "accepted_rewrites": rsi_summary.accepted_rewrites,
            "rejected_rewrites": rsi_summary.rejected_rewrites,
            "reward_hacking_attempts": rsi_summary.reward_hacking_attempts,
            "sealed_test_metrics": sealed_metrics,
            "val_bootstrap_ci_vs_equal_weight": ci_vs_eq,
            "iterations": [asdict(it) for it in rsi_summary.iteration_logs],
        }
    )
    print(
        json.dumps(
            {
                "status": "OK",
                "run_id": "phase4_modular_rsi",
                "preset": preset,
                "diffusion_params": summary["parameter_breakdown"]["ast_diffusion_params"],
                "hrm_params": summary["parameter_breakdown"]["hrm_reasoner_params"],
                "best_val_sharpe": best.val_sharpe_mean,
                "sealed_test_sharpe": sealed_metrics.get("net_sharpe"),
                "accepted_rewrites": rsi_summary.accepted_rewrites,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
