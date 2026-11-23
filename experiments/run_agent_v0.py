"""Phase 3 CLI: Run Multi-Agent System v0 (<=8B) before any self-improvement, compare vs baselines with CIs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vn_rsi.agents import MultiAgentResearchPipeline
from vn_rsi.data import SnapshotManager, load_market_rules
from vn_rsi.eval import SealedEvaluator, paired_bootstrap_ci, run_all_baselines
from vn_rsi.llm import build_llm_backend
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
    parser = argparse.ArgumentParser(description="Phase 3: Run Multi-Agent System v0 (<=8B)")
    parser.add_argument("--agent-config", default="configs/agent_v0.yaml")
    parser.add_argument("--llm-config", default="configs/llm.yaml")
    parser.add_argument("--sandbox-config", default="configs/sandbox.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--market-rules", default="configs/market_rules.yaml")
    parser.add_argument("--logs-dir", default="logs")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    agent_cfg = load_yaml(repo_root / args.agent_config)
    llm_cfg = load_yaml(repo_root / args.llm_config)
    sandbox_cfg = load_yaml(repo_root / args.sandbox_config)
    data_cfg = load_yaml(repo_root / args.data_config)
    rules = load_market_rules(repo_root / args.market_rules)

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
    llm = build_llm_backend(llm_cfg, use_teacher=False)

    eval_seeds = [101, 202, 303]
    b_cfg = llm_cfg.get("budgets", {})
    vram = float(llm_cfg.get("hardware", {}).get("configured_vram_gb", 0.0))

    seed_runs = []
    total_tokens = 0
    total_accepted = 0

    for sd in eval_seeds:
        set_global_seed(sd)
        budget = BudgetTracker(
            BudgetConfig(
                max_tokens=int(b_cfg.get("inner_loop_max_tokens", 60_000)),
                max_wall_seconds=float(b_cfg.get("inner_loop_max_wall_seconds", 180.0)),
                max_gpu_hours=float(b_cfg.get("inner_loop_max_gpu_hours", 0.25)),
                configured_vram_gb=vram,
            ),
            scope_name=f"agent_v0_s{sd}",
        )
        pipeline = MultiAgentResearchPipeline(
            evaluator=evaluator,
            llm=llm,
            genome=dict(agent_cfg["genome"]),
            acceptance_cfg=dict(agent_cfg.get("acceptance_criteria", {})),
            enable_memory=True,
            enable_auditor=True,
        )
        res = pipeline.run(
            agent_version="agent_v0",
            n_rounds=int(agent_cfg.get("n_rounds", 3)),
            n_hypotheses_per_round=int(agent_cfg.get("n_hypotheses_per_round", 4)),
            budget=budget,
            seed=sd,
            run_id=f"phase3_agent_v0_s{sd}",
        )
        seed_runs.append(res)
        total_tokens += int(res.budget_snapshot["tokens_used"])
        total_accepted += len(res.accepted_specs)

    # Pick median-validation seed run as representative frozen v0 ensemble for ONCE-ONLY sealed_test scoring
    seed_runs_sorted = sorted(seed_runs, key=lambda r: r.validation_outcome.metrics["net_sharpe"])
    rep_run = seed_runs_sorted[len(seed_runs_sorted) // 2]

    sealed_v0 = evaluator.evaluate_factor_ensemble(
        specs=rep_run.accepted_specs,
        weights=rep_run.accepted_weights,
        zone="sealed_test",
        candidate_id="agent_v0_frozen_rep",
        run_id="phase3_agent_v0",
        allow_sealed=True,
    )

    # Compare against equal_weight and ridge on validation returns for paired bootstrap CIs
    val_baselines = run_all_baselines(evaluator, zone="validation", run_id="phase3_ref_val", score_sealed_test=False)
    ci_vs_eq = paired_bootstrap_ci(rep_run.validation_outcome.daily_net_returns, val_baselines["equal_weight"].daily_net_returns)
    ci_vs_ridge = paired_bootstrap_ci(rep_run.validation_outcome.daily_net_returns, val_baselines["ridge"].daily_net_returns)

    val_sharpes = [r.validation_outcome.metrics["net_sharpe"] for r in seed_runs]
    val_ics = [r.validation_outcome.metrics["rank_ic"] for r in seed_runs]

    cfg_hash = compute_config_hash({"agent": agent_cfg, "llm": llm_cfg, "sandbox": sandbox_cfg})[:16]
    logger = RunLogger(
        run_id="phase3_agent_v0",
        logs_dir=repo_root / args.logs_dir,
        git_commit=get_git_commit_hash(repo_root),
        config_hash=cfg_hash,
        evaluator_seal_hash=evaluator.seal_hash,
        metadata={
            "phase": 3,
            "llm_backend": llm.backend_id,
            "is_synthetic_data": snapshot.is_synthetic,
        },
    )
    summary = logger.finalize_summary(
        {
            "seeds": eval_seeds,
            "val_metrics_mean": {
                "rank_ic": round(float(np.mean(val_ics)), 5),
                "net_sharpe": round(float(np.mean(val_sharpes)), 4),
            },
            "val_metrics_std": {
                "rank_ic": round(float(np.std(val_ics, ddof=1)), 5),
                "net_sharpe": round(float(np.std(val_sharpes, ddof=1)), 4),
            },
            "rep_validation_metrics": rep_run.validation_outcome.metrics,
            "sealed_test_metrics": sealed_v0.metrics,
            "bootstrap_ci_val_vs_equal_weight": ci_vs_eq,
            "bootstrap_ci_val_vs_ridge": ci_vs_ridge,
            "cost_per_accepted_finding": {
                "total_tokens": total_tokens,
                "total_accepted_findings": total_accepted,
                "tokens_per_accepted_finding": round(total_tokens / max(1, total_accepted), 1),
            },
            "belief_memory_final": rep_run.memory_snapshot,
        }
    )
    print(
        json.dumps(
            {
                "status": "OK",
                "run_id": "phase3_agent_v0",
                "llm_backend": llm.backend_id,
                "val_sharpe_mean": summary["val_metrics_mean"]["net_sharpe"],
                "sealed_test_sharpe": sealed_v0.metrics["net_sharpe"],
                "tokens_per_finding": summary["cost_per_accepted_finding"]["tokens_per_accepted_finding"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
