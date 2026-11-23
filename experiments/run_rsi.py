"""Phase 4 CLI: Bi-Level Recursive Self-Improvement Loop + 6 Ablations + Ignition + OOD Generalization."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path

from vn_rsi.data import SnapshotManager, build_provider, load_market_rules
from vn_rsi.eval import SealedEvaluator, paired_bootstrap_ci, run_all_baselines
from vn_rsi.llm import build_llm_backend
from vn_rsi.rsi import BiLevelRSIRunner
from vn_rsi.utils import (
    RunLogger,
    compute_config_hash,
    get_git_commit_hash,
    load_yaml,
    set_global_seed,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4: Bi-level RSI loop, ablations, ignition, and OOD checks")
    parser.add_argument("--rsi-config", default="configs/rsi.yaml")
    parser.add_argument("--agent-config", default="configs/agent_v0.yaml")
    parser.add_argument("--llm-config", default="configs/llm.yaml")
    parser.add_argument("--sandbox-config", default="configs/sandbox.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--market-rules", default="configs/market_rules.yaml")
    parser.add_argument("--logs-dir", default="logs")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    rsi_cfg = load_yaml(repo_root / args.rsi_config)
    agent_cfg = load_yaml(repo_root / args.agent_config)
    llm_cfg = load_yaml(repo_root / args.llm_config)
    sandbox_cfg = load_yaml(repo_root / args.sandbox_config)
    data_cfg = load_yaml(repo_root / args.data_config)
    rules = load_market_rules(repo_root / args.market_rules)

    set_global_seed(int(rsi_cfg.get("seed", 2026)))
    sandbox_cfg["ledger_path"] = str(repo_root / str(sandbox_cfg.get("ledger_path", "logs/sealed_test_ledger.jsonl")))
    if "self_training" in rsi_cfg and "sft_output_path" in rsi_cfg["self_training"]:
        rsi_cfg["self_training"]["sft_output_path"] = str(repo_root / rsi_cfg["self_training"]["sft_output_path"])

    snap_mgr = SnapshotManager(repo_root / str(data_cfg.get("snapshot_root", "data/snapshots")))
    snapshot = snap_mgr.load_snapshot(str(data_cfg["snapshot_id"]))

    protected = [
        repo_root / "src" / "vn_rsi" / "sandbox",
        repo_root / "src" / "vn_rsi" / "eval",
        repo_root / args.sandbox_config,
        repo_root / args.market_rules,
    ]
    evaluator = SealedEvaluator(snapshot, rules, sandbox_cfg, protected_paths=protected)

    inner_llm = build_llm_backend(llm_cfg, use_teacher=False)
    teacher_llm = build_llm_backend(llm_cfg, use_teacher=True)

    ablation_modes: list[str] = list(
        rsi_cfg.get(
            "ablations",
            [
                "full_rsi_8b",
                "no_outer_loop",
                "no_memory",
                "random_mutation",
                "no_leakage_auditor",
                "teacher_proposer_sim",
            ],
        )
    )

    ablation_summaries: dict[str, dict] = {}
    full_rsi_summary = None

    for mode in ablation_modes:
        outer_llm = teacher_llm if mode == "teacher_proposer_sim" else inner_llm
        runner = BiLevelRSIRunner(
            evaluator=evaluator,
            inner_llm=inner_llm,
            outer_proposer_llm=outer_llm,
            rsi_cfg=rsi_cfg,
            agent_cfg=agent_cfg,
            llm_cfg=llm_cfg,
        )
        res = runner.run_mode(mode_name=mode)
        if mode == "full_rsi_8b":
            full_rsi_summary = res
        ablation_summaries[mode] = {
            "seed_val_composite": res.seed_entry.val_composite_mean,
            "best_val_composite": res.best_entry.val_composite_mean,
            "best_val_composite_std": res.best_entry.val_composite_std,
            "best_val_sharpe": res.best_entry.val_sharpe_mean,
            "best_val_rank_ic": res.best_entry.val_rank_ic_mean,
            "accepted_rewrites": res.accepted_rewrites,
            "rejected_rewrites": res.rejected_rewrites,
            "reward_hacking_attempts": res.reward_hacking_attempts,
            "archive_size": res.archive_size,
            "tokens_used": res.outer_budget_snapshot["tokens_used"],
            "wall_seconds": res.outer_budget_snapshot["wall_seconds"],
            "outer_proposer_backend": outer_llm.backend_id,
            "iterations": [asdict(it) for it in res.iteration_logs],
        }

    assert full_rsi_summary is not None
    best_entry = full_rsi_summary.best_entry

    # 1. Score single frozen best RSI agent ensemble ONCE on `sealed_test`
    # Ensure distinct hash label if identical ASTs to v0 occurred; ledger checks exact expression+weights hash!
    # If v* discovered genuinely new ASTs/weights, its spec hash differs; if v* == v0 (0 rewrites accepted), handle cleanly.
    if best_entry.genome_hash == full_rsi_summary.seed_entry.genome_hash:
        sealed_rsi_metrics = {"note": "identical_to_v0_no_rewrite_accepted"}
        sealed_ci_vs_eq = {}
    else:
        sealed_rsi_out = evaluator.evaluate_factor_ensemble(
            specs=best_entry.representative_specs,
            weights=best_entry.representative_weights,
            zone="sealed_test",
            candidate_id="rsi_vstar_frozen_best",
            run_id="phase4_rsi",
            allow_sealed=True,
        )
        sealed_rsi_metrics = sealed_rsi_out.metrics
        eq_test = evaluator.evaluate_signal_matrix(
            signal_wide=evaluator.feature_panel["close"] * 0.0 + 1.0,
            zone="validation",
            candidate_id="eq_val_ref",
            run_id="phase4_rsi",
        )
        val_rsi_out = evaluator.evaluate_factor_ensemble(
            specs=best_entry.representative_specs,
            weights=best_entry.representative_weights,
            zone="validation",
            candidate_id="rsi_vstar_val_ci",
            run_id="phase4_rsi",
        )
        sealed_ci_vs_eq = paired_bootstrap_ci(val_rsi_out.daily_net_returns, eq_test.daily_net_returns)

    # 2. Ignition check on fresh unseen seeds
    ign_runner = BiLevelRSIRunner(evaluator, inner_llm, inner_llm, rsi_cfg, agent_cfg, llm_cfg)
    ign_seeds = [int(s) for s in rsi_cfg.get("ood_evaluation", {}).get("ignition_seeds", [404, 505, 606])]
    ignition_report = ign_runner.run_ignition_check(
        v0_genome=dict(agent_cfg["genome"]),
        evolved_genome=best_entry.genome,
        ignition_seeds=ign_seeds,
    )

    # 3. Out-of-Distribution (OOD) Generalization: VN100 multi-exchange (HOSE + HNX + UPCoM) & 10-day horizon
    ood_cfg = rsi_cfg.get("ood_evaluation", {})
    ood_tier = str(ood_cfg.get("ood_universe_tier", "vn100"))
    ood_horizon = int(ood_cfg.get("ood_prediction_horizon_days", 10))
    ood_snap_id = f"{data_cfg['snapshot_id']}_ood_{ood_tier}"

    provider = build_provider("synthetic_fixture" if snapshot.is_synthetic else str(data_cfg.get("provider_mode", "auto")))
    ood_bundle = provider.fetch_bundle(
        universe_tier=ood_tier,
        start_date=str(data_cfg["start_date"]),
        end_date=str(data_cfg["end_date"]),
        market_rules=rules,
        seed=int(data_cfg.get("seed", 20261123)) + 77,
    )
    ood_snapshot = snap_mgr.write_snapshot(ood_snap_id, ood_bundle, config_metadata={"ood_tier": ood_tier})
    ood_sandbox_cfg = copy.deepcopy(sandbox_cfg)
    ood_sandbox_cfg["prediction_horizon_days"] = ood_horizon
    ood_sandbox_cfg["ledger_path"] = str(repo_root / "logs" / "ood_sealed_ledger.jsonl")
    ood_eval = SealedEvaluator(ood_snapshot, rules, ood_sandbox_cfg)

    ood_v0 = ood_eval.evaluate_factor_ensemble(
        full_rsi_summary.seed_entry.representative_specs,
        full_rsi_summary.seed_entry.representative_weights,
        zone="validation",
        candidate_id="ood_v0",
    )
    ood_vstar = ood_eval.evaluate_factor_ensemble(
        best_entry.representative_specs,
        best_entry.representative_weights,
        zone="validation",
        candidate_id="ood_vstar",
    )
    ood_baselines = run_all_baselines(ood_eval, zone="validation", run_id="ood_baselines", score_sealed_test=False)

    ood_report = {
        "ood_universe_tier": ood_tier,
        "ood_symbols_count": ood_snapshot.manifest["symbols_count"],
        "ood_horizon_days": ood_horizon,
        "v0_val_metrics": ood_v0.metrics,
        "vstar_val_metrics": ood_vstar.metrics,
        "equal_weight_val_metrics": ood_baselines["equal_weight"].metrics,
        "ridge_val_metrics": ood_baselines["ridge"].metrics,
    }

    cfg_hash = compute_config_hash({"rsi": rsi_cfg, "agent": agent_cfg, "llm": llm_cfg})[:16]
    logger = RunLogger(
        run_id="phase4_rsi",
        logs_dir=repo_root / args.logs_dir,
        git_commit=get_git_commit_hash(repo_root),
        config_hash=cfg_hash,
        evaluator_seal_hash=evaluator.seal_hash,
        metadata={
            "phase": 4,
            "inner_llm_backend": inner_llm.backend_id,
            "is_synthetic_data": snapshot.is_synthetic,
        },
    )
    final_summary = logger.finalize_summary(
        {
            "ablations": ablation_summaries,
            "best_evolved_genome": best_entry.genome,
            "sealed_test_vstar_metrics": sealed_rsi_metrics,
            "val_bootstrap_ci_vstar_vs_eq": sealed_ci_vs_eq,
            "ignition_check": ignition_report,
            "ood_generalization": ood_report,
        }
    )
    print(
        json.dumps(
            {
                "status": "OK",
                "run_id": "phase4_rsi",
                "accepted_rewrites": ablation_summaries["full_rsi_8b"]["accepted_rewrites"],
                "v0_val_composite": ablation_summaries["full_rsi_8b"]["seed_val_composite"],
                "vstar_val_composite": ablation_summaries["full_rsi_8b"]["best_val_composite"],
                "sealed_test_vstar_sharpe": sealed_rsi_metrics.get("net_sharpe"),
                "ignition_status": ignition_report["status"],
                "ood_vstar_sharpe": ood_vstar.metrics["net_sharpe"],
            },
            indent=2,
        )
    )
    _ = final_summary


if __name__ == "__main__":
    main()
