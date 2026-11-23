"""Frontier Combat CLI: Pit MAVE-Q (Orthogonal diffu-GRPO + HRM Specialist) vs 70B / 72B / 671B Frontier Models."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from vn_rsi.data import SnapshotManager, load_market_rules
from vn_rsi.eval import SealedEvaluator
from vn_rsi.models import FrontierCombatArena
from vn_rsi.utils import (
    RunLogger,
    compute_config_hash,
    get_git_commit_hash,
    load_yaml,
    set_global_seed,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Frontier Combat Arena: MAVE-Q vs 50B-671B Frontier Models")
    parser.add_argument("--combat-config", default="configs/frontier_combat.yaml")
    parser.add_argument("--sandbox-config", default="configs/sandbox.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--market-rules", default="configs/market_rules.yaml")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--preset", default=None, help="cpu_fast | kaggle_t4_optimal | kaggle_max_100m")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    combat_cfg = load_yaml(repo_root / args.combat_config)
    sandbox_cfg = load_yaml(repo_root / args.sandbox_config)
    data_cfg = load_yaml(repo_root / args.data_config)
    rules = load_market_rules(repo_root / args.market_rules)

    seed = int(combat_cfg.get("seed", 2026))
    set_global_seed(seed)

    preset = args.preset or str(combat_cfg.get("specialist_preset", "auto"))
    if preset == "auto":
        try:
            import torch

            preset = "kaggle_max_100m" if torch.cuda.is_available() else "cpu_fast"
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

    arena = FrontierCombatArena(evaluator=evaluator, combat_cfg=combat_cfg, preset=preset)
    res = arena.run_combat(run_id="phase5_frontier_combat")

    cfg_hash = compute_config_hash({"combat": combat_cfg, "preset": preset})[:16]
    logger = RunLogger(
        run_id="phase5_frontier_combat",
        logs_dir=repo_root / args.logs_dir,
        git_commit=get_git_commit_hash(repo_root),
        config_hash=cfg_hash,
        evaluator_seal_hash=evaluator.seal_hash,
        metadata={
            "phase": 5,
            "preset": preset,
            "is_synthetic_data": snapshot.is_synthetic,
        },
    )
    logger.finalize_summary(
        {
            "specialist_total_params": res.specialist_total_params,
            "specialist_vram_mb": res.specialist_vram_mb,
            "frontier_validation_scores": res.frontier_validation_scores,
            "frontier_sealed_test_scores": res.frontier_sealed_test_scores,
            "combat_rounds": [asdict(r) for r in res.combat_rounds],
            "mave_q_won_validation": res.mave_q_won_validation,
            "winning_round": res.winning_round,
            "champion_validation_metrics": res.champion_validation_metrics,
            "champion_sealed_test_metrics": res.champion_sealed_test_metrics,
            "champion_bootstrap_vs_strongest_frontier": res.champion_bootstrap_vs_strongest_frontier,
            "champion_expressions": res.champion_expressions,
            "verdict": res.verdict,
        }
    )
    print(
        json.dumps(
            {
                "status": "OK",
                "run_id": "phase5_frontier_combat",
                "preset": preset,
                "verdict": res.verdict,
                "winning_round": res.winning_round,
                "frontier_val_sharpes": {k: v["net_sharpe"] for k, v in res.frontier_validation_scores.items()},
                "mave_q_val_sharpe": res.champion_validation_metrics["net_sharpe"],
                "frontier_test_sharpes": {k: v.get("net_sharpe") for k, v in res.frontier_sealed_test_scores.items()},
                "mave_q_test_sharpe": res.champion_sealed_test_metrics.get("net_sharpe"),
                "bootstrap_vs_strongest_frontier": res.champion_bootstrap_vs_strongest_frontier,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
