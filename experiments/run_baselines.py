"""Phase 2 CLI: Lock Sealed Evaluator, verify Canary null-signal checks, and run all 7 human baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vn_rsi.data import SnapshotManager, load_market_rules
from vn_rsi.eval import SealedEvaluator, run_all_baselines
from vn_rsi.utils import (
    RunLogger,
    compute_config_hash,
    get_git_commit_hash,
    load_yaml,
    set_global_seed,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2: Run human baselines in sealed sandbox")
    parser.add_argument("--sandbox-config", default="configs/sandbox.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--market-rules", default="configs/market_rules.yaml")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--reset-ledger", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    sandbox_cfg = load_yaml(repo_root / args.sandbox_config)
    data_cfg = load_yaml(repo_root / args.data_config)
    rules = load_market_rules(repo_root / args.market_rules)
    set_global_seed(int(sandbox_cfg.get("seed", 42)))

    ledger_path = repo_root / str(sandbox_cfg.get("ledger_path", "logs/sealed_test_ledger.jsonl"))
    if args.reset_ledger and ledger_path.exists():
        ledger_path.unlink()
    sandbox_cfg["ledger_path"] = str(ledger_path)

    snap_mgr = SnapshotManager(repo_root / str(data_cfg.get("snapshot_root", "data/snapshots")))
    snapshot = snap_mgr.load_snapshot(str(data_cfg["snapshot_id"]))

    protected = [
        repo_root / "src" / "vn_rsi" / "sandbox",
        repo_root / "src" / "vn_rsi" / "eval",
        repo_root / args.sandbox_config,
        repo_root / args.market_rules,
    ]
    evaluator = SealedEvaluator(snapshot, rules, sandbox_cfg, protected_paths=protected)
    canary = evaluator.run_canary_checks(seed=999)
    if not canary["passed"]:
        raise SystemExit(f"Canary anti-reward-hacking check FAILED: {canary}")

    val_baselines = run_all_baselines(evaluator, zone="validation", run_id="phase2_baselines_val", score_sealed_test=False)
    test_baselines = run_all_baselines(evaluator, zone="sealed_test", run_id="phase2_baselines_test", score_sealed_test=True)

    cfg_hash = compute_config_hash({"sandbox": sandbox_cfg, "data": data_cfg})[:16]
    logger = RunLogger(
        run_id="phase2_baselines",
        logs_dir=repo_root / args.logs_dir,
        git_commit=get_git_commit_hash(repo_root),
        config_hash=cfg_hash,
        evaluator_seal_hash=evaluator.seal_hash,
        metadata={
            "phase": 2,
            "snapshot_id": snapshot.snapshot_id,
            "is_synthetic_data": snapshot.is_synthetic,
        },
    )
    summary = logger.finalize_summary(
        {
            "canary_check": canary,
            "validation_baselines": {k: v.metrics for k, v in val_baselines.items()},
            "sealed_test_baselines": {k: v.metrics for k, v in test_baselines.items()},
            "validation_fold_std": {k: v.fold_std for k, v in val_baselines.items()},
        }
    )
    print(
        json.dumps(
            {
                "status": "OK",
                "run_id": "phase2_baselines",
                "seal_hash": evaluator.seal_hash[:16],
                "canary_passed": canary["passed"],
                "val_sharpes": {k: v["net_sharpe"] for k, v in summary["validation_baselines"].items()},
                "test_sharpes": {k: v["net_sharpe"] for k, v in summary["sealed_test_baselines"].items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
