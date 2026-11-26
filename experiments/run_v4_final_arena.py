"""Task 09 (Phase 10) CLI: Pre-Seal Audit + Single-Shot Sealed Test Arena for Frozen MAVE-Q v4 & Global SOTA.

STRICT SINGLE-SHOT SEALED TEST PROTOCOL:
1. Executes mandatory Pre-Seal Audit: DataQASuite + SHA-256 EvaluatorSeal check + zero prior v4 sealed entries.
2. Freezes `configs/maveq_v4.yaml` (seed=2026) and scores MAVE-Q v4 ONCE on `sealed_test`.
3. Scores the 5 Global Multi-Agent / Quant SOTA Competitors (`TradingAgents`, `FinMem`, `FinAgent`,
   `RD-Agent(Q)`, `AlphaForge-DE`) ONCE on `sealed_test` under identical rules & fresh SealedEvaluators.
4. Computes full Extended Metrics, Paired Block-Bootstrap 95% CIs, Hansen SPA & White Reality Check,
   and 4-Regime Breakdown, logging immutably to `logs/experiment_registry.jsonl` (`phase10_v4_final_arena`)
   plus saving daily return trajectories to `logs/v4_final_arena_trajectories.json` for publication figures.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from vn_rsi.benchmarks.extended_metrics import (
    classify_market_regimes,
    hansen_spa_and_white_rc,
    paired_block_bootstrap_metrics,
)
from vn_rsi.benchmarks.global_competitors import run_global_competitor_arena
from vn_rsi.data import DataQASuite, SnapshotManager, load_market_rules
from vn_rsi.eval import SealedEvaluator
from vn_rsi.models.cloud_presets import detect_cloud_compute
from vn_rsi.models.maveq_v4_veteran import MAVEQv4Engine
from vn_rsi.sandbox.seal import EvaluatorSeal
from vn_rsi.utils import (
    RunLogger,
    compute_config_hash,
    get_git_commit_hash,
    load_yaml,
    set_global_seed,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "phase10_v4_final_arena"
EXPECTED_SEAL_HASH = "4a0f5ad331bd982f5290e6f62e9ea5122a29f21675a2e197080e83e2f8f9331e"


def _series_to_dict(s: pd.Series) -> dict[str, float]:
    return {str(pd.Timestamp(k).date()): round(float(v), 6) for k, v in s.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 09: Pre-Seal Audit + Single-Shot Sealed Test Arena")
    parser.add_argument("--config", default="configs/maveq_v4.yaml")
    parser.add_argument("--sandbox-config", default="configs/sandbox.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--market-rules", default="configs/market_rules.yaml")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--preset", default="cpu_fast")
    parser.add_argument("--dev", action="store_true", help="Validation-only dry run (never touches sealed_test)")
    args = parser.parse_args()

    t_start = time.time()
    v4_cfg = load_yaml(REPO_ROOT / args.config)
    sandbox_cfg = load_yaml(REPO_ROOT / args.sandbox_config)
    data_cfg = load_yaml(REPO_ROOT / args.data_config)
    rules = load_market_rules(REPO_ROOT / args.market_rules)
    seed = int(v4_cfg.get("seed", 2026))
    set_global_seed(seed)

    preset = args.preset
    if preset == "auto":
        preset = str(detect_cloud_compute()["recommended_preset"])

    protected = [
        REPO_ROOT / "src" / "vn_rsi" / "sandbox",
        REPO_ROOT / "src" / "vn_rsi" / "eval",
        REPO_ROOT / args.sandbox_config,
        REPO_ROOT / args.market_rules,
    ]

    # --- Mandatory Pre-Seal Audit ---
    seal_hash = EvaluatorSeal(protected).verify_intact()
    if seal_hash != EXPECTED_SEAL_HASH:
        raise SystemExit(f"PRE-SEAL AUDIT FAILED: EvaluatorSeal mismatch {seal_hash} != {EXPECTED_SEAL_HASH}")

    snapshot = SnapshotManager(REPO_ROOT / str(data_cfg.get("snapshot_root", "data/snapshots"))).load_snapshot(
        str(data_cfg["snapshot_id"])
    )
    qa = DataQASuite(rules).run_all(snapshot)
    if not qa.passed:
        raise SystemExit(f"PRE-SEAL AUDIT FAILED: DataQASuite failed: {qa.checks}")

    if args.dev:
        sandbox_cfg["ledger_path"] = str(REPO_ROOT / "logs/dev/phase10_dev_unused_ledger.jsonl")
    else:
        sandbox_cfg["ledger_path"] = str(REPO_ROOT / str(sandbox_cfg.get("ledger_path", "logs/sealed_test_ledger.jsonl")))

    def new_evaluator() -> SealedEvaluator:
        return SealedEvaluator(snapshot, rules, sandbox_cfg, protected_paths=protected)

    score_sealed = not args.dev
    ev_ref = new_evaluator()
    bm_val_df = ev_ref.zone_guard._slice_zone(snapshot.benchmark, "validation").sort_values("date")
    bm_val = pd.Series(bm_val_df["return_1d"].to_numpy(dtype=float), index=bm_val_df["date"])
    bm_test_df = ev_ref.zone_guard._slice_zone(snapshot.benchmark, "sealed_test").sort_values("date")
    bm_test = pd.Series(bm_test_df["return_1d"].to_numpy(dtype=float), index=bm_test_df["date"])

    # 1. Frozen MAVE-Q v4 (Single-Shot Sealed Test!)
    t0 = time.time()
    ev_v4 = new_evaluator()
    v4_res = MAVEQv4Engine(ev_v4, v4_cfg, preset=preset).run(run_id=RUN_ID, score_sealed=score_sealed)
    t_v4 = round(time.time() - t0, 2)

    # 2. 5 Global SOTA Trading / Quant Competitors (Single-Shot Sealed Test!)
    t1 = time.time()
    global_res = run_global_competitor_arena(new_evaluator, run_id=RUN_ID, score_sealed=score_sealed, seed=seed)
    t_glob = round(time.time() - t1, 2)

    # 3. Statistical Dominance Tests (Paired Block-Bootstrap 95% CIs + Hansen SPA & White RC)
    val_series_map: dict[str, pd.Series] = {"MAVE-Q_v4": v4_res.validation_daily_returns}
    test_series_map: dict[str, pd.Series] = {}
    if score_sealed and len(v4_res.sealed_test_daily_returns) > 0:
        test_series_map["MAVE-Q_v4"] = v4_res.sealed_test_daily_returns

    for gk, gres in global_res.items():
        val_series_map[gk] = gres.validation_daily_returns
        if score_sealed and len(gres.sealed_test_daily_returns) > 0:
            test_series_map[gk] = gres.sealed_test_daily_returns

    val_boot: dict[str, Any] = {
        "vs_vnindex": paired_block_bootstrap_metrics(v4_res.validation_daily_returns, bm_val, n_boot=1000, seed=seed)
    }
    for gk, gres in global_res.items():
        val_boot[f"vs_{gk}"] = paired_block_bootstrap_metrics(
            v4_res.validation_daily_returns, gres.validation_daily_returns, n_boot=1000, seed=seed
        )
    val_spa = hansen_spa_and_white_rc(val_series_map, bm_val, n_boot=1000, seed=seed)

    test_boot: dict[str, Any] = {}
    test_spa: dict[str, Any] = {}
    if score_sealed and "MAVE-Q_v4" in test_series_map:
        test_boot["vs_vnindex"] = paired_block_bootstrap_metrics(
            v4_res.sealed_test_daily_returns, bm_test, n_boot=1000, seed=seed
        )
        for gk, gres in global_res.items():
            if len(gres.sealed_test_daily_returns) > 0:
                test_boot[f"vs_{gk}"] = paired_block_bootstrap_metrics(
                    v4_res.sealed_test_daily_returns, gres.sealed_test_daily_returns, n_boot=1000, seed=seed
                )
        test_spa = hansen_spa_and_white_rc(test_series_map, bm_test, n_boot=1000, seed=seed)

    v4_dict = asdict(v4_res)
    v4_dict.pop("validation_daily_returns", None)
    v4_dict.pop("sealed_test_daily_returns", None)

    global_summary: dict[str, Any] = {}
    for gk, gres in global_res.items():
        gd = asdict(gres)
        gd.pop("validation_daily_returns", None)
        gd.pop("sealed_test_daily_returns", None)
        global_summary[gk] = gd

    summary: dict[str, Any] = {
        "mode": "dev_validation_only" if args.dev else "final_single_shot_sealed_test",
        "preset": preset,
        "seed": seed,
        "is_synthetic_data": snapshot.is_synthetic,
        "pre_seal_audit": {
            "evaluator_seal_hash": seal_hash,
            "seal_matches_expected": seal_hash == EXPECTED_SEAL_HASH,
            "data_qa_passed": qa.passed,
        },
        "maveq_v4": v4_dict,
        "global_competitors": global_summary,
        "validation_bootstrap_v4": val_boot,
        "validation_spa_white_rc": val_spa,
        "sealed_test_bootstrap_v4": test_boot,
        "sealed_test_spa_white_rc": test_spa,
        "timings_sec": {
            "maveq_v4": t_v4,
            "global_competitors": t_glob,
            "total": round(time.time() - t_start, 2),
        },
    }

    # Save exact daily return trajectories + regime labels for publication figures (Task 10)
    val_regimes = classify_market_regimes(bm_val)
    test_regimes = classify_market_regimes(bm_test)
    trajectories = {
        "run_id": RUN_ID,
        "mode": summary["mode"],
        "validation": {
            "benchmark_vnindex": _series_to_dict(bm_val),
            "regimes": {str(pd.Timestamp(k).date()): str(v) for k, v in val_regimes.items()},
            "models": {k: _series_to_dict(s) for k, s in val_series_map.items()},
        },
        "sealed_test": {
            "benchmark_vnindex": _series_to_dict(bm_test),
            "regimes": {str(pd.Timestamp(k).date()): str(v) for k, v in test_regimes.items()},
            "models": {k: _series_to_dict(s) for k, s in test_series_map.items()},
        },
    }
    traj_path = REPO_ROOT / args.logs_dir / "v4_final_arena_trajectories.json"
    traj_path.parent.mkdir(parents=True, exist_ok=True)
    traj_path.write_text(json.dumps(trajectories, indent=2), encoding="utf-8")

    out_json = REPO_ROOT / args.logs_dir / "v4_final_arena.json"
    out_json.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    if not args.dev:
        logger = RunLogger(
            run_id=RUN_ID,
            logs_dir=REPO_ROOT / args.logs_dir,
            git_commit=get_git_commit_hash(REPO_ROOT),
            config_hash=compute_config_hash({"v4_cfg": v4_cfg, "preset": preset, "seed": seed})[:16],
            evaluator_seal_hash=seal_hash,
            metadata={"phase": 10, "preset": preset, "is_synthetic_data": snapshot.is_synthetic},
        )
        logger.finalize_summary(summary)

    brief = {
        "status": "OK",
        "mode": summary["mode"],
        "seal_hash": seal_hash,
        "maveq_v4_val": {
            k: v4_res.champion_validation_extended[k]
            for k in ("rank_ic", "icir", "net_sharpe", "net_sortino", "calmar_ratio", "ann_net_return", "max_drawdown", "daily_turnover", "deflated_sharpe_prob")
        },
        "maveq_v4_test": (
            {
                k: v4_res.champion_sealed_test_extended[k]
                for k in ("rank_ic", "icir", "net_sharpe", "net_sortino", "calmar_ratio", "ann_net_return", "max_drawdown", "daily_turnover", "deflated_sharpe_prob")
            }
            if v4_res.champion_sealed_test_extended
            else v4_res.champion_sealed_test_metrics
        ),
        "global_test": {
            gk: (
                {
                    k: gres.sealed_test_extended[k]
                    for k in ("rank_ic", "net_sharpe", "calmar_ratio", "ann_net_return", "max_drawdown", "deflated_sharpe_prob")
                }
                if gres.sealed_test_extended
                else gres.sealed_test_metrics
            )
            for gk, gres in global_res.items()
        },
        "sealed_test_bootstrap_v4_vs_vnindex": test_boot.get("vs_vnindex", {}),
        "sealed_test_spa_white_rc": test_spa,
        "timings_sec": summary["timings_sec"],
    }
    print(json.dumps(brief, indent=2, default=str))


if __name__ == "__main__":
    main()
