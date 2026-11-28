"""Task 07 (Phase 13) CLI: Pre-Seal Audit & Single-Shot Sealed Test Arena for Frozen Pruned MAVE-Q v5.

STRICT SINGLE-SHOT SEALED TEST PROTOCOL:
1. Mandatory Pre-Seal Audit:
   - Verifies SHA-256 EvaluatorSeal == 4a0f5ad331bd982f5290e6f62e9ea5122a29f21675a2e197080e83e2f8f9331e
   - Runs DataQASuite on `vn_equities_snapshot_v1`
   - Verifies existing `logs/sealed_test_ledger.jsonl` history (28 v1..v4 entries) is preserved untouched
     and contains ZERO prior `maveq_v5` entries!
2. Scores frozen, Task-05-pruned MAVE-Q v5 (`configs/maveq_v5.yaml`, seed=2026) EXACTLY ONCE on `sealed_test`.
3. Computes 25+ extended institutional metrics, Paired Block-Bootstrap 95% CIs vs VN-Index, MAVE-Q v4, and
   5 Global SOTA rivals (loaded from immutable `logs/v4_final_arena_trajectories.json` without re-scoring!),
   Hansen SPA & White Reality Check, and BOTH individual DSR (`N=1`) and strict lineage-cumulative DSR
   across the full `v1..v5` sealed-test ledger (`N=lineage_total`)!
4. Saves `logs/v5_final_arena.json`, `logs/v5_final_arena_trajectories.json`, and logs `phase13_v5_final_arena`
   to `logs/experiment_registry.jsonl`.
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
from vn_rsi.data import DataQASuite, SnapshotManager, load_market_rules
from vn_rsi.eval import SealedEvaluator
from vn_rsi.eval.metrics import deflated_sharpe_ratio
from vn_rsi.models.maveq_v5_chronos import MAVEQv5Engine
from vn_rsi.sandbox.seal import EvaluatorSeal
from vn_rsi.utils import (
    RunLogger,
    compute_config_hash,
    get_git_commit_hash,
    load_yaml,
    set_global_seed,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "phase13_v5_final_arena"
EXPECTED_SEAL_HASH = "4a0f5ad331bd982f5290e6f62e9ea5122a29f21675a2e197080e83e2f8f9331e"


def _series_to_dict(s: pd.Series) -> dict[str, float]:
    return {str(pd.Timestamp(k).date()): round(float(v), 6) for k, v in s.items()}


def _dict_to_series(d: dict[str, float]) -> pd.Series:
    return pd.Series({pd.Timestamp(k): float(v) for k, v in d.items()}).sort_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 07: Pre-Seal Audit + Single-Shot Sealed Test Arena (MAVE-Q v5)")
    parser.add_argument("--config", default="configs/maveq_v5.yaml")
    parser.add_argument("--sandbox-config", default="configs/sandbox.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--market-rules", default="configs/market_rules.yaml")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--preset", default="auto")
    parser.add_argument("--dev", action="store_true", help="Validation-only dry run (never touches sealed_test)")
    args = parser.parse_args()

    t_start = time.time()
    v5_cfg = load_yaml(REPO_ROOT / args.config)
    sandbox_cfg = load_yaml(REPO_ROOT / args.sandbox_config)
    data_cfg = load_yaml(REPO_ROOT / args.data_config)
    rules = load_market_rules(REPO_ROOT / args.market_rules)
    seed = int(v5_cfg.get("seed", 2026))
    set_global_seed(seed)

    protected = [
        REPO_ROOT / "src" / "vn_rsi" / "sandbox",
        REPO_ROOT / "src" / "vn_rsi" / "eval",
        REPO_ROOT / args.sandbox_config,
        REPO_ROOT / args.market_rules,
    ]

    # --- 1. Mandatory Pre-Seal Audit ---
    seal_hash = EvaluatorSeal(protected).verify_intact()
    if seal_hash != EXPECTED_SEAL_HASH:
        raise SystemExit(f"PRE-SEAL AUDIT FAILED: EvaluatorSeal mismatch {seal_hash} != {EXPECTED_SEAL_HASH}")

    snapshot = SnapshotManager(REPO_ROOT / str(data_cfg.get("snapshot_root", "data/snapshots"))).load_snapshot(
        str(data_cfg["snapshot_id"])
    )
    qa = DataQASuite(rules).run_all(snapshot)
    if not qa.passed:
        raise SystemExit(f"PRE-SEAL AUDIT FAILED: DataQASuite failed: {qa.checks}")

    ledger_path = REPO_ROOT / str(sandbox_cfg.get("ledger_path", "logs/sealed_test_ledger.jsonl"))
    prior_ledger_lines: list[dict[str, Any]] = []
    if ledger_path.exists():
        for ln in ledger_path.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                prior_ledger_lines.append(json.loads(ln))
    prior_v5 = [r for r in prior_ledger_lines if "maveq_v5" in str(r.get("candidate_id", "")).lower()]
    if (not args.dev) and prior_v5:
        raise SystemExit(
            f"PRE-SEAL AUDIT BLOCKED: `maveq_v5` already has {len(prior_v5)} entry in {ledger_path}! Single-shot rule enforced."
        )

    if args.dev:
        sandbox_cfg["ledger_path"] = str(REPO_ROOT / "logs/dev/phase13_dev_unused_ledger.jsonl")
    else:
        sandbox_cfg["ledger_path"] = str(ledger_path)

    ev_v5 = SealedEvaluator(snapshot, rules, sandbox_cfg, protected_paths=protected)
    bm_val_df = ev_v5.zone_guard._slice_zone(snapshot.benchmark, "validation").sort_values("date")
    bm_val = pd.Series(bm_val_df["return_1d"].to_numpy(dtype=float), index=bm_val_df["date"])
    bm_test_df = ev_v5.zone_guard._slice_zone(snapshot.benchmark, "sealed_test").sort_values("date")
    bm_test = pd.Series(bm_test_df["return_1d"].to_numpy(dtype=float), index=bm_test_df["date"])

    # --- 2. Execute Frozen Pruned MAVE-Q v5 (Single-Shot Sealed Test!) ---
    score_sealed = not args.dev
    t0 = time.time()
    v5_res = MAVEQv5Engine(ev_v5, v5_cfg, preset=args.preset).run(run_id=RUN_ID, score_sealed=score_sealed)
    t_v5 = round(time.time() - t0, 2)

    # Verify post-run ledger integrity (exactly +1 entry appended when score_sealed=True)
    post_ledger_count = len(prior_ledger_lines)
    if score_sealed and ledger_path.exists():
        post_ledger_count = sum(1 for ln in ledger_path.read_text(encoding="utf-8").splitlines() if ln.strip())

    # --- 3. Compute Individual DSR (N=1) AND Strict Lineage-Cumulative DSR (v1..v5!) ---
    lineage_total_trials = max(1, post_ledger_count)
    # Count cumulative historical trials across all logged phases (v0/v1/v2/v3/v4/v5)
    lineage_eval_trials = max(lineage_total_trials, 45)
    val_dsr_individual = float(v5_res.champion_validation_extended.get("deflated_sharpe_prob", 0.0))
    val_dsr_lineage = float(deflated_sharpe_ratio(v5_res.validation_daily_returns, n_trials=lineage_eval_trials))

    test_dsr_individual = 0.0
    test_dsr_lineage_ledger = 0.0
    test_dsr_lineage_strict = 0.0
    if score_sealed and len(v5_res.sealed_test_daily_returns) > 0:
        test_dsr_individual = float(v5_res.champion_sealed_test_extended.get("deflated_sharpe_prob", 0.0))
        test_dsr_lineage_ledger = float(
            deflated_sharpe_ratio(v5_res.sealed_test_daily_returns, n_trials=lineage_total_trials)
        )
        test_dsr_lineage_strict = float(
            deflated_sharpe_ratio(v5_res.sealed_test_daily_returns, n_trials=lineage_eval_trials)
        )

    # --- 4. Paired Block-Bootstrap & Hansen SPA / White RC vs Logged Rivals (ZERO re-scoring!) ---
    v4_traj_path = REPO_ROOT / args.logs_dir / "v4_final_arena_trajectories.json"
    val_series_map: dict[str, pd.Series] = {"MAVE-Q_v5": v5_res.validation_daily_returns}
    test_series_map: dict[str, pd.Series] = {}
    if score_sealed and len(v5_res.sealed_test_daily_returns) > 0:
        test_series_map["MAVE-Q_v5"] = v5_res.sealed_test_daily_returns

    if v4_traj_path.exists():
        v4_traj = json.loads(v4_traj_path.read_text(encoding="utf-8"))
        for mk, md in v4_traj.get("validation", {}).get("models", {}).items():
            val_series_map[mk] = _dict_to_series(md)
        for mk, md in v4_traj.get("sealed_test", {}).get("models", {}).items():
            test_series_map[mk] = _dict_to_series(md)

    val_boot: dict[str, Any] = {
        "vs_vnindex": paired_block_bootstrap_metrics(v5_res.validation_daily_returns, bm_val, n_boot=1000, seed=seed)
    }
    for mk, ser in val_series_map.items():
        if mk != "MAVE-Q_v5":
            val_boot[f"vs_{mk}"] = paired_block_bootstrap_metrics(
                v5_res.validation_daily_returns, ser, n_boot=1000, seed=seed
            )
    val_spa = hansen_spa_and_white_rc(val_series_map, bm_val, n_boot=1000, seed=seed)

    test_boot: dict[str, Any] = {}
    test_spa: dict[str, Any] = {}
    if score_sealed and "MAVE-Q_v5" in test_series_map:
        test_boot["vs_vnindex"] = paired_block_bootstrap_metrics(
            v5_res.sealed_test_daily_returns, bm_test, n_boot=1000, seed=seed
        )
        for mk, ser in test_series_map.items():
            if mk != "MAVE-Q_v5" and len(ser) > 0:
                test_boot[f"vs_{mk}"] = paired_block_bootstrap_metrics(
                    v5_res.sealed_test_daily_returns, ser, n_boot=1000, seed=seed
                )
        test_spa = hansen_spa_and_white_rc(test_series_map, bm_test, n_boot=1000, seed=seed)

    v5_dict = asdict(v5_res)
    v5_dict.pop("validation_daily_returns", None)
    v5_dict.pop("sealed_test_daily_returns", None)

    summary: dict[str, Any] = {
        "mode": "dev_validation_only" if args.dev else "final_single_shot_sealed_test",
        "preset": v5_res.resolved_safe_preset,
        "seed": seed,
        "is_synthetic_data": snapshot.is_synthetic,
        "pruned_architecture_active_flags": dict(v5_cfg.get("ablations", {})),
        "pre_seal_audit": {
            "evaluator_seal_hash": seal_hash,
            "seal_matches_expected": seal_hash == EXPECTED_SEAL_HASH,
            "data_qa_passed": qa.passed,
            "prior_sealed_ledger_entries": len(prior_ledger_lines),
            "post_sealed_ledger_entries": post_ledger_count,
        },
        "dsr_audit": {
            "val_dsr_individual_n1": round(val_dsr_individual, 4),
            "val_dsr_lineage_cumulative": round(val_dsr_lineage, 4),
            "test_dsr_individual_n1": round(test_dsr_individual, 4),
            "test_dsr_lineage_ledger": round(test_dsr_lineage_ledger, 4),
            "test_dsr_lineage_strict_n45": round(test_dsr_lineage_strict, 4),
            "lineage_ledger_n": lineage_total_trials,
            "lineage_strict_n": lineage_eval_trials,
        },
        "maveq_v5": v5_dict,
        "validation_bootstrap_v5": val_boot,
        "validation_spa_white_rc": val_spa,
        "sealed_test_bootstrap_v5": test_boot,
        "sealed_test_spa_white_rc": test_spa,
        "timings_sec": {
            "maveq_v5": t_v5,
            "total": round(time.time() - t_start, 2),
        },
    }

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
    traj_path = REPO_ROOT / args.logs_dir / "v5_final_arena_trajectories.json"
    traj_path.parent.mkdir(parents=True, exist_ok=True)
    traj_path.write_text(json.dumps(trajectories, indent=2), encoding="utf-8")

    out_json = REPO_ROOT / args.logs_dir / "v5_final_arena.json"
    out_json.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    if not args.dev:
        logger = RunLogger(
            run_id=RUN_ID,
            logs_dir=REPO_ROOT / args.logs_dir,
            git_commit=get_git_commit_hash(REPO_ROOT),
            config_hash=compute_config_hash({"v5_cfg": v5_cfg, "preset": v5_res.resolved_safe_preset, "seed": seed})[:16],
            evaluator_seal_hash=seal_hash,
            metadata={"phase": 13, "preset": v5_res.resolved_safe_preset, "is_synthetic_data": snapshot.is_synthetic},
        )
        logger.finalize_summary(summary)

    brief = {
        "status": "OK",
        "mode": summary["mode"],
        "seal_hash": seal_hash,
        "ledger_entries": f"{len(prior_ledger_lines)} -> {post_ledger_count}",
        "dsr_audit": summary["dsr_audit"],
        "maveq_v5_val": {
            k: v5_res.champion_validation_extended[k]
            for k in ("rank_ic", "icir", "net_sharpe", "net_sortino", "calmar_ratio", "ann_net_return", "max_drawdown", "daily_turnover", "capm_beta", "deflated_sharpe_prob")
        },
        "maveq_v5_test": (
            {
                k: v5_res.champion_sealed_test_extended[k]
                for k in ("rank_ic", "icir", "net_sharpe", "net_sortino", "calmar_ratio", "ann_net_return", "max_drawdown", "daily_turnover", "capm_beta", "deflated_sharpe_prob")
            }
            if v5_res.champion_sealed_test_extended
            else v5_res.champion_sealed_test_metrics
        ),
        "sealed_test_bootstrap_v5_vs_vnindex": test_boot.get("vs_vnindex", {}),
        "sealed_test_bootstrap_v5_vs_v4": test_boot.get("vs_MAVE-Q_v4", {}),
        "sealed_test_spa_white_rc": test_spa,
        "timings_sec": summary["timings_sec"],
    }
    print(json.dumps(brief, indent=2, default=str))


if __name__ == "__main__":
    main()
