"""Phase 1 CLI: Build immutable Parquet market snapshot and run mandatory Data QA + Leakage tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vn_rsi.data import DataQASuite, SnapshotManager, build_provider, load_market_rules
from vn_rsi.dsl import ASTNode, FactorSpec, compute_feature_panel, evaluate_factor_spec
from vn_rsi.utils import (
    RunLogger,
    compute_config_hash,
    get_git_commit_hash,
    load_yaml,
    set_global_seed,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1: Build & verify leak-free VN equity snapshot")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--market-rules", default="configs/market_rules.yaml")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    data_cfg = load_yaml(repo_root / args.config)
    rules = load_market_rules(repo_root / args.market_rules)
    seed = int(data_cfg.get("seed", 20261123))
    set_global_seed(seed)

    provider = build_provider(
        mode=str(data_cfg.get("provider_mode", "auto")),
        sources=list(data_cfg.get("vnstock_sources", ["VCI", "TCBS"])),
        rate_limit_seconds=float(data_cfg.get("rate_limit_seconds", 1.0)),
    )
    snap_mgr = SnapshotManager(repo_root / str(data_cfg.get("snapshot_root", "data/snapshots")))
    snap_id = str(data_cfg["snapshot_id"])

    bundle = provider.fetch_bundle(
        universe_tier=str(data_cfg.get("universe_tier", "vn30")),
        start_date=str(data_cfg["start_date"]),
        end_date=str(data_cfg["end_date"]),
        market_rules=rules,
        seed=seed,
    )
    snapshot = snap_mgr.write_snapshot(
        snapshot_id=snap_id,
        bundle=bundle,
        config_metadata={"universe_tier": data_cfg.get("universe_tier", "vn30"), "seed": seed},
        overwrite=args.overwrite,
    )

    # Build representative causal DSL feature functions to run dynamic future-perturbation shift-test on subset
    mini_syms = sorted(snapshot.bars["symbol"].unique().tolist())[:4]
    mini_bars = snapshot.bars[snapshot.bars["symbol"].isin(mini_syms)].copy().reset_index(drop=True)
    mini_fund = snapshot.fundamentals_pit[snapshot.fundamentals_pit["symbol"].isin(mini_syms)].copy()

    sample_spec = FactorSpec(
        factor_id="qa_mom_vol",
        hypothesis_id="qa_0",
        description="15d return over 20d volatility",
        root=ASTNode(
            node_type="op",
            name="safe_div",
            args=[
                ASTNode(node_type="op", name="ts_return", window=15, args=[ASTNode(node_type="terminal", name="close")]),
                ASTNode(node_type="op", name="ts_std", window=20, args=[ASTNode(node_type="terminal", name="return_1d")]),
            ],
        ),
    )

    def _dsl_feature_fn(bars_in):  # type: ignore[no-untyped-def]
        p = compute_feature_panel(bars_in, mini_fund)
        mat = evaluate_factor_spec(sample_spec, p)
        stacked = mat.stack(future_stack=True)
        return bars_in.set_index(["date", "symbol"]).index.map(stacked).to_series().fillna(0.0)

    qa_suite = DataQASuite(
        market_rules=rules,
        max_source_diff_pct=float(data_cfg.get("qa_checks", {}).get("max_cross_source_adj_close_diff_pct", 0.02)),
    )
    qa_report = qa_suite.run_all(
        snapshot=snapshot,
        require_delisted=bool(data_cfg.get("qa_checks", {}).get("require_survivorship_delisted_presence", True)),
    )
    dsl_shift_ok = qa_suite.verify_feature_causality_shift_test(mini_bars, _dsl_feature_fn)
    qa_report.checks["dsl_shift_causality"] = {"passed": dsl_shift_ok}
    qa_report.passed = qa_report.passed and dsl_shift_ok

    cfg_hash = compute_config_hash({"data": data_cfg, "market_rules": rules.model_dump()})[:16]
    logger = RunLogger(
        run_id="phase1_data_qa",
        logs_dir=repo_root / args.logs_dir,
        git_commit=get_git_commit_hash(repo_root),
        config_hash=cfg_hash,
        evaluator_seal_hash="n/a_phase1",
        metadata={"phase": 1, "snapshot_id": snap_id, "is_synthetic_data": snapshot.is_synthetic},
    )
    summary = logger.finalize_summary(
        {
            "qa_passed": qa_report.passed,
            "qa_checks": qa_report.checks,
            "manifest_sha256": snapshot.manifest_sha256,
            "symbols_count": snapshot.manifest["symbols_count"],
            "bars_rows": snapshot.manifest["row_counts"]["bars"],
            "unverified_market_rules": rules.unverified_items(),
        }
    )
    if not qa_report.passed:
        raise SystemExit(f"Phase 1 Data QA FAILED: {json.dumps(qa_report.checks, indent=2)}")
    print(json.dumps({"status": "OK", "run_id": "phase1_data_qa", "qa_passed": summary["qa_passed"], "is_synthetic": snapshot.is_synthetic}, indent=2))


if __name__ == "__main__":
    main()
