"""Task 08 (Phase 9) CLI: Multi-Seed Architectural Ablations (N=8 Seeds) & Multi-Market OOD Generalization.

STRICT VALIDATION/OOD PROTOCOL:
- NEVER touches main VN30 `sealed_test` or `logs/sealed_test_ledger.jsonl`.
- Part 1: 8-seed paired ablation study (`11,22,33,44,55,66,77,2026`) on `vn_equities_snapshot_v1` validation:
  `full_maveq_v4`, `no_case_memory`, `no_pre_mortem`, `no_regime_controller`, `no_zero_val_burn_cv`,
  and `no_typed_tree_mdlm` with paired Bootstrap 95% CIs and paired one-sided p-values.
- Part 2: Out-of-Distribution (OOD) generalization across:
  (a) `VN100_10d` (100-stock Vietnam universe, 10-day forecast horizon, T+2.5 & +/-7/10/15% bands),
  (b) `US_LIQUID_T1` (US Liquid Equities, T+1, no daily price bands, 2bps comm),
  (c) `CSI_EMERGING_T1` (CSI-style Emerging Market, T+1, +/-10% bands, 5bps stamp duty),
  benchmarked against Buy & Hold and all 5 Global SOTA Trading/Quant Agents.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vn_rsi.benchmarks.extended_metrics import (
    compute_extended_metrics,
    paired_block_bootstrap_metrics,
)
from vn_rsi.benchmarks.global_competitors import run_global_competitor_arena
from vn_rsi.benchmarks.multi_market import MultiMarketLoader
from vn_rsi.data import SnapshotManager, load_market_rules
from vn_rsi.eval import SealedEvaluator
from vn_rsi.eval.baselines import run_all_baselines
from vn_rsi.models.maveq_v4_veteran import MAVEQv4Engine
from vn_rsi.utils import RunLogger, compute_config_hash, get_git_commit_hash, load_yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "phase9_v4_ablation_and_ood"

ABLATION_VARIANTS: tuple[str, ...] = (
    "full_maveq_v4",
    "no_case_memory",
    "no_pre_mortem",
    "no_regime_controller",
    "no_zero_val_burn_cv",
    "no_typed_tree_mdlm",
)


def _ablation_cfg(base_cfg: dict[str, Any], variant: str, seed: int) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg["seed"] = seed
    ab = cfg.setdefault("ablations", {})
    ab["use_case_memory"] = variant != "no_case_memory"
    ab["use_pre_mortem"] = variant != "no_pre_mortem"
    ab["use_regime_controller"] = variant != "no_regime_controller"
    ab["use_zero_val_burn_cv"] = variant != "no_zero_val_burn_cv"
    ab["use_typed_tree_mdlm"] = variant != "no_typed_tree_mdlm"
    return cfg


def _run_ablation_job(args: tuple[str, int, str]) -> dict[str, Any]:
    variant, seed, preset = args
    rules = load_market_rules(REPO_ROOT / "configs/market_rules.yaml")
    sandbox_cfg = copy.deepcopy(load_yaml(REPO_ROOT / "configs/sandbox.yaml"))
    sandbox_cfg["ledger_path"] = str(REPO_ROOT / "logs/dev/v4_ablation_unused_ledger.jsonl")
    data_cfg = load_yaml(REPO_ROOT / "configs/data.yaml")
    snap = SnapshotManager(REPO_ROOT / str(data_cfg.get("snapshot_root", "data/snapshots"))).load_snapshot(
        str(data_cfg["snapshot_id"])
    )
    ev = SealedEvaluator(snap, rules, sandbox_cfg)
    base_cfg = load_yaml(REPO_ROOT / "configs/maveq_v4.yaml")
    cfg = _ablation_cfg(base_cfg, variant, seed)

    t0 = time.time()
    res = MAVEQv4Engine(ev, cfg, preset=preset).run(run_id=f"{RUN_ID}_{variant}_s{seed}", score_sealed=False)
    ext = res.champion_validation_extended
    return {
        "variant": variant,
        "seed": seed,
        "elapsed_sec": round(time.time() - t0, 2),
        "selected_controller": res.selected_controller,
        "trials": int(res.total_evaluator_trials),
        "val_queries": int(res.validation_queries_used),
        "rank_ic": float(ext["rank_ic"]),
        "icir": float(ext["icir"]),
        "net_sharpe": float(ext["net_sharpe"]),
        "net_sortino": float(ext["net_sortino"]),
        "calmar": float(ext["calmar_ratio"]),
        "ann_net_return": float(ext["ann_net_return"]),
        "max_drawdown": float(ext["max_drawdown"]),
        "daily_turnover": float(ext["daily_turnover"]),
        "dsr": float(ext["deflated_sharpe_prob"]),
        "val_returns": {str(k.date()): float(v) for k, v in res.validation_daily_returns.items()},
    }


def _paired_seed_bootstrap(
    full_vals: np.ndarray,
    ab_vals: np.ndarray,
    n_boot: int = 2000,
    seed: int = 2026,
) -> dict[str, float]:
    """Paired bootstrap 95% CI & one-sided p-value for (full - ablated) across independent seeds."""
    diffs = np.asarray(full_vals, dtype=float) - np.asarray(ab_vals, dtype=float)
    n = len(diffs)
    if n == 0:
        return {"diff_mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_one_sided": 1.0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diffs[idx].mean(axis=1)
    return {
        "diff_mean": round(float(diffs.mean()), 4),
        "ci_low": round(float(np.quantile(boot_means, 0.025)), 4),
        "ci_high": round(float(np.quantile(boot_means, 0.975)), 4),
        "p_one_sided": round(float(np.mean(boot_means <= 0.0)), 4),
    }


def _run_ood_market(
    label: str,
    profile: str,
    universe_tier: str,
    horizon_days: int,
    preset: str,
    seed: int,
) -> dict[str, Any]:
    """Run OOD validation benchmark comparing MAVE-Q v4 vs Buy & Hold and 5 Global SOTA Competitors."""
    loader = MultiMarketLoader(REPO_ROOT / "data/snapshots")
    env = loader.load_or_build(
        profile=profile,
        start_date="2019-01-02",
        end_date="2025-12-31",
        mode="offline_fixture",
        universe_tier=universe_tier,
        seed=20261123 if profile == "VN_HOSE_T25" else 20261199,
    )
    sandbox_cfg = copy.deepcopy(load_yaml(REPO_ROOT / "configs/sandbox.yaml"))
    sandbox_cfg["ledger_path"] = str(REPO_ROOT / f"logs/dev/ood_{label.lower()}_unused_ledger.jsonl")
    sandbox_cfg["prediction_horizon_days"] = horizon_days
    if profile != "VN_HOSE_T25":
        sandbox_cfg["rebalance_every_days"] = 3

    def new_ev() -> SealedEvaluator:
        return SealedEvaluator(env.snapshot, env.market_rules, sandbox_cfg)

    ev_ref = new_ev()
    bm_val_df = ev_ref.zone_guard._slice_zone(env.snapshot.benchmark, "validation").sort_values("date")
    bm_val = pd.Series(bm_val_df["return_1d"].to_numpy(dtype=float), index=bm_val_df["date"])
    bars_val = ev_ref.zone_guard._slice_zone(env.snapshot.bars, "validation")

    results: dict[str, dict[str, Any]] = {}

    # 1. Buy & Hold Benchmark
    b_outs = run_all_baselines(new_ev(), zone="validation", run_id=f"ood_{label}_bm", score_sealed_test=False)
    for bkey in ("vnindex_buy_hold", "equal_weight", "ridge"):
        if bkey in b_outs:
            bext = compute_extended_metrics(b_outs[bkey], bm_val, bars_slice=bars_val, n_trials=1, n_boot=200).to_dict()
            results[bkey] = {
                "trials": 1,
                "rank_ic": bext["rank_ic"],
                "net_sharpe": bext["net_sharpe"],
                "net_sortino": bext["net_sortino"],
                "calmar": bext["calmar_ratio"],
                "ann_net_return": bext["ann_net_return"],
                "max_drawdown": bext["max_drawdown"],
                "daily_turnover": bext["daily_turnover"],
                "dsr": bext["deflated_sharpe_prob"],
            }

    # 2. Global SOTA Competitors
    g_outs = run_global_competitor_arena(new_ev, run_id=f"ood_{label}_global", score_sealed=False, seed=seed)
    for gk, gres in g_outs.items():
        gext = gres.validation_extended
        results[gk] = {
            "trials": gres.total_evaluator_trials,
            "rank_ic": gext["rank_ic"],
            "net_sharpe": gext["net_sharpe"],
            "net_sortino": gext["net_sortino"],
            "calmar": gext["calmar_ratio"],
            "ann_net_return": gext["ann_net_return"],
            "max_drawdown": gext["max_drawdown"],
            "daily_turnover": gext["daily_turnover"],
            "dsr": gext["deflated_sharpe_prob"],
        }

    # 3. MAVE-Q v4 (Frozen config — zero market-specific retuning!)
    v4_cfg = copy.deepcopy(load_yaml(REPO_ROOT / "configs/maveq_v4.yaml"))
    v4_cfg["seed"] = seed
    ev_v4 = new_ev()
    v4_res = MAVEQv4Engine(ev_v4, v4_cfg, preset=preset).run(run_id=f"ood_{label}_v4", score_sealed=False)
    vext = v4_res.champion_validation_extended
    results["MAVE-Q_v4"] = {
        "trials": v4_res.total_evaluator_trials,
        "selected_controller": v4_res.selected_controller,
        "rank_ic": vext["rank_ic"],
        "net_sharpe": vext["net_sharpe"],
        "net_sortino": vext["net_sortino"],
        "calmar": vext["calmar_ratio"],
        "ann_net_return": vext["ann_net_return"],
        "max_drawdown": vext["max_drawdown"],
        "daily_turnover": vext["daily_turnover"],
        "dsr": vext["deflated_sharpe_prob"],
    }

    return {
        "market_label": label,
        "profile": profile,
        "universe_tier": universe_tier,
        "horizon_days": horizon_days,
        "is_synthetic": env.is_synthetic,
        "n_symbols": int(env.snapshot.bars["symbol"].nunique()),
        "models": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 08: MAVE-Q v4 8-Seed Ablations & Multi-Market OOD Suite")
    parser.add_argument("--seeds", default="11,22,33,44,55,66,77,2026")
    parser.add_argument("--preset", default="cpu_fast")
    parser.add_argument("--workers", type=int, default=min(8, max(2, os.cpu_count() or 4)))
    parser.add_argument("--dev", action="store_true", help="Do not write to experiment_registry.jsonl")
    args = parser.parse_args()

    t_start = time.time()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    jobs = [(v, s, args.preset) for v in ABLATION_VARIANTS for s in seeds]

    print(f"=== [Task 08] Running {len(jobs)} ablation jobs ({len(ABLATION_VARIANTS)} variants x {len(seeds)} seeds) with {args.workers} workers ===")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        ab_rows = list(pool.map(_run_ablation_job, jobs))

    metrics_keys = (
        "trials",
        "rank_ic",
        "icir",
        "net_sharpe",
        "net_sortino",
        "calmar",
        "ann_net_return",
        "max_drawdown",
        "daily_turnover",
        "dsr",
    )

    by_variant: dict[str, list[dict[str, Any]]] = {
        v: sorted([r for r in ab_rows if r["variant"] == v], key=lambda x: x["seed"])
        for v in ABLATION_VARIANTS
    }
    full_rows = by_variant["full_maveq_v4"]
    full_sharpes = np.asarray([r["net_sharpe"] for r in full_rows], dtype=float)
    full_calmars = np.asarray([r["calmar"] for r in full_rows], dtype=float)
    full_dsrs = np.asarray([r["dsr"] for r in full_rows], dtype=float)

    # Representative median-Sharpe seed daily series for daily block-bootstrap
    full_rep_idx = int(np.argsort(full_sharpes)[len(full_sharpes) // 2])
    full_rep_ser = pd.Series({pd.Timestamp(k): v for k, v in full_rows[full_rep_idx]["val_returns"].items()}).sort_index()

    ablation_agg: dict[str, dict[str, Any]] = {}
    for v in ABLATION_VARIANTS:
        rs = by_variant[v]
        entry: dict[str, Any] = {"n_seeds": len(rs)}
        for k in metrics_keys:
            arr = np.asarray([float(r[k]) for r in rs], dtype=float)
            entry[f"{k}_mean"] = round(float(arr.mean()), 4)
            entry[f"{k}_std"] = round(float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, 4)

        if v != "full_maveq_v4":
            ab_sharpes = np.asarray([r["net_sharpe"] for r in rs], dtype=float)
            ab_calmars = np.asarray([r["calmar"] for r in rs], dtype=float)
            ab_dsrs = np.asarray([r["dsr"] for r in rs], dtype=float)
            ab_rep_idx = int(np.argsort(ab_sharpes)[len(ab_sharpes) // 2])
            ab_rep_ser = pd.Series({pd.Timestamp(k): val for k, val in rs[ab_rep_idx]["val_returns"].items()}).sort_index()
            entry["paired_seed_boot_sharpe"] = _paired_seed_bootstrap(full_sharpes, ab_sharpes, seed=2026)
            entry["paired_seed_boot_calmar"] = _paired_seed_bootstrap(full_calmars, ab_calmars, seed=2027)
            entry["paired_seed_boot_dsr"] = _paired_seed_bootstrap(full_dsrs, ab_dsrs, seed=2028)
            entry["paired_daily_block_boot"] = paired_block_bootstrap_metrics(full_rep_ser, ab_rep_ser, n_boot=500, seed=2026)
        ablation_agg[v] = entry

    print("=== [Task 08] Running Multi-Market & Horizon OOD Generalization (VN100-10d, US_LIQUID_T1, CSI_EMERGING_T1) ===")
    ood_specs = [
        ("VN100_10d", "VN_HOSE_T25", "vn100", 10),
        ("US_LIQUID_T1", "US_LIQUID_T1", "us30", 5),
        ("CSI_EMERGING_T1", "CSI_EMERGING_T1", "csi30", 5),
    ]
    ood_results: dict[str, dict[str, Any]] = {}
    for lbl, prof, tier, h_days in ood_specs:
        ood_results[lbl] = _run_ood_market(lbl, prof, tier, h_days, preset=args.preset, seed=2026)

    # Strip large daily return dicts before saving registry summary
    compact_rows = [{k: val for k, val in r.items() if k != "val_returns"} for r in ab_rows]
    summary: dict[str, Any] = {
        "mode": "validation_and_ood_only_zero_sealed_access",
        "seeds": seeds,
        "ablation_aggregate": ablation_agg,
        "ablation_rows": compact_rows,
        "ood_generalization": ood_results,
        "elapsed_sec": round(time.time() - t_start, 2),
    }

    out_path = REPO_ROOT / "logs/v4_ablation_and_ood.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if not args.dev:
        logger = RunLogger(
            run_id=RUN_ID,
            logs_dir=REPO_ROOT / "logs",
            git_commit=get_git_commit_hash(REPO_ROOT),
            config_hash=compute_config_hash(
                {"variants": list(ABLATION_VARIANTS), "seeds": seeds, "ood": [s[0] for s in ood_specs]}
            )[:16],
            metadata={"phase": 9, "zone": "validation_and_ood_only", "n_seeds": len(seeds)},
        )
        logger.finalize_summary(summary)

    print(
        json.dumps(
            {
                "status": "OK",
                "ablation_aggregate": ablation_agg,
                "ood_generalization": ood_results,
                "elapsed_sec": summary["elapsed_sec"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
