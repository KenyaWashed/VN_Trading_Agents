"""Task 05: Autonomous Multi-Seed Self-Improvement & Validation-Only Dev Arena for MAVE-Q v4 (VetQuant-RSI).

STRICTLY `--dev` / VALIDATION-ONLY PROTOCOL:
- NEVER touches `sealed_test` or `logs/sealed_test_ledger.jsonl`.
- Runs MAVE-Q v4 across >= 5 independent seeds against ALL rivals (VN-Index Buy & Hold, 7 traditional
  baselines, 3 Frontier 70B-671B proxies, 5 Global SOTA Trading/Quant Agents [TradingAgents, FinMem,
  FinAgent, RD-Agent(Q), AlphaForge-DE], 7 Alpha Miners from registry, and Holo-MAVE-Q v3).
- Computes extended metrics (Net Sharpe, Sortino, Calmar, Ann. Return, MaxDD, Rank IC, DSR, 4-Regime
  breakdown, Paired Block-Bootstrap 95% CI, Hansen SPA p-value) and appends full evolution trace to
  `logs/v4_self_improvement_trace.jsonl`.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vn_rsi.benchmarks.extended_metrics import (
    compute_extended_metrics,
    hansen_spa_and_white_rc,
    paired_block_bootstrap_metrics,
)
from vn_rsi.benchmarks.global_competitors import run_global_competitor_arena
from vn_rsi.data import SnapshotManager, load_market_rules
from vn_rsi.eval import SealedEvaluator, run_all_baselines
from vn_rsi.models.frontier_arena import FrontierChallengerRunner, FrontierChallengerSpec
from vn_rsi.models.holo_maveq import HoloMAVEQEngine
from vn_rsi.models.maveq_v4_veteran import MAVEQv4Engine
from vn_rsi.utils import compute_config_hash, load_yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_v4_single_seed(args: tuple[int, dict[str, Any], str]) -> dict[str, Any]:
    seed, base_cfg, preset = args
    rules = load_market_rules(REPO_ROOT / "configs/market_rules.yaml")
    sandbox_cfg = copy.deepcopy(load_yaml(REPO_ROOT / "configs/sandbox.yaml"))
    sandbox_cfg["ledger_path"] = str(REPO_ROOT / "logs/dev/v4_dev_unused_ledger.jsonl")
    data_cfg = load_yaml(REPO_ROOT / "configs/data.yaml")
    snap = SnapshotManager(REPO_ROOT / str(data_cfg.get("snapshot_root", "data/snapshots"))).load_snapshot(
        str(data_cfg["snapshot_id"])
    )
    ev = SealedEvaluator(snap, rules, sandbox_cfg)
    cfg = copy.deepcopy(base_cfg)
    cfg["seed"] = seed

    t0 = time.time()
    res = MAVEQv4Engine(ev, cfg, preset=preset).run(run_id=f"v4_dev_s{seed}", score_sealed=False)
    elapsed = round(time.time() - t0, 2)
    return {
        "seed": seed,
        "elapsed_sec": elapsed,
        "selected_controller": res.selected_controller,
        "trials": res.total_evaluator_trials,
        "val_queries": res.validation_queries_used,
        "search_screened": res.search_candidates_screened,
        "params": res.specialist_total_params,
        "vram_mb": res.estimated_vram_mb,
        "metrics": res.champion_validation_metrics,
        "extended": res.champion_validation_extended,
        "expressions": res.champion_expressions,
        "weights": res.champion_weights,
        "niches": res.champion_niches,
        "regime_exposure": res.regime_exposure_summary,
        "val_returns": {str(k.date()): float(v) for k, v in res.validation_daily_returns.items()},
    }


def _summarize_ext(ext: dict[str, Any], trials: int) -> dict[str, Any]:
    rb = ext.get("regime_breakdown", {})
    return {
        "trials": trials,
        "rank_ic": ext["rank_ic"],
        "icir": ext["icir"],
        "net_sharpe": ext["net_sharpe"],
        "net_sortino": ext["net_sortino"],
        "calmar": ext["calmar_ratio"],
        "ann_net_return": ext["ann_net_return"],
        "max_drawdown": ext["max_drawdown"],
        "daily_turnover": ext["daily_turnover"],
        "dsr": ext["deflated_sharpe_prob"],
        "bull_ret": rb.get("Bull", {}).get("ann_net_return", 0.0),
        "bear_sharpe": rb.get("Bear_Crash", {}).get("net_sharpe", 0.0),
        "sideways_sharpe": rb.get("Sideways", {}).get("net_sharpe", 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MAVE-Q v4 Multi-Seed Validation-Only Self-Improvement Arena (--dev)")
    parser.add_argument("--config", default="configs/maveq_v4.yaml")
    parser.add_argument("--seeds", default="11,22,33,44,2026")
    parser.add_argument("--preset", default="cpu_fast")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--iteration-tag", default="iter_1")
    parser.add_argument("--hypothesis", default="Initial MAVE-Q v4 VetQuant-RSI baseline evaluation")
    parser.add_argument("--skip-v3-recompute", action="store_true")
    args = parser.parse_args()

    t_all = time.time()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    v4_cfg = load_yaml(REPO_ROOT / args.config)
    rules = load_market_rules(REPO_ROOT / "configs/market_rules.yaml")
    sandbox_cfg = copy.deepcopy(load_yaml(REPO_ROOT / "configs/sandbox.yaml"))
    sandbox_cfg["ledger_path"] = str(REPO_ROOT / "logs/dev/v4_dev_unused_ledger.jsonl")
    data_cfg = load_yaml(REPO_ROOT / "configs/data.yaml")
    snap = SnapshotManager(REPO_ROOT / str(data_cfg.get("snapshot_root", "data/snapshots"))).load_snapshot(
        str(data_cfg["snapshot_id"])
    )

    def new_ev() -> SealedEvaluator:
        return SealedEvaluator(snap, rules, sandbox_cfg)

    ev_ref = new_ev()
    bm_val_df = ev_ref.zone_guard._slice_zone(snap.benchmark, "validation").sort_values("date")
    bm_val = pd.Series(bm_val_df["return_1d"].to_numpy(dtype=float), index=bm_val_df["date"])
    bars_val = ev_ref.zone_guard._slice_zone(snap.bars, "validation")

    # 1. Run MAVE-Q v4 across >= 5 independent seeds in parallel
    with ProcessPoolExecutor(max_workers=min(args.workers, len(seeds))) as pool:
        v4_seed_rows = list(pool.map(_run_v4_single_seed, [(s, v4_cfg, args.preset) for s in seeds]))

    # Aggregate MAVE-Q v4 across seeds
    v4_series_list = [
        pd.Series({pd.Timestamp(k): v for k, v in r["val_returns"].items()}).sort_index()
        for r in v4_seed_rows
    ]
    # Representative median-Sharpe seed & cross-seed mean
    v4_sharpes = [r["extended"]["net_sharpe"] for r in v4_seed_rows]
    rep_idx = int(np.argsort(v4_sharpes)[len(v4_sharpes) // 2])
    v4_rep_series = v4_series_list[rep_idx]

    def _agg_mean_sd(vals: list[float]) -> dict[str, float]:
        arr = np.asarray(vals, dtype=float)
        return {
            "mean": round(float(arr.mean()), 4),
            "std": round(float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, 4),
            "min": round(float(arr.min()), 4),
            "max": round(float(arr.max()), 4),
        }

    v4_agg = {
        "n_seeds": len(v4_seed_rows),
        "trials": _agg_mean_sd([float(r["trials"]) for r in v4_seed_rows]),
        "rank_ic": _agg_mean_sd([float(r["extended"]["rank_ic"]) for r in v4_seed_rows]),
        "icir": _agg_mean_sd([float(r["extended"]["icir"]) for r in v4_seed_rows]),
        "net_sharpe": _agg_mean_sd([float(r["extended"]["net_sharpe"]) for r in v4_seed_rows]),
        "net_sortino": _agg_mean_sd([float(r["extended"]["net_sortino"]) for r in v4_seed_rows]),
        "calmar": _agg_mean_sd([float(r["extended"]["calmar_ratio"]) for r in v4_seed_rows]),
        "ann_net_return": _agg_mean_sd([float(r["extended"]["ann_net_return"]) for r in v4_seed_rows]),
        "max_drawdown": _agg_mean_sd([float(r["extended"]["max_drawdown"]) for r in v4_seed_rows]),
        "daily_turnover": _agg_mean_sd([float(r["extended"]["daily_turnover"]) for r in v4_seed_rows]),
        "dsr": _agg_mean_sd([float(r["extended"]["deflated_sharpe_prob"]) for r in v4_seed_rows]),
        "bull_ret": _agg_mean_sd(
            [float(r["extended"]["regime_breakdown"].get("Bull", {}).get("ann_net_return", 0.0)) for r in v4_seed_rows]
        ),
        "bear_sharpe": _agg_mean_sd(
            [float(r["extended"]["regime_breakdown"].get("Bear_Crash", {}).get("net_sharpe", 0.0)) for r in v4_seed_rows]
        ),
        "sideways_sharpe": _agg_mean_sd(
            [float(r["extended"]["regime_breakdown"].get("Sideways", {}).get("net_sharpe", 0.0)) for r in v4_seed_rows]
        ),
    }

    # 2. Evaluate all reference rivals strictly on VALIDATION
    scoreboard: dict[str, dict[str, Any]] = {}
    rival_returns: dict[str, pd.Series] = {}

    # 2a. 7 Traditional Baselines (incl. VN-Index Buy & Hold)
    b_outs = run_all_baselines(new_ev(), zone="validation", run_id="dev_baselines", score_sealed_test=False)
    for bk, bout in b_outs.items():
        bext = compute_extended_metrics(bout, bm_val, bars_slice=bars_val, n_trials=1, n_boot=200).to_dict()
        scoreboard[f"baseline:{bk}"] = _summarize_ext(bext, 1)
        rival_returns[f"baseline:{bk}"] = bout.daily_net_returns

    # 2b. 5 Global Multi-Agent / Quant SOTA Competitors
    global_res = run_global_competitor_arena(new_ev, run_id="dev_global", score_sealed=False, seed=2026)
    for gk, gres in global_res.items():
        scoreboard[f"global:{gk}"] = _summarize_ext(gres.validation_extended, gres.total_evaluator_trials)
        rival_returns[f"global:{gk}"] = gres.validation_daily_returns

    # 2c. 3 Frozen Frontier 70B-671B Challengers
    frontier_cfg = load_yaml(REPO_ROOT / "configs/frontier_combat.yaml")
    for ch in frontier_cfg.get("frontier_challengers", []):
        ev_f = new_ev()
        spec = FrontierChallengerSpec(
            name=str(ch["name"]),
            parameter_count_billions=float(ch["parameter_count_billions"]),
            api_env_var=str(ch.get("api_env_var", "")),
            endpoint_url=str(ch.get("endpoint_url", "")),
            api_model_id=str(ch.get("api_model_id", "")),
            prior_strength=float(ch.get("prior_strength", 0.8)),
        )
        runner = FrontierChallengerRunner(spec, ev_f)
        specs = runner._frozen_frontier_textbook_factors()
        weights = [max(0.05, ev_f.evaluate_factor(s, zone="search", run_id="dev_f").metrics["rank_ic"] + 0.02) for s in specs]
        fout = ev_f.evaluate_factor_ensemble(specs, weights, zone="validation", candidate_id=spec.name, run_id="dev_f")
        fext = compute_extended_metrics(fout, bm_val, bars_slice=bars_val, n_trials=ev_f.n_trials_evaluated, n_boot=200).to_dict()
        scoreboard[f"frontier:{spec.name}"] = _summarize_ext(fext, ev_f.n_trials_evaluated)
        rival_returns[f"frontier:{spec.name}"] = fout.daily_net_returns

    # 2d. Holo-MAVE-Q v3 (direct validation comparison)
    if not args.skip_v3_recompute:
        holo_cfg = load_yaml(REPO_ROOT / "configs/holo_maveq.yaml")
        ev_v3 = new_ev()
        v3_res = HoloMAVEQEngine(ev_v3, holo_cfg, preset=args.preset).run(run_id="dev_v3", score_sealed=False)
        from vn_rsi.eval.evaluator import EvaluationOutcome

        v3_out = EvaluationOutcome(
            candidate_id="holo_maveq_v3",
            candidate_hash="v3",
            zone="validation",
            metrics=v3_res.champion_validation_metrics,
            daily_net_returns=v3_res.validation_daily_returns,
        )
        v3_ext = compute_extended_metrics(
            v3_out, bm_val, bars_slice=bars_val, n_trials=v3_res.total_evaluator_trials, n_boot=200
        ).to_dict()
        scoreboard["maveq:holo_v3"] = _summarize_ext(v3_ext, v3_res.total_evaluator_trials)
        rival_returns["maveq:holo_v3"] = v3_res.validation_daily_returns

    # 2e. Include logged 7 Alpha Miners & MAVE-Q v1/v2 validation summaries from experiment_registry.jsonl
    reg_path = REPO_ROOT / "logs/experiment_registry.jsonl"
    if reg_path.exists():
        by_id: dict[str, dict[str, Any]] = {}
        for ln in reg_path.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                rec = json.loads(ln)
                by_id[str(rec["run_id"])] = rec
        if "phase7_holo_maveq_combat" in by_id:
            p7 = by_id["phase7_holo_maveq_combat"]
            for ck, cv in p7.get("competitors", {}).items():
                vm = cv["validation_metrics"]
                mdd = abs(float(vm["max_drawdown"]))
                scoreboard[f"miner:{ck}"] = {
                    "trials": int(cv["total_evaluator_trials"]),
                    "rank_ic": vm["rank_ic"],
                    "icir": vm["icir"],
                    "net_sharpe": vm["net_sharpe"],
                    "net_sortino": vm["net_sortino"],
                    "calmar": round(float(vm["ann_net_return"]) / max(mdd, 1e-6), 4),
                    "ann_net_return": vm["ann_net_return"],
                    "max_drawdown": vm["max_drawdown"],
                    "daily_turnover": vm["daily_turnover"],
                    "dsr": vm["deflated_sharpe_prob"],
                }

    # 3. Statistical dominance tests: Paired Block-Bootstrap & Hansen SPA
    strongest_rival_name = max(rival_returns, key=lambda k: scoreboard[k]["net_sharpe"])
    boot_vs_strongest = paired_block_bootstrap_metrics(
        v4_rep_series, rival_returns[strongest_rival_name], n_boot=600, seed=2026
    )
    boot_vs_bm = paired_block_bootstrap_metrics(v4_rep_series, bm_val, n_boot=600, seed=2026)
    all_spa_map = dict(rival_returns)
    all_spa_map["MAVE-Q_v4_rep"] = v4_rep_series
    spa_test = hansen_spa_and_white_rc(all_spa_map, bm_val, n_boot=400, seed=2026)

    # Gap analysis against best rival on every core dimension
    best_rival_sharpe = max(v["net_sharpe"] for v in scoreboard.values())
    best_rival_calmar = max(v["calmar"] for v in scoreboard.values())
    best_rival_ret = max(v["ann_net_return"] for v in scoreboard.values())
    best_rival_ic = max(v["rank_ic"] for v in scoreboard.values())
    best_rival_dsr = max(v["dsr"] for v in scoreboard.values())

    gaps = {
        "sharpe_margin_vs_best_rival": round(v4_agg["net_sharpe"]["mean"] - best_rival_sharpe, 4),
        "calmar_margin_vs_best_rival": round(v4_agg["calmar"]["mean"] - best_rival_calmar, 4),
        "ann_ret_margin_vs_best_rival": round(v4_agg["ann_net_return"]["mean"] - best_rival_ret, 4),
        "rank_ic_margin_vs_best_rival": round(v4_agg["rank_ic"]["mean"] - best_rival_ic, 4),
        "dsr_margin_vs_best_rival": round(v4_agg["dsr"]["mean"] - best_rival_dsr, 4),
        "strongest_rival_by_sharpe": strongest_rival_name,
    }

    trace_record = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "iteration_tag": args.iteration_tag,
        "hypothesis": args.hypothesis,
        "config_hash": compute_config_hash(v4_cfg)[:16],
        "mode": "dev_validation_only_zero_sealed_access",
        "seeds": seeds,
        "v4_aggregate": v4_agg,
        "v4_per_seed": [
            {k: v for k, v in r.items() if k != "val_returns"} for r in v4_seed_rows
        ],
        "rival_scoreboard": scoreboard,
        "gaps": gaps,
        "bootstrap_vs_strongest_rival": boot_vs_strongest,
        "bootstrap_vs_vnindex": boot_vs_bm,
        "hansen_spa_white_rc": spa_test,
        "elapsed_sec": round(time.time() - t_all, 2),
    }

    trace_path = REPO_ROOT / "logs/v4_self_improvement_trace.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(trace_record, default=str) + "\n")

    print(
        json.dumps(
            {
                "iteration_tag": args.iteration_tag,
                "v4_aggregate": v4_agg,
                "gaps": gaps,
                "bootstrap_vs_strongest_rival": boot_vs_strongest,
                "bootstrap_vs_vnindex": boot_vs_bm,
                "hansen_spa_white_rc": spa_test,
                "top_rivals": {
                    k: scoreboard[k]
                    for k in sorted(scoreboard, key=lambda x: scoreboard[x]["net_sharpe"], reverse=True)[:8]
                },
                "elapsed_sec": trace_record["elapsed_sec"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
