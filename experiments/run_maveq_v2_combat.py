"""Phase 6 CLI: MAVE-Q v2 (Friction-Aware QD Evolution + Portfolio-Level Credit) vs same-topic alpha miners.

Protocol
- Every method gets its OWN SealedEvaluator instance, so its Deflated Sharpe Ratio is deflated by the number
  of trials *that method* actually ran (search screens + search ensemble evaluations + validation looks).
- `--dev` never touches `sealed_test` and never writes to the experiment registry (design iteration happens
  on search/validation only). The final run (no `--dev`) scores every frozen champion on `sealed_test` once.
- Frozen-frontier proxies and MAVE-Q v1 sealed-test numbers are NOT re-scored (their hashes are already in
  the append-only ledger); the report pulls them from `phase5_frontier_combat`. Their *validation* returns are
  recomputed here (validation is not ledgered) only to run paired block-bootstrap tests.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from vn_rsi.data import SnapshotManager, load_market_rules
from vn_rsi.eval import SealedEvaluator
from vn_rsi.eval.metrics import paired_bootstrap_ci
from vn_rsi.models.alpha_mining_baselines import (
    Alpha101StyleLibrary,
    CompetitorResult,
    GPAlphaMiner,
    RandomDSLSearch,
)
from vn_rsi.models.frontier_arena import FrontierChallengerRunner, FrontierChallengerSpec
from vn_rsi.models.maveq_v2 import MAVEQv2Engine
from vn_rsi.utils import (
    RunLogger,
    compute_config_hash,
    get_git_commit_hash,
    load_yaml,
    set_global_seed,
)

RUN_ID = "phase6_maveq_v2_combat"


def _competitor_summary(res: CompetitorResult) -> dict[str, Any]:
    d = asdict(res)
    d.pop("validation_daily_returns", None)
    return d


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6: MAVE-Q v2 vs same-topic alpha miners")
    parser.add_argument("--config", default="configs/maveq_v2.yaml")
    parser.add_argument("--frontier-config", default="configs/frontier_combat.yaml")
    parser.add_argument("--sandbox-config", default="configs/sandbox.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--market-rules", default="configs/market_rules.yaml")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--preset", default=None)
    parser.add_argument("--dev", action="store_true", help="validation-only design iteration (no sealed test)")
    parser.add_argument("--skip-competitors", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_yaml(repo_root / args.config)
    frontier_cfg = load_yaml(repo_root / args.frontier_config)
    sandbox_cfg = load_yaml(repo_root / args.sandbox_config)
    data_cfg = load_yaml(repo_root / args.data_config)
    rules = load_market_rules(repo_root / args.market_rules)
    seed = int(cfg.get("seed", 2026))
    set_global_seed(seed)

    preset = args.preset or str(cfg.get("specialist_preset", "auto"))
    if preset == "auto":
        try:
            import torch

            preset = "kaggle_max_100m" if torch.cuda.is_available() else "cpu_fast"
        except Exception:
            preset = "cpu_fast"

    sandbox_cfg["ledger_path"] = str(repo_root / str(sandbox_cfg.get("ledger_path", "logs/sealed_test_ledger.jsonl")))
    snapshot = SnapshotManager(repo_root / str(data_cfg.get("snapshot_root", "data/snapshots"))).load_snapshot(
        str(data_cfg["snapshot_id"])
    )
    protected = [
        repo_root / "src" / "vn_rsi" / "sandbox",
        repo_root / "src" / "vn_rsi" / "eval",
        repo_root / args.sandbox_config,
        repo_root / args.market_rules,
    ]

    def new_evaluator() -> SealedEvaluator:
        return SealedEvaluator(snapshot, rules, sandbox_cfg, protected_paths=protected)

    score_sealed = not args.dev
    timings: dict[str, float] = {}

    # 1. MAVE-Q v2
    t0 = time.time()
    ev_v2 = new_evaluator()
    v2 = MAVEQv2Engine(ev_v2, cfg, preset=preset).run(run_id=RUN_ID, score_sealed=score_sealed)
    timings["maveq_v2"] = round(time.time() - t0, 1)

    # 2. Budget-matched same-topic competitors
    competitors: dict[str, CompetitorResult] = {}
    if not args.skip_competitors:
        ccfg = cfg.get("competitors", {})
        t0 = time.time()
        g = ccfg.get("gp_alpha_miner", {})
        competitors["gp_alpha_miner"] = GPAlphaMiner(
            new_evaluator(), population=int(g.get("population", 32)), generations=int(g.get("generations", 8)),
            pool_size=int(g.get("pool_size", 5)), seed=seed,
        ).run(run_id=RUN_ID, score_sealed=score_sealed)
        timings["gp_alpha_miner"] = round(time.time() - t0, 1)
        t0 = time.time()
        rr = ccfg.get("random_dsl_search", {})
        competitors["random_dsl_search"] = RandomDSLSearch(
            new_evaluator(), budget=int(rr.get("budget", 256)), checkpoint_every=int(rr.get("checkpoint_every", 32)),
            pool_size=int(rr.get("pool_size", 5)), seed=seed,
        ).run(run_id=RUN_ID, score_sealed=score_sealed)
        timings["random_dsl_search"] = round(time.time() - t0, 1)
        competitors["alpha101_style_library"] = Alpha101StyleLibrary(new_evaluator()).run(
            run_id=RUN_ID, score_sealed=score_sealed
        )

    # 3. Frontier proxies: validation-only re-evaluation for paired bootstrap (never re-touch sealed test)
    ev_front = new_evaluator()
    frontier_val: dict[str, dict[str, Any]] = {}
    frontier_val_returns = {}
    for ch in frontier_cfg.get("frontier_challengers", []):
        spec = FrontierChallengerSpec(
            name=str(ch["name"]), parameter_count_billions=float(ch["parameter_count_billions"]),
            api_env_var=str(ch.get("api_env_var", "")), endpoint_url=str(ch.get("endpoint_url", "")),
            api_model_id=str(ch.get("api_model_id", "")), prior_strength=float(ch.get("prior_strength", 0.8)),
        )
        runner = FrontierChallengerRunner(spec, ev_front)
        specs = runner._frozen_frontier_textbook_factors()
        weights = [max(0.05, ev_front.evaluate_factor(s, zone="search", run_id=RUN_ID).metrics["rank_ic"] + 0.02)
                   for s in specs]
        out = ev_front.evaluate_factor_ensemble(specs, weights, zone="validation",
                                                candidate_id=f"frontier_{spec.name}_val", run_id=RUN_ID)
        frontier_val[spec.name] = {"mode": "offline_frontier_policy_sim", **out.metrics}
        frontier_val_returns[spec.name] = out.daily_net_returns

    # 4. Paired block-bootstrap on validation: MAVE-Q v2 vs every rival
    bootstrap: dict[str, dict[str, float]] = {}
    for name, rets in frontier_val_returns.items():
        bootstrap[name] = paired_bootstrap_ci(v2.validation_daily_returns, rets, n_boot=1000, seed=seed)
    for name, res in competitors.items():
        bootstrap[name] = paired_bootstrap_ci(v2.validation_daily_returns, res.validation_daily_returns,
                                              n_boot=1000, seed=seed)

    v2_summary = asdict(v2)
    v2_summary.pop("validation_daily_returns", None)
    summary = {
        "preset": preset,
        "mode": "dev_validation_only" if args.dev else "final_single_shot_sealed_test",
        "maveq_v2": v2_summary,
        "competitors": {k: _competitor_summary(v) for k, v in competitors.items()},
        "frontier_proxy_validation_recomputed": frontier_val,
        "validation_bootstrap_maveq_v2_vs": bootstrap,
        "timings_sec": timings,
        "is_synthetic_data": snapshot.is_synthetic,
    }

    if args.dev:
        out_dir = repo_root / args.logs_dir / "dev"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{RUN_ID}_dev_{int(time.time())}.json"
        out_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    else:
        logger = RunLogger(
            run_id=RUN_ID,
            logs_dir=repo_root / args.logs_dir,
            git_commit=get_git_commit_hash(repo_root),
            config_hash=compute_config_hash({"cfg": cfg, "preset": preset})[:16],
            evaluator_seal_hash=ev_v2.seal_hash,
            metadata={"phase": 6, "preset": preset, "is_synthetic_data": snapshot.is_synthetic},
        )
        logger.finalize_summary(summary)

    brief = {
        "status": "OK",
        "mode": summary["mode"],
        "maveq_v2": {
            "val": {k: v2.champion_validation_metrics[k] for k in ("rank_ic", "net_sharpe", "daily_turnover", "deflated_sharpe_prob")},
            "search": {k: v2.champion_search_metrics[k] for k in ("rank_ic", "net_sharpe", "daily_turnover")},
            "test": v2.champion_sealed_test_metrics,
            "expr": v2.champion_expressions,
            "weights": v2.champion_weights,
            "round": v2.champion_round,
            "trials": v2.total_evaluator_trials,
        },
        "competitors_val": {k: {m: v.validation_metrics[m] for m in ("rank_ic", "net_sharpe", "daily_turnover")}
                            for k, v in competitors.items()},
        "competitors_test": {k: v.sealed_test_metrics for k, v in competitors.items()},
        "frontier_val": {k: {m: v[m] for m in ("rank_ic", "net_sharpe")} for k, v in frontier_val.items()},
        "bootstrap_p": {k: v["p_one_sided"] for k, v in bootstrap.items()},
        "rounds": [(r.round_idx, r.ensemble_size, r.search_objective, r.val_net_sharpe, r.val_rank_ic, r.archive_coverage)
                   for r in v2.round_logs],
        "timings": timings,
    }
    print(json.dumps(brief, indent=1, default=str))


if __name__ == "__main__":
    main()
