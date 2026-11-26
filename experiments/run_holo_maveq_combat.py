"""Phase 7 CLI: Holo-MAVE-Q (v3) vs 7 Same-Topic Alpha Miners, MAVE-Q v1/v2, and 70B-671B Frontier Models.

Strict Sealed-Sandbox Protocol:
- Every evaluated method gets its OWN fresh SealedEvaluator instance so Deflated Sharpe Ratio (DSR)
  is deflated strictly by that method's exact trial count.
- `--dev` runs strictly on `search` + scalar `validation` view (NEVER touches `sealed_test` or registry).
- Final mode (without `--dev`) freezes all champions and scores each on `sealed_test` ONCE.
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
    AlphaGPTIterativeMiner,
    CompetitorResult,
    GPAlphaMiner,
    MAPElitesAlphaMiner,
    MCTSAlphaMiner,
    PPOAlphaMiner,
    RandomDSLSearch,
)
from vn_rsi.models.cloud_presets import detect_cloud_compute
from vn_rsi.models.frontier_arena import FrontierChallengerRunner, FrontierChallengerSpec
from vn_rsi.models.holo_maveq import HoloMAVEQEngine
from vn_rsi.models.maveq_v2 import MAVEQv2Engine
from vn_rsi.utils import (
    RunLogger,
    compute_config_hash,
    get_git_commit_hash,
    load_yaml,
    set_global_seed,
)

RUN_ID = "phase7_holo_maveq_combat"


def _strip_returns(d: dict[str, Any]) -> dict[str, Any]:
    out = dict(d)
    out.pop("validation_daily_returns", None)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Holo-MAVE-Q v3 Unified Same-Topic & Frontier Arena")
    parser.add_argument("--config", default="configs/holo_maveq.yaml")
    parser.add_argument("--v2-config", default="configs/maveq_v2.yaml")
    parser.add_argument("--frontier-config", default="configs/frontier_combat.yaml")
    parser.add_argument("--sandbox-config", default="configs/sandbox.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--market-rules", default="configs/market_rules.yaml")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--preset", default=None)
    parser.add_argument("--dev", action="store_true", help="Validation-only dev run (never touches sealed_test)")
    parser.add_argument("--skip-competitors", action="store_true")
    parser.add_argument("--skip-v2", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_yaml(repo_root / args.config)
    v2_cfg = load_yaml(repo_root / args.v2_config)
    frontier_cfg = load_yaml(repo_root / args.frontier_config)
    sandbox_cfg = load_yaml(repo_root / args.sandbox_config)
    data_cfg = load_yaml(repo_root / args.data_config)
    rules = load_market_rules(repo_root / args.market_rules)
    seed = int(cfg.get("seed", 2026))
    set_global_seed(seed)

    preset = args.preset or str(cfg.get("specialist_preset", "auto"))
    if preset == "auto":
        preset = str(detect_cloud_compute()["recommended_preset"])

    sandbox_cfg["ledger_path"] = str(
        repo_root / str(sandbox_cfg.get("ledger_path", "logs/sealed_test_ledger.jsonl"))
    )
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

    # 1. Holo-MAVE-Q (v3 Flagship)
    t0 = time.time()
    ev_v3 = new_evaluator()
    v3 = HoloMAVEQEngine(ev_v3, cfg, preset=preset).run(run_id=RUN_ID, score_sealed=score_sealed)
    timings["holo_maveq_v3"] = round(time.time() - t0, 2)

    # 2. MAVE-Q v2 (Intermediate QD + Greedy PLCA Baseline)
    v2_summary: dict[str, Any] | None = None
    v2_val_rets = None
    if not args.skip_v2:
        t0 = time.time()
        v2_res = MAVEQv2Engine(new_evaluator(), v2_cfg, preset=preset).run(
            run_id=f"{RUN_ID}_v2", score_sealed=score_sealed
        )
        timings["maveq_v2"] = round(time.time() - t0, 2)
        v2_val_rets = v2_res.validation_daily_returns
        v2_summary = _strip_returns(asdict(v2_res))

    # 3. 7 Same-Topic Automated Alpha-Mining Competitors
    competitors: dict[str, CompetitorResult] = {}
    if not args.skip_competitors:
        ccfg = cfg.get("competitors", {})

        t0 = time.time()
        competitors["alpha101_style_library"] = Alpha101StyleLibrary(new_evaluator()).run(
            run_id=RUN_ID, score_sealed=score_sealed
        )
        timings["alpha101_style_library"] = round(time.time() - t0, 2)

        t0 = time.time()
        rc = ccfg.get("random_dsl_search", {})
        competitors["random_dsl_search"] = RandomDSLSearch(
            new_evaluator(),
            budget=int(rc.get("budget", 192)),
            checkpoint_every=int(rc.get("checkpoint_every", 24)),
            pool_size=int(rc.get("pool_size", 5)),
            seed=seed,
        ).run(run_id=RUN_ID, score_sealed=score_sealed)
        timings["random_dsl_search"] = round(time.time() - t0, 2)

        t0 = time.time()
        gc = ccfg.get("gp_alpha_miner", {})
        competitors["gp_alpha_miner"] = GPAlphaMiner(
            new_evaluator(),
            population=int(gc.get("population", 24)),
            generations=int(gc.get("generations", 8)),
            pool_size=int(gc.get("pool_size", 5)),
            seed=seed,
        ).run(run_id=RUN_ID, score_sealed=score_sealed)
        timings["gp_alpha_miner"] = round(time.time() - t0, 2)

        t0 = time.time()
        pc = ccfg.get("ppo_alpha_miner", {})
        competitors["ppo_alpha_miner"] = PPOAlphaMiner(
            new_evaluator(),
            episodes=int(pc.get("episodes", 8)),
            batch_size=int(pc.get("batch_size", 24)),
            ppo_epochs=int(pc.get("ppo_epochs", 4)),
            pool_size=int(pc.get("pool_size", 5)),
            seed=seed,
        ).run(run_id=RUN_ID, score_sealed=score_sealed)
        timings["ppo_alpha_miner"] = round(time.time() - t0, 2)

        t0 = time.time()
        mc = ccfg.get("mcts_alpha_miner", {})
        competitors["mcts_alpha_miner"] = MCTSAlphaMiner(
            new_evaluator(),
            budget=int(mc.get("budget", 192)),
            c_puct=float(mc.get("c_puct", 1.25)),
            checkpoint_every=int(mc.get("checkpoint_every", 24)),
            pool_size=int(mc.get("pool_size", 5)),
            seed=seed,
        ).run(run_id=RUN_ID, score_sealed=score_sealed)
        timings["mcts_alpha_miner"] = round(time.time() - t0, 2)

        t0 = time.time()
        qc = ccfg.get("map_elites_alpha_miner", {})
        competitors["map_elites_alpha_miner"] = MAPElitesAlphaMiner(
            new_evaluator(),
            budget=int(qc.get("budget", 192)),
            init_random=int(qc.get("init_random", 36)),
            checkpoint_every=int(qc.get("checkpoint_every", 24)),
            pool_size=int(qc.get("pool_size", 5)),
            seed=seed,
        ).run(run_id=RUN_ID, score_sealed=score_sealed)
        timings["map_elites_alpha_miner"] = round(time.time() - t0, 2)

        t0 = time.time()
        ac = ccfg.get("alphagpt_iterative_miner", {})
        competitors["alphagpt_iterative_miner"] = AlphaGPTIterativeMiner(
            new_evaluator(),
            rounds=int(ac.get("rounds", 8)),
            ideas_per_round=int(ac.get("ideas_per_round", 12)),
            pool_size=int(ac.get("pool_size", 5)),
            seed=seed,
        ).run(run_id=RUN_ID, score_sealed=score_sealed)
        timings["alphagpt_iterative_miner"] = round(time.time() - t0, 2)

    # 4. Frontier 70B-671B Challengers (validation returns for paired bootstrap; sealed_test read from phase5)
    ev_front = new_evaluator()
    frontier_val: dict[str, dict[str, Any]] = {}
    frontier_val_returns: dict[str, Any] = {}
    for ch in frontier_cfg.get("frontier_challengers", []):
        spec = FrontierChallengerSpec(
            name=str(ch["name"]),
            parameter_count_billions=float(ch["parameter_count_billions"]),
            api_env_var=str(ch.get("api_env_var", "")),
            endpoint_url=str(ch.get("endpoint_url", "")),
            api_model_id=str(ch.get("api_model_id", "")),
            prior_strength=float(ch.get("prior_strength", 0.8)),
        )
        runner = FrontierChallengerRunner(spec, ev_front)
        specs = runner._frozen_frontier_textbook_factors()
        weights = [
            max(0.05, ev_front.evaluate_factor(s, zone="search", run_id=RUN_ID).metrics["rank_ic"] + 0.02)
            for s in specs
        ]
        out = ev_front.evaluate_factor_ensemble(
            specs, weights, zone="validation", candidate_id=f"frontier_{spec.name}_val", run_id=RUN_ID
        )
        frontier_val[spec.name] = {"mode": runner.mode, **out.metrics}
        frontier_val_returns[spec.name] = out.daily_net_returns

    # 5. Paired block-bootstrap on validation: Holo-MAVE-Q v3 vs every rival
    bootstrap: dict[str, dict[str, float]] = {}
    for name, rets in frontier_val_returns.items():
        bootstrap[name] = paired_bootstrap_ci(v3.validation_daily_returns, rets, n_boot=1000, seed=seed)
    if v2_val_rets is not None:
        bootstrap["maveq_v2"] = paired_bootstrap_ci(v3.validation_daily_returns, v2_val_rets, n_boot=1000, seed=seed)
    for name, res in competitors.items():
        bootstrap[name] = paired_bootstrap_ci(
            v3.validation_daily_returns, res.validation_daily_returns, n_boot=1000, seed=seed
        )

    summary: dict[str, Any] = {
        "preset": preset,
        "mode": "dev_validation_only" if args.dev else "final_single_shot_sealed_test",
        "holo_maveq_v3": _strip_returns(asdict(v3)),
        "maveq_v2": v2_summary,
        "competitors": {k: _strip_returns(asdict(v)) for k, v in competitors.items()},
        "frontier_proxy_validation_recomputed": frontier_val,
        "validation_bootstrap_holo_v3_vs": bootstrap,
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
            evaluator_seal_hash=ev_v3.seal_hash,
            metadata={"phase": 7, "preset": preset, "is_synthetic_data": snapshot.is_synthetic},
        )
        logger.finalize_summary(summary)

    brief = {
        "status": "OK",
        "mode": summary["mode"],
        "holo_maveq_v3": {
            "params": v3.specialist_total_params,
            "trials": v3.total_evaluator_trials,
            "val": {
                k: v3.champion_validation_metrics[k]
                for k in ("rank_ic", "net_sharpe", "daily_turnover", "deflated_sharpe_prob")
            },
            "test": v3.champion_sealed_test_metrics,
            "expr": v3.champion_expressions,
            "weights": v3.champion_weights,
            "niches": v3.champion_niches,
        },
        "maveq_v2_val": (
            {
                k: v2_summary["champion_validation_metrics"][k]
                for k in ("rank_ic", "net_sharpe", "daily_turnover", "deflated_sharpe_prob")
            }
            if v2_summary
            else None
        ),
        "maveq_v2_test": v2_summary["champion_sealed_test_metrics"] if v2_summary else None,
        "competitors_val": {
            k: {
                m: v.validation_metrics[m]
                for m in ("rank_ic", "net_sharpe", "daily_turnover", "deflated_sharpe_prob")
            }
            for k, v in competitors.items()
        },
        "competitors_test": {k: v.sealed_test_metrics for k, v in competitors.items()},
        "frontier_val": {k: {m: v[m] for m in ("rank_ic", "net_sharpe")} for k, v in frontier_val.items()},
        "bootstrap_p": {k: v["p_one_sided"] for k, v in bootstrap.items()},
        "timings": timings,
    }
    print(json.dumps(brief, indent=2, default=str))


if __name__ == "__main__":
    main()
