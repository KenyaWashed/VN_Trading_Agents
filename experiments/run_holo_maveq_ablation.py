"""Phase 7b CLI: Holo-MAVE-Q v3 Architectural Ablation Study across multiple seeds (VALIDATION ONLY).

Evaluates each mechanism's contribution without ever touching `sealed_test`:
- `full_holo_maveq_v3`: Full architecture (TypedASTGuard + TreeDiffusion + HoloHRM + MSSF + SRSS)
- `no_typed_guard`:     Disable Physical-Exponent Dimensional & Algebraic Guard
- `no_tree_diffusion`:  Disable Topological Tree-Coordinate Conditioning in Diffusion
- `no_holo_hrm`:        Disable Holographic Compositional Tree-Recursive Surrogate & Attribution Infilling
- `no_mssf`:            Disable Multi-Scale Causal State-Space Filter Bank (raw w=1 only)
- `no_srss`:            Disable Regime-Invariant Spectral Risk-Parity Shrinkage Solver (equal weights)
Plus multi-seed comparisons against Same-Topic SOTA Miners (GP, PPO, MCTS, MAP-Elites, Alpha-GPT, Random).
"""

from __future__ import annotations

import argparse
import copy
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RUN_ID = "phase7_holo_maveq_ablation"
KEYS = ("rank_ic", "net_sharpe", "daily_turnover", "deflated_sharpe_prob")


def _variant_cfg(base: dict[str, Any], variant: str, seed: int) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    cfg["seed"] = seed
    ab = cfg.setdefault("ablations", {})
    if variant == "no_typed_guard":
        ab["typed_guard"] = False
    elif variant == "no_tree_diffusion":
        ab["tree_diffusion"] = False
    elif variant == "no_holo_hrm":
        ab["holo_hrm"] = False
    elif variant == "no_mssf":
        ab["mssf_smoothing"] = False
    elif variant == "no_srss":
        ab["srss_spectral_allocation"] = False
    return cfg


def _job(args: tuple[str, int]) -> dict[str, Any]:
    variant, seed = args
    from vn_rsi.data import SnapshotManager, load_market_rules
    from vn_rsi.eval import SealedEvaluator
    from vn_rsi.models.alpha_mining_baselines import (
        AlphaGPTIterativeMiner,
        GPAlphaMiner,
        MAPElitesAlphaMiner,
        MCTSAlphaMiner,
        PPOAlphaMiner,
        RandomDSLSearch,
    )
    from vn_rsi.models.holo_maveq import HoloMAVEQEngine
    from vn_rsi.utils import load_yaml

    sandbox_cfg = load_yaml(REPO / "configs/sandbox.yaml")
    sandbox_cfg["ledger_path"] = str(REPO / "logs" / "ablation_no_sealed_access.jsonl")
    data_cfg = load_yaml(REPO / "configs/data.yaml")
    rules = load_market_rules(REPO / "configs/market_rules.yaml")
    snap = SnapshotManager(REPO / "data/snapshots").load_snapshot(str(data_cfg["snapshot_id"]))
    ev = SealedEvaluator(snap, rules, sandbox_cfg)
    base = load_yaml(REPO / "configs/holo_maveq.yaml")

    if variant == "random_dsl_search":
        r = RandomDSLSearch(ev, budget=192, checkpoint_every=24, seed=seed).run(RUN_ID)
        vm, trials = r.validation_metrics, r.total_evaluator_trials
    elif variant == "gp_alpha_miner":
        r = GPAlphaMiner(ev, population=24, generations=8, seed=seed).run(RUN_ID)
        vm, trials = r.validation_metrics, r.total_evaluator_trials
    elif variant == "ppo_alpha_miner":
        r = PPOAlphaMiner(ev, episodes=8, batch_size=24, seed=seed).run(RUN_ID)
        vm, trials = r.validation_metrics, r.total_evaluator_trials
    elif variant == "mcts_alpha_miner":
        r = MCTSAlphaMiner(ev, budget=192, checkpoint_every=24, seed=seed).run(RUN_ID)
        vm, trials = r.validation_metrics, r.total_evaluator_trials
    elif variant == "map_elites_alpha_miner":
        r = MAPElitesAlphaMiner(ev, budget=192, checkpoint_every=24, seed=seed).run(RUN_ID)
        vm, trials = r.validation_metrics, r.total_evaluator_trials
    elif variant == "alphagpt_iterative_miner":
        r = AlphaGPTIterativeMiner(ev, rounds=8, ideas_per_round=12, seed=seed).run(RUN_ID)
        vm, trials = r.validation_metrics, r.total_evaluator_trials
    else:
        r_holo = HoloMAVEQEngine(ev, _variant_cfg(base, variant, seed)).run(RUN_ID, score_sealed=False)
        vm, trials = r_holo.champion_validation_metrics, r_holo.total_evaluator_trials

    return {"variant": variant, "seed": seed, "trials": int(trials), **{k: float(vm[k]) for k in KEYS}}


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed Holo-MAVE-Q v3 mechanism ablations (validation only)")
    parser.add_argument("--seeds", default="11,22,33,2026")
    parser.add_argument(
        "--variants",
        default="full_holo_maveq_v3,no_typed_guard,no_tree_diffusion,no_holo_hrm,no_mssf,no_srss,gp_alpha_miner,ppo_alpha_miner,mcts_alpha_miner,map_elites_alpha_miner,alphagpt_iterative_miner,random_dsl_search",
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--dev", action="store_true", help="Do not write to main experiment_registry.jsonl")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    jobs = [(v, s) for v in variants for s in seeds]

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(_job, jobs))

    agg: dict[str, dict[str, float]] = {}
    for v in variants:
        rs = [r for r in rows if r["variant"] == v]
        agg[v] = {
            "n_seeds": len(rs),
            "trials_mean": round(float(np.mean([r["trials"] for r in rs])), 1),
        }
        for k in KEYS:
            vals = np.asarray([r[k] for r in rs], dtype=float)
            agg[v][f"{k}_mean"] = round(float(vals.mean()), 4)
            agg[v][f"{k}_std"] = round(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, 4)

    if not args.dev:
        from vn_rsi.utils import RunLogger, compute_config_hash, get_git_commit_hash

        logger = RunLogger(
            run_id=RUN_ID,
            logs_dir=REPO / "logs",
            git_commit=get_git_commit_hash(REPO),
            config_hash=compute_config_hash({"variants": variants, "seeds": seeds})[:16],
            metadata={"phase": "7b", "zone": "validation_only", "seeds": seeds},
        )
        logger.finalize_summary({"rows": rows, "aggregate": agg})

    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
