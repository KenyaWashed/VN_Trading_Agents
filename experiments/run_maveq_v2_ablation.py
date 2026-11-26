"""Phase 6b CLI: MAVE-Q v2 mechanism ablations x seeds on VALIDATION only (never touches sealed_test).

Variants: full | no_dcg (Dimensional-Consistency Guard off) | no_cftc (Cross-Fold Transfer Calibration replaced
by its static shrinkage prior + raw search Sharpe objective) | no_fats (no Friction-Aware Temporal Smoothing),
plus budget-matched Random DSL Search and GP Alpha Miner on the same seeds.
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
RUN_ID = "phase6_maveq_v2_ablation"
KEYS = ("rank_ic", "net_sharpe", "daily_turnover")


def _variant_cfg(base: dict[str, Any], variant: str, seed: int) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    cfg["seed"] = seed
    if variant == "no_dcg":
        cfg["dimensional_guard"] = False
    elif variant == "no_cftc":
        cfg.setdefault("transfer_calibration", {})["min_rows"] = 10**9
        cfg.setdefault("objective", {}).update({"transfer_weight": 0.0, "full_sharpe_weight": 1.0})
    elif variant == "no_fats":
        cfg.setdefault("loop", {})["smoothing_grid"] = [1]
    return cfg


def _job(args: tuple[str, int]) -> dict[str, Any]:
    variant, seed = args
    from vn_rsi.data import SnapshotManager, load_market_rules
    from vn_rsi.eval import SealedEvaluator
    from vn_rsi.models.alpha_mining_baselines import GPAlphaMiner, RandomDSLSearch
    from vn_rsi.models.maveq_v2 import MAVEQv2Engine
    from vn_rsi.utils import load_yaml

    sandbox_cfg = load_yaml(REPO / "configs/sandbox.yaml")
    sandbox_cfg["ledger_path"] = str(REPO / "logs" / "ablation_no_sealed_access.jsonl")  # never written: val only
    data_cfg = load_yaml(REPO / "configs/data.yaml")
    rules = load_market_rules(REPO / "configs/market_rules.yaml")
    snap = SnapshotManager(REPO / "data/snapshots").load_snapshot(str(data_cfg["snapshot_id"]))
    ev = SealedEvaluator(snap, rules, sandbox_cfg)
    base = load_yaml(REPO / "configs/maveq_v2.yaml")
    if variant == "random_dsl_search":
        r = RandomDSLSearch(ev, budget=256, checkpoint_every=32, seed=seed).run(RUN_ID)
        vm, trials = r.validation_metrics, r.total_evaluator_trials
    elif variant == "gp_alpha_miner":
        r2 = GPAlphaMiner(ev, population=32, generations=8, seed=seed).run(RUN_ID)
        vm, trials = r2.validation_metrics, r2.total_evaluator_trials
    else:
        r3 = MAVEQv2Engine(ev, _variant_cfg(base, variant, seed)).run(RUN_ID, score_sealed=False)
        vm, trials = r3.champion_validation_metrics, r3.total_evaluator_trials
    return {"variant": variant, "seed": seed, "trials": int(trials), **{k: float(vm[k]) for k in KEYS}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="11,22,33")
    parser.add_argument("--variants", default="full,no_dcg,no_cftc,no_fats,random_dsl_search,gp_alpha_miner")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    variants = args.variants.split(",")
    jobs = [(v, s) for v in variants for s in seeds]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(_job, jobs))

    agg: dict[str, dict[str, float]] = {}
    for v in variants:
        rs = [r for r in rows if r["variant"] == v]
        agg[v] = {"n_seeds": len(rs)}
        for k in KEYS:
            vals = np.asarray([r[k] for r in rs])
            agg[v][f"{k}_mean"] = round(float(vals.mean()), 5)
            agg[v][f"{k}_std"] = round(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, 5)

    from vn_rsi.utils import RunLogger, get_git_commit_hash

    logger = RunLogger(run_id=RUN_ID, logs_dir=REPO / "logs", git_commit=get_git_commit_hash(REPO),
                       metadata={"phase": "6b", "zone": "validation_only", "seeds": seeds})
    logger.finalize_summary({"rows": rows, "aggregate": agg})
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
