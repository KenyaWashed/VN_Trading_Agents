"""Task 05 (Phase 11) CLI: Ruthless 8-Seed Pruning & Ablation Audit + Multi-Market OOD Generalization for MAVE-Q v5.

STRICT VALIDATION/OOD PROTOCOL:
- NEVER touches main VN30 `sealed_test` or `logs/sealed_test_ledger.jsonl`.
- Part 1: 8-seed paired ablation & scaling audit (`11,22,33,44,55,66,77,2026`) on `vn_equities_snapshot_v1`:
    - `full_maveq_v5` (active config from `configs/maveq_v5.yaml`)
    - `no_dca_backbone` (`preln_transformer` instead of Google `DCA-QuantFormer`)
    - `no_chronos_kalman_hedge` (static prior weights instead of causal H-day delayed online Kalman-Hedge)
    - `no_mose_hypergate` (removing Group-Lasso nonlinear MoSE HyperGate)
    - `no_conformal_veto` (removing Split-Conformal Dormant Veto & Epistemic Abstention)
    - `no_asym_beta_ntz` (removing AsymBeta-NTZ controller)
    - `scale_5m` & `scale_20m` (direct 8-seed scaling comparison vs 0.5M Occam reference)
  Computes Paired Seed-Bootstrap 95% CIs + one-sided p-values (Sharpe, Calmar, IC, Tail-CVaR, DSR) and
  Paired Daily Block-Bootstrap 95% CIs. Flags any module whose removal improves or leaves performance unchanged
  (`p > 0.10`) for honest Popperian pruning!
- Part 2: Out-of-Distribution (OOD) generalization across `VN100_10d`, `US_LIQUID_T1`, and `CSI_EMERGING_T1`
  benchmarked against Buy & Hold, Equal Weight, Ridge, 5 Global SOTA rivals, and MAVE-Q v4.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
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
from vn_rsi.models.maveq_v5_chronos import MAVEQv5Engine
from vn_rsi.utils import RunLogger, compute_config_hash, get_git_commit_hash, load_yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "phase11_v5_ablation_and_ood"

ABLATION_VARIANTS: tuple[str, ...] = (
    "full_maveq_v5",
    "no_dca_backbone",
    "no_chronos_kalman_hedge",
    "no_mose_hypergate",
    "no_conformal_veto",
    "no_asym_beta_ntz",
    "scale_5m",
    "scale_20m",
)


def _ablation_cfg(base_cfg: dict[str, Any], variant: str, seed: int) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg["seed"] = seed
    ab = cfg.setdefault("ablations", {})
    if variant == "no_dca_backbone":
        ab["use_dca_backbone"] = False
    elif variant == "no_chronos_kalman_hedge":
        ab["use_chronos_kalman_hedge"] = False
    elif variant == "no_mose_hypergate":
        ab["use_mose_hypergate"] = False
    elif variant == "no_conformal_veto":
        ab["use_conformal_veto"] = False
    elif variant == "no_asym_beta_ntz":
        ab["use_asym_beta_ntz"] = False
    elif variant == "scale_5m":
        cfg["backbone_zoo"]["scale_rung"] = "5m"
    elif variant == "scale_20m":
        cfg["backbone_zoo"]["scale_rung"] = "20m"
    return cfg


def _run_ablation_job(args: tuple[str, int, str, str]) -> dict[str, Any]:
    variant, seed, preset, config_rel = args
    rules = load_market_rules(REPO_ROOT / "configs/market_rules.yaml")
    sandbox_cfg = copy.deepcopy(load_yaml(REPO_ROOT / "configs/sandbox.yaml"))
    sandbox_cfg["ledger_path"] = str(REPO_ROOT / "logs/dev/v5_ablation_unused_ledger.jsonl")
    data_cfg = load_yaml(REPO_ROOT / "configs/data.yaml")
    snap = SnapshotManager(REPO_ROOT / str(data_cfg.get("snapshot_root", "data/snapshots"))).load_snapshot(
        str(data_cfg["snapshot_id"])
    )
    ev = SealedEvaluator(snap, rules, sandbox_cfg)
    base_cfg = load_yaml(REPO_ROOT / config_rel)
    cfg = _ablation_cfg(base_cfg, variant, seed)

    t0 = time.time()
    res = MAVEQv5Engine(ev, cfg, preset=preset).run(run_id=f"{RUN_ID}_{variant}_s{seed}", score_sealed=False)
    ext = res.champion_validation_extended
    gc.collect()
    return {
        "variant": variant,
        "seed": seed,
        "elapsed_sec": round(time.time() - t0, 2),
        "backbone_family": res.backbone_family,
        "scale_rung": res.scale_rung,
        "backbone_exact_params": int(res.backbone_exact_params),
        "selected_controller": res.selected_controller,
        "trials": int(res.total_evaluator_trials),
        "val_queries": int(res.validation_queries_used),
        "dormant_veto_fraction": float(res.dormant_veto_fraction),
        "epistemic_abstain_rate": float(res.epistemic_abstain_rate),
        "rank_ic": float(ext["rank_ic"]),
        "icir": float(ext["icir"]),
        "net_sharpe": float(ext["net_sharpe"]),
        "net_sortino": float(ext["net_sortino"]),
        "calmar": float(ext["calmar_ratio"]),
        "ann_net_return": float(ext["ann_net_return"]),
        "max_drawdown": float(ext["max_drawdown"]),
        "cvar_95": float(ext.get("cvar_95", 0.0)),
        "daily_turnover": float(ext["daily_turnover"]),
        "capm_beta": float(ext.get("capm_beta", 0.0)),
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
    config_rel: str,
) -> dict[str, Any]:
    """Run OOD validation benchmark comparing MAVE-Q v5 vs Buy & Hold, 5 Global SOTA rivals, and MAVE-Q v4."""
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
    sandbox_cfg["ledger_path"] = str(REPO_ROOT / f"logs/dev/v5_ood_{label.lower()}_unused_ledger.jsonl")
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

    # 1. Baselines
    b_outs = run_all_baselines(new_ev(), zone="validation", run_id=f"v5_ood_{label}_bm", score_sealed_test=False)
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
    g_outs = run_global_competitor_arena(new_ev, run_id=f"v5_ood_{label}_global", score_sealed=False, seed=seed)
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

    # 3. MAVE-Q v4 (reference)
    v4_cfg = copy.deepcopy(load_yaml(REPO_ROOT / "configs/maveq_v4.yaml"))
    v4_cfg["seed"] = seed
    v4_res = MAVEQv4Engine(new_ev(), v4_cfg, preset="cpu_fast").run(run_id=f"v5_ood_{label}_v4", score_sealed=False)
    v4ext = v4_res.champion_validation_extended
    results["MAVE-Q_v4"] = {
        "trials": v4_res.total_evaluator_trials,
        "selected_controller": v4_res.selected_controller,
        "rank_ic": v4ext["rank_ic"],
        "net_sharpe": v4ext["net_sharpe"],
        "net_sortino": v4ext["net_sortino"],
        "calmar": v4ext["calmar_ratio"],
        "ann_net_return": v4ext["ann_net_return"],
        "max_drawdown": v4ext["max_drawdown"],
        "daily_turnover": v4ext["daily_turnover"],
        "dsr": v4ext["deflated_sharpe_prob"],
    }

    # 4. MAVE-Q v5 (Frozen config — zero market-specific retuning!)
    v5_cfg = copy.deepcopy(load_yaml(REPO_ROOT / config_rel))
    v5_cfg["seed"] = seed
    v5_cfg["chronos_kalman_hedge"]["horizon_delay_days"] = horizon_days
    v5_res = MAVEQv5Engine(new_ev(), v5_cfg, preset=preset).run(run_id=f"v5_ood_{label}_v5", score_sealed=False)
    v5ext = v5_res.champion_validation_extended
    results["MAVE-Q_v5"] = {
        "trials": v5_res.total_evaluator_trials,
        "selected_controller": v5_res.selected_controller,
        "rank_ic": v5ext["rank_ic"],
        "net_sharpe": v5ext["net_sharpe"],
        "net_sortino": v5ext["net_sortino"],
        "calmar": v5ext["calmar_ratio"],
        "ann_net_return": v5ext["ann_net_return"],
        "max_drawdown": v5ext["max_drawdown"],
        "daily_turnover": v5ext["daily_turnover"],
        "dsr": v5ext["deflated_sharpe_prob"],
        "epistemic_abstain_rate": v5_res.epistemic_abstain_rate,
    }

    gc.collect()
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
    parser = argparse.ArgumentParser(description="Task 05: MAVE-Q v5 Ruthless 8-Seed Pruning, Ablation & OOD Suite")
    parser.add_argument("--config", default="configs/maveq_v5.yaml")
    parser.add_argument("--seeds", default="11,22,33,44,55,66,77,2026")
    parser.add_argument("--preset", default="auto")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dev", action="store_true", help="Do not write to experiment_registry.jsonl")
    parser.add_argument("--skip-ood", action="store_true", help="Run only 8-seed ablations (useful for pruning audit)")
    args = parser.parse_args()

    t_start = time.time()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    base_cfg = load_yaml(REPO_ROOT / args.config)
    ab_active = base_cfg.get("ablations", {})

    # Only ablate mechanisms currently enabled in config (+ scaling rungs)
    active_variants: list[str] = ["full_maveq_v5"]
    toggle_map = {
        "no_dca_backbone": "use_dca_backbone",
        "no_chronos_kalman_hedge": "use_chronos_kalman_hedge",
        "no_mose_hypergate": "use_mose_hypergate",
        "no_conformal_veto": "use_conformal_veto",
        "no_asym_beta_ntz": "use_asym_beta_ntz",
    }
    for v_name, flag in toggle_map.items():
        if bool(ab_active.get(flag, True)):
            active_variants.append(v_name)
    active_variants.extend(["scale_5m", "scale_20m"])

    print(
        f"=== [Task 05] Running {len(active_variants)} variants x {len(seeds)} seeds with {args.workers} workers ===",
        flush=True,
    )
    ab_rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=min(args.workers, 4)) as pool:
        for v_name in active_variants:
            v_jobs = [(v_name, s, args.preset, args.config) for s in seeds]
            v_res = list(pool.map(_run_ablation_job, v_jobs))
            ab_rows.extend(v_res)
            sr_arr = np.asarray([float(r["net_sharpe"]) for r in v_res], dtype=float)
            ic_arr = np.asarray([float(r["rank_ic"]) for r in v_res], dtype=float)
            print(
                f"  [done] {v_name:26s} | IC={ic_arr.mean():.4f} | SR={sr_arr.mean():.3f}±{sr_arr.std(ddof=1):.3f}",
                flush=True,
            )

    metrics_keys = (
        "trials",
        "rank_ic",
        "icir",
        "net_sharpe",
        "net_sortino",
        "calmar",
        "ann_net_return",
        "max_drawdown",
        "cvar_95",
        "daily_turnover",
        "capm_beta",
        "dsr",
        "dormant_veto_fraction",
        "epistemic_abstain_rate",
    )

    by_variant: dict[str, list[dict[str, Any]]] = {
        v: sorted([r for r in ab_rows if r["variant"] == v], key=lambda x: x["seed"])
        for v in active_variants
    }
    full_rows = by_variant["full_maveq_v5"]
    full_sharpes = np.asarray([r["net_sharpe"] for r in full_rows], dtype=float)
    full_calmars = np.asarray([r["calmar"] for r in full_rows], dtype=float)
    full_ics = np.asarray([r["rank_ic"] for r in full_rows], dtype=float)
    full_cvars = np.asarray([r["cvar_95"] for r in full_rows], dtype=float)
    full_dsrs = np.asarray([r["dsr"] for r in full_rows], dtype=float)

    full_rep_idx = int(np.argsort(full_sharpes)[len(full_sharpes) // 2])
    full_rep_ser = pd.Series({pd.Timestamp(k): v for k, v in full_rows[full_rep_idx]["val_returns"].items()}).sort_index()

    ablation_agg: dict[str, dict[str, Any]] = {}
    pruning_audit: dict[str, dict[str, Any]] = {}

    for v in active_variants:
        rs = by_variant[v]
        entry: dict[str, Any] = {
            "n_seeds": len(rs),
            "backbone_family": rs[0]["backbone_family"],
            "scale_rung": rs[0]["scale_rung"],
            "backbone_exact_params": int(rs[0]["backbone_exact_params"]),
        }
        for k in metrics_keys:
            arr = np.asarray([float(r[k]) for r in rs], dtype=float)
            entry[f"{k}_mean"] = round(float(arr.mean()), 4)
            entry[f"{k}_std"] = round(float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, 4)

        if v != "full_maveq_v5":
            ab_sharpes = np.asarray([r["net_sharpe"] for r in rs], dtype=float)
            ab_calmars = np.asarray([r["calmar"] for r in rs], dtype=float)
            ab_ics = np.asarray([r["rank_ic"] for r in rs], dtype=float)
            ab_cvars = np.asarray([r["cvar_95"] for r in rs], dtype=float)
            ab_dsrs = np.asarray([r["dsr"] for r in rs], dtype=float)

            ab_rep_idx = int(np.argsort(ab_sharpes)[len(ab_sharpes) // 2])
            ab_rep_ser = pd.Series({pd.Timestamp(k): val for k, val in rs[ab_rep_idx]["val_returns"].items()}).sort_index()

            bs_sr = _paired_seed_bootstrap(full_sharpes, ab_sharpes, seed=2026)
            bs_cal = _paired_seed_bootstrap(full_calmars, ab_calmars, seed=2027)
            bs_ic = _paired_seed_bootstrap(full_ics, ab_ics, seed=2028)
            bs_cvar = _paired_seed_bootstrap(full_cvars, ab_cvars, seed=2029)
            bs_dsr = _paired_seed_bootstrap(full_dsrs, ab_dsrs, seed=2030)

            entry["paired_seed_boot_sharpe"] = bs_sr
            entry["paired_seed_boot_calmar"] = bs_cal
            entry["paired_seed_boot_ic"] = bs_ic
            entry["paired_seed_boot_cvar95"] = bs_cvar
            entry["paired_seed_boot_dsr"] = bs_dsr
            entry["paired_daily_block_boot"] = paired_block_bootstrap_metrics(full_rep_ser, ab_rep_ser, n_boot=500, seed=2026)

            if v.startswith("no_"):
                # Ruthless Pruning Rule: module survives ONLY if full beats ablated with statistical support
                # on at least one primary pillar (Sharpe, Calmar, IC, or Tail-CVaR) without degrading Sharpe!
                helps_any = (
                    (bs_sr["diff_mean"] > 0.0 and bs_sr["p_one_sided"] <= 0.10)
                    or (bs_cal["diff_mean"] > 0.0 and bs_cal["p_one_sided"] <= 0.10 and bs_sr["diff_mean"] >= -0.01)
                    or (bs_ic["diff_mean"] > 0.001 and bs_ic["p_one_sided"] <= 0.10 and bs_sr["diff_mean"] >= -0.01)
                    or (bs_cvar["diff_mean"] > 0.0005 and bs_cvar["p_one_sided"] <= 0.10 and bs_sr["diff_mean"] >= -0.01)
                )
                pruning_audit[v] = {
                    "verdict": "KEEP_VERIFIED" if helps_any else "PRUNE_CANDIDATE",
                    "delta_sharpe": bs_sr["diff_mean"],
                    "p_sharpe": bs_sr["p_one_sided"],
                    "delta_calmar": bs_cal["diff_mean"],
                    "p_calmar": bs_cal["p_one_sided"],
                    "delta_ic": bs_ic["diff_mean"],
                    "p_ic": bs_ic["p_one_sided"],
                    "delta_cvar95": bs_cvar["diff_mean"],
                    "p_cvar95": bs_cvar["p_one_sided"],
                }

        ablation_agg[v] = entry
        print(
            f"  {v:26s} | IC={entry['rank_ic_mean']:.4f} | SR={entry['net_sharpe_mean']:.3f}±{entry['net_sharpe_std']:.3f} | "
            f"Cal={entry['calmar_mean']:.2f} | MDD={entry['max_drawdown_mean']*100:.1f}% | DSR={entry['dsr_mean']:.3f}",
            flush=True,
        )

    ood_specs = [
        ("VN100_10d", "VN_HOSE_T25", "vn100", 10),
        ("US_LIQUID_T1", "US_LIQUID_T1", "us30", 5),
        ("CSI_EMERGING_T1", "CSI_EMERGING_T1", "csi30", 5),
    ]
    ood_results: dict[str, dict[str, Any]] = {}
    if not args.skip_ood:
        print("=== [Task 05] Running Multi-Market OOD Generalization (VN100-10d, US_LIQUID_T1, CSI_EMERGING_T1) ===", flush=True)
        for lbl, prof, tier, h_days in ood_specs:
            ood_results[lbl] = _run_ood_market(lbl, prof, tier, h_days, preset=args.preset, seed=2026, config_rel=args.config)
            v5m = ood_results[lbl]["models"]["MAVE-Q_v5"]
            best_r = max(m["net_sharpe"] for k, m in ood_results[lbl]["models"].items() if k != "MAVE-Q_v5")
            print(
                f"  [OOD {lbl:15s}] v5_SR={v5m['net_sharpe']:+.3f} vs best_rival={best_r:+.3f} "
                f"(Rank#1={v5m['net_sharpe'] > best_r}) IC={v5m['rank_ic']:.4f} Cal={v5m['calmar']:+.2f} MDD={v5m['max_drawdown']*100:.1f}%",
                flush=True,
            )

    compact_rows = [{k: val for k, val in r.items() if k != "val_returns"} for r in ab_rows]
    summary: dict[str, Any] = {
        "mode": "validation_and_ood_only_zero_sealed_access",
        "config_path": args.config,
        "active_ablations_config": dict(ab_active),
        "seeds": seeds,
        "ablation_aggregate": ablation_agg,
        "pruning_audit": pruning_audit,
        "ablation_rows": compact_rows,
        "ood_generalization": ood_results,
        "elapsed_sec": round(time.time() - t_start, 2),
    }

    out_path = REPO_ROOT / "logs/v5_ablation_and_ood.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if not args.dev:
        logger = RunLogger(
            run_id=RUN_ID,
            logs_dir=REPO_ROOT / "logs",
            git_commit=get_git_commit_hash(REPO_ROOT),
            config_hash=compute_config_hash(
                {"variants": active_variants, "seeds": seeds, "ood": [s[0] for s in ood_specs], "ab": dict(ab_active)}
            )[:16],
            metadata={"phase": 11, "zone": "validation_and_ood_only", "n_seeds": len(seeds)},
        )
        logger.finalize_summary(summary)

    print(f"\n=== [Task 05] Pruning Audit Summary ({summary['elapsed_sec']}s) ===")
    print(json.dumps(pruning_audit, indent=2))


if __name__ == "__main__":
    main()
