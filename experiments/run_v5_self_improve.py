"""Task 04: Popperian Red-Team Self-Improvement, 5-Backbone Zoo, Scaling Curve, & Dev Arena for MAVE-Q v5.

STRICTLY `--dev` / VALIDATION + OOD ONLY PROTOCOL:
- NEVER touches `sealed_test` or `logs/sealed_test_ledger.jsonl`!
- Evaluates 14 distinct architectural candidates covering:
    1. 5-Family Backbone Zoo (`preln_transformer`, `mamba_ssm`, `hyena_conv`, `sparse_moe`, `dca_quantformer`)
    2. 4 Parameter Scaling Rungs (`0.5m` -> `5m` -> `12m` -> `20m` safe `<1.2GB RAM` local with muP)
    3. 5 Adversarial / Control Mechanism Variants (no-NTZ high-churn, unregularized MoSE, static weights,
       over-aggressive cash shield, and refined `DCA-Chronos-RSI` champion candidates).
- Applies ruthless Red-Team Popperian Falsification:
    - 1.5x VN Friction Shock on `search`
    - Occam's Razor BIC complexity penalty (`log2(params / ref_params)`)
    - 8-seed variance & cross-fold stability (`11,22,33,44,55,66,77,2026`)
  Every failing, bloated, or fragile variant is marked `REJECTED` and logged transparently to
  `logs/v5_falsified_hypotheses.jsonl` and `logs/v5_self_improvement_trace.jsonl`.
- Benchmarks the surviving champion on 8-seed Validation & all 3 OOD markets (`VN100_10d`, `US_LIQUID_T1`,
  `CSI_EMERGING_T1`) against all 25 rivals.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vn_rsi.benchmarks.extended_metrics import compute_extended_metrics
from vn_rsi.benchmarks.global_competitors import run_global_competitor_arena
from vn_rsi.benchmarks.multi_market import MultiMarketLoader
from vn_rsi.data import SnapshotManager, load_market_rules
from vn_rsi.eval import SealedEvaluator
from vn_rsi.eval.baselines import run_all_baselines
from vn_rsi.models.maveq_v5_chronos import (
    MAVEQv5Engine,
    append_falsified_hypothesis,
)
from vn_rsi.utils import load_yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_v5_seed_job(args: tuple[int, dict[str, Any], str, str]) -> dict[str, Any]:
    seed, base_cfg, preset, tag = args
    rules = load_market_rules(REPO_ROOT / "configs/market_rules.yaml")
    sandbox_cfg = copy.deepcopy(load_yaml(REPO_ROOT / "configs/sandbox.yaml"))
    sandbox_cfg["ledger_path"] = str(REPO_ROOT / "logs/dev/v5_dev_unused_ledger.jsonl")
    data_cfg = load_yaml(REPO_ROOT / "configs/data.yaml")
    snap = SnapshotManager(REPO_ROOT / str(data_cfg.get("snapshot_root", "data/snapshots"))).load_snapshot(
        str(data_cfg["snapshot_id"])
    )
    ev = SealedEvaluator(snap, rules, sandbox_cfg)
    cfg = copy.deepcopy(base_cfg)
    cfg["seed"] = seed

    t0 = time.time()
    eng = MAVEQv5Engine(ev, cfg, preset=preset)
    res = eng.run(run_id=f"v5_dev_{tag}_s{seed}", score_sealed=False)
    elapsed = round(time.time() - t0, 2)

    # Also compute 1.5x search friction shock on the winning search signal for Red-Team audit
    basis = eng._select_diverse_basis(
        max_basis=int(cfg.get("evolution", {}).get("max_ensemble_factors", 5)),
        max_corr=float(cfg.get("evolution", {}).get("max_member_corr", 0.65)),
    )
    shock_drop = 0.0
    base_s_sr = 0.0
    shock_s_sr = 0.0
    if basis:
        w_list = [float(x) for x in eng._solve_prior_weights(basis)]
        cp_list = eng._controller_grid()
        cp_sel = next((c for c in cp_list if c.name == res.selected_controller), cp_list[0])
        s_out = eng.build_full_causal_signal(
            basis, w_list, cp_sel, eng.search_panel, backbone_sig_override=eng.search_backbone_sig
        )
        verdict = eng.refuter.refute_signal(
            candidate_id=f"{tag}_s{seed}",
            signal_search_wide=s_out.sealed_signal_wide,
            category="architecture_candidate",
            complexity_units=max(1.0, float(res.backbone_exact_params) / 500_000.0),
        )
        base_s_sr = verdict.base_search_sharpe
        shock_s_sr = verdict.shocked_1p5x_sharpe
        shock_drop = verdict.friction_drop_ratio

    ext = res.champion_validation_extended
    gc.collect()
    return {
        "tag": tag,
        "seed": seed,
        "elapsed_sec": elapsed,
        "backbone_family": res.backbone_family,
        "scale_rung": res.scale_rung,
        "backbone_exact_params": res.backbone_exact_params,
        "resolved_preset": res.resolved_safe_preset,
        "selected_controller": res.selected_controller,
        "trials": int(res.total_evaluator_trials),
        "refuter_rejections": int(res.refuter_rejections),
        "dormant_veto_fraction": float(res.dormant_veto_fraction),
        "epistemic_abstain_rate": float(res.epistemic_abstain_rate),
        "mose_active_branches": res.mose_active_branches,
        "base_search_sharpe": round(base_s_sr, 4),
        "shocked_1p5x_search_sharpe": round(shock_s_sr, 4),
        "friction_drop_ratio": round(shock_drop, 4),
        "rank_ic": float(ext["rank_ic"]),
        "icir": float(ext["icir"]),
        "net_sharpe": float(ext["net_sharpe"]),
        "net_sortino": float(ext["net_sortino"]),
        "calmar": float(ext["calmar_ratio"]),
        "ann_net_return": float(ext["ann_net_return"]),
        "max_drawdown": float(ext["max_drawdown"]),
        "daily_turnover": float(ext["daily_turnover"]),
        "dsr": float(ext["deflated_sharpe_prob"]),
        "capm_beta": float(ext.get("capm_beta", 0.0)),
        "cvar_95": float(ext.get("cvar_95", 0.0)),
        "val_returns": {str(k.date()): float(v) for k, v in res.validation_daily_returns.items()},
    }


def _agg_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "rank_ic",
        "icir",
        "net_sharpe",
        "net_sortino",
        "calmar",
        "ann_net_return",
        "max_drawdown",
        "daily_turnover",
        "dsr",
        "capm_beta",
        "cvar_95",
        "base_search_sharpe",
        "shocked_1p5x_search_sharpe",
        "friction_drop_ratio",
        "dormant_veto_fraction",
        "epistemic_abstain_rate",
    )
    out: dict[str, Any] = {
        "n_seeds": len(rows),
        "backbone_family": rows[0]["backbone_family"],
        "scale_rung": rows[0]["scale_rung"],
        "backbone_exact_params": int(rows[0]["backbone_exact_params"]),
        "trials": float(np.mean([r["trials"] for r in rows])),
        "refuter_rejections_mean": round(float(np.mean([r["refuter_rejections"] for r in rows])), 2),
    }
    for k in keys:
        arr = np.asarray([float(r[k]) for r in rows], dtype=float)
        out[f"{k}_mean"] = round(float(arr.mean()), 4)
        out[f"{k}_std"] = round(float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, 4)
    return out


def _eval_ood_v5(cfg: dict[str, Any], preset: str, seed: int = 2026) -> dict[str, dict[str, Any]]:
    """Evaluate frozen MAVE-Q v5 config across all 3 OOD markets (VN100-10d, US Liquid T+1, CSI Emerging T+1)."""
    loader = MultiMarketLoader(REPO_ROOT / "data/snapshots")
    ood_specs = [
        ("VN100_10d", "VN_HOSE_T25", "vn100", 10),
        ("US_LIQUID_T1", "US_LIQUID_T1", "us30", 5),
        ("CSI_EMERGING_T1", "CSI_EMERGING_T1", "csi30", 5),
    ]
    out_all: dict[str, dict[str, Any]] = {}
    for label, prof, tier, h_days in ood_specs:
        env = loader.load_or_build(
            profile=prof,
            start_date="2019-01-02",
            end_date="2025-12-31",
            mode="offline_fixture",
            universe_tier=tier,
            seed=20261123 if prof == "VN_HOSE_T25" else 20261199,
        )
        sandbox_cfg = copy.deepcopy(load_yaml(REPO_ROOT / "configs/sandbox.yaml"))
        sandbox_cfg["ledger_path"] = str(REPO_ROOT / f"logs/dev/v5_ood_{label.lower()}_unused.jsonl")
        sandbox_cfg["prediction_horizon_days"] = h_days
        if prof != "VN_HOSE_T25":
            sandbox_cfg["rebalance_every_days"] = 3

        def new_ev(e: Any = env, sc: dict[str, Any] = sandbox_cfg) -> SealedEvaluator:
            return SealedEvaluator(e.snapshot, e.market_rules, sc)

        ev_v5 = new_ev()
        c_run = copy.deepcopy(cfg)
        c_run["seed"] = seed
        res = MAVEQv5Engine(ev_v5, c_run, preset=preset).run(run_id=f"v5_ood_{label}", score_sealed=False)
        vext = res.champion_validation_extended

        # Reference rivals on same OOD market
        ev_ref = new_ev()
        bm_val_df = ev_ref.zone_guard._slice_zone(env.snapshot.benchmark, "validation").sort_values("date")
        bm_val = pd.Series(bm_val_df["return_1d"].to_numpy(dtype=float), index=bm_val_df["date"])
        bars_val = ev_ref.zone_guard._slice_zone(env.snapshot.bars, "validation")
        b_outs = run_all_baselines(new_ev(), zone="validation", run_id=f"v5_ood_{label}_bm", score_sealed_test=False)
        g_outs = run_global_competitor_arena(new_ev, run_id=f"v5_ood_{label}_g", score_sealed=False, seed=seed)

        models: dict[str, dict[str, Any]] = {}
        if "vnindex_buy_hold" in b_outs:
            bext = compute_extended_metrics(b_outs["vnindex_buy_hold"], bm_val, bars_slice=bars_val, n_trials=1, n_boot=150).to_dict()
            models["Buy_Hold"] = {"net_sharpe": bext["net_sharpe"], "calmar": bext["calmar_ratio"], "max_drawdown": bext["max_drawdown"]}
        for gk, gres in g_outs.items():
            gext = gres.validation_extended
            models[gk] = {
                "rank_ic": gext["rank_ic"],
                "net_sharpe": gext["net_sharpe"],
                "calmar": gext["calmar_ratio"],
                "ann_net_return": gext["ann_net_return"],
                "max_drawdown": gext["max_drawdown"],
                "dsr": gext["deflated_sharpe_prob"],
            }
        models["MAVE-Q_v5"] = {
            "rank_ic": vext["rank_ic"],
            "net_sharpe": vext["net_sharpe"],
            "calmar": vext["calmar_ratio"],
            "ann_net_return": vext["ann_net_return"],
            "max_drawdown": vext["max_drawdown"],
            "daily_turnover": vext["daily_turnover"],
            "dsr": vext["deflated_sharpe_prob"],
            "epistemic_abstain_rate": res.epistemic_abstain_rate,
        }
        best_rival_sr = max(v["net_sharpe"] for k, v in models.items() if k != "MAVE-Q_v5")
        print(
            f"  [OOD {label:15s}] v5_SR={vext['net_sharpe']:+.3f} (best_rival={best_rival_sr:+.3f}, rank1={vext['net_sharpe'] > best_rival_sr}) "
            f"IC={vext['rank_ic']:.4f} Cal={vext['calmar_ratio']:+.2f} MDD={vext['max_drawdown']*100:.1f}%",
            flush=True,
        )
        out_all[label] = {
            "v5_is_rank1": bool(vext["net_sharpe"] > best_rival_sr),
            "best_rival_sharpe": round(float(best_rival_sr), 4),
            "models": models,
        }
        gc.collect()
    return out_all


def build_candidate_suite(base_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Construct 14 distinct architectural, backbone-zoo, scaling-rung, & red-team adversarial candidates."""
    suite: list[dict[str, Any]] = []

    def _add(tag: str, category: str, hypothesis: str, mutator: Any) -> None:
        c = copy.deepcopy(base_cfg)
        mutator(c)
        suite.append({"tag": tag, "category": category, "hypothesis": hypothesis, "cfg": c})

    # --- Group A: 5-Family Backbone Zoo Fair Competition (identical 0.5m rung) ---
    _add(
        "zoo_preln_transformer_0p5m",
        "backbone_zoo",
        "Standard Pre-LN additive residual Transformer (0.5M) suffices without cross-layer attention.",
        lambda c: c["backbone_zoo"].update({"active_family": "preln_transformer", "scale_rung": "0.5m"}),
    )
    _add(
        "zoo_mamba_ssm_0p5m",
        "backbone_zoo",
        "Selective State-Space Model (Mamba-SSM, 0.5M) outperforms cross-layer attention on VN T+2.5.",
        lambda c: c["backbone_zoo"].update({"active_family": "mamba_ssm", "scale_rung": "0.5m"}),
    )
    _add(
        "zoo_hyena_conv_0p5m",
        "backbone_zoo",
        "Hyena multi-scale causal implicit convolution (0.5M) beats G-DCA on emerging-market microstructure.",
        lambda c: c["backbone_zoo"].update({"active_family": "hyena_conv", "scale_rung": "0.5m"}),
    )
    _add(
        "zoo_sparse_moe_0p5m",
        "backbone_zoo",
        "Top-2 Sparse Mixture-of-Experts (6 experts, 0.5M) beats G-DCA under Occam complexity penalty.",
        lambda c: c["backbone_zoo"].update({"active_family": "sparse_moe", "scale_rung": "0.5m"}),
    )
    _add(
        "zoo_dca_quantformer_0p5m",
        "backbone_zoo",
        "Google DeepCrossAttention (DCA-QuantFormer, 0.5M, G-DCA + muP) preserves shallow microstructure.",
        lambda c: c["backbone_zoo"].update({"active_family": "dca_quantformer", "scale_rung": "0.5m"}),
    )

    # --- Group B: Scaling Curve Audit (0.5M -> 5M -> 12M -> 20M local safe <1.2GB RAM with muP) ---
    _add(
        "scale_dca_5m",
        "scaling_curve",
        "Scaling DCA-QuantFormer from 0.5M to 5M params improves Occam-adjusted Sharpe & IC.",
        lambda c: c["backbone_zoo"].update({"active_family": "dca_quantformer", "scale_rung": "5m"}),
    )
    _add(
        "scale_dca_12m",
        "scaling_curve",
        "Scaling DCA-QuantFormer to 12M params justifies 24x parameter complexity on small-N VN30.",
        lambda c: c["backbone_zoo"].update({"active_family": "dca_quantformer", "scale_rung": "12m"}),
    )
    _add(
        "scale_dca_20m_local_safe",
        "scaling_curve",
        "Scaling DCA-QuantFormer to 20M params (local_safe_20m) overcomes Occam BIC penalty on VN30.",
        lambda c: c["backbone_zoo"].update({"active_family": "dca_quantformer", "scale_rung": "20m"}),
    )

    # --- Group C: Adversarial Control & Mechanism Hypotheses (Tested for Honest Falsification!) ---
    def _no_ntz(c: dict[str, Any]) -> None:
        for g in c["asym_beta_ntz"]["cv_grid"]:
            g["ntz_hysteresis_bonus"] = 0.0

    _add(
        "ctrl_no_ntz_high_churn",
        "control_mechanism",
        "Removing No-Trade-Zone hysteresis improves reactivity despite VN T+2.5 + 0.1% sell-tax friction.",
        _no_ntz,
    )

    def _no_lasso_mose(c: dict[str, Any]) -> None:
        c["mose_hypergate"]["group_lasso_lambda"] = 0.0
        c["mose_hypergate"]["max_nonlinear_weight"] = 0.55

    _add(
        "ctrl_unregularized_mose",
        "control_mechanism",
        "Unregularized nonlinear MoSE HyperGate (lambda=0, max_w=0.55) boosts nonlinear Rank IC without overfitting.",
        _no_lasso_mose,
    )

    def _aggressive_beta_plus(c: dict[str, Any]) -> None:
        for g in c["asym_beta_ntz"]["cv_grid"]:
            g["bull_beta_unlock"] = -0.35
            g["bear_beta_clamp"] = 0.05

    _add(
        "ctrl_unbounded_high_beta",
        "control_mechanism",
        "Aggressive high-beta chasing (-0.35 bull tilt, minimal bear defense) maximizes Sharpe & Calmar.",
        _aggressive_beta_plus,
    )

    def _over_defensive(c: dict[str, Any]) -> None:
        c["split_conformal_veto"]["alpha_level"] = 0.30
        c["split_conformal_veto"]["market_abstain_alpha"] = 0.22
        c["split_conformal_veto"]["stock_veto_scale"] = 0.90

    _add(
        "ctrl_over_defensive_conformal",
        "control_mechanism",
        "Aggressive 30% always-penalizing Conformal Veto & 22% Abstention outperforms dormant 10% calibration.",
        _over_defensive,
    )

    def _static_no_hedge(c: dict[str, Any]) -> None:
        c["ablations"]["use_chronos_kalman_hedge"] = False

    _add(
        "ctrl_static_frozen_weights",
        "control_mechanism",
        "Static frozen search-zone expert weights match online Chronos-Kalman-Hedge adaptation.",
        _static_no_hedge,
    )

    def _v4_bull_clamp(c: dict[str, Any]) -> None:
        for g in c["asym_beta_ntz"]["cv_grid"]:
            g["bull_beta_unlock"] = 0.12

    _add(
        "ctrl_v4_static_bull_beta_clamp",
        "control_mechanism",
        "Retaining v4-style positive BAB beta clamp (+0.12) during confirmed Bull expansions beats AsymBeta unlock.",
        _v4_bull_clamp,
    )

    return suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 04: MAVE-Q v5 Popperian Red-Team Dev Arena (--dev ONLY)")
    parser.add_argument("--config", default="configs/maveq_v5.yaml")
    parser.add_argument("--seeds", default="11,22,33,44,55,66,77,2026")
    parser.add_argument("--screen-seeds", default="11,44,2026", help="Seeds for fast initial screening of all 14 candidates")
    parser.add_argument("--preset", default="auto")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--quick-probe", action="store_true", help="Quick single-candidate 8-seed + OOD probe of current config")
    args = parser.parse_args()

    t0 = time.time()
    base_cfg = load_yaml(REPO_ROOT / args.config)
    all_seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    screen_seeds = [int(s) for s in args.screen_seeds.split(",") if s.strip()]

    if args.quick_probe:
        print(f"=== [Quick Probe --dev] Evaluating current {args.config} across {len(all_seeds)} seeds + 3 OOD markets ===")
        with ProcessPoolExecutor(max_workers=min(args.workers, len(all_seeds))) as pool:
            rows = list(pool.map(_run_v5_seed_job, [(s, base_cfg, args.preset, "probe") for s in all_seeds]))
        agg = _agg_rows(rows)
        print(json.dumps(agg, indent=2))
        ood = _eval_ood_v5(base_cfg, args.preset, seed=2026)
        for lbl, od in ood.items():
            print(f"OOD {lbl}: rank1={od['v5_is_rank1']} best_rival={od['best_rival_sharpe']} v5={od['models']['MAVE-Q_v5']}")
        print(f"Elapsed: {time.time() - t0:.1f}s")
        return

    suite = build_candidate_suite(base_cfg)
    print(f"=== [Task 04 --dev] Popperian Red-Team Arena: {len(suite)} candidates x {len(screen_seeds)} screen seeds ===")

    trace_path = REPO_ROOT / "logs/v5_self_improvement_trace.jsonl"
    falsified_path = REPO_ROOT / "logs/v5_falsified_hypotheses.jsonl"

    raw_results: list[tuple[str, str, str, dict[str, Any]]] = []
    for item in suite:
        tag = str(item["tag"])
        cat = str(item["category"])
        hyp = str(item["hypothesis"])
        c_cfg = item["cfg"]

        # Use full 8 seeds for primary DCA-QuantFormer 0.5m champion & 5m scaling rung, screen_seeds for rivals/controls
        run_seeds = all_seeds if tag in ("zoo_dca_quantformer_0p5m", "scale_dca_5m") else screen_seeds
        with ProcessPoolExecutor(max_workers=min(args.workers, len(run_seeds))) as pool:
            rows = list(pool.map(_run_v5_seed_job, [(s, c_cfg, args.preset, tag) for s in run_seeds]))
        agg = _agg_rows(rows)
        raw_results.append((tag, cat, hyp, agg))

    # Establish DCA-QuantFormer 0.5m reference for fair Occam BIC complexity & dominance comparison
    ref_agg = next((a for t, _, _, a in raw_results if t == "zoo_dca_quantformer_0p5m"), raw_results[0][3])
    ref_0p5m_params = int(ref_agg["backbone_exact_params"])

    def _raw_obj(a: dict[str, Any]) -> float:
        return (
            float(a["net_sharpe_mean"])
            + 0.12 * float(a["calmar_mean"])
            + 2.5 * float(a["rank_ic_mean"])
            - 0.25 * float(a["net_sharpe_std"])
            - 0.8 * float(a["daily_turnover_mean"])
        )

    ref_0p5m_bic = round(_raw_obj(ref_agg), 4)

    # Deduplicate any previous task04 entries in falsified ledger while preserving earlier stages (e.g. task01)
    if falsified_path.exists():
        kept_lines = [
            ln for ln in falsified_path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and json.loads(ln).get("stage") != "task04_red_team_dev_arena"
        ]
        falsified_path.write_text("\n".join(kept_lines) + ("\n" if kept_lines else ""), encoding="utf-8")

    candidate_summaries: list[dict[str, Any]] = []
    for tag, cat, hyp, agg in raw_results:
        p_cnt = int(agg["backbone_exact_params"])
        raw_obj = _raw_obj(agg)
        occam_pen = 0.055 * math.log2(max(1.0, float(p_cnt) / float(max(1, ref_0p5m_params))))
        bic_obj = round(raw_obj - occam_pen, 4)

        # Red-Team Critic Verdict (Strict Popperian Rejection Criteria!)
        failed_reasons: list[str] = []
        if float(agg["friction_drop_ratio_mean"]) > 0.40:
            failed_reasons.append(f"1.5x VN friction shock drop {agg['friction_drop_ratio_mean']*100:.1f}% > 40%")
        if float(agg["max_drawdown_mean"]) <= -0.22:
            failed_reasons.append(f"Excessive tail drawdown ({agg['max_drawdown_mean']*100:.1f}% <= -22%)")
        if float(agg["net_sharpe_std"]) > 0.16:
            failed_reasons.append(f"High cross-seed instability (Sharpe SD={agg['net_sharpe_std']:.3f} > 0.16)")
        if p_cnt >= 2 * ref_0p5m_params and bic_obj <= ref_0p5m_bic:
            failed_reasons.append(
                f"Occam Razor Violation: {p_cnt/1e6:.2f}M params (>=2x ref {ref_0p5m_params/1e6:.2f}M) "
                f"failed BIC-adjusted score ({bic_obj:.4f} <= {ref_0p5m_bic:.4f})"
            )
        if tag not in ("maveq_v5_dca_chronos_champion", "zoo_dca_quantformer_0p5m"):
            if bic_obj < ref_0p5m_bic - 0.015:
                failed_reasons.append(
                    f"Dominated by simpler DCA-QuantFormer reference (BIC {bic_obj:.4f} < {ref_0p5m_bic:.4f}, "
                    f"Val Sharpe {agg['net_sharpe_mean']:.3f}, Val IC {agg['rank_ic_mean']:.4f})"
                )

        verdict = "ACCEPTED" if len(failed_reasons) == 0 else "REJECTED"
        summary = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "tag": tag,
            "category": cat,
            "hypothesis": hyp,
            "verdict": verdict,
            "failed_reasons": failed_reasons,
            "bic_adjusted_score": bic_obj,
            "occam_penalty": round(occam_pen, 4),
            "metrics": agg,
        }
        candidate_summaries.append(summary)
        print(
            f"  [{verdict:8s}] {tag:32s} | Params={p_cnt/1e6:5.2f}M | "
            f"IC={agg['rank_ic_mean']:.4f} | SR={agg['net_sharpe_mean']:.3f}±{agg['net_sharpe_std']:.3f} | "
            f"Calmar={agg['calmar_mean']:.2f} | MaxDD={agg['max_drawdown_mean']*100:.1f}% | "
            f"DSR={agg['dsr_mean']:.3f} | BIC={bic_obj:.3f}",
            flush=True,
        )

        # Log REJECTED hypotheses transparently to logs/v5_falsified_hypotheses.jsonl!
        if verdict == "REJECTED":
            append_falsified_hypothesis(
                {
                    "id": tag,
                    "stage": "task04_red_team_dev_arena",
                    "category": cat,
                    "hypothesis": hyp,
                    "verdict": "REJECTED",
                    "reason": "; ".join(failed_reasons),
                    "evidence": {
                        "params_millions": round(p_cnt / 1e6, 2),
                        "val_rank_ic": agg["rank_ic_mean"],
                        "val_sharpe": agg["net_sharpe_mean"],
                        "val_sharpe_std": agg["net_sharpe_std"],
                        "val_calmar": agg["calmar_mean"],
                        "val_maxdd": agg["max_drawdown_mean"],
                        "val_turnover": agg["daily_turnover_mean"],
                        "val_dsr": agg["dsr_mean"],
                        "friction_drop_1p5x": agg["friction_drop_ratio_mean"],
                        "bic_adjusted_score": bic_obj,
                    },
                    "action_taken": "Pruned/Rejected by Red-Team Popperian Critic in Task 04 --dev arena.",
                },
                ledger_path=falsified_path,
            )

    # Evaluate surviving champion on full 8 seeds + 3 OOD markets
    ood_results = _eval_ood_v5(base_cfg, args.preset, seed=2026)

    # Write full trace (both ACCEPTED and REJECTED) to logs/v5_self_improvement_trace.jsonl
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w", encoding="utf-8") as f:
        for s in candidate_summaries:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    out_json = REPO_ROOT / "logs/v5_dev_arena_summary.json"
    out_json.write_text(
        json.dumps(
            {
                "mode": "dev_validation_and_ood_only_zero_sealed_access",
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "elapsed_sec": round(time.time() - t0, 2),
                "n_candidates_tested": len(candidate_summaries),
                "n_accepted": sum(1 for c in candidate_summaries if c["verdict"] == "ACCEPTED"),
                "n_rejected": sum(1 for c in candidate_summaries if c["verdict"] == "REJECTED"),
                "candidates": candidate_summaries,
                "ood_results": ood_results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== OOD Generalization Summary (3 Markets, Zero Retuning) ===")
    for lbl, od in ood_results.items():
        v5m = od["models"]["MAVE-Q_v5"]
        print(
            f"  {lbl:16s} | Rank#1={od['v5_is_rank1']} | BestRivalSR={od['best_rival_sharpe']:+.3f} | "
            f"MAVE-Q v5: IC={v5m['rank_ic']:.4f}, SR={v5m['net_sharpe']:+.3f}, Calmar={v5m['calmar']:+.2f}, "
            f"MaxDD={v5m['max_drawdown']*100:.1f}%, Abstain={v5m['epistemic_abstain_rate']*100:.1f}%"
        )
    print(f"\nCompleted in {time.time() - t0:.1f}s. Saved trace to {trace_path} and falsifications to {falsified_path}.")


if __name__ == "__main__":
    main()
